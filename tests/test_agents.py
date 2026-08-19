"""Agent tests.

These assert on *behaviour*, not on prose: which real tools the agent invoked,
and whether the structured contract was honoured. The model is scripted; the
agent loop, tool dispatch and output parsing are real.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from etl_migrator.agents.discovery import DiscoveryAgent, discovery_task
from etl_migrator.agents.planner import PlannerAgent, planning_task
from etl_migrator.agents.spark_engineer import SparkEngineerAgent, codegen_task
from etl_migrator.domain.enums import RiskLevel, TransformKind
from etl_migrator.domain.errors import AgentContractError, ScriptExhaustedError
from etl_migrator.domain.plan import MigrationPlan
from etl_migrator.domain.spec import MigrationSpec
from etl_migrator.llm.factory import ScriptedModelClientFactory
from etl_migrator.llm.scripted import ScriptedChatCompletionClient, ScriptedTurn
from etl_migrator.tools.code_gate import analyze_generated_code


@pytest.fixture
def spec(fixture_payload: dict) -> MigrationSpec:
    return MigrationSpec.model_validate(fixture_payload["discovery"][-1]["content"])


@pytest.fixture
def plan(fixture_payload: dict) -> MigrationPlan:
    return MigrationPlan.model_validate(fixture_payload["planner"][-1]["content"])


class TestDiscoveryAgent:
    async def test_calls_real_tools_before_answering(
        self, factory: ScriptedModelClientFactory, legacy_source: Path, example_input_dir: Path
    ) -> None:
        agent = DiscoveryAgent(
            factory.client_for("discovery"),
            source_path=legacy_source,
            input_dir=example_input_dir,
        )
        run = await agent.run(discovery_task(legacy_source, example_input_dir))

        assert "inspect_legacy_source" in run.tools_used
        assert run.tools_used.count("profile_input_data") == 2
        assert all(not t.is_error for t in run.tool_invocations)

    async def test_tool_output_is_real_not_replayed(
        self, factory: ScriptedModelClientFactory, legacy_source: Path, example_input_dir: Path
    ) -> None:
        """The *responses* are scripted; the tool *results* are computed live.

        If this ever became a recording too, the agents would be testing nothing.
        """
        agent = DiscoveryAgent(
            factory.client_for("discovery"),
            source_path=legacy_source,
            input_dir=example_input_dir,
        )
        run = await agent.run(discovery_task(legacy_source, example_input_dir))
        inspect_result = next(t for t in run.tool_invocations if t.name == "inspect_legacy_source")
        assert "python_pandas" in inspect_result.result_preview

    async def test_produces_a_validated_specification(
        self, factory: ScriptedModelClientFactory, legacy_source: Path, example_input_dir: Path
    ) -> None:
        agent = DiscoveryAgent(
            factory.client_for("discovery"),
            source_path=legacy_source,
            input_dir=example_input_dir,
        )
        run = await agent.run(discovery_task(legacy_source, example_input_dir))
        assert isinstance(run.output, MigrationSpec)
        assert run.output.max_risk is RiskLevel.HIGH
        assert {t.kind for t in run.output.transformations} >= {
            TransformKind.JOIN,
            TransformKind.AGGREGATE,
            TransformKind.FILTER,
        }

    async def test_profiler_tool_refuses_path_traversal(
        self, factory: ScriptedModelClientFactory, legacy_source: Path, example_input_dir: Path
    ) -> None:
        agent = DiscoveryAgent(
            factory.client_for("discovery"),
            source_path=legacy_source,
            input_dir=example_input_dir,
        )
        tool = agent._make_profile_tool()
        assert "refused" in tool("../../../etc/passwd")


class TestPlannerAgent:
    async def test_consults_the_pattern_catalogue(
        self, factory: ScriptedModelClientFactory, spec: MigrationSpec
    ) -> None:
        agent = PlannerAgent(factory.client_for("planner"), spec=spec)
        run = await agent.run(planning_task(spec))
        assert run.tools_used.count("lookup_migration_pattern") == 4
        assert "dataset_sizes" in run.tools_used

    async def test_declares_semantic_differences_for_the_null_key_trap(
        self, factory: ScriptedModelClientFactory, spec: MigrationSpec
    ) -> None:
        agent = PlannerAgent(factory.client_for("planner"), spec=spec)
        plan = (await agent.run(planning_task(spec))).output

        aggregate_step = next(s for s in plan.steps if s.kind is TransformKind.AGGREGATE)
        assert aggregate_step.risk_level is RiskLevel.HIGH
        assert any("dropna" in d.description or "null" in d.description.lower()
                   for d in aggregate_step.semantic_differences)

    async def test_semantic_differences_expand_required_checks(
        self, factory: ScriptedModelClientFactory, spec: MigrationSpec
    ) -> None:
        agent = PlannerAgent(factory.client_for("planner"), spec=spec)
        plan = (await agent.run(planning_task(spec))).output
        assert set(plan.effective_required_checks()) >= set(plan.validation_plan.required_checks)

    async def test_reset_index_maps_to_nothing(
        self, factory: ScriptedModelClientFactory, spec: MigrationSpec
    ) -> None:
        agent = PlannerAgent(factory.client_for("planner"), spec=spec)
        plan = (await agent.run(planning_task(spec))).output
        step = next(s for s in plan.steps if s.kind is TransformKind.RESET_INDEX)
        assert step.spark_construct.startswith("(none")

    def test_dataset_sizes_refuses_to_guess_without_profiles(self, spec: MigrationSpec) -> None:
        agent = PlannerAgent(
            ScriptedChatCompletionClient([]), spec=spec, profiles=[]
        )
        assert "Do not broadcast" in agent._make_sizes_tool()()

    def test_unknown_pattern_kind_lists_alternatives(self, spec: MigrationSpec) -> None:
        agent = PlannerAgent(ScriptedChatCompletionClient([]), spec=spec)
        assert "Known kinds" in agent._make_pattern_tool()("teleport")


class TestSparkEngineerAgent:
    async def test_self_corrects_through_the_gate(
        self, factory: ScriptedModelClientFactory, spec: MigrationSpec, plan: MigrationPlan
    ) -> None:
        """The recorded first submission is genuinely broken; the agent must
        submit twice, and the second submission must be gate-clean."""
        agent = SparkEngineerAgent(factory.client_for("spark_engineer"), plan=plan, spec=spec)
        run = await agent.run(codegen_task(spec, plan, "out.py"))

        assert run.tools_used.count("check_spark_code") == 2
        assert agent.gate_calls == 2

        first, second = run.tool_invocations
        assert "gate: FAIL" in first.result_preview
        assert "gate: PASS" in second.result_preview

    async def test_final_code_independently_passes_the_gate(
        self, factory: ScriptedModelClientFactory, spec: MigrationSpec, plan: MigrationPlan
    ) -> None:
        agent = SparkEngineerAgent(factory.client_for("spark_engineer"), plan=plan, spec=spec)
        run = await agent.run(codegen_task(spec, plan, "out.py"))
        assert analyze_generated_code(run.output.content).passed

    async def test_generated_code_implements_the_declared_mitigations(
        self, factory: ScriptedModelClientFactory, spec: MigrationSpec, plan: MigrationPlan
    ) -> None:
        agent = SparkEngineerAgent(factory.client_for("spark_engineer"), plan=plan, spec=spec)
        content = (await agent.run(codegen_task(spec, plan, "out.py"))).output.content

        assert "isNotNull" in content, "pandas groupby(dropna=True) not reproduced"
        assert "coalesce" in content, "pandas all-NaN sum -> 0.0 not reproduced"
        assert "broadcast" in content, "planner's broadcast decision not implemented"
        assert "monotonically_increasing_id" not in content, "invented an index column"


class TestAgentContract:
    async def test_missing_structured_output_raises(self) -> None:
        client = ScriptedChatCompletionClient([ScriptedTurn(content="here you go, boss")])
        agent = PlannerAgent(client, spec=MigrationSpec(
            source_path="x.py", source_language="python_pandas", summary="s", confidence=0.5
        ))
        with pytest.raises(AgentContractError, match="did not produce a MigrationPlan"):
            await agent.run("go")

    async def test_exhausted_script_fails_loudly(
        self, legacy_source: Path, example_input_dir: Path
    ) -> None:
        """A silent fallback would let a broken agent loop look like a pass."""
        agent = DiscoveryAgent(
            ScriptedChatCompletionClient([], label="empty"),
            source_path=legacy_source,
            input_dir=example_input_dir,
        )
        with pytest.raises(ScriptExhaustedError, match="exhausted"):
            await agent.run("go")
