"""Tests for the Testing and Validation agents.

The Validation agent is the one place in the system where an LLM is shown a
correctness verdict, so its constraints get the most attention here: it must be
impossible to invoke on a passing migration, and its output type must have no
field capable of overturning the differ.
"""

from __future__ import annotations

import pytest

# Imported as modules, not names: `TestingAgent` and `testing_task` both match
# pytest's default collection patterns, and pytest would try to run them.
from etl_migrator.agents import testing as testing_mod
from etl_migrator.agents import validation as validation_mod
from etl_migrator.domain.artifacts import GeneratedCode
from etl_migrator.domain.enums import RiskCategory, ValidationStatus
from etl_migrator.domain.plan import MigrationPlan
from etl_migrator.domain.spec import MigrationSpec
from etl_migrator.domain.validation import (
    CheckResult,
    Difference,
    GeneratedTests,
    ValidationDiagnosis,
    ValidationReport,
)
from etl_migrator.llm.factory import ScriptedModelClientFactory
from etl_migrator.llm.scripted import ScriptedChatCompletionClient
from etl_migrator.tools.code_gate import GateOptions, analyze_generated_code


@pytest.fixture
def spec(fixture_payload: dict) -> MigrationSpec:
    return MigrationSpec.model_validate(fixture_payload["discovery"][-1]["content"])


@pytest.fixture
def plan(fixture_payload: dict) -> MigrationPlan:
    return MigrationPlan.model_validate(fixture_payload["planner"][-1]["content"])


@pytest.fixture
def code(fixture_payload: dict) -> GeneratedCode:
    return GeneratedCode.model_validate(fixture_payload["spark_engineer"][-1]["content"])


def failing_report() -> ValidationReport:
    """The real failure shape: one cause, several symptoms."""
    return ValidationReport(
        migration_id="mig-fail",
        checks=[
            CheckResult(
                name="row_count",
                passed=False,
                detail="reference=5 candidate=6",
                differences=[
                    Difference(
                        check="row_count",
                        category=RiskCategory.NULL_SEMANTICS,
                        reference="5",
                        candidate="6",
                        detail="row counts differ by +1",
                    )
                ],
            ),
            CheckResult(
                name="null_counts",
                passed=False,
                detail="country differs",
                differences=[
                    Difference(
                        check="null_counts",
                        column="country",
                        category=RiskCategory.NULL_SEMANTICS,
                        reference="0",
                        candidate="1",
                        detail="null counts differ",
                    )
                ],
            ),
        ],
    )


class TestTestingAgent:
    async def test_reads_the_generated_pipeline_before_writing_tests(
        self,
        factory: ScriptedModelClientFactory,
        spec: MigrationSpec,
        plan: MigrationPlan,
        code: GeneratedCode,
    ) -> None:
        agent = testing_mod.TestingAgent(
            factory.client_for("testing"), spec=spec, plan=plan, code_content=code.content
        )
        run = await agent.run(testing_mod.testing_task(spec, plan, "test_pipeline.py"))
        assert "read_generated_pipeline" in run.tools_used

    async def test_self_corrects_through_the_test_gate(
        self,
        factory: ScriptedModelClientFactory,
        spec: MigrationSpec,
        plan: MigrationPlan,
        code: GeneratedCode,
    ) -> None:
        """The recorded first submission imports os; the gate rejects it."""
        agent = testing_mod.TestingAgent(
            factory.client_for("testing"), spec=spec, plan=plan, code_content=code.content
        )
        run = await agent.run(testing_mod.testing_task(spec, plan, "test_pipeline.py"))

        gate_calls = [t for t in run.tool_invocations if t.name == "check_test_code"]
        assert len(gate_calls) == 2
        assert "gate: FAIL" in gate_calls[0].result_preview
        assert "gate: PASS" in gate_calls[1].result_preview

    async def test_final_suite_passes_an_independent_gate(
        self,
        factory: ScriptedModelClientFactory,
        spec: MigrationSpec,
        plan: MigrationPlan,
        code: GeneratedCode,
    ) -> None:
        agent = testing_mod.TestingAgent(
            factory.client_for("testing"), spec=spec, plan=plan, code_content=code.content
        )
        run = await agent.run(testing_mod.testing_task(spec, plan, "test_pipeline.py"))
        assert analyze_generated_code(run.output.content, GateOptions.for_tests()).passed

    async def test_covers_every_declared_semantic_difference(
        self,
        factory: ScriptedModelClientFactory,
        spec: MigrationSpec,
        plan: MigrationPlan,
        code: GeneratedCode,
    ) -> None:
        """The suite exists to pin the differences the planner declared. If it
        covers none of them it is decoration."""
        agent = testing_mod.TestingAgent(
            factory.client_for("testing"), spec=spec, plan=plan, code_content=code.content
        )
        tests: GeneratedTests = (
            await agent.run(testing_mod.testing_task(spec, plan, "test_pipeline.py"))
        ).output
        declared = {d.validation_check for d in plan.all_semantic_differences}
        assert set(tests.covers_checks) & declared
        assert len(tests.test_names) >= len(plan.all_semantic_differences) // 2

    def test_task_prompt_carries_the_differences_and_their_mitigations(
        self, spec: MigrationSpec, plan: MigrationPlan
    ) -> None:
        prompt = testing_mod.testing_task(spec, plan, "test_pipeline.py")
        assert "SEMANTIC DIFFERENCES THAT MUST EACH GET A TEST" in prompt
        assert "mitigation:" in prompt


class TestValidationAgentConstraints:
    def test_refuses_to_be_constructed_on_a_passing_report(self, plan: MigrationPlan) -> None:
        """Inviting an opinion on a passing migration is inviting it to be
        overturned. The only safe answer is to make the call impossible."""
        passing = ValidationReport(
            migration_id="m", checks=[CheckResult(name="schema", passed=True)]
        )
        assert passing.status is ValidationStatus.PASS
        with pytest.raises(ValueError, match="only for failures"):
            validation_mod.ValidationAgent(
                ScriptedChatCompletionClient([]), report=passing, plan=plan
            )

    def test_diagnosis_model_cannot_express_a_verdict(self) -> None:
        """A structural guarantee, not a prompt instruction: there is no field on
        `ValidationDiagnosis` that could flip a FAIL to a PASS."""
        fields = set(ValidationDiagnosis.model_fields)
        assert not fields & {"status", "passed", "valid", "approved", "override"}


class TestValidationAgentBehaviour:
    async def test_reads_the_report_and_the_implicated_step(
        self, factory: ScriptedModelClientFactory, plan: MigrationPlan
    ) -> None:
        agent = validation_mod.ValidationAgent(
            factory.client_for("validation"), report=failing_report(), plan=plan
        )
        run = await agent.run(validation_mod.diagnosis_task(failing_report(), plan))
        assert "get_validation_report" in run.tools_used
        assert "get_plan_step" in run.tools_used

    async def test_identifies_the_single_root_cause(
        self, factory: ScriptedModelClientFactory, plan: MigrationPlan
    ) -> None:
        """Two failing checks, one mechanism. Reporting two independent problems
        would send the repair loop in two directions."""
        agent = validation_mod.ValidationAgent(
            factory.client_for("validation"), report=failing_report(), plan=plan
        )
        diagnosis = (await agent.run(validation_mod.diagnosis_task(failing_report(), plan))).output

        assert diagnosis.root_cause_category is RiskCategory.NULL_SEMANTICS
        assert diagnosis.implicated_step_ids == ["s6"]
        assert diagnosis.confidence > 0.5

    async def test_evidence_quotes_measured_values(
        self, factory: ScriptedModelClientFactory, plan: MigrationPlan
    ) -> None:
        """Evidence not traceable to the report is a guess wearing a citation."""
        agent = validation_mod.ValidationAgent(
            factory.client_for("validation"), report=failing_report(), plan=plan
        )
        diagnosis = (await agent.run(validation_mod.diagnosis_task(failing_report(), plan))).output
        assert any("reference=5" in item and "candidate=6" in item for item in diagnosis.evidence)

    async def test_suggests_a_concrete_code_change(
        self, factory: ScriptedModelClientFactory, plan: MigrationPlan
    ) -> None:
        agent = validation_mod.ValidationAgent(
            factory.client_for("validation"), report=failing_report(), plan=plan
        )
        diagnosis = (await agent.run(validation_mod.diagnosis_task(failing_report(), plan))).output
        assert "isNotNull" in diagnosis.suggested_fix
        assert "groupBy" in diagnosis.suggested_fix

    def test_unknown_check_lookup_lists_alternatives(self, plan: MigrationPlan) -> None:
        agent = validation_mod.ValidationAgent(
            ScriptedChatCompletionClient([]), report=failing_report(), plan=plan
        )
        result = agent._make_failed_check_tool()("does_not_exist")
        assert "Available:" in result
