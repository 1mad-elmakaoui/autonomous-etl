"""Activity tests, driven through Temporal's real `ActivityEnvironment`.

No server is involved, but nothing is faked either: these are the actual
`@activity.defn` callables the worker registers, invoked the way the worker
invokes them, with `activity.info()` populated.

What is being checked is the activity contract — one argument in, one
serialisable value out, no lifecycle decisions taken — rather than the agent
behaviour, which `test_agents.py` already covers.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from temporalio.testing import ActivityEnvironment

from etl_migrator.activities.migration import (
    DeliveryActivities,
    MigrationActivities,
    ValidationActivities,
)
from etl_migrator.config import Settings
from etl_migrator.domain.artifacts import GeneratedCode, MigrationRecord
from etl_migrator.domain.enums import MigrationStage, RiskLevel
from etl_migrator.domain.messages import (
    AgentTelemetry,
    CodegenInput,
    DiscoveryInput,
    PersistInput,
    PlanningInput,
    StaticAnalysisInput,
)
from etl_migrator.domain.plan import MigrationPlan
from etl_migrator.domain.spec import MigrationSpec
from etl_migrator.llm.factory import ScriptedModelClientFactory


@pytest.fixture
def env() -> ActivityEnvironment:
    return ActivityEnvironment()


@pytest.fixture
def activities(settings: Settings) -> MigrationActivities:
    """Activities that build a fresh scripted factory per call.

    A shared factory would hand out a client whose script cursor has already
    advanced — which is exactly what a Temporal retry must not do.
    """
    return MigrationActivities(settings)


@pytest.fixture
def spec(fixture_payload: dict) -> MigrationSpec:
    return MigrationSpec.model_validate(fixture_payload["discovery"][-1]["content"])


@pytest.fixture
def plan(fixture_payload: dict) -> MigrationPlan:
    return MigrationPlan.model_validate(fixture_payload["planner"][-1]["content"])


@pytest.fixture
def generated(fixture_payload: dict) -> GeneratedCode:
    return GeneratedCode.model_validate(fixture_payload["spark_engineer"][-1]["content"])


class TestDiscoveryActivity:
    async def test_returns_spec_and_telemetry(
        self,
        env: ActivityEnvironment,
        activities: MigrationActivities,
        legacy_source: Path,
        example_input_dir: Path,
    ) -> None:
        out = await env.run(
            activities.analyze_legacy_pipeline,
            DiscoveryInput(
                migration_id="mig-1",
                source_path=str(legacy_source),
                input_dir=str(example_input_dir),
                scenario="customer_pipeline",
            ),
        )
        assert isinstance(out.spec, MigrationSpec)
        assert out.spec.max_risk is RiskLevel.HIGH
        assert "inspect_legacy_source" in out.telemetry.tools_used
        assert out.telemetry.agent == "discovery"

    async def test_output_round_trips_as_json(
        self,
        env: ActivityEnvironment,
        activities: MigrationActivities,
        legacy_source: Path,
        example_input_dir: Path,
    ) -> None:
        """Activity results cross a process boundary; if a field cannot be
        serialised, the failure must show up here and not in production."""
        out = await env.run(
            activities.analyze_legacy_pipeline,
            DiscoveryInput(
                migration_id="mig-1",
                source_path=str(legacy_source),
                input_dir=str(example_input_dir),
                scenario="customer_pipeline",
            ),
        )
        assert out.model_validate_json(out.model_dump_json()) == out


class TestPlanningActivity:
    async def test_produces_a_plan_and_measures_inputs_itself(
        self,
        env: ActivityEnvironment,
        activities: MigrationActivities,
        example_input_dir: Path,
        spec: MigrationSpec,
    ) -> None:
        out = await env.run(
            activities.generate_migration_plan,
            PlanningInput(
                migration_id="mig-1",
                input_dir=str(example_input_dir),
                scenario="customer_pipeline",
                spec=spec,
            ),
        )
        assert isinstance(out.plan, MigrationPlan)
        assert out.plan.overall_risk is RiskLevel.HIGH
        assert "dataset_sizes" in out.telemetry.tools_used

    async def test_activity_returns_the_agent_plan_untouched(
        self,
        env: ActivityEnvironment,
        activities: MigrationActivities,
        example_input_dir: Path,
        spec: MigrationSpec,
        plan: MigrationPlan,
    ) -> None:
        """Approval policy belongs to the workflow, where the decision is durable.

        The activity must hand back exactly what the agent produced, so the
        comparison is against the recorded plan itself.
        """
        out = await env.run(
            activities.generate_migration_plan,
            PlanningInput(
                migration_id="mig-1",
                input_dir=str(example_input_dir),
                scenario="customer_pipeline",
                spec=spec,
            ),
        )
        assert out.plan == plan


class TestCodegenActivity:
    async def test_generates_code_and_reports_gate_iterations(
        self,
        env: ActivityEnvironment,
        activities: MigrationActivities,
        spec: MigrationSpec,
        plan: MigrationPlan,
    ) -> None:
        out = await env.run(
            activities.generate_spark_code,
            CodegenInput(
                migration_id="mig-1",
                scenario="customer_pipeline",
                output_filename="pipeline_spark.py",
                spec=spec,
                plan=plan,
            ),
        )
        assert out.code.filename == "pipeline_spark.py"
        assert out.gate_iterations == 2, "the recorded first submission is rejected"
        assert out.telemetry.gate_iterations == 2


class TestStaticAnalysisActivity:
    async def test_passes_clean_code(
        self,
        env: ActivityEnvironment,
        activities: MigrationActivities,
        generated: GeneratedCode,
    ) -> None:
        out = await env.run(
            activities.run_static_analysis,
            StaticAnalysisInput(migration_id="mig-1", code=generated, gate_iterations=2),
        )
        assert out.passed
        assert out.result.gate_iterations == 2

    async def test_reports_failure_as_data_rather_than_raising(
        self, env: ActivityEnvironment, activities: MigrationActivities
    ) -> None:
        """A failing gate is an outcome the workflow reasons about, since it
        routes to repair. Raising would hand that decision to a retry policy."""
        bad = GeneratedCode(
            filename="bad.py",
            content="import os\ndef run(spark, input_dir, output_dir):\n    os.system('x')\n",
        )
        out = await env.run(
            activities.run_static_analysis,
            StaticAnalysisInput(migration_id="mig-1", code=bad),
        )
        assert out.passed is False
        assert {f.code for f in out.report.errors} >= {"GATE002"}


class TestPersistActivity:
    def _record(self, spec: MigrationSpec, plan: MigrationPlan) -> MigrationRecord:
        from datetime import UTC, datetime

        record = MigrationRecord(
            migration_id="mig-persist",
            source_path="legacy.py",
            created_at=datetime(2026, 8, 10, tzinfo=UTC),
            stage=MigrationStage.STATIC_ANALYSIS,
        )
        record.spec = spec
        record.plan = plan
        return record

    async def test_writes_the_artifact_set(
        self,
        env: ActivityEnvironment,
        activities: MigrationActivities,
        spec: MigrationSpec,
        plan: MigrationPlan,
    ) -> None:
        record = self._record(spec, plan)
        out = await env.run(
            activities.persist_artifacts,
            PersistInput(
                migration_id="mig-persist",
                record_json=record.model_dump_json(),
                telemetry=[AgentTelemetry(agent="discovery", tools_used=["inspect"])],
            ),
        )
        written = set(out.files_written)
        assert {"migration_spec.json", "migration_plan.json", "migration_record.json",
                "agent_trace.json"} <= written

        trace = json.loads((Path(out.artifact_dir) / "agent_trace.json").read_text())
        assert trace["discovery"]["tools_used"] == ["inspect"]

    async def test_is_idempotent_under_retry(
        self,
        env: ActivityEnvironment,
        activities: MigrationActivities,
        spec: MigrationSpec,
        plan: MigrationPlan,
    ) -> None:
        """Temporal delivers activities at least once. Running this twice must
        produce byte-identical output, not duplicates."""
        record = self._record(spec, plan)
        params = PersistInput(migration_id="mig-persist", record_json=record.model_dump_json())

        first = await env.run(activities.persist_artifacts, params)
        before = (Path(first.artifact_dir) / "migration_record.json").read_bytes()

        second = await env.run(activities.persist_artifacts, params)
        after = (Path(second.artifact_dir) / "migration_record.json").read_bytes()

        assert first.files_written == second.files_written
        assert before == after

    async def test_persists_a_failed_record_too(
        self,
        env: ActivityEnvironment,
        activities: MigrationActivities,
        spec: MigrationSpec,
    ) -> None:
        """A failed migration needs *more* artifacts than a successful one — they
        are what a human uses to intervene."""
        from datetime import UTC, datetime

        record = MigrationRecord(
            migration_id="mig-failed",
            source_path="legacy.py",
            created_at=datetime(2026, 8, 10, tzinfo=UTC),
            failed=True,
            failure_reason="rejected by amira",
        )
        record.spec = spec
        out = await env.run(
            activities.persist_artifacts,
            PersistInput(migration_id="mig-failed", record_json=record.model_dump_json()),
        )
        stored = MigrationRecord.model_validate_json(
            (Path(out.artifact_dir) / "migration_record.json").read_text()
        )
        assert stored.failed and stored.failure_reason == "rejected by amira"
        assert "migration_spec.json" in out.files_written


class TestRegistration:
    def test_all_activities_are_registered_and_named(
        self, activities: MigrationActivities, settings: Settings
    ) -> None:
        from temporalio.activity import _Definition

        from etl_migrator.temporal.worker import ACTIVITY_NAMES

        registered = {
            _Definition.must_from_callable(a).name
            for a in [
                *activities.all(),
                *ValidationActivities(settings).all(),
                *DeliveryActivities(settings).all(),
            ]
        }
        assert registered == set(ACTIVITY_NAMES)

    def test_scripted_factory_is_rebuilt_per_activity(
        self, settings: Settings, fixture_payload: dict
    ) -> None:
        """An injected factory is reused; the default path builds a fresh one so a
        retried attempt replays the script from turn zero."""
        acts = MigrationActivities(settings)
        first = acts._context("customer_pipeline").factory
        second = acts._context("customer_pipeline").factory
        assert first is not second

        shared = ScriptedModelClientFactory.from_fixture(
            settings.llm_fixture_dir / "customer_pipeline.json"
        )
        pinned = MigrationActivities(settings, factory=shared)
        assert pinned._context("customer_pipeline").factory is shared
