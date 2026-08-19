"""End-to-end local pipeline: legacy source in, gate-clean PySpark out. Offline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from etl_migrator.config import Settings
from etl_migrator.domain import lifecycle
from etl_migrator.domain.artifacts import MigrationRecord
from etl_migrator.domain.enums import MigrationStage, RiskLevel
from etl_migrator.domain.errors import StaticGateError
from etl_migrator.domain.plan import MigrationPlan
from etl_migrator.ids import new_migration_id
from etl_migrator.llm.factory import ScriptedModelClientFactory
from etl_migrator.pipeline.local import LocalMigrationPipeline, MigrationRequest
from etl_migrator.tools.code_gate import analyze_generated_code


@pytest.fixture
async def record(
    settings: Settings,
    factory: ScriptedModelClientFactory,
    legacy_source: Path,
    example_input_dir: Path,
) -> MigrationRecord:
    # Generation-only: validation executes Spark and is covered by the
    # spark-marked suite in test_validation_e2e.py.
    pipeline = LocalMigrationPipeline(settings, factory, run_validation=False)
    request = MigrationRequest(legacy_source, example_input_dir)
    return await pipeline.run(request)


class TestEndToEnd:
    async def test_completes_and_reaches_static_analysis(self, record: MigrationRecord) -> None:
        assert not record.failed
        assert record.stage is MigrationStage.STATIC_ANALYSIS
        assert record.spec is not None and record.plan is not None and record.codegen is not None

    async def test_generated_code_passes_an_independent_gate(
        self, record: MigrationRecord
    ) -> None:
        assert record.codegen is not None
        assert analyze_generated_code(record.codegen.code.content).passed

    async def test_records_that_the_agent_needed_two_attempts(
        self, record: MigrationRecord
    ) -> None:
        assert record.codegen is not None
        assert record.codegen.gate_iterations == 2

    async def test_every_stage_is_timed(self, record: MigrationRecord) -> None:
        stages = [s.stage for s in record.stages]
        assert stages == [
            MigrationStage.DISCOVERY,
            MigrationStage.PLANNING,
            MigrationStage.APPROVAL,
            MigrationStage.CODE_GENERATION,
            MigrationStage.STATIC_ANALYSIS,
        ]
        assert all(s.succeeded and s.duration_seconds is not None for s in record.stages)
        assert record.total_duration_seconds > 0

    async def test_correlation_id_is_unique_and_sortable(self) -> None:
        first, second = new_migration_id(), new_migration_id()
        assert first != second
        assert first.startswith("mig-")


class TestArtifacts:
    async def test_writes_the_full_artifact_set(self, record: MigrationRecord) -> None:
        assert record.artifact_dir is not None
        artifacts = Path(record.artifact_dir)
        expected = {
            "migration_spec.json",
            "migration_plan.json",
            "static_analysis.json",
            "migration_record.json",
            "agent_trace.json",
            "legacy_pipeline_spark.py",
        }
        assert expected <= {p.name for p in artifacts.iterdir()}

    async def test_agent_trace_records_which_tools_ran(self, record: MigrationRecord) -> None:
        assert record.artifact_dir is not None
        trace = json.loads((Path(record.artifact_dir) / "agent_trace.json").read_text())
        assert set(trace) == {"discovery", "planner", "spark_engineer"}
        assert "inspect_legacy_source" in trace["discovery"]["tools_used"]
        assert trace["spark_engineer"]["tools_used"].count("check_spark_code") == 2

    async def test_record_round_trips_through_json(self, record: MigrationRecord) -> None:
        """Temporal stores this as workflow state, so it must serialise."""
        assert record.artifact_dir is not None
        raw = (Path(record.artifact_dir) / "migration_record.json").read_text()
        restored = MigrationRecord.model_validate_json(raw)
        assert restored.migration_id == record.migration_id
        assert restored.plan is not None


class TestApprovalPolicy:
    async def test_high_risk_migration_requires_approval(self, record: MigrationRecord) -> None:
        assert record.plan is not None
        assert record.plan.overall_risk is RiskLevel.HIGH
        assert record.plan.requires_human_approval is True

    def test_agent_cannot_waive_its_own_approval_gate(self, fixture_payload: dict) -> None:
        """An agent that sets requires_human_approval=False on a HIGH-risk plan
        is overruled by policy, not believed."""
        plan = MigrationPlan.model_validate(fixture_payload["planner"][-1]["content"])
        sneaky = plan.model_copy(update={"requires_human_approval": False})
        enforced = lifecycle.enforce_approval_policy(sneaky)
        assert enforced.requires_human_approval is True

    def test_low_risk_migration_does_not_require_approval(self, fixture_payload: dict) -> None:
        plan = MigrationPlan.model_validate(fixture_payload["planner"][-1]["content"])
        low = plan.model_copy(
            update={"overall_risk": RiskLevel.LOW, "requires_human_approval": True}
        )
        assert lifecycle.enforce_approval_policy(low).requires_human_approval is False


class TestFailureHandling:
    async def test_lying_agent_is_caught_by_the_independent_gate(
        self,
        settings: Settings,
        legacy_source: Path,
        example_input_dir: Path,
        fixture_payload: dict,
        tmp_path: Path,
    ) -> None:
        """If an agent returns code it never validated, the re-run gate fails the
        migration rather than shipping it."""
        payload = json.loads(json.dumps(fixture_payload))
        payload["spark_engineer"] = [
            {
                "content": {
                    "filename": "bad.py",
                    "content": (
                        "import os\n"
                        "def run(spark, input_dir, output_dir):\n"
                        "    os.system('x')\n"
                    ),
                    "entrypoint": "run",
                }
            }
        ]
        fixture_file = tmp_path / "lying.json"
        fixture_file.write_text(json.dumps(payload), encoding="utf-8")

        factory = ScriptedModelClientFactory.from_fixture(fixture_file)
        pipeline = LocalMigrationPipeline(settings, factory, run_validation=False)
        with pytest.raises(StaticGateError, match="failed the independent static gate"):
            await pipeline.run(MigrationRequest(legacy_source, example_input_dir))

    async def test_failure_is_recorded_on_the_stage(
        self,
        settings: Settings,
        legacy_source: Path,
        example_input_dir: Path,
        tmp_path: Path,
    ) -> None:
        fixture_file = tmp_path / "empty.json"
        fixture_file.write_text(json.dumps({"discovery": []}), encoding="utf-8")
        factory = ScriptedModelClientFactory.from_fixture(fixture_file)
        pipeline = LocalMigrationPipeline(settings, factory, run_validation=False)
        with pytest.raises(Exception, match="exhausted"):
            await pipeline.run(MigrationRequest(legacy_source, example_input_dir))
