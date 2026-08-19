"""End-to-end validation: both pipelines executed, tests run, outputs diffed.

This is the test that justifies the validation tier. Everything upstream produces a
*claim* that a migration is correct; this executes both implementations against
the same data and checks the claim.

The counterfactual class at the bottom matters as much as the happy path. A
validation stage that passes everything is indistinguishable from no validation
stage, so these deliberately break the generated pipeline in the exact ways the
plan warned about, and assert that each one is caught.

Marked `spark` because it needs a JVM:  pytest -m "not spark"  skips it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from etl_migrator.config import Settings
from etl_migrator.domain.artifacts import GeneratedCode
from etl_migrator.domain.enums import RiskCategory, ValidationStatus
from etl_migrator.domain.plan import MigrationPlan
from etl_migrator.domain.spec import MigrationSpec
from etl_migrator.domain.validation import GeneratedTests
from etl_migrator.llm.factory import ScriptedModelClientFactory
from etl_migrator.pipeline import steps
from etl_migrator.pipeline.local import LocalMigrationPipeline, MigrationRequest
from etl_migrator.sandbox.execute import spark_conf_from
from etl_migrator.sandbox.pytest_runner import run_generated_tests

pytestmark = pytest.mark.spark

pytest.importorskip("pyspark", reason="pyspark extra not installed")


#: The two mitigations that make the customer migration correct. Removing either
#: is precisely the mistake a naive translation makes.
DROPNA_EMULATION = (
    'joined.filter(F.col("country").isNotNull())\n        .groupBy("country")'
)
COALESCE_EMULATION = 'F.coalesce(F.sum("revenue"), F.lit(0.0)).alias("revenue")'


@pytest.fixture(scope="module")
def spec(fixture_payload: dict) -> MigrationSpec:
    return MigrationSpec.model_validate(fixture_payload["discovery"][-1]["content"])


@pytest.fixture(scope="module")
def plan(fixture_payload: dict) -> MigrationPlan:
    return MigrationPlan.model_validate(fixture_payload["planner"][-1]["content"])


@pytest.fixture(scope="module")
def code(fixture_payload: dict) -> GeneratedCode:
    return GeneratedCode.model_validate(fixture_payload["spark_engineer"][-1]["content"])


@pytest.fixture(scope="module")
def generated_tests(fixture_payload: dict) -> GeneratedTests:
    return GeneratedTests.model_validate(fixture_payload["testing"][-1]["content"])


def break_dropna(source: str) -> str:
    broken = source.replace(DROPNA_EMULATION, 'joined.groupBy("country")')
    assert broken != source, "the dropna mitigation moved; update this counterfactual"
    return broken


def break_coalesce(source: str) -> str:
    broken = source.replace(COALESCE_EMULATION, 'F.sum("revenue").alias("revenue")')
    assert broken != source, "the coalesce mitigation moved; update this counterfactual"
    return broken


def validate_source(
    source: str,
    *,
    tmp_path: Path,
    plan: MigrationPlan,
    example_input_dir: Path,
    legacy_source: Path,
    name: str,
):
    """Run both pipelines against the same input and diff their outputs."""
    module = steps.materialize(tmp_path / name, "pipeline_spark.py", source)

    legacy = steps.execute_legacy(
        source_path=legacy_source,
        input_dir=example_input_dir,
        output_dir=tmp_path / name / "reference",
    )
    assert legacy.succeeded, legacy.error

    spark = steps.execute_spark(
        module_path=module,
        input_dir=example_input_dir,
        output_dir=tmp_path / name / "candidate",
        strategy=plan.execution_strategy,
    )
    assert spark.succeeded, spark.error

    return steps.diff_outputs(
        migration_id=f"mig-{name}",
        reference_path=steps.resolve_output_root(tmp_path / name / "reference"),
        candidate_path=steps.resolve_output_root(tmp_path / name / "candidate"),
        plan=plan,
    )


class TestCorrectMigrationPasses:
    @pytest.fixture(scope="class")
    @classmethod
    def report(
        cls,
        tmp_path_factory: pytest.TempPathFactory,
        plan: MigrationPlan,
        code: GeneratedCode,
        example_input_dir: Path,
        legacy_source: Path,
    ):
        return validate_source(
            code.content,
            tmp_path=tmp_path_factory.mktemp("good"),
            plan=plan,
            example_input_dir=example_input_dir,
            legacy_source=legacy_source,
            name="good",
        )

    def test_status_is_pass(self, report) -> None:
        assert report.status is ValidationStatus.PASS, report.render()

    def test_every_required_check_actually_ran(
        self, report, plan: MigrationPlan
    ) -> None:
        """A check that silently did not run must not be counted as a pass."""
        ran = {c.name for c in report.checks if not c.skipped}
        assert set(plan.effective_required_checks()) <= ran

    def test_no_differences(self, report) -> None:
        assert report.differences == []

    def test_flat_summary_matches_the_documented_shape(self, report) -> None:
        assert report.schema_match
        assert report.row_count_match
        assert report.numeric_tolerance_passed


class TestBrokenMigrationsAreCaught:
    """Each case removes one mitigation the plan declared and asserts it is caught."""

    def test_missing_dropna_emulation_is_caught(
        self,
        tmp_path: Path,
        plan: MigrationPlan,
        code: GeneratedCode,
        example_input_dir: Path,
        legacy_source: Path,
    ) -> None:
        """The headline failure: a phantom null-country group carrying real
        revenue, produced with no error and no warning."""
        report = validate_source(
            break_dropna(code.content),
            tmp_path=tmp_path,
            plan=plan,
            example_input_dir=example_input_dir,
            legacy_source=legacy_source,
            name="no-dropna",
        )
        assert report.status is ValidationStatus.FAIL
        assert not report.row_count_match

        failed = {c.name for c in report.failed_checks}
        assert {"row_count", "aggregate_checksums"} <= failed
        assert any(
            d.category is RiskCategory.NULL_SEMANTICS for d in report.differences
        )

    def test_the_revenue_total_difference_is_reported_exactly(
        self,
        tmp_path: Path,
        plan: MigrationPlan,
        code: GeneratedCode,
        example_input_dir: Path,
        legacy_source: Path,
    ) -> None:
        """The differ must say *how much* is wrong, not just that something is —
        that number is what a human uses to judge the blast radius."""
        report = validate_source(
            break_dropna(code.content),
            tmp_path=tmp_path,
            plan=plan,
            example_input_dir=example_input_dir,
            legacy_source=legacy_source,
            name="revenue-delta",
        )
        revenue = next(
            d
            for d in report.differences
            if d.check == "aggregate_checksums" and d.column == "revenue"
        )
        assert float(revenue.candidate) > float(revenue.reference)


class TestGeneratedSuiteIsNotVacuous:
    """The generated tests must fail on a broken pipeline.

    A suite that passes against anything is worse than no suite: it reports
    assurance it does not provide.
    """

    @pytest.fixture(scope="class")
    @classmethod
    def pipeline_dir(cls, tmp_path_factory: pytest.TempPathFactory) -> Path:
        return tmp_path_factory.mktemp("suite")

    def test_suite_passes_against_the_correct_pipeline(
        self,
        pipeline_dir: Path,
        code: GeneratedCode,
        generated_tests: GeneratedTests,
        plan: MigrationPlan,
        example_input_dir: Path,
    ) -> None:
        module = steps.materialize(pipeline_dir / "good", code.filename, code.content)
        result = run_generated_tests(
            test_source=generated_tests.content,
            test_filename=generated_tests.filename,
            pipeline_path=module,
            input_dir=example_input_dir,
            spark_conf=spark_conf_from(plan.execution_strategy),
        )
        assert result.succeeded, result.output_tail
        assert result.passed == len(generated_tests.test_names)
        assert result.failed == 0

    def test_suite_fails_against_a_broken_pipeline(
        self,
        pipeline_dir: Path,
        code: GeneratedCode,
        generated_tests: GeneratedTests,
        plan: MigrationPlan,
        example_input_dir: Path,
    ) -> None:
        module = steps.materialize(
            pipeline_dir / "broken", code.filename, break_coalesce(break_dropna(code.content))
        )
        result = run_generated_tests(
            test_source=generated_tests.content,
            test_filename=generated_tests.filename,
            pipeline_path=module,
            input_dir=example_input_dir,
            spark_conf=spark_conf_from(plan.execution_strategy),
        )
        assert not result.succeeded
        assert result.failed >= 2, "both declared mitigations should have a failing test"


class TestFullLocalPipelineWithValidation:
    """The whole lifecycle in one run: generation through to a measured verdict."""

    async def test_migration_completes_and_validates(
        self,
        settings: Settings,
        factory: ScriptedModelClientFactory,
        legacy_source: Path,
        example_input_dir: Path,
    ) -> None:
        record = await LocalMigrationPipeline(
            settings, factory, run_validation=True, run_generated_tests=True
        ).run(MigrationRequest(legacy_source, example_input_dir))

        assert not record.failed, record.failure_reason
        assert record.validation is not None
        assert record.validation.passed
        assert record.validation.report.status is ValidationStatus.PASS

        assert record.validation.legacy_execution is not None
        assert record.validation.spark_execution is not None
        assert record.validation.spark_execution.metrics["engine"] == "spark"

        assert record.validation.test_run is not None
        assert record.validation.test_run.succeeded

        # No failure, so no diagnosis: the agent is never invited to opine on a
        # migration that already passed.
        assert record.validation.diagnosis is None

    async def test_artifacts_include_the_executed_outputs(
        self,
        settings: Settings,
        factory: ScriptedModelClientFactory,
        legacy_source: Path,
        example_input_dir: Path,
    ) -> None:
        record = await LocalMigrationPipeline(
            settings, factory, run_validation=True, run_generated_tests=False
        ).run(MigrationRequest(legacy_source, example_input_dir))

        assert record.artifact_dir is not None
        artifacts = Path(record.artifact_dir)
        assert (artifacts / "reference_output").is_dir()
        assert (artifacts / "candidate_output").is_dir()
        assert (artifacts / "migration_record.json").is_file()
