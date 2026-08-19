"""Builders for `MigrationRecord` and its parts.

Several suites need a record shaped a particular way — a failed validation, a
rejected optimisation, a repair that exhausted. Building them inline in each
file drifts: one copy gains a field, another does not, and a test starts passing
for a reason nobody intended.

Everything here builds *real* models through their real constructors. Nothing is
stubbed or mocked, which matters because the verdicts are computed fields: the
only way to get a FAIL out of `ValidationReport` is to fail a check, exactly as
the differ would. A helper that could hand back an arbitrary status would let a
test assert something the system cannot actually produce.

Named `helpers_records` rather than `test_records` so pytest does not try to
collect it.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from etl_migrator.domain.artifacts import CodeGenResult, MigrationRecord
from etl_migrator.domain.code import GeneratedCode, StaticAnalysisReport
from etl_migrator.domain.delivery import DeliveryOutcome
from etl_migrator.domain.enums import RiskCategory, RiskLevel, TransformKind, ValidationStatus
from etl_migrator.domain.optimization import (
    BenchmarkResult,
    OptimizationOutcome,
    OptimizationStrategy,
)
from etl_migrator.domain.plan import (
    ExecutionStrategy,
    MigrationPlan,
    PlanStep,
    SemanticDifference,
    ValidationPlan,
)
from etl_migrator.domain.repair import RepairOutcome
from etl_migrator.domain.validation import (
    CheckResult,
    DatasetStats,
    ValidationOutcome,
    ValidationReport,
)

EPOCH = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)


def at(seconds: float) -> datetime:
    """A deterministic timestamp, `seconds` after a fixed epoch."""
    return EPOCH + timedelta(seconds=seconds)


def benchmark(durations: list[float], *, label: str = "b") -> BenchmarkResult:
    return BenchmarkResult(label=label, durations=durations)


def strategy(approach: str, *, expected: float = 1.5) -> OptimizationStrategy:
    return OptimizationStrategy(
        approach=approach,
        description=f"Apply {approach}.",
        rationale="Grounded in the measured baseline.",
        expected_speedup=expected,
    )


def plan(risk: RiskLevel = RiskLevel.HIGH) -> MigrationPlan:
    """A minimal plan carrying one declared semantic difference."""
    return MigrationPlan(
        summary="Migrate the revenue pipeline to PySpark.",
        steps=[
            PlanStep(
                id="s1",
                transformation_id="t1",
                kind=TransformKind.AGGREGATE,
                legacy_construct="pandas.DataFrame.groupby",
                spark_construct="DataFrame.groupBy",
                rationale="Aggregate revenue per country.",
                semantic_differences=[
                    SemanticDifference(
                        category=RiskCategory.NULL_SEMANTICS,
                        description="pandas drops null group keys; Spark keeps them.",
                        mitigation="Filter null countries before the groupBy.",
                        validation_check="row_count",
                    )
                ],
                risk_level=risk,
            )
        ],
        execution_strategy=ExecutionStrategy(),
        validation_plan=ValidationPlan(required_checks=["row_count"]),
        overall_risk=risk,
    )


def validation_outcome(
    status: ValidationStatus = ValidationStatus.PASS,
    *,
    checks: list[CheckResult] | None = None,
    rows: int = 5,
) -> ValidationOutcome:
    """A validation outcome whose status is *produced*, never asserted.

    When `checks` is given it is used verbatim and the resulting status is
    whatever the checks imply — which is the point. Otherwise a minimal set is
    built to produce `status`, and an assertion confirms it did.
    """
    if checks is None:
        if status is ValidationStatus.PASS:
            checks = [CheckResult(name="row_count", passed=True)]
        elif status is ValidationStatus.FAIL:
            checks = [CheckResult(name="row_count", passed=False, detail="4 != 5")]
        else:
            checks = [CheckResult(name="row_count", passed=False, skipped=True)]
        expected: ValidationStatus | None = status
    else:
        expected = None

    report = ValidationReport(
        migration_id="mig-1",
        checks=checks,
        reference=DatasetStats(path="reference", row_count=rows),
        candidate=DatasetStats(path="candidate", row_count=rows),
    )
    if expected is not None:
        assert report.status is expected, "the helper did not produce the status it claims"
    return ValidationOutcome(report=report)


def migration_record(
    *,
    migration_id: str = "mig-1",
    failed: bool = False,
    risk: RiskLevel = RiskLevel.HIGH,
    with_code: bool = True,
    with_validation: bool = True,
    status: ValidationStatus = ValidationStatus.PASS,
    checks: list[CheckResult] | None = None,
    repair: RepairOutcome | None = None,
    optimization: OptimizationOutcome | None = None,
    delivery: DeliveryOutcome | None = None,
) -> MigrationRecord:
    """A finished record, shaped by the keywords."""
    record = MigrationRecord(
        migration_id=migration_id, source_path="examples/legacy_pipeline.py", created_at=at(0)
    )
    record.plan = plan(risk)
    if with_code:
        record.codegen = CodeGenResult(
            code=GeneratedCode(filename="pipeline_spark.py", content="def run(): ...\n"),
            static_analysis=StaticAnalysisReport(passed=True),
            gate_iterations=1,
        )
    if with_validation:
        record.validation = validation_outcome(status, checks=checks)
    record.repair = repair
    record.optimization = optimization
    record.delivery = delivery
    record.failed = failed
    if failed:
        record.failure_reason = "the sandbox timed out"
    return record
