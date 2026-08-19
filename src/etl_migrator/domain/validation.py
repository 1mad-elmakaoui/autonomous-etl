"""Validation contracts: execution results, the differ's verdict, and diagnosis.

The single most important rule in this file is the split between
`ValidationReport` and `ValidationDiagnosis`:

* `ValidationReport` is produced by the **differ**, deterministically, from two
  executed outputs. Its `status` is computed, never asserted. No agent can
  construct one.
* `ValidationDiagnosis` is produced by the **Validation agent**, and only when
  the report already says FAIL. It explains *why*, and proposes what to change.
  It cannot make a failure into a pass.

That asymmetry is the whole reason this system is not a confident-sounding code
translator. The model gets to interpret evidence; it never gets to be evidence.
"""

from __future__ import annotations

from pydantic import Field, computed_field

from etl_migrator.domain.enums import RiskCategory, ValidationStatus
from etl_migrator.domain.spec import StrictModel


class ExecutionResult(StrictModel):
    """Outcome of running one pipeline in the sandbox."""

    engine: str = Field(description="'pandas' or 'spark'.")
    succeeded: bool
    duration_seconds: float = Field(ge=0.0)
    output_path: str | None = None
    stdout_tail: str = ""
    stderr_tail: str = ""
    exit_code: int | None = None
    timed_out: bool = False
    error: str | None = None
    metrics: dict[str, str] = Field(
        default_factory=dict,
        description="Engine-reported numbers (Spark conf, partition counts). "
        "Phase 5 reads these for benchmarking.",
    )

    def render(self) -> str:
        head = (
            f"{self.engine}: {'ok' if self.succeeded else 'FAILED'} "
            f"in {self.duration_seconds:.2f}s"
        )
        if self.succeeded:
            return f"{head} -> {self.output_path}"
        detail = self.error or self.stderr_tail or f"exit code {self.exit_code}"
        return f"{head}\n{detail}"


class ColumnStat(StrictModel):
    """One column's measured shape, computed identically on both sides."""

    name: str
    dtype: str
    null_count: int = Field(ge=0)
    distinct_count: int = Field(ge=0)
    numeric_sum: float | None = None
    numeric_min: float | None = None
    numeric_max: float | None = None


class DatasetStats(StrictModel):
    """Everything the differ measures about one side of the comparison."""

    path: str
    row_count: int = Field(ge=0)
    columns: list[ColumnStat] = Field(default_factory=list)
    duplicate_row_count: int = Field(default=0, ge=0)
    duplicate_key_count: int = Field(default=0, ge=0)

    @property
    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]

    def column(self, name: str) -> ColumnStat | None:
        return next((c for c in self.columns if c.name == name), None)


class Difference(StrictModel):
    """One concrete, located disagreement between reference and candidate.

    Every field here exists because the repair agent needs it: `check` says
    which invariant broke, `column` narrows it, `category` maps it onto a
    root-cause class, and `reference`/`candidate` are the actual values so a
    human reading the PR does not have to re-run anything.
    """

    check: str = Field(description="Check that produced it, e.g. 'row_count'.")
    column: str | None = None
    category: RiskCategory | None = Field(
        default=None,
        description="Root-cause class, when the differ can infer it. Drives the "
        "phase 4 repair strategy, which must be distinct per class.",
    )
    reference: str
    candidate: str
    detail: str

    def render(self) -> str:
        where = f" [{self.column}]" if self.column else ""
        return (
            f"{self.check}{where}: reference={self.reference} "
            f"candidate={self.candidate} — {self.detail}"
        )


class CheckResult(StrictModel):
    """Outcome of one named check from the plan's `required_checks`."""

    name: str
    passed: bool
    detail: str = ""
    differences: list[Difference] = Field(default_factory=list)
    skipped: bool = Field(
        default=False,
        description="True when the check could not run (e.g. no join keys declared). "
        "A skipped check is never counted as a pass.",
    )


class ValidationReport(StrictModel):
    """The differ's verdict. Deterministic, computed, not asserted.

    The four flat booleans mirror the shape the brief asks for and are the
    summary a PR body quotes; `checks` carries the full detail underneath.
    """

    migration_id: str
    reference: DatasetStats | None = None
    candidate: DatasetStats | None = None
    checks: list[CheckResult] = Field(default_factory=list)
    error: str | None = Field(
        default=None, description="Set when validation could not run at all."
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def status(self) -> ValidationStatus:
        """PASS only if every required check ran and passed.

        A check that could not run is ERROR, not PASS. Treating "we did not
        measure it" as "it is fine" is how silently-wrong migrations ship.
        """
        if self.error is not None:
            return ValidationStatus.ERROR
        if not self.checks:
            return ValidationStatus.ERROR
        if any(c.skipped for c in self.checks):
            return ValidationStatus.ERROR
        return (
            ValidationStatus.PASS
            if all(c.passed for c in self.checks)
            else ValidationStatus.FAIL
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def schema_match(self) -> bool:
        return self._check_passed("schema")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def row_count_match(self) -> bool:
        return self._check_passed("row_count")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def numeric_tolerance_passed(self) -> bool:
        return self._check_passed("numeric_tolerance")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def differences(self) -> list[Difference]:
        return [d for c in self.checks for d in c.differences]

    def _check_passed(self, name: str) -> bool:
        check = next((c for c in self.checks if c.name == name), None)
        return bool(check and check.passed and not check.skipped)

    @property
    def failed_checks(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed or c.skipped]

    def render(self) -> str:
        lines = [f"validation: {self.status.value}"]
        if self.error:
            lines.append(f"  error: {self.error}")
        for check in self.checks:
            mark = "skip" if check.skipped else ("ok" if check.passed else "FAIL")
            lines.append(f"  [{mark:>4}] {check.name}: {check.detail}")
            lines.extend(f"         - {d.render()}" for d in check.differences[:5])
            if len(check.differences) > 5:
                lines.append(f"         ... and {len(check.differences) - 5} more")
        return "\n".join(lines)


class ValidationDiagnosis(StrictModel):
    """The Validation agent's reading of a failure.

    Produced only when the differ already said FAIL. Note what is absent: any
    field that could flip the verdict. The agent's job is to explain and to
    point at the responsible plan step, not to grade.
    """

    summary: str = Field(description="What actually differs, in one or two sentences.")
    root_cause_category: RiskCategory
    implicated_step_ids: list[str] = Field(
        default_factory=list, description="Plan step ids (s3, s6) most likely responsible."
    )
    evidence: list[str] = Field(
        default_factory=list,
        description="Specific differences cited from the report. Anything not "
        "traceable to the report is a guess and must not appear here.",
    )
    suggested_fix: str = Field(description="Concrete change to the generated Spark code.")
    confidence: float = Field(ge=0.0, le=1.0)


class GeneratedTests(StrictModel):
    """A pytest module produced by the Testing agent."""

    filename: str
    content: str
    test_names: list[str] = Field(default_factory=list)
    covers_checks: list[str] = Field(
        default_factory=list,
        description="Validation checks these tests exercise, so a gap between the "
        "plan's required checks and the suite is visible.",
    )
    notes: list[str] = Field(default_factory=list)


class TestRunResult(StrictModel):
    """Outcome of executing the generated pytest suite in the sandbox."""

    succeeded: bool
    passed: int = Field(default=0, ge=0)
    failed: int = Field(default=0, ge=0)
    errors: int = Field(default=0, ge=0)
    skipped: int = Field(default=0, ge=0)
    duration_seconds: float = Field(default=0.0, ge=0.0)
    output_tail: str = ""

    @property
    def total(self) -> int:
        return self.passed + self.failed + self.errors + self.skipped

    def render(self) -> str:
        return (
            f"tests: {self.passed} passed, {self.failed} failed, "
            f"{self.errors} errors, {self.skipped} skipped "
            f"({self.duration_seconds:.2f}s)"
        )


class ValidationOutcome(StrictModel):
    """Everything ValidationWorkflow returns to its parent."""

    report: ValidationReport
    legacy_execution: ExecutionResult | None = None
    spark_execution: ExecutionResult | None = None
    tests: GeneratedTests | None = None
    test_run: TestRunResult | None = None
    diagnosis: ValidationDiagnosis | None = None

    @property
    def passed(self) -> bool:
        return self.report.status is ValidationStatus.PASS
