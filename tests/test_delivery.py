"""The delivery policy and the claim audit, tested by trying to get past them.

Two interlocks are under test here and they guard different things. The policy
decides whether a migration has earned a pull request at all; the audit decides
whether the prose attached to it is allowed to say what it says. Both are pure
functions over a `MigrationRecord`, so both can be driven with records
constructed to be exactly as awkward as needed.

The records are built rather than stubbed. `ValidationReport.status` is a
computed field, so the only way to get a FAIL here is to fail a check, which is
the same path the differ takes.

No JVM, no network.
"""

from __future__ import annotations

import pytest

from etl_migrator.domain.artifacts import CodeGenResult, MigrationRecord
from etl_migrator.domain.code import GeneratedCode, StaticAnalysisReport
from etl_migrator.domain.delivery import (
    NEEDS_HUMAN_LABEL,
    ORIGIN_LABEL,
    DeliveryDisposition,
    PullRequestNarrative,
)
from etl_migrator.domain.delivery_policy import (
    audit_numeric_claims,
    decide_delivery,
    supported_numbers,
)
from etl_migrator.domain.enums import ValidationStatus
from etl_migrator.domain.optimization import (
    BenchmarkResult,
    OptimizationAttempt,
    OptimizationOutcome,
    OptimizationStrategy,
)
from etl_migrator.domain.repair import RepairOutcome
from etl_migrator.domain.validation import (
    CheckResult,
    DatasetStats,
    ValidationOutcome,
    ValidationReport,
)
from etl_migrator.tools.pr_body import (
    EVIDENCE_END,
    EVIDENCE_START,
    render_evidence,
    render_pr_body,
)


def code_result() -> CodeGenResult:
    return CodeGenResult(
        code=GeneratedCode(filename="pipeline_spark.py", content="def run(): ...\n"),
        static_analysis=StaticAnalysisReport(passed=True),
        gate_iterations=1,
    )


def validation(status: ValidationStatus, *, rows: int = 5) -> ValidationOutcome:
    if status is ValidationStatus.PASS:
        checks = [
            CheckResult(name="schema", passed=True),
            CheckResult(name="row_count", passed=True),
        ]
    elif status is ValidationStatus.FAIL:
        checks = [
            CheckResult(name="schema", passed=True),
            CheckResult(name="row_count", passed=False, detail="4 != 5"),
        ]
    else:
        checks = [CheckResult(name="schema", passed=False, skipped=True)]
    report = ValidationReport(
        migration_id="mig-1",
        checks=checks,
        reference=DatasetStats(path="ref", row_count=rows),
        candidate=DatasetStats(path="cand", row_count=rows),
    )
    assert report.status is status
    return ValidationOutcome(report=report)


def record(
    *,
    status: ValidationStatus | None = ValidationStatus.PASS,
    failed: bool = False,
    with_code: bool = True,
    repair: RepairOutcome | None = None,
    optimization: OptimizationOutcome | None = None,
) -> MigrationRecord:
    rec = MigrationRecord(migration_id="mig-1", source_path="legacy.py")
    if with_code:
        rec.codegen = code_result()
    if status is not None:
        rec.validation = validation(status)
    rec.repair = repair
    rec.optimization = optimization
    rec.failed = failed
    if failed:
        rec.failure_reason = "the sandbox timed out"
    return rec


# --------------------------------------------------------------------------
# Who gets a pull request
# --------------------------------------------------------------------------


class TestDeliveryPolicy:
    def test_a_validated_migration_is_ready(self) -> None:
        decision = decide_delivery(record())
        assert decision.disposition is DeliveryDisposition.READY
        assert not decision.draft
        assert decision.should_open
        assert ORIGIN_LABEL in decision.labels
        assert NEEDS_HUMAN_LABEL not in decision.labels

    def test_an_unvalidated_migration_is_refused(self) -> None:
        """The important refusal.

        Code that was never executed against the legacy pipeline is a
        hypothesis. Opening a PR for it asks a reviewer to approve something
        nobody has checked, and the reviewer has no way to tell that from a PR
        whose validation passed.
        """
        decision = decide_delivery(record(status=None))
        assert decision.disposition is DeliveryDisposition.REFUSED
        assert not decision.should_open
        assert "never validated" in decision.reason

    def test_a_failed_validation_becomes_a_labelled_draft(self) -> None:
        """Not silence, and not a merge request.

        Hiding the failure would waste the work; opening a normal PR would
        invite an approval. A draft carrying `needs-human` is the honest third
        option.
        """
        decision = decide_delivery(record(status=ValidationStatus.FAIL))
        assert decision.disposition is DeliveryDisposition.DRAFT
        assert decision.draft
        assert decision.should_open
        assert NEEDS_HUMAN_LABEL in decision.labels
        assert "validation-fail" in decision.labels

    def test_an_errored_validation_is_also_a_draft(self) -> None:
        decision = decide_delivery(record(status=ValidationStatus.ERROR))
        assert decision.disposition is DeliveryDisposition.DRAFT
        assert "validation-error" in decision.labels

    def test_a_successful_repair_restores_readiness(self) -> None:
        """What matters is the final state, not that it took two goes."""
        rec = record(status=ValidationStatus.PASS, repair=RepairOutcome(succeeded=True))
        assert decide_delivery(rec).disposition is DeliveryDisposition.READY

    def test_an_exhausted_repair_stays_a_draft(self) -> None:
        rec = record(
            status=ValidationStatus.FAIL,
            repair=RepairOutcome(succeeded=False, exhausted=True),
        )
        decision = decide_delivery(rec)
        assert decision.disposition is DeliveryDisposition.DRAFT
        assert NEEDS_HUMAN_LABEL in decision.labels

    def test_a_failed_migration_is_a_draft_not_a_refusal(self) -> None:
        decision = decide_delivery(record(failed=True))
        assert decision.disposition is DeliveryDisposition.DRAFT
        assert "migration-failed" in decision.labels

    def test_nothing_generated_means_nothing_to_deliver(self) -> None:
        decision = decide_delivery(record(with_code=False))
        assert decision.disposition is DeliveryDisposition.REFUSED
        assert "nothing to deliver" in decision.reason

    def test_a_kept_optimisation_is_labelled(self) -> None:
        rec = record(
            optimization=OptimizationOutcome(
                applied=True,
                baseline=BenchmarkResult(label="baseline", durations=[10.0] * 4),
                final=BenchmarkResult(label="candidate", durations=[8.0] * 4),
                accepted_strategy=OptimizationStrategy(
                    approach="reduce_shuffle_partitions",
                    description="fewer partitions",
                    rationale="measured 200 for a tiny input",
                ),
            )
        )
        assert "optimised" in decide_delivery(rec).labels

    def test_the_reason_given_is_the_actionable_one(self) -> None:
        """A record that is wrong in two ways reports the failure, not the risk.

        Ordering matters for the human reading the label set: "your validation
        failed" is what to act on, and burying it under "this is high risk"
        makes it look like a routine review request.
        """
        rec = record(status=ValidationStatus.FAIL, failed=True)
        decision = decide_delivery(rec)
        assert "migration-failed" in decision.labels
        assert decision.draft


# --------------------------------------------------------------------------
# What the prose is allowed to say
# --------------------------------------------------------------------------


def narrative(summary: str, **kwargs: list[str]) -> PullRequestNarrative:
    return PullRequestNarrative(
        title="Migrate the customer revenue pipeline to PySpark",
        summary=summary.ljust(40),
        **kwargs,  # type: ignore[arg-type]
    )


def optimised_record(speedup_from: float = 10.0, speedup_to: float = 8.0) -> MigrationRecord:
    return record(
        optimization=OptimizationOutcome(
            applied=True,
            baseline=BenchmarkResult(label="baseline", durations=[speedup_from] * 4),
            final=BenchmarkResult(label="candidate", durations=[speedup_to] * 4),
            accepted_strategy=OptimizationStrategy(
                approach="reduce_shuffle_partitions",
                description="fewer partitions",
                rationale="measured",
            ),
        )
    )


class TestClaimAudit:
    def test_prose_without_numbers_passes_trivially(self) -> None:
        audit = audit_numeric_claims(
            narrative("This migrates a pandas revenue pipeline to PySpark."), record()
        )
        assert audit.passed
        assert audit.checked == 0

    def test_a_fabricated_speedup_is_caught(self) -> None:
        """The headline failure.

        The record measured 1.25x. The prose says 3x. A reviewer who reads the
        summary and skims the evidence block would come away with the wrong
        number, which is exactly the person this protects.
        """
        audit = audit_numeric_claims(
            narrative("The optimised pipeline is 3x faster than the original."),
            optimised_record(),
        )
        assert not audit.passed
        violation = audit.violations[0]
        assert violation.kind == "speedup"
        assert violation.claimed == "3"
        assert "1.25" in violation.supported
        assert "3x faster" in violation.excerpt

    def test_the_measured_speedup_is_permitted(self) -> None:
        audit = audit_numeric_claims(
            narrative("Benchmarking showed it runs 1.25x faster."), optimised_record()
        )
        assert audit.passed
        assert audit.checked == 1

    def test_the_typographic_multiplication_sign_is_not_a_loophole(self) -> None:
        """An audit that only reads ASCII is one keystroke from useless."""
        audit = audit_numeric_claims(
            # Built from an escape so the source stays ASCII and the two
            # characters cannot be mistaken for one another in review.
            narrative("The optimised pipeline is 9\u00d7 faster."),
            optimised_record(),
        )
        assert not audit.passed
        assert audit.violations[0].claimed == "9"

    def test_an_inflated_check_count_is_caught(self) -> None:
        audit = audit_numeric_claims(
            narrative("All 47 checks passed against the reference output."), record()
        )
        assert not audit.passed
        assert audit.violations[0].kind == "checks"

    def test_the_real_check_count_is_permitted(self) -> None:
        audit = audit_numeric_claims(
            narrative("All 2 checks passed against the reference output."), record()
        )
        assert audit.passed

    def test_a_wrong_row_count_is_caught(self) -> None:
        audit = audit_numeric_claims(
            narrative("The pipeline produced 10000 rows, matching pandas."), record()
        )
        assert not audit.passed
        assert audit.violations[0].kind == "rows"

    def test_thousands_separators_are_not_a_loophole(self) -> None:
        """"1,000,000 rows" and "1000000 rows" are the same claim."""
        rec = record()
        rec.validation = validation(ValidationStatus.PASS, rows=1_000_000)
        assert audit_numeric_claims(narrative("It emitted 1,000,000 rows."), rec).passed
        assert audit_numeric_claims(narrative("It emitted 1000000 rows."), rec).passed
        assert not audit_numeric_claims(narrative("It emitted 2,000,000 rows."), rec).passed

    def test_every_authored_field_is_audited_not_just_the_summary(self) -> None:
        """A claim moved into a bullet list is still a claim."""
        for field in ("reviewer_focus", "risk_callouts"):
            audit = audit_numeric_claims(
                narrative(
                    "A migration of the revenue pipeline.",
                    **{field: ["Confirm the 5x speedup is real."]},
                ),
                optimised_record(),
            )
            assert not audit.passed, field

    def test_an_unchanged_pipeline_may_be_called_1x(self) -> None:
        """Not every number needs a measurement behind it to be honest."""
        assert audit_numeric_claims(
            narrative("Performance is unchanged at 1x."), record()
        ).passed

    def test_ordinary_numbers_in_prose_are_not_flagged(self) -> None:
        """The audit has to stay narrow enough that nobody switches it off.

        None of these are claims about what the migration measured, and an
        audit that argued with them would be noise.
        """
        audit = audit_numeric_claims(
            narrative(
                "Targets Python 3.11 and Spark 4.0. Reads 2 CSV files and writes 1 "
                "output directory. See lines 40-55 of the module."
            ),
            record(),
        )
        assert audit.passed

    def test_violations_explain_themselves(self) -> None:
        audit = audit_numeric_claims(
            narrative("It is 3x faster now."), optimised_record()
        )
        rendered = audit.render()
        assert "FAIL" in rendered
        assert "3" in rendered
        assert "1.25" in rendered

    def test_supported_numbers_are_read_from_the_record(self) -> None:
        """The permitted set is derived, never configured.

        This is what stops the audit drifting into a list of numbers somebody
        once decided were fine.
        """
        permitted = supported_numbers(optimised_record())
        assert "1.25" in permitted["speedup"]
        assert "5" in permitted["rows"]
        assert permitted["tests"] == set()

    @pytest.mark.parametrize("claimed", ["2.50", "2.5", "0.5"])
    def test_near_misses_are_still_misses(self, claimed: str) -> None:
        audit = audit_numeric_claims(
            narrative(f"Roughly {claimed}x faster after tuning."), optimised_record()
        )
        assert not audit.passed


# --------------------------------------------------------------------------
# What ends up in the body
# --------------------------------------------------------------------------


class TestPullRequestBody:
    def test_the_evidence_block_is_rendered_from_the_record(self) -> None:
        rendered = render_evidence(record(), decide_delivery(record()))
        assert "✅ **PASS**" in rendered
        assert "`schema`" in rendered
        assert "`row_count`" in rendered
        assert EVIDENCE_START in rendered and EVIDENCE_END in rendered

    def test_agent_prose_cannot_reach_the_evidence_block(self) -> None:
        """The structural guarantee the split exists for.

        Whatever the narrative says — even if it contradicts the record outright
        — the evidence half is byte-identical, because it is rendered from the
        record and the narrative is not one of its inputs.
        """
        rec = record(status=ValidationStatus.FAIL)
        decision = decide_delivery(rec)
        honest = narrative("Validation failed; this needs a human.")
        lying = narrative("Everything passed perfectly, ship it.")

        assert render_evidence(rec, decision) == render_evidence(rec, decision)
        first = render_pr_body(rec, decision, honest)
        second = render_pr_body(rec, decision, lying)

        evidence_of = lambda body: body[body.index(EVIDENCE_START) : body.index(EVIDENCE_END)]  # noqa: E731
        assert evidence_of(first) == evidence_of(second)
        assert "❌ **FAIL**" in evidence_of(second)

    def test_a_draft_body_opens_with_a_do_not_merge_warning(self) -> None:
        rec = record(status=ValidationStatus.FAIL)
        body = render_pr_body(rec, decide_delivery(rec), narrative("It failed."))
        assert "[!WARNING]" in body
        assert "should not be merged" in body

    def test_a_ready_body_carries_no_warning(self) -> None:
        rec = record()
        body = render_pr_body(rec, decide_delivery(rec), narrative("It worked."))
        assert "should not be merged" not in body

    def test_the_narrative_appears_above_the_evidence(self) -> None:
        rec = record()
        body = render_pr_body(
            rec,
            decide_delivery(rec),
            narrative(
                "A pandas to PySpark migration of the revenue pipeline.",
                reviewer_focus=["Check the join key."],
                risk_callouts=["Null group keys are dropped."],
            ),
        )
        assert body.index("A pandas to PySpark migration") < body.index(EVIDENCE_START)
        assert "Check the join key." in body
        assert "Null group keys are dropped." in body

    def test_the_measured_speedup_is_quoted_from_the_record(self) -> None:
        rec = optimised_record()
        body = render_pr_body(rec, decide_delivery(rec), narrative("Faster now."))
        assert "**1.25x faster**" in body
        assert "reduce_shuffle_partitions" in body

    def test_a_refused_optimisation_still_reports_its_attempts(self) -> None:
        """"Nothing kept" with the reasons underneath is a result; on its own it
        is indistinguishable from a stage that never ran."""
        rec = record(
            optimization=OptimizationOutcome(
                applied=False,
                attempts=[
                    OptimizationAttempt(
                        attempt=1,
                        strategy=OptimizationStrategy(
                            approach="broadcast_small_side",
                            description="broadcast",
                            rationale="measured 159 KB",
                        ),
                        verdict="rejected: 1.02x is below the 1.10x threshold",
                    )
                ],
            )
        )
        body = render_pr_body(rec, decide_delivery(rec), narrative("No win found."))
        assert "No optimisation was kept" in body
        assert "below the 1.10x threshold" in body

    def test_the_body_states_where_its_numbers_came_from(self) -> None:
        """A reviewer should not have to work out which half to trust."""
        rec = record()
        body = render_pr_body(rec, decide_delivery(rec), narrative("A migration."))
        assert "None of it is authored by a language model" in body
        assert "audited against the migration record" in body

    def test_provenance_names_the_source_and_the_risk(self) -> None:
        rec = record()
        body = render_pr_body(rec, decide_delivery(rec), narrative("A migration."))
        assert rec.migration_id in body
        assert "legacy.py" in body
