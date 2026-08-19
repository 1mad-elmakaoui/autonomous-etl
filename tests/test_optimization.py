"""The acceptance rule, tested with controlled timings.

`evaluate_optimization` is the one function in the system an agent cannot argue
with, so it gets tested the way a safety interlock gets tested: by trying to
sneak past it. Every test here hands it a proposal that *looks* like a win —
plausible numbers, a confident `expected_speedup` — and asserts on the reason it
is turned down.

Synthetic durations are used deliberately. The e2e counterpart in
`test_optimization_e2e.py` proves the rule holds on real Spark timings; this
file proves it holds on timings chosen to be adversarial, which no real workload
can be relied upon to produce on demand.

No JVM needed: nothing here executes Spark.
"""

from __future__ import annotations

import pytest

from etl_migrator.domain.enums import DataFormat, ValidationStatus
from etl_migrator.domain.optimization import (
    DEFAULT_MAX_NOISE_RATIO,
    DEFAULT_MIN_SPEEDUP,
    BenchmarkComparison,
    BenchmarkResult,
    OptimizationAttempt,
    OptimizationOutcome,
    OptimizationStrategy,
    SparkRunMetrics,
    evaluate_optimization,
)
from etl_migrator.domain.validation import CheckResult, ValidationReport
from etl_migrator.tools.data_profiler import DatasetProfile
from etl_migrator.tools.plan_analyzer import analyze_plan


def report(status: ValidationStatus) -> ValidationReport:
    """A validation report with the requested status, built from real checks.

    Constructed rather than stubbed: `status` is a computed field, so the only
    way to get a FAIL is to fail a check, exactly as the differ would.
    """
    if status is ValidationStatus.PASS:
        checks = [CheckResult(name="row_count", passed=True)]
    elif status is ValidationStatus.FAIL:
        checks = [CheckResult(name="row_count", passed=False, detail="4 != 5")]
    else:
        checks = [CheckResult(name="row_count", passed=False, skipped=True)]
    built = ValidationReport(migration_id="m-1", checks=checks)
    assert built.status is status, "helper does not produce the status it claims"
    return built


def comparison(
    baseline: list[float], candidate: list[float], **kwargs: float
) -> BenchmarkComparison:
    return BenchmarkComparison(
        baseline=BenchmarkResult(label="baseline", durations=baseline),
        candidate=BenchmarkResult(label="candidate", durations=candidate),
        **kwargs,  # type: ignore[arg-type]
    )


# --------------------------------------------------------------------------
# The statistics a verdict rests on
# --------------------------------------------------------------------------


class TestBenchmarkResult:
    def test_reports_median_p75_and_samples(self) -> None:
        result = BenchmarkResult(label="b", durations=[10.0, 10.0, 12.0, 20.0])
        assert result.median == 11.0
        assert result.p75 == 12.0
        assert result.samples == 4
        # The p75 is the point: the 20s outlier moves it, but not to 20.
        assert result.p75 < max(result.durations)

    def test_a_stable_measurement_reports_low_noise(self) -> None:
        result = BenchmarkResult(label="b", durations=[10.0, 10.1, 9.9, 10.0])
        assert result.noise_ratio < 0.02

    def test_one_interrupted_run_does_not_void_seven_agreeing_ones(self) -> None:
        """The real sample that motivated the median-absolute-deviation estimator.

        Seven runs agree to within a second and one took twice as long because
        something else on the machine wanted the CPU. Under a standard
        deviation that sample reads as 35% noise and nothing can be concluded
        from it — which throws away seven good measurements on account of one
        bad one.
        """
        stalled = BenchmarkResult(
            label="b",
            durations=[9.556, 10.736, 20.208, 10.331, 9.641, 9.667, 10.122, 10.64],
        )
        assert stalled.noise_ratio < DEFAULT_MAX_NOISE_RATIO
        assert stalled.outliers == 1
        # The outlier is reported, never quietly dropped.
        assert "outlier" in stalled.render()
        assert 20.208 in stalled.durations

    def test_genuinely_scattered_timings_are_still_noisy(self) -> None:
        """The guard on the guard above.

        A robust estimator must not become a permissive one. Here the runs
        disagree with *each other*, rather than one run disagreeing with the
        rest, and that is the case the ceiling exists to catch.
        """
        scattered = BenchmarkResult(label="b", durations=[2.0, 5.0, 5.0, 12.0])
        assert scattered.noise_ratio > DEFAULT_MAX_NOISE_RATIO

    def test_resistance_extends_to_a_minority_and_no_further(self) -> None:
        """Resistant, not blind.

        One slow run in four is treated as the machine misbehaving. Three slow
        runs in four is the pipeline being slow, and the statistics say so.
        """
        occasional = BenchmarkResult(label="b", durations=[10.0, 10.0, 10.0, 20.0])
        assert occasional.median == 10.0
        assert occasional.noise_ratio == 0.0

        habitual = BenchmarkResult(label="b", durations=[10.0, 20.0, 20.0, 20.0])
        assert habitual.median == 20.0
        assert habitual.p75 == 20.0

    def test_a_stall_is_never_erased_from_the_record(self) -> None:
        """Whatever the statistics make of it, the raw timing survives so a
        human can look at the run that took twice as long."""
        stalled = BenchmarkResult(label="b", durations=[10.0, 10.0, 10.0, 20.0])
        assert stalled.durations == [10.0, 10.0, 10.0, 20.0]
        assert stalled.p75 == 10.0

    def test_warmups_are_excluded_from_every_statistic(self) -> None:
        """A warm-up run is recorded, never counted.

        The first Spark run pays JVM start-up; folding it into the median would
        make an unchanged pipeline look like it got faster on the second
        benchmark simply because that one was measured warm.
        """
        result = BenchmarkResult(
            label="b", durations=[10.0, 10.0], discarded_warmups=[45.0]
        )
        assert result.median == 10.0
        assert result.samples == 2
        assert 45.0 not in result.durations

    def test_no_samples_yields_zeroes_rather_than_an_exception(self) -> None:
        empty = BenchmarkResult(label="b", failed=True, error="boom")
        assert empty.median == 0.0
        assert empty.p75 == 0.0
        assert empty.noise_ratio == 0.0

    def test_render_surfaces_the_error_bar_next_to_the_number(self) -> None:
        """Whatever the optimiser reads, it reads the uncertainty too."""
        rendered = BenchmarkResult(
            label="baseline",
            durations=[10.0, 11.0],
            metrics=SparkRunMetrics(jobs=3, stages=6, tasks=408, shuffle_partitions=200),
        ).render()
        assert "median=" in rendered
        assert "noise=" in rendered
        assert "tasks=408" in rendered


# --------------------------------------------------------------------------
# The comparison
# --------------------------------------------------------------------------


class TestBenchmarkComparison:
    def test_speedup_is_baseline_over_candidate(self) -> None:
        assert comparison([12.0] * 4, [6.0] * 4).speedup == pytest.approx(2.0)

    def test_a_failed_run_makes_the_comparison_inconclusive(self) -> None:
        """Half a measurement is not a measurement.

        The tempting bug is to compare against whatever the working side
        produced and report a number anyway.
        """
        cmp = BenchmarkComparison(
            baseline=BenchmarkResult(label="baseline", durations=[10.0] * 4),
            candidate=BenchmarkResult(label="candidate", failed=True, error="OOM"),
        )
        assert cmp.inconclusive
        assert not cmp.improved

    def test_a_single_sample_is_inconclusive(self) -> None:
        assert comparison([10.0], [5.0]).inconclusive

    def test_noise_above_the_ceiling_is_inconclusive(self) -> None:
        """A 2x median difference proves nothing under 40% run-to-run spread."""
        cmp = comparison([10.0, 10.0, 10.0, 10.0], [2.0, 5.0, 5.0, 12.0])
        assert cmp.candidate.noise_ratio > DEFAULT_MAX_NOISE_RATIO
        assert cmp.inconclusive
        assert not cmp.improved

    def test_robustness_requires_the_p75_to_beat_the_baseline_median(self) -> None:
        """One lucky run cannot carry the verdict.

        The candidate's median is genuinely lower here, but most of its runs are
        no faster than the baseline — the median is dragged down by a single
        fast execution.
        """
        cmp = comparison([10.0, 10.0, 10.0, 10.0], [7.9, 8.1, 10.0, 10.2])
        assert cmp.speedup >= DEFAULT_MIN_SPEEDUP
        # Not merely noisy — the measurement is clean enough to conclude from,
        # and robustness is the only thing standing in the way.
        assert not cmp.inconclusive
        assert not cmp.robust
        assert not cmp.improved

    def test_improved_requires_all_three_conditions(self) -> None:
        cmp = comparison([12.0, 12.1, 11.9, 12.0], [9.0, 9.1, 8.9, 9.0])
        assert not cmp.inconclusive
        assert cmp.robust
        assert cmp.speedup >= DEFAULT_MIN_SPEEDUP
        assert cmp.improved


# --------------------------------------------------------------------------
# The rule itself
# --------------------------------------------------------------------------


class TestEvaluateOptimization:
    def test_accepts_a_measured_robust_win_on_a_passing_validation(self) -> None:
        accepted, verdict = evaluate_optimization(
            validation=report(ValidationStatus.PASS),
            comparison=comparison([12.0, 12.1, 11.9, 12.0], [9.0, 9.1, 8.9, 9.0]),
        )
        assert accepted
        assert "accepted" in verdict
        assert "1.33x" in verdict

    def test_a_broken_output_is_rejected_however_fast_it_is(self) -> None:
        """The central rule. A 10x speedup that changes the answer is a bug.

        Note the numbers: this comparison is clean, robust and enormous. Only
        correctness rejects it.
        """
        cmp = comparison([100.0, 100.1, 99.9, 100.0], [10.0, 10.1, 9.9, 10.0])
        assert cmp.improved, "the timing side must be impeccable for this test to bite"

        accepted, verdict = evaluate_optimization(
            validation=report(ValidationStatus.FAIL), comparison=cmp
        )
        assert not accepted
        assert "regression" in verdict

    def test_correctness_is_checked_before_the_stopwatch(self) -> None:
        """When both are bad, the reported reason is the correctness one.

        The order matters for the human reading the verdict: "your change broke
        the output" is the actionable finding, and burying it under "the
        measurement was noisy" invites someone to re-run the benchmark and try
        again with the same broken code.
        """
        _, verdict = evaluate_optimization(
            validation=report(ValidationStatus.FAIL),
            comparison=comparison([10.0], [5.0]),  # also inconclusive
        )
        assert "regression" in verdict
        assert "inconclusive" not in verdict

    def test_an_unverified_change_is_rejected(self) -> None:
        """No validation report at all is not a pass.

        This is the "we did not measure it, so it is fine" failure, which is how
        a stage that silently fails to run becomes a stage that silently
        approves everything.
        """
        accepted, verdict = evaluate_optimization(
            validation=None,
            comparison=comparison([12.0, 12.1, 11.9, 12.0], [9.0, 9.1, 8.9, 9.0]),
        )
        assert not accepted
        assert "not re-verified" in verdict

    def test_an_error_status_is_not_a_pass(self) -> None:
        accepted, _ = evaluate_optimization(
            validation=report(ValidationStatus.ERROR),
            comparison=comparison([12.0, 12.1, 11.9, 12.0], [9.0, 9.1, 8.9, 9.0]),
        )
        assert not accepted

    def test_an_unmeasurable_result_is_refused_not_guessed(self) -> None:
        accepted, verdict = evaluate_optimization(
            validation=report(ValidationStatus.PASS),
            comparison=comparison([10.0, 10.0, 10.0, 10.0], [2.0, 5.0, 5.0, 12.0]),
        )
        assert not accepted
        assert "inconclusive" in verdict

    def test_a_gain_resting_on_one_lucky_run_is_refused(self) -> None:
        accepted, verdict = evaluate_optimization(
            validation=report(ValidationStatus.PASS),
            comparison=comparison([10.0, 10.0, 10.0, 10.0], [7.9, 8.1, 10.0, 10.2]),
        )
        assert not accepted
        assert "not robust" in verdict

    def test_a_real_but_small_gain_is_below_the_threshold(self) -> None:
        """3% is real and still not worth a reviewer's time."""
        accepted, verdict = evaluate_optimization(
            validation=report(ValidationStatus.PASS),
            comparison=comparison([10.0, 10.0, 10.0, 10.0], [9.7, 9.7, 9.7, 9.7]),
        )
        assert not accepted
        assert "below the" in verdict

    def test_the_threshold_is_configurable_and_is_actually_consulted(self) -> None:
        timings = ([10.0, 10.0, 10.0, 10.0], [9.7, 9.7, 9.7, 9.7])
        strict, _ = evaluate_optimization(
            validation=report(ValidationStatus.PASS),
            comparison=comparison(*timings),
        )
        lenient, _ = evaluate_optimization(
            validation=report(ValidationStatus.PASS),
            comparison=comparison(*timings, min_speedup=1.02),
        )
        assert not strict
        assert lenient

    def test_a_slower_candidate_is_rejected(self) -> None:
        accepted, _ = evaluate_optimization(
            validation=report(ValidationStatus.PASS),
            comparison=comparison([10.0, 10.0, 10.0, 10.0], [14.0, 14.0, 14.0, 14.0]),
        )
        assert not accepted


class TestTheAgentCannotSetTheVerdict:
    """`expected_speedup` is recorded, then ignored.

    This is the guard against the failure the whole phase exists to prevent: an
    agent asserting an improvement and the system believing it.
    """

    @pytest.mark.parametrize("claim", [1.0, 2.0, 50.0])
    def test_expected_speedup_does_not_enter_the_decision(self, claim: float) -> None:
        """A 3% measured change is rejected whether the agent promised 1x or 50x.

        Asserting on the *identical* verdict string, rather than merely on
        rejection, is what makes this bite: it would fail if the claim leaked
        into the reasoning at all, even in the explanation.
        """
        strategy = OptimizationStrategy(
            approach="reduce_shuffle_partitions",
            description="Lower the partition count.",
            rationale="Measured 200 partitions over a 230 KB input.",
            expected_speedup=claim,
        )
        assert strategy.expected_speedup == claim  # recorded, for later comparison

        accepted, verdict = evaluate_optimization(
            validation=report(ValidationStatus.PASS),
            comparison=comparison([10.0, 10.0, 10.0, 10.0], [9.7, 9.7, 9.7, 9.7]),
        )
        assert not accepted, f"a claim of {claim}x moved the verdict"
        assert verdict == "rejected: 1.03x is below the 1.10x threshold"

    def test_outcome_speedup_is_one_when_nothing_was_kept(self) -> None:
        """No accepted change means no speedup to report, whatever was measured."""
        outcome = OptimizationOutcome(
            baseline=BenchmarkResult(label="baseline", durations=[10.0] * 4),
            final=BenchmarkResult(label="candidate", durations=[1.0] * 4),
            applied=False,
        )
        assert outcome.speedup == 1.0

    def test_a_rejected_attempt_is_kept_in_the_record(self) -> None:
        """Rejections are the interesting half of the audit trail."""
        outcome = OptimizationOutcome(
            attempts=[
                OptimizationAttempt(
                    attempt=1,
                    strategy=OptimizationStrategy(
                        approach="broadcast_small_side",
                        description="Broadcast orders.",
                        rationale="Profiler measured 159 KB.",
                    ),
                    verdict="rejected: 1.02x is below the 1.10x threshold",
                )
            ]
        )
        rendered = outcome.render()
        assert "none kept" in rendered
        assert "broadcast_small_side" in rendered
        assert "below the 1.10x threshold" in rendered


# --------------------------------------------------------------------------
# What the optimiser is allowed to see
# --------------------------------------------------------------------------


OPTIMISED = '''\
from pyspark.sql import SparkSession, functions as F

def run(spark: SparkSession, input_dir: str, output_dir: str) -> None:
    left = spark.read.csv(f"{input_dir}/a.csv", header=True)
    right = spark.read.csv(f"{input_dir}/b.csv", header=True)
    joined = left.join(F.broadcast(right), on="id", how="left")
    joined.groupBy("country").agg(F.sum("revenue")).write.csv(f"{output_dir}/out")
'''


class TestPlanAnalyzer:
    def test_reports_nothing_on_already_optimised_code(self) -> None:
        """The honesty test.

        An analyser that always finds something is an analyser that tells you
        nothing, and it would push the optimiser into inventing work. This code
        already broadcasts the small side; there is no opportunity to report.
        """
        analysis = analyze_plan(OPTIMISED)
        assert analysis.opportunities == []
        assert analysis.broadcast_hints
        assert any("already present" in note for note in analysis.notes)

    def test_counts_wide_transformations(self) -> None:
        analysis = analyze_plan(OPTIMISED)
        assert analysis.wide_transform_count == 2  # join + groupBy

    def test_flags_coalesce_one_with_its_line(self) -> None:
        code = OPTIMISED.replace(".write.csv", ".coalesce(1).write.csv")
        assert code != OPTIMISED
        found = [o for o in analyze_plan(code).opportunities if o.code == "OPT001"]
        assert len(found) == 1
        assert found[0].line is not None
        assert found[0].suggested_approach == "remove_single_partition_coalesce"

    def test_flags_a_python_udf(self) -> None:
        code = OPTIMISED.replace(
            'joined.groupBy', 'F.udf(lambda x: x)\n    joined.groupBy'
        )
        found = [o for o in analyze_plan(code).opportunities if o.code == "OPT004"]
        assert len(found) == 1

    def test_broadcast_opportunities_come_from_measured_sizes(self) -> None:
        """OPT006 is grounded in the profiler, never in a guess about the data."""
        without_hint = OPTIMISED.replace("F.broadcast(right)", "right")
        small = DatasetProfile(
            path="b.csv", format=DataFormat.CSV, exists=True, size_bytes=159_340
        )
        huge = DatasetProfile(
            path="b.csv", format=DataFormat.CSV, exists=True, size_bytes=500 * 1024 * 1024
        )

        flagged = [
            o for o in analyze_plan(without_hint, [small]).opportunities
            if o.code == "OPT006"
        ]
        assert len(flagged) == 1
        assert "159340" in flagged[0].summary

        assert not [
            o for o in analyze_plan(without_hint, [huge]).opportunities
            if o.code == "OPT006"
        ]

    def test_does_not_repeat_a_broadcast_already_present(self) -> None:
        small = DatasetProfile(
            path="b.csv", format=DataFormat.CSV, exists=True, size_bytes=159_340
        )
        assert not [
            o for o in analyze_plan(OPTIMISED, [small]).opportunities if o.code == "OPT006"
        ]

    def test_says_so_when_there_is_no_shuffle_to_optimise(self) -> None:
        narrow = '''\
def run(spark, input_dir, output_dir):
    df = spark.read.csv(f"{input_dir}/a.csv", header=True)
    df.filter(df.age > 18).write.csv(f"{output_dir}/out")
'''
        analysis = analyze_plan(narrow)
        assert analysis.wide_transform_count == 0
        assert any("no wide transformations" in note for note in analysis.notes)

    def test_a_syntax_error_is_reported_not_raised(self) -> None:
        analysis = analyze_plan("def run(:\n")
        assert analysis.opportunities == []
        assert any("could not parse" in note for note in analysis.notes)
