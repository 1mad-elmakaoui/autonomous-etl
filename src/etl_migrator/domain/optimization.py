"""Benchmarking and optimisation: measured, or not accepted.

`evaluate_optimization` applies three rules, none of them left to a prompt:

1. Correctness first. Validation is re-run in full and a FAIL rejects the
   proposal whatever the timing said.
2. The gain must clear `min_speedup` (default 1.10), not merely exist.
3. The gain must survive the noise. The candidate's p75 must beat the
   baseline's median, and if either side is too noisy to measure through the
   comparison is inconclusive and nothing is accepted.

`OptimizationOutcome.applied` is a conjunction of those measured facts. No
field on any agent output can set it.

When a comparison comes back inconclusive the fix is more measurement, never a
laxer threshold. `DEFAULT_RUNS` and the `noise_ratio` estimator were both
changed during development; the thresholds were not.
"""

from __future__ import annotations

import statistics

from pydantic import Field, computed_field

from etl_migrator.domain.code import GeneratedCode
from etl_migrator.domain.plan import ExecutionStrategy
from etl_migrator.domain.spec import StrictModel
from etl_migrator.domain.validation import ValidationReport

#: Below this, a "speedup" is indistinguishable from scheduling jitter on a
#: local Spark run and is not worth a reviewer's attention.
DEFAULT_MIN_SPEEDUP = 1.10

#: Ceiling on `BenchmarkResult.noise_ratio` — a robust relative dispersion, see
#: that property — above which a measurement cannot support a 10% claim. Noisier
#: than this and the honest answer is "we cannot tell".
DEFAULT_MAX_NOISE_RATIO = 0.25

#: Timed runs per configuration, plus discarded warm-ups. Measured, not guessed
#: — see the note in `tools/benchmark.py`: four samples reported 36.8% noise on
#: the example workload and could conclude nothing, while eight reported 4.0%
#: and a robust 1.19x on the very same comparison. Four samples could not see
#: past their own error bar.
DEFAULT_RUNS = 8
DEFAULT_WARMUPS = 1


class SparkRunMetrics(StrictModel):
    """What one execution reported about itself.

    Deliberately limited to numbers Spark reliably exposes to PySpark after a
    run. Stage-level shuffle byte counts need a JVM listener, so they are not
    claimed here rather than being estimated and presented as measured.
    """

    jobs: int = Field(default=0, ge=0)
    stages: int = Field(default=0, ge=0)
    tasks: int = Field(default=0, ge=0)
    shuffle_partitions: int | None = None
    adaptive_enabled: bool | None = None
    default_parallelism: int | None = None

    def render(self) -> str:
        return (
            f"jobs={self.jobs} stages={self.stages} tasks={self.tasks} "
            f"shuffle_partitions={self.shuffle_partitions} aqe={self.adaptive_enabled}"
        )


class BenchmarkResult(StrictModel):
    """Timings from repeated execution of one configuration."""

    label: str
    durations: list[float] = Field(default_factory=list, description="Kept runs, seconds.")
    discarded_warmups: list[float] = Field(
        default_factory=list,
        description="Warm-up runs, excluded from the statistics. A first Spark run "
        "pays JVM start-up and JIT costs that say nothing about the pipeline.",
    )
    metrics: SparkRunMetrics | None = None
    failed: bool = False
    error: str | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def median(self) -> float:
        return statistics.median(self.durations) if self.durations else 0.0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def p75(self) -> float:
        """Robust upper estimate. With few samples, the max is too jumpy and the
        mean is dragged by a single outlier; the 75th percentile is neither."""
        if not self.durations:
            return 0.0
        ordered = sorted(self.durations)
        if len(ordered) == 1:
            return ordered[0]
        index = min(len(ordered) - 1, round(0.75 * (len(ordered) - 1)))
        return ordered[index]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def noise_ratio(self) -> float:
        """The measurement's own error bar, as a fraction of the median.

        Median absolute deviation rather than standard deviation, scaled by
        1.4826 so it stays comparable to a relative standard deviation.

        A real run came back `9.6 9.7 10.1 10.3 10.6 10.7 20.2` plus one more
        near ten: seven consistent measurements and one that hit CPU contention.
        Its standard deviation is 35% of the median, so the comparison was ruled
        unmeasurable even though seven runs agreed to within a second. The
        robustness check already reads the p75, which a lone outlier cannot
        reach, so an outlier-sensitive estimator here let the same data support
        two conclusions.

        Not a loosened threshold. Genuinely scattered timings still produce a
        large deviation and are still refused; `test_optimization.py` pins it.
        """
        if len(self.durations) < 2:
            return 0.0
        median = self.median
        if median <= 0:
            return 0.0
        deviations = [abs(d - median) for d in self.durations]
        return 1.4826 * statistics.median(deviations) / median

    @computed_field  # type: ignore[prop-decorator]
    @property
    def outliers(self) -> int:
        """Runs more than 50% off the median — almost always the machine, not
        the code. Counted and surfaced rather than silently absorbed, so a
        measurement taken on a busy host is visible to whoever reads it."""
        median = self.median
        if median <= 0:
            return 0
        return sum(1 for d in self.durations if abs(d - median) > 0.5 * median)

    @property
    def samples(self) -> int:
        return len(self.durations)

    def render(self) -> str:
        if self.failed:
            return f"{self.label}: FAILED — {self.error}"
        return (
            f"{self.label}: median={self.median:.3f}s p75={self.p75:.3f}s "
            f"noise={self.noise_ratio:.1%} over {self.samples} run(s)"
            + (f", {self.outliers} outlier(s)" if self.outliers else "")
            + (f" [{self.metrics.render()}]" if self.metrics else "")
        )


class BenchmarkComparison(StrictModel):
    """Whether the candidate is measurably faster. Computed, never asserted."""

    baseline: BenchmarkResult
    candidate: BenchmarkResult
    min_speedup: float = Field(default=DEFAULT_MIN_SPEEDUP, gt=1.0)
    max_noise_ratio: float = Field(default=DEFAULT_MAX_NOISE_RATIO, gt=0.0)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def speedup(self) -> float:
        """Baseline median over candidate median. >1 means the candidate is faster."""
        if self.candidate.median <= 0 or self.baseline.median <= 0:
            return 0.0
        return self.baseline.median / self.candidate.median

    @computed_field  # type: ignore[prop-decorator]
    @property
    def inconclusive(self) -> bool:
        """True when the measurement cannot support any conclusion.

        A failed run, a single sample, or noise wider than the effect we are
        trying to detect. Reporting a speedup from data like this is how
        benchmark theatre happens.
        """
        if self.baseline.failed or self.candidate.failed:
            return True
        if self.baseline.samples < 2 or self.candidate.samples < 2:
            return True
        return max(self.baseline.noise_ratio, self.candidate.noise_ratio) > self.max_noise_ratio

    @computed_field  # type: ignore[prop-decorator]
    @property
    def robust(self) -> bool:
        """The candidate's p75 beats the baseline's median.

        Requires the improvement to hold across most runs rather than resting on
        one fast execution.
        """
        if self.baseline.median <= 0 or self.candidate.p75 <= 0:
            return False
        return self.candidate.p75 < self.baseline.median

    @computed_field  # type: ignore[prop-decorator]
    @property
    def improved(self) -> bool:
        """Faster by enough, robustly, and measurably."""
        return (
            not self.inconclusive
            and self.speedup >= self.min_speedup
            and self.robust
        )

    def render(self) -> str:
        lines = [self.baseline.render(), self.candidate.render()]
        if self.inconclusive:
            lines.append(
                f"verdict: INCONCLUSIVE — noise "
                f"({max(self.baseline.noise_ratio, self.candidate.noise_ratio):.1%}) "
                f"exceeds the {self.max_noise_ratio:.0%} ceiling, or too few samples. "
                "No optimisation can be accepted on this measurement."
            )
        else:
            lines.append(
                f"verdict: {self.speedup:.2f}x "
                f"(threshold {self.min_speedup:.2f}x, robust={self.robust}) -> "
                f"{'ACCEPT' if self.improved else 'REJECT'}"
            )
        return "\n".join(lines)


class OptimizationStrategy(StrictModel):
    """What the optimiser intends to change, named so attempts can be compared."""

    approach: str = Field(
        pattern=r"^[a-z][a-z0-9_]{2,48}$",
        description="Slug for the technique, e.g. 'reduce_shuffle_partitions'.",
    )
    description: str
    rationale: str = Field(
        description="Grounded in the measured metrics or the plan analysis, not in "
        "general advice about Spark."
    )
    expected_speedup: float = Field(
        default=1.0, ge=1.0, description="The agent's claim. Recorded, then checked."
    )


class OptimizationProposal(StrictModel):
    """One optimisation to try.

    Either or both halves may change: configuration alone is often the larger
    win and carries less risk than rewriting the pipeline.
    """

    strategy: OptimizationStrategy
    execution_strategy: ExecutionStrategy | None = Field(
        default=None, description="Replacement Spark configuration, if it changes."
    )
    code: GeneratedCode | None = Field(
        default=None, description="Replacement module, if the code changes."
    )

    @property
    def changes_code(self) -> bool:
        return self.code is not None

    @property
    def changes_config(self) -> bool:
        return self.execution_strategy is not None


class OptimizationAttempt(StrictModel):
    """The durable record of one optimisation attempt and its measured fate."""

    attempt: int = Field(ge=1)
    strategy: OptimizationStrategy
    admitted: bool = True
    rejection_reason: str | None = None
    validation_status: str | None = None
    comparison: BenchmarkComparison | None = None
    accepted: bool = False
    verdict: str = ""

    def render(self) -> str:
        head = f"attempt {self.attempt} [{self.strategy.approach}]"
        if not self.admitted:
            return f"{head}: REJECTED — {self.rejection_reason}"
        return f"{head}: {self.verdict}"


class OptimizationOutcome(StrictModel):
    """Everything `OptimizationWorkflow` returns.

    `applied` is only ever set by `evaluate_optimization`, from measurements.
    """

    attempts: list[OptimizationAttempt] = Field(default_factory=list)
    applied: bool = False
    accepted_strategy: OptimizationStrategy | None = None
    optimized_code: GeneratedCode | None = None
    optimized_execution_strategy: ExecutionStrategy | None = None
    baseline: BenchmarkResult | None = None
    final: BenchmarkResult | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def speedup(self) -> float:
        """The kept improvement, or 1.0 when nothing was accepted."""
        if not self.applied or self.baseline is None or self.final is None:
            return 1.0
        if self.final.median <= 0:
            return 1.0
        return self.baseline.median / self.final.median

    def render(self) -> str:
        lines = [
            f"optimisation: {'applied' if self.applied else 'none kept'} "
            f"after {len(self.attempts)} attempt(s)"
        ]
        lines += [f"  {a.render()}" for a in self.attempts]
        if self.applied:
            lines.append(f"  kept: {self.speedup:.2f}x faster")
        return "\n".join(lines)


def evaluate_optimization(
    *,
    validation: ValidationReport | None,
    comparison: BenchmarkComparison,
) -> tuple[bool, str]:
    """Decide whether to keep an optimisation. Returns `(accept, verdict)`.

    The order of the checks is the argument: correctness is not traded against
    speed at any exchange rate, so it is settled before the stopwatch is
    consulted at all.
    """
    if validation is None:
        return False, "rejected: correctness was not re-verified after the change"
    if validation.status.value != "PASS":
        return False, (
            f"rejected: validation {validation.status.value} after the change — "
            "an optimisation that alters the output is a regression, not a speedup"
        )
    if comparison.inconclusive:
        return False, (
            "rejected: the measurement is inconclusive "
            f"(noise {max(comparison.baseline.noise_ratio, comparison.candidate.noise_ratio):.1%} "
            f"> {comparison.max_noise_ratio:.0%} ceiling, or too few samples)"
        )
    if not comparison.robust:
        return False, (
            f"rejected: {comparison.speedup:.2f}x is not robust — the candidate's p75 "
            f"({comparison.candidate.p75:.3f}s) does not beat the baseline's median "
            f"({comparison.baseline.median:.3f}s), so the gain rests on one lucky run"
        )
    if comparison.speedup < comparison.min_speedup:
        return False, (
            f"rejected: {comparison.speedup:.2f}x is below the "
            f"{comparison.min_speedup:.2f}x threshold"
        )
    return True, f"accepted: {comparison.speedup:.2f}x faster, validation still PASS"
