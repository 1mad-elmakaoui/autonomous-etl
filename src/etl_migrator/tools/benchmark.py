"""Repeated, warm-up-corrected timing of a Spark pipeline.

A single execution is not a measurement. The first Spark run in a fresh JVM pays
class loading, JIT warm-up and shuffle-service start-up, and the runs after it
still vary by more than the effect sizes we are trying to detect.

So run N times, discard the first `warmups`, and hand the rest to
`BenchmarkResult`, which reports a median, a p75 and its own noise ratio. Every
number the optimiser sees carries an error bar.

N is eight because four did not work. On the example workload, four samples put
the candidate's noise ratio at 36.8% and every comparison came back
INCONCLUSIVE. The same comparison at eight reported 0.7% and 4.0% noise and a
robust 1.19x. The speedup was always there; four samples could not see past
their own error bar. Raising `DEFAULT_MAX_NOISE_RATIO` instead would have
produced the same headline on evidence that did not support it.
"""

from __future__ import annotations

from pathlib import Path

from etl_migrator.domain.optimization import (
    DEFAULT_RUNS,
    DEFAULT_WARMUPS,
    BenchmarkResult,
    SparkRunMetrics,
)
from etl_migrator.domain.plan import ExecutionStrategy
from etl_migrator.observability import get_logger
from etl_migrator.sandbox.execute import run_spark_pipeline

log = get_logger(__name__)

__all__ = ["DEFAULT_RUNS", "DEFAULT_WARMUPS", "benchmark_pipeline"]


def _metrics_from(raw: dict[str, str]) -> SparkRunMetrics:
    def as_int(key: str) -> int:
        try:
            return int(raw.get(key, "0"))
        except ValueError:
            return 0

    def as_optional_int(key: str) -> int | None:
        try:
            return int(raw[key])
        except (KeyError, ValueError):
            return None

    adaptive = raw.get("adaptive_enabled")
    return SparkRunMetrics(
        jobs=as_int("jobs"),
        stages=as_int("stages"),
        tasks=as_int("tasks"),
        shuffle_partitions=as_optional_int("shuffle_partitions"),
        adaptive_enabled=None if adaptive is None else adaptive.lower() == "true",
        default_parallelism=as_optional_int("default_parallelism"),
    )


def benchmark_pipeline(
    *,
    label: str,
    module_path: Path,
    input_dir: Path,
    output_dir: Path,
    strategy: ExecutionStrategy | None = None,
    runs: int = DEFAULT_RUNS,
    warmups: int = DEFAULT_WARMUPS,
) -> BenchmarkResult:
    """Execute the pipeline `runs + warmups` times and summarise the timings.

    A failure on any run aborts the benchmark and is reported: half a
    measurement is not a measurement, and `BenchmarkComparison` treats a failed
    side as inconclusive rather than drawing a conclusion from the other one.
    """
    warmup_durations: list[float] = []
    durations: list[float] = []
    metrics: SparkRunMetrics | None = None

    total = runs + warmups
    for index in range(total):
        is_warmup = index < warmups
        result = run_spark_pipeline(
            module_path=module_path,
            input_dir=input_dir,
            output_dir=output_dir,
            strategy=strategy,
        )
        if not result.succeeded:
            log.warning(
                "benchmark.run_failed", label=label, run=index + 1, error=result.error
            )
            return BenchmarkResult(
                label=label,
                durations=durations,
                discarded_warmups=warmup_durations,
                failed=True,
                error=f"run {index + 1}/{total} failed: {result.error}",
            )

        if is_warmup:
            warmup_durations.append(round(result.duration_seconds, 4))
        else:
            durations.append(round(result.duration_seconds, 4))
            metrics = _metrics_from(result.metrics)

    outcome = BenchmarkResult(
        label=label,
        durations=durations,
        discarded_warmups=warmup_durations,
        metrics=metrics,
    )
    log.info(
        "benchmark.done",
        label=label,
        median=round(outcome.median, 3),
        noise=round(outcome.noise_ratio, 4),
        samples=outcome.samples,
    )
    return outcome
