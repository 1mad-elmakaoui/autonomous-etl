"""Executing the two pipelines: the reference and the candidate.

Both go through the same sandbox with the same contract, differing only in the
engine profile. That symmetry matters: if the legacy pipeline ran in-process and
only the generated one were sandboxed, a difference in the *harness* could show
up as a difference in the *data*, and the differ would blame the migration.
"""

from __future__ import annotations

from pathlib import Path

from etl_migrator.domain.plan import ExecutionStrategy
from etl_migrator.domain.validation import ExecutionResult
from etl_migrator.sandbox.runner import SandboxLimits, SandboxRunner, SubprocessSandbox

#: Baseline Spark configuration. The optimizer varies these and
#: re-benchmarks; correctness must hold under all of them.
BASE_SPARK_CONF: dict[str, str] = {
    "spark.sql.adaptive.enabled": "true",
    "spark.sql.shuffle.partitions": "8",
    "spark.ui.enabled": "false",
    "spark.driver.memory": "2g",
    "spark.sql.session.timeZone": "UTC",
}


def spark_conf_from(strategy: ExecutionStrategy | None) -> dict[str, str]:
    """Turn the planner's declared strategy into real Spark settings.

    The plan is not decorative — if the planner said AQE and eight shuffle
    partitions, that is what the pipeline is executed with, and the benchmark
    later measures that exact configuration.
    """
    conf = dict(BASE_SPARK_CONF)
    if strategy is None:
        return conf
    conf["spark.sql.shuffle.partitions"] = str(strategy.shuffle_partitions)
    conf["spark.sql.adaptive.enabled"] = str(strategy.adaptive_query_execution).lower()
    return conf


def run_legacy_pipeline(
    *,
    module_path: Path,
    input_dir: Path,
    output_dir: Path,
    runner: SandboxRunner | None = None,
    limits: SandboxLimits | None = None,
) -> ExecutionResult:
    """Execute the legacy pipeline to produce the reference output."""
    sandbox = runner or SubprocessSandbox()
    result = sandbox.run(
        kind="legacy",
        module_path=module_path,
        input_dir=input_dir,
        output_dir=output_dir,
        limits=limits or SandboxLimits.for_pandas(),
    )
    return ExecutionResult(
        engine="pandas",
        succeeded=result.ok,
        duration_seconds=result.duration_seconds,
        output_path=str(output_dir) if result.ok else None,
        stdout_tail=result.stdout,
        stderr_tail=result.stderr,
        exit_code=result.exit_code,
        timed_out=result.timed_out,
        error=_error_of(result),
        metrics=result.metrics,
    )


def run_spark_pipeline(
    *,
    module_path: Path,
    input_dir: Path,
    output_dir: Path,
    strategy: ExecutionStrategy | None = None,
    runner: SandboxRunner | None = None,
    limits: SandboxLimits | None = None,
    master: str = "local[2]",
) -> ExecutionResult:
    """Execute the generated PySpark pipeline to produce the candidate output."""
    sandbox = runner or SubprocessSandbox()
    result = sandbox.run(
        kind="spark",
        module_path=module_path,
        input_dir=input_dir,
        output_dir=output_dir,
        limits=limits or SandboxLimits.for_spark(),
        extra={
            "spark_conf": spark_conf_from(strategy),
            "master": master,
            "app_name": f"etlm-{module_path.stem}",
        },
    )
    return ExecutionResult(
        engine="spark",
        succeeded=result.ok,
        duration_seconds=result.duration_seconds,
        output_path=str(output_dir) if result.ok else None,
        stdout_tail=result.stdout,
        stderr_tail=result.stderr,
        exit_code=result.exit_code,
        timed_out=result.timed_out,
        error=_error_of(result),
        metrics=result.metrics,
    )


def _error_of(result: object) -> str | None:
    """Prefer the child's own error, falling back to its traceback tail."""
    error = getattr(result, "error", None)
    if error:
        return str(error)
    tb = getattr(result, "traceback", None)
    return str(tb)[-1500:] if tb else None
