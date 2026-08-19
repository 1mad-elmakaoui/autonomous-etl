"""`ValidationWorkflow` — the child that decides whether the migration is correct.

A separate workflow rather than four more stages inline, for reasons that are
operational rather than aesthetic:

* Its activities have a completely different failure profile. A Spark job that
  OOMs should retry twice with a long timeout; an LLM call should retry four
  times with backoff. One retry budget cannot serve both.
* It is the unit the repair loop re-runs. Repair means "change the code
  and validate again", so validation has to be a thing you can invoke as a
  whole, repeatedly, with each attempt visible separately in the Temporal UI.
* Execution here can run for tens of minutes. Isolating it keeps a slow
  benchmark from sharing a timeout budget with a fast planning call.

The ordering below is deliberate. Tests run *before* the differ, because a
generated suite that fails tells you which behaviour broke, while an output diff
only tells you that something did. Cheap, specific signal first.
"""

from __future__ import annotations

from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from etl_migrator.activities.migration import ValidationActivities
    from etl_migrator.domain.enums import ValidationStatus
    from etl_migrator.domain.messages import (
        DiagnosisInput,
        LegacyExecutionInput,
        OutputValidationInput,
        SparkExecutionInput,
        TestExecutionInput,
        TestGenerationInput,
        ValidationWorkflowInput,
    )
    from etl_migrator.domain.validation import (
        CheckResult,
        ValidationOutcome,
        ValidationReport,
    )

NON_RETRYABLE: list[str] = [
    "ConfigurationError",
    "UnsupportedSourceError",
    "NonRetryableMigrationError",
    "ScriptExhaustedError",
]

#: Sandboxed execution: expensive, occasionally killed by a resource limit, and
#: not worth many attempts — a pipeline that OOMs twice will OOM a third time.
EXECUTION_RETRY = RetryPolicy(
    initial_interval=timedelta(seconds=5),
    backoff_coefficient=2.0,
    maximum_attempts=2,
    non_retryable_error_types=NON_RETRYABLE,
)
LEGACY_TIMEOUT = timedelta(minutes=15)
SPARK_TIMEOUT = timedelta(minutes=30)

#: The differ is deterministic. A second failure means a bug, not bad luck.
DIFF_RETRY = RetryPolicy(maximum_attempts=2, non_retryable_error_types=NON_RETRYABLE)
DIFF_TIMEOUT = timedelta(minutes=10)

AGENT_RETRY = RetryPolicy(
    initial_interval=timedelta(seconds=2),
    backoff_coefficient=2.0,
    maximum_interval=timedelta(seconds=30),
    maximum_attempts=3,
    non_retryable_error_types=NON_RETRYABLE,
)
AGENT_TIMEOUT = timedelta(minutes=10)
TEST_RUN_TIMEOUT = timedelta(minutes=30)


def _execution_failure(migration_id: str, engine: str, detail: str) -> ValidationReport:
    """A report describing an execution that never produced an output.

    Deliberately built with `error` set rather than with failing checks: nothing
    was measured, so the status must be ERROR, not FAIL. Reporting "the outputs
    differ" when one pipeline never ran would be a lie about what we know.
    """
    return ValidationReport(
        migration_id=migration_id,
        error=f"{engine} pipeline did not produce an output: {detail}",
    )


def execution_queue(configured: str) -> str | None:
    """Where the untrusted-code activities should run.

    `None` tells Temporal "the workflow's own queue", which is what a single
    worker wants. A configured value routes them to the worker deployed with no
    internet egress — the split that makes the sandbox's network story real
    rather than aspirational (see `k8s/` and `temporal/worker.py`).
    """
    return configured or None


@workflow.defn(name="ValidationWorkflow")
class ValidationWorkflow:
    """Execute both pipelines, run the generated tests, and diff the outputs."""

    def __init__(self) -> None:
        self._outcome: ValidationOutcome | None = None

    @workflow.query
    def outcome(self) -> ValidationOutcome | None:
        """Live view for a status poller, including partial progress."""
        return self._outcome

    @workflow.run
    async def run(self, params: ValidationWorkflowInput) -> ValidationOutcome:
        workflow.logger.info("validation started for %s", params.migration_id)

        # --- reference execution -----------------------------------------
        legacy = await workflow.execute_activity_method(
            ValidationActivities.run_legacy_pipeline,
            LegacyExecutionInput(
                migration_id=params.migration_id,
                source_path=params.legacy_source_path,
                input_dir=params.input_dir,
            ),
            start_to_close_timeout=LEGACY_TIMEOUT,
            retry_policy=EXECUTION_RETRY,
            task_queue=execution_queue(params.execution_task_queue),
            summary="run legacy pipeline",
        )
        if not legacy.succeeded:
            # The reference is the yardstick. Without it there is nothing to
            # compare against, and the migration cannot be assessed at all.
            return self._finish(
                ValidationOutcome(
                    report=_execution_failure(
                        params.migration_id, "legacy", legacy.error or "unknown error"
                    ),
                    legacy_execution=legacy,
                )
            )

        # --- candidate execution ------------------------------------------
        spark = await workflow.execute_activity_method(
            ValidationActivities.run_spark_pipeline,
            SparkExecutionInput(
                migration_id=params.migration_id,
                input_dir=params.input_dir,
                code=params.code,
                execution_strategy=params.plan.execution_strategy,
            ),
            start_to_close_timeout=SPARK_TIMEOUT,
            retry_policy=EXECUTION_RETRY,
            task_queue=execution_queue(params.execution_task_queue),
            summary="run generated PySpark pipeline",
        )
        if not spark.succeeded:
            return self._finish(
                ValidationOutcome(
                    report=_execution_failure(
                        params.migration_id, "spark", spark.error or "unknown error"
                    ),
                    legacy_execution=legacy,
                    spark_execution=spark,
                )
            )

        # --- generated tests ----------------------------------------------
        tests = None
        test_run = None
        if params.run_generated_tests:
            generated = await workflow.execute_activity_method(
                ValidationActivities.generate_tests,
                TestGenerationInput(
                    migration_id=params.migration_id,
                    scenario=params.scenario,
                    spec=params.spec,
                    plan=params.plan,
                    code=params.code,
                ),
                start_to_close_timeout=AGENT_TIMEOUT,
                retry_policy=AGENT_RETRY,
                summary="generate pytest suite",
            )
            tests = generated.tests
            # A suite that fails its own gate is not executed. Running unsafe
            # generated code because it happens to be labelled "test" would
            # defeat the point of having a gate.
            if generated.static_analysis.passed:
                test_run = await workflow.execute_activity_method(
                    ValidationActivities.run_tests,
                    TestExecutionInput(
                        migration_id=params.migration_id,
                        tests=generated.tests,
                        code=params.code,
                        input_dir=params.input_dir,
                        execution_strategy=params.plan.execution_strategy,
                    ),
                    start_to_close_timeout=TEST_RUN_TIMEOUT,
                    retry_policy=EXECUTION_RETRY,
                    task_queue=execution_queue(params.execution_task_queue),
                    summary="run generated tests",
                )
            else:
                workflow.logger.warning(
                    "generated test suite failed the static gate; not executing it"
                )

        # --- the verdict ---------------------------------------------------
        report = await workflow.execute_activity_method(
            ValidationActivities.validate_outputs,
            OutputValidationInput(
                migration_id=params.migration_id,
                reference_path=legacy.output_path or "",
                candidate_path=spark.output_path or "",
                plan=params.plan,
            ),
            start_to_close_timeout=DIFF_TIMEOUT,
            retry_policy=DIFF_RETRY,
            task_queue=execution_queue(params.execution_task_queue),
            summary="compare outputs",
        )

        # A failing generated test is a real failure even when the aggregate
        # outputs happen to agree — it means a behaviour the plan promised is
        # not implemented, and the sample data simply did not exercise it.
        if test_run is not None and not test_run.succeeded:
            report.checks.append(
                CheckResult(
                    name="generated_tests",
                    passed=False,
                    detail=test_run.render(),
                )
            )
        elif test_run is not None:
            report.checks.append(
                CheckResult(
                    name="generated_tests", passed=True, detail=test_run.render()
                )
            )

        outcome = ValidationOutcome(
            report=report,
            legacy_execution=legacy,
            spark_execution=spark,
            tests=tests,
            test_run=test_run,
        )

        # --- diagnosis, only on failure ------------------------------------
        if report.status is not ValidationStatus.PASS and report.error is None:
            diagnosis = await workflow.execute_activity_method(
                ValidationActivities.diagnose_validation_failure,
                DiagnosisInput(
                    migration_id=params.migration_id,
                    scenario=params.scenario,
                    report=report,
                    plan=params.plan,
                ),
                start_to_close_timeout=AGENT_TIMEOUT,
                retry_policy=AGENT_RETRY,
                summary="diagnose validation failure",
            )
            outcome.diagnosis = diagnosis.diagnosis

        workflow.logger.info(
            "validation finished: %s (%d differences)",
            report.status.value,
            len(report.differences),
        )
        return self._finish(outcome)

    def _finish(self, outcome: ValidationOutcome) -> ValidationOutcome:
        self._outcome = outcome
        return outcome
