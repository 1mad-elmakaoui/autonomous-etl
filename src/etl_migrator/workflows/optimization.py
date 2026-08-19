"""`OptimizationWorkflow`: measure, change, re-validate, re-measure, keep or revert.

    baseline benchmark -> propose -> static gate (if the code changed)
        -> re-validate in full -> benchmark the candidate
        -> evaluate_optimization -> keep or revert

The order is the argument. Correctness is settled before the stopwatch is
consulted, so validation runs before the candidate benchmark and a FAIL
short-circuits the attempt. There is nothing worth timing about a wrong
pipeline.

`evaluate_optimization` is a pure function over a `ValidationReport` and a
`BenchmarkComparison`. The agent's `expected_speedup` is recorded so it can be
compared against what happened, and is never an input to the decision.

If no attempt is accepted the parent keeps the original code and configuration.
A revert is the default, not an error path.
"""

from __future__ import annotations

from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from etl_migrator.activities.migration import MigrationActivities, ValidationActivities
    from etl_migrator.domain.messages import (
        BenchmarkInput,
        OptimizationProposalInput,
        OptimizationWorkflowInput,
        StaticAnalysisInput,
        ValidationWorkflowInput,
    )
    from etl_migrator.domain.optimization import (
        BenchmarkComparison,
        OptimizationAttempt,
        OptimizationOutcome,
        evaluate_optimization,
    )
    from etl_migrator.workflows.validation import ValidationWorkflow

NON_RETRYABLE: list[str] = [
    "ConfigurationError",
    "UnsupportedSourceError",
    "NonRetryableMigrationError",
    "ScriptExhaustedError",
]

AGENT_RETRY = RetryPolicy(
    initial_interval=timedelta(seconds=2),
    backoff_coefficient=2.0,
    maximum_attempts=3,
    non_retryable_error_types=NON_RETRYABLE,
)
AGENT_TIMEOUT = timedelta(minutes=10)

#: Benchmarks are long — several full executions — and a repeat of a failed one
#: is unlikely to succeed, so the budget is deliberately small.
BENCHMARK_RETRY = RetryPolicy(maximum_attempts=2, non_retryable_error_types=NON_RETRYABLE)
BENCHMARK_TIMEOUT = timedelta(hours=1)
VALIDATION_TIMEOUT = timedelta(hours=2)

#: A slug the agent may return to decline. Declining costs nothing and is a
#: better answer than a change invented to look busy.
NO_CHANGE = "no_change"


def execution_queue(configured: str) -> str | None:
    """Where the untrusted-code activities should run.

    `None` tells Temporal "the workflow's own queue", which is what a single
    worker wants. A configured value routes them to the worker deployed with no
    internet egress — the split that makes the sandbox's network story real
    rather than aspirational (see `k8s/` and `temporal/worker.py`).
    """
    return configured or None


@workflow.defn(name="OptimizationWorkflow")
class OptimizationWorkflow:
    """Try bounded optimisations, keeping only those that measure better."""

    def __init__(self) -> None:
        self._outcome = OptimizationOutcome()

    @workflow.query
    def outcome(self) -> OptimizationOutcome:
        """Live view, including attempts already measured and rejected."""
        return self._outcome

    @workflow.run
    async def run(self, params: OptimizationWorkflowInput) -> OptimizationOutcome:
        baseline = await workflow.execute_activity_method(
            ValidationActivities.benchmark_spark,
            BenchmarkInput(
                migration_id=params.migration_id,
                label="baseline",
                code=params.code,
                input_dir=params.input_dir,
                execution_strategy=params.plan.execution_strategy,
                runs=params.benchmark_runs,
                warmups=params.benchmark_warmups,
            ),
            start_to_close_timeout=BENCHMARK_TIMEOUT,
            retry_policy=BENCHMARK_RETRY,
            task_queue=execution_queue(params.execution_task_queue),
            summary="benchmark baseline",
        )
        self._outcome.baseline = baseline
        self._outcome.final = baseline

        if baseline.failed:
            workflow.logger.warning(
                "baseline benchmark failed; nothing can be measured against it"
            )
            return self._outcome

        workflow.logger.info(
            "optimisation starting: baseline median %.3fs, noise %.1f%%",
            baseline.median,
            baseline.noise_ratio * 100,
        )

        tried: list[str] = []
        current_code = params.code
        current_strategy = params.plan.execution_strategy

        for attempt in range(1, params.max_attempts + 1):
            proposal_result = await workflow.execute_activity_method(
                ValidationActivities.propose_optimization,
                OptimizationProposalInput(
                    migration_id=params.migration_id,
                    scenario=params.scenario,
                    attempt=attempt,
                    max_attempts=params.max_attempts,
                    plan=params.plan,
                    code=current_code,
                    input_dir=params.input_dir,
                    baseline=baseline,
                    history=self._outcome.attempts,
                ),
                start_to_close_timeout=AGENT_TIMEOUT,
                retry_policy=AGENT_RETRY,
                summary=f"propose optimisation {attempt}/{params.max_attempts}",
            )
            proposal = proposal_result.proposal
            approach = proposal.strategy.approach

            # --- the agent declined -------------------------------------
            if approach == NO_CHANGE:
                workflow.logger.info("optimizer declined: %s", proposal.strategy.rationale)
                self._outcome.attempts.append(
                    OptimizationAttempt(
                        attempt=attempt,
                        strategy=proposal.strategy,
                        admitted=False,
                        rejection_reason="the optimizer found no grounded opportunity",
                        verdict="declined: no grounded opportunity",
                    )
                )
                break

            # --- already spent, refused before measuring -----------------
            if approach in tried:
                workflow.logger.info("attempt %d repeats '%s'; refused", attempt, approach)
                self._outcome.attempts.append(
                    OptimizationAttempt(
                        attempt=attempt,
                        strategy=proposal.strategy,
                        admitted=False,
                        rejection_reason=f"approach '{approach}' was already measured",
                        verdict=f"refused: '{approach}' already measured",
                    )
                )
                continue

            tried.append(approach)
            record = OptimizationAttempt(attempt=attempt, strategy=proposal.strategy)
            self._outcome.attempts.append(record)

            candidate_code = proposal.code or current_code
            candidate_strategy = proposal.execution_strategy or current_strategy

            # --- the gate, only when code changed ------------------------
            if proposal.code is not None:
                analysis = await workflow.execute_activity_method(
                    MigrationActivities.run_static_analysis,
                    StaticAnalysisInput(
                        migration_id=params.migration_id,
                        code=proposal.code,
                        gate_iterations=proposal_result.gate_iterations,
                    ),
                    start_to_close_timeout=timedelta(minutes=2),
                    retry_policy=BENCHMARK_RETRY,
                    summary=f"gate optimisation {attempt}",
                )
                if not analysis.passed:
                    record.verdict = "rejected: optimised code failed the static gate"
                    continue

            # --- correctness, before the stopwatch -----------------------
            validation = await workflow.execute_child_workflow(
                ValidationWorkflow.run,
                ValidationWorkflowInput(
                    migration_id=params.migration_id,
                    legacy_source_path=params.legacy_source_path,
                    input_dir=params.input_dir,
                    scenario=params.scenario,
                    spec=params.spec,
                    plan=params.plan.model_copy(
                        update={"execution_strategy": candidate_strategy}
                    ),
                    code=candidate_code,
                    run_generated_tests=False,
                    execution_task_queue=params.execution_task_queue,
                ),
                id=f"{params.migration_id}-optimize-{attempt}-validation",
                execution_timeout=VALIDATION_TIMEOUT,
            )
            record.validation_status = validation.report.status.value

            if not validation.passed:
                # No point timing a wrong answer.
                record.verdict = (
                    f"rejected: validation {validation.report.status.value} after the "
                    "change — an optimisation that alters the output is a regression"
                )
                workflow.logger.info("attempt %d broke correctness; reverted", attempt)
                continue

            # --- now measure ---------------------------------------------
            candidate = await workflow.execute_activity_method(
                ValidationActivities.benchmark_spark,
                BenchmarkInput(
                    migration_id=params.migration_id,
                    label=f"candidate{attempt}",
                    code=candidate_code,
                    input_dir=params.input_dir,
                    execution_strategy=candidate_strategy,
                    runs=params.benchmark_runs,
                    warmups=params.benchmark_warmups,
                ),
                start_to_close_timeout=BENCHMARK_TIMEOUT,
                retry_policy=BENCHMARK_RETRY,
                task_queue=execution_queue(params.execution_task_queue),
                summary=f"benchmark candidate {attempt}",
            )
            comparison = BenchmarkComparison(
                baseline=baseline, candidate=candidate, min_speedup=params.min_speedup
            )
            record.comparison = comparison

            accepted, verdict = evaluate_optimization(
                validation=validation.report, comparison=comparison
            )
            record.accepted = accepted
            record.verdict = verdict
            workflow.logger.info("attempt %d: %s", attempt, verdict)

            if accepted:
                self._outcome.applied = True
                self._outcome.accepted_strategy = proposal.strategy
                self._outcome.optimized_code = proposal.code
                self._outcome.optimized_execution_strategy = candidate_strategy
                self._outcome.final = candidate
                # Subsequent attempts would need a new baseline to be meaningful,
                # so one accepted change per run keeps every number attributable.
                break

        workflow.logger.info(
            "optimisation finished: %s",
            "applied" if self._outcome.applied else "nothing kept",
        )
        return self._outcome
