"""`RepairWorkflow`: bounded, durable, and refuses to try the same thing twice.

    propose -> admit? -> static gate -> re-validate -> pass or next attempt

The ledger decides admissibility, not the agent. A repeated strategy signature,
or code byte-equivalent to something already tried, is rejected before any Spark
job runs, so the attempt budget means "distinct ideas tried".

A rejected proposal still consumes an attempt and the rejection is fed into the
next prompt. An agent that cannot produce a distinct strategy has run out of
ideas.

Exhaustion returns a `RepairOutcome` carrying every attempt and the nearest
miss, so a human inherits a diagnosis rather than a stack trace.

Validation re-runs as a nested child workflow, one per attempt, so each shows
separately in the Temporal UI with its own retry budget.
"""

from __future__ import annotations

from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from etl_migrator.activities.migration import MigrationActivities, ValidationActivities
    from etl_migrator.domain.messages import (
        RepairProposalInput,
        RepairWorkflowInput,
        StaticAnalysisInput,
        ValidationWorkflowInput,
    )
    from etl_migrator.domain.repair import (
        RepairAttempt,
        RepairLedger,
        RepairOutcome,
        code_fingerprint,
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
    maximum_interval=timedelta(seconds=30),
    maximum_attempts=3,
    non_retryable_error_types=NON_RETRYABLE,
)
AGENT_TIMEOUT = timedelta(minutes=10)

#: Ceiling for one attempt's validation child. Generous: it executes two
#: pipelines and a generated test suite.
VALIDATION_TIMEOUT = timedelta(hours=2)


@workflow.defn(name="RepairWorkflow")
class RepairWorkflow:
    """Try a bounded number of *distinct* fixes, re-validating after each."""

    def __init__(self) -> None:
        self._outcome = RepairOutcome()

    @workflow.query
    def outcome(self) -> RepairOutcome:
        """Live view of the loop, including attempts already spent."""
        return self._outcome

    @workflow.run
    async def run(self, params: RepairWorkflowInput) -> RepairOutcome:
        ledger = RepairLedger(params.max_attempts)
        ledger.register_baseline(params.code)

        report = params.report
        diagnosis = params.diagnosis
        current_code = params.code

        workflow.logger.info(
            "repair loop starting: %d differences, budget %d attempts",
            len(report.differences),
            params.max_attempts,
        )

        for attempt in range(1, params.max_attempts + 1):
            proposal_result = await workflow.execute_activity_method(
                ValidationActivities.propose_repair,
                RepairProposalInput(
                    migration_id=params.migration_id,
                    scenario=params.scenario,
                    attempt=attempt,
                    max_attempts=params.max_attempts,
                    plan=params.plan,
                    code=current_code,
                    report=report,
                    diagnosis=diagnosis,
                    history=self._outcome.attempts,
                ),
                start_to_close_timeout=AGENT_TIMEOUT,
                retry_policy=AGENT_RETRY,
                summary=f"propose repair {attempt}/{params.max_attempts}",
            )
            proposal = proposal_result.proposal

            # --- admissibility, decided without an LLM and without Spark ----
            admissible, reason = ledger.admits(proposal)
            if not admissible:
                workflow.logger.info("attempt %d rejected: %s", attempt, reason)
                self._outcome.attempts.append(
                    RepairAttempt(
                        attempt=attempt,
                        strategy=proposal.strategy,
                        code_sha256=code_fingerprint(proposal.code),
                        admitted=False,
                        rejection_reason=reason,
                        expected_effect=proposal.expected_effect,
                    )
                )
                continue

            ledger.record(proposal)
            record = RepairAttempt(
                attempt=attempt,
                strategy=proposal.strategy,
                code_sha256=code_fingerprint(proposal.code),
                expected_effect=proposal.expected_effect,
            )
            self._outcome.attempts.append(record)

            # --- the gate, before paying for execution ----------------------
            analysis = await workflow.execute_activity_method(
                MigrationActivities.run_static_analysis,
                StaticAnalysisInput(
                    migration_id=params.migration_id,
                    code=proposal.code,
                    gate_iterations=proposal_result.gate_iterations,
                ),
                start_to_close_timeout=timedelta(minutes=2),
                retry_policy=RetryPolicy(
                    maximum_attempts=2, non_retryable_error_types=NON_RETRYABLE
                ),
                summary=f"gate repair {attempt}",
            )
            record.static_analysis = analysis.result.static_analysis
            if not analysis.passed:
                workflow.logger.info(
                    "attempt %d failed the static gate; not executing it", attempt
                )
                continue

            # --- re-validate as a nested child ------------------------------
            outcome = await workflow.execute_child_workflow(
                ValidationWorkflow.run,
                ValidationWorkflowInput(
                    migration_id=params.migration_id,
                    legacy_source_path=params.legacy_source_path,
                    input_dir=params.input_dir,
                    scenario=params.scenario,
                    spec=params.spec,
                    plan=params.plan,
                    code=proposal.code,
                    run_generated_tests=params.run_generated_tests,
                    execution_task_queue=params.execution_task_queue,
                ),
                id=f"{params.migration_id}-repair-{attempt}-validation",
                execution_timeout=VALIDATION_TIMEOUT,
            )
            record.validation_status = outcome.report.status
            record.differences = len(outcome.report.differences)

            if outcome.passed:
                workflow.logger.info("repair succeeded on attempt %d", attempt)
                self._outcome.succeeded = True
                self._outcome.repaired_code = proposal.code
                self._outcome.final_report = outcome.report
                return self._outcome

            # Carry the *new* failure forward: the next attempt should repair
            # what is wrong now, not what was wrong before this change.
            report = outcome.report
            diagnosis = outcome.diagnosis
            current_code = proposal.code
            self._outcome.final_report = outcome.report

        self._outcome.exhausted = True
        workflow.logger.warning(
            "repair exhausted after %d attempts; escalating to a human",
            params.max_attempts,
        )
        return self._outcome
