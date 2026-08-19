"""`DeliveryWorkflow`: decide, narrate, audit, and only then open.

    decide_delivery -> propose narrative -> audit against the record
        -> revise, up to a bound -> branch, commit, PR, label

Two gates, neither the agent's to open.

`decide_delivery` runs first, in workflow code, from the record the parent
already holds. An unvalidated migration never reaches the agent, because there
is no point paying for prose describing a pull request that will not exist.

The audit runs second, computed in the activity against the same record. The
workflow only decides what to do when it fails: one more revision, or stop.
Exhausting the revisions means no PR, not a PR with a disclaimer.

Everything after that is idempotent by construction (see `github/client.py`), so
a retried activity re-attaches to the branch and PR it already made.
"""

from __future__ import annotations

from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from etl_migrator.activities.migration import DeliveryActivities
    from etl_migrator.domain.delivery import ClaimAudit, DeliveryOutcome
    from etl_migrator.domain.delivery_policy import decide_delivery
    from etl_migrator.domain.messages import (
        DeliverPullRequestInput,
        DeliveryWorkflowInput,
        PullRequestNarrativeInput,
    )

NON_RETRYABLE: list[str] = [
    "ConfigurationError",
    "UnsupportedSourceError",
    "NonRetryableMigrationError",
    "ScriptExhaustedError",
    "GitHubError",
]

AGENT_RETRY = RetryPolicy(
    initial_interval=timedelta(seconds=2),
    backoff_coefficient=2.0,
    maximum_attempts=3,
    non_retryable_error_types=NON_RETRYABLE,
)
AGENT_TIMEOUT = timedelta(minutes=10)

#: GitHub failures split cleanly. A 5xx or a timeout is worth retrying; a 403,
#: a 404 or a 422 will say the same thing on the fourth attempt, so `GitHubError`
#: is non-retryable and the transport's own exceptions are not.
DELIVERY_RETRY = RetryPolicy(
    initial_interval=timedelta(seconds=5),
    backoff_coefficient=2.0,
    maximum_attempts=4,
    non_retryable_error_types=NON_RETRYABLE,
)
DELIVERY_TIMEOUT = timedelta(minutes=10)


@workflow.defn(name="DeliveryWorkflow")
class DeliveryWorkflow:
    """Open the pull request, if the migration has earned one."""

    def __init__(self) -> None:
        self._outcome = DeliveryOutcome()

    @workflow.query
    def outcome(self) -> DeliveryOutcome:
        """Live view, including a refusal and the reason for it."""
        return self._outcome

    @workflow.run
    async def run(self, params: DeliveryWorkflowInput) -> DeliveryOutcome:
        decision = decide_delivery(params.record)
        self._outcome.decision = decision
        workflow.logger.info(
            "delivery decided: %s — %s", decision.disposition.value, decision.reason
        )

        if not decision.should_open:
            # No branch, no agent call, no PR. There is nothing a reviewer could
            # usefully do with this migration.
            self._outcome.skipped_reason = decision.reason
            return self._outcome

        audit: ClaimAudit | None = None
        for revision in range(params.max_narrative_revisions + 1):
            proposal = await workflow.execute_activity_method(
                DeliveryActivities.propose_pr_narrative,
                PullRequestNarrativeInput(
                    migration_id=params.migration_id,
                    scenario=params.scenario,
                    record=params.record,
                    decision=decision,
                    previous_audit=audit if audit is not None and not audit.passed else None,
                ),
                start_to_close_timeout=AGENT_TIMEOUT,
                retry_policy=AGENT_RETRY,
                summary=f"write PR narrative (revision {revision})",
            )
            audit = proposal.audit
            self._outcome.audit = audit
            self._outcome.narrative_revisions = revision

            if audit.passed:
                delivered = await workflow.execute_activity_method(
                    DeliveryActivities.deliver_pull_request,
                    DeliverPullRequestInput(
                        migration_id=params.migration_id,
                        record=params.record,
                        decision=decision,
                        narrative=proposal.narrative,
                        branch=f"{params.branch_prefix}/{params.migration_id}",
                        directory=params.directory,
                    ),
                    start_to_close_timeout=DELIVERY_TIMEOUT,
                    retry_policy=DELIVERY_RETRY,
                    summary="open pull request",
                )
                delivered.audit = audit
                delivered.narrative_revisions = revision
                self._outcome = delivered
                workflow.logger.info(
                    "delivery opened PR #%s",
                    delivered.pull_request.number if delivered.pull_request else "?",
                )
                return self._outcome

            workflow.logger.warning(
                "narrative rejected: %d claim(s) the record does not support",
                len(audit.violations),
            )

        # The prose could not be made to match the measurements. Opening the PR
        # anyway would publish the overstatement.
        self._outcome.skipped_reason = (
            "the PR narrative made claims the migration record does not support, "
            f"after {params.max_narrative_revisions} revision(s)"
        )
        return self._outcome
