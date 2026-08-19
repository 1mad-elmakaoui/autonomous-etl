"""Pure state transitions for a migration.

A Temporal workflow must be deterministic, so it cannot read a clock, generate a
uuid, touch the filesystem or call an LLM. That leaves the workflow body as
glue: await an activity, mutate state, branch.

Keeping that mutation inline would make it testable only by running a workflow,
which needs a server. Here it is testable as ordinary functions, and the
workflow stays short enough to read in one sitting.

Every function takes `now` explicitly rather than reading a clock. The workflow
passes `workflow.now()`, the local pipeline passes `datetime.now(UTC)`, and
neither can introduce non-determinism because there is no clock in here to
reach for.
"""

from __future__ import annotations

from datetime import datetime

from etl_migrator.domain.artifacts import MigrationRecord, StageRecord
from etl_migrator.domain.enums import MigrationStage, RiskLevel
from etl_migrator.domain.messages import ApprovalDecision
from etl_migrator.domain.plan import MigrationPlan

#: Stages of the migration lifecycle, in execution order. Extended, not
#: reordered, by later phases.
STAGE_ORDER: tuple[MigrationStage, ...] = (
    MigrationStage.DISCOVERY,
    MigrationStage.PLANNING,
    MigrationStage.APPROVAL,
    MigrationStage.CODE_GENERATION,
    MigrationStage.STATIC_ANALYSIS,
    MigrationStage.VALIDATION,
    MigrationStage.OPTIMIZATION,
)


def new_record(migration_id: str, source_path: str, now: datetime) -> MigrationRecord:
    """Create a record with an explicit creation time.

    `MigrationRecord.created_at` has a `datetime.now` default factory, which is
    fine locally and wrong inside a workflow. Constructing records through this
    function is how the workflow avoids ever triggering that default.
    """
    return MigrationRecord(
        migration_id=migration_id,
        source_path=source_path,
        created_at=now,
        stage=MigrationStage.DISCOVERY,
    )


def begin_stage(
    record: MigrationRecord, stage: MigrationStage, now: datetime, *, attempt: int = 1
) -> StageRecord:
    """Open a stage on the record and return its entry for later completion."""
    record.stage = stage
    entry = StageRecord(stage=stage, started_at=now, attempt=attempt)
    record.stages.append(entry)
    return entry


def complete_stage(entry: StageRecord, now: datetime, *, detail: str | None = None) -> None:
    entry.ended_at = now
    entry.succeeded = True
    entry.detail = detail


def fail_stage(
    record: MigrationRecord, entry: StageRecord, now: datetime, *, reason: str
) -> None:
    entry.ended_at = now
    entry.succeeded = False
    entry.detail = reason
    record.failed = True
    record.failure_reason = reason


def requires_approval(plan: MigrationPlan) -> bool:
    """Policy, evaluated by the orchestrator rather than believed from the agent.

    A planner that sets `requires_human_approval=False` on a HIGH-risk plan is
    overruled. Letting the model answer "may I skip review?" would make the gate
    decorative.
    """
    return plan.overall_risk is RiskLevel.HIGH


def enforce_approval_policy(plan: MigrationPlan) -> MigrationPlan:
    """Return a plan whose approval flag reflects policy, not the agent's wish."""
    required = requires_approval(plan)
    if plan.requires_human_approval == required:
        return plan
    return plan.model_copy(update={"requires_human_approval": required})


def apply_approval(
    record: MigrationRecord,
    entry: StageRecord,
    decision: ApprovalDecision,
    now: datetime,
) -> bool:
    """Record a human decision. Returns True if the migration should continue."""
    detail = f"{'approved' if decision.approved else 'rejected'} by {decision.actor}"
    if decision.reason:
        detail = f"{detail}: {decision.reason}"

    if decision.approved:
        complete_stage(entry, now, detail=detail)
        return True
    fail_stage(record, entry, now, reason=detail)
    return False


def apply_approval_timeout(
    record: MigrationRecord, entry: StageRecord, now: datetime, *, timeout_seconds: int
) -> None:
    """No human answered in time. Terminal, and clearly labelled as such.

    Not a retry: re-asking a human who did not answer produces the same silence.
    """
    fail_stage(
        record,
        entry,
        now,
        reason=(
            f"approval not granted within {timeout_seconds}s; "
            "migration parked for human intervention"
        ),
    )


def apply_abort(record: MigrationRecord, now: datetime, *, reason: str) -> None:
    """Operator-initiated cancellation. Distinguished from a failure in the record."""
    open_stages = [s for s in record.stages if s.ended_at is None]
    for stage in open_stages:
        stage.ended_at = now
        stage.succeeded = False
        stage.detail = f"aborted: {reason}"
    record.failed = True
    record.failure_reason = f"aborted: {reason}"


def resume_point(record: MigrationRecord) -> MigrationStage:
    """The first stage that has not completed successfully.

    Temporal already resumes a *running* workflow mid-flight. This answers the
    different question of where a *resubmitted* migration should pick up, given
    a record recovered from artifacts — so re-running a migration that died at
    code generation does not re-pay for discovery and planning.
    """
    completed = {s.stage for s in record.stages if s.succeeded}
    for stage in STAGE_ORDER:
        if stage is MigrationStage.APPROVAL and not _approval_was_needed(record):
            continue
        if stage not in completed:
            return stage
    return MigrationStage.OPTIMIZATION


def _approval_was_needed(record: MigrationRecord) -> bool:
    return record.plan is not None and record.plan.requires_human_approval


def summarise(record: MigrationRecord) -> str:
    """One-line human summary, used in logs and the CLI."""
    if record.failed:
        return f"{record.migration_id}: FAILED at {record.stage.value} — {record.failure_reason}"
    return (
        f"{record.migration_id}: {record.stage.value} "
        f"(risk={record.risk.value}, {len(record.stages)} stages, "
        f"{record.total_duration_seconds:.2f}s)"
    )
