"""Temporal client construction and the submit/signal/query helpers.

One place decides the data converter, because client and worker must agree on
it. Using `pydantic_data_converter` means the domain models cross the wire as
themselves — no hand-written serialisation, and a schema violation surfaces as
a `ValidationError` at the boundary instead of a `dict` masquerading as a
`MigrationPlan` three stages later.
"""

from __future__ import annotations

from pathlib import Path

from temporalio.client import Client, WorkflowHandle
from temporalio.common import WorkflowIDConflictPolicy
from temporalio.contrib.pydantic import pydantic_data_converter

from etl_migrator.config import Settings
from etl_migrator.domain.artifacts import MigrationRecord
from etl_migrator.domain.messages import (
    DEFAULT_APPROVAL_TIMEOUT_SECONDS,
    ApprovalDecision,
    MigrationStatus,
    MigrationWorkflowInput,
)
from etl_migrator.ids import new_migration_id
from etl_migrator.observability import get_logger

log = get_logger(__name__)

WORKFLOW_NAME = "ETLMigrationWorkflow"

__all__ = [
    "WORKFLOW_NAME",
    "build_input",
    "connect",
    "handle_for",
    "new_migration_id",
    "query_report",
    "query_status",
    "send_abort",
    "send_approval",
    "submit",
    "wait_for_record",
]


async def connect(settings: Settings) -> Client:
    log.info(
        "temporal.connect",
        host=settings.temporal_host,
        namespace=settings.temporal_namespace,
    )
    return await Client.connect(
        settings.temporal_host,
        namespace=settings.temporal_namespace,
        data_converter=pydantic_data_converter,
    )


def build_input(
    source_path: Path,
    input_dir: Path,
    *,
    migration_id: str | None = None,
    output_filename: str | None = None,
    scenario: str = "customer_pipeline",
    approval_timeout_seconds: int = DEFAULT_APPROVAL_TIMEOUT_SECONDS,
) -> MigrationWorkflowInput:
    source = source_path.resolve()
    return MigrationWorkflowInput(
        migration_id=migration_id or new_migration_id(),
        source_path=str(source),
        input_dir=str(input_dir.resolve()),
        output_filename=output_filename or f"{source.stem}_spark.py",
        scenario=scenario,
        approval_timeout_seconds=approval_timeout_seconds,
    )


async def submit(
    client: Client, settings: Settings, params: MigrationWorkflowInput
) -> WorkflowHandle[object, MigrationRecord]:
    """Start a migration, or attach to the one already running under this id.

    The workflow id *is* the migration id, which is what makes a double submit
    safe: `USE_EXISTING` returns a handle to the running execution instead of
    raising. Without it the second submit fails with
    `WorkflowAlreadyStartedError` — and a retried submit is exactly the case
    this id scheme exists to make harmless.

    `id_reuse_policy` is a different question and not the one being answered
    here: it governs starting a *new* run once this one has closed.

    `result_type` is not decoration. Called by workflow *name* rather than by
    method reference, the SDK has no type to decode into, and `handle.result()`
    hands back a bare `dict` that fails on the first attribute access.
    """
    handle: WorkflowHandle[object, MigrationRecord] = await client.start_workflow(
        WORKFLOW_NAME,
        params,
        id=params.migration_id,
        task_queue=settings.temporal_task_queue,
        id_conflict_policy=WorkflowIDConflictPolicy.USE_EXISTING,
        result_type=MigrationRecord,
    )
    # `handle.run_id` is documented as unset by `start_workflow` -- it exists to
    # pin operations to an exact run, which only `get_workflow_handle` does. The
    # run that was actually started is `first_execution_run_id`; logging the
    # other one printed a bare "None" that reads as a failure.
    log.info(
        "migration.submitted",
        migration_id=params.migration_id,
        run_id=handle.first_execution_run_id,
    )
    return handle


def handle_for(client: Client, migration_id: str) -> WorkflowHandle[object, MigrationRecord]:
    """A handle that decodes its result as a `MigrationRecord`.

    `client.get_workflow_handle(id)` alone carries no result type, so awaiting
    its `.result()` yields a `dict`. Anything that waits on a migration should
    come through here.
    """
    return client.get_workflow_handle(migration_id, result_type=MigrationRecord)


async def wait_for_record(client: Client, migration_id: str) -> MigrationRecord:
    """Block until the migration finishes and return its record."""
    return await handle_for(client, migration_id).result()


async def send_approval(
    client: Client, migration_id: str, decision: ApprovalDecision
) -> None:
    await client.get_workflow_handle(migration_id).signal("approve", decision)
    log.info(
        "approval.sent",
        migration_id=migration_id,
        approved=decision.approved,
        actor=decision.actor,
    )


async def send_abort(client: Client, migration_id: str, reason: str) -> None:
    await client.get_workflow_handle(migration_id).signal("abort", reason)
    log.info("abort.sent", migration_id=migration_id, reason=reason)


async def query_status(client: Client, migration_id: str) -> MigrationStatus:
    result = await client.get_workflow_handle(migration_id).query("status")
    return MigrationStatus.model_validate(result)


async def query_report(client: Client, migration_id: str) -> MigrationRecord | None:
    result = await client.get_workflow_handle(migration_id).query("report")
    return None if result is None else MigrationRecord.model_validate(result)
