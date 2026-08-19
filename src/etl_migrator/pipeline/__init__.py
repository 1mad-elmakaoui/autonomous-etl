"""Orchestration.

`steps` holds the work. `local` runs it sequentially in-process; the Temporal
workflow in `etl_migrator.workflows` runs the same steps durably. Neither
orchestrator owns an implementation of its own.
"""

from etl_migrator.pipeline.local import (
    ApprovalResolver,
    LocalMigrationPipeline,
    MigrationRequest,
    auto_approve,
    require_manual_approval,
)
from etl_migrator.pipeline.steps import StepContext

__all__ = [
    "ApprovalResolver",
    "LocalMigrationPipeline",
    "MigrationRequest",
    "StepContext",
    "auto_approve",
    "require_manual_approval",
]
