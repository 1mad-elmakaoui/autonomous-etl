"""The run-level record of a migration.

Everything here is id-keyed so that a Temporal retry that re-executes an
activity overwrites byte-identical output instead of appending duplicates
(idempotency requirement, R8).

`Finding`, `StaticAnalysisReport`, `GeneratedCode` and `CodeGenResult` live in
`domain.code` and are re-exported here, because they were part of this module's
public surface before the layering split and callers should not have to care.
"""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import Field

from etl_migrator.domain.code import (
    CodeGenResult,
    Finding,
    GeneratedCode,
    StaticAnalysisReport,
)
from etl_migrator.domain.delivery import DeliveryOutcome
from etl_migrator.domain.enums import MigrationStage, RiskLevel
from etl_migrator.domain.optimization import OptimizationOutcome
from etl_migrator.domain.plan import MigrationPlan
from etl_migrator.domain.repair import RepairOutcome
from etl_migrator.domain.spec import MigrationSpec, StrictModel
from etl_migrator.domain.validation import ValidationOutcome

__all__ = [
    "CodeGenResult",
    "Finding",
    "GeneratedCode",
    "MigrationRecord",
    "StageRecord",
    "StaticAnalysisReport",
]


def _utcnow() -> datetime:
    return datetime.now(UTC)


class StageRecord(StrictModel):
    """One entry per stage attempt. Feeds Prometheus and the history harvester."""

    stage: MigrationStage
    started_at: datetime
    ended_at: datetime | None = None
    succeeded: bool | None = None
    attempt: int = Field(default=1, ge=1)
    detail: str | None = None

    @property
    def duration_seconds(self) -> float | None:
        if self.ended_at is None:
            return None
        return (self.ended_at - self.started_at).total_seconds()


class MigrationRecord(StrictModel):
    """The durable, queryable state of one migration.

    This is the Temporal workflow's state object, returned by a
    `@workflow.query`. It is JSON-serialisable by construction.
    """

    migration_id: str
    source_path: str
    created_at: datetime = Field(default_factory=_utcnow)
    stage: MigrationStage = MigrationStage.DISCOVERY
    spec: MigrationSpec | None = None
    plan: MigrationPlan | None = None
    codegen: CodeGenResult | None = None
    validation: ValidationOutcome | None = None
    repair: RepairOutcome | None = None
    optimization: OptimizationOutcome | None = None
    delivery: DeliveryOutcome | None = None
    stages: list[StageRecord] = Field(default_factory=list)
    artifact_dir: str | None = None
    failed: bool = False
    failure_reason: str | None = None

    @property
    def risk(self) -> RiskLevel:
        if self.plan is not None:
            return self.plan.overall_risk
        if self.spec is not None:
            return self.spec.max_risk
        return RiskLevel.LOW

    @property
    def total_duration_seconds(self) -> float:
        return sum(s.duration_seconds or 0.0 for s in self.stages)
