"""The migration plan: how each legacy transformation becomes Spark, and what
could go wrong when it does.

The important type here is `SemanticDifference`. It is the mechanism that turns
"pandas and Spark differ subtly" from tribal knowledge into machine-readable
obligations: every difference the planner declares becomes (a) a required
validation check and (b) a generated test. A planner that stays silent about
null semantics therefore produces a *weaker* validation suite, and the
difference shows up as a real diff later — so silence is not free.
"""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from etl_migrator.domain.enums import RiskCategory, RiskLevel, TransformKind
from etl_migrator.domain.spec import StrictModel


class SemanticDifference(StrictModel):
    category: RiskCategory
    description: str = Field(description="What differs between pandas/SQL and Spark, concretely.")
    mitigation: str = Field(description="What the generated Spark code does about it.")
    validation_check: str = Field(
        description="Name of the differ check that proves the mitigation worked, "
        "e.g. 'null_counts', 'row_count', 'numeric_tolerance', 'duplicate_counts'."
    )


class PlanStep(StrictModel):
    id: str = Field(pattern=r"^s\d+$")
    transformation_id: str = Field(pattern=r"^t\d+$")
    kind: TransformKind
    legacy_construct: str = Field(description="e.g. 'pandas.DataFrame.merge'")
    spark_construct: str = Field(description="e.g. 'DataFrame.join'")
    rationale: str = Field(description="Why this mapping, in one or two sentences.")
    semantic_differences: list[SemanticDifference] = Field(default_factory=list)
    risk_level: RiskLevel = RiskLevel.LOW


class ExecutionStrategy(StrictModel):
    """Cluster-level decisions, distinct from per-step mapping decisions."""

    shuffle_partitions: int = Field(default=200, ge=1, le=10_000)
    adaptive_query_execution: bool = True
    broadcast_datasets: list[str] = Field(
        default_factory=list, description="DataSource names small enough to broadcast."
    )
    cache_datasets: list[str] = Field(
        default_factory=list, description="Datasets consumed more than once."
    )
    repartition_by: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class ValidationPlan(StrictModel):
    """What 'equivalent' means for *this* migration. Set before any code runs, so
    the criteria cannot be relaxed after seeing the diff."""

    join_keys: list[str] = Field(
        default_factory=list,
        description="Columns that identify a row for record-level comparison. Empty means "
        "the differ falls back to sorted full-row comparison.",
    )
    numeric_tolerance: float = Field(default=1e-9, ge=0.0)
    ignore_row_order: bool = True
    required_checks: list[str] = Field(
        default_factory=lambda: [
            "schema",
            "row_count",
            "null_counts",
            "numeric_tolerance",
            "duplicate_counts",
            "aggregate_checksums",
        ]
    )


class MigrationPlan(StrictModel):
    """Produced by PlannerAgent from a MigrationSpec."""

    summary: str
    steps: list[PlanStep] = Field(min_length=1)
    execution_strategy: ExecutionStrategy = Field(default_factory=ExecutionStrategy)
    validation_plan: ValidationPlan = Field(default_factory=ValidationPlan)
    high_risk_transformation_ids: list[str] = Field(default_factory=list)
    overall_risk: RiskLevel = RiskLevel.LOW
    requires_human_approval: bool = Field(
        default=False,
        description="Advisory only. The workflow recomputes this from overall_risk; an agent "
        "cannot talk its way out of an approval gate.",
    )

    @model_validator(mode="after")
    def _unique_step_ids(self) -> Self:
        ids = [s.id for s in self.steps]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate plan step ids: {ids}")
        return self

    @property
    def all_semantic_differences(self) -> list[SemanticDifference]:
        return [d for s in self.steps for d in s.semantic_differences]

    def effective_required_checks(self) -> list[str]:
        """Base checks plus every check demanded by a declared semantic difference."""
        checks = list(self.validation_plan.required_checks)
        for diff in self.all_semantic_differences:
            if diff.validation_check not in checks:
                checks.append(diff.validation_check)
        return checks
