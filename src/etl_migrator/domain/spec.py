"""The migration specification: what Discovery perceives about a legacy pipeline.

This is the single artifact that decouples "understanding the legacy code" from
"writing Spark". Everything downstream consumes `MigrationSpec` and never reads
the legacy source again.
"""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from etl_migrator.domain.enums import (
    DataFormat,
    JoinType,
    RiskCategory,
    RiskLevel,
    SourceLanguage,
    TransformKind,
    WriteMode,
)


class StrictModel(BaseModel):
    """Base for every domain model.

    `extra="forbid"` matters more than it looks: when an LLM invents a field, we
    want a loud ValidationError inside a retryable activity, not a silently
    dropped instruction.
    """

    model_config = ConfigDict(extra="forbid", frozen=False, use_enum_values=False)

    @model_validator(mode="before")
    @classmethod
    def _drop_computed_fields(cls, data: object) -> object:
        """Let a model that was serialised with computed fields load again.

        Pydantic writes `@computed_field` values on dump and refuses them on
        load, so `extra="forbid"` makes any model carrying one impossible to
        round-trip. Every verdict here is a computed field precisely so no
        caller can set one, which means the models that matter most were the
        ones that could not survive the trip. `MigrationRecord` crosses JSON on
        the durable path and again when artifacts are re-read, so this failed
        for real.

        Dropping exactly the computed-field keys keeps `forbid` doing its job on
        hallucinated fields, and the values are recomputed on load.
        """
        if isinstance(data, dict) and cls.model_computed_fields:
            computed = cls.model_computed_fields
            present = [key for key in data if key in computed]
            if present:
                return {key: value for key, value in data.items() if key not in computed}
        return data


class ColumnSpec(StrictModel):
    name: str
    dtype: str = Field(description="Source dtype verbatim, e.g. 'int64', 'object', 'float64'.")
    nullable: bool = True
    description: str | None = None
    example_values: list[str] = Field(default_factory=list, max_length=5)


class DataSource(StrictModel):
    name: str = Field(description="Stable identifier used by transformations, e.g. 'customers'.")
    path_expression: str = Field(description="Path/URI/table as it appears in the legacy source.")
    format: DataFormat = DataFormat.UNKNOWN
    read_options: dict[str, str] = Field(default_factory=dict)
    columns: list[ColumnSpec] = Field(default_factory=list)
    row_estimate: int | None = Field(default=None, ge=0)
    profiled: bool = Field(
        default=False,
        description="True only if a real profiler read the file. False means the schema was "
        "inferred from source code alone and must be treated as a hypothesis.",
    )


class DataSink(StrictModel):
    name: str
    path_expression: str
    format: DataFormat = DataFormat.UNKNOWN
    mode: WriteMode = WriteMode.OVERWRITE
    write_options: dict[str, str] = Field(default_factory=dict)
    columns: list[ColumnSpec] = Field(default_factory=list)


class JoinDetail(StrictModel):
    left: str
    right: str
    how: JoinType
    left_keys: list[str] = Field(min_length=1)
    right_keys: list[str] = Field(min_length=1)
    broadcast_candidate: bool = False

    @model_validator(mode="after")
    def _keys_align(self) -> Self:
        if self.how is not JoinType.CROSS and len(self.left_keys) != len(self.right_keys):
            raise ValueError(
                f"join key arity mismatch: {self.left_keys} vs {self.right_keys}"
            )
        return self


class AggregationDetail(StrictModel):
    group_by: list[str] = Field(default_factory=list)
    aggregations: dict[str, str] = Field(
        default_factory=dict,
        description="column -> aggregate function name, e.g. {'revenue': 'sum'}.",
    )


class Transformation(StrictModel):
    id: str = Field(pattern=r"^t\d+$", description="Stable ordinal id: t1, t2, ...")
    kind: TransformKind
    description: str
    inputs: list[str] = Field(default_factory=list, description="Names of upstream datasets.")
    output: str = Field(description="Name of the dataset produced by this step.")
    source_lines: list[int] = Field(
        default_factory=list, description="1-based line numbers in the legacy source."
    )
    expression: str | None = Field(
        default=None, description="Legacy expression verbatim, for traceability."
    )
    columns_read: list[str] = Field(default_factory=list)
    columns_written: list[str] = Field(default_factory=list)
    join: JoinDetail | None = None
    aggregation: AggregationDetail | None = None


class BusinessRule(StrictModel):
    id: str = Field(pattern=r"^br\d+$")
    description: str = Field(description="Stated in domain language, not code language.")
    transformation_ids: list[str] = Field(default_factory=list)
    criticality: RiskLevel = RiskLevel.MEDIUM


class MigrationRisk(StrictModel):
    id: str = Field(pattern=r"^r\d+$")
    category: RiskCategory
    severity: RiskLevel
    description: str
    transformation_ids: list[str] = Field(default_factory=list)
    mitigation: str = Field(description="Concrete action, not 'be careful'.")


class MigrationSpec(StrictModel):
    """Structured understanding of a legacy pipeline. Produced by DiscoveryAgent."""

    source_path: str
    source_language: SourceLanguage
    summary: str = Field(
        description="Two or three sentences: what this pipeline does, in domain terms."
    )
    inputs: list[DataSource] = Field(default_factory=list)
    outputs: list[DataSink] = Field(default_factory=list)
    transformations: list[Transformation] = Field(default_factory=list)
    business_rules: list[BusinessRule] = Field(default_factory=list)
    risks: list[MigrationRisk] = Field(default_factory=list)
    dependencies: list[str] = Field(
        default_factory=list, description="Third-party modules imported by the legacy pipeline."
    )
    assumptions: list[str] = Field(
        default_factory=list,
        description="Anything the agent could not verify with a tool. Surfaced in the PR body.",
    )
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("transformations")
    @classmethod
    def _unique_ids(cls, v: list[Transformation]) -> list[Transformation]:
        ids = [t.id for t in v]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate transformation ids: {ids}")
        return v

    @property
    def max_risk(self) -> RiskLevel:
        if not self.risks:
            return RiskLevel.LOW
        return max((r.severity for r in self.risks), key=lambda s: s.rank)

    def transformation(self, tid: str) -> Transformation:
        for t in self.transformations:
            if t.id == tid:
                return t
        raise KeyError(f"unknown transformation id: {tid}")
