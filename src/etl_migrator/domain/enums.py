"""Closed vocabularies shared by every agent, activity and workflow.

These enums are deliberately small and closed. They are the interface between
the *reasoning* layer (which chooses a member) and the *deterministic* layer
(which switches on it). A free-text string here would push the switch back into
the LLM, which is exactly what this project avoids.
"""

from __future__ import annotations

from enum import StrEnum


class SourceLanguage(StrEnum):
    PYTHON_PANDAS = "python_pandas"
    SQL = "sql"
    SHELL = "shell"
    PYTHON_PLAIN = "python_plain"


class DataFormat(StrEnum):
    CSV = "csv"
    PARQUET = "parquet"
    JSON = "json"
    JDBC = "jdbc"
    UNKNOWN = "unknown"


class WriteMode(StrEnum):
    OVERWRITE = "overwrite"
    APPEND = "append"
    ERROR_IF_EXISTS = "errorifexists"


class TransformKind(StrEnum):
    """What a legacy operation *does*, independent of how it was written.

    Discovery normalises `df[df.x > 1]`, `df.query(...)` and `WHERE x > 1` to
    FILTER. The planner then maps FILTER -> Spark once, instead of once per
    syntactic form.
    """

    READ = "read"
    WRITE = "write"
    FILTER = "filter"
    PROJECT = "project"
    DERIVE_COLUMN = "derive_column"
    JOIN = "join"
    AGGREGATE = "aggregate"
    SORT = "sort"
    DEDUPLICATE = "deduplicate"
    RENAME = "rename"
    FILLNA = "fillna"
    DROPNA = "dropna"
    ASTYPE = "astype"
    WINDOW = "window"
    PIVOT = "pivot"
    UNION = "union"
    APPLY_UDF = "apply_udf"
    RESET_INDEX = "reset_index"
    UNKNOWN = "unknown"


class JoinType(StrEnum):
    INNER = "inner"
    LEFT = "left"
    RIGHT = "right"
    OUTER = "outer"
    CROSS = "cross"
    SEMI = "semi"
    ANTI = "anti"


class RiskLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

    @property
    def rank(self) -> int:
        return {"low": 0, "medium": 1, "high": 2}[self.value]


class RiskCategory(StrEnum):
    NULL_SEMANTICS = "null_semantics"
    TYPE_COERCION = "type_coercion"
    ROW_ORDER = "row_order"
    DUPLICATE_EXPLOSION = "duplicate_explosion"
    FLOATING_POINT = "floating_point"
    INDEX_SEMANTICS = "index_semantics"
    UDF_SEMANTICS = "udf_semantics"
    NON_DETERMINISM = "non_determinism"
    UNSUPPORTED_CONSTRUCT = "unsupported_construct"
    DATA_SKEW = "data_skew"
    SCALE = "scale"
    EXTERNAL_DEPENDENCY = "external_dependency"


class Severity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class MigrationStage(StrEnum):
    """Every stage of the lifecycle. Used as the durable cursor so a migration
    resumes at the failed stage instead of restarting from zero."""

    DISCOVERY = "discovery"
    PLANNING = "planning"
    APPROVAL = "approval"
    CODE_GENERATION = "code_generation"
    STATIC_ANALYSIS = "static_analysis"
    TEST_GENERATION = "test_generation"
    LEGACY_EXECUTION = "legacy_execution"
    SPARK_EXECUTION = "spark_execution"
    VALIDATION = "validation"
    REPAIR = "repair"
    OPTIMIZATION = "optimization"
    BENCHMARK = "benchmark"
    PULL_REQUEST = "pull_request"
    DEPLOYMENT = "deployment"
    VERIFICATION = "verification"


class ValidationStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    ERROR = "ERROR"
