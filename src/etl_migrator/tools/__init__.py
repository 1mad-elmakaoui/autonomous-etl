"""Deterministic tools: the system's ground truth.

Nothing in this package calls an LLM. Agents invoke these to *observe* reality;
the observations are what agent conclusions are checked against.
"""

from etl_migrator.tools.code_gate import (
    ALLOWED_IMPORT_ROOTS,
    REQUIRED_ENTRYPOINT,
    REQUIRED_PARAMETERS,
    GateOptions,
    analyze_generated_code,
)
from etl_migrator.tools.data_profiler import (
    ColumnProfile,
    DatasetProfile,
    profile_dataset,
    profile_directory,
)
from etl_migrator.tools.source_inspector import (
    BooleanFilter,
    ColumnAssignment,
    SourceCall,
    SourceInspection,
    detect_language,
    inspect_source,
)

__all__ = [
    "ALLOWED_IMPORT_ROOTS",
    "REQUIRED_ENTRYPOINT",
    "REQUIRED_PARAMETERS",
    "BooleanFilter",
    "ColumnAssignment",
    "ColumnProfile",
    "DatasetProfile",
    "GateOptions",
    "SourceCall",
    "SourceInspection",
    "analyze_generated_code",
    "detect_language",
    "inspect_source",
    "profile_dataset",
    "profile_directory",
]
