"""Deterministic profiling of the pipeline's real input data.

Discovery infers schemas from source code, which is a hypothesis. This tool
turns the hypothesis into an observation by actually reading the files. The
distinction is recorded on `DataSource.profiled` and it matters: a plan built on
an inferred `int64` that is really `object` with nulls produces a Spark job that
fails at hour three of a production run.

Reading input data is safe (it is the user's own data, not generated code), so
this runs in-process. Generated *code* never does — see tools/code_gate.py and
the sandbox runner.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pydantic import Field

from etl_migrator.domain.enums import DataFormat
from etl_migrator.domain.spec import ColumnSpec, StrictModel

_SAMPLE_ROWS = 50_000
_MAX_EXAMPLES = 3


class ColumnProfile(StrictModel):
    name: str
    dtype: str
    null_count: int = Field(ge=0)
    null_fraction: float = Field(ge=0.0, le=1.0)
    distinct_count: int = Field(ge=0)
    min_value: str | None = None
    max_value: str | None = None
    examples: list[str] = Field(default_factory=list)

    def to_column_spec(self) -> ColumnSpec:
        return ColumnSpec(
            name=self.name,
            dtype=self.dtype,
            nullable=self.null_count > 0,
            example_values=self.examples,
        )


class DatasetProfile(StrictModel):
    path: str
    format: DataFormat
    exists: bool
    row_count: int = Field(default=0, ge=0)
    sampled: bool = Field(
        default=False, description="True when row_count/stats come from a capped sample."
    )
    size_bytes: int = Field(default=0, ge=0)
    columns: list[ColumnProfile] = Field(default_factory=list)
    error: str | None = None

    @property
    def broadcast_candidate(self) -> bool:
        """Under Spark's default 10 MB autoBroadcastJoinThreshold."""
        return self.exists and 0 < self.size_bytes < 10 * 1024 * 1024

    def render(self) -> str:
        if not self.exists:
            return f"{self.path}: NOT FOUND ({self.error})"
        head = (
            f"{self.path} [{self.format.value}] rows={self.row_count}"
            f"{' (sampled)' if self.sampled else ''} size={self.size_bytes}B "
            f"broadcast_candidate={self.broadcast_candidate}"
        )
        rows = [
            f"  {c.name}: {c.dtype} nulls={c.null_count} ({c.null_fraction:.1%}) "
            f"distinct={c.distinct_count} range=[{c.min_value}..{c.max_value}] "
            f"examples={c.examples}"
            for c in self.columns
        ]
        return "\n".join([head, *rows])


def _format_of(path: Path) -> DataFormat:
    match path.suffix.lower():
        case ".csv" | ".tsv" | ".txt":
            return DataFormat.CSV
        case ".parquet" | ".pq":
            return DataFormat.PARQUET
        case ".json" | ".jsonl" | ".ndjson":
            return DataFormat.JSON
        case _:
            return DataFormat.UNKNOWN


def _read(path: Path, fmt: DataFormat) -> tuple[pd.DataFrame, bool]:
    """Return (frame, sampled). Large CSVs are capped so profiling stays bounded."""
    match fmt:
        case DataFormat.CSV:
            frame = pd.read_csv(path, nrows=_SAMPLE_ROWS + 1)
            if len(frame) > _SAMPLE_ROWS:
                return frame.head(_SAMPLE_ROWS), True
            return frame, False
        case DataFormat.PARQUET:
            return pd.read_parquet(path), False
        case DataFormat.JSON:
            return pd.read_json(path, lines=path.suffix.lower() in {".jsonl", ".ndjson"}), False
        case _:
            raise ValueError(f"cannot profile format {fmt.value} at {path}")


def _profile_column(series: pd.Series) -> ColumnProfile:
    null_count = int(series.isna().sum())
    total = len(series)
    non_null = series.dropna()
    min_value = max_value = None
    if not non_null.empty:
        try:
            min_value = str(non_null.min())
            max_value = str(non_null.max())
        except TypeError:
            # Mixed-type object column: not orderable. That is itself a finding.
            min_value = max_value = None
    return ColumnProfile(
        name=str(series.name),
        dtype=str(series.dtype),
        null_count=null_count,
        null_fraction=(null_count / total) if total else 0.0,
        distinct_count=int(non_null.nunique()),
        min_value=min_value,
        max_value=max_value,
        examples=[str(v) for v in non_null.head(_MAX_EXAMPLES).tolist()],
    )


def profile_dataset(path: Path) -> DatasetProfile:
    fmt = _format_of(path)
    if not path.is_file():
        return DatasetProfile(path=str(path), format=fmt, exists=False, error="file not found")
    try:
        frame, sampled = _read(path, fmt)
    except Exception as exc:
        return DatasetProfile(path=str(path), format=fmt, exists=False, error=str(exc))

    return DatasetProfile(
        path=str(path),
        format=fmt,
        exists=True,
        row_count=len(frame),
        sampled=sampled,
        size_bytes=path.stat().st_size,
        columns=[_profile_column(frame[c]) for c in frame.columns],
    )


def profile_directory(directory: Path) -> list[DatasetProfile]:
    """Profile every readable dataset in a directory, sorted for determinism."""
    if not directory.is_dir():
        return []
    candidates = sorted(
        p for p in directory.iterdir() if p.is_file() and _format_of(p) is not DataFormat.UNKNOWN
    )
    return [profile_dataset(p) for p in candidates]
