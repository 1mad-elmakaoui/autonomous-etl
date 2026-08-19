"""The output differ: the only thing in this system allowed to say PASS.

No LLM is involved anywhere in this file. Every other component can be
persuaded or mistaken; this one measures two directories of data.

Worth knowing before reading:

* A check that cannot run is not a pass. Unknown check names and missing join
  keys set `skipped=True`, which `ValidationReport.status` maps to ERROR.
* Dtypes are compared by family. `int32` vs `int64` is a storage detail;
  `int64` vs `float64` is the pandas null-promotion trap and must be reported.
* Row order is never assumed. pandas preserves left-frame order through a merge
  and Spark does not, so every comparison is key-joined or sorted first.
* Duplicate keys are detected before they are relied on. Joining on a non-unique
  key multiplies rows, so the record comparison falls back to sorted rows.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from etl_migrator.domain.enums import RiskCategory
from etl_migrator.domain.plan import MigrationPlan, ValidationPlan
from etl_migrator.domain.validation import (
    CheckResult,
    ColumnStat,
    DatasetStats,
    Difference,
    ValidationReport,
)
from etl_migrator.observability import get_logger

log = get_logger(__name__)

MAX_SAMPLE_DIFFERENCES = 10

#: Coarse dtype families. Storage width is noise; family changes are signal.
_FAMILIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("boolean", ("bool",)),
    ("integer", ("int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64")),
    ("float", ("float16", "float32", "float64")),
    ("timestamp", ("datetime64", "datetime64[ns]", "period")),
    ("string", ("object", "string", "category")),
)


def dtype_family(dtype: object) -> str:
    name = str(dtype).lower()
    for family, members in _FAMILIES:
        if name in members or any(name.startswith(m) for m in members):
            return family
    return name


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_dataset(path: Path) -> pd.DataFrame:
    """Read an output, whether it is one file or a directory of part files.

    pandas writes `revenue_by_country.csv`; Spark writes a directory containing
    `part-00000-....csv` plus `_SUCCESS`. Both are legitimate representations of
    the same table, and the differ must not care which it was handed.
    """
    if path.is_file():
        return _read_one(path)

    if not path.is_dir():
        raise FileNotFoundError(f"no output at {path}")

    parts = sorted(
        p
        for p in path.rglob("*")
        if p.is_file() and p.suffix.lower() in {".csv", ".parquet", ".json"}
    )
    if not parts:
        raise FileNotFoundError(f"no readable data files under {path}")

    frames = [_read_one(p) for p in parts]
    frames = [f for f in frames if not f.empty or len(frames) == 1]
    return pd.concat(frames, ignore_index=True) if frames else frames[0]


def _read_one(path: Path) -> pd.DataFrame:
    match path.suffix.lower():
        case ".csv":
            return pd.read_csv(path)
        case ".parquet":
            return pd.read_parquet(path)
        case ".json":
            return pd.read_json(path, lines=True)
        case _:
            raise ValueError(f"cannot read {path}")


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def compute_stats(frame: pd.DataFrame, path: Path, join_keys: list[str]) -> DatasetStats:
    """Measure one side. Both sides go through this identical code path."""
    usable_keys = [k for k in join_keys if k in frame.columns]
    return DatasetStats(
        path=str(path),
        row_count=len(frame),
        columns=[_column_stat(frame[c]) for c in frame.columns],
        duplicate_row_count=int(frame.duplicated().sum()),
        duplicate_key_count=(
            int(frame.duplicated(subset=usable_keys).sum()) if usable_keys else 0
        ),
    )


def _column_stat(series: pd.Series) -> ColumnStat:
    non_null = series.dropna()
    numeric_sum = numeric_min = numeric_max = None
    if pd.api.types.is_numeric_dtype(series) and not non_null.empty:
        numeric_sum = float(non_null.sum())
        numeric_min = float(non_null.min())
        numeric_max = float(non_null.max())
    return ColumnStat(
        name=str(series.name),
        dtype=str(series.dtype),
        null_count=int(series.isna().sum()),
        distinct_count=int(non_null.nunique()),
        numeric_sum=numeric_sum,
        numeric_min=numeric_min,
        numeric_max=numeric_max,
    )


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------
def check_schema(ref: pd.DataFrame, cand: pd.DataFrame) -> CheckResult:
    differences: list[Difference] = []
    ref_cols, cand_cols = list(ref.columns), list(cand.columns)

    for missing in [c for c in ref_cols if c not in cand_cols]:
        differences.append(
            Difference(
                check="schema",
                column=missing,
                reference="present",
                candidate="absent",
                detail="column produced by the legacy pipeline is missing from the Spark output",
            )
        )
    for extra in [c for c in cand_cols if c not in ref_cols]:
        differences.append(
            Difference(
                check="schema",
                column=extra,
                category=RiskCategory.INDEX_SEMANTICS,
                reference="absent",
                candidate="present",
                detail="Spark output has a column the reference does not — a synthesised "
                "index column is the usual cause",
            )
        )

    for column in [c for c in ref_cols if c in cand_cols]:
        ref_family = dtype_family(ref[column].dtype)
        cand_family = dtype_family(cand[column].dtype)
        if ref_family != cand_family:
            differences.append(
                Difference(
                    check="schema",
                    column=column,
                    category=RiskCategory.TYPE_COERCION,
                    reference=f"{ref[column].dtype} ({ref_family})",
                    candidate=f"{cand[column].dtype} ({cand_family})",
                    detail="dtype family differs; an integer column that gained nulls is "
                    "promoted to float by pandas but not by Spark",
                )
            )

    if not differences and ref_cols != cand_cols:
        differences.append(
            Difference(
                check="schema",
                reference=str(ref_cols),
                candidate=str(cand_cols),
                detail="same columns in a different order",
            )
        )

    return CheckResult(
        name="schema",
        passed=not differences,
        detail=f"{len(ref_cols)} reference columns, {len(cand_cols)} candidate columns",
        differences=differences,
    )


def check_row_count(ref: pd.DataFrame, cand: pd.DataFrame) -> CheckResult:
    equal = len(ref) == len(cand)
    differences: list[Difference] = []
    if not equal:
        differences.append(
            Difference(
                check="row_count",
                category=RiskCategory.NULL_SEMANTICS,
                reference=str(len(ref)),
                candidate=str(len(cand)),
                detail=(
                    "row counts differ by "
                    f"{len(cand) - len(ref):+d}. A group key containing nulls is the most "
                    "common cause: pandas groupby drops them, Spark groupBy keeps them"
                ),
            )
        )
    return CheckResult(
        name="row_count",
        passed=equal,
        detail=f"reference={len(ref)} candidate={len(cand)}",
        differences=differences,
    )


def check_null_counts(ref: pd.DataFrame, cand: pd.DataFrame) -> CheckResult:
    differences: list[Difference] = []
    for column in [c for c in ref.columns if c in cand.columns]:
        ref_nulls = int(ref[column].isna().sum())
        cand_nulls = int(cand[column].isna().sum())
        if ref_nulls != cand_nulls:
            differences.append(
                Difference(
                    check="null_counts",
                    column=column,
                    category=RiskCategory.NULL_SEMANTICS,
                    reference=str(ref_nulls),
                    candidate=str(cand_nulls),
                    detail="null counts differ; pandas sum() of an all-NaN group returns 0.0 "
                    "while Spark sum() returns null",
                )
            )
    return CheckResult(
        name="null_counts",
        passed=not differences,
        detail=f"compared {len(set(ref.columns) & set(cand.columns))} shared columns",
        differences=differences,
    )


def check_duplicate_counts(ref: pd.DataFrame, cand: pd.DataFrame, keys: list[str]) -> CheckResult:
    differences: list[Difference] = []
    ref_dupes, cand_dupes = int(ref.duplicated().sum()), int(cand.duplicated().sum())
    if ref_dupes != cand_dupes:
        differences.append(
            Difference(
                check="duplicate_counts",
                category=RiskCategory.DUPLICATE_EXPLOSION,
                reference=str(ref_dupes),
                candidate=str(cand_dupes),
                detail="duplicate full-row counts differ; a join key that is non-unique on "
                "one side multiplies rows",
            )
        )

    usable = [k for k in keys if k in ref.columns and k in cand.columns]
    if usable:
        ref_key_dupes = int(ref.duplicated(subset=usable).sum())
        cand_key_dupes = int(cand.duplicated(subset=usable).sum())
        if ref_key_dupes != cand_key_dupes:
            differences.append(
                Difference(
                    check="duplicate_counts",
                    column=",".join(usable),
                    category=RiskCategory.DUPLICATE_EXPLOSION,
                    reference=str(ref_key_dupes),
                    candidate=str(cand_key_dupes),
                    detail="duplicate counts on the declared key columns differ",
                )
            )

    return CheckResult(
        name="duplicate_counts",
        passed=not differences,
        detail=f"reference={ref_dupes} candidate={cand_dupes} duplicate rows",
        differences=differences,
    )


def check_aggregate_checksums(
    ref: pd.DataFrame, cand: pd.DataFrame, tolerance: float
) -> CheckResult:
    """Order-independent totals: numeric sums plus a hash over sorted string values.

    Catches the case where every row-level check passes on the intersection but
    the two sides hold different data overall.
    """
    differences: list[Difference] = []
    for column in [c for c in ref.columns if c in cand.columns]:
        ref_series, cand_series = ref[column], cand[column]
        if pd.api.types.is_numeric_dtype(ref_series) and pd.api.types.is_numeric_dtype(
            cand_series
        ):
            ref_sum = float(ref_series.dropna().sum())
            cand_sum = float(cand_series.dropna().sum())
            if not _within_tolerance(ref_sum, cand_sum, tolerance):
                differences.append(
                    Difference(
                        check="aggregate_checksums",
                        column=column,
                        category=RiskCategory.FLOATING_POINT,
                        reference=f"{ref_sum:.10g}",
                        candidate=f"{cand_sum:.10g}",
                        detail=f"column totals differ beyond a relative tolerance of {tolerance:g}",
                    )
                )
        else:
            ref_hash = _value_hash(ref_series)
            cand_hash = _value_hash(cand_series)
            if ref_hash != cand_hash:
                differences.append(
                    Difference(
                        check="aggregate_checksums",
                        column=column,
                        reference=ref_hash[:16],
                        candidate=cand_hash[:16],
                        detail="order-independent checksum of the column's values differs",
                    )
                )
    return CheckResult(
        name="aggregate_checksums",
        passed=not differences,
        detail=f"checksummed {len(set(ref.columns) & set(cand.columns))} shared columns",
        differences=differences,
    )


def check_column_statistics(ref: pd.DataFrame, cand: pd.DataFrame, tolerance: float) -> CheckResult:
    differences: list[Difference] = []
    for column in [c for c in ref.columns if c in cand.columns]:
        ref_series, cand_series = ref[column].dropna(), cand[column].dropna()
        if not (
            pd.api.types.is_numeric_dtype(ref[column])
            and pd.api.types.is_numeric_dtype(cand[column])
        ):
            ref_distinct, cand_distinct = ref_series.nunique(), cand_series.nunique()
            if ref_distinct != cand_distinct:
                differences.append(
                    Difference(
                        check="column_statistics",
                        column=column,
                        reference=str(ref_distinct),
                        candidate=str(cand_distinct),
                        detail="distinct value counts differ",
                    )
                )
            continue
        if ref_series.empty or cand_series.empty:
            continue
        for label, ref_value, cand_value in (
            ("min", float(ref_series.min()), float(cand_series.min())),
            ("max", float(ref_series.max()), float(cand_series.max())),
            ("mean", float(ref_series.mean()), float(cand_series.mean())),
        ):
            if not _within_tolerance(ref_value, cand_value, tolerance):
                differences.append(
                    Difference(
                        check="column_statistics",
                        column=column,
                        category=RiskCategory.FLOATING_POINT,
                        reference=f"{label}={ref_value:.10g}",
                        candidate=f"{label}={cand_value:.10g}",
                        detail=f"{label} differs beyond tolerance",
                    )
                )
    return CheckResult(
        name="column_statistics",
        passed=not differences,
        detail="compared min/max/mean for numeric columns, cardinality otherwise",
        differences=differences,
    )


def check_numeric_tolerance(
    ref: pd.DataFrame, cand: pd.DataFrame, plan: ValidationPlan
) -> CheckResult:
    """Record-level comparison, keyed where possible.

    Falls back to a sorted positional comparison when there are no usable keys
    or when the keys are not unique — and says which strategy it used, because
    the two have genuinely different strength.
    """
    tolerance = plan.numeric_tolerance
    keys = [k for k in plan.join_keys if k in ref.columns and k in cand.columns]

    if plan.join_keys and not keys:
        return CheckResult(
            name="numeric_tolerance",
            passed=False,
            skipped=True,
            detail=(
                f"declared join keys {plan.join_keys} are absent from the outputs "
                f"(reference has {list(ref.columns)}); cannot compare records"
            ),
        )

    keys_unique = bool(keys) and not (
        ref.duplicated(subset=keys).any() or cand.duplicated(subset=keys).any()
    )

    if keys_unique:
        return _compare_on_keys(ref, cand, keys, tolerance)
    strategy = "sorted rows (no usable keys)" if not keys else "sorted rows (keys not unique)"
    return _compare_sorted(ref, cand, tolerance, strategy)


def _compare_on_keys(
    ref: pd.DataFrame, cand: pd.DataFrame, keys: list[str], tolerance: float
) -> CheckResult:
    merged = ref.merge(cand, on=keys, how="outer", suffixes=("__ref", "__cand"), indicator=True)
    differences: list[Difference] = []

    for side, label in (("left_only", "reference"), ("right_only", "candidate")):
        missing = merged[merged["_merge"] == side]
        for _, row in missing.head(MAX_SAMPLE_DIFFERENCES).iterrows():
            key_repr = ", ".join(f"{k}={row[k]!r}" for k in keys)
            differences.append(
                Difference(
                    check="numeric_tolerance",
                    reference=key_repr if label == "reference" else "absent",
                    candidate="absent" if label == "reference" else key_repr,
                    detail=f"row present only in the {label} output",
                )
            )

    both = merged[merged["_merge"] == "both"]
    shared = [c for c in ref.columns if c not in keys and c in cand.columns]
    for column in shared:
        left, right = both[f"{column}__ref"], both[f"{column}__cand"]
        if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
            mismatched = ~_series_within_tolerance(left, right, tolerance)
        else:
            mismatched = ~((left == right) | (left.isna() & right.isna()))
        count = int(mismatched.sum())
        if not count:
            continue
        for _, row in both[mismatched].head(MAX_SAMPLE_DIFFERENCES).iterrows():
            differences.append(
                Difference(
                    check="numeric_tolerance",
                    column=column,
                    category=RiskCategory.FLOATING_POINT,
                    reference=f"{row[f'{column}__ref']!r}",
                    candidate=f"{row[f'{column}__cand']!r}",
                    detail=f"at {', '.join(f'{k}={row[k]!r}' for k in keys)} "
                    f"({count} row(s) differ in this column)",
                )
            )
    return CheckResult(
        name="numeric_tolerance",
        passed=not differences,
        detail=f"compared {len(both)} records keyed on {keys} at tolerance {tolerance:g}",
        differences=differences[: MAX_SAMPLE_DIFFERENCES * 2],
    )


def _compare_sorted(
    ref: pd.DataFrame, cand: pd.DataFrame, tolerance: float, strategy: str
) -> CheckResult:
    shared = [c for c in ref.columns if c in cand.columns]
    if len(ref) != len(cand):
        return CheckResult(
            name="numeric_tolerance",
            passed=False,
            detail=f"{strategy}: row counts differ, records cannot be aligned",
            differences=[
                Difference(
                    check="numeric_tolerance",
                    reference=str(len(ref)),
                    candidate=str(len(cand)),
                    detail="cannot compare records when row counts differ and no unique key "
                    "is available",
                )
            ],
        )

    left = ref[shared].sort_values(by=shared).reset_index(drop=True)
    right = cand[shared].sort_values(by=shared).reset_index(drop=True)
    differences: list[Difference] = []

    for column in shared:
        if pd.api.types.is_numeric_dtype(left[column]) and pd.api.types.is_numeric_dtype(
            right[column]
        ):
            mismatched = ~_series_within_tolerance(left[column], right[column], tolerance)
        else:
            mismatched = ~(
                (left[column] == right[column])
                | (left[column].isna() & right[column].isna())
            )
        positions = list(mismatched[mismatched].index[:MAX_SAMPLE_DIFFERENCES])
        for position in positions:
            differences.append(
                Difference(
                    check="numeric_tolerance",
                    column=column,
                    reference=f"{left[column].iloc[position]!r}",
                    candidate=f"{right[column].iloc[position]!r}",
                    detail=f"{strategy}: differs at sorted position {position}",
                )
            )
    return CheckResult(
        name="numeric_tolerance",
        passed=not differences,
        detail=f"{strategy}: compared {len(left)} rows at tolerance {tolerance:g}",
        differences=differences,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _within_tolerance(left: float, right: float, tolerance: float) -> bool:
    if pd.isna(left) and pd.isna(right):
        return True
    if pd.isna(left) or pd.isna(right):
        return False
    if left == right:
        return True
    scale = max(abs(left), abs(right), 1e-12)
    return abs(left - right) / scale <= tolerance


def _series_within_tolerance(
    left: pd.Series, right: pd.Series, tolerance: float
) -> pd.Series:
    both_null = left.isna() & right.isna()
    one_null = left.isna() ^ right.isna()
    scale = pd.concat([left.abs(), right.abs()], axis=1).max(axis=1).clip(lower=1e-12)
    relative = (left - right).abs() / scale
    return (both_null | (relative <= tolerance)) & ~one_null


def _value_hash(series: pd.Series) -> str:
    """Order-independent hash of a column's values."""
    values = sorted(str(v) for v in series.dropna().tolist())
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\x00")
    digest.update(f"|nulls={int(series.isna().sum())}".encode())
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def compare_outputs(
    *,
    migration_id: str,
    reference_path: Path,
    candidate_path: Path,
    plan: MigrationPlan | None = None,
    required_checks: list[str] | None = None,
    validation_plan: ValidationPlan | None = None,
) -> ValidationReport:
    """Run every required check and return the verdict.

    The checks come from the plan, which was written before any code existed —
    so the acceptance criteria cannot be relaxed after an inconvenient diff
    appears.
    """
    vplan = validation_plan or (plan.validation_plan if plan else ValidationPlan())
    checks = required_checks or (
        plan.effective_required_checks() if plan else list(vplan.required_checks)
    )

    try:
        reference = load_dataset(reference_path)
        candidate = load_dataset(candidate_path)
    except (FileNotFoundError, ValueError) as exc:
        return ValidationReport(migration_id=migration_id, error=str(exc))

    keys = list(vplan.join_keys)
    report = ValidationReport(
        migration_id=migration_id,
        reference=compute_stats(reference, reference_path, keys),
        candidate=compute_stats(candidate, candidate_path, keys),
    )

    runners = {
        "schema": lambda: check_schema(reference, candidate),
        "row_count": lambda: check_row_count(reference, candidate),
        "null_counts": lambda: check_null_counts(reference, candidate),
        "duplicate_counts": lambda: check_duplicate_counts(reference, candidate, keys),
        "aggregate_checksums": lambda: check_aggregate_checksums(
            reference, candidate, vplan.numeric_tolerance
        ),
        "column_statistics": lambda: check_column_statistics(
            reference, candidate, vplan.numeric_tolerance
        ),
        "numeric_tolerance": lambda: check_numeric_tolerance(reference, candidate, vplan),
    }

    for name in checks:
        runner = runners.get(name)
        if runner is None:
            # An unknown check name is a contract violation, not something to
            # shrug off — it means the plan demanded an assurance we cannot give.
            report.checks.append(
                CheckResult(
                    name=name,
                    passed=False,
                    skipped=True,
                    detail=f"no differ implements '{name}'; known checks: "
                    f"{', '.join(sorted(runners))}",
                )
            )
            continue
        report.checks.append(runner())

    log.info(
        "validation.complete",
        migration_id=migration_id,
        status=report.status.value,
        checks=len(report.checks),
        differences=len(report.differences),
    )
    return report
