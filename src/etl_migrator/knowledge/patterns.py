"""Structured migration knowledge: the seed of the learning loop.

This is deliberately **not** RAG. It is a typed catalogue keyed by
`TransformKind`, so a lookup is a dictionary access with a guaranteed shape, not
a similarity search that may return something adjacent. The planner agent
queries it as a tool and cites what it found.

`domain/history.py` is the empirical half of this: it harvests the same kind of
evidence from completed migrations and keys it by `(RiskCategory, approach)` for
repairs and by approach slug for optimisations, so the Repair and Optimizer
agents can look up what they are about to propose *before* proposing it. Both
halves share one rule — a rate is reported only above `MIN_SUPPORT`
observations, because a confidence that rises the moment anything is observed
looks like knowledge and an agent will act on it.
"""

from __future__ import annotations

from pydantic import Field

from etl_migrator.domain.enums import RiskCategory, TransformKind
from etl_migrator.domain.spec import StrictModel


class ObservedOutcome(StrictModel):
    """One empirical data point from a completed migration."""

    migration_id: str
    passed_validation: bool
    repair_attempts: int = Field(default=0, ge=0)
    speedup: float | None = None
    note: str | None = None


class MigrationPattern(StrictModel):
    kind: TransformKind
    legacy_construct: str
    spark_construct: str
    guidance: str
    pitfalls: list[str] = Field(default_factory=list)
    risk_categories: list[RiskCategory] = Field(default_factory=list)
    recommended_validation_checks: list[str] = Field(default_factory=list)
    observed_outcomes: list[ObservedOutcome] = Field(default_factory=list)

    @property
    def success_rate(self) -> float | None:
        """Observed pass rate, or None when there is not enough to say.

        The support threshold is `domain.history.MIN_SUPPORT` and it matters:
        without it a single successful migration would report this pattern as
        100% reliable, and the planner would read that as knowledge. One
        observation is an anecdote with a percent sign.
        """
        from etl_migrator.domain.history import MIN_SUPPORT

        if len(self.observed_outcomes) < MIN_SUPPORT:
            return None
        passed = sum(1 for o in self.observed_outcomes if o.passed_validation)
        return passed / len(self.observed_outcomes)

    def render(self) -> str:
        lines = [
            f"{self.legacy_construct}  ->  {self.spark_construct}",
            f"  guidance: {self.guidance}",
        ]
        if self.pitfalls:
            lines += ["  pitfalls:"] + [f"    - {p}" for p in self.pitfalls]
        if self.recommended_validation_checks:
            lines.append(
                f"  validation checks: {', '.join(self.recommended_validation_checks)}"
            )
        rate = self.success_rate
        if rate is None:
            lines.append("  observed outcomes: none recorded yet (curated guidance)")
        else:
            lines.append(
                f"  observed outcomes: {len(self.observed_outcomes)} runs, "
                f"{rate:.0%} passed validation first time"
            )
        return "\n".join(lines)


_SEED: tuple[MigrationPattern, ...] = (
    MigrationPattern(
        kind=TransformKind.FILTER,
        legacy_construct="df[df['col'] > x]  /  df.query(...)  /  SQL WHERE",
        spark_construct="DataFrame.filter(F.col('col') > x)",
        guidance="Direct mapping. Push the filter as early as possible so less data is shuffled.",
        pitfalls=[
            "pandas comparison against NaN yields False and drops the row; Spark comparison "
            "against null yields null, which filter() also drops. Identical outcome for `>`, "
            "but NOT for `!=`: pandas keeps NaN != x as True, Spark drops it. Use "
            "`F.col('c').isNull() | (F.col('c') != x)` when the legacy code relied on that.",
        ],
        risk_categories=[RiskCategory.NULL_SEMANTICS],
        recommended_validation_checks=["row_count", "null_counts"],
    ),
    MigrationPattern(
        kind=TransformKind.DERIVE_COLUMN,
        legacy_construct="df['new'] = df['a'] * df['b']",
        spark_construct="DataFrame.withColumn('new', F.col('a') * F.col('b'))",
        guidance="Use withColumn, or withColumns for several at once to avoid a deep plan.",
        pitfalls=[
            "pandas int64 * int64 stays int64 and overflows silently; Spark LongType "
            "arithmetic returns null on overflow under ANSI mode. Check ranges when profiling.",
            "Chaining dozens of withColumn calls builds a deep logical plan and slows analysis; "
            "prefer a single withColumns(dict).",
        ],
        risk_categories=[RiskCategory.TYPE_COERCION, RiskCategory.FLOATING_POINT],
        recommended_validation_checks=["numeric_tolerance", "schema"],
    ),
    MigrationPattern(
        kind=TransformKind.JOIN,
        legacy_construct="pd.merge(left, right, on=..., how=...)",
        spark_construct="left.join(right, on=..., how=...)",
        guidance=(
            "Map how= directly. Broadcast the smaller side with F.broadcast() when it is under "
            "the autoBroadcastJoinThreshold (default 10 MB) — the profiler reports file size."
        ),
        pitfalls=[
            "pandas merge preserves left-frame row order; Spark join does not. Never compare "
            "outputs positionally — compare on keys or on sorted rows.",
            "Duplicate join keys multiply rows identically in both engines, but the row count "
            "explosion is far more visible at Spark scale. Check duplicate counts on both sides "
            "before the join, not after.",
            "pandas merge on columns with different dtypes raises; Spark silently upcasts "
            "(e.g. int vs string comparison under non-ANSI mode). Assert key dtypes match.",
        ],
        risk_categories=[
            RiskCategory.ROW_ORDER,
            RiskCategory.DUPLICATE_EXPLOSION,
            RiskCategory.TYPE_COERCION,
        ],
        recommended_validation_checks=["row_count", "duplicate_counts", "null_counts"],
    ),
    MigrationPattern(
        kind=TransformKind.AGGREGATE,
        legacy_construct="df.groupby(keys)[col].sum().reset_index()",
        spark_construct="df.groupBy(*keys).agg(F.sum(col).alias(col))",
        guidance=(
            "groupBy + agg with explicit aliases. reset_index() has no Spark equivalent and "
            "needs no translation: Spark aggregates already return flat DataFrames."
        ),
        pitfalls=[
            "pandas groupby drops null keys by default (dropna=True); Spark groupBy keeps null "
            "as its own group. This is the single most common silent row-count difference. "
            "Either filter the null keys explicitly or set dropna=False in the legacy baseline.",
            "pandas sum() of an all-NaN group returns 0.0; Spark sum() of an all-null group "
            "returns null. Wrap with F.coalesce(F.sum(c), F.lit(0)) when the legacy behaviour "
            "is load-bearing.",
            "Aggregate output column names differ: pandas keeps the source name, Spark produces "
            "'sum(col)' unless aliased. Always alias.",
        ],
        risk_categories=[RiskCategory.NULL_SEMANTICS, RiskCategory.ROW_ORDER],
        recommended_validation_checks=["row_count", "null_counts", "numeric_tolerance", "schema"],
    ),
    MigrationPattern(
        kind=TransformKind.WINDOW,
        legacy_construct="df.rolling(n).mean()  /  df.shift()  /  df.cumsum()",
        spark_construct="F.avg(c).over(Window.partitionBy(...).orderBy(...).rowsBetween(...))",
        guidance=(
            "Rolling operations require an explicit Window with both partitionBy and orderBy. "
            "A Window without partitionBy moves the entire dataset into one partition."
        ),
        pitfalls=[
            "pandas rolling is implicitly ordered by the index; Spark has no index, so the "
            "ordering column must be identified explicitly or results are non-deterministic.",
            "rowsBetween(-(n-1), 0) reproduces pandas rolling(n); rangeBetween does not — it is "
            "value-based, not position-based.",
        ],
        risk_categories=[RiskCategory.INDEX_SEMANTICS, RiskCategory.NON_DETERMINISM],
        recommended_validation_checks=["numeric_tolerance", "row_count"],
    ),
    MigrationPattern(
        kind=TransformKind.DEDUPLICATE,
        legacy_construct="df.drop_duplicates(subset=[...], keep='first')",
        spark_construct="df.dropDuplicates([...])  or  row_number() over a Window",
        guidance=(
            "dropDuplicates keeps an arbitrary row, not the first. If keep='first' mattered, "
            "use row_number() over Window.partitionBy(subset).orderBy(<explicit order>) == 1."
        ),
        pitfalls=[
            "keep='first' is meaningless without an ordering column in a distributed engine; "
            "silently mapping it to dropDuplicates yields non-deterministic output.",
        ],
        risk_categories=[RiskCategory.NON_DETERMINISM, RiskCategory.ROW_ORDER],
        recommended_validation_checks=["row_count", "duplicate_counts"],
    ),
    MigrationPattern(
        kind=TransformKind.APPLY_UDF,
        legacy_construct="df['c'].apply(fn)  /  df.apply(fn, axis=1)",
        spark_construct="Built-in pyspark.sql.functions expression, UDF only as a last resort",
        guidance=(
            "Decompose the Python function into built-in column expressions. A Python UDF is "
            "opaque to Catalyst, serialises every row, and typically costs 5-50x."
        ),
        pitfalls=[
            "A UDF returning Python None must declare a nullable return type or Spark fails "
            "at execution, not at analysis.",
            "Python float arithmetic inside a UDF and Spark DoubleType arithmetic can differ in "
            "the last ulp; this shows up as a tolerance failure, not a crash.",
        ],
        risk_categories=[RiskCategory.UDF_SEMANTICS, RiskCategory.FLOATING_POINT],
        recommended_validation_checks=["numeric_tolerance", "null_counts"],
    ),
    MigrationPattern(
        kind=TransformKind.RESET_INDEX,
        legacy_construct="df.reset_index()  /  df.set_index(...)",
        spark_construct="(no equivalent — omit)",
        guidance=(
            "Spark DataFrames have no index. reset_index() after a groupby is pure pandas "
            "bookkeeping and must produce no Spark code. Do not invent a monotonic id column "
            "for it: that would add a column the legacy output does not have."
        ),
        pitfalls=[
            "Emitting monotonically_increasing_id() to 'preserve the index' breaks the schema "
            "comparison and is non-deterministic across runs.",
        ],
        risk_categories=[RiskCategory.INDEX_SEMANTICS],
        recommended_validation_checks=["schema"],
    ),
    MigrationPattern(
        kind=TransformKind.SORT,
        legacy_construct="df.sort_values(by=[...])",
        spark_construct="df.orderBy(...)",
        guidance="Direct mapping, but note that a global orderBy forces a full shuffle.",
        pitfalls=[
            "pandas sort is stable; Spark orderBy is not. Rows with equal sort keys can appear "
            "in any relative order, so add a tiebreaker column if the order is load-bearing.",
            "NULL ordering differs: pandas puts NaN last by default, Spark puts null first for "
            "ascending. Use asc_nulls_last() to match pandas.",
        ],
        risk_categories=[RiskCategory.ROW_ORDER, RiskCategory.NULL_SEMANTICS],
        recommended_validation_checks=["row_count"],
    ),
    MigrationPattern(
        kind=TransformKind.WRITE,
        legacy_construct="df.to_csv(path, index=False)",
        spark_construct="df.write.mode('overwrite').option('header', True).csv(path)",
        guidance=(
            "Spark writes a directory of part files, pandas writes one file. For comparison "
            "purposes that is fine — the differ reads the directory. Do not call coalesce(1) "
            "to force a single file unless the downstream consumer truly requires it."
        ),
        pitfalls=[
            "index=False has no Spark equivalent and needs no translation.",
            "coalesce(1) funnels the whole dataset through one task and can OOM the executor.",
        ],
        risk_categories=[RiskCategory.SCALE],
        recommended_validation_checks=["schema", "row_count"],
    ),
)


class PatternCatalogue:
    """Typed lookup over migration patterns."""

    def __init__(self, patterns: tuple[MigrationPattern, ...] = _SEED) -> None:
        self._patterns = patterns

    def lookup(self, kind: TransformKind) -> list[MigrationPattern]:
        return [p for p in self._patterns if p.kind is kind]

    def render(self, kind: TransformKind) -> str:
        found = self.lookup(kind)
        if not found:
            return (
                f"no recorded pattern for '{kind.value}'. Map it from first principles and "
                f"declare every semantic difference you rely on."
            )
        return "\n\n".join(p.render() for p in found)

    @property
    def kinds(self) -> list[str]:
        return sorted({p.kind.value for p in self._patterns})


DEFAULT_CATALOGUE = PatternCatalogue()
