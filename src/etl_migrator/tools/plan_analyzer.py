"""Deterministic detection of optimisation opportunities in generated Spark code.

This is the Optimizer agent's eyes, the counterpart of `source_inspector` for
the Discovery agent. It contains no LLM and offers no opinions — it reports
structural facts with line numbers, and each fact carries the *reason* it is
worth looking at.

Why static analysis rather than reading Spark's query plan: PySpark does not
reliably expose a physical plan for a pipeline whose entrypoint writes and
returns nothing, and shuffle-byte metrics need a JVM listener. Rather than
estimate those and present the estimates as measurements, this reports what can
actually be established — the code's structure — and `SparkRunMetrics` reports
what Spark actually said. Two honest sources beat one overreaching one.
"""

from __future__ import annotations

import ast

from pydantic import Field

from etl_migrator.domain.spec import StrictModel
from etl_migrator.tools.data_profiler import DatasetProfile

#: Methods that force a full shuffle of the whole dataset.
_WIDE_TRANSFORMS = {"groupBy", "join", "orderBy", "sort", "distinct", "repartition"}


class OptimizationOpportunity(StrictModel):
    """One structural observation worth considering, with its justification."""

    code: str = Field(description="Stable identifier, e.g. 'OPT001'.")
    line: int | None = None
    summary: str
    detail: str = Field(description="Why this matters, in Spark terms.")
    suggested_approach: str = Field(
        description="Slug the optimiser may use as its strategy, or empty when the "
        "observation is informational."
    )

    def render(self) -> str:
        where = f" (line {self.line})" if self.line else ""
        return f"[{self.code}]{where} {self.summary} — {self.detail}"


class PlanAnalysis(StrictModel):
    """Structural view of a generated pipeline, plus measured input sizes."""

    wide_transform_count: int = 0
    opportunities: list[OptimizationOpportunity] = Field(default_factory=list)
    broadcast_hints: list[str] = Field(default_factory=list)
    cached_datasets: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    def render(self) -> str:
        lines = [
            f"wide transformations (each a full shuffle): {self.wide_transform_count}",
            f"broadcast hints present: {self.broadcast_hints or 'none'}",
            f"cache/persist calls: {self.cached_datasets or 'none'}",
            "",
            "opportunities:",
        ]
        lines += [f"  {o.render()}" for o in self.opportunities] or ["  none detected"]
        if self.notes:
            lines += ["", "notes:"] + [f"  {n}" for n in self.notes]
        return "\n".join(lines)


class _Visitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.opportunities: list[OptimizationOpportunity] = []
        self.wide_transforms = 0
        self.broadcast_hints: list[str] = []
        self.cached: list[str] = []
        self.assignments: dict[str, int] = {}
        self.name_reads: dict[str, int] = {}
        self.udf_lines: list[int] = []

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.assignments[target.id] = node.lineno
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            self.name_reads[node.id] = self.name_reads.get(node.id, 0) + 1
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        name = node.func.attr if isinstance(node.func, ast.Attribute) else (
            node.func.id if isinstance(node.func, ast.Name) else ""
        )

        if name in _WIDE_TRANSFORMS:
            self.wide_transforms += 1

        if name == "broadcast":
            self.broadcast_hints.append(ast.unparse(node)[:60])

        if name in {"cache", "persist"}:
            self.cached.append(ast.unparse(node)[:60])

        if name == "coalesce" and node.args and _is_literal_one(node.args[0]):
            self.opportunities.append(
                OptimizationOpportunity(
                    code="OPT001",
                    line=node.lineno,
                    summary="coalesce(1) funnels the entire dataset through one task",
                    detail="Every row is written by a single executor, which serialises the "
                    "write and can exhaust that executor's memory. Only justified when a "
                    "downstream consumer genuinely requires one file.",
                    suggested_approach="remove_single_partition_coalesce",
                )
            )

        if name == "repartition" and node.args and _is_literal_one(node.args[0]):
            self.opportunities.append(
                OptimizationOpportunity(
                    code="OPT002",
                    line=node.lineno,
                    summary="repartition(1) collapses parallelism to a single task",
                    detail="A full shuffle followed by single-threaded execution.",
                    suggested_approach="remove_single_partition_repartition",
                )
            )

        if name in {"udf", "pandas_udf"}:
            self.udf_lines.append(node.lineno)

        if name == "distinct":
            self.opportunities.append(
                OptimizationOpportunity(
                    code="OPT003",
                    line=node.lineno,
                    summary="distinct() is a full shuffle",
                    detail="If the data is already unique on the relevant key, this is a "
                    "shuffle bought for nothing; dropDuplicates on a subset is cheaper "
                    "when only some columns need to be unique.",
                    suggested_approach="narrow_distinct_to_subset",
                )
            )

        self.generic_visit(node)


def _is_literal_one(node: ast.expr) -> bool:
    return isinstance(node, ast.Constant) and node.value == 1


def analyze_plan(
    code: str, profiles: list[DatasetProfile] | None = None
) -> PlanAnalysis:
    """Report structural optimisation opportunities in a generated module."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:  # pragma: no cover - the gate catches this first
        return PlanAnalysis(notes=[f"could not parse the module: {exc}"])

    visitor = _Visitor()
    visitor.visit(tree)
    opportunities = list(visitor.opportunities)

    for line in visitor.udf_lines:
        opportunities.append(
            OptimizationOpportunity(
                code="OPT004",
                line=line,
                summary="Python UDF is opaque to Catalyst",
                detail="Rows are serialised to the Python worker one at a time and the "
                "expression cannot be pushed down or fused. A built-in equivalent is "
                "typically several times faster.",
                suggested_approach="replace_udf_with_builtin",
            )
        )

    # A DataFrame read more than once is recomputed from source each time unless
    # it is cached. Only worth flagging when nothing is cached already.
    if not visitor.cached:
        for name, reads in sorted(visitor.name_reads.items()):
            if name in visitor.assignments and reads >= 3:
                opportunities.append(
                    OptimizationOpportunity(
                        code="OPT005",
                        line=visitor.assignments[name],
                        summary=f"'{name}' is referenced {reads} times and never cached",
                        detail="Spark recomputes a DataFrame from its source on every "
                        "action unless it is cached, so a branch that is consumed twice "
                        "does the upstream work twice.",
                        suggested_approach="cache_reused_dataframe",
                    )
                )

    # Broadcast opportunities are grounded in measured file sizes, never guessed.
    for profile in profiles or []:
        if profile.broadcast_candidate and not visitor.broadcast_hints:
            opportunities.append(
                OptimizationOpportunity(
                    code="OPT006",
                    summary=f"{profile.path} is {profile.size_bytes} bytes and could be broadcast",
                    detail="Below the 10 MB autoBroadcastJoinThreshold. Broadcasting the "
                    "small side of a join removes the shuffle entirely.",
                    suggested_approach="broadcast_small_side",
                )
            )

    notes: list[str] = []
    if visitor.wide_transforms == 0:
        notes.append(
            "no wide transformations detected; this pipeline has no shuffle to optimise "
            "and configuration changes are unlikely to help"
        )
    if visitor.broadcast_hints:
        notes.append(
            "a broadcast hint is already present, so the join shuffle is already avoided"
        )

    return PlanAnalysis(
        wide_transform_count=visitor.wide_transforms,
        opportunities=opportunities,
        broadcast_hints=visitor.broadcast_hints,
        cached_datasets=visitor.cached,
        notes=notes,
    )
