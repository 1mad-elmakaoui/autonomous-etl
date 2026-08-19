"""Planner agent: `MigrationSpec` -> `MigrationPlan`.

The planner exists as a separate agent from the code generator for one reason:
**the mapping decision and the code that implements it must be independently
reviewable.** When a migration later fails validation, the repair agent needs to
know whether the plan was wrong or the implementation was — and it can only tell
those apart if the plan was written down first, in machine-readable form, before
any code existed.

The planner also owns the definition of "correct" for this migration
(`ValidationPlan`). Fixing the acceptance criteria before generating code means
they cannot be quietly relaxed once a diff appears.
"""

from __future__ import annotations

from collections.abc import Callable

from autogen_core.models import ChatCompletionClient

from etl_migrator.agents.base import StructuredAgent
from etl_migrator.domain.enums import TransformKind
from etl_migrator.domain.plan import MigrationPlan
from etl_migrator.domain.spec import MigrationSpec
from etl_migrator.knowledge.patterns import DEFAULT_CATALOGUE, PatternCatalogue
from etl_migrator.tools.data_profiler import DatasetProfile

SYSTEM_MESSAGE = """\
You are a Migration Planner Agent. You receive a structured specification of a
legacy pipeline and decide how each transformation maps onto PySpark. You do not
write code — you decide what the code must do and what could go wrong.

Rules:

1. Produce exactly one plan step per transformation in the specification, in
   order, with ids s1, s2, ... Each step references its `transformation_id`.
2. Call `lookup_migration_pattern` for every transformation kind before you map
   it. It returns curated guidance plus outcomes observed on previous
   migrations. If you deviate from the recorded guidance, say why in `rationale`.
3. `semantic_differences` is the most important field you produce. Every
   difference you declare becomes a mandatory validation check and a generated
   test. Declare a difference when pandas/SQL and Spark could produce different
   results for the same input — null handling in group keys, join row order,
   integer/float coercion, duplicate handling, sort stability, index semantics.
   Each one needs a concrete `mitigation` (what the Spark code will do) and a
   `validation_check` naming the differ check that proves it worked.
   Valid check names: schema, row_count, null_counts, numeric_tolerance,
   duplicate_counts, aggregate_checksums, column_statistics.
4. Call `dataset_sizes` before deciding `broadcast_datasets`. Only broadcast a
   dataset the profiler shows is small. Never broadcast on a guess.
5. Transformations that reorganise a pandas index (reset_index, set_index) map
   to no Spark construct at all. Say so explicitly with spark_construct
   "(none - pandas index bookkeeping)" rather than inventing an equivalent.
6. Set `risk_level` per step and `overall_risk` for the migration. HIGH means a
   human must approve before code is generated, so use it when a semantic
   difference could change business numbers rather than merely row order.
7. `validation_plan.join_keys` should name the columns that identify a row in
   the final output, so the differ can compare records rather than positions.

Respond only with the structured plan.
"""


class PlannerAgent(StructuredAgent[MigrationPlan]):
    key = "planner"
    description = "Maps each legacy transformation onto PySpark and declares semantic risks."

    def __init__(
        self,
        model_client: ChatCompletionClient,
        *,
        spec: MigrationSpec,
        profiles: list[DatasetProfile] | None = None,
        catalogue: PatternCatalogue = DEFAULT_CATALOGUE,
        max_tool_iterations: int = 6,
    ) -> None:
        self.spec = spec
        self.profiles = profiles or []
        self.catalogue = catalogue
        super().__init__(
            model_client,
            MigrationPlan,
            system_message=SYSTEM_MESSAGE,
            tools=[
                self._make_pattern_tool(),
                self._make_transformation_tool(),
                self._make_sizes_tool(),
            ],
            max_tool_iterations=max_tool_iterations,
        )

    def _make_pattern_tool(self) -> Callable[[str], str]:
        catalogue = self.catalogue

        def lookup_migration_pattern(kind: str) -> str:
            """Look up the recorded strategy for a transformation kind: the recommended
            Spark construct, known pitfalls, required validation checks, and how
            previous migrations using this strategy actually turned out.

            Args:
                kind: a transformation kind such as 'join', 'aggregate', 'filter',
                    'window', 'deduplicate', 'apply_udf', 'reset_index', 'sort', 'write'.
            """
            try:
                parsed = TransformKind(kind)
            except ValueError:
                return (
                    f"unknown transformation kind '{kind}'. "
                    f"Known kinds with recorded patterns: {', '.join(catalogue.kinds)}"
                )
            return catalogue.render(parsed)

        return lookup_migration_pattern

    def _make_transformation_tool(self) -> Callable[[str], str]:
        spec = self.spec

        def get_transformation(transformation_id: str) -> str:
            """Return the full detail of one transformation from the specification,
            including join keys, aggregation functions, source line numbers and the
            verbatim legacy expression.

            Args:
                transformation_id: an id such as 't3'.
            """
            try:
                return spec.transformation(transformation_id).model_dump_json(indent=2)
            except KeyError:
                available = ", ".join(t.id for t in spec.transformations)
                return f"no transformation '{transformation_id}'. Available: {available}"

        return get_transformation

    def _make_sizes_tool(self) -> Callable[[], str]:
        profiles = self.profiles
        spec = self.spec

        def dataset_sizes() -> str:
            """Report measured size, row count and broadcast eligibility for every input
            dataset. Use this before deciding which datasets to broadcast or cache."""
            if not profiles:
                declared = "\n".join(
                    f"  {s.name}: {s.path_expression} (row_estimate="
                    f"{s.row_estimate if s.row_estimate is not None else 'unknown'}, "
                    f"profiled={s.profiled})"
                    for s in spec.inputs
                )
                return (
                    "no measured profiles available; only declared sources:\n"
                    + (declared or "  none")
                    + "\nDo not broadcast anything on the basis of this."
                )
            return "\n".join(
                f"{p.path}: rows={p.row_count} size={p.size_bytes}B "
                f"broadcast_eligible={p.broadcast_candidate}"
                for p in profiles
            )

        return dataset_sizes


def planning_task(spec: MigrationSpec) -> str:
    return (
        "Produce the migration plan for the following specification.\n\n"
        f"{spec.model_dump_json(indent=2)}"
    )
