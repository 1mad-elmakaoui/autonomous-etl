"""Discovery agent: turns a legacy source file into a `MigrationSpec`.

The agent is *not* given the source text in its prompt. It is given tools and
must call them. That is deliberate: an agent handed 200 lines of pandas will
confidently describe columns that do not exist, whereas an agent that has to
call `inspect_legacy_source()` gets line-numbered facts and `profile_input_data()`
gets real dtypes and null counts. The difference between the two designs is the
difference between a demo and a system.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from autogen_core.models import ChatCompletionClient

from etl_migrator.agents.base import StructuredAgent
from etl_migrator.domain.spec import MigrationSpec
from etl_migrator.tools.data_profiler import profile_dataset, profile_directory
from etl_migrator.tools.source_inspector import SourceInspection, inspect_source

SYSTEM_MESSAGE = """\
You are a Discovery Agent in an automated ETL migration system. You analyse a
legacy data pipeline and produce a precise, structured specification of what it
does. A downstream agent will write PySpark from your specification alone and
will never see the original source, so anything you omit is lost.

Rules you must follow:

1. Call `inspect_legacy_source` first. It returns line-numbered facts extracted
   from the AST. Never describe an operation that does not appear there.
2. Call `profile_input_data` for every input dataset before you state a schema.
   Set `profiled=true` on a DataSource only when the profiler actually read it.
   If you did not profile it, set `profiled=false` and add an assumption.
3. Use `read_source_lines` when you need the verbatim text of a region to
   understand a business rule.
4. Transformation ids are t1, t2, ... in execution order. Every transformation
   must name its `output` dataset and list its `inputs`, so the chain from
   source to sink is unbroken.
5. Business rules are stated in domain language ("only adults are counted"),
   not code language ("filter age >= 18"), and reference the transformation ids
   that implement them.
6. Risks must be specific and actionable. "pandas and Spark differ" is useless.
   "Left join on customer_id will produce nulls in revenue for customers with no
   orders; pandas sum() skips NaN while Spark sum() skips null, so totals agree,
   but count() does not" is useful.
7. `confidence` reflects how much you verified with tools, not how fluent your
   answer reads.

Respond only with the structured specification.
"""


def _preview(text: str, limit: int = 6000) -> str:
    return text if len(text) <= limit else text[:limit] + f"\n... [truncated, {len(text)} chars]"


class DiscoveryAgent(StructuredAgent[MigrationSpec]):
    key = "discovery"
    description = "Analyses legacy ETL source and produces a structured migration specification."

    def __init__(
        self,
        model_client: ChatCompletionClient,
        *,
        source_path: Path,
        input_dir: Path,
        max_tool_iterations: int = 6,
    ) -> None:
        self.source_path = source_path
        self.input_dir = input_dir
        super().__init__(
            model_client,
            MigrationSpec,
            system_message=SYSTEM_MESSAGE,
            tools=[
                self._make_inspect_tool(),
                self._make_profile_tool(),
                self._make_read_lines_tool(),
            ],
            max_tool_iterations=max_tool_iterations,
        )

    # -- tools -------------------------------------------------------------
    def _make_inspect_tool(self) -> Callable[[], str]:
        source_path = self.source_path

        def inspect_legacy_source() -> str:
            """Parse the legacy pipeline and return line-numbered structural facts:
            imports, reads, writes, boolean filters, derived columns, joins,
            aggregations and referenced column names. Call this before anything else."""
            inspection: SourceInspection = inspect_source(source_path)
            return inspection.render()

        return inspect_legacy_source

    def _make_profile_tool(self) -> Callable[[str], str]:
        input_dir = self.input_dir

        def profile_input_data(dataset: str = "") -> str:
            """Read real input data and report per-column dtype, null count, distinct
            count, min/max, example values, row count and file size.

            Args:
                dataset: file name inside the input directory (e.g. 'customers.csv').
                    Leave empty to profile every dataset in the directory.
            """
            if not dataset:
                profiles = profile_directory(input_dir)
                if not profiles:
                    return f"no readable datasets found in {input_dir}"
                return "\n\n".join(p.render() for p in profiles)

            # Path traversal guard: generated/agent-supplied names never escape input_dir.
            candidate = (input_dir / dataset).resolve()
            if not candidate.is_relative_to(input_dir.resolve()):
                return f"refused: '{dataset}' resolves outside the input directory"
            return profile_dataset(candidate).render()

        return profile_input_data

    def _make_read_lines_tool(self) -> Callable[[int, int], str]:
        source_path = self.source_path

        def read_source_lines(start: int, end: int) -> str:
            """Return verbatim legacy source for lines [start, end] (1-based, inclusive).

            Args:
                start: first line number, 1-based.
                end: last line number, inclusive.
            """
            lines = source_path.read_text(encoding="utf-8").splitlines()
            lo = max(1, start)
            hi = min(len(lines), max(lo, end))
            body = "\n".join(f"{n:4d}| {lines[n - 1]}" for n in range(lo, hi + 1))
            return _preview(body)

        return read_source_lines


def discovery_task(source_path: Path, input_dir: Path) -> str:
    return (
        f"Analyse the legacy ETL pipeline at `{source_path}`. Its input datasets live in "
        f"`{input_dir}`. Inspect the source, profile every input dataset, and produce the "
        f"migration specification."
    )
