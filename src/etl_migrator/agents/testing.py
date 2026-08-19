"""Testing agent: writes a pytest suite for the generated pipeline.

The suite it produces is a genuinely different instrument from the differ. The
differ compares two *whole outputs* after both pipelines have run end to end;
the tests pin *individual behaviours* on small, hand-built frames — an empty
group, an all-null column, a duplicated key. Those are the cases the sample data
may never contain, and they are what turns a passing migration into one you
would let near production.

Like the Spark Engineer, this agent validates its own output through the same
gate the orchestrator re-runs afterwards, so a syntactically broken or unsafe
test module is caught inside the agent's loop rather than at execution time.
"""

from __future__ import annotations

from collections.abc import Callable

from autogen_core.models import ChatCompletionClient

from etl_migrator.agents.base import StructuredAgent
from etl_migrator.domain.plan import MigrationPlan
from etl_migrator.domain.spec import MigrationSpec
from etl_migrator.domain.validation import GeneratedTests
from etl_migrator.tools.code_gate import GateOptions, analyze_generated_code

SYSTEM_MESSAGE = """\
You are a Testing Agent. You write a pytest module that exercises a generated
PySpark pipeline, with particular attention to the semantic differences the
migration plan declared.

Fixtures the harness provides — use these, never build your own SparkSession:

    spark        an active SparkSession, configured exactly as the pipeline runs
    pipeline     the generated module, already imported (call pipeline.run(...))
    input_dir    path to the real input data as a string

Hard contract, enforced by a static gate you can call:

* Import only from: pytest, pyspark, and the standard library modules typing,
  datetime, decimal, math, collections, functools, itertools, dataclasses, enum.
* Never import os, sys, subprocess, pandas, or anything that touches the host.
* No work at module level. Only imports, constants and function definitions —
  pytest imports the module to collect it.
* Every test function name starts with `test_`.
* You may call .collect() here: test frames are tiny and collecting three rows
  to assert on them is correct.

What to test, in rough priority order:

1. One test per declared semantic difference. If the plan says null group keys
   are dropped to match pandas, build a frame containing a null key and assert
   the output excludes it. These are the highest-value tests you can write.
2. Aggregation correctness on a small hand-built frame with known totals.
3. Null handling: an all-null column in an aggregate, nulls in a join key.
4. Edge cases: an empty input frame, a single row, duplicated join keys.
5. Schema: the output column names and types the specification declares.

Rules that make the tests worth having:

* Assert on specific values, not just that something ran. `assert result == 42`
  beats `assert result is not None`.
* Build test frames with spark.createDataFrame and an explicit schema, so a
  type change is caught rather than inferred away.
* Never assert on row order unless the pipeline sorts. Collect and sort, or
  compare sets.
* A test that cannot fail is worse than no test. Do not write one.

Process: draft the module, call `check_test_code` on the complete source, fix
every ERROR finding, and only return once the gate passes. `covers_checks` must
list the validation checks your tests exercise, using the same names the plan
uses.
"""


class TestingAgent(StructuredAgent[GeneratedTests]):
    key = "testing"
    description = "Writes a pytest suite pinning the semantic differences the plan declared."

    def __init__(
        self,
        model_client: ChatCompletionClient,
        *,
        spec: MigrationSpec,
        plan: MigrationPlan,
        code_content: str,
        max_tool_iterations: int = 8,
    ) -> None:
        self.spec = spec
        self.plan = plan
        self.code_content = code_content
        self.gate_options = GateOptions.for_tests()
        self.gate_calls = 0
        super().__init__(
            model_client,
            GeneratedTests,
            system_message=SYSTEM_MESSAGE,
            tools=[self._make_gate_tool(), self._make_source_tool()],
            max_tool_iterations=max_tool_iterations,
        )

    def _make_gate_tool(self) -> Callable[[str], str]:
        options = self.gate_options
        agent = self

        def check_test_code(code: str) -> str:
            """Run the static gate on a candidate pytest module and return every
            finding with severity, code and line number.

            Args:
                code: the complete test module source to validate.
            """
            agent.gate_calls += 1
            report = analyze_generated_code(code, options)
            verdict = "PASS" if report.passed else "FAIL"
            return f"gate: {verdict} (submission #{agent.gate_calls})\n{report.render()}"

        return check_test_code

    def _make_source_tool(self) -> Callable[[], str]:
        content = self.code_content

        def read_generated_pipeline() -> str:
            """Return the full source of the generated PySpark module under test, so
            tests assert on what the code actually does rather than on what the plan
            said it would do."""
            return content

        return read_generated_pipeline


def testing_task(spec: MigrationSpec, plan: MigrationPlan, filename: str) -> str:
    differences = "\n".join(
        f"- [{d.category.value}] {d.description}\n  mitigation: {d.mitigation}\n"
        f"  proves it worked: {d.validation_check}"
        for d in plan.all_semantic_differences
    )
    return (
        f"Write the pytest module `{filename}` for the generated pipeline.\n\n"
        f"Read the generated source first.\n\n"
        f"=== SEMANTIC DIFFERENCES THAT MUST EACH GET A TEST ===\n{differences}\n\n"
        f"=== SPECIFICATION ===\n{spec.model_dump_json(indent=2)}\n\n"
        f"=== PLAN ===\n{plan.model_dump_json(indent=2)}"
    )
