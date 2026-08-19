"""Repair agent: turns a measured failure into a different piece of code.

This agent is handed the most information of any in the system — the differ's
report, the Validation agent's diagnosis, the current source, the plan, and
every previous attempt — and it has the narrowest remit: produce code that is
*meaningfully different* and that addresses the stated root cause.

Two things constrain it, and only one of them is a prompt:

* The prompt tells it not to repeat a strategy and to name its approach with a
  slug.
* `RepairLedger` enforces that, deterministically, before any Spark job runs.
  A repeated signature or byte-equivalent code is rejected without an LLM call
  or an execution, and the rejection is fed into the next attempt.

The second is what makes the first true. "Don't repeat yourself" is advice a
model can forget; a set membership test cannot.
"""

from __future__ import annotations

import json
from collections.abc import Callable

from autogen_core.models import ChatCompletionClient

from etl_migrator.agents.base import StructuredAgent
from etl_migrator.domain.artifacts import GeneratedCode
from etl_migrator.domain.history import MigrationHistory
from etl_migrator.domain.plan import MigrationPlan
from etl_migrator.domain.repair import (
    RepairAttempt,
    RepairProposal,
    build_diagnosis_summary,
    summarise_history,
)
from etl_migrator.domain.validation import ValidationDiagnosis, ValidationReport
from etl_migrator.tools.code_gate import (
    REQUIRED_ENTRYPOINT,
    REQUIRED_PARAMETERS,
    GateOptions,
    analyze_generated_code,
)

SYSTEM_MESSAGE = f"""\
You are a Repair Agent. A generated PySpark pipeline executed successfully but
produced output that does not match the legacy reference. A deterministic differ
measured the disagreement and a diagnosis names the likely cause. Your job is to
produce a corrected version of the module.

What you are not doing: arguing with the differ, widening a tolerance, or
explaining why the difference is acceptable. The difference is real and the
migration is wrong until the numbers agree.

Rules:

1. Call `get_failure` first, then `get_current_code`. Read `previous_attempts`
   before proposing anything — strategies listed there have already been spent
   and will be rejected without being executed.
2. Choose one root cause and fix it. If the report shows five differences with
   one mechanism behind them, fix the mechanism. Scattering unrelated changes
   across a module makes the next failure impossible to attribute.
3. `strategy.approach` is a slug naming the *technique*, not the symptom:
   `filter_null_group_keys`, `coalesce_null_aggregate`, `declare_explicit_schema`,
   `dedupe_join_keys_with_window`, `cast_key_columns`. Two attempts with the same
   slug are the same idea and the second is rejected.
4. Return the **complete** module, not a patch. It must still satisfy the
   contract: exactly `def {REQUIRED_ENTRYPOINT}({", ".join(REQUIRED_PARAMETERS)})`,
   no SparkSession construction, no work at import time, no `.collect()` or
   `.toPandas()`, and imports only from the permitted roots.
5. Preserve every mitigation the code already implements correctly. A repair
   that fixes one difference by removing an unrelated correct behaviour trades
   one failure for another.
6. Call `check_spark_code` on the full source and fix every ERROR before you
   return. Code that fails the static gate never reaches validation, so a
   careless submission simply wastes an attempt from a small budget.
7. `expected_effect` states what the differ should show afterwards, in terms of
   the specific checks that are currently failing. It is compared against the
   actual outcome, so an over-optimistic claim becomes visible.

Common mechanisms and their fixes:

* extra output row with a null group key -> `filter_null_group_keys`: apply
  `.filter(F.col(key).isNotNull())` before `.groupBy(key)` to reproduce
  pandas `groupby(dropna=True)`
* null where the reference has 0.0 -> `coalesce_null_aggregate`: wrap the
  aggregate as `F.coalesce(F.sum(c), F.lit(0.0))`
* an integer column arriving as float, or vice versa ->
  `declare_explicit_schema`: read with an explicit StructType instead of
  `inferSchema`
* an extra column -> `drop_synthesised_index`: emit nothing for `reset_index`
* multiplied row counts -> `dedupe_join_keys_with_window`: deduplicate the
  non-unique side with `row_number()` over an explicit Window
* aggregate column named `sum(x)` -> `alias_aggregates`: alias every aggregate

Before settling on an approach, call `strategy_track_record` for it. That is the
record of how the same strategy fared on *other* migrations, and it is worth a
tool call: an approach that has failed the last four times it was tried on this
category is unlikely to be the one. If it reports too little evidence, that is
an honest answer and not a reason to hesitate — proceed on the diagnosis.
"""


class RepairAgent(StructuredAgent[RepairProposal]):
    key = "repair"
    description = "Rewrites generated PySpark to fix a measured validation failure."

    def __init__(
        self,
        model_client: ChatCompletionClient,
        *,
        report: ValidationReport,
        diagnosis: ValidationDiagnosis | None,
        plan: MigrationPlan,
        code: GeneratedCode,
        history: list[RepairAttempt] | None = None,
        past: MigrationHistory | None = None,
        gate_options: GateOptions | None = None,
        max_tool_iterations: int = 8,
    ) -> None:
        self.report = report
        self.diagnosis = diagnosis
        self.plan = plan
        self.code = code
        self.history = history or []
        self.past = past or MigrationHistory()
        self.gate_options = gate_options or GateOptions()
        self.gate_calls = 0
        super().__init__(
            model_client,
            RepairProposal,
            system_message=SYSTEM_MESSAGE,
            tools=[
                self._make_failure_tool(),
                self._make_code_tool(),
                self._make_history_tool(),
                self._make_track_record_tool(),
                self._make_gate_tool(),
                self._make_plan_step_tool(),
            ],
            max_tool_iterations=max_tool_iterations,
        )

    def _make_failure_tool(self) -> Callable[[], str]:
        report, diagnosis = self.report, self.diagnosis

        def get_failure() -> str:
            """Return the differ's measured report and the diagnosis of its cause:
            every failing check, the reference and candidate values, and the plan
            step believed responsible."""
            return build_diagnosis_summary(report, diagnosis)

        return get_failure

    def _make_code_tool(self) -> Callable[[], str]:
        content = self.code.content

        def get_current_code() -> str:
            """Return the full source of the pipeline that failed validation. Your
            repair must be a complete replacement for this module."""
            return content

        return get_current_code

    def _make_history_tool(self) -> Callable[[], str]:
        history = self.history

        def previous_attempts() -> str:
            """Return every repair already attempted, its strategy, and how it fared.

            Strategies listed here are spent: proposing one again is rejected
            before execution, wasting an attempt from a small budget.
            """
            return summarise_history(history)

        return previous_attempts

    def _make_track_record_tool(self) -> Callable[[str, str], str]:
        past = self.past

        def strategy_track_record(category: str, approach: str) -> str:
            """Return how a repair strategy has fared on *previous* migrations.

            Distinct from `previous_attempts`, which covers only this migration.
            This is the corpus: whether an approach has fixed this class of
            failure before, and how often.

            An unseen strategy is reported as unseen, and a strategy with fewer
            than three attempts is reported without a rate — two data points are
            not a success rate, and a number would invite you to act as though
            they were.

            Args:
                category: the root-cause class, e.g. 'null_semantics'.
                approach: the strategy slug you are considering.
            """
            evidence = past.repair_evidence(category, approach)
            if evidence.attempts == 0:
                return (
                    f"{category}/{approach}: never tried on a recorded migration. "
                    "That is not a reason to avoid it — only a reason not to "
                    "expect anything in particular."
                )
            return evidence.render()

        return strategy_track_record

    def _make_gate_tool(self) -> Callable[[str], str]:
        options = self.gate_options
        agent = self

        def check_spark_code(code: str) -> str:
            """Run the static gate on the repaired module and return every finding
            with severity, code and line number.

            Args:
                code: the complete repaired module source.
            """
            agent.gate_calls += 1
            report = analyze_generated_code(code, options)
            verdict = "PASS" if report.passed else "FAIL"
            return f"gate: {verdict} (submission #{agent.gate_calls})\n{report.render()}"

        return check_spark_code

    def _make_plan_step_tool(self) -> Callable[[str], str]:
        plan = self.plan

        def get_plan_step(step_id: str) -> str:
            """Return one plan step with the semantic differences it declared and the
            mitigation the code was supposed to implement.

            Args:
                step_id: a step id such as 's6'.
            """
            step = next((s for s in plan.steps if s.id == step_id), None)
            if step is None:
                return f"no step '{step_id}'. Available: {', '.join(s.id for s in plan.steps)}"
            return json.dumps(step.model_dump(mode="json"), indent=2)

        return get_plan_step


def repair_task(
    report: ValidationReport,
    diagnosis: ValidationDiagnosis | None,
    history: list[RepairAttempt],
    attempt: int,
    max_attempts: int,
) -> str:
    failing = ", ".join(c.name for c in report.failed_checks) or "unknown"
    return (
        f"Repair attempt {attempt} of {max_attempts}.\n"
        f"Failing checks: {failing}. Differences measured: {len(report.differences)}.\n\n"
        f"{summarise_history(history)}\n\n"
        "Read the failure and the current code, choose one root cause, and return the "
        "complete corrected module once the static gate passes."
    )
