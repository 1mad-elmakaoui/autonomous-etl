"""Data Validation agent: diagnoses a failure the differ already found.

Read the constructor signature carefully — this agent is only ever built when
`ValidationReport.status` is already FAIL. It cannot be invoked on a passing
migration, and its output type has no field capable of changing a verdict. It
explains and it points; it does not grade.

That constraint is what makes it useful rather than dangerous. An agent asked
"did this migration work?" will find reasons to say yes. An agent handed
"row_count: reference=5 candidate=6" and asked "which plan step caused this?"
is doing work only a model can do — mapping a symptom onto an intention — with
no room to be optimistic about it.

Its output is the input to the repair loop: `root_cause_category` keys
the repair strategy, and each attempt must pick a category it has not tried.
"""

from __future__ import annotations

import json
from collections.abc import Callable

from autogen_core.models import ChatCompletionClient

from etl_migrator.agents.base import StructuredAgent
from etl_migrator.domain.plan import MigrationPlan
from etl_migrator.domain.validation import ValidationDiagnosis, ValidationReport

SYSTEM_MESSAGE = """\
You are a Data Validation Agent. A deterministic differ has executed both the
legacy pipeline and the generated PySpark pipeline against the same input and
found that their outputs disagree. Your job is to explain why and to name the
plan step responsible.

You are not being asked whether the migration is correct. That question is
already answered: it is not. Do not argue with the differ, do not describe the
differences as acceptable, and do not suggest loosening a tolerance.

Rules:

1. Call `get_validation_report` first. Everything you claim must be traceable to
   something in it. Call `get_failed_check` for the detail of a specific check.
2. Call `get_plan_step` for any step you intend to implicate. Naming a step
   without reading it is guessing.
3. `evidence` must quote actual values from the report — "reference=5
   candidate=6 on row_count", not "the row counts differ".
4. `root_cause_category` drives the repair strategy, so choose the *mechanism*,
   not the symptom. An extra output row from a null group key is
   null_semantics, not row_order.
5. `suggested_fix` must be a concrete change to the Spark code: which expression
   to add or replace, and where. "Handle nulls properly" is not a fix.
6. If several differences share one cause, say so — one root cause with several
   symptoms is the common case, and listing five independent problems when there
   is one sends the repair loop in five directions.
7. `confidence` reflects how directly the evidence supports your conclusion.

Common mechanisms, for reference:

* extra output row with a null group key -> pandas groupby(dropna=True) was not
  reproduced; filter the null keys before groupBy
* null where the reference has 0.0 -> pandas sum() of an all-NaN group is 0.0,
  Spark returns null; wrap in coalesce
* integer column became float -> pandas promoted an int column containing nulls;
  declare the type explicitly when reading
* an extra column -> an index column was synthesised for reset_index, which has
  no Spark equivalent and should emit nothing
* small last-digit numeric differences -> float ulp; only a real failure if it
  exceeds the declared tolerance, which the differ has already accounted for
* multiplied row counts -> a non-unique join key
"""


class ValidationAgent(StructuredAgent[ValidationDiagnosis]):
    key = "validation"
    description = "Diagnoses why the generated pipeline's output differs from the reference."

    def __init__(
        self,
        model_client: ChatCompletionClient,
        *,
        report: ValidationReport,
        plan: MigrationPlan,
        max_tool_iterations: int = 6,
    ) -> None:
        if report.status.value == "PASS":
            raise ValueError(
                "ValidationAgent is only for failures; a passing report has nothing to "
                "diagnose and inviting an opinion on it would be an invitation to overrule "
                "the differ"
            )
        self.report = report
        self.plan = plan
        super().__init__(
            model_client,
            ValidationDiagnosis,
            system_message=SYSTEM_MESSAGE,
            tools=[
                self._make_report_tool(),
                self._make_failed_check_tool(),
                self._make_plan_step_tool(),
            ],
            max_tool_iterations=max_tool_iterations,
        )

    def _make_report_tool(self) -> Callable[[], str]:
        report = self.report

        def get_validation_report() -> str:
            """Return the differ's full verdict: every check, whether it passed, and
            every measured difference with reference and candidate values."""
            summary = report.render()
            stats = ""
            if report.reference and report.candidate:
                stats = (
                    f"\n\nreference: {report.reference.row_count} rows, "
                    f"columns={report.reference.column_names}"
                    f"\ncandidate: {report.candidate.row_count} rows, "
                    f"columns={report.candidate.column_names}"
                )
            return summary + stats

        return get_validation_report

    def _make_failed_check_tool(self) -> Callable[[str], str]:
        report = self.report

        def get_failed_check(name: str) -> str:
            """Return every difference recorded by one check, in full.

            Args:
                name: check name, e.g. 'row_count', 'null_counts', 'numeric_tolerance'.
            """
            check = next((c for c in report.checks if c.name == name), None)
            if check is None:
                available = ", ".join(c.name for c in report.checks)
                return f"no check named '{name}'. Available: {available}"
            return json.dumps(check.model_dump(mode="json"), indent=2)

        return get_failed_check

    def _make_plan_step_tool(self) -> Callable[[str], str]:
        plan = self.plan

        def get_plan_step(step_id: str) -> str:
            """Return one plan step: the mapping decision, its rationale, and every
            semantic difference it declared with the mitigation the code was supposed
            to implement.

            Args:
                step_id: a step id such as 's6'.
            """
            step = next((s for s in plan.steps if s.id == step_id), None)
            if step is None:
                return f"no step '{step_id}'. Available: {', '.join(s.id for s in plan.steps)}"
            return json.dumps(step.model_dump(mode="json"), indent=2)

        return get_plan_step


def diagnosis_task(report: ValidationReport, plan: MigrationPlan) -> str:
    failed = ", ".join(c.name for c in report.failed_checks)
    return (
        f"The migration failed validation. Failing checks: {failed}.\n"
        f"There are {len(report.differences)} recorded differences.\n\n"
        "Read the report, inspect the failing checks and the plan steps you suspect, "
        "and produce the diagnosis."
    )
