"""Delivery agent: writes the prose a reviewer reads first.

Its authority is narrow by construction. It cannot state the verdict — there is
no field on `PullRequestNarrative` for one — and every number it writes is
checked against the migration record before the PR is opened. What is left is
the genuinely useful part that measurements cannot supply: what this migration
*did*, which specific lines deserve a human's attention, and which semantic
differences a reviewer is being asked to accept.

It gets the same self-check affordance as the Spark Engineer. `check_claims`
runs the real audit — the same function the delivery step runs at the boundary
— so the agent can find its own overstatement and fix it, rather than having a
revision round spent on something it could have caught itself.
"""

from __future__ import annotations

from collections.abc import Callable

from autogen_core.models import ChatCompletionClient

from etl_migrator.agents.base import StructuredAgent
from etl_migrator.domain.artifacts import MigrationRecord
from etl_migrator.domain.delivery import (
    ClaimAudit,
    DeliveryDecision,
    PullRequestNarrative,
)
from etl_migrator.domain.delivery_policy import audit_numeric_claims

SYSTEM_MESSAGE = """\
You are a Delivery Agent. A migration has finished and is about to become a pull
request. You write the prose a reviewer reads before anything else.

Understand what you are and are not writing. The PR body has two halves. The
evidence half — the validation verdict, the check table, the measured speedup,
the semantic differences, the repair history — is rendered directly from the
migration record and you do not author it. Restating any of it is wasted words
at best; getting it wrong is worse, and will be caught.

Your half is what the record cannot say:

1. **summary** — what this pipeline computes, and how the translation was
   approached. Write for someone who has never seen the legacy code. Describe
   the *work*, not the outcome.
2. **risk_callouts** — the pandas↔Spark divergences a reviewer is being asked to
   accept, in plain terms. "pandas drops null group keys and Spark keeps them,
   so the generated code filters nulls before the groupBy" is useful. "There are
   semantic differences" is not.
3. **reviewer_focus** — specific places to look. Name a function, a column, a
   transformation. "Check the join" is not focus; "the left join on customer_id
   keeps adults with no orders, so revenue is null for them — confirm 0.0 is the
   intended value" is.

Rules:

- Call `get_migration_facts` first. Write nothing about the migration before you
  have read what actually happened to it.
- Call `get_generated_code` when you need to point at something specific. Do not
  describe code you have not read.
- **Every number you write is checked against the record.** A speedup, a row
  count, a check tally, an attempt count — if the record does not contain it,
  the PR is refused and you will be asked to rewrite. Call `check_claims` before
  you answer and fix anything it flags.
- If you are unsure of a figure, leave it out. The evidence block already states
  it, correctly, immediately below your prose.
- Do not congratulate the system. A reviewer wants to know what to check, not
  that the migration went well.
- The title should name the pipeline and what happened, under 120 characters.

If the migration failed or its validation did not pass, say so plainly in the
summary and use reviewer_focus to point at what needs human judgement. A draft
PR asking for help is a legitimate and useful outcome; dressing it up as a
success is not.
"""


class DeliveryAgent(StructuredAgent[PullRequestNarrative]):
    key = "delivery"
    description = "Writes the reviewer-facing prose for a migration pull request."

    def __init__(
        self,
        model_client: ChatCompletionClient,
        *,
        record: MigrationRecord,
        decision: DeliveryDecision,
        max_tool_iterations: int = 8,
    ) -> None:
        self.record = record
        self.decision = decision
        self.audit_calls = 0
        self.last_audit: ClaimAudit | None = None
        super().__init__(
            model_client,
            PullRequestNarrative,
            system_message=SYSTEM_MESSAGE,
            tools=[
                self._make_facts_tool(),
                self._make_code_tool(),
                self._make_plan_tool(),
                self._make_audit_tool(),
            ],
            max_tool_iterations=max_tool_iterations,
        )

    def _make_facts_tool(self) -> Callable[[], str]:
        record, decision = self.record, self.decision

        def get_migration_facts() -> str:
            """Return what actually happened to this migration: the validation
            verdict and every check, the optimisation result, the repair history,
            the declared risk, and how this PR will be opened."""
            lines = [
                f"migration: {record.migration_id}",
                f"source: {record.source_path}",
                f"risk: {record.risk.value}",
                f"disposition: {decision.disposition.value} — {decision.reason}",
            ]
            if record.validation is not None:
                report = record.validation.report
                lines.append(f"validation: {report.status.value}")
                lines += [
                    f"  [{'skip' if c.skipped else ('ok' if c.passed else 'FAIL')}] "
                    f"{c.name}: {c.detail}"
                    for c in report.checks
                ]
                if record.validation.test_run is not None:
                    lines.append(f"  tests: {record.validation.test_run.render()}")
            else:
                lines.append("validation: NOT RUN")
            if record.repair is not None:
                lines.append(f"repair: {'succeeded' if record.repair.succeeded else 'exhausted'}")
                lines += [f"  {a.render()}" for a in record.repair.attempts]
            if record.optimization is not None:
                lines.append(record.optimization.render())
            return "\n".join(lines)

        return get_migration_facts

    def _make_code_tool(self) -> Callable[[], str]:
        record = self.record

        def get_generated_code() -> str:
            """Return the full source of the PySpark module this PR delivers."""
            if record.codegen is None:
                return "No code was generated."
            return record.codegen.code.content

        return get_generated_code

    def _make_plan_tool(self) -> Callable[[], str]:
        record = self.record

        def get_semantic_differences() -> str:
            """Return every pandas/Spark divergence the plan identified, with the
            mitigation applied and the check that proves it worked."""
            if record.plan is None:
                return "No plan is available."
            differences = record.plan.all_semantic_differences
            if not differences:
                return "The plan declared no semantic differences."
            return "\n".join(
                f"- [{d.category.value}] {d.description}\n"
                f"  mitigation: {d.mitigation}\n"
                f"  proven by: {d.validation_check}"
                for d in differences
            )

        return get_semantic_differences

    def _make_audit_tool(self) -> Callable[[str], str]:
        agent = self

        def check_claims(text: str) -> str:
            """Check the numbers in a draft against the migration record. Call this
            before answering.

            Args:
                text: the prose you intend to submit — summary, callouts and focus
                    points together are fine.
            """
            agent.audit_calls += 1
            draft = PullRequestNarrative(
                title="Draft title placeholder for auditing",
                summary=text if len(text) >= 40 else text.ljust(40),
            )
            audit = audit_numeric_claims(draft, agent.record)
            agent.last_audit = audit
            return f"(submission #{agent.audit_calls})\n{audit.render()}"

        return check_claims


def delivery_task(record: MigrationRecord, decision: DeliveryDecision) -> str:
    return (
        f"Write the pull request narrative for migration {record.migration_id}.\n"
        f"It will be opened as: {decision.disposition.value} — {decision.reason}\n\n"
        "Read the facts, read the generated code, then write the summary, the risk "
        "callouts and the reviewer focus points. Check your numbers before you answer."
    )


def revision_task(audit: ClaimAudit) -> str:
    return (
        "The narrative was rejected: it contains figures the migration record does "
        f"not support.\n\n{audit.render()}\n\n"
        "Rewrite it. Either use the measured value or drop the claim — the evidence "
        "block below your prose already states every figure correctly, so leaving a "
        "number out costs the reviewer nothing."
    )
