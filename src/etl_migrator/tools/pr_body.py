"""Rendering the pull request body.

The split is the whole design. Everything a reviewer might *act* on — the
validation verdict, the check table, the measured speedup, the semantic
differences, the repair history — is rendered here, from the `MigrationRecord`,
by code with no model anywhere near it. The agent's prose is inserted into
clearly labelled sections and nowhere else.

That is not a stylistic preference. A PR body is the artefact a reviewer reads
first and often the only one they read closely, so it is exactly where an
overstated claim does the most damage. The optimiser already treats an agent's
`expected_speedup` as worth recording and not believing; this applies the same
rule to a paragraph of English.

The `<!-- etl-migrator:… -->` markers exist so a future update can replace the
generated evidence while leaving human review comments and any hand-edits to the
narrative intact.
"""

from __future__ import annotations

from etl_migrator.domain.artifacts import MigrationRecord
from etl_migrator.domain.delivery import (
    DeliveryDecision,
    DeliveryDisposition,
    PullRequestNarrative,
)
from etl_migrator.domain.enums import ValidationStatus

EVIDENCE_START = "<!-- etl-migrator:evidence:start -->"
EVIDENCE_END = "<!-- etl-migrator:evidence:end -->"

_STATUS_BADGE = {
    ValidationStatus.PASS: "✅ **PASS**",
    ValidationStatus.FAIL: "❌ **FAIL**",
    ValidationStatus.ERROR: "⚠️ **ERROR**",
}


def _validation_section(record: MigrationRecord) -> list[str]:
    outcome = record.validation
    if outcome is None:
        return [
            "### Validation",
            "",
            "⚠️ **Not run.** This migration's correctness is unverified — the "
            "generated pipeline was never executed against the legacy one.",
        ]

    report = outcome.report
    lines = [
        "### Validation",
        "",
        f"Verdict: {_STATUS_BADGE.get(report.status, report.status.value)} "
        "— from executing both pipelines and diffing their real output.",
        "",
        "| Check | Result | Detail |",
        "|---|---|---|",
    ]
    for check in report.checks:
        mark = "⏭️ skipped" if check.skipped else ("✅ pass" if check.passed else "❌ **fail**")
        lines.append(f"| `{check.name}` | {mark} | {check.detail or '—'} |")

    if outcome.test_run is not None:
        run = outcome.test_run
        lines += [
            "",
            f"Generated test suite: **{run.passed} passed**, {run.failed} failed, "
            f"{run.errors} errors, {run.skipped} skipped.",
        ]

    differences = report.differences
    if differences:
        lines += ["", "**Differences found:**", ""]
        lines += [f"- {d.render()}" for d in differences[:10]]
        if len(differences) > 10:
            lines.append(f"- …and {len(differences) - 10} more.")

    if outcome.diagnosis is not None:
        d = outcome.diagnosis
        lines += [
            "",
            f"**Diagnosis** ({d.root_cause_category.value}, confidence "
            f"{d.confidence:.0%}): {d.summary}",
            "",
            f"Suggested fix: {d.suggested_fix}",
        ]
    return lines


def _optimization_section(record: MigrationRecord) -> list[str]:
    outcome = record.optimization
    if outcome is None:
        return []

    lines = ["### Performance", ""]
    if outcome.baseline is not None:
        lines.append(f"- Baseline: `{outcome.baseline.render()}`")
    if outcome.final is not None and outcome.final is not outcome.baseline:
        lines.append(f"- Optimised: `{outcome.final.render()}`")

    if outcome.applied and outcome.accepted_strategy is not None:
        strategy = outcome.accepted_strategy
        lines += [
            "",
            f"**{outcome.speedup:.2f}x faster** via `{strategy.approach}`, kept because "
            "validation still passed *and* the gain cleared the threshold robustly "
            "across repeated runs with warm-up discarded.",
            "",
            f"> {strategy.rationale}",
            "",
            f"The agent predicted {strategy.expected_speedup:.2f}x; the measurement is "
            "what was kept.",
        ]
    else:
        lines += [
            "",
            "No optimisation was kept. Every attempt and the reason it was refused:",
            "",
        ]
        lines += [f"- {a.render()}" for a in outcome.attempts] or [
            "- (none proposed)"
        ]
    return lines


def _plan_section(record: MigrationRecord) -> list[str]:
    if record.plan is None:
        return []
    differences = record.plan.all_semantic_differences
    if not differences:
        return []
    lines = [
        "### Semantic differences handled",
        "",
        f"{len(differences)} pandas↔Spark divergence(s) the plan identified. Each one "
        "became a required validation check:",
        "",
    ]
    lines += [
        f"- **{d.category.value}** — {d.description} "
        f"_Mitigation:_ {d.mitigation} _(checked by `{d.validation_check}`)_"
        for d in differences[:12]
    ]
    if len(differences) > 12:
        lines.append(f"- …and {len(differences) - 12} more.")
    return lines


def _repair_section(record: MigrationRecord) -> list[str]:
    outcome = record.repair
    if outcome is None or not outcome.attempts:
        return []
    verdict = "recovered to a passing validation" if outcome.succeeded else "did not recover"
    lines = [
        "### Autonomous repair",
        "",
        f"The first generated version failed validation. The repair loop {verdict} "
        f"after {len(outcome.attempts)} attempt(s):",
        "",
    ]
    lines += [f"- {a.render()}" for a in outcome.attempts]
    return lines


def render_evidence(record: MigrationRecord, decision: DeliveryDecision) -> str:
    """The measured half of the body. Contains no agent-authored text."""
    lines: list[str] = [EVIDENCE_START, ""]

    if decision.disposition is DeliveryDisposition.DRAFT:
        lines += [
            "> [!WARNING]",
            f"> **This is a draft and should not be merged.** {decision.reason}.",
            "",
        ]

    lines += ["## Evidence", ""]
    lines += _validation_section(record)
    for section in (
        _optimization_section(record),
        _plan_section(record),
        _repair_section(record),
    ):
        if section:
            lines += ["", *section]

    lines += [
        "",
        "### Provenance",
        "",
        f"- Migration `{record.migration_id}`",
        f"- Source: `{record.source_path}`",
        f"- Overall risk: **{record.risk.value}**",
        f"- Stages: {len(record.stages)}, total {record.total_duration_seconds:.1f}s",
    ]
    if record.plan is not None and record.plan.requires_human_approval:
        lines.append("- Required human approval before code generation")
    lines += [
        "",
        "_Every number above is read from the migration record, which is written "
        "from executed runs. None of it is authored by a language model._",
        "",
        EVIDENCE_END,
    ]
    return "\n".join(lines)


def render_pr_body(
    record: MigrationRecord,
    decision: DeliveryDecision,
    narrative: PullRequestNarrative,
) -> str:
    """Assemble the full body: agent prose first, then measured evidence."""
    lines = [narrative.summary.strip(), ""]

    if narrative.risk_callouts:
        lines += ["## What a reviewer must accept", ""]
        lines += [f"- {item}" for item in narrative.risk_callouts]
        lines.append("")

    if narrative.reviewer_focus:
        lines += ["## Where to look", ""]
        lines += [f"- {item}" for item in narrative.reviewer_focus]
        lines.append("")

    lines.append(render_evidence(record, decision))
    lines += [
        "",
        "---",
        "",
        "🤖 Opened by the [Autonomous ETL Migration Agent]"
        "(https://github.com/1mad-elmakaoui/autonomous-etl). "
        "The prose above the evidence block is agent-authored and was audited "
        "against the migration record before this PR was opened; every figure in "
        "the evidence block is rendered from measurements.",
    ]
    return "\n".join(lines)
