"""The two rules that decide what a migration is allowed to claim.

A PR is a claim about work, and claims can be inflated. Left to narrate freely,
the agent will write "all checks passed" for a migration whose validation
errored, or "3x faster" for a measured 1.38x, because those are the sentences
that usually follow. So the body is assembled in two parts: the evidence is
rendered from the `MigrationRecord` (see `tools/pr_body.py`), and the prose is
checked by `audit_numeric_claims` before it reaches a branch.

Not every migration should become a PR at all. One whose validation failed has
produced code that is known to be wrong, and opening a normal PR invites a
reviewer to approve it. `decide_delivery` refuses, or opens a draft carrying a
`needs-human` label.

Neither rule is enforced by a prompt, and neither takes an agent field as input.
"""

from __future__ import annotations

import re

from etl_migrator.domain.artifacts import MigrationRecord
from etl_migrator.domain.delivery import (
    NEEDS_HUMAN_LABEL,
    ORIGIN_LABEL,
    ClaimAudit,
    ClaimViolation,
    DeliveryDecision,
    DeliveryDisposition,
    PullRequestNarrative,
)
from etl_migrator.domain.enums import RiskLevel, ValidationStatus


def decide_delivery(record: MigrationRecord) -> DeliveryDecision:
    """Decide whether this migration may become a pull request, and in what form.

    The ordering mirrors `evaluate_optimization`: the disqualifying conditions
    are checked first, so the reason a reviewer is given is the actionable one
    rather than whichever check happened to run last.
    """
    labels = [ORIGIN_LABEL]

    if record.codegen is None:
        return DeliveryDecision(
            disposition=DeliveryDisposition.REFUSED,
            reason="no code was generated; there is nothing to deliver",
        )

    if record.failed:
        return DeliveryDecision(
            disposition=DeliveryDisposition.DRAFT,
            labels=[*labels, NEEDS_HUMAN_LABEL, "migration-failed"],
            reason=(
                f"the migration failed ({record.failure_reason}); opened as a draft so "
                "the work is visible without inviting an approval"
            ),
        )

    if record.validation is None:
        return DeliveryDecision(
            disposition=DeliveryDisposition.REFUSED,
            reason=(
                "the migration was never validated, so its correctness is unknown. A PR "
                "would ask a reviewer to approve an unproven translation"
            ),
        )

    status = record.validation.report.status
    if status is not ValidationStatus.PASS:
        repaired = record.repair is not None and record.repair.succeeded
        if not repaired:
            return DeliveryDecision(
                disposition=DeliveryDisposition.DRAFT,
                labels=[*labels, NEEDS_HUMAN_LABEL, f"validation-{status.value.lower()}"],
                reason=(
                    f"validation {status.value} and repair did not recover it. The code is "
                    "known to be wrong, so this is a draft asking for help, not a merge "
                    "request"
                ),
            )

    if record.risk is RiskLevel.HIGH:
        labels.append("high-risk")

    if record.optimization is not None and record.optimization.applied:
        labels.append("optimised")

    return DeliveryDecision(
        disposition=DeliveryDisposition.READY,
        labels=labels,
        reason="validation passed on executed output; ready for review",
    )


# ---------------------------------------------------------------------------
# The claim audit
# ---------------------------------------------------------------------------
#: Each pattern captures one number in a context that makes it a *claim about
#: this migration*, rather than any number that happens to appear in prose. The
#: narrowness is deliberate: an audit that flags "Python 3.11" or "2 CSV files"
#: would be noise, and noise gets switched off.
_CLAIM_PATTERNS: tuple[tuple[str, str], ...] = (
    # Both the ASCII "x" and U+00D7 are matched, since models write speedups
    # either way and catching only one is easy to slip past. Spelled as an
    # escape to keep the source ASCII. The word boundary applies only to the
    # ASCII branch: "x" needs one so "matrix" is not a speedup claim, while
    # U+00D7 is already a separator and a boundary after it would never match.
    ("speedup", "(\\d+(?:\\.\\d+)?)\\s*(?:x\\b|\u00d7)"),
    ("checks", r"(\d+)\s+(?:of\s+\d+\s+)?checks?\b"),
    ("tests", r"(\d+)\s+tests?\b"),
    ("rows", r"([\d,]+)\s+rows?\b"),
    ("differences", r"(\d+)\s+(?:semantic\s+)?(?:differences?|diffs?)\b"),
    ("attempts", r"(\d+)\s+(?:repair\s+)?attempts?\b"),
)


def _sentence_around(text: str, index: int) -> str:
    start = max(text.rfind(".", 0, index), text.rfind("\n", 0, index)) + 1
    end = min(
        (i for i in (text.find(".", index), text.find("\n", index)) if i != -1),
        default=len(text),
    )
    return text[start : end + 1].strip()[:160]


def supported_numbers(record: MigrationRecord) -> dict[str, set[str]]:
    """The numbers this migration actually measured, by claim kind.

    Everything here is read from the record. A claim is permitted only if it
    appears in this mapping, so the audit's strictness is bounded by what was
    genuinely observed rather than by a list of forbidden phrases.
    """
    permitted: dict[str, set[str]] = {kind: set() for kind, _ in _CLAIM_PATTERNS}

    # An unchanged pipeline is honestly "1x", so it is always permitted.
    permitted["speedup"].add("1")
    permitted["speedup"].add("1.0")
    permitted["speedup"].add("1.00")

    optimization = record.optimization
    if optimization is not None:
        speedup = optimization.speedup
        permitted["speedup"] |= {f"{speedup:.2f}", f"{speedup:.1f}", f"{speedup:g}"}
        permitted["attempts"].add(str(len(optimization.attempts)))
        for attempt in optimization.attempts:
            claimed = attempt.strategy.expected_speedup
            # The agent may quote its own earlier prediction, but only as one.
            permitted["speedup"] |= {f"{claimed:.2f}", f"{claimed:.1f}", f"{claimed:g}"}

    validation = record.validation
    if validation is not None:
        checks = validation.report.checks
        permitted["checks"] |= {
            str(len(checks)),
            str(sum(1 for c in checks if c.passed)),
            str(sum(1 for c in checks if not c.passed)),
        }
        permitted["differences"].add(str(len(validation.report.differences)))
        for stats in (validation.report.reference, validation.report.candidate):
            if stats is not None:
                permitted["rows"] |= {str(stats.row_count), f"{stats.row_count:,}"}
        run = validation.test_run
        if run is not None:
            permitted["tests"] |= {
                str(run.passed),
                str(run.failed),
                str(run.errors),
                str(run.total),
            }

    if record.plan is not None:
        permitted["differences"].add(str(len(record.plan.all_semantic_differences)))

    if record.repair is not None:
        permitted["attempts"].add(str(len(record.repair.attempts)))

    return permitted


def audit_numeric_claims(narrative: PullRequestNarrative, record: MigrationRecord) -> ClaimAudit:
    """Check every number in the agent's prose against what was measured.

    This does not attempt to understand the prose. It answers one narrow
    question — "does the record contain this number?" — which is enough to catch
    the failure that matters: a PR body announcing a speedup, a row count or a
    check tally that never happened. A reviewer who trusts the summary and skims
    the evidence is the person this protects.
    """
    permitted = supported_numbers(record)
    text = "\n".join(
        [narrative.title, narrative.summary, *narrative.reviewer_focus, *narrative.risk_callouts]
    )

    violations: list[ClaimViolation] = []
    checked = 0
    for kind, pattern in _CLAIM_PATTERNS:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            checked += 1
            claimed = match.group(1)
            allowed = permitted[kind]
            if claimed in allowed or claimed.replace(",", "") in {
                a.replace(",", "") for a in allowed
            }:
                continue
            violations.append(
                ClaimViolation(
                    kind=kind,
                    claimed=claimed,
                    supported=sorted(allowed),
                    excerpt=_sentence_around(text, match.start()),
                )
            )
    return ClaimAudit(violations=violations, checked=checked)
