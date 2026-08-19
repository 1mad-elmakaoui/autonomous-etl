"""The repair loop: strategies, attempts, and the ledger that bounds it.

Two failure modes dominate a loop like this. Oscillation, where the agent
proposes fix A, then B, then A again until the budget runs out having tried two
ideas. And cosmetic churn, where the returned code differs only in whitespace
and the system pays for a full Spark run to find out it behaves identically.

`RepairLedger` closes both without an LLM call. A strategy is identified by
`(root cause category, approach slug)`, and proposing a signature already tried
is rejected. So is code whose content hash matches an earlier attempt.

The rejection is recorded and fed to the next attempt, so the agent observes
that its idea was already spent. That is what makes the bound a constraint
rather than a cap on how long the flailing lasts.
"""

from __future__ import annotations

import hashlib

from pydantic import Field

from etl_migrator.domain.code import GeneratedCode, StaticAnalysisReport
from etl_migrator.domain.enums import RiskCategory, ValidationStatus
from etl_migrator.domain.spec import StrictModel
from etl_migrator.domain.validation import ValidationDiagnosis, ValidationReport


class RepairStrategy(StrictModel):
    """What the agent intends to change, and on what theory.

    `approach` is a slug rather than prose precisely so that two attempts can be
    compared mechanically. "Filter the null keys" and "filter out null country
    values" are the same idea in different words; `filter_null_group_keys` twice
    is detectably the same idea.
    """

    category: RiskCategory = Field(
        description="Root-cause class this attempt addresses, from the diagnosis."
    )
    approach: str = Field(
        pattern=r"^[a-z][a-z0-9_]{2,48}$",
        description="Stable slug for the technique, e.g. 'filter_null_group_keys'.",
    )
    description: str = Field(description="What changes in the code, concretely.")
    target_step_ids: list[str] = Field(
        default_factory=list, description="Plan steps the change touches."
    )

    @property
    def signature(self) -> str:
        """Identity for the purposes of 'have we tried this already?'."""
        return f"{self.category.value}:{self.approach}"


class RepairProposal(StrictModel):
    """One repair the agent wants to try. Produced by `RepairAgent`."""

    strategy: RepairStrategy
    code: GeneratedCode
    rationale: str = Field(description="Why this addresses the measured differences.")
    expected_effect: str = Field(
        description="What the differ should show afterwards. Checkable after the fact, "
        "which makes an over-optimistic claim visible rather than free."
    )


class RepairAttempt(StrictModel):
    """The durable record of one attempt, accepted or not."""

    attempt: int = Field(ge=1)
    strategy: RepairStrategy | None = None
    code_sha256: str | None = None
    admitted: bool = Field(
        default=True, description="False when the ledger rejected it before execution."
    )
    rejection_reason: str | None = None
    static_analysis: StaticAnalysisReport | None = None
    validation_status: ValidationStatus | None = None
    differences: int | None = Field(default=None, ge=0)
    expected_effect: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.validation_status is ValidationStatus.PASS

    def render(self) -> str:
        if not self.admitted:
            return f"attempt {self.attempt}: REJECTED — {self.rejection_reason}"
        head = f"attempt {self.attempt}"
        if self.strategy is not None:
            head += f" [{self.strategy.signature}]"
        if self.static_analysis is not None and not self.static_analysis.passed:
            return f"{head}: failed the static gate"
        status = self.validation_status.value if self.validation_status else "not run"
        return f"{head}: validation {status} ({self.differences} differences)"


class RepairOutcome(StrictModel):
    """Everything `RepairWorkflow` returns to its parent."""

    succeeded: bool = False
    attempts: list[RepairAttempt] = Field(default_factory=list)
    repaired_code: GeneratedCode | None = None
    final_report: ValidationReport | None = None
    exhausted: bool = Field(
        default=False,
        description="True when the attempt budget ran out. Terminal, and routed to a "
        "human rather than treated as a crash.",
    )

    @property
    def attempts_used(self) -> int:
        return len(self.attempts)

    @property
    def best_attempt(self) -> RepairAttempt | None:
        """The attempt that got closest, by measured difference count.

        When the budget is exhausted a human needs the *nearest miss*, not the
        last thing tried — the final attempt is often a wilder guess than the
        second one.
        """
        scored = [
            a for a in self.attempts if a.admitted and a.differences is not None
        ]
        if not scored:
            return None
        return min(scored, key=lambda a: (a.differences or 0, a.attempt))

    def render(self) -> str:
        lines = [
            f"repair: {'SUCCEEDED' if self.succeeded else 'exhausted'} "
            f"after {self.attempts_used} attempt(s)"
        ]
        lines += [f"  {a.render()}" for a in self.attempts]
        best = self.best_attempt
        if not self.succeeded and best is not None:
            lines.append(f"  closest: attempt {best.attempt} with {best.differences} differences")
        return "\n".join(lines)


def code_fingerprint(code: GeneratedCode) -> str:
    """Hash of the code's *meaning-bearing* text.

    Comments and blank lines are stripped before hashing so that a proposal
    differing only in commentary is recognised as the same attempt. It is a
    deliberately cheap approximation — not an AST normalisation — because the
    cost of a false negative is one wasted Spark run, while the cost of a false
    positive is refusing a genuine fix.
    """
    meaningful = [
        line.rstrip()
        for line in code.content.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    return hashlib.sha256("\n".join(meaningful).encode("utf-8")).hexdigest()


class RepairLedger:
    """Decides whether a proposal is worth executing. Pure and deterministic.

    Deliberately not an agent, and deliberately not a prompt instruction: "do
    not repeat yourself" is advice a model can forget, whereas a set membership
    test cannot.
    """

    def __init__(self, max_attempts: int) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        self.max_attempts = max_attempts
        self._signatures: set[str] = set()
        self._fingerprints: set[str] = set()
        self._baseline_fingerprint: str | None = None

    def register_baseline(self, code: GeneratedCode) -> None:
        """Record the failing code, so 'repairing' it into itself is rejected."""
        self._baseline_fingerprint = code_fingerprint(code)
        self._fingerprints.add(self._baseline_fingerprint)

    def admits(self, proposal: RepairProposal) -> tuple[bool, str | None]:
        """Return `(admissible, reason_if_not)`."""
        signature = proposal.strategy.signature
        if signature in self._signatures:
            return False, (
                f"strategy '{signature}' was already tried and did not fix the failure; "
                "a different root-cause approach is required"
            )

        fingerprint = code_fingerprint(proposal.code)
        if fingerprint == self._baseline_fingerprint:
            return False, (
                "the proposed code is identical to the code that failed validation; "
                "no change was actually made"
            )
        if fingerprint in self._fingerprints:
            return False, (
                "this exact code was already tried under a different strategy label; "
                "identical code cannot behave differently"
            )
        return True, None

    def record(self, proposal: RepairProposal) -> None:
        """Mark a proposal as spent. Called for admitted proposals only."""
        self._signatures.add(proposal.strategy.signature)
        self._fingerprints.add(code_fingerprint(proposal.code))

    @property
    def tried_signatures(self) -> list[str]:
        return sorted(self._signatures)


def summarise_history(attempts: list[RepairAttempt]) -> str:
    """Render prior attempts for the agent's next prompt.

    This is the feedback half of the loop: the agent is told what has already
    been spent, so a repeat is a choice rather than an accident.
    """
    if not attempts:
        return "No previous repair attempts."
    lines = ["Previous repair attempts (do not repeat any of these strategies):"]
    for attempt in attempts:
        lines.append(f"  {attempt.render()}")
        if attempt.strategy is not None:
            lines.append(f"    approach: {attempt.strategy.description}")
        if attempt.expected_effect and attempt.validation_status is not None:
            lines.append(f"    expected: {attempt.expected_effect}")
    return "\n".join(lines)


def build_diagnosis_summary(
    report: ValidationReport, diagnosis: ValidationDiagnosis | None
) -> str:
    """Compact statement of what is wrong, for the repair prompt."""
    lines = [report.render()]
    if diagnosis is not None:
        lines += [
            "",
            f"Diagnosis ({diagnosis.root_cause_category.value}, "
            f"confidence {diagnosis.confidence:.0%}): {diagnosis.summary}",
            f"Implicated steps: {', '.join(diagnosis.implicated_step_ids) or 'none named'}",
            f"Suggested fix: {diagnosis.suggested_fix}",
        ]
    return "\n".join(lines)
