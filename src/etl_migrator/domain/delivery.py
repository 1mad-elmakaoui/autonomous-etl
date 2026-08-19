"""Models for delivering a finished migration as a pull request.

The types live here and the two rules that police them live in
`domain/delivery_policy.py`, which is the module that may read a
`MigrationRecord`. Splitting them keeps `artifacts.MigrationRecord` able to
carry a `DeliveryOutcome` without the two modules importing each other.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field, computed_field

from etl_migrator.domain.spec import StrictModel

#: Applied to every PR this system opens, so the whole population is findable
#: with one query — including the ones that went wrong.
ORIGIN_LABEL = "autonomous-etl"

#: Applied when the system does not believe its own output is ready to merge.
NEEDS_HUMAN_LABEL = "needs-human"


class DeliveryDisposition(StrEnum):
    """What should happen to this migration."""

    #: Validated, measured, ready for a reviewer.
    READY = "ready"
    #: Opened as a draft, labelled, with the failure stated in the body. A human
    #: is being shown the work, not asked to approve it.
    DRAFT = "draft"
    #: No branch, no PR. There is nothing a reviewer could usefully do.
    REFUSED = "refused"


class FileChange(StrictModel):
    """One file to write on the migration branch."""

    path: str = Field(min_length=1, max_length=400)
    content: str
    message: str = Field(description="Commit message for this file.")

    @property
    def line_count(self) -> int:
        return len(self.content.splitlines())


class BranchRef(StrictModel):
    name: str
    sha: str
    created: bool = Field(
        default=False, description="False when the branch already existed and was reused."
    )


class PullRequestRef(StrictModel):
    number: int = Field(ge=1)
    url: str
    draft: bool = False
    labels: list[str] = Field(default_factory=list)
    created: bool = Field(
        default=False, description="False when an open PR for this branch already existed."
    )


class PullRequestNarrative(StrictModel):
    """The part of the PR body an agent is entitled to author.

    Deliberately excludes anything the record already knows. There is no
    `validation_status` field, no `speedup` field, no summary of what the checks
    found — those are rendered from measurements, so the agent cannot restate
    them incorrectly and a reviewer never has to work out which of two
    disagreeing accounts is the real one.
    """

    title: str = Field(min_length=10, max_length=120)
    summary: str = Field(
        min_length=40,
        description="What this migration does and how the translation was approached.",
    )
    reviewer_focus: list[str] = Field(
        default_factory=list,
        max_length=8,
        description="Specific things a human should look at, each pointing at real "
        "code or a real declared risk.",
    )
    risk_callouts: list[str] = Field(
        default_factory=list,
        max_length=8,
        description="Semantic differences a reviewer must accept, in their own terms.",
    )


class ClaimViolation(StrictModel):
    """One number in the agent's prose that the record does not support."""

    kind: str = Field(description="What sort of claim, e.g. 'speedup'.")
    claimed: str
    supported: list[str] = Field(
        default_factory=list, description="What the record actually measured."
    )
    excerpt: str = Field(description="The sentence it appeared in.")

    def render(self) -> str:
        supported = ", ".join(self.supported) if self.supported else "nothing measured"
        return (
            f"[{self.kind}] claims {self.claimed}; the record supports {supported} "
            f'— in: "{self.excerpt}"'
        )


class ClaimAudit(StrictModel):
    """Whether the prose's numbers match the measurements."""

    violations: list[ClaimViolation] = Field(default_factory=list)
    checked: int = Field(default=0, ge=0, description="Numeric claims examined.")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def passed(self) -> bool:
        return not self.violations

    def render(self) -> str:
        if self.passed:
            return (
                f"claims: OK ({self.checked} numeric claim(s) checked against the record)"
            )
        return "claims: FAIL\n" + "\n".join(f"  {v.render()}" for v in self.violations)


class DeliveryDecision(StrictModel):
    """Computed from the record. No agent field is an input."""

    disposition: DeliveryDisposition
    labels: list[str] = Field(default_factory=list)
    reason: str

    @property
    def should_open(self) -> bool:
        return self.disposition is not DeliveryDisposition.REFUSED

    @property
    def draft(self) -> bool:
        return self.disposition is DeliveryDisposition.DRAFT

    def render(self) -> str:
        return f"delivery: {self.disposition.value} — {self.reason}"


class DeliveryOutcome(StrictModel):
    """Everything the delivery stage produced."""

    decision: DeliveryDecision | None = None
    branch: BranchRef | None = None
    pull_request: PullRequestRef | None = None
    files: list[str] = Field(default_factory=list)
    audit: ClaimAudit | None = None
    narrative_revisions: int = Field(
        default=0, ge=0, description="Times the agent had to revise to pass the audit."
    )
    skipped_reason: str | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def opened(self) -> bool:
        return self.pull_request is not None

    def render(self) -> str:
        if self.skipped_reason is not None:
            return f"delivery: skipped — {self.skipped_reason}"
        lines = [self.decision.render() if self.decision else "delivery: no decision"]
        if self.branch is not None:
            verb = "created" if self.branch.created else "reused"
            lines.append(f"  branch {verb}: {self.branch.name} @ {self.branch.sha[:8]}")
        for path in self.files:
            lines.append(f"  file: {path}")
        if self.pull_request is not None:
            pr = self.pull_request
            verb = "opened" if pr.created else "already open"
            kind = "draft PR" if pr.draft else "PR"
            lines.append(f"  {kind} #{pr.number} {verb}: {pr.url}")
            if pr.labels:
                lines.append(f"  labels: {', '.join(pr.labels)}")
        if self.audit is not None:
            lines.append(f"  {self.audit.render()}")
        return "\n".join(lines)
