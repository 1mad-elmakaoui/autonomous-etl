"""Generated-code artifacts and the static gate's verdict.

A layer of its own because both `validation` and `repair` need these types, and
`artifacts.MigrationRecord` needs all three. Without the split the imports form
a cycle — which is a signal that these types are lower-level than the record
that aggregates them, not that Python is being awkward.
"""

from __future__ import annotations

import hashlib

from pydantic import Field

from etl_migrator.domain.enums import Severity
from etl_migrator.domain.spec import StrictModel


class Finding(StrictModel):
    """One result from the deterministic static gate."""

    code: str = Field(description="Stable machine code, e.g. 'GATE001'.")
    severity: Severity
    message: str
    line: int | None = Field(default=None, ge=1)

    def render(self) -> str:
        where = f" (line {self.line})" if self.line else ""
        return f"[{self.severity.value.upper()}] {self.code}{where}: {self.message}"


class StaticAnalysisReport(StrictModel):
    passed: bool
    findings: list[Finding] = Field(default_factory=list)

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.severity is Severity.ERROR]

    def render(self) -> str:
        if not self.findings:
            return "static analysis: clean"
        return "\n".join(f.render() for f in self.findings)


class GeneratedCode(StrictModel):
    """A generated PySpark module plus the agent's own account of it."""

    filename: str
    content: str
    entrypoint: str = Field(
        default="run",
        description="Module-level callable implementing the pipeline contract.",
    )
    imports_used: list[str] = Field(default_factory=list)
    notes: list[str] = Field(
        default_factory=list, description="Decisions worth surfacing in the PR body."
    )

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()

    @property
    def line_count(self) -> int:
        return self.content.count("\n") + 1


class CodeGenResult(StrictModel):
    """What SparkEngineerAgent returns: the code and the gate verdict that the
    agent already saw. The gate is re-run outside the agent before the result is
    trusted, so a lying agent changes nothing."""

    code: GeneratedCode
    static_analysis: StaticAnalysisReport
    gate_iterations: int = Field(
        default=1, ge=1, description="How many times the agent had to fix its own code."
    )
