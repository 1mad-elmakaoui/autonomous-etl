"""The work units of a migration, free of any orchestration framework.

There are two orchestrators over these steps and there must only ever be one
implementation beneath them:

* `pipeline.local.LocalMigrationPipeline` — sequential, in-process, no server.
  What you run during development and what most tests drive.
* `activities.migration.MigrationActivities` — the same steps wrapped as
  Temporal Activities, with retries, timeouts and durable state.

If the two orchestrators had their own copies of "call the discovery agent",
they would drift, and the local path would stop being a faithful rehearsal of
the durable one. So everything that actually does work lives here, and the
orchestrators only decide *when* to call it and *what happens on failure*.

Nothing in this module knows about Temporal, and nothing here holds state
between calls.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from etl_migrator.agents.base import AgentRun, StructuredAgent
from etl_migrator.agents.delivery import DeliveryAgent, delivery_task, revision_task
from etl_migrator.agents.discovery import DiscoveryAgent, discovery_task
from etl_migrator.agents.optimizer import OptimizerAgent, optimization_task
from etl_migrator.agents.planner import PlannerAgent, planning_task
from etl_migrator.agents.repair import RepairAgent, repair_task
from etl_migrator.agents.spark_engineer import SparkEngineerAgent, codegen_task
from etl_migrator.agents.testing import TestingAgent, testing_task
from etl_migrator.agents.validation import ValidationAgent, diagnosis_task
from etl_migrator.config import Settings
from etl_migrator.domain.artifacts import (
    CodeGenResult,
    GeneratedCode,
    MigrationRecord,
    StaticAnalysisReport,
)
from etl_migrator.domain.delivery import (
    ClaimAudit,
    DeliveryDecision,
    DeliveryOutcome,
    FileChange,
    PullRequestNarrative,
)
from etl_migrator.domain.history import MigrationHistory
from etl_migrator.domain.messages import AgentTelemetry, ToolResult
from etl_migrator.domain.optimization import (
    DEFAULT_RUNS,
    DEFAULT_WARMUPS,
    BenchmarkResult,
    OptimizationAttempt,
    OptimizationProposal,
)
from etl_migrator.domain.plan import ExecutionStrategy, MigrationPlan
from etl_migrator.domain.repair import RepairAttempt, RepairProposal
from etl_migrator.domain.spec import MigrationSpec
from etl_migrator.domain.validation import (
    ExecutionResult,
    GeneratedTests,
    TestRunResult,
    ValidationDiagnosis,
    ValidationReport,
)
from etl_migrator.github.client import GitHubClient
from etl_migrator.llm.factory import ModelClientFactory
from etl_migrator.observability import get_logger
from etl_migrator.sandbox.execute import run_legacy_pipeline, run_spark_pipeline, spark_conf_from
from etl_migrator.sandbox.pytest_runner import run_generated_tests
from etl_migrator.tools.benchmark import benchmark_pipeline
from etl_migrator.tools.code_gate import GateOptions, analyze_generated_code
from etl_migrator.tools.data_profiler import DatasetProfile, profile_directory
from etl_migrator.tools.differ import compare_outputs
from etl_migrator.tools.plan_analyzer import PlanAnalysis, analyze_plan
from etl_migrator.tools.pr_body import render_pr_body

M = TypeVar("M", bound=BaseModel)

log = get_logger(__name__)


@dataclass(frozen=True)
class StepContext:
    """Everything a step needs from its environment, passed explicitly.

    Explicit rather than global because a Temporal worker may serve several
    migrations concurrently, each potentially against a different fixture
    scenario.
    """

    settings: Settings
    factory: ModelClientFactory
    gate_options: GateOptions = field(default_factory=GateOptions)


@asynccontextmanager
async def _closing(agent: StructuredAgent[M]) -> AsyncIterator[StructuredAgent[M]]:
    """Release the model client even when a step raises."""
    try:
        yield agent
    finally:
        await agent.close()


def telemetry_of(run: AgentRun[M], *, gate_iterations: int | None = None) -> AgentTelemetry:
    """Flatten an `AgentRun` into a wire-safe telemetry record.

    `AgentRun` is generic over the agent's output type; Temporal payloads are
    easier to reason about when the envelope is concrete, so the output travels
    as its own field and the evidence travels as this.
    """
    return AgentTelemetry(
        agent=run.agent,
        tools_used=run.tools_used,
        llm_turns=run.llm_turns,
        duration_seconds=round(run.duration_seconds, 3),
        gate_iterations=gate_iterations,
        tool_results=[
            ToolResult(name=t.name, is_error=t.is_error, preview=t.result_preview)
            for t in run.tool_invocations
        ],
    )


def profile_inputs(input_dir: Path) -> list[DatasetProfile]:
    """Measure the real input data. Deterministic, no LLM."""
    return profile_directory(input_dir)


async def discover(
    ctx: StepContext, *, source_path: Path, input_dir: Path
) -> AgentRun[MigrationSpec]:
    agent = DiscoveryAgent(
        ctx.factory.client_for(DiscoveryAgent.key),
        source_path=source_path,
        input_dir=input_dir,
        max_tool_iterations=ctx.settings.llm_max_tool_iterations,
    )
    async with _closing(agent):
        return await agent.run(discovery_task(source_path, input_dir))


async def plan_migration(
    ctx: StepContext, *, spec: MigrationSpec, profiles: Sequence[DatasetProfile]
) -> AgentRun[MigrationPlan]:
    agent = PlannerAgent(
        ctx.factory.client_for(PlannerAgent.key),
        spec=spec,
        profiles=list(profiles),
        max_tool_iterations=ctx.settings.llm_max_tool_iterations,
    )
    async with _closing(agent):
        return await agent.run(planning_task(spec))


async def generate_code(
    ctx: StepContext, *, spec: MigrationSpec, plan: MigrationPlan, filename: str
) -> tuple[AgentRun[GeneratedCode], int]:
    """Return the agent run plus how many times it submitted code to the gate."""
    agent = SparkEngineerAgent(
        ctx.factory.client_for(SparkEngineerAgent.key),
        plan=plan,
        spec=spec,
        gate_options=ctx.gate_options,
        max_tool_iterations=max(ctx.settings.llm_max_tool_iterations, 8),
    )
    async with _closing(agent):
        run = await agent.run(codegen_task(spec, plan, filename))
    run.output.filename = filename
    return run, max(agent.gate_calls, 1)


def verify_gate(
    code: GeneratedCode, *, gate_options: GateOptions, gate_iterations: int
) -> CodeGenResult:
    """Re-run the static gate outside the agent.

    The agent already ran this gate as a tool and told us it passed. That claim
    is not evidence — it is a claim made by the thing being judged. This is the
    independent re-check, and it is the value the rest of the system builds on.
    """
    report: StaticAnalysisReport = analyze_generated_code(code.content, gate_options)
    log.info(
        "gate.verified",
        passed=report.passed,
        findings=len(report.findings),
        errors=len(report.errors),
        agent_iterations=gate_iterations,
    )
    return CodeGenResult(code=code, static_analysis=report, gate_iterations=gate_iterations)


def write_artifacts(
    artifact_dir: Path,
    record: MigrationRecord,
    telemetry: Sequence[AgentTelemetry],
) -> None:
    """Write the full artifact set.

    Every path is derived from `migration_id`, and every write is a full
    overwrite, so a Temporal retry that re-executes this activity produces
    byte-identical output instead of duplicating anything (idempotency, R8).
    """
    artifact_dir.mkdir(parents=True, exist_ok=True)

    if record.spec is not None:
        (artifact_dir / "migration_spec.json").write_text(
            record.spec.model_dump_json(indent=2), encoding="utf-8"
        )
    if record.plan is not None:
        (artifact_dir / "migration_plan.json").write_text(
            record.plan.model_dump_json(indent=2), encoding="utf-8"
        )
    if record.codegen is not None:
        (artifact_dir / record.codegen.code.filename).write_text(
            record.codegen.code.content, encoding="utf-8"
        )
        (artifact_dir / "static_analysis.json").write_text(
            record.codegen.static_analysis.model_dump_json(indent=2), encoding="utf-8"
        )

    (artifact_dir / "migration_record.json").write_text(
        record.model_dump_json(indent=2), encoding="utf-8"
    )
    (artifact_dir / "agent_trace.json").write_text(
        json.dumps(
            {t.agent: t.model_dump(mode="json", exclude={"agent"}) for t in telemetry},
            indent=2,
        ),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Validation stage
# ---------------------------------------------------------------------------
def materialize(artifact_dir: Path, filename: str, content: str) -> Path:
    """Write generated source to disk so a subprocess can execute it.

    Idempotent by construction: same migration, same filename, full overwrite.
    """
    artifact_dir.mkdir(parents=True, exist_ok=True)
    path = artifact_dir / filename
    path.write_text(content, encoding="utf-8")
    return path


def execute_legacy(
    *, source_path: Path, input_dir: Path, output_dir: Path
) -> ExecutionResult:
    """Produce the reference output by running the legacy pipeline sandboxed."""
    return run_legacy_pipeline(
        module_path=source_path, input_dir=input_dir, output_dir=output_dir
    )


def execute_spark(
    *,
    module_path: Path,
    input_dir: Path,
    output_dir: Path,
    strategy: ExecutionStrategy | None,
) -> ExecutionResult:
    """Produce the candidate output by running the generated pipeline sandboxed."""
    return run_spark_pipeline(
        module_path=module_path,
        input_dir=input_dir,
        output_dir=output_dir,
        strategy=strategy,
    )


def diff_outputs(
    *, migration_id: str, reference_path: Path, candidate_path: Path, plan: MigrationPlan
) -> ValidationReport:
    """The verdict. No LLM anywhere on this path."""
    return compare_outputs(
        migration_id=migration_id,
        reference_path=reference_path,
        candidate_path=candidate_path,
        plan=plan,
    )


def resolve_output_root(directory: Path) -> Path:
    """Find the actual dataset inside a pipeline's output directory.

    pandas writes `revenue_by_country.csv` directly into the output directory;
    Spark writes a `revenue_by_country/` directory of part files into it. Both
    are one logical table, and the differ should be handed the same thing in
    either case rather than being taught about engine conventions.
    """
    if not directory.is_dir():
        return directory
    entries = [p for p in directory.iterdir() if not p.name.startswith(("_", "."))]
    data_files = [
        p
        for p in entries
        if p.is_file() and p.suffix.lower() in {".csv", ".parquet", ".json"}
    ]
    subdirs = [p for p in entries if p.is_dir()]
    if len(data_files) == 1 and not subdirs:
        return data_files[0]
    if len(subdirs) == 1 and not data_files:
        return subdirs[0]
    return directory


async def generate_test_suite(
    ctx: StepContext, *, spec: MigrationSpec, plan: MigrationPlan, code: GeneratedCode,
    filename: str,
) -> tuple[AgentRun[GeneratedTests], int]:
    """Ask the Testing agent for a pytest suite, returning its gate submissions too."""
    agent = TestingAgent(
        ctx.factory.client_for(TestingAgent.key),
        spec=spec,
        plan=plan,
        code_content=code.content,
        max_tool_iterations=max(ctx.settings.llm_max_tool_iterations, 8),
    )
    async with _closing(agent):
        run = await agent.run(testing_task(spec, plan, filename))
    run.output.filename = filename
    return run, max(agent.gate_calls, 1)


def verify_test_gate(tests: GeneratedTests) -> StaticAnalysisReport:
    """Re-run the test gate outside the agent, exactly as for pipeline code."""
    report = analyze_generated_code(tests.content, GateOptions.for_tests())
    log.info(
        "test_gate.verified",
        passed=report.passed,
        findings=len(report.findings),
        tests=len(tests.test_names),
    )
    return report


def execute_test_suite(
    *,
    tests: GeneratedTests,
    pipeline_path: Path,
    input_dir: Path,
    strategy: ExecutionStrategy | None,
) -> TestRunResult:
    """Run the generated suite against the generated pipeline, sandboxed."""
    return run_generated_tests(
        test_source=tests.content,
        test_filename=tests.filename,
        pipeline_path=pipeline_path,
        input_dir=input_dir,
        spark_conf=spark_conf_from(strategy),
    )


async def diagnose_failure(
    ctx: StepContext, *, report: ValidationReport, plan: MigrationPlan
) -> AgentRun[ValidationDiagnosis]:
    """Ask the Validation agent why the outputs disagree.

    Only ever called on a FAIL; the agent's constructor refuses anything else.
    """
    agent = ValidationAgent(
        ctx.factory.client_for(ValidationAgent.key),
        report=report,
        plan=plan,
        max_tool_iterations=ctx.settings.llm_max_tool_iterations,
    )
    async with _closing(agent):
        return await agent.run(diagnosis_task(report, plan))


# ---------------------------------------------------------------------------
# Repair loop
# ---------------------------------------------------------------------------
async def propose_repair(
    ctx: StepContext,
    *,
    report: ValidationReport,
    diagnosis: ValidationDiagnosis | None,
    plan: MigrationPlan,
    code: GeneratedCode,
    history: list[RepairAttempt],
    attempt: int,
    max_attempts: int,
    past: MigrationHistory | None = None,
) -> tuple[AgentRun[RepairProposal], int]:
    """Ask the Repair agent for a different implementation.

    The agent is shown the history so a repeat is a choice rather than an
    accident; `RepairLedger` is what makes it a rejected one.
    """
    agent = RepairAgent(
        ctx.factory.client_for(RepairAgent.key),
        report=report,
        diagnosis=diagnosis,
        plan=plan,
        code=code,
        history=history,
        gate_options=ctx.gate_options,
        max_tool_iterations=max(ctx.settings.llm_max_tool_iterations, 8),
    )
    async with _closing(agent):
        run = await agent.run(
            repair_task(report, diagnosis, history, attempt, max_attempts)
        )
    run.output.code.filename = code.filename
    return run, max(agent.gate_calls, 1)


# ---------------------------------------------------------------------------
# Optimisation
# ---------------------------------------------------------------------------
def benchmark(
    *,
    label: str,
    module_path: Path,
    input_dir: Path,
    output_dir: Path,
    strategy: ExecutionStrategy | None,
    runs: int = DEFAULT_RUNS,
    warmups: int = DEFAULT_WARMUPS,
) -> BenchmarkResult:
    """Time a pipeline repeatedly, discarding warm-ups. No LLM on this path."""
    return benchmark_pipeline(
        label=label,
        module_path=module_path,
        input_dir=input_dir,
        output_dir=output_dir,
        strategy=strategy,
        runs=runs,
        warmups=warmups,
    )


def analyze_optimization_opportunities(
    code: GeneratedCode, input_dir: Path
) -> PlanAnalysis:
    """Structural analysis, grounded in measured input sizes."""
    return analyze_plan(code.content, profile_directory(input_dir))


async def propose_optimization(
    ctx: StepContext,
    *,
    plan: MigrationPlan,
    code: GeneratedCode,
    baseline: BenchmarkResult,
    analysis: PlanAnalysis,
    history: list[OptimizationAttempt],
    attempt: int,
    max_attempts: int,
    past: MigrationHistory | None = None,
) -> tuple[AgentRun[OptimizationProposal], int]:
    """Ask the Optimizer agent for one grounded, measurable change."""
    agent = OptimizerAgent(
        ctx.factory.client_for(OptimizerAgent.key),
        plan=plan,
        code=code,
        baseline=baseline,
        analysis=analysis,
        history=history,
        past=past or MigrationHistory(),
        gate_options=ctx.gate_options,
        max_tool_iterations=max(ctx.settings.llm_max_tool_iterations, 8),
    )
    async with _closing(agent):
        run = await agent.run(
            optimization_task(baseline, history, attempt, max_attempts)
        )
    if run.output.code is not None:
        run.output.code.filename = code.filename
    return run, max(agent.gate_calls, 1)


# ---------------------------------------------------------------------------
# Delivery
# ---------------------------------------------------------------------------
async def propose_pr_narrative(
    ctx: StepContext,
    *,
    record: MigrationRecord,
    decision: DeliveryDecision,
    revision_of: ClaimAudit | None = None,
) -> tuple[AgentRun[PullRequestNarrative], int]:
    """Ask the Delivery agent for the reviewer-facing prose.

    `revision_of` carries a failed audit back to the agent so a rewrite is
    directed at the specific figures that were wrong, rather than being a second
    blind attempt at the same paragraph.
    """
    agent = DeliveryAgent(
        ctx.factory.client_for(DeliveryAgent.key),
        record=record,
        decision=decision,
        max_tool_iterations=max(ctx.settings.llm_max_tool_iterations, 8),
    )
    task = revision_task(revision_of) if revision_of is not None else delivery_task(
        record, decision
    )
    async with _closing(agent):
        run = await agent.run(task)
    return run, max(agent.audit_calls, 1)


def build_file_changes(record: MigrationRecord, *, directory: str) -> list[FileChange]:
    """The files this migration puts on the branch.

    The generated pipeline, its generated test suite, and a machine-readable
    record of how the code was arrived at. The last one matters more than it
    looks: without it a reviewer six months from now has a PySpark module and no
    way to tell which legacy file it came from or what was checked.
    """
    if record.codegen is None:
        return []

    stem = f"{directory.rstrip('/')}/{record.migration_id}"
    code = record.codegen.code
    changes = [
        FileChange(
            path=f"{stem}/{code.filename}",
            content=code.content,
            message=f"Add migrated PySpark pipeline for {record.source_path}",
        )
    ]

    tests = record.validation.tests if record.validation is not None else None
    if tests is not None:
        changes.append(
            FileChange(
                path=f"{stem}/{tests.filename}",
                content=tests.content,
                message=f"Add generated test suite for {record.migration_id}",
            )
        )

    changes.append(
        FileChange(
            path=f"{stem}/migration_record.json",
            content=record.model_dump_json(indent=2),
            message=f"Record how {record.migration_id} was produced and verified",
        )
    )
    return changes


def deliver_pull_request(
    client: GitHubClient,
    *,
    record: MigrationRecord,
    decision: DeliveryDecision,
    narrative: PullRequestNarrative,
    files: list[FileChange],
    branch: str,
    base: str | None = None,
) -> DeliveryOutcome:
    """Branch, commit, open, label. Every step is lookup-then-create.

    No LLM on this path: the narrative arrives already audited, and everything
    else is rendered from the record.
    """
    outcome = DeliveryOutcome(decision=decision)
    base_branch = base or client.default_branch()

    branch_ref = client.ensure_branch(branch, base=base_branch)
    outcome.branch = branch_ref

    for change in files:
        client.ensure_file(change, branch=branch)
        outcome.files.append(change.path)

    pull = client.ensure_pull_request(
        head=branch,
        base=base_branch,
        title=narrative.title,
        body=render_pr_body(record, decision, narrative),
        draft=decision.draft,
    )
    if decision.labels:
        pull.labels = client.add_labels(pull.number, decision.labels)
    outcome.pull_request = pull
    log.info(
        "delivery.done",
        migration_id=record.migration_id,
        number=pull.number,
        draft=pull.draft,
        created=pull.created,
    )
    return outcome
