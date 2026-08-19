"""Temporal Activities: every side effect in the system lives behind one of these.

Activities are instance methods on a class rather than module-level functions so
the worker can inject `Settings` and a model-client factory once at startup.
That also makes them directly unit-testable with
`temporalio.testing.ActivityEnvironment`, with no server and no worker.

Each activity is a thin wrapper over `pipeline.steps`. The rule is that an
activity may do I/O and translate errors, but must not make lifecycle decisions
— those belong to the workflow, where they are durable.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path

from temporalio import activity

from etl_migrator.config import Settings
from etl_migrator.domain.artifacts import MigrationRecord
from etl_migrator.domain.delivery import DeliveryOutcome
from etl_migrator.domain.delivery_policy import audit_numeric_claims
from etl_migrator.domain.messages import (
    BenchmarkInput,
    CodegenInput,
    CodegenOutput,
    DeliverPullRequestInput,
    DiagnosisInput,
    DiagnosisOutput,
    DiscoveryInput,
    DiscoveryOutput,
    LegacyExecutionInput,
    OptimizationProposalInput,
    OptimizationProposalOutput,
    OutputValidationInput,
    PersistInput,
    PersistOutput,
    PlanningInput,
    PlanningOutput,
    PullRequestNarrativeInput,
    PullRequestNarrativeOutput,
    RepairProposalInput,
    RepairProposalOutput,
    SparkExecutionInput,
    StaticAnalysisInput,
    StaticAnalysisOutput,
    TestExecutionInput,
    TestGenerationInput,
    TestGenerationOutput,
)
from etl_migrator.domain.optimization import BenchmarkResult
from etl_migrator.domain.validation import ExecutionResult, TestRunResult, ValidationReport
from etl_migrator.github.client import GitHubClient
from etl_migrator.knowledge.history import load_history
from etl_migrator.llm.factory import ModelClientFactory, build_model_client_factory
from etl_migrator.observability import get_logger, get_metrics
from etl_migrator.pipeline import steps
from etl_migrator.pipeline.steps import StepContext
from etl_migrator.tools.code_gate import GateOptions

log = get_logger(__name__)


class MigrationActivities:
    """Activity implementations bound to one worker's configuration."""

    def __init__(
        self,
        settings: Settings,
        *,
        factory: ModelClientFactory | None = None,
        gate_options: GateOptions | None = None,
    ) -> None:
        self._settings = settings
        self._factory = factory
        self._gate_options = gate_options or GateOptions()

    # -- helpers -----------------------------------------------------------
    def _context(self, scenario: str) -> StepContext:
        """Build a step context for one activity execution.

        A model client is created per activity rather than shared: the scripted
        provider replays a per-agent script and must start at turn zero on every
        attempt, or a Temporal retry would resume mid-script and replay the wrong
        response.
        """
        factory = self._factory or build_model_client_factory(self._settings, scenario=scenario)
        return StepContext(
            settings=self._settings, factory=factory, gate_options=self._gate_options
        )

    def artifact_dir(self, migration_id: str) -> Path:
        return self._settings.workspace_dir / migration_id

    @staticmethod
    def _log_start(name: str, migration_id: str) -> None:
        info = activity.info() if activity.in_activity() else None
        log.info(
            "activity.start",
            activity=name,
            migration_id=migration_id,
            attempt=info.attempt if info else 1,
        )

    # -- activities --------------------------------------------------------
    @activity.defn(name="analyze_legacy_pipeline")
    async def analyze_legacy_pipeline(self, params: DiscoveryInput) -> DiscoveryOutput:
        """Inspect the legacy source and profile its inputs into a `MigrationSpec`."""
        self._log_start("analyze_legacy_pipeline", params.migration_id)
        run = await steps.discover(
            self._context(params.scenario),
            source_path=Path(params.source_path),
            input_dir=Path(params.input_dir),
        )
        return DiscoveryOutput(spec=run.output, telemetry=steps.telemetry_of(run))

    @activity.defn(name="generate_migration_plan")
    async def generate_migration_plan(self, params: PlanningInput) -> PlanningOutput:
        """Map every transformation onto Spark and declare the semantic differences."""
        self._log_start("generate_migration_plan", params.migration_id)
        profiles = steps.profile_inputs(Path(params.input_dir))
        run = await steps.plan_migration(
            self._context(params.scenario), spec=params.spec, profiles=profiles
        )
        return PlanningOutput(plan=run.output, telemetry=steps.telemetry_of(run))

    @activity.defn(name="generate_spark_code")
    async def generate_spark_code(self, params: CodegenInput) -> CodegenOutput:
        """Implement the approved plan as an executable PySpark module."""
        self._log_start("generate_spark_code", params.migration_id)
        run, gate_iterations = await steps.generate_code(
            self._context(params.scenario),
            spec=params.spec,
            plan=params.plan,
            filename=params.output_filename,
        )
        return CodegenOutput(
            code=run.output,
            telemetry=steps.telemetry_of(run, gate_iterations=gate_iterations),
            gate_iterations=gate_iterations,
        )

    @activity.defn(name="run_static_analysis")
    async def run_static_analysis(self, params: StaticAnalysisInput) -> StaticAnalysisOutput:
        """Re-run the static gate independently of the agent that wrote the code.

        Returns the verdict as data. A failing gate is an outcome the workflow
        decides about, not an exception for a retry policy to guess at.
        """
        self._log_start("run_static_analysis", params.migration_id)
        result = steps.verify_gate(
            params.code,
            gate_options=self._gate_options,
            gate_iterations=params.gate_iterations,
        )
        return StaticAnalysisOutput(result=result)

    @activity.defn(name="persist_artifacts")
    async def persist_artifacts(self, params: PersistInput) -> PersistOutput:
        """Write spec, plan, generated code, gate report, record and agent trace.

        Idempotent: every path derives from `migration_id` and every write is a
        full overwrite, so a retried attempt produces byte-identical output.
        """
        self._log_start("persist_artifacts", params.migration_id)
        record = MigrationRecord.model_validate_json(params.record_json)
        directory = self.artifact_dir(params.migration_id)
        steps.write_artifacts(directory, record, params.telemetry)
        written = sorted(p.name for p in directory.iterdir() if p.is_file())

        # The one place the durable path sees a finished record, so the one
        # place it can emit metrics. `observe` is deduplicated per migration
        # within the process, which makes a retry of this activity harmless.
        metrics = get_metrics()
        metrics.observe(record)
        for entry in params.telemetry:
            metrics.observe_telemetry(entry)

        log.info("artifacts.written", migration_id=params.migration_id, files=len(written))
        return PersistOutput(artifact_dir=str(directory), files_written=written)

    # -- registration ------------------------------------------------------
    def all(self) -> list[Callable[..., object]]:
        """Every activity, in the shape `Worker(activities=...)` expects."""
        return [
            self.analyze_legacy_pipeline,
            self.generate_migration_plan,
            self.generate_spark_code,
            self.run_static_analysis,
            self.persist_artifacts,
        ]


class ValidationActivities:
    """Activities for the validation stage.

    Split from `MigrationActivities` because they have a different character:
    these are long, CPU-heavy and touch the sandbox, whereas the generation
    activities are mostly waiting on a model. Keeping them apart makes it
    possible to run them on a separate task queue with different worker sizing
    later, without changing a workflow.

    Every blocking call goes through `asyncio.to_thread`. An activity that
    blocks the event loop stops its worker from heartbeating or accepting other
    work, and a Spark run blocks for minutes.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        factory: ModelClientFactory | None = None,
        gate_options: GateOptions | None = None,
    ) -> None:
        self._settings = settings
        self._factory = factory
        self._gate_options = gate_options or GateOptions()

    def _context(self, scenario: str) -> StepContext:
        factory = self._factory or build_model_client_factory(self._settings, scenario=scenario)
        return StepContext(
            settings=self._settings, factory=factory, gate_options=self._gate_options
        )

    def artifact_dir(self, migration_id: str) -> Path:
        return self._settings.workspace_dir / migration_id

    @activity.defn(name="run_legacy_pipeline")
    async def run_legacy_pipeline(self, params: LegacyExecutionInput) -> ExecutionResult:
        """Execute the legacy pipeline sandboxed to produce the reference output."""
        MigrationActivities._log_start("run_legacy_pipeline", params.migration_id)
        output_dir = self.artifact_dir(params.migration_id) / "reference_output"
        result = await asyncio.to_thread(
            steps.execute_legacy,
            source_path=Path(params.source_path),
            input_dir=Path(params.input_dir),
            output_dir=output_dir,
        )
        if result.succeeded:
            result.output_path = str(steps.resolve_output_root(output_dir))
        return result

    @activity.defn(name="run_spark_pipeline")
    async def run_spark_pipeline(self, params: SparkExecutionInput) -> ExecutionResult:
        """Execute the generated PySpark pipeline sandboxed to produce the candidate."""
        MigrationActivities._log_start("run_spark_pipeline", params.migration_id)
        artifacts = self.artifact_dir(params.migration_id)
        module_path = await asyncio.to_thread(
            steps.materialize, artifacts, params.code.filename, params.code.content
        )
        output_dir = artifacts / params.output_name
        result = await asyncio.to_thread(
            steps.execute_spark,
            module_path=module_path,
            input_dir=Path(params.input_dir),
            output_dir=output_dir,
            strategy=params.execution_strategy,
        )
        if result.succeeded:
            result.output_path = str(steps.resolve_output_root(output_dir))
        return result

    @activity.defn(name="validate_outputs")
    async def validate_outputs(self, params: OutputValidationInput) -> ValidationReport:
        """Compare the two executed outputs. The only thing that may say PASS."""
        MigrationActivities._log_start("validate_outputs", params.migration_id)
        return await asyncio.to_thread(
            steps.diff_outputs,
            migration_id=params.migration_id,
            reference_path=Path(params.reference_path),
            candidate_path=Path(params.candidate_path),
            plan=params.plan,
        )

    @activity.defn(name="generate_tests")
    async def generate_tests(self, params: TestGenerationInput) -> TestGenerationOutput:
        """Ask the Testing agent for a pytest suite, then re-gate it independently."""
        MigrationActivities._log_start("generate_tests", params.migration_id)
        run, gate_iterations = await steps.generate_test_suite(
            self._context(params.scenario),
            spec=params.spec,
            plan=params.plan,
            code=params.code,
            filename=params.filename,
        )
        report = steps.verify_test_gate(run.output)
        return TestGenerationOutput(
            tests=run.output,
            static_analysis=report,
            telemetry=steps.telemetry_of(run, gate_iterations=gate_iterations),
            gate_iterations=gate_iterations,
        )

    @activity.defn(name="run_tests")
    async def run_tests(self, params: TestExecutionInput) -> TestRunResult:
        """Execute the generated suite against the generated pipeline, sandboxed."""
        MigrationActivities._log_start("run_tests", params.migration_id)
        artifacts = self.artifact_dir(params.migration_id)
        pipeline_path = await asyncio.to_thread(
            steps.materialize, artifacts, params.code.filename, params.code.content
        )
        await asyncio.to_thread(
            steps.materialize, artifacts, params.tests.filename, params.tests.content
        )
        return await asyncio.to_thread(
            steps.execute_test_suite,
            tests=params.tests,
            pipeline_path=pipeline_path,
            input_dir=Path(params.input_dir),
            strategy=params.execution_strategy,
        )

    @activity.defn(name="diagnose_validation_failure")
    async def diagnose_validation_failure(self, params: DiagnosisInput) -> DiagnosisOutput:
        """Ask the Validation agent why the outputs disagree.

        Reached only when the differ already said FAIL; the agent refuses to be
        constructed on a passing report.
        """
        MigrationActivities._log_start("diagnose_validation_failure", params.migration_id)
        run = await steps.diagnose_failure(
            self._context(params.scenario), report=params.report, plan=params.plan
        )
        return DiagnosisOutput(diagnosis=run.output, telemetry=steps.telemetry_of(run))

    @activity.defn(name="propose_repair")
    async def propose_repair(self, params: RepairProposalInput) -> RepairProposalOutput:
        """Ask the Repair agent for a corrected module.

        Only proposes. Whether the proposal is *admissible* is decided by the
        workflow through `RepairLedger`, because that decision must be durable
        and must not cost an LLM call.
        """
        MigrationActivities._log_start("propose_repair", params.migration_id)
        # The corpus, read inside the activity: it is a filesystem read, which
        # is exactly the kind of thing a workflow may not do and an activity
        # exists for.
        past = await asyncio.to_thread(load_history, self._settings.workspace_dir)
        run, gate_iterations = await steps.propose_repair(
            self._context(params.scenario),
            report=params.report,
            diagnosis=params.diagnosis,
            plan=params.plan,
            code=params.code,
            history=params.history,
            attempt=params.attempt,
            max_attempts=params.max_attempts,
            past=past,
        )
        return RepairProposalOutput(
            proposal=run.output,
            telemetry=steps.telemetry_of(run, gate_iterations=gate_iterations),
            gate_iterations=gate_iterations,
        )

    @activity.defn(name="benchmark_spark")
    async def benchmark_spark(self, params: BenchmarkInput) -> BenchmarkResult:
        """Time a pipeline repeatedly under one configuration.

        Deterministic in intent but expensive in practice: it executes Spark
        `runs + warmups` times, which is why it is its own activity with its own
        timeout rather than folded into the optimisation agent's call.
        """
        MigrationActivities._log_start("benchmark_spark", params.migration_id)
        artifacts = self.artifact_dir(params.migration_id)
        module_path = await asyncio.to_thread(
            steps.materialize,
            artifacts,
            f"{params.label}_{params.code.filename}",
            params.code.content,
        )
        return await asyncio.to_thread(
            steps.benchmark,
            label=params.label,
            module_path=module_path,
            input_dir=Path(params.input_dir),
            output_dir=artifacts / f"benchmark_{params.label}",
            strategy=params.execution_strategy,
            runs=params.runs,
            warmups=params.warmups,
        )

    @activity.defn(name="propose_optimization")
    async def propose_optimization(
        self, params: OptimizationProposalInput
    ) -> OptimizationProposalOutput:
        """Ask the Optimizer agent for one grounded change.

        Whether the change is *kept* is decided later by measurement, in the
        workflow, via `evaluate_optimization`.
        """
        MigrationActivities._log_start("propose_optimization", params.migration_id)
        analysis = await asyncio.to_thread(
            steps.analyze_optimization_opportunities, params.code, Path(params.input_dir)
        )
        past = await asyncio.to_thread(load_history, self._settings.workspace_dir)
        run, gate_iterations = await steps.propose_optimization(
            self._context(params.scenario),
            plan=params.plan,
            code=params.code,
            baseline=params.baseline,
            analysis=analysis,
            history=params.history,
            attempt=params.attempt,
            max_attempts=params.max_attempts,
            past=past,
        )
        return OptimizationProposalOutput(
            proposal=run.output,
            telemetry=steps.telemetry_of(run, gate_iterations=gate_iterations),
            gate_iterations=gate_iterations,
        )

    # -- trust tiers -------------------------------------------------------
    #
    # These two lists are the seam the NetworkPolicy in k8s/ cuts along. One
    # executes untrusted generated code, the other talks to an LLM provider, and
    # co-locating them means the untrusted code inherits the pod's internet
    # egress. A test keeps the partition honest by checking that nothing in
    # `execution_activities` reaches for a model client.

    def execution_activities(self) -> list[Callable[..., object]]:
        """Activities that run untrusted code, or read what it produced.

        None of these needs a network beyond Temporal itself.
        """
        return [
            self.run_legacy_pipeline,
            self.run_spark_pipeline,
            self.validate_outputs,
            self.run_tests,
            self.benchmark_spark,
        ]

    def reasoning_activities(self) -> list[Callable[..., object]]:
        """Activities that call a model. None of these executes generated code."""
        return [
            self.generate_tests,
            self.diagnose_validation_failure,
            self.propose_repair,
            self.propose_optimization,
        ]

    def all(self) -> list[Callable[..., object]]:
        """Both tiers, for a single-worker deployment such as compose or a laptop."""
        return [*self.execution_activities(), *self.reasoning_activities()]


class DeliveryActivities:
    """Activities for the pull request stage.

    Separate again, and for a reason that matters operationally: these are the
    only activities that hold a GitHub credential. Keeping them in their own
    class means a deployment can run them on their own task queue, on workers
    that have the token, while the Spark and agent workers never see it.

    The client is built once per worker and injected, so a test can substitute
    `InMemoryGitHub` without any of this code knowing.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        github: GitHubClient | None = None,
        factory: ModelClientFactory | None = None,
    ) -> None:
        self._settings = settings
        self._github = github
        self._factory = factory

    def _context(self, scenario: str) -> StepContext:
        factory = self._factory or build_model_client_factory(self._settings, scenario=scenario)
        return StepContext(settings=self._settings, factory=factory)

    #: Why delivery was not attempted. Phrased for a reader of the record, and
    #: identical in meaning to the local pipeline's version of the same skip.
    NOT_CONFIGURED = (
        "no GitHub client configured; set ETLM_GITHUB_TOKEN and "
        "ETLM_GITHUB_REPOSITORY on the worker, or pass deliver_enabled=false "
        "to stop asking"
    )

    @activity.defn(name="propose_pr_narrative")
    async def propose_pr_narrative(
        self, params: PullRequestNarrativeInput
    ) -> PullRequestNarrativeOutput:
        """Ask the Delivery agent for the reviewer-facing prose, then audit it.

        The audit runs *here*, in the activity, against the same record the
        workflow holds — so the workflow receives a verdict it did not have to
        compute and cannot disagree with. What the workflow decides is only what
        to do about a failure.
        """
        MigrationActivities._log_start("propose_pr_narrative", params.migration_id)
        run, audit_calls = await steps.propose_pr_narrative(
            self._context(params.scenario),
            record=params.record,
            decision=params.decision,
            revision_of=params.previous_audit,
        )
        audit = audit_numeric_claims(run.output, params.record)
        return PullRequestNarrativeOutput(
            narrative=run.output,
            audit=audit,
            telemetry=steps.telemetry_of(run, gate_iterations=audit_calls),
        )

    @activity.defn(name="deliver_pull_request")
    async def deliver_pull_request(self, params: DeliverPullRequestInput) -> DeliveryOutcome:
        """Branch, commit, open, label — every step lookup-then-create.

        Safe to retry by construction, which is the only reason it can be an
        activity at all: Temporal may run it again after the side effects have
        already landed.
        """
        MigrationActivities._log_start("deliver_pull_request", params.migration_id)
        if self._github is None:
            # A missing token is a configuration gap, not a failed migration.
            # Raising here would fail the activity, fail DeliveryWorkflow, and
            # fail a migration whose Spark output was already validated as
            # correct -- discarding a good result over an absent credential.
            # The local pipeline has always skipped instead; this is the
            # durable path agreeing with it.
            log.info("delivery.skipped", reason=self.NOT_CONFIGURED)
            return DeliveryOutcome(skipped_reason=self.NOT_CONFIGURED)

        client = self._github
        files = steps.build_file_changes(params.record, directory=params.directory)
        return await asyncio.to_thread(
            steps.deliver_pull_request,
            client,
            record=params.record,
            decision=params.decision,
            narrative=params.narrative,
            files=files,
            branch=params.branch,
            base=params.base,
        )

    def all(self) -> list[Callable[..., object]]:
        return [self.propose_pr_narrative, self.deliver_pull_request]
