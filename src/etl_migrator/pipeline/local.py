"""Sequential, in-process orchestration — the same migration without a server.

This exists alongside `ETLMigrationWorkflow` on purpose, and it is not a toy:

* it is what you run while developing an agent, with no Temporal to stand up;
* it keeps the whole system testable in CI, where no server is available;
* running the two side by side is how you notice if the durable path has
  quietly diverged from the direct one.

The two orchestrators share `pipeline.steps` (the work) and `domain.lifecycle`
(the state transitions and policy), so what differs between them is exactly
what *should* differ: durability, retries, and how the approval gate is
satisfied. Here, approval is resolved by an injected callback and the process
must stay alive; in Temporal it is a signal that survives a reboot.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from etl_migrator.config import Settings
from etl_migrator.domain import lifecycle
from etl_migrator.domain.artifacts import MigrationRecord
from etl_migrator.domain.delivery import (
    ClaimAudit,
    DeliveryOutcome,
    PullRequestNarrative,
)
from etl_migrator.domain.delivery_policy import audit_numeric_claims, decide_delivery
from etl_migrator.domain.enums import MigrationStage
from etl_migrator.domain.errors import StaticGateError
from etl_migrator.domain.history import MigrationHistory
from etl_migrator.domain.messages import AgentTelemetry, ApprovalDecision
from etl_migrator.domain.optimization import (
    DEFAULT_MIN_SPEEDUP,
    DEFAULT_RUNS,
    BenchmarkComparison,
    OptimizationAttempt,
    OptimizationOutcome,
    evaluate_optimization,
)
from etl_migrator.domain.repair import (
    RepairAttempt,
    RepairLedger,
    RepairOutcome,
    code_fingerprint,
)
from etl_migrator.domain.validation import CheckResult, ValidationOutcome, ValidationReport
from etl_migrator.github.client import GitHubClient
from etl_migrator.ids import new_migration_id
from etl_migrator.knowledge.history import load_history
from etl_migrator.llm.factory import ModelClientFactory
from etl_migrator.observability import (
    get_logger,
    get_metrics,
    migration_context,
    stage_context,
)
from etl_migrator.pipeline import steps
from etl_migrator.pipeline.steps import StepContext
from etl_migrator.tools.code_gate import GateOptions

log = get_logger(__name__)

#: Resolves the approval gate. Returning None means "nobody answered".
ApprovalResolver = Callable[[MigrationRecord], ApprovalDecision | None]


def auto_approve(actor: str = "local-auto-approve") -> ApprovalResolver:
    """Approve automatically, loudly.

    Convenient for local runs and CI. It logs a warning every time because an
    approval gate that silently approves itself is worse than no gate at all.
    """

    def resolver(record: MigrationRecord) -> ApprovalDecision:
        log.warning(
            "approval.auto_granted",
            risk=record.risk.value,
            note="local pipeline only; the Temporal workflow requires a real signal",
        )
        return ApprovalDecision(
            approved=True, actor=actor, reason="auto-approved by the local pipeline"
        )

    return resolver


def require_manual_approval() -> ApprovalResolver:
    """Refuse to proceed. The default, so a HIGH-risk plan stops by default."""

    def resolver(record: MigrationRecord) -> None:
        return None

    return resolver


@dataclass
class MigrationRequest:
    """Inputs for one local migration run."""

    source_path: Path
    input_dir: Path
    migration_id: str | None = None
    output_filename: str | None = None
    scenario: str = "customer_pipeline"

    def __post_init__(self) -> None:
        self.source_path = self.source_path.resolve()
        self.input_dir = self.input_dir.resolve()
        self.migration_id = self.migration_id or new_migration_id()
        self.output_filename = self.output_filename or f"{self.source_path.stem}_spark.py"


class LocalMigrationPipeline:
    """Runs discovery → planning → approval → code generation → static gate."""

    def __init__(
        self,
        settings: Settings,
        factory: ModelClientFactory,
        *,
        gate_options: GateOptions | None = None,
        approval: ApprovalResolver | None = None,
        run_validation: bool = True,
        run_generated_tests: bool = True,
        run_repair: bool = True,
        max_repair_attempts: int = 3,
        run_optimization: bool = True,
        max_optimization_attempts: int = 2,
        benchmark_runs: int = DEFAULT_RUNS,
        min_speedup: float = DEFAULT_MIN_SPEEDUP,
        github: GitHubClient | None = None,
        run_delivery: bool = True,
        delivery_branch_prefix: str = "etl-migration",
        delivery_directory: str = "migrations",
        max_narrative_revisions: int = 2,
        learn_from_history: bool = True,
    ) -> None:
        self.settings = settings
        self.ctx = StepContext(
            settings=settings, factory=factory, gate_options=gate_options or GateOptions()
        )
        self.approval = approval or auto_approve()
        self.run_validation = run_validation
        self.run_generated_tests = run_generated_tests
        self.run_repair = run_repair
        self.max_repair_attempts = max_repair_attempts
        self.run_optimization = run_optimization
        self.max_optimization_attempts = max_optimization_attempts
        self.benchmark_runs = benchmark_runs
        self.min_speedup = min_speedup
        self.github = github
        self.run_delivery = run_delivery
        self.delivery_branch_prefix = delivery_branch_prefix
        self.delivery_directory = delivery_directory
        self.max_narrative_revisions = max_narrative_revisions
        self.learn_from_history = learn_from_history

    @staticmethod
    def _now() -> datetime:
        return datetime.now(UTC)

    def _past(self) -> MigrationHistory:
        """What previous migrations established.

        Recomputed from the persisted artifacts rather than held as state, so it
        can never drift from what actually happened. Empty when disabled or when
        nothing has run yet, and both agents treat empty as "nothing known"
        rather than as evidence of anything.
        """
        if not self.learn_from_history:
            return MigrationHistory()
        return load_history(self.settings.workspace_dir)

    async def run(self, request: MigrationRequest) -> MigrationRecord:
        assert request.migration_id is not None and request.output_filename is not None
        record = lifecycle.new_record(
            request.migration_id, str(request.source_path), self._now()
        )
        telemetry: list[AgentTelemetry] = []

        with migration_context(record.migration_id, source=request.source_path.name):
            log.info(
                "migration.start",
                source=str(request.source_path),
                input_dir=str(request.input_dir),
                provider=self.settings.llm_provider.value,
            )

            # --- discovery ------------------------------------------------
            with stage_context(MigrationStage.DISCOVERY.value):
                entry = lifecycle.begin_stage(record, MigrationStage.DISCOVERY, self._now())
                try:
                    discovery = await steps.discover(
                        self.ctx,
                        source_path=request.source_path,
                        input_dir=request.input_dir,
                    )
                except Exception as exc:
                    lifecycle.fail_stage(
                        record, entry, self._now(), reason=f"{type(exc).__name__}: {exc}"
                    )
                    self._persist(record, telemetry)
                    raise
                record.spec = discovery.output
                telemetry.append(steps.telemetry_of(discovery))
                lifecycle.complete_stage(
                    entry, self._now(), detail=f"tools={discovery.tools_used}"
                )

            # --- planning -------------------------------------------------
            with stage_context(MigrationStage.PLANNING.value):
                entry = lifecycle.begin_stage(record, MigrationStage.PLANNING, self._now())
                profiles = steps.profile_inputs(request.input_dir)
                try:
                    planning = await steps.plan_migration(
                        self.ctx, spec=discovery.output, profiles=profiles
                    )
                except Exception as exc:
                    lifecycle.fail_stage(
                        record, entry, self._now(), reason=f"{type(exc).__name__}: {exc}"
                    )
                    self._persist(record, telemetry)
                    raise
                plan = lifecycle.enforce_approval_policy(planning.output)
                record.plan = plan
                telemetry.append(steps.telemetry_of(planning))
                lifecycle.complete_stage(
                    entry, self._now(), detail=f"risk={plan.overall_risk.value}"
                )

            # --- approval -------------------------------------------------
            if plan.requires_human_approval and not self._resolve_approval(record):
                self._persist(record, telemetry)
                return record

            # --- code generation ------------------------------------------
            with stage_context(MigrationStage.CODE_GENERATION.value):
                entry = lifecycle.begin_stage(
                    record, MigrationStage.CODE_GENERATION, self._now()
                )
                try:
                    codegen, gate_iterations = await steps.generate_code(
                        self.ctx,
                        spec=discovery.output,
                        plan=plan,
                        filename=request.output_filename,
                    )
                except Exception as exc:
                    lifecycle.fail_stage(
                        record, entry, self._now(), reason=f"{type(exc).__name__}: {exc}"
                    )
                    self._persist(record, telemetry)
                    raise
                telemetry.append(
                    steps.telemetry_of(codegen, gate_iterations=gate_iterations)
                )
                lifecycle.complete_stage(
                    entry, self._now(), detail=f"gate_submissions={gate_iterations}"
                )

            # --- independent static gate ----------------------------------
            with stage_context(MigrationStage.STATIC_ANALYSIS.value):
                entry = lifecycle.begin_stage(
                    record, MigrationStage.STATIC_ANALYSIS, self._now()
                )
                result = steps.verify_gate(
                    codegen.output,
                    gate_options=self.ctx.gate_options,
                    gate_iterations=gate_iterations,
                )
                record.codegen = result
                if not result.static_analysis.passed:
                    reason = (
                        "generated code failed the independent static gate:\n"
                        + result.static_analysis.render()
                    )
                    lifecycle.fail_stage(record, entry, self._now(), reason=reason)
                    self._persist(record, telemetry)
                    raise StaticGateError(reason)
                lifecycle.complete_stage(
                    entry,
                    self._now(),
                    detail=f"findings={len(result.static_analysis.findings)}",
                )

            # --- validation -----------------------------------------------
            if self.run_validation:
                with stage_context(MigrationStage.VALIDATION.value):
                    entry = lifecycle.begin_stage(
                        record, MigrationStage.VALIDATION, self._now()
                    )
                    outcome = await self._validate(
                        request, discovery.output, plan, result.code, telemetry
                    )
                    record.validation = outcome
                    if outcome.passed:
                        lifecycle.complete_stage(
                            entry,
                            self._now(),
                            detail=f"status={outcome.report.status.value}",
                        )
                        await self._optimize(
                            request, record, discovery.output, plan, result.code, telemetry
                        )
                    else:
                        lifecycle.fail_stage(
                            record,
                            entry,
                            self._now(),
                            reason=f"validation {outcome.report.status.value}: "
                            + (
                                "; ".join(
                                    d.render() for d in outcome.report.differences[:5]
                                )
                                or outcome.report.error
                                or "no differences recorded"
                            ),
                        )

                # --- autonomous repair --------------------------------------
                if (
                    not outcome.passed
                    and self.run_repair
                    and outcome.report.error is None
                ):
                    with stage_context(MigrationStage.REPAIR.value):
                        entry = lifecycle.begin_stage(
                            record, MigrationStage.REPAIR, self._now()
                        )
                        repair = await self._repair(
                            request,
                            discovery.output,
                            plan,
                            result.code,
                            outcome,
                            telemetry,
                        )
                        record.repair = repair
                        if repair.succeeded and repair.repaired_code is not None:
                            record.codegen = result.model_copy(
                                update={"code": repair.repaired_code}
                            )
                            if repair.final_report is not None:
                                record.validation = outcome.model_copy(
                                    update={
                                        "report": repair.final_report,
                                        "diagnosis": None,
                                    }
                                )
                            record.failed = False
                            record.failure_reason = None
                            lifecycle.complete_stage(
                                entry,
                                self._now(),
                                detail=f"repaired after {repair.attempts_used} attempt(s)",
                            )
                            await self._optimize(
                                request,
                                record,
                                discovery.output,
                                plan,
                                repair.repaired_code,
                                telemetry,
                            )
                        else:
                            best = repair.best_attempt
                            closest = (
                                f"; closest attempt {best.attempt} left "
                                f"{best.differences} differences"
                                if best is not None
                                else ""
                            )
                            lifecycle.fail_stage(
                                record,
                                entry,
                                self._now(),
                                reason=(
                                    f"repair exhausted after {repair.attempts_used} "
                                    f"attempt(s){closest}. Human intervention required."
                                ),
                            )

            # Delivery runs whatever happened above. A migration that failed
            # still has work worth showing a human — as a labelled draft, never
            # as something to approve — and `decide_delivery` is what tells the
            # two apart.
            await self._deliver(record, telemetry)

            self._persist(record, telemetry)
            log.info("migration.complete", summary=lifecycle.summarise(record))
            return record

    # -- internals ---------------------------------------------------------
    async def _deliver(
        self, record: MigrationRecord, telemetry: list[AgentTelemetry]
    ) -> None:
        """Open the pull request, if this migration has earned one.

        Three things have to hold, and the order matters: the caller must have
        configured a repository at all, the *policy* must permit a PR, and the
        agent's prose must survive the claim audit. Only then does anything
        reach GitHub.
        """
        if not self.run_delivery:
            return

        outcome = DeliveryOutcome()
        record.delivery = outcome

        if self.github is None:
            outcome.skipped_reason = (
                "no GitHub client configured; set ETLM_GITHUB_TOKEN and "
                "ETLM_GITHUB_REPOSITORY, or pass --no-pr to stop asking"
            )
            log.info("delivery.skipped", reason=outcome.skipped_reason)
            return

        with stage_context(MigrationStage.PULL_REQUEST.value):
            entry = lifecycle.begin_stage(record, MigrationStage.PULL_REQUEST, self._now())
            decision = decide_delivery(record)
            outcome.decision = decision
            log.info("delivery.decided", disposition=decision.disposition.value)

            if not decision.should_open:
                outcome.skipped_reason = decision.reason
                lifecycle.complete_stage(
                    entry, self._now(), detail=f"refused: {decision.reason}"
                )
                return

            narrative: PullRequestNarrative | None = None
            audit: ClaimAudit | None = None
            for revision in range(self.max_narrative_revisions + 1):
                run, audit_calls = await steps.propose_pr_narrative(
                    self.ctx,
                    record=record,
                    decision=decision,
                    revision_of=audit if audit is not None and not audit.passed else None,
                )
                telemetry.append(steps.telemetry_of(run, gate_iterations=audit_calls))
                narrative = run.output
                audit = audit_numeric_claims(narrative, record)
                outcome.audit = audit
                outcome.narrative_revisions = revision
                if audit.passed:
                    break
                log.warning("delivery.claims_rejected", violations=len(audit.violations))

            if audit is None or not audit.passed or narrative is None:
                # The prose could not be made to match the measurements. Opening
                # the PR anyway would publish the overstatement, which is the one
                # outcome this stage exists to prevent.
                outcome.skipped_reason = (
                    "the PR narrative made claims the migration record does not "
                    f"support, after {self.max_narrative_revisions} revision(s)"
                )
                lifecycle.fail_stage(
                    record, entry, self._now(), reason=outcome.skipped_reason
                )
                return

            files = steps.build_file_changes(record, directory=self.delivery_directory)
            delivered = steps.deliver_pull_request(
                self.github,
                record=record,
                decision=decision,
                narrative=narrative,
                files=files,
                branch=f"{self.delivery_branch_prefix}/{record.migration_id}",
            )
            delivered.audit = audit
            delivered.narrative_revisions = outcome.narrative_revisions
            record.delivery = delivered
            pr = delivered.pull_request
            lifecycle.complete_stage(
                entry,
                self._now(),
                detail=f"PR #{pr.number} ({'draft' if pr.draft else 'ready'})"
                if pr is not None
                else "no PR opened",
            )

    async def _optimize(
        self,
        request: MigrationRequest,
        record: MigrationRecord,
        spec: object,
        plan: object,
        code: object,
        telemetry: list[AgentTelemetry],
    ) -> None:
        """The same measure/change/re-verify/re-measure loop the workflow runs.

        `evaluate_optimization` is shared with `OptimizationWorkflow`, so the
        acceptance rule cannot drift between the local rehearsal and the durable
        run — which matters more here than anywhere else, because the rule is
        the entire point of the stage.
        """
        if not self.run_optimization or code is None:
            return

        with stage_context(MigrationStage.OPTIMIZATION.value):
            entry = lifecycle.begin_stage(record, MigrationStage.OPTIMIZATION, self._now())
            outcome = OptimizationOutcome()
            artifacts = self.settings.workspace_dir / str(request.migration_id)
            module_path = steps.materialize(artifacts, code.filename, code.content)  # type: ignore[attr-defined]
            strategy = plan.execution_strategy  # type: ignore[attr-defined]

            baseline = steps.benchmark(
                label="baseline",
                module_path=module_path,
                input_dir=request.input_dir,
                output_dir=artifacts / "benchmark_baseline",
                strategy=strategy,
                runs=self.benchmark_runs,
            )
            outcome.baseline = baseline
            outcome.final = baseline
            record.optimization = outcome

            if baseline.failed:
                lifecycle.complete_stage(
                    entry, self._now(), detail="baseline benchmark failed; nothing measured"
                )
                return

            analysis = steps.analyze_optimization_opportunities(code, request.input_dir)  # type: ignore[arg-type]
            # Read once per stage, not once per attempt: the corpus cannot
            # change mid-loop, and re-reading it would be work for no answer.
            past = self._past()
            tried: list[str] = []
            for attempt in range(1, self.max_optimization_attempts + 1):
                run, gate_iterations = await steps.propose_optimization(
                    self.ctx,
                    plan=plan,  # type: ignore[arg-type]
                    code=code,  # type: ignore[arg-type]
                    baseline=baseline,
                    analysis=analysis,
                    history=outcome.attempts,
                    attempt=attempt,
                    max_attempts=self.max_optimization_attempts,
                    past=past,
                )
                telemetry.append(steps.telemetry_of(run, gate_iterations=gate_iterations))
                proposal = run.output
                approach = proposal.strategy.approach

                if approach == "no_change":
                    outcome.attempts.append(
                        OptimizationAttempt(
                            attempt=attempt,
                            strategy=proposal.strategy,
                            admitted=False,
                            rejection_reason="the optimizer found no grounded opportunity",
                            verdict="declined: no grounded opportunity",
                        )
                    )
                    break
                if approach in tried:
                    outcome.attempts.append(
                        OptimizationAttempt(
                            attempt=attempt,
                            strategy=proposal.strategy,
                            admitted=False,
                            rejection_reason=f"approach '{approach}' was already measured",
                            verdict=f"refused: '{approach}' already measured",
                        )
                    )
                    continue

                tried.append(approach)
                record_attempt = OptimizationAttempt(
                    attempt=attempt, strategy=proposal.strategy
                )
                outcome.attempts.append(record_attempt)

                candidate_code = proposal.code or code
                candidate_strategy = proposal.execution_strategy or strategy

                if proposal.code is not None:
                    gate = steps.verify_gate(
                        proposal.code,
                        gate_options=self.ctx.gate_options,
                        gate_iterations=gate_iterations,
                    )
                    if not gate.static_analysis.passed:
                        record_attempt.verdict = (
                            "rejected: optimised code failed the static gate"
                        )
                        continue

                revalidated = await self._validate(
                    request,
                    spec,
                    plan.model_copy(update={"execution_strategy": candidate_strategy}),  # type: ignore[attr-defined]
                    candidate_code,
                    telemetry,
                )
                record_attempt.validation_status = revalidated.report.status.value
                if not revalidated.passed:
                    record_attempt.verdict = (
                        f"rejected: validation {revalidated.report.status.value} after "
                        "the change"
                    )
                    continue

                candidate_path = steps.materialize(
                    artifacts, f"candidate{attempt}_{code.filename}", candidate_code.content  # type: ignore[attr-defined]
                )
                candidate = steps.benchmark(
                    label=f"candidate{attempt}",
                    module_path=candidate_path,
                    input_dir=request.input_dir,
                    output_dir=artifacts / f"benchmark_candidate{attempt}",
                    strategy=candidate_strategy,
                    runs=self.benchmark_runs,
                )
                comparison = BenchmarkComparison(
                    baseline=baseline, candidate=candidate, min_speedup=self.min_speedup
                )
                record_attempt.comparison = comparison
                accepted, verdict = evaluate_optimization(
                    validation=revalidated.report, comparison=comparison
                )
                record_attempt.accepted = accepted
                record_attempt.verdict = verdict
                log.info("optimization.attempt", attempt=attempt, verdict=verdict)

                if accepted:
                    outcome.applied = True
                    outcome.accepted_strategy = proposal.strategy
                    outcome.optimized_code = proposal.code
                    outcome.optimized_execution_strategy = candidate_strategy
                    outcome.final = candidate
                    if proposal.code is not None and record.codegen is not None:
                        record.codegen = record.codegen.model_copy(
                            update={"code": proposal.code}
                        )
                    if record.plan is not None:
                        record.plan = record.plan.model_copy(
                            update={"execution_strategy": candidate_strategy}
                        )
                    break

            lifecycle.complete_stage(
                entry,
                self._now(),
                detail=(
                    f"{outcome.speedup:.2f}x applied"
                    if outcome.applied
                    else f"no change kept after {len(outcome.attempts)} attempt(s)"
                ),
            )

    async def _repair(
        self,
        request: MigrationRequest,
        spec: object,
        plan: object,
        code: object,
        outcome: ValidationOutcome,
        telemetry: list[AgentTelemetry],
    ) -> RepairOutcome:
        """The same bounded loop `RepairWorkflow` runs, without the durability.

        `RepairLedger` is shared between the two, so the rule that makes the
        budget mean "distinct ideas" cannot drift between the local rehearsal
        and the durable run.
        """
        result = RepairOutcome()
        past = self._past()
        ledger = RepairLedger(self.max_repair_attempts)
        ledger.register_baseline(code)  # type: ignore[arg-type]

        report = outcome.report
        diagnosis = outcome.diagnosis
        current = code

        for attempt in range(1, self.max_repair_attempts + 1):
            run, gate_iterations = await steps.propose_repair(
                self.ctx,
                report=report,
                diagnosis=diagnosis,
                plan=plan,  # type: ignore[arg-type]
                code=current,  # type: ignore[arg-type]
                history=result.attempts,
                attempt=attempt,
                max_attempts=self.max_repair_attempts,
                past=past,
            )
            telemetry.append(steps.telemetry_of(run, gate_iterations=gate_iterations))
            proposal = run.output

            admissible, reason = ledger.admits(proposal)
            if not admissible:
                log.info("repair.rejected", attempt=attempt, reason=reason)
                result.attempts.append(
                    RepairAttempt(
                        attempt=attempt,
                        strategy=proposal.strategy,
                        code_sha256=code_fingerprint(proposal.code),
                        admitted=False,
                        rejection_reason=reason,
                        expected_effect=proposal.expected_effect,
                    )
                )
                continue

            ledger.record(proposal)
            record = RepairAttempt(
                attempt=attempt,
                strategy=proposal.strategy,
                code_sha256=code_fingerprint(proposal.code),
                expected_effect=proposal.expected_effect,
            )
            result.attempts.append(record)

            gate = steps.verify_gate(
                proposal.code,
                gate_options=self.ctx.gate_options,
                gate_iterations=gate_iterations,
            )
            record.static_analysis = gate.static_analysis
            if not gate.static_analysis.passed:
                log.info("repair.gate_failed", attempt=attempt)
                continue

            revalidated = await self._validate(
                request, spec, plan, proposal.code, telemetry
            )
            record.validation_status = revalidated.report.status
            record.differences = len(revalidated.report.differences)
            result.final_report = revalidated.report

            if revalidated.passed:
                log.info("repair.succeeded", attempt=attempt)
                result.succeeded = True
                result.repaired_code = proposal.code
                return result

            report = revalidated.report
            diagnosis = revalidated.diagnosis
            current = proposal.code

        result.exhausted = True
        log.warning("repair.exhausted", attempts=result.attempts_used)
        return result

    async def _validate(
        self,
        request: MigrationRequest,
        spec: object,
        plan: object,
        code: object,
        telemetry: list[AgentTelemetry],
    ) -> ValidationOutcome:
        """The same sequence `ValidationWorkflow` runs, without the durability.

        Kept structurally parallel on purpose: if the two ever disagree about
        what validation means, the local run stops being a rehearsal of the
        real one and becomes a second, weaker system.
        """
        artifacts = self.settings.workspace_dir / str(request.migration_id)
        module_path = steps.materialize(artifacts, code.filename, code.content)  # type: ignore[attr-defined]
        strategy = plan.execution_strategy  # type: ignore[attr-defined]

        legacy = steps.execute_legacy(
            source_path=request.source_path,
            input_dir=request.input_dir,
            output_dir=artifacts / "reference_output",
        )
        legacy.output_path = str(steps.resolve_output_root(artifacts / "reference_output"))
        if not legacy.succeeded:
            return ValidationOutcome(
                report=ValidationReport(
                    migration_id=str(request.migration_id),
                    error=f"legacy pipeline did not produce an output: {legacy.error}",
                ),
                legacy_execution=legacy,
            )

        spark = steps.execute_spark(
            module_path=module_path,
            input_dir=request.input_dir,
            output_dir=artifacts / "candidate_output",
            strategy=strategy,
        )
        if spark.succeeded:
            spark.output_path = str(
                steps.resolve_output_root(artifacts / "candidate_output")
            )
        if not spark.succeeded:
            return ValidationOutcome(
                report=ValidationReport(
                    migration_id=str(request.migration_id),
                    error=f"spark pipeline did not produce an output: {spark.error}",
                ),
                legacy_execution=legacy,
                spark_execution=spark,
            )

        tests = None
        test_run = None
        if self.run_generated_tests:
            run, gate_iterations = await steps.generate_test_suite(
                self.ctx, spec=spec, plan=plan, code=code,  # type: ignore[arg-type]
                filename="test_generated_pipeline.py",
            )
            telemetry.append(steps.telemetry_of(run, gate_iterations=gate_iterations))
            tests = run.output
            gate_report = steps.verify_test_gate(tests)
            if gate_report.passed:
                steps.materialize(artifacts, tests.filename, tests.content)
                test_run = steps.execute_test_suite(
                    tests=tests,
                    pipeline_path=module_path,
                    input_dir=request.input_dir,
                    strategy=strategy,
                )
            else:
                log.warning("tests.gate_failed", findings=gate_report.render())

        report = steps.diff_outputs(
            migration_id=str(request.migration_id),
            reference_path=Path(legacy.output_path or ""),
            candidate_path=Path(spark.output_path or ""),
            plan=plan,  # type: ignore[arg-type]
        )
        if test_run is not None:
            report.checks.append(
                CheckResult(
                    name="generated_tests",
                    passed=test_run.succeeded,
                    detail=test_run.render(),
                )
            )

        outcome = ValidationOutcome(
            report=report,
            legacy_execution=legacy,
            spark_execution=spark,
            tests=tests,
            test_run=test_run,
        )
        if not outcome.passed and report.error is None:
            diagnosis_run = await steps.diagnose_failure(
                self.ctx, report=report, plan=plan  # type: ignore[arg-type]
            )
            telemetry.append(steps.telemetry_of(diagnosis_run))
            outcome.diagnosis = diagnosis_run.output
        return outcome

    def _resolve_approval(self, record: MigrationRecord) -> bool:
        entry = lifecycle.begin_stage(record, MigrationStage.APPROVAL, self._now())
        decision = self.approval(record)
        if decision is None:
            lifecycle.apply_approval_timeout(record, entry, self._now(), timeout_seconds=0)
            log.warning(
                "approval.not_granted",
                risk=record.risk.value,
                note="submit through Temporal to wait durably for a real decision",
            )
            return False
        return lifecycle.apply_approval(record, entry, decision, self._now())

    def _persist(self, record: MigrationRecord, telemetry: list[AgentTelemetry]) -> None:
        directory = self.settings.workspace_dir / record.migration_id
        record.artifact_dir = str(directory)
        steps.write_artifacts(directory, record, telemetry)
        # Metrics are derived from the same finished record the artifacts hold,
        # so the two can never disagree. Emitted here rather than sprinkled
        # through the stages: an instrumented call site is one more thing to
        # keep in step, and one that silently stops firing when a branch moves.
        metrics = get_metrics()
        metrics.observe(record)
        for entry in telemetry:
            metrics.observe_telemetry(entry)
