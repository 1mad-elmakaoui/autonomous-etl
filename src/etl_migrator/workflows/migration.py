"""`ETLMigrationWorkflow`: the durable spine of a migration.

Determinism rules obeyed here:

* No I/O, no LLM calls, no filesystem. All of those are Activities. Replay
  re-executes this function from the start.
* No clock, no uuid, no randomness. Time comes from `workflow.now()`. The
  migration id comes from the caller and doubles as the workflow id, so
  submitting twice attaches to the running execution.
* No state mutation inline. It lives in `domain.lifecycle` as pure functions
  taking `now` explicitly, so it is testable without a Temporal server.
* Imports are passed through the sandbox for the modules below. The sandbox
  reloads unpassed modules per workflow instance, and dragging pandas and
  AutoGen through that would be slow for no gain since the workflow only needs
  their type definitions.

`ValidationWorkflow`, `RepairWorkflow`, `OptimizationWorkflow` and
`DeliveryWorkflow` attach as children at the marked branch points.
"""

from __future__ import annotations

from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from etl_migrator.activities.migration import MigrationActivities
    from etl_migrator.domain import lifecycle
    from etl_migrator.domain.artifacts import MigrationRecord
    from etl_migrator.domain.enums import MigrationStage, RiskLevel
    from etl_migrator.domain.messages import (
        AgentTelemetry,
        ApprovalDecision,
        CodegenInput,
        DeliveryWorkflowInput,
        DiscoveryInput,
        MigrationStatus,
        MigrationWorkflowInput,
        OptimizationWorkflowInput,
        PersistInput,
        PlanningInput,
        RepairWorkflowInput,
        StaticAnalysisInput,
        ValidationWorkflowInput,
    )
    from etl_migrator.workflows.delivery import DeliveryWorkflow
    from etl_migrator.workflows.optimization import OptimizationWorkflow
    from etl_migrator.workflows.repair import RepairWorkflow
    from etl_migrator.workflows.validation import ValidationWorkflow

#: Retrying these cannot help: the input or the configuration is wrong. Listed by
#: class name because Temporal surfaces an unhandled Python exception as an
#: ApplicationError whose `type` is the exception class name.
NON_RETRYABLE: list[str] = [
    "ConfigurationError",
    "UnsupportedSourceError",
    "NonRetryableMigrationError",
    "ScriptExhaustedError",
]

#: LLM-backed activities: slow, occasionally flaky, worth several attempts.
AGENT_RETRY = RetryPolicy(
    initial_interval=timedelta(seconds=2),
    backoff_coefficient=2.0,
    maximum_interval=timedelta(seconds=30),
    maximum_attempts=4,
    non_retryable_error_types=NON_RETRYABLE,
)
AGENT_TIMEOUT = timedelta(minutes=10)

#: Deterministic local work: fast, and a repeated failure means a real bug.
LOCAL_RETRY = RetryPolicy(
    initial_interval=timedelta(seconds=1),
    backoff_coefficient=2.0,
    maximum_attempts=3,
    non_retryable_error_types=NON_RETRYABLE,
)
LOCAL_TIMEOUT = timedelta(minutes=2)


@workflow.defn(name="ETLMigrationWorkflow")
class ETLMigrationWorkflow:
    """Discovery → planning → (approval) → code generation → static gate."""

    def __init__(self) -> None:
        self._record: MigrationRecord | None = None
        self._telemetry: list[AgentTelemetry] = []
        self._approval: ApprovalDecision | None = None
        self._abort_reason: str | None = None
        self._awaiting_approval = False
        self._finished = False

    # -- signals -----------------------------------------------------------
    @workflow.signal
    def approve(self, decision: ApprovalDecision) -> None:
        """Release or reject a migration paused at the approval gate.

        Durable: a decision that arrives while every worker is down is delivered
        when one returns, and a workflow already waiting simply wakes up.
        """
        workflow.logger.info(
            "approval signal received: approved=%s actor=%s", decision.approved, decision.actor
        )
        self._approval = decision

    @workflow.signal
    def abort(self, reason: str) -> None:
        """Operator cancellation. Recorded distinctly from a failure."""
        workflow.logger.info("abort signal received: %s", reason)
        self._abort_reason = reason

    # -- queries -----------------------------------------------------------
    @workflow.query
    def status(self) -> MigrationStatus:
        """Cheap live status. Safe to poll; never mutates state."""
        record = self._record
        if record is None:
            return MigrationStatus(
                migration_id=workflow.info().workflow_id,
                stage=MigrationStage.DISCOVERY,
                risk=RiskLevel.LOW,
            )
        return MigrationStatus(
            migration_id=record.migration_id,
            stage=record.stage,
            risk=record.risk,
            awaiting_approval=self._awaiting_approval,
            approval=self._approval,
            finished=self._finished,
            failed=record.failed,
            failure_reason=record.failure_reason,
            stages=record.stages,
        )

    @workflow.query
    def report(self) -> MigrationRecord | None:
        """The full durable record, including spec, plan and gate verdict."""
        return self._record

    # -- run ---------------------------------------------------------------
    @workflow.run
    async def run(self, params: MigrationWorkflowInput) -> MigrationRecord:
        record = lifecycle.new_record(
            params.migration_id, params.source_path, workflow.now()
        )
        self._record = record
        workflow.logger.info("migration started: %s", params.migration_id)

        # --- discovery ----------------------------------------------------
        entry = lifecycle.begin_stage(record, MigrationStage.DISCOVERY, workflow.now())
        discovery = await workflow.execute_activity_method(
            MigrationActivities.analyze_legacy_pipeline,
            DiscoveryInput(
                migration_id=params.migration_id,
                source_path=params.source_path,
                input_dir=params.input_dir,
                scenario=params.scenario,
            ),
            start_to_close_timeout=AGENT_TIMEOUT,
            retry_policy=AGENT_RETRY,
            summary="analyse legacy pipeline",
        )
        record.spec = discovery.spec
        self._telemetry.append(discovery.telemetry)
        lifecycle.complete_stage(
            entry, workflow.now(), detail=f"tools={discovery.telemetry.tools_used}"
        )
        if await self._aborted(record):
            return await self._finish(params, record)

        # --- planning -----------------------------------------------------
        entry = lifecycle.begin_stage(record, MigrationStage.PLANNING, workflow.now())
        planning = await workflow.execute_activity_method(
            MigrationActivities.generate_migration_plan,
            PlanningInput(
                migration_id=params.migration_id,
                input_dir=params.input_dir,
                scenario=params.scenario,
                spec=discovery.spec,
            ),
            start_to_close_timeout=AGENT_TIMEOUT,
            retry_policy=AGENT_RETRY,
            summary="plan the migration",
        )
        # Policy is applied here, in durable code, not inside the agent.
        plan = lifecycle.enforce_approval_policy(planning.plan)
        record.plan = plan
        self._telemetry.append(planning.telemetry)
        lifecycle.complete_stage(
            entry, workflow.now(), detail=f"risk={plan.overall_risk.value}"
        )
        if await self._aborted(record):
            return await self._finish(params, record)

        # --- human approval ----------------------------------------------
        if plan.requires_human_approval and not await self._await_approval(params, record):
            return await self._finish(params, record)

        # --- code generation ----------------------------------------------
        entry = lifecycle.begin_stage(record, MigrationStage.CODE_GENERATION, workflow.now())
        codegen = await workflow.execute_activity_method(
            MigrationActivities.generate_spark_code,
            CodegenInput(
                migration_id=params.migration_id,
                scenario=params.scenario,
                output_filename=params.output_filename,
                spec=discovery.spec,
                plan=plan,
            ),
            start_to_close_timeout=AGENT_TIMEOUT,
            retry_policy=AGENT_RETRY,
            summary="generate PySpark",
        )
        self._telemetry.append(codegen.telemetry)
        lifecycle.complete_stage(
            entry, workflow.now(), detail=f"gate_submissions={codegen.gate_iterations}"
        )

        # --- independent static gate --------------------------------------
        entry = lifecycle.begin_stage(record, MigrationStage.STATIC_ANALYSIS, workflow.now())
        analysis = await workflow.execute_activity_method(
            MigrationActivities.run_static_analysis,
            StaticAnalysisInput(
                migration_id=params.migration_id,
                code=codegen.code,
                gate_iterations=codegen.gate_iterations,
            ),
            start_to_close_timeout=LOCAL_TIMEOUT,
            retry_policy=LOCAL_RETRY,
            summary="verify generated code",
        )
        record.codegen = analysis.result

        if not analysis.passed:
            # Routed into RepairWorkflow rather than failing: the findings are
            # exactly the root-cause input a repair strategy needs.
            lifecycle.fail_stage(
                record,
                entry,
                workflow.now(),
                reason="generated code failed the independent static gate: "
                + "; ".join(f.render() for f in analysis.report.errors),
            )
            return await self._finish(params, record)

        lifecycle.complete_stage(
            entry, workflow.now(), detail=f"findings={len(analysis.report.findings)}"
        )

        # --- validation (child workflow) ----------------------------------
        # A child rather than four more stages inline: execution and diffing
        # have a different failure profile from agent calls, and the repair
        # loop re-runs validation as a unit after each attempt.
        entry = lifecycle.begin_stage(record, MigrationStage.VALIDATION, workflow.now())
        outcome = await workflow.execute_child_workflow(
            ValidationWorkflow.run,
            ValidationWorkflowInput(
                migration_id=params.migration_id,
                legacy_source_path=params.source_path,
                input_dir=params.input_dir,
                scenario=params.scenario,
                spec=discovery.spec,
                plan=plan,
                code=analysis.result.code,
                run_generated_tests=params.run_generated_tests,
                execution_task_queue=params.execution_task_queue,
            ),
            id=f"{params.migration_id}-validation",
            execution_timeout=timedelta(seconds=params.validation_timeout_seconds),
        )
        record.validation = outcome

        if outcome.passed:
            lifecycle.complete_stage(
                entry, workflow.now(), detail=f"status={outcome.report.status.value}"
            )
            await self._optimize(params, record, analysis.result.code, plan, discovery.spec)
            return await self._finish(params, record)

        reason = f"validation {outcome.report.status.value}: " + (
            "; ".join(d.render() for d in outcome.report.differences[:5])
            or outcome.report.error
            or "no differences recorded"
        )
        if outcome.diagnosis is not None:
            reason += f" | diagnosis: {outcome.diagnosis.summary}"
        lifecycle.fail_stage(record, entry, workflow.now(), reason=reason)

        # --- autonomous repair --------------------------------------------
        # Only attempted when the differ actually measured a disagreement. An
        # ERROR report means a pipeline never produced an output, and there is
        # nothing for a code fix to act on.
        if not params.repair_enabled or outcome.report.error is not None:
            return await self._finish(params, record)

        entry = lifecycle.begin_stage(record, MigrationStage.REPAIR, workflow.now())
        repair = await workflow.execute_child_workflow(
            RepairWorkflow.run,
            RepairWorkflowInput(
                migration_id=params.migration_id,
                legacy_source_path=params.source_path,
                input_dir=params.input_dir,
                scenario=params.scenario,
                spec=discovery.spec,
                plan=plan,
                code=analysis.result.code,
                report=outcome.report,
                diagnosis=outcome.diagnosis,
                max_attempts=params.max_repair_attempts,
                run_generated_tests=params.run_generated_tests,
                execution_task_queue=params.execution_task_queue,
            ),
            id=f"{params.migration_id}-repair",
            execution_timeout=timedelta(seconds=params.repair_timeout_seconds),
        )
        record.repair = repair

        if repair.succeeded and repair.repaired_code is not None:
            # The migration is correct again, by the same measurement that
            # condemned it. The record keeps both the original failure and the
            # repair history, because "passed after two attempts" is materially
            # different from "passed" and a reviewer deserves to see which.
            record.codegen = record.codegen.model_copy(
                update={"code": repair.repaired_code}
            ) if record.codegen is not None else None
            if repair.final_report is not None and record.validation is not None:
                record.validation = record.validation.model_copy(
                    update={"report": repair.final_report, "diagnosis": None}
                )
            record.failed = False
            record.failure_reason = None
            lifecycle.complete_stage(
                entry,
                workflow.now(),
                detail=f"repaired after {repair.attempts_used} attempt(s)",
            )
            await self._optimize(
                params, record, repair.repaired_code, plan, discovery.spec
            )
        else:
            best = repair.best_attempt
            closest = (
                f"; closest attempt {best.attempt} left {best.differences} differences"
                if best is not None
                else ""
            )
            lifecycle.fail_stage(
                record,
                entry,
                workflow.now(),
                reason=(
                    f"repair exhausted after {repair.attempts_used} attempt(s){closest}. "
                    "Human intervention required."
                ),
            )

        return await self._finish(params, record)

    # -- internals ---------------------------------------------------------
    async def _await_approval(
        self, params: MigrationWorkflowInput, record: MigrationRecord
    ) -> bool:
        """Block until a human decides, or the deadline passes.

        This is the reason Temporal is in the architecture at all. The wait is
        durable: workers can be redeployed, the machine can reboot, and the
        migration is still sitting here when a decision arrives days later.
        """
        entry = lifecycle.begin_stage(record, MigrationStage.APPROVAL, workflow.now())
        self._awaiting_approval = True
        workflow.logger.info(
            "awaiting human approval (risk=%s, timeout=%ss)",
            record.risk.value,
            params.approval_timeout_seconds,
        )

        # `wait_condition` returns None and raises on expiry — the timeout is an
        # exception, not a falsy return value.
        try:
            await workflow.wait_condition(
                lambda: self._approval is not None or self._abort_reason is not None,
                timeout=timedelta(seconds=params.approval_timeout_seconds),
                timeout_summary="human approval window",
            )
        except TimeoutError:
            self._awaiting_approval = False
            lifecycle.apply_approval_timeout(
                record,
                entry,
                workflow.now(),
                timeout_seconds=params.approval_timeout_seconds,
            )
            return False

        self._awaiting_approval = False
        if self._abort_reason is not None:
            lifecycle.apply_abort(record, workflow.now(), reason=self._abort_reason)
            return False
        if self._approval is None:  # pragma: no cover - the wait condition guarantees one
            lifecycle.apply_approval_timeout(
                record, entry, workflow.now(), timeout_seconds=params.approval_timeout_seconds
            )
            return False
        return lifecycle.apply_approval(record, entry, self._approval, workflow.now())

    async def _aborted(self, record: MigrationRecord) -> bool:
        """Checked between stages so an abort takes effect promptly.

        Not checked mid-activity: cancelling an in-flight LLM call would leave
        the record describing work whose result was thrown away.
        """
        if self._abort_reason is None:
            return False
        lifecycle.apply_abort(record, workflow.now(), reason=self._abort_reason)
        return True

    async def _optimize(
        self,
        params: MigrationWorkflowInput,
        record: MigrationRecord,
        code: object,
        plan: object,
        spec: object,
    ) -> None:
        """Run the optimisation child, but only on a migration already proven correct.

        Optimising something that does not work is optimising the wrong thing,
        so this is reached only from a passing validation — either the first one
        or the one that cleared a repair.
        """
        if not params.optimize_enabled:
            return

        entry = lifecycle.begin_stage(record, MigrationStage.OPTIMIZATION, workflow.now())
        outcome = await workflow.execute_child_workflow(
            OptimizationWorkflow.run,
            OptimizationWorkflowInput(
                migration_id=params.migration_id,
                legacy_source_path=params.source_path,
                input_dir=params.input_dir,
                scenario=params.scenario,
                spec=spec,  # type: ignore[arg-type]
                plan=plan,  # type: ignore[arg-type]
                code=code,  # type: ignore[arg-type]
                max_attempts=params.max_optimization_attempts,
                benchmark_runs=params.benchmark_runs,
                min_speedup=params.min_speedup,
                execution_task_queue=params.execution_task_queue,
            ),
            id=f"{params.migration_id}-optimize",
            execution_timeout=timedelta(seconds=params.optimization_timeout_seconds),
        )
        record.optimization = outcome

        if outcome.applied:
            # Only the code is swapped when the optimisation changed code; a
            # configuration-only win leaves the module untouched by design.
            if outcome.optimized_code is not None and record.codegen is not None:
                record.codegen = record.codegen.model_copy(
                    update={"code": outcome.optimized_code}
                )
            if outcome.optimized_execution_strategy is not None and record.plan is not None:
                record.plan = record.plan.model_copy(
                    update={"execution_strategy": outcome.optimized_execution_strategy}
                )

        # An optimisation stage that keeps nothing is a success, not a failure:
        # the migration is correct either way, and "measured, found nothing
        # worth keeping" is a legitimate result.
        lifecycle.complete_stage(
            entry,
            workflow.now(),
            detail=(
                f"{outcome.speedup:.2f}x applied"
                if outcome.applied
                else f"no change kept after {len(outcome.attempts)} attempt(s)"
            ),
        )

    async def _deliver(
        self, params: MigrationWorkflowInput, record: MigrationRecord
    ) -> None:
        """Run the delivery child on whatever the migration turned out to be.

        Unlike optimisation, this is *not* gated on success. A migration that
        failed still has work worth showing a human, and `decide_delivery` is
        what distinguishes "ready to review" from "here is a draft, please look"
        from "there is nothing here worth opening". Putting that judgement in
        the child rather than in a condition here keeps one implementation of
        it, shared with the local pipeline.
        """
        if not params.deliver_enabled:
            return

        entry = lifecycle.begin_stage(record, MigrationStage.PULL_REQUEST, workflow.now())
        outcome = await workflow.execute_child_workflow(
            DeliveryWorkflow.run,
            DeliveryWorkflowInput(
                migration_id=params.migration_id,
                scenario=params.scenario,
                record=record,
                branch_prefix=params.delivery_branch_prefix,
                directory=params.delivery_directory,
            ),
            id=f"{params.migration_id}-deliver",
            execution_timeout=timedelta(seconds=params.delivery_timeout_seconds),
        )
        record.delivery = outcome

        # A refusal is a correct outcome, not a stage failure: the migration is
        # exactly as good or bad as it was before delivery ran.
        pull = outcome.pull_request
        lifecycle.complete_stage(
            entry,
            workflow.now(),
            detail=(
                f"PR #{pull.number} ({'draft' if pull.draft else 'ready'})"
                if pull is not None
                else f"no PR: {outcome.skipped_reason}"
            ),
        )

    async def _finish(
        self, params: MigrationWorkflowInput, record: MigrationRecord
    ) -> MigrationRecord:
        """Deliver, persist artifacts, and return.

        Runs on every exit path, including failure. A failed migration is worth
        more artifacts than a successful one, not fewer — the spec, plan and
        gate findings are what a human needs in order to intervene.
        """
        await self._deliver(params, record)

        persisted = await workflow.execute_activity_method(
            MigrationActivities.persist_artifacts,
            PersistInput(
                migration_id=params.migration_id,
                record_json=record.model_dump_json(),
                telemetry=self._telemetry,
            ),
            start_to_close_timeout=LOCAL_TIMEOUT,
            retry_policy=LOCAL_RETRY,
            summary="persist artifacts",
        )
        record.artifact_dir = persisted.artifact_dir
        self._finished = True
        workflow.logger.info("migration finished: %s", lifecycle.summarise(record))
        return record
