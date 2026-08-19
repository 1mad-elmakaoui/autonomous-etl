"""Exhaustive tests for the pure lifecycle logic.

This is where the workflow's branching is actually verified. The Temporal
workflow is thin glue over these functions precisely so that its decisions can
be tested as ordinary code, without a server, in milliseconds — and so that the
same transitions are provably shared with the local pipeline.

Every function takes `now` explicitly, so there is no clock to stub.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from etl_migrator.domain import lifecycle
from etl_migrator.domain.artifacts import MigrationRecord
from etl_migrator.domain.enums import MigrationStage, RiskLevel, TransformKind
from etl_migrator.domain.messages import ApprovalDecision
from etl_migrator.domain.plan import MigrationPlan, PlanStep

T0 = datetime(2026, 8, 10, 12, 0, 0, tzinfo=UTC)


def at(seconds: int) -> datetime:
    return T0 + timedelta(seconds=seconds)


def plan_with(risk: RiskLevel, *, agent_says: bool = False) -> MigrationPlan:
    return MigrationPlan(
        summary="s",
        overall_risk=risk,
        requires_human_approval=agent_says,
        steps=[
            PlanStep(
                id="s1",
                transformation_id="t1",
                kind=TransformKind.FILTER,
                legacy_construct="mask",
                spark_construct="filter",
                rationale="r",
            )
        ],
    )


@pytest.fixture
def record() -> MigrationRecord:
    return lifecycle.new_record("mig-test", "pipeline.py", T0)


class TestRecordCreation:
    def test_creation_time_is_the_supplied_clock(self, record: MigrationRecord) -> None:
        """The workflow passes `workflow.now()`. If this ever fell back to the
        model's `datetime.now` default factory, replay would diverge."""
        assert record.created_at == T0

    def test_starts_at_discovery_and_not_failed(self, record: MigrationRecord) -> None:
        assert record.stage is MigrationStage.DISCOVERY
        assert not record.failed and record.failure_reason is None


class TestStageTransitions:
    def test_begin_stage_moves_the_cursor_and_opens_an_entry(
        self, record: MigrationRecord
    ) -> None:
        entry = lifecycle.begin_stage(record, MigrationStage.PLANNING, at(1))
        assert record.stage is MigrationStage.PLANNING
        assert entry.started_at == at(1)
        assert entry.ended_at is None and entry.succeeded is None
        assert record.stages == [entry]

    def test_complete_stage_records_duration(self, record: MigrationRecord) -> None:
        entry = lifecycle.begin_stage(record, MigrationStage.DISCOVERY, at(0))
        lifecycle.complete_stage(entry, at(5), detail="tools=['x']")
        assert entry.succeeded is True
        assert entry.duration_seconds == 5.0
        assert entry.detail == "tools=['x']"
        assert not record.failed

    def test_fail_stage_marks_the_whole_migration(self, record: MigrationRecord) -> None:
        entry = lifecycle.begin_stage(record, MigrationStage.CODE_GENERATION, at(0))
        lifecycle.fail_stage(record, entry, at(3), reason="boom")
        assert entry.succeeded is False
        assert record.failed is True
        assert record.failure_reason == "boom"

    def test_attempts_are_tracked_separately(self, record: MigrationRecord) -> None:
        """The repair loop reruns stages; attempt numbers keep the timeline
        readable rather than collapsing retries into one entry."""
        lifecycle.begin_stage(record, MigrationStage.VALIDATION, at(0), attempt=1)
        lifecycle.begin_stage(record, MigrationStage.VALIDATION, at(9), attempt=2)
        assert [s.attempt for s in record.stages] == [1, 2]

    def test_total_duration_sums_completed_stages(self, record: MigrationRecord) -> None:
        first = lifecycle.begin_stage(record, MigrationStage.DISCOVERY, at(0))
        lifecycle.complete_stage(first, at(2))
        second = lifecycle.begin_stage(record, MigrationStage.PLANNING, at(2))
        lifecycle.complete_stage(second, at(5))
        assert record.total_duration_seconds == 5.0


class TestApprovalPolicy:
    @pytest.mark.parametrize(
        ("risk", "expected"),
        [(RiskLevel.LOW, False), (RiskLevel.MEDIUM, False), (RiskLevel.HIGH, True)],
    )
    def test_only_high_risk_requires_approval(self, risk: RiskLevel, expected: bool) -> None:
        assert lifecycle.requires_approval(plan_with(risk)) is expected

    def test_agent_cannot_waive_a_required_gate(self) -> None:
        """The whole point of computing this outside the agent."""
        sneaky = plan_with(RiskLevel.HIGH, agent_says=False)
        assert lifecycle.enforce_approval_policy(sneaky).requires_human_approval is True

    def test_agent_cannot_invent_a_gate_that_policy_does_not_require(self) -> None:
        anxious = plan_with(RiskLevel.LOW, agent_says=True)
        assert lifecycle.enforce_approval_policy(anxious).requires_human_approval is False

    def test_policy_returns_the_same_object_when_already_correct(self) -> None:
        correct = plan_with(RiskLevel.HIGH, agent_says=True)
        assert lifecycle.enforce_approval_policy(correct) is correct

    def test_policy_does_not_mutate_the_input(self) -> None:
        sneaky = plan_with(RiskLevel.HIGH, agent_says=False)
        lifecycle.enforce_approval_policy(sneaky)
        assert sneaky.requires_human_approval is False


class TestApprovalOutcomes:
    def test_approval_continues_and_records_the_actor(self, record: MigrationRecord) -> None:
        entry = lifecycle.begin_stage(record, MigrationStage.APPROVAL, at(0))
        decision = ApprovalDecision(approved=True, actor="amira", reason="checked the plan")
        assert lifecycle.apply_approval(record, entry, decision, at(60)) is True
        assert not record.failed
        assert entry.detail == "approved by amira: checked the plan"

    def test_rejection_stops_the_migration(self, record: MigrationRecord) -> None:
        entry = lifecycle.begin_stage(record, MigrationStage.APPROVAL, at(0))
        decision = ApprovalDecision(approved=False, actor="amira", reason="wrong source table")
        assert lifecycle.apply_approval(record, entry, decision, at(60)) is False
        assert record.failed is True
        assert "rejected by amira" in (record.failure_reason or "")

    def test_decision_without_a_reason_is_still_attributed(
        self, record: MigrationRecord
    ) -> None:
        entry = lifecycle.begin_stage(record, MigrationStage.APPROVAL, at(0))
        lifecycle.apply_approval(
            record, entry, ApprovalDecision(approved=True, actor="ci"), at(1)
        )
        assert entry.detail == "approved by ci"

    def test_timeout_is_terminal_and_says_so(self, record: MigrationRecord) -> None:
        """Not a retry: re-asking a human who did not answer yields the same silence."""
        entry = lifecycle.begin_stage(record, MigrationStage.APPROVAL, at(0))
        lifecycle.apply_approval_timeout(record, entry, at(99), timeout_seconds=86400)
        assert record.failed is True
        assert "86400s" in (record.failure_reason or "")
        assert "human intervention" in (record.failure_reason or "")


class TestAbort:
    def test_abort_closes_every_open_stage(self, record: MigrationRecord) -> None:
        done = lifecycle.begin_stage(record, MigrationStage.DISCOVERY, at(0))
        lifecycle.complete_stage(done, at(1))
        open_stage = lifecycle.begin_stage(record, MigrationStage.PLANNING, at(1))

        lifecycle.apply_abort(record, at(10), reason="wrong source")

        assert done.succeeded is True, "a completed stage must not be rewritten"
        assert open_stage.succeeded is False
        assert open_stage.ended_at == at(10)
        assert record.failure_reason == "aborted: wrong source"

    def test_abort_is_distinguishable_from_a_failure(self, record: MigrationRecord) -> None:
        lifecycle.apply_abort(record, at(1), reason="operator changed their mind")
        assert (record.failure_reason or "").startswith("aborted:")


class TestResumePoint:
    def test_fresh_record_resumes_at_discovery(self, record: MigrationRecord) -> None:
        assert lifecycle.resume_point(record) is MigrationStage.DISCOVERY

    def test_resumes_after_the_last_successful_stage(self, record: MigrationRecord) -> None:
        entry = lifecycle.begin_stage(record, MigrationStage.DISCOVERY, at(0))
        lifecycle.complete_stage(entry, at(1))
        assert lifecycle.resume_point(record) is MigrationStage.PLANNING

    def test_a_failed_stage_is_re_run_not_skipped(self, record: MigrationRecord) -> None:
        entry = lifecycle.begin_stage(record, MigrationStage.DISCOVERY, at(0))
        lifecycle.fail_stage(record, entry, at(1), reason="llm timeout")
        assert lifecycle.resume_point(record) is MigrationStage.DISCOVERY

    def test_approval_is_skipped_when_the_plan_does_not_require_it(
        self, record: MigrationRecord
    ) -> None:
        for stage in (MigrationStage.DISCOVERY, MigrationStage.PLANNING):
            lifecycle.complete_stage(lifecycle.begin_stage(record, stage, at(0)), at(1))
        record.plan = plan_with(RiskLevel.LOW)
        assert lifecycle.resume_point(record) is MigrationStage.CODE_GENERATION

    def test_approval_is_required_before_codegen_on_a_high_risk_plan(
        self, record: MigrationRecord
    ) -> None:
        for stage in (MigrationStage.DISCOVERY, MigrationStage.PLANNING):
            lifecycle.complete_stage(lifecycle.begin_stage(record, stage, at(0)), at(1))
        record.plan = plan_with(RiskLevel.HIGH, agent_says=True)
        assert lifecycle.resume_point(record) is MigrationStage.APPROVAL

    def test_fully_complete_record_reports_the_final_stage(
        self, record: MigrationRecord
    ) -> None:
        record.plan = plan_with(RiskLevel.LOW)
        for stage in lifecycle.STAGE_ORDER:
            if stage is MigrationStage.APPROVAL:
                continue
            lifecycle.complete_stage(lifecycle.begin_stage(record, stage, at(0)), at(1))
        # Derived from the order rather than named, so appending a stage does not
        # silently leave this asserting on the second-to-last one.
        assert lifecycle.resume_point(record) is lifecycle.STAGE_ORDER[-1]


class TestSummary:
    def test_failure_summary_names_the_stage_and_reason(
        self, record: MigrationRecord
    ) -> None:
        entry = lifecycle.begin_stage(record, MigrationStage.CODE_GENERATION, at(0))
        lifecycle.fail_stage(record, entry, at(1), reason="gate rejected the module")
        summary = lifecycle.summarise(record)
        assert "FAILED at code_generation" in summary
        assert "gate rejected the module" in summary

    def test_success_summary_reports_risk_and_duration(
        self, record: MigrationRecord
    ) -> None:
        entry = lifecycle.begin_stage(record, MigrationStage.DISCOVERY, at(0))
        lifecycle.complete_stage(entry, at(2))
        record.plan = plan_with(RiskLevel.MEDIUM)
        assert "risk=medium" in lifecycle.summarise(record)
