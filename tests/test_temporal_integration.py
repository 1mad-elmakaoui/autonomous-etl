"""End-to-end workflow tests against a real Temporal server.

These are the tests that prove durability actually works: a workflow that pauses
for a human signal, a rejection that stops the migration, an abort mid-flight.
None of that can be verified without a server, so nothing here is faked to make
CI look green.

Running them:

    docker compose up -d temporal
    pytest -m integration

They skip — with the command above in the skip reason — when no server answers.
`ETLM_TEMPORAL_HOST` selects a different address.

Why they are not in the default run: the Temporal SDK's own time-skipping test
server is downloaded from the network on first use, which is unavailable in
sandboxed and air-gapped CI. Rather than pretend, the deterministic half of the
workflow's behaviour is covered exhaustively in `test_lifecycle.py` and its
structure in `test_workflow_definition.py`.
"""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

import pytest
from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter

from etl_migrator.config import LLMProvider, Settings
from etl_migrator.domain.artifacts import MigrationRecord
from etl_migrator.domain.enums import MigrationStage, RiskLevel, ValidationStatus
from etl_migrator.domain.messages import ApprovalDecision, MigrationWorkflowInput
from etl_migrator.temporal import client as tc
from etl_migrator.temporal.worker import build_worker

pytestmark = pytest.mark.integration

CONNECT_TIMEOUT_SECONDS = 5
FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "llm"
EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "customer_pipeline"


@pytest.fixture(scope="module")
def temporal_settings(tmp_path_factory: pytest.TempPathFactory) -> Settings:
    return Settings(
        llm_provider=LLMProvider.SCRIPTED,
        llm_fixture_dir=FIXTURE_DIR,
        workspace_dir=tmp_path_factory.mktemp("temporal-workspace"),
        temporal_task_queue=f"etlm-test-{uuid.uuid4().hex[:8]}",
        log_format="json",
    )


@pytest.fixture(scope="module")
async def temporal_client(temporal_settings: Settings) -> Client:
    try:
        return await asyncio.wait_for(
            Client.connect(
                temporal_settings.temporal_host,
                namespace=temporal_settings.temporal_namespace,
                data_converter=pydantic_data_converter,
            ),
            timeout=CONNECT_TIMEOUT_SECONDS,
        )
    except Exception as exc:
        pytest.skip(
            f"no Temporal server at {temporal_settings.temporal_host} ({type(exc).__name__}). "
            "Start one with:  docker compose up -d temporal"
        )


@pytest.fixture
async def worker(temporal_client: Client, temporal_settings: Settings):
    """A worker running for the duration of one test, on a unique task queue."""
    w = build_worker(temporal_client, temporal_settings)
    async with w:
        yield w


def workflow_input(migration_id: str) -> MigrationWorkflowInput:
    """Input for a durability test, not a performance one.

    Optimisation and the generated pytest suite are switched off: this file is
    about signals, queries and replay, and eight benchmark rounds per test add
    five minutes to prove nothing these assertions look at. Validation still
    runs, so the pipeline is still really executed.

    Delivery stays *on* deliberately. With no GitHub token configured it must
    skip and let the migration finish; that is a behaviour worth holding onto.
    """
    params = tc.build_input(
        EXAMPLE / "legacy_pipeline.py",
        EXAMPLE / "input",
        migration_id=migration_id,
        approval_timeout_seconds=30,
    )
    return params.model_copy(update={"optimize_enabled": False, "run_generated_tests": False})


async def _wait_for_approval_gate(client: Client, migration_id: str) -> None:
    """Poll the query until the workflow parks at the approval gate."""
    for _ in range(100):
        status = await tc.query_status(client, migration_id)
        if status.awaiting_approval:
            return
        await asyncio.sleep(0.1)
    pytest.fail("workflow never reached the approval gate")


class TestDurableMigration:
    async def test_high_risk_migration_pauses_for_a_human(
        self, temporal_client: Client, temporal_settings: Settings, worker: object
    ) -> None:
        """The load-bearing behaviour: the workflow stops and waits."""
        params = workflow_input(f"mig-pause-{uuid.uuid4().hex[:8]}")
        await tc.submit(temporal_client, temporal_settings, params)
        await _wait_for_approval_gate(temporal_client, params.migration_id)

        status = await tc.query_status(temporal_client, params.migration_id)
        assert status.stage is MigrationStage.APPROVAL
        assert status.risk is RiskLevel.HIGH
        assert status.finished is False

        await tc.send_approval(
            temporal_client,
            params.migration_id,
            ApprovalDecision(approved=True, actor="pytest", reason="reviewed"),
        )
        record = await tc.wait_for_record(temporal_client, params.migration_id)

        assert not record.failed
        assert record.codegen is not None and record.codegen.static_analysis.passed
        assert isinstance(record, MigrationRecord), (
            "handle.result() decoded to a bare dict; the handle carries no result type"
        )

        # A worker with no GitHub credentials must skip delivery, not fail the
        # migration. Raising here would throw away a Spark pipeline that was
        # already validated as correct, over a missing token.
        assert record.delivery is not None
        assert record.delivery.skipped_reason is not None
        assert record.delivery.pull_request is None
        # Assert the migration *passed through* static analysis, not that it
        # stopped there. The original assertion pinned the terminal stage, which
        # was STATIC_ANALYSIS when this test was written and has moved with
        # every phase added since; it only survived that long because delivery
        # was failing the workflow before execution ever reached this line.
        completed = {s.stage for s in record.stages if s.succeeded}
        assert MigrationStage.STATIC_ANALYSIS in completed, sorted(s.value for s in completed)
        assert record.validation is not None
        assert record.validation.report is not None
        assert record.validation.report.status is ValidationStatus.PASS

    async def test_rejection_stops_the_migration_before_codegen(
        self, temporal_client: Client, temporal_settings: Settings, worker: object
    ) -> None:
        params = workflow_input(f"mig-reject-{uuid.uuid4().hex[:8]}")
        await tc.submit(temporal_client, temporal_settings, params)
        await _wait_for_approval_gate(temporal_client, params.migration_id)

        await tc.send_approval(
            temporal_client,
            params.migration_id,
            ApprovalDecision(approved=False, actor="pytest", reason="wrong source table"),
        )
        record = await tc.wait_for_record(temporal_client, params.migration_id)

        assert record.failed
        assert "rejected by pytest" in (record.failure_reason or "")
        assert record.codegen is None, "no code should be generated after a rejection"
        assert record.plan is not None, "the plan is still persisted for the human to read"

    async def test_abort_signal_stops_a_waiting_migration(
        self, temporal_client: Client, temporal_settings: Settings, worker: object
    ) -> None:
        params = workflow_input(f"mig-abort-{uuid.uuid4().hex[:8]}")
        await tc.submit(temporal_client, temporal_settings, params)
        await _wait_for_approval_gate(temporal_client, params.migration_id)

        await tc.send_abort(temporal_client, params.migration_id, "operator changed their mind")
        record = await tc.wait_for_record(temporal_client, params.migration_id)

        assert record.failed
        assert (record.failure_reason or "").startswith("aborted:")

    async def test_artifacts_are_written_even_when_rejected(
        self, temporal_client: Client, temporal_settings: Settings, worker: object
    ) -> None:
        params = workflow_input(f"mig-artifacts-{uuid.uuid4().hex[:8]}")
        await tc.submit(temporal_client, temporal_settings, params)
        await _wait_for_approval_gate(temporal_client, params.migration_id)
        await tc.send_approval(
            temporal_client,
            params.migration_id,
            ApprovalDecision(approved=False, actor="pytest"),
        )
        record = await tc.wait_for_record(temporal_client, params.migration_id)

        assert record.artifact_dir is not None
        written = {p.name for p in Path(record.artifact_dir).iterdir()}
        assert {"migration_spec.json", "migration_plan.json", "migration_record.json"} <= written


class TestIdempotencyAndQueries:
    async def test_resubmitting_the_same_id_attaches_to_the_running_execution(
        self, temporal_client: Client, temporal_settings: Settings, worker: object
    ) -> None:
        """The migration id is the workflow id, so a double submit is not a
        double migration — a duplicated LLM bill is a real failure mode."""
        params = workflow_input(f"mig-dedupe-{uuid.uuid4().hex[:8]}")
        first = await tc.submit(temporal_client, temporal_settings, params)
        await _wait_for_approval_gate(temporal_client, params.migration_id)
        second = await tc.submit(temporal_client, temporal_settings, params)

        assert first.run_id == second.run_id

        await tc.send_abort(temporal_client, params.migration_id, "cleanup")
        await tc.wait_for_record(temporal_client, params.migration_id)

    async def test_report_query_returns_the_durable_record(
        self, temporal_client: Client, temporal_settings: Settings, worker: object
    ) -> None:
        params = workflow_input(f"mig-report-{uuid.uuid4().hex[:8]}")
        await tc.submit(temporal_client, temporal_settings, params)
        await _wait_for_approval_gate(temporal_client, params.migration_id)

        report = await tc.query_report(temporal_client, params.migration_id)
        assert report is not None
        assert report.spec is not None and len(report.spec.transformations) == 9
        assert report.plan is not None and report.plan.requires_human_approval

        await tc.send_abort(temporal_client, params.migration_id, "cleanup")
        await tc.wait_for_record(temporal_client, params.migration_id)


class TestReplayDeterminism:
    async def test_completed_history_replays_without_divergence(
        self, temporal_client: Client, temporal_settings: Settings, worker: object
    ) -> None:
        """The definitive determinism check.

        Temporal replays a completed workflow's history against the current code
        and fails if any decision differs. This is what catches a
        non-determinism the static guards cannot see.
        """
        from temporalio.worker import Replayer

        from etl_migrator.workflows.migration import ETLMigrationWorkflow

        params = workflow_input(f"mig-replay-{uuid.uuid4().hex[:8]}")
        await tc.submit(temporal_client, temporal_settings, params)
        await _wait_for_approval_gate(temporal_client, params.migration_id)
        await tc.send_approval(
            temporal_client,
            params.migration_id,
            ApprovalDecision(approved=True, actor="pytest"),
        )
        handle = tc.handle_for(temporal_client, params.migration_id)
        await handle.result()

        replayer = Replayer(
            workflows=[ETLMigrationWorkflow], data_converter=pydantic_data_converter
        )
        await replayer.replay_workflow(await handle.fetch_history())
