"""Delivery end to end: a real migration becomes a real pull request.

The migration is genuine — both pipelines execute, the outputs are diffed — and
the only substituted things are the model (recorded fixtures) and the network
(`InMemoryGitHub`). So this exercises the actual delivery agent loop, the actual
claim audit, the actual client, and the actual request bodies.

Two claims are under test, and the second is the one that matters:

* a validated migration produces a labelled PR carrying the generated code, the
  generated tests and a machine-readable record of how it was produced;
* the agent's prose was **checked against the record** before any of that
  happened, and a migration that had not earned a PR would not have got one.

Marked `spark`: validation executes both pipelines.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from etl_migrator.config import LLMProvider, Settings
from etl_migrator.domain.delivery import DeliveryDisposition
from etl_migrator.domain.enums import MigrationStage, ValidationStatus
from etl_migrator.github import GitHubClient, InMemoryGitHub
from etl_migrator.llm.factory import ScriptedModelClientFactory
from etl_migrator.pipeline.local import LocalMigrationPipeline, MigrationRequest
from etl_migrator.tools.pr_body import EVIDENCE_END, EVIDENCE_START

pytestmark = pytest.mark.spark

pytest.importorskip("pyspark", reason="pyspark extra not installed")


@pytest.fixture(scope="module")
def hub() -> InMemoryGitHub:
    return InMemoryGitHub(repository="acme/data-platform", default_branch="main")


@pytest.fixture(scope="module")
async def delivered(
    tmp_path_factory: pytest.TempPathFactory, repo_root: Path, hub: InMemoryGitHub
) -> Any:
    """A full migration with validation, ending in a pull request."""
    example = repo_root / "examples" / "customer_pipeline"
    settings = Settings(
        llm_provider=LLMProvider.SCRIPTED,
        llm_fixture_dir=repo_root / "fixtures" / "llm",
        workspace_dir=tmp_path_factory.mktemp("delivery"),
        log_format="json",
    )
    pipeline = LocalMigrationPipeline(
        settings,
        ScriptedModelClientFactory.from_fixture(
            repo_root / "fixtures" / "llm" / "customer_pipeline.json"
        ),
        run_validation=True,
        run_generated_tests=False,
        run_repair=False,
        run_optimization=False,
        github=GitHubClient(hub, hub.repository),
    )
    return await pipeline.run(
        MigrationRequest(
            source_path=example / "legacy_pipeline.py",
            input_dir=example / "input",
        )
    )


class TestThePullRequestIsOpened:
    def test_a_validated_migration_gets_a_ready_pull_request(
        self, delivered: Any
    ) -> None:
        outcome = delivered.delivery
        assert outcome is not None, "the delivery stage did not run"
        assert outcome.skipped_reason is None, outcome.skipped_reason
        assert outcome.decision.disposition is DeliveryDisposition.READY
        assert outcome.opened
        assert not outcome.pull_request.draft

    def test_it_is_labelled_so_the_population_is_findable(
        self, delivered: Any, hub: InMemoryGitHub
    ) -> None:
        labels = hub.pulls[0].labels
        assert "autonomous-etl" in labels
        assert "high-risk" in labels, "the customer example is HIGH risk"
        assert "needs-human" not in labels

    def test_the_branch_carries_the_code_the_migration_produced(
        self, delivered: Any, hub: InMemoryGitHub
    ) -> None:
        branch = delivered.delivery.branch.name
        written = {path for (b, path) in hub.files if b == branch}
        code_path = next(p for p in written if p.endswith("_spark.py"))
        assert hub.file_content(branch, code_path) == delivered.codegen.code.content

    def test_it_ships_a_machine_readable_record_of_its_own_provenance(
        self, delivered: Any, hub: InMemoryGitHub
    ) -> None:
        """Without this a reviewer six months from now has a PySpark module and
        no way to tell what it came from or what was checked."""
        branch = delivered.delivery.branch.name
        raw = hub.file_content(branch, f"migrations/{delivered.migration_id}/migration_record.json")
        assert raw is not None
        parsed = json.loads(raw)
        assert parsed["migration_id"] == delivered.migration_id
        assert parsed["validation"]["report"]["status"] == "PASS"

    def test_the_stage_is_recorded_in_the_lifecycle(self, delivered: Any) -> None:
        entry = next(
            e for e in delivered.stages if e.stage is MigrationStage.PULL_REQUEST
        )
        assert entry.succeeded is True
        assert "PR #" in (entry.detail or "")


class TestTheBodyIsTrustworthy:
    def test_the_prose_was_audited_against_the_record(self, delivered: Any) -> None:
        """And non-vacuously: the fixture's narrative does contain a figure.

        An audit that passed because there was nothing to check would prove
        nothing, so this asserts a real claim was examined.
        """
        audit = delivered.delivery.audit
        assert audit is not None
        assert audit.passed
        assert audit.checked > 0, "the narrative made no checkable claim"
        assert delivered.delivery.narrative_revisions == 0

    def test_the_evidence_block_reports_the_measured_verdict(
        self, delivered: Any, hub: InMemoryGitHub
    ) -> None:
        body = hub.pulls[0].body
        evidence = body[body.index(EVIDENCE_START) : body.index(EVIDENCE_END)]
        assert "✅ **PASS**" in evidence
        assert "`numeric_tolerance`" in evidence
        assert "row_count" in evidence

    def test_every_declared_semantic_difference_reaches_the_reviewer(
        self, delivered: Any, hub: InMemoryGitHub
    ) -> None:
        body = hub.pulls[0].body
        declared = delivered.plan.all_semantic_differences
        assert f"{len(declared)} pandas" in body
        assert declared[0].description[:40] in body

    def test_the_agent_prose_is_present_and_above_the_evidence(
        self, delivered: Any, hub: InMemoryGitHub
    ) -> None:
        body = hub.pulls[0].body
        assert body.index("This migrates") < body.index(EVIDENCE_START)
        assert "Where to look" in body

    def test_the_title_came_from_the_agent(
        self, delivered: Any, hub: InMemoryGitHub
    ) -> None:
        assert hub.pulls[0].title.startswith("Migrate customer revenue pipeline")


class TestDeliveryIsIdempotent:
    """The retry property, exercised through the whole stage rather than the
    client alone. A Temporal activity may run this twice."""

    async def test_a_second_delivery_reuses_the_branch_and_the_pull_request(
        self, delivered: Any, hub: InMemoryGitHub, repo_root: Path
    ) -> None:
        from etl_migrator.domain.delivery_policy import decide_delivery
        from etl_migrator.pipeline import steps

        before = len(hub.pulls)
        record = delivered
        decision = decide_delivery(record)
        narrative_run, _ = await steps.propose_pr_narrative(
            LocalMigrationPipeline(
                Settings(
                    llm_provider=LLMProvider.SCRIPTED,
                    llm_fixture_dir=repo_root / "fixtures" / "llm",
                ),
                ScriptedModelClientFactory.from_fixture(
                    repo_root / "fixtures" / "llm" / "customer_pipeline.json"
                ),
            ).ctx,
            record=record,
            decision=decision,
        )
        again = steps.deliver_pull_request(
            GitHubClient(hub, hub.repository),
            record=record,
            decision=decision,
            narrative=narrative_run.output,
            files=steps.build_file_changes(record, directory="migrations"),
            branch=record.delivery.branch.name,
        )

        assert len(hub.pulls) == before, "a duplicate pull request was opened"
        assert not again.branch.created
        assert not again.pull_request.created
        assert again.pull_request.number == delivered.delivery.pull_request.number


class TestAnUnearnedPullRequestIsRefused:
    async def test_an_unvalidated_migration_opens_nothing(
        self, tmp_path: Path, repo_root: Path
    ) -> None:
        """The counterfactual that makes the happy path mean something.

        Same fixture, same repository, same code — validation switched off. If a
        PR appeared anyway, the policy would be decoration.
        """
        example = repo_root / "examples" / "customer_pipeline"
        empty_hub = InMemoryGitHub(repository="acme/data-platform")
        pipeline = LocalMigrationPipeline(
            Settings(
                llm_provider=LLMProvider.SCRIPTED,
                llm_fixture_dir=repo_root / "fixtures" / "llm",
                workspace_dir=tmp_path / "unvalidated",
            ),
            ScriptedModelClientFactory.from_fixture(
                repo_root / "fixtures" / "llm" / "customer_pipeline.json"
            ),
            run_validation=False,
            run_generated_tests=False,
            run_repair=False,
            run_optimization=False,
            github=GitHubClient(empty_hub, empty_hub.repository),
        )
        record = await pipeline.run(
            MigrationRequest(
                source_path=example / "legacy_pipeline.py", input_dir=example / "input"
            )
        )

        assert record.validation is None
        assert record.delivery.decision.disposition is DeliveryDisposition.REFUSED
        assert not record.delivery.opened
        assert empty_hub.pulls == []
        assert empty_hub.branches == {"main": empty_hub.branches["main"]}, (
            "a branch was created for a migration that earned no pull request"
        )

    async def test_nothing_is_pushed_when_github_is_not_configured(
        self, tmp_path: Path, repo_root: Path
    ) -> None:
        """A missing token is an ordinary local run, not a failed migration."""
        example = repo_root / "examples" / "customer_pipeline"
        pipeline = LocalMigrationPipeline(
            Settings(
                llm_provider=LLMProvider.SCRIPTED,
                llm_fixture_dir=repo_root / "fixtures" / "llm",
                workspace_dir=tmp_path / "no-github",
            ),
            ScriptedModelClientFactory.from_fixture(
                repo_root / "fixtures" / "llm" / "customer_pipeline.json"
            ),
            run_validation=False,
            run_generated_tests=False,
            run_repair=False,
            run_optimization=False,
            github=None,
        )
        record = await pipeline.run(
            MigrationRequest(
                source_path=example / "legacy_pipeline.py", input_dir=example / "input"
            )
        )
        assert not record.failed
        assert record.delivery.skipped_reason is not None
        assert "ETLM_GITHUB_TOKEN" in record.delivery.skipped_reason


class TestAFailedMigrationBecomesADraft:
    async def test_a_broken_migration_opens_a_labelled_draft(
        self, tmp_path: Path, repo_root: Path
    ) -> None:
        """The third outcome, and the one most systems get wrong.

        The `customer_pipeline_broken` fixture generates code that passes the
        static gate and fails validation. With repair disabled it stays broken.
        The work is still worth showing a human — as a draft that says so, never
        as something to approve.
        """
        example = repo_root / "examples" / "customer_pipeline"
        broken_hub = InMemoryGitHub(repository="acme/data-platform")
        pipeline = LocalMigrationPipeline(
            Settings(
                llm_provider=LLMProvider.SCRIPTED,
                llm_fixture_dir=repo_root / "fixtures" / "llm",
                workspace_dir=tmp_path / "broken",
            ),
            ScriptedModelClientFactory.from_fixture(
                repo_root / "fixtures" / "llm" / "customer_pipeline_broken.json"
            ),
            run_validation=True,
            run_generated_tests=False,
            run_repair=False,
            run_optimization=False,
            github=GitHubClient(broken_hub, broken_hub.repository),
        )
        record = await pipeline.run(
            MigrationRequest(
                source_path=example / "legacy_pipeline.py",
                input_dir=example / "input",
                scenario="customer_pipeline_broken",
            )
        )

        assert record.validation.report.status is not ValidationStatus.PASS
        outcome = record.delivery
        assert outcome.decision.disposition is DeliveryDisposition.DRAFT
        assert outcome.pull_request.draft
        assert "needs-human" in broken_hub.pulls[0].labels

        body = broken_hub.pulls[0].body
        assert "[!WARNING]" in body
        assert "should not be merged" in body
        assert "❌ **FAIL**" in body
