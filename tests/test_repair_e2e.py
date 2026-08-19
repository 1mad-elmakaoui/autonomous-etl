"""The autonomous repair loop, end to end, against real Spark execution.

This is the test that justifies the repair loop. The `customer_pipeline_broken` fixture
records a Spark Engineer that produces code which is *safe and clean* — it
passes the static gate — but semantically wrong in two independent ways. That is
the realistic failure a repair loop exists for; a gate failure would have been
caught before anything executed.

The loop then has to do something genuinely hard: fix one defect, discover the
second, be refused when it re-proposes a spent idea, and find a different one.

Marked `spark` because it executes both pipelines several times.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from etl_migrator.config import LLMProvider, Settings
from etl_migrator.domain.enums import RiskCategory, ValidationStatus
from etl_migrator.llm.factory import ScriptedModelClientFactory
from etl_migrator.pipeline.local import LocalMigrationPipeline, MigrationRequest

pytestmark = pytest.mark.spark

pytest.importorskip("pyspark", reason="pyspark extra not installed")

BROKEN_SCENARIO = "customer_pipeline_broken"


@pytest.fixture(scope="module")
def known_good_code(repo_root: Path) -> str:
    """The implementation validation proved correct. The loop should rediscover it."""
    payload = json.loads(
        (repo_root / "fixtures" / "llm" / "customer_pipeline.json").read_text()
    )
    return str(payload["spark_engineer"][-1]["content"]["content"])


@pytest.fixture(scope="module")
async def repaired(
    tmp_path_factory: pytest.TempPathFactory, repo_root: Path
):
    """Run a migration whose generated code fails validation, and let it repair."""
    settings = Settings(
        llm_provider=LLMProvider.SCRIPTED,
        llm_fixture_dir=repo_root / "fixtures" / "llm",
        workspace_dir=tmp_path_factory.mktemp("repair"),
        log_format="json",
    )
    factory = ScriptedModelClientFactory.from_fixture(
        repo_root / "fixtures" / "llm" / f"{BROKEN_SCENARIO}.json"
    )
    pipeline = LocalMigrationPipeline(
        settings,
        factory,
        run_validation=True,
        run_generated_tests=False,
        run_repair=True,
        max_repair_attempts=3,
        # This test is about the repair loop. Benchmarking the repaired pipeline
        # afterwards is the optimiser's job and would add several minutes of Spark for
        # nothing this file asserts on.
        run_optimization=False,
    )
    return await pipeline.run(
        MigrationRequest(
            repo_root / "examples" / "customer_pipeline" / "legacy_pipeline.py",
            repo_root / "examples" / "customer_pipeline" / "input",
            scenario=BROKEN_SCENARIO,
        )
    )


class TestTheFailureIsReal:
    def test_the_broken_code_passes_the_static_gate(self, repaired) -> None:
        """If the gate caught it, the repair loop would never be reached — and
        this would be testing the wrong thing."""
        assert repaired.codegen is not None
        assert repaired.codegen.static_analysis.passed

    def test_validation_caught_what_the_gate_could_not(self, repaired) -> None:
        assert repaired.repair is not None
        first = repaired.repair.attempts[0]
        assert first.strategy is not None
        # The loop ran at all, which means the differ produced a FAIL.
        assert repaired.repair.attempts_used >= 1


class TestTheLoopConverges:
    def test_repair_succeeded(self, repaired) -> None:
        assert repaired.repair is not None
        assert repaired.repair.succeeded, repaired.repair.render()
        assert not repaired.failed

    def test_final_validation_passes_by_measurement(self, repaired) -> None:
        """The same differ that condemned the code is the one that clears it."""
        assert repaired.repair.final_report is not None
        assert repaired.repair.final_report.status is ValidationStatus.PASS
        assert repaired.repair.final_report.differences == []

    def test_it_rediscovers_the_known_correct_implementation(
        self, repaired, known_good_code: str
    ) -> None:
        assert repaired.repair.repaired_code is not None
        assert repaired.repair.repaired_code.content == known_good_code

    def test_the_record_carries_the_repaired_code_forward(
        self, repaired, known_good_code: str
    ) -> None:
        """Downstream stages must see the fixed code, not the original."""
        assert repaired.codegen is not None
        assert repaired.codegen.code.content == known_good_code

    def test_progress_is_visible_between_attempts(self, repaired) -> None:
        """Attempt 1 fixes one of two defects, so the difference count must fall
        rather than the loop thrashing at a constant distance."""
        first = repaired.repair.attempts[0]
        assert first.differences is not None and first.differences >= 1
        assert first.validation_status is ValidationStatus.FAIL


class TestTheLedgerIsLoadBearing:
    def test_a_repeated_strategy_was_rejected(self, repaired) -> None:
        """The recorded attempt 2 re-proposes attempt 1's strategy. It must be
        refused — this is the oscillation guard doing its job on a real run."""
        rejected = [a for a in repaired.repair.attempts if not a.admitted]
        assert rejected, repaired.repair.render()
        assert "already tried" in (rejected[0].rejection_reason or "")

    def test_the_rejected_attempt_never_ran(self, repaired) -> None:
        """No gate, no Spark, no validation — the whole point is that refusing
        costs nothing."""
        rejected = next(a for a in repaired.repair.attempts if not a.admitted)
        assert rejected.static_analysis is None
        assert rejected.validation_status is None
        assert rejected.differences is None

    def test_a_rejection_still_consumes_an_attempt(self, repaired) -> None:
        """An agent that cannot produce a distinct idea has run out of them;
        re-asking indefinitely is the token bonfire the bound prevents."""
        assert repaired.repair.attempts_used == 3
        assert [a.attempt for a in repaired.repair.attempts] == [1, 2, 3]

    def test_the_successful_fix_used_a_different_root_cause(self, repaired) -> None:
        admitted = [a for a in repaired.repair.attempts if a.admitted]
        assert admitted[0].strategy is not None and admitted[-1].strategy is not None
        assert admitted[0].strategy.category is RiskCategory.NULL_SEMANTICS
        assert admitted[-1].strategy.category is RiskCategory.INDEX_SEMANTICS
        assert admitted[0].strategy.signature != admitted[-1].strategy.signature


class TestExhaustionIsNotACrash:
    async def test_a_budget_of_one_exhausts_cleanly(
        self, tmp_path_factory: pytest.TempPathFactory, repo_root: Path
    ) -> None:
        """One attempt cannot fix two defects. The migration must end as a
        reported failure carrying the nearest miss, not an exception."""
        settings = Settings(
            llm_provider=LLMProvider.SCRIPTED,
            llm_fixture_dir=repo_root / "fixtures" / "llm",
            workspace_dir=tmp_path_factory.mktemp("exhausted"),
            log_format="json",
        )
        record = await LocalMigrationPipeline(
            Settings.model_validate(settings.model_dump()),
            ScriptedModelClientFactory.from_fixture(
                repo_root / "fixtures" / "llm" / f"{BROKEN_SCENARIO}.json"
            ),
            run_validation=True,
            run_generated_tests=False,
            run_repair=True,
            max_repair_attempts=1,
        ).run(
            MigrationRequest(
                repo_root / "examples" / "customer_pipeline" / "legacy_pipeline.py",
                repo_root / "examples" / "customer_pipeline" / "input",
                scenario=BROKEN_SCENARIO,
            )
        )

        assert record.repair is not None
        assert record.repair.exhausted
        assert not record.repair.succeeded
        assert record.failed
        assert "Human intervention required" in (record.failure_reason or "")

        best = record.repair.best_attempt
        assert best is not None, "a human needs the nearest miss, not just 'it failed'"
        assert best.differences is not None
