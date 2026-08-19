"""The optimisation loop end to end, against real Spark and a real stopwatch.

This is the test that justifies the optimiser. Everything in `test_optimization.py`
proves the acceptance rule is correct on timings chosen by hand; this proves the
rule survives contact with a real JVM, where durations are noisy, warm-up costs
are large, and the effect being measured is a couple of seconds.

The `customer_pipeline_slow` fixture records a planner that emits a deliberately
naive Spark configuration — adaptive query execution disabled, a static 200
shuffle partitions — for a 230 KB input that groups into five countries. That is
a real inefficiency, not a contrived one: Spark schedules ~400 tasks for a job
whose output is five rows, and with AQE off it cannot coalesce them afterwards.
The optimiser has to notice it from the measured metrics and propose the fix.

Two things are asserted, and the second matters more than the first:

* a genuine speedup is accepted, with the numbers to back it;
* the acceptance is *earned* — the same machinery refuses a change that is
  faster but wrong, and refuses one whose measurement is too noisy to read.

Marked `spark`: this executes Spark roughly twenty times and takes minutes.
`pytest -m "not spark"` skips it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from etl_migrator.config import LLMProvider, Settings
from etl_migrator.domain.code import GeneratedCode
from etl_migrator.domain.enums import MigrationStage, ValidationStatus
from etl_migrator.domain.optimization import (
    DEFAULT_MAX_NOISE_RATIO,
    DEFAULT_MIN_SPEEDUP,
    BenchmarkComparison,
    evaluate_optimization,
)
from etl_migrator.domain.plan import ExecutionStrategy
from etl_migrator.llm.factory import ScriptedModelClientFactory
from etl_migrator.pipeline import steps
from etl_migrator.pipeline.local import LocalMigrationPipeline, MigrationRequest

pytestmark = pytest.mark.spark

pytest.importorskip("pyspark", reason="pyspark extra not installed")

SLOW_SCENARIO = "customer_pipeline_slow"


@pytest.fixture(scope="module")
def slow_payload(repo_root: Path) -> dict[str, Any]:
    path = repo_root / "fixtures" / "llm" / f"{SLOW_SCENARIO}.json"
    return json.loads(path.read_text(encoding="utf-8"))  # type: ignore[no-any-return]


@pytest.fixture(scope="module")
def slow_code(slow_payload: dict[str, Any]) -> GeneratedCode:
    return GeneratedCode.model_validate(slow_payload["spark_engineer"][-1]["content"])


@pytest.fixture(scope="module")
async def migrated(
    tmp_path_factory: pytest.TempPathFactory, repo_root: Path
) -> Any:
    """Run the whole migration, optimisation stage included."""
    example = repo_root / "examples" / "customer_pipeline"
    settings = Settings(
        llm_provider=LLMProvider.SCRIPTED,
        llm_fixture_dir=repo_root / "fixtures" / "llm",
        workspace_dir=tmp_path_factory.mktemp("optimize"),
        log_format="json",
    )
    factory = ScriptedModelClientFactory.from_fixture(
        repo_root / "fixtures" / "llm" / f"{SLOW_SCENARIO}.json"
    )
    pipeline = LocalMigrationPipeline(
        settings,
        factory,
        run_validation=True,
        run_generated_tests=False,
        run_optimization=True,
        max_optimization_attempts=2,
    )
    return await pipeline.run(
        MigrationRequest(
            source_path=example / "legacy_pipeline.py",
            input_dir=example / "input",
            scenario=SLOW_SCENARIO,
        )
    )


class TestMeasuredAcceptance:
    """A real optimisation, accepted on real numbers."""

    def test_the_optimisation_stage_ran_and_kept_something(self, migrated: Any) -> None:
        outcome = migrated.optimization
        assert outcome is not None
        assert outcome.applied, (
            "no optimisation was kept; the measured verdicts were:\n"
            + "\n".join(a.render() for a in outcome.attempts)
        )

    def test_the_kept_change_clears_the_threshold_on_measured_time(
        self, migrated: Any
    ) -> None:
        outcome = migrated.optimization
        assert outcome.baseline is not None and outcome.final is not None
        assert outcome.speedup >= DEFAULT_MIN_SPEEDUP, outcome.render()
        # Real wall-clock seconds, not a modelled estimate.
        assert outcome.baseline.median > 0
        assert outcome.final.median < outcome.baseline.median

    def test_the_measurement_it_rests_on_is_readable(self, migrated: Any) -> None:
        """The gain has to be bigger than the error bar around it.

        Without this assertion the suite would pass on a lucky pair of runs, and
        the acceptance rule would be an ornament.
        """
        accepted = [a for a in migrated.optimization.attempts if a.accepted]
        assert len(accepted) == 1
        comparison = accepted[0].comparison
        assert comparison is not None
        assert not comparison.inconclusive
        assert comparison.robust
        assert comparison.baseline.noise_ratio <= DEFAULT_MAX_NOISE_RATIO
        assert comparison.candidate.noise_ratio <= DEFAULT_MAX_NOISE_RATIO
        assert comparison.baseline.samples >= 4

    def test_correctness_was_re_established_after_the_change(
        self, migrated: Any
    ) -> None:
        """The optimised pipeline was re-validated in full, not assumed correct."""
        accepted = [a for a in migrated.optimization.attempts if a.accepted]
        assert accepted[0].validation_status == ValidationStatus.PASS.value
        assert migrated.validation is not None
        assert migrated.validation.report.status is ValidationStatus.PASS

    def test_spark_confirms_the_change_actually_took_effect(
        self, migrated: Any
    ) -> None:
        """The configuration reached the JVM; the speedup is not a coincidence.

        Spark's own `statusTracker` reports the task count, and it collapses from
        several hundred to a handful — which is the mechanism the optimiser
        claimed. A speedup with unchanged metrics would mean something else got
        faster.
        """
        outcome = migrated.optimization
        base_metrics = outcome.baseline.metrics
        final_metrics = outcome.final.metrics
        assert base_metrics is not None and final_metrics is not None
        assert base_metrics.adaptive_enabled is False
        assert base_metrics.shuffle_partitions == 200
        assert final_metrics.adaptive_enabled is True
        assert final_metrics.tasks < base_metrics.tasks

    def test_the_agents_claim_is_recorded_but_not_believed(
        self, migrated: Any
    ) -> None:
        """`expected_speedup` is kept for comparison, and the verdict quotes the
        measurement instead."""
        accepted = next(a for a in migrated.optimization.attempts if a.accepted)
        assert accepted.strategy.expected_speedup > 1.0
        measured = migrated.optimization.speedup
        assert accepted.verdict == (
            f"accepted: {measured:.2f}x faster, validation still PASS"
        )

    def test_the_second_attempt_declines_rather_than_inventing_work(
        self, migrated: Any
    ) -> None:
        """With the opportunity taken, the honest move is to stop."""
        attempts = migrated.optimization.attempts
        assert len(attempts) == 1, [a.render() for a in attempts]
        # One accepted change per run keeps every number attributable to it.
        assert attempts[0].accepted

    def test_the_stage_is_recorded_in_the_lifecycle(self, migrated: Any) -> None:
        stages = [e.stage for e in migrated.stages]
        assert MigrationStage.OPTIMIZATION in stages
        entry = next(e for e in migrated.stages if e.stage is MigrationStage.OPTIMIZATION)
        assert entry.ended_at is not None
        assert entry.succeeded is True
        assert "x applied" in (entry.detail or "")

    def test_the_accepted_configuration_is_what_the_migration_carries_forward(
        self, migrated: Any
    ) -> None:
        """A verdict nobody acts on is a log line, not an optimisation."""
        assert migrated.plan is not None
        strategy = migrated.plan.execution_strategy
        assert strategy.adaptive_query_execution is True
        assert strategy.shuffle_partitions == 8


class TestTheAcceptanceIsEarned:
    """The same machinery, shown refusing.

    These run against real Spark too, because a rule that only refuses synthetic
    data is not evidence about the system that ships.
    """

    def test_an_unchanged_pipeline_does_not_measure_as_faster(
        self, migrated: Any, slow_code: GeneratedCode, repo_root: Path, tmp_path: Path
    ) -> None:
        """Benchmark the accepted configuration against itself.

        This is the null experiment, and it is the one that would expose a
        biased harness: if the second benchmark of an identical pipeline came
        out 10% faster — because warm-ups leaked, or because the OS page cache
        was warmer the second time — every verdict in this file would be
        worthless.
        """
        module = steps.materialize(tmp_path, slow_code.filename, slow_code.content)
        inputs = repo_root / "examples" / "customer_pipeline" / "input"
        strategy = ExecutionStrategy(shuffle_partitions=8, adaptive_query_execution=True)
        first = steps.benchmark(
            label="first", module_path=module, input_dir=inputs,
            output_dir=tmp_path / "first", strategy=strategy, runs=4,
        )
        second = steps.benchmark(
            label="second", module_path=module, input_dir=inputs,
            output_dir=tmp_path / "second", strategy=strategy, runs=4,
        )
        assert not first.failed and not second.failed

        comparison = BenchmarkComparison(baseline=first, candidate=second)
        accepted, verdict = evaluate_optimization(
            validation=migrated.validation.report, comparison=comparison
        )
        assert not accepted, (
            "an identical pipeline measured as an improvement — the harness is "
            f"biased:\n{comparison.render()}"
        )
        assert verdict.startswith("rejected")

    def test_a_faster_but_wrong_pipeline_is_rejected(self, migrated: Any) -> None:
        """The measured speedup, paired with a failing validation.

        Reuses the real benchmark comparison that was accepted above and swaps
        in a genuinely failed report, so the only thing that changed is
        correctness. It must be enough on its own to reject.
        """
        accepted_attempt = next(
            a for a in migrated.optimization.attempts if a.accepted
        )
        comparison = accepted_attempt.comparison
        assert comparison is not None and comparison.improved

        wrong = migrated.validation.report.model_copy(
            update={"error": "row counts differ: 4 != 5"}
        )
        assert wrong.status is ValidationStatus.ERROR

        accepted, verdict = evaluate_optimization(
            validation=wrong, comparison=comparison
        )
        assert not accepted
        assert "regression" in verdict
