"""The learning loop, tested for what it refuses to claim.

The failure mode here is not a crash. It is a system that has seen two
migrations, reports a 50% success rate with a straight face, and an agent that
acts on it. A confidence which rises the moment anything is observed is worse
than no confidence at all, because it looks like knowledge.

So most of this file is about the boundary: what happens at one, two and three
observations, and whether the thing that comes out says "not enough evidence" or
a number. That is the benchmarking discipline, refusing to answer an unanswerable
question — applied to learning instead of to benchmarks.

The other rule under test is what counts as a success. A repair succeeded only
if the differ then returned PASS; an optimisation succeeded only if
`evaluate_optimization` accepted it. Neither counts the agent's own opinion.

No JVM, no network.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from tests.helpers_records import benchmark, migration_record, strategy

from etl_migrator.domain.enums import RiskCategory, ValidationStatus
from etl_migrator.domain.history import (
    DISCOURAGED_RATE,
    MIN_SUPPORT,
    MigrationHistory,
    StrategyEvidence,
    harvest,
    outcomes_by_category,
)
from etl_migrator.domain.optimization import (
    BenchmarkComparison,
    OptimizationAttempt,
    OptimizationOutcome,
)
from etl_migrator.domain.repair import RepairAttempt, RepairOutcome, RepairStrategy
from etl_migrator.knowledge.history import RECORD_FILENAME, load_history, load_records


def repair_attempt(
    attempt: int,
    *,
    approach: str,
    category: RiskCategory = RiskCategory.NULL_SEMANTICS,
    fixed: bool,
    admitted: bool = True,
) -> RepairAttempt:
    return RepairAttempt(
        attempt=attempt,
        strategy=RepairStrategy(
            category=category, approach=approach, description=f"Apply {approach}."
        ),
        admitted=admitted,
        rejection_reason=None if admitted else "already tried",
        validation_status=ValidationStatus.PASS if fixed else ValidationStatus.FAIL,
    )


def optimization_attempt(
    attempt: int,
    *,
    approach: str,
    accepted: bool,
    baseline: float = 10.0,
    candidate: float = 8.0,
    predicted: float = 1.5,
    admitted: bool = True,
) -> OptimizationAttempt:
    return OptimizationAttempt(
        attempt=attempt,
        strategy=strategy(approach, expected=predicted),
        admitted=admitted,
        rejection_reason=None if admitted else "already measured",
        accepted=accepted,
        validation_status="PASS",
        comparison=BenchmarkComparison(
            baseline=benchmark([baseline] * 4), candidate=benchmark([candidate] * 4)
        )
        if admitted
        else None,
    )


def with_repairs(migration_id: str, attempts: list[RepairAttempt]):
    return migration_record(
        migration_id=migration_id,
        repair=RepairOutcome(
            succeeded=any(a.succeeded for a in attempts), attempts=attempts
        ),
    )


def with_optimizations(migration_id: str, attempts: list[OptimizationAttempt]):
    return migration_record(
        migration_id=migration_id,
        optimization=OptimizationOutcome(
            applied=any(a.accepted for a in attempts), attempts=attempts
        ),
    )


class TestSupportThreshold:
    """The refusal this module exists for."""

    @pytest.mark.parametrize("attempts", [1, 2])
    def test_below_the_threshold_there_is_no_rate(self, attempts: int) -> None:
        """Two repairs, one of which worked, is not a 50% success rate. It is
        two data points, and `rate` says so by being None."""
        evidence = StrategyEvidence(key="k", attempts=attempts, successes=attempts)
        assert evidence.rate is None
        assert not evidence.sufficient
        assert "not enough evidence" in evidence.render()

    def test_at_the_threshold_a_rate_appears(self) -> None:
        evidence = StrategyEvidence(key="k", attempts=MIN_SUPPORT, successes=MIN_SUPPORT)
        assert evidence.rate == 1.0
        assert evidence.sufficient
        assert "100%" in evidence.render()

    def test_a_thin_record_is_never_discouraged(self) -> None:
        """One failure must not blacklist a strategy. `discouraged` requires
        both enough evidence and a bad rate, in that order."""
        assert not StrategyEvidence(key="k", attempts=1, successes=0).discouraged
        assert not StrategyEvidence(key="k", attempts=2, successes=0).discouraged
        assert StrategyEvidence(key="k", attempts=MIN_SUPPORT, successes=0).discouraged

    def test_the_discouraged_boundary_is_where_it_says_it_is(self) -> None:
        one_in_three = StrategyEvidence(key="k", attempts=3, successes=1)
        two_in_three = StrategyEvidence(key="k", attempts=3, successes=2)
        assert one_in_three.rate is not None and one_in_three.rate <= DISCOURAGED_RATE
        assert one_in_three.discouraged
        assert not two_in_three.discouraged

    def test_a_thin_corpus_says_so_in_words(self) -> None:
        history = MigrationHistory(migrations_observed=2, validated=2)
        assert not history.sufficient
        assert "anecdote" in history.render()

    def test_an_empty_corpus_does_not_pretend(self) -> None:
        rendered = MigrationHistory().render()
        assert "No completed migrations" in rendered
        assert "%" not in rendered


class TestWhatCountsAsSuccess:
    def test_a_repair_counts_only_if_the_differ_then_passed(self) -> None:
        """Not if the agent said it would work, not if the code changed."""
        history = harvest([
            with_repairs("m1", [repair_attempt(1, approach="filter_nulls", fixed=True)]),
            with_repairs("m2", [repair_attempt(1, approach="filter_nulls", fixed=False)]),
        ])
        evidence = history.repair_evidence(RiskCategory.NULL_SEMANTICS, "filter_nulls")
        assert evidence.attempts == 2
        assert evidence.successes == 1

    def test_an_optimisation_counts_only_if_it_was_accepted(self) -> None:
        """Accepted means correctness held *and* the speedup was measurable and
        robust. A proposal that looked fast and was refused is an attempt."""
        history = harvest([
            with_optimizations(
                "m1", [optimization_attempt(1, approach="broadcast", accepted=False)]
            ),
            with_optimizations(
                "m2", [optimization_attempt(1, approach="broadcast", accepted=True)]
            ),
        ])
        evidence = history.optimization_evidence("broadcast")
        assert evidence.attempts == 2
        assert evidence.successes == 1

    def test_a_refused_attempt_is_not_counted_at_all(self) -> None:
        """The ledger refusing a repeated strategy says nothing about whether it
        works — it was never tried. Counting it as a failure would punish an
        approach for being proposed twice in one migration.
        """
        history = harvest([
            with_repairs("m1", [
                repair_attempt(1, approach="filter_nulls", fixed=False),
                repair_attempt(2, approach="filter_nulls", fixed=False, admitted=False),
            ])
        ])
        evidence = history.repair_evidence(RiskCategory.NULL_SEMANTICS, "filter_nulls")
        assert evidence.attempts == 1, "a ledger refusal was counted as a real attempt"

    def test_the_same_migration_is_never_counted_twice(self) -> None:
        """Artifacts get re-read. Double-counting would inflate confidence with
        no new information behind it."""
        record = with_repairs("m1", [repair_attempt(1, approach="filter_nulls", fixed=True)])
        history = harvest([record, record])
        assert history.migrations_observed == 1
        assert history.repair_evidence(RiskCategory.NULL_SEMANTICS, "filter_nulls").attempts == 1


class TestHarvest:
    def test_evidence_is_keyed_by_category_and_approach(self) -> None:
        """The same approach against a different root cause is a different
        question, and the key keeps them apart."""
        history = harvest([
            with_repairs("m1", [
                repair_attempt(
                    1, approach="declare_schema",
                    category=RiskCategory.TYPE_COERCION, fixed=True,
                ),
                repair_attempt(
                    2, approach="declare_schema",
                    category=RiskCategory.NULL_SEMANTICS, fixed=False,
                ),
            ])
        ])
        assert history.repair_evidence(RiskCategory.TYPE_COERCION, "declare_schema").successes == 1
        assert history.repair_evidence(RiskCategory.NULL_SEMANTICS, "declare_schema").successes == 0

    def test_an_unseen_strategy_returns_empty_evidence_not_none(self) -> None:
        """"Nothing known" is a real answer, and callers should not have to
        branch on None to hear it."""
        evidence = harvest([]).repair_evidence(RiskCategory.ROW_ORDER, "never_tried")
        assert evidence.attempts == 0
        assert evidence.rate is None

    def test_measured_speedups_are_recorded_only_for_accepted_attempts(self) -> None:
        history = harvest([
            with_optimizations("m1", [
                optimization_attempt(1, approach="shuffle", accepted=True, candidate=5.0),
            ]),
            with_optimizations("m2", [
                optimization_attempt(1, approach="shuffle", accepted=False, candidate=1.0),
            ]),
        ])
        evidence = history.optimization_evidence("shuffle")
        assert evidence.speedups == [2.0], "a rejected ratio leaked into the record"

    def test_agent_optimism_is_quantified_from_matched_pairs(self) -> None:
        """The agent's own calibration, measured rather than asserted — it
        predicted 2.0x three times and delivered 1.25x."""
        history = harvest([
            with_optimizations(f"m{i}", [
                optimization_attempt(
                    1, approach="shuffle", accepted=True, candidate=8.0, predicted=2.0
                )
            ])
            for i in range(MIN_SUPPORT)
        ])
        evidence = history.optimization_evidence("shuffle")
        assert evidence.median_speedup == pytest.approx(1.25)
        assert evidence.optimism == pytest.approx(0.75)
        assert "optimistic" in evidence.render()

    def test_optimism_needs_support_too(self) -> None:
        history = harvest([
            with_optimizations("m1", [
                optimization_attempt(1, approach="shuffle", accepted=True, predicted=5.0)
            ])
        ])
        assert history.optimization_evidence("shuffle").optimism is None

    def test_validated_migrations_are_counted(self) -> None:
        history = harvest([
            migration_record(migration_id="m1", status=ValidationStatus.PASS),
            migration_record(migration_id="m2", status=ValidationStatus.FAIL),
            migration_record(migration_id="m3", with_validation=False),
        ])
        assert history.migrations_observed == 3
        assert history.validated == 1

    def test_harvest_is_deterministic(self) -> None:
        """It is recomputed from artifacts on every call, so the same corpus
        must always produce the same answer — otherwise the evidence an agent
        sees would depend on when it asked."""
        records = [
            with_repairs("m1", [repair_attempt(1, approach="filter_nulls", fixed=True)]),
            with_optimizations(
                "m2", [optimization_attempt(1, approach="reduce_shuffle", accepted=True)]
            ),
        ]
        assert harvest(records) == harvest(records)

    def test_discouraged_strategies_are_listed(self) -> None:
        history = harvest([
            with_repairs(f"m{i}", [repair_attempt(1, approach="hopeless", fixed=False)])
            for i in range(MIN_SUPPORT)
        ])
        discouraged = history.discouraged_repairs()
        assert [e.key for e in discouraged] == ["null_semantics/hopeless"]
        assert "rarely works" in discouraged[0].render()


class TestOutcomesByCategory:
    def test_it_counts_passes_per_declared_category(self) -> None:
        """Which kinds of semantic difference actually cause trouble — measured
        rather than assumed."""
        outcomes = outcomes_by_category([
            migration_record(migration_id="m1", status=ValidationStatus.PASS),
            migration_record(migration_id="m2", status=ValidationStatus.FAIL),
        ])
        # The helper's plan declares one null_semantics difference.
        assert outcomes["null_semantics"] == (1, 2)

    def test_a_migration_without_validation_is_not_counted(self) -> None:
        """It has no outcome, and recording it as a failure would blame a
        category for a migration that was never checked."""
        assert outcomes_by_category(
            [migration_record(migration_id="m1", with_validation=False)]
        ) == {}


class TestLoadingFromDisk:
    def write(self, workspace: Path, record) -> None:
        directory = workspace / record.migration_id
        directory.mkdir(parents=True, exist_ok=True)
        (directory / RECORD_FILENAME).write_text(record.model_dump_json(indent=2))

    def test_records_round_trip_through_the_artifacts(self, tmp_path: Path) -> None:
        """The corpus is the files the system already writes — no separate
        store to fall out of step with what happened."""
        self.write(
            tmp_path,
            with_repairs("m1", [repair_attempt(1, approach="filter_nulls", fixed=True)]),
        )
        history = load_history(tmp_path)
        assert history.migrations_observed == 1
        assert history.repair_evidence(RiskCategory.NULL_SEMANTICS, "filter_nulls").successes == 1

    def test_a_missing_workspace_is_an_empty_history(self, tmp_path: Path) -> None:
        assert load_history(tmp_path / "nope").migrations_observed == 0

    def test_a_corrupt_record_is_skipped_not_raised(self, tmp_path: Path) -> None:
        """The history is advisory. Refusing to plan a migration because an
        unrelated one from last month has a truncated JSON file would be a poor
        trade.
        """
        self.write(tmp_path, migration_record(migration_id="good"))
        broken = tmp_path / "broken"
        broken.mkdir()
        (broken / RECORD_FILENAME).write_text("{not json")

        history = load_history(tmp_path)
        assert history.migrations_observed == 1

    def test_a_record_missing_required_fields_is_skipped(self, tmp_path: Path) -> None:
        self.write(tmp_path, migration_record(migration_id="good"))
        wrong = tmp_path / "wrong"
        wrong.mkdir()
        (wrong / RECORD_FILENAME).write_text(json.dumps({"unexpected": True}))
        assert load_history(tmp_path).migrations_observed == 1

    def test_the_limit_takes_the_most_recent(self, tmp_path: Path) -> None:
        """A worker running for months should not read ten thousand files to
        answer one tool call, and the recent past is the relevant one."""
        import os
        import time

        for index in range(5):
            record = migration_record(migration_id=f"m{index}")
            self.write(tmp_path, record)
            path = tmp_path / record.migration_id / RECORD_FILENAME
            stamp = time.time() + index
            os.utime(path, (stamp, stamp))

        recent = load_records(tmp_path, limit=2)
        assert [r.migration_id for r in recent] == ["m4", "m3"]


class TestAgentsSeeTheEvidence:
    """The point of all of it: an agent can ask before it proposes."""

    @staticmethod
    def registered_tools(module_name: str, class_name: str) -> set[str]:
        """Tool function names actually passed to `tools=` on the agent.

        Read from the AST, because what matters is registration: a tool factory
        that exists but was never added to the list is a method nobody calls,
        and a prompt naming a tool that is not registered makes the agent try
        to call something that does not exist.
        """
        import ast
        import importlib
        import inspect

        tree = ast.parse(
            Path(inspect.getfile(importlib.import_module(module_name))).read_text("utf-8")
        )
        cls = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef) and node.name == class_name
        )
        # The factory methods named in `tools=[...]`, mapped to the inner
        # function each one defines and returns.
        registered = {
            ast.unparse(element).removeprefix("self.").removesuffix("()")
            for call in ast.walk(cls)
            if isinstance(call, ast.Call)
            for keyword in call.keywords
            if keyword.arg == "tools" and isinstance(keyword.value, ast.List)
            for element in keyword.value.elts
        }
        names: set[str] = set()
        for method in ast.walk(cls):
            if isinstance(method, ast.FunctionDef) and method.name in registered:
                names |= {
                    inner.name
                    for inner in ast.walk(method)
                    if isinstance(inner, ast.FunctionDef) and inner is not method
                }
        return names

    def test_the_repair_agent_registers_the_track_record_tool(self) -> None:
        tools = self.registered_tools("etl_migrator.agents.repair", "RepairAgent")
        assert "strategy_track_record" in tools, f"registered: {sorted(tools)}"

    def test_the_optimizer_registers_the_track_record_tool(self) -> None:
        tools = self.registered_tools("etl_migrator.agents.optimizer", "OptimizerAgent")
        assert "approach_track_record" in tools, f"registered: {sorted(tools)}"

    def test_both_prompts_name_the_tool_they_register(self) -> None:
        """A registered tool the prompt never mentions is one the agent will not
        think to call."""
        from etl_migrator.agents.optimizer import SYSTEM_MESSAGE as OPTIMIZER
        from etl_migrator.agents.repair import SYSTEM_MESSAGE as REPAIR

        assert "strategy_track_record" in REPAIR
        assert "approach_track_record" in OPTIMIZER

    def test_an_unseen_strategy_is_reported_as_unseen(self) -> None:
        """And explicitly not as a reason to avoid it — absence of evidence is
        not evidence of absence, and an agent told otherwise would narrow its
        options for no reason."""
        evidence = MigrationHistory().repair_evidence(RiskCategory.ROW_ORDER, "new_idea")
        assert evidence.attempts == 0
        assert evidence.rate is None


class TestTheReadmeExampleIsReal:
    """The README shows sample `render()` output. A hand-typed approximation of
    it is a small lie that survives every test in this file, so the sample is
    generated here and checked against the file verbatim.

    This is not hypothetical: the optimism figure in that block was wrong when
    it was written, and nothing else in the suite could have caught it.
    """

    #: Exactly the four cases the README's Historical Learning block shows.
    CASES = (
        StrategyEvidence(key="null_semantics/coalesce_sum", attempts=2, successes=1),
        StrategyEvidence(key="null_semantics/filter_null_keys", attempts=4, successes=4),
        StrategyEvidence(key="broadcast_small_side", attempts=4, successes=0),
        StrategyEvidence(
            key="reduce_shuffle_partitions",
            attempts=5,
            successes=3,
            speedups=[1.20, 1.31, 1.44],
            predicted_speedups=[1.7, 1.8, 1.8],
        ),
    )

    @classmethod
    def readme(cls) -> str:
        return (Path(__file__).resolve().parents[1] / "README.md").read_text("utf-8")

    @pytest.mark.parametrize("evidence", CASES, ids=lambda e: e.key)
    def test_the_readme_quotes_what_the_code_prints(
        self, evidence: StrategyEvidence
    ) -> None:
        line = evidence.render()
        assert line in self.readme(), f"README does not contain this line verbatim:\n{line}"

    def test_the_check_would_notice_a_drifting_number(self) -> None:
        """Guard the guard: prove the assertion above is sensitive to the digits
        and not passing on the strategy name alone."""
        drifted = StrategyEvidence(
            key="reduce_shuffle_partitions",
            attempts=5,
            successes=3,
            speedups=[1.20, 1.31, 1.44],
            predicted_speedups=[2.9, 2.9, 2.9],
        )
        assert drifted.render() not in self.readme()
