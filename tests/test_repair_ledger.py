"""Tests for the repair ledger — the thing that makes the attempt budget mean something.

Without it, "three attempts" means "three executions", which an oscillating
agent spends on two ideas. With it, an attempt is a *distinct idea*, and that
distinction is enforced by set membership rather than by asking a model to
remember what it already tried.

Everything here is pure: no LLM, no Spark, no server.
"""

from __future__ import annotations

import pytest

from etl_migrator.domain.code import GeneratedCode
from etl_migrator.domain.enums import RiskCategory, ValidationStatus
from etl_migrator.domain.repair import (
    RepairAttempt,
    RepairLedger,
    RepairOutcome,
    RepairProposal,
    RepairStrategy,
    code_fingerprint,
    summarise_history,
)

BASELINE = "def run(spark, input_dir, output_dir):\n    return None\n"


def code(body: str, filename: str = "p.py") -> GeneratedCode:
    return GeneratedCode(filename=filename, content=body)


def proposal(
    approach: str,
    body: str,
    category: RiskCategory = RiskCategory.NULL_SEMANTICS,
) -> RepairProposal:
    return RepairProposal(
        strategy=RepairStrategy(
            category=category, approach=approach, description="d", target_step_ids=["s1"]
        ),
        code=code(body),
        rationale="r",
        expected_effect="e",
    )


class TestStrategyIdentity:
    def test_signature_combines_cause_and_technique(self) -> None:
        strategy = RepairStrategy(
            category=RiskCategory.NULL_SEMANTICS,
            approach="filter_null_group_keys",
            description="d",
        )
        assert strategy.signature == "null_semantics:filter_null_group_keys"

    def test_same_technique_under_a_different_cause_is_a_different_strategy(self) -> None:
        """Filtering nulls to fix a row count and filtering nulls to fix a type
        coercion are different theories, and both deserve an attempt."""
        a = RepairStrategy(
            category=RiskCategory.NULL_SEMANTICS, approach="cast_key_columns", description="d"
        )
        b = RepairStrategy(
            category=RiskCategory.TYPE_COERCION, approach="cast_key_columns", description="d"
        )
        assert a.signature != b.signature

    @pytest.mark.parametrize("bad", ["Filter Nulls", "filter-nulls", "x", "filter nulls", ""])
    def test_approach_must_be_a_slug(self, bad: str) -> None:
        """Prose cannot be compared mechanically; a slug can."""
        with pytest.raises(ValueError):
            RepairStrategy(
                category=RiskCategory.NULL_SEMANTICS, approach=bad, description="d"
            )


class TestFingerprinting:
    def test_identical_code_has_the_same_fingerprint(self) -> None:
        assert code_fingerprint(code("x = 1\n")) == code_fingerprint(code("x = 1\n"))

    def test_comments_and_blank_lines_do_not_change_it(self) -> None:
        """A proposal differing only in commentary is the same attempt, and paying
        for a Spark run to discover that is the waste this prevents."""
        plain = code("x = 1\ny = 2\n")
        commented = code("# explains itself\nx = 1\n\n  # more\ny = 2\n")
        assert code_fingerprint(plain) == code_fingerprint(commented)

    def test_a_real_change_changes_it(self) -> None:
        assert code_fingerprint(code("x = 1\n")) != code_fingerprint(code("x = 2\n"))

    def test_filename_does_not_affect_it(self) -> None:
        assert code_fingerprint(code("x = 1\n", "a.py")) == code_fingerprint(
            code("x = 1\n", "b.py")
        )


class TestLedgerAdmission:
    def test_a_fresh_strategy_is_admitted(self) -> None:
        ledger = RepairLedger(max_attempts=3)
        ledger.register_baseline(code(BASELINE))
        admitted, reason = ledger.admits(proposal("filter_null_group_keys", "x = 1\n"))
        assert admitted and reason is None

    def test_a_repeated_signature_is_rejected(self) -> None:
        """The oscillation guard: A, B, A must not spend a third execution."""
        ledger = RepairLedger(max_attempts=3)
        ledger.register_baseline(code(BASELINE))
        first = proposal("filter_null_group_keys", "x = 1\n")
        ledger.record(first)

        admitted, reason = ledger.admits(proposal("filter_null_group_keys", "x = 99\n"))
        assert not admitted
        assert "already tried" in (reason or "")

    def test_returning_the_unchanged_code_is_rejected(self) -> None:
        """An agent that 'repairs' the failing code into itself has done nothing,
        and would otherwise cost a full re-validation to discover that."""
        ledger = RepairLedger(max_attempts=3)
        ledger.register_baseline(code(BASELINE))
        admitted, reason = ledger.admits(proposal("some_new_idea", BASELINE))
        assert not admitted
        assert "identical to the code that failed" in (reason or "")

    def test_relabelled_identical_code_is_rejected(self) -> None:
        """A new slug on byte-identical code is a new label, not a new idea."""
        ledger = RepairLedger(max_attempts=3)
        ledger.register_baseline(code(BASELINE))
        ledger.record(proposal("first_idea", "x = 1\n"))

        admitted, reason = ledger.admits(proposal("second_idea", "x = 1\n"))
        assert not admitted
        assert "identical code cannot behave differently" in (reason or "")

    def test_cosmetic_rewording_is_rejected(self) -> None:
        ledger = RepairLedger(max_attempts=3)
        ledger.register_baseline(code(BASELINE))
        ledger.record(proposal("first_idea", "x = 1\ny = 2\n"))

        reworded = proposal("second_idea", "# now with a comment\nx = 1\n\ny = 2\n")
        admitted, _ = ledger.admits(reworded)
        assert not admitted

    def test_a_genuinely_different_fix_is_admitted_after_a_failure(self) -> None:
        ledger = RepairLedger(max_attempts=3)
        ledger.register_baseline(code(BASELINE))
        ledger.record(proposal("filter_null_group_keys", "x = 1\n"))

        admitted, _ = ledger.admits(
            proposal("drop_synthesised_index", "x = 2\n", RiskCategory.INDEX_SEMANTICS)
        )
        assert admitted

    def test_tried_signatures_are_reportable(self) -> None:
        ledger = RepairLedger(max_attempts=2)
        ledger.register_baseline(code(BASELINE))
        ledger.record(proposal("a_first_idea", "x = 1\n"))
        assert ledger.tried_signatures == ["null_semantics:a_first_idea"]

    def test_a_zero_attempt_budget_is_rejected_at_construction(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            RepairLedger(max_attempts=0)


class TestOutcomeReporting:
    def _attempt(self, n: int, differences: int | None, admitted: bool = True) -> RepairAttempt:
        return RepairAttempt(
            attempt=n,
            strategy=RepairStrategy(
                category=RiskCategory.NULL_SEMANTICS, approach=f"idea_{n}", description="d"
            ),
            admitted=admitted,
            differences=differences,
            validation_status=ValidationStatus.FAIL if differences else None,
        )

    def test_best_attempt_is_the_nearest_miss_not_the_last(self) -> None:
        """When the budget runs out a human needs the closest attempt; the final
        one is often a wilder guess than the second."""
        outcome = RepairOutcome(
            attempts=[self._attempt(1, 5), self._attempt(2, 1), self._attempt(3, 9)]
        )
        best = outcome.best_attempt
        assert best is not None and best.attempt == 2

    def test_rejected_attempts_are_not_candidates_for_best(self) -> None:
        outcome = RepairOutcome(
            attempts=[self._attempt(1, 4), self._attempt(2, None, admitted=False)]
        )
        best = outcome.best_attempt
        assert best is not None and best.attempt == 1

    def test_ties_prefer_the_earlier_attempt(self) -> None:
        outcome = RepairOutcome(attempts=[self._attempt(1, 3), self._attempt(2, 3)])
        best = outcome.best_attempt
        assert best is not None and best.attempt == 1

    def test_no_scored_attempts_yields_no_best(self) -> None:
        assert RepairOutcome().best_attempt is None

    def test_render_shows_rejections_distinctly(self) -> None:
        outcome = RepairOutcome(
            attempts=[
                RepairAttempt(attempt=1, admitted=False, rejection_reason="already tried")
            ]
        )
        assert "REJECTED" in outcome.render()


class TestHistoryFeedback:
    def test_empty_history_says_so(self) -> None:
        assert "No previous" in summarise_history([])

    def test_history_tells_the_agent_what_is_spent(self) -> None:
        """The correction half of the loop: a repeat becomes a choice, not an
        accident — and then the ledger makes it a rejected one."""
        history = [
            RepairAttempt(
                attempt=1,
                strategy=RepairStrategy(
                    category=RiskCategory.NULL_SEMANTICS,
                    approach="filter_null_group_keys",
                    description="filter nulls before grouping",
                ),
                differences=2,
                validation_status=ValidationStatus.FAIL,
            )
        ]
        rendered = summarise_history(history)
        assert "do not repeat" in rendered
        assert "null_semantics:filter_null_group_keys" in rendered
        assert "filter nulls before grouping" in rendered
