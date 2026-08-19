"""What past migrations established, and what they did not.

Deliberately not RAG. There is no embedding and no similarity search: a lookup
is a dictionary access on a typed key, `(RiskCategory, approach)` for a repair
and `approach` for an optimisation, so an agent asking "has this been tried?"
gets an answer about that thing rather than something adjacent. The evidence
comes from the `MigrationRecord`s the system already writes.

The point of the file is to make one refusal easy: saying "not enough evidence"
instead of a number. Two repairs, one of which worked, is not a 50% success
rate. So `StrategyEvidence.rate` is `None` below `MIN_SUPPORT` and `render()`
says so in words.

What counts as a success is held to the same standard as everywhere else. A
repair worked only if the differ then returned PASS, not if the agent said it
would. An optimisation worked only if `evaluate_optimization` accepted it, which
means correctness held and the speedup was measurable and robust.
"""

from __future__ import annotations

import statistics
from collections import defaultdict

from pydantic import Field, computed_field

from etl_migrator.domain.artifacts import MigrationRecord
from etl_migrator.domain.enums import RiskCategory, ValidationStatus
from etl_migrator.domain.spec import StrictModel

#: Observations required before a rate is reported at all. Three is not a
#: statistically satisfying number — it is the smallest one where "it worked
#: every time" is worth saying out loud, and it is stated here so nobody has to
#: guess what the threshold is.
MIN_SUPPORT = 3

#: A strategy that has failed this often, with support behind it, is worth
#: warning about explicitly rather than leaving an agent to infer from a rate.
DISCOURAGED_RATE = 0.34


class StrategyEvidence(StrictModel):
    """The track record of one named strategy.

    `rate` is `None` rather than a number when support is thin, and every caller
    has to handle that. That is the point: an optional value forces the question
    "do we actually know?" at the call site, where a default of 0.5 would have
    quietly answered it wrong.
    """

    key: str = Field(description="The strategy, e.g. 'null_semantics/filter_nulls'.")
    attempts: int = Field(default=0, ge=0)
    successes: int = Field(default=0, ge=0)
    #: Measured speedups of accepted optimisations. Empty for repairs.
    speedups: list[float] = Field(default_factory=list)
    #: What the agent predicted, for the attempts where it predicted anything.
    #: Kept so over-optimism can be quantified rather than asserted.
    predicted_speedups: list[float] = Field(default_factory=list)
    migrations: list[str] = Field(
        default_factory=list, description="Ids this evidence came from, for audit."
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def sufficient(self) -> bool:
        return self.attempts >= MIN_SUPPORT

    @computed_field  # type: ignore[prop-decorator]
    @property
    def rate(self) -> float | None:
        """Success rate, or None when there is not enough to say."""
        if not self.sufficient or self.attempts == 0:
            return None
        return self.successes / self.attempts

    @computed_field  # type: ignore[prop-decorator]
    @property
    def discouraged(self) -> bool:
        """Enough evidence, and most of it bad."""
        rate = self.rate
        return rate is not None and rate <= DISCOURAGED_RATE

    @property
    def median_speedup(self) -> float | None:
        return statistics.median(self.speedups) if self.speedups else None

    @property
    def optimism(self) -> float | None:
        """How much the agent over-promised, as predicted minus measured.

        Positive means it claimed more than it delivered. Reported only when
        both sides have been observed the required number of times, because a
        calibration figure from one attempt is a coincidence with a decimal
        point.
        """
        if len(self.speedups) < MIN_SUPPORT or len(self.predicted_speedups) < MIN_SUPPORT:
            return None
        return statistics.median(self.predicted_speedups) - statistics.median(self.speedups)

    def render(self) -> str:
        if not self.sufficient:
            return (
                f"{self.key}: {self.successes}/{self.attempts} — not enough evidence to "
                f"report a rate (needs {MIN_SUPPORT} attempts)"
            )
        rate = self.rate
        assert rate is not None  # implied by `sufficient`
        parts = [f"{self.key}: {self.successes}/{self.attempts} succeeded ({rate:.0%})"]
        if self.discouraged:
            parts.append("— rarely works; prefer something else")
        median = self.median_speedup
        if median is not None:
            parts.append(f"— median measured speedup {median:.2f}x")
        optimism = self.optimism
        if optimism is not None and optimism > 0.05:
            parts.append(f"(predictions have run {optimism:.2f}x optimistic)")
        return " ".join(parts)


class MigrationHistory(StrictModel):
    """Everything learned from completed migrations.

    Keyed rather than searched. `repair_strategies` is keyed by
    `"<category>/<approach>"` and `optimization_approaches` by the approach
    slug, which is exactly the vocabulary the agents already emit — so an agent
    can look up the thing it is about to propose, before proposing it.
    """

    migrations_observed: int = Field(default=0, ge=0)
    validated: int = Field(
        default=0, ge=0, description="Migrations whose differ verdict was PASS."
    )
    repair_strategies: dict[str, StrategyEvidence] = Field(default_factory=dict)
    optimization_approaches: dict[str, StrategyEvidence] = Field(default_factory=dict)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def sufficient(self) -> bool:
        """Whether the corpus as a whole is worth consulting."""
        return self.migrations_observed >= MIN_SUPPORT

    def repair_evidence(self, category: RiskCategory | str, approach: str) -> StrategyEvidence:
        """Evidence for one repair strategy. Never None — an unseen strategy is
        a real answer ("nothing known") and callers should not have to branch."""
        name = category.value if isinstance(category, RiskCategory) else category
        key = f"{name}/{approach}"
        return self.repair_strategies.get(key, StrategyEvidence(key=key))

    def optimization_evidence(self, approach: str) -> StrategyEvidence:
        return self.optimization_approaches.get(approach, StrategyEvidence(key=approach))

    def discouraged_repairs(self) -> list[StrategyEvidence]:
        return sorted(
            (e for e in self.repair_strategies.values() if e.discouraged),
            key=lambda e: e.key,
        )

    def discouraged_optimizations(self) -> list[StrategyEvidence]:
        return sorted(
            (e for e in self.optimization_approaches.values() if e.discouraged),
            key=lambda e: e.key,
        )

    def render(self) -> str:
        if self.migrations_observed == 0:
            return (
                "No completed migrations on record yet. Nothing here has been "
                "learned, and this tool will say so rather than guess."
            )
        lines = [
            f"{self.migrations_observed} migration(s) on record, "
            f"{self.validated} validated."
        ]
        if not self.sufficient:
            lines.append(
                f"Fewer than {MIN_SUPPORT} migrations: treat everything below as "
                "anecdote, not evidence."
            )
        if self.repair_strategies:
            lines += ["", "repair strategies:"]
            lines += [
                f"  {self.repair_strategies[k].render()}"
                for k in sorted(self.repair_strategies)
            ]
        if self.optimization_approaches:
            lines += ["", "optimisation approaches:"]
            lines += [
                f"  {self.optimization_approaches[k].render()}"
                for k in sorted(self.optimization_approaches)
            ]
        return "\n".join(lines)


def _validated(record: MigrationRecord) -> bool:
    """Did the differ ultimately say PASS?

    The only definition of success this file recognises, and the same one every
    other stage uses.
    """
    return (
        record.validation is not None
        and record.validation.report.status is ValidationStatus.PASS
    )


def harvest(records: list[MigrationRecord]) -> MigrationHistory:
    """Aggregate completed migrations into keyed evidence.

    Pure and deterministic: same records in, same history out, so the result can
    be recomputed from the artifacts at any time and never has to be trusted as
    durable state of its own. There is no separate store to fall out of step
    with what actually happened.
    """
    repairs: dict[str, StrategyEvidence] = {}
    optimizations: dict[str, StrategyEvidence] = {}
    seen: set[str] = set()
    validated = 0

    def evidence(store: dict[str, StrategyEvidence], key: str) -> StrategyEvidence:
        return store.setdefault(key, StrategyEvidence(key=key))

    for record in records:
        if record.migration_id in seen:
            # Artifacts can be re-read; the same migration must not count twice.
            continue
        seen.add(record.migration_id)
        if _validated(record):
            validated += 1

        if record.repair is not None:
            for repair_attempt in record.repair.attempts:
                if not repair_attempt.admitted or repair_attempt.strategy is None:
                    # A refusal by the ledger says nothing about whether the
                    # strategy works — it was never tried. Counting it as a
                    # failure would punish an approach for being proposed twice.
                    continue
                strategy = repair_attempt.strategy
                key = f"{strategy.category.value}/{strategy.approach}"
                item = evidence(repairs, key)
                item.attempts += 1
                item.successes += int(repair_attempt.succeeded)
                item.migrations.append(record.migration_id)

        if record.optimization is not None:
            for opt_attempt in record.optimization.attempts:
                if not opt_attempt.admitted:
                    continue
                item = evidence(optimizations, opt_attempt.strategy.approach)
                item.attempts += 1
                item.migrations.append(record.migration_id)
                if opt_attempt.accepted:
                    item.successes += 1
                    if opt_attempt.comparison is not None:
                        item.speedups.append(opt_attempt.comparison.speedup)
                        item.predicted_speedups.append(
                            opt_attempt.strategy.expected_speedup
                        )

    return MigrationHistory(
        migrations_observed=len(seen),
        validated=validated,
        repair_strategies=repairs,
        optimization_approaches=optimizations,
    )


def outcomes_by_category(records: list[MigrationRecord]) -> dict[str, tuple[int, int]]:
    """Validation outcome per declared risk category, as `(passed, total)`.

    Which *kinds* of semantic difference actually cause trouble, measured rather
    than assumed. A category that has never failed is either genuinely benign or
    never really exercised, and the counts let a reader tell which.
    """
    tally: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for record in records:
        if record.plan is None or record.validation is None:
            continue
        passed = _validated(record)
        for difference in record.plan.all_semantic_differences:
            entry = tally[difference.category.value]
            entry[0] += int(passed)
            entry[1] += 1
    return {key: (v[0], v[1]) for key, v in sorted(tally.items())}
