"""Optimizer agent: proposes a change, and is then measured.

This agent has the least authority of any in the system. It proposes; the
benchmark and the differ dispose. `evaluate_optimization` accepts nothing on the
strength of `expected_speedup` — that field exists only so an over-optimistic
claim can be compared against the measurement afterwards and shown to be one.

Its inputs are deliberately all measurements: the baseline timings with their
noise ratio, Spark's own job/stage/task counts, the measured input file sizes,
and a structural analysis of the code. If it cannot ground a proposal in one of
those, it is asked to propose nothing — an honest "no opportunity" is a better
outcome than a change that costs a validation run and a benchmark to disprove.
"""

from __future__ import annotations

from collections.abc import Callable

from autogen_core.models import ChatCompletionClient

from etl_migrator.agents.base import StructuredAgent
from etl_migrator.domain.code import GeneratedCode
from etl_migrator.domain.history import MigrationHistory
from etl_migrator.domain.optimization import (
    BenchmarkResult,
    OptimizationAttempt,
    OptimizationProposal,
)
from etl_migrator.domain.plan import MigrationPlan
from etl_migrator.tools.code_gate import GateOptions, analyze_generated_code
from etl_migrator.tools.plan_analyzer import PlanAnalysis

SYSTEM_MESSAGE = """\
You are a Spark Optimizer Agent. A migration is already correct: its output
matches the legacy reference exactly. Your job is to make it faster without
changing that output.

Understand your position before proposing anything. Whatever you claim, the
system will re-run the full validation and re-benchmark with warm-up discarded.
An optimisation is kept only if validation still passes AND the measured speedup
clears a threshold AND the gain is robust across runs. Your `expected_speedup`
is recorded and compared against reality; it persuades nobody.

Rules:

1. Call `get_baseline` first. It gives the measured timings, their noise ratio,
   and Spark's own job/stage/task counts. If the noise ratio is already above
   the acceptance ceiling, say so in your rationale — no optimisation can be
   demonstrated on that measurement, and proposing one wastes a benchmark.
2. Call `analyze_current_plan`. It reports structural opportunities with line
   numbers, and it reports honestly when there are none.
3. Ground every proposal in something those two tools reported. "Broadcast the
   small side" is only a proposal if the profiler measured a small side and no
   broadcast hint is already present. General Spark advice is not evidence.
4. Prefer configuration changes to code changes. Shuffle partitions, AQE and
   coalesce thresholds are reversible, carry no correctness risk of their own,
   and are frequently the larger win. Set `execution_strategy` and leave `code`
   null when configuration alone suffices.
5. If you do change code, return the complete module and keep every correctness
   mitigation intact. An optimisation that breaks a null-handling fix will be
   caught by validation and rejected, having wasted an attempt.
6. One change per proposal. Two changes measured together cannot be attributed.
7. `strategy.approach` is a slug naming the technique:
   `reduce_shuffle_partitions`, `enable_adaptive_coalescing`,
   `broadcast_small_side`, `cache_reused_dataframe`,
   `remove_single_partition_coalesce`, `replace_udf_with_builtin`.
   A repeated slug is refused before it is measured.

Before proposing an approach, call `approach_track_record` for it. That is how
the same approach has *measured* on other migrations — how often it was accepted
and what speedup it actually delivered. An approach accepted zero times out of
four has already cost four validation runs and eight benchmarks to disprove, and
proposing it again spends them a fifth time. Too little evidence is an honest
answer; act on the baseline and the plan analysis instead.

If neither tool shows an opportunity — for example the pipeline has no wide
transformations, or the small side is already broadcast — propose the
`no_change` strategy with `expected_speedup` 1.0 and say why. Declining is a
valid, useful answer and costs nothing. Inventing work is not.
"""


class OptimizerAgent(StructuredAgent[OptimizationProposal]):
    key = "optimizer"
    description = "Proposes a measurable Spark optimisation grounded in real metrics."

    def __init__(
        self,
        model_client: ChatCompletionClient,
        *,
        plan: MigrationPlan,
        code: GeneratedCode,
        baseline: BenchmarkResult,
        analysis: PlanAnalysis,
        history: list[OptimizationAttempt] | None = None,
        past: MigrationHistory | None = None,
        gate_options: GateOptions | None = None,
        max_tool_iterations: int = 8,
    ) -> None:
        self.plan = plan
        self.code = code
        self.baseline = baseline
        self.analysis = analysis
        self.history = history or []
        self.past = past or MigrationHistory()
        self.gate_options = gate_options or GateOptions()
        self.gate_calls = 0
        super().__init__(
            model_client,
            OptimizationProposal,
            system_message=SYSTEM_MESSAGE,
            tools=[
                self._make_baseline_tool(),
                self._make_analysis_tool(),
                self._make_code_tool(),
                self._make_history_tool(),
                self._make_track_record_tool(),
                self._make_gate_tool(),
            ],
            max_tool_iterations=max_tool_iterations,
        )

    def _make_baseline_tool(self) -> Callable[[], str]:
        baseline, plan = self.baseline, self.plan

        def get_baseline() -> str:
            """Return the measured baseline: timings, their noise ratio, and Spark's
            reported job, stage and task counts, plus the configuration in effect."""
            strategy = plan.execution_strategy
            return (
                f"{baseline.render()}\n"
                f"discarded warm-up runs: {baseline.discarded_warmups}\n"
                f"configured shuffle_partitions={strategy.shuffle_partitions} "
                f"aqe={strategy.adaptive_query_execution} "
                f"broadcast={strategy.broadcast_datasets}"
            )

        return get_baseline

    def _make_analysis_tool(self) -> Callable[[], str]:
        analysis = self.analysis

        def analyze_current_plan() -> str:
            """Return the structural analysis of the current pipeline: wide
            transformations, broadcast hints, cache calls, and every detected
            optimisation opportunity with its line number and justification."""
            return analysis.render()

        return analyze_current_plan

    def _make_code_tool(self) -> Callable[[], str]:
        content = self.code.content

        def get_current_code() -> str:
            """Return the full source of the pipeline being optimised."""
            return content

        return get_current_code

    def _make_history_tool(self) -> Callable[[], str]:
        history = self.history

        def previous_attempts() -> str:
            """Return every optimisation already tried and how it measured.

            Approaches listed here are spent; proposing one again is refused
            before it is benchmarked.
            """
            if not history:
                return "No previous optimisation attempts."
            return "\n".join(f"  {a.render()}" for a in history)

        return previous_attempts

    def _make_track_record_tool(self) -> Callable[[str], str]:
        past = self.past

        def approach_track_record(approach: str) -> str:
            """Return how an optimisation approach has measured on *previous*
            migrations: how often it was accepted, the median speedup it
            actually delivered, and how far its predictions have run ahead of
            the measurements.

            `previous_attempts` covers this migration only; this is the corpus.
            An approach accepted zero times out of several is worth knowing
            before you spend a validation run and two benchmarks rediscovering
            it.

            Args:
                approach: the strategy slug you are considering.
            """
            evidence = past.optimization_evidence(approach)
            if evidence.attempts == 0:
                return (
                    f"{approach}: never measured on a recorded migration. Nothing "
                    "is known about it either way."
                )
            return evidence.render()

        return approach_track_record

    def _make_gate_tool(self) -> Callable[[str], str]:
        options = self.gate_options
        agent = self

        def check_spark_code(code: str) -> str:
            """Run the static gate on an optimised module. Only needed when the
            proposal changes code.

            Args:
                code: the complete optimised module source.
            """
            agent.gate_calls += 1
            report = analyze_generated_code(code, options)
            return (
                f"gate: {'PASS' if report.passed else 'FAIL'} "
                f"(submission #{agent.gate_calls})\n{report.render()}"
            )

        return check_spark_code


def optimization_task(
    baseline: BenchmarkResult,
    history: list[OptimizationAttempt],
    attempt: int,
    max_attempts: int,
) -> str:
    spent = (
        "\n".join(f"  {a.render()}" for a in history)
        if history
        else "  none"
    )
    return (
        f"Optimisation attempt {attempt} of {max_attempts}.\n"
        f"Baseline: {baseline.render()}\n\n"
        f"Already tried:\n{spent}\n\n"
        "Read the baseline and the plan analysis, then propose one grounded change — "
        "or the no_change strategy if neither shows an opportunity."
    )
