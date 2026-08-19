"""Prometheus metrics, derived from the migration record.

Nothing is instrumented at the call site. Every number is read from a finished
`MigrationRecord`, the same object the PR body renders from, so a dashboard and
a pull request cannot disagree. `etl_optimizations_kept_total` moves when
`OptimizationOutcome.applied` is true, and only `evaluate_optimization` sets
that.

Three things this layer gets wrong easily, handled explicitly:

Cardinality. A `migration_id` label would add a series per migration. Every
label here draws from a bounded set and a test asserts it; the id stays in the
log line and the artifact.

Bucket ranges. The default buckets top out at ten seconds, so a migration that
takes minutes would land entirely in `+Inf`. The buckets below come from what
this system actually measures.

Double counting. Temporal retries activities, so the same record can arrive
twice. Counters are deduplicated per migration within a worker process, which
makes them exactly-once for a retry in the same process and at-least-once
across a worker restart.
"""

from __future__ import annotations

from collections import OrderedDict

from prometheus_client import CollectorRegistry, Counter, Histogram

from etl_migrator.domain.artifacts import MigrationRecord
from etl_migrator.domain.delivery import DeliveryDisposition
from etl_migrator.domain.enums import RiskLevel, ValidationStatus
from etl_migrator.domain.optimization import OptimizationAttempt
from etl_migrator.observability.logging import get_logger

log = get_logger(__name__)

#: A migration is minutes, not milliseconds. The default buckets would put every
#: observation in +Inf.
MIGRATION_BUCKETS = (10.0, 30.0, 60.0, 120.0, 300.0, 600.0, 1800.0, 3600.0, float("inf"))

#: A stage ranges from a sub-second gate to a multi-minute benchmark.
STAGE_BUCKETS = (0.1, 0.5, 1.0, 5.0, 15.0, 30.0, 60.0, 180.0, 600.0, float("inf"))

#: One Spark execution on a small dataset is ~10s; a large one is minutes.
EXECUTION_BUCKETS = (1.0, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0, 320.0, float("inf"))

#: Speedups worth distinguishing. 1.10 is the acceptance threshold, so the
#: bucket boundary sits there deliberately: `rate` below it is the rejected
#: population, above it the kept one.
SPEEDUP_BUCKETS = (1.0, 1.05, 1.10, 1.25, 1.5, 2.0, 3.0, 5.0, float("inf"))

#: The measurement's own error bar. 0.25 is the ceiling above which a comparison
#: is refused, so the distribution shows how close the system runs to it.
NOISE_BUCKETS = (0.01, 0.02, 0.05, 0.10, 0.25, 0.50, 1.0, float("inf"))


class OptimizationVerdict:
    """Bounded classes for why an optimisation was or was not kept.

    Derived from the structured fields of an `OptimizationAttempt`, never parsed
    out of its rendered prose. The prose is for humans and changes freely; a
    label that tracked it would silently fragment the time series the first time
    somebody improved a sentence.
    """

    ACCEPTED = "accepted"
    DECLINED = "declined"
    REFUSED_REPEAT = "refused_repeat"
    REJECTED_GATE = "rejected_gate"
    REJECTED_CORRECTNESS = "rejected_correctness"
    REJECTED_INCONCLUSIVE = "rejected_inconclusive"
    REJECTED_NOT_ROBUST = "rejected_not_robust"
    REJECTED_BELOW_THRESHOLD = "rejected_below_threshold"

    ALL = (
        ACCEPTED,
        DECLINED,
        REFUSED_REPEAT,
        REJECTED_GATE,
        REJECTED_CORRECTNESS,
        REJECTED_INCONCLUSIVE,
        REJECTED_NOT_ROBUST,
        REJECTED_BELOW_THRESHOLD,
    )


def classify_optimization(attempt: OptimizationAttempt) -> str:
    """Map an attempt onto one bounded verdict class.

    The order mirrors `evaluate_optimization`, so the label a reviewer sees in
    Grafana is the same reason the workflow acted on.
    """
    if attempt.accepted:
        return OptimizationVerdict.ACCEPTED
    if not attempt.admitted:
        reason = attempt.rejection_reason or ""
        return (
            OptimizationVerdict.DECLINED
            if "no grounded opportunity" in reason
            else OptimizationVerdict.REFUSED_REPEAT
        )
    if attempt.comparison is None:
        # Never benchmarked: either the gate refused the code or validation did.
        return (
            OptimizationVerdict.REJECTED_CORRECTNESS
            if attempt.validation_status not in (None, ValidationStatus.PASS.value)
            else OptimizationVerdict.REJECTED_GATE
        )
    if attempt.validation_status != ValidationStatus.PASS.value:
        return OptimizationVerdict.REJECTED_CORRECTNESS
    comparison = attempt.comparison
    if comparison.inconclusive:
        return OptimizationVerdict.REJECTED_INCONCLUSIVE
    if not comparison.robust:
        return OptimizationVerdict.REJECTED_NOT_ROBUST
    return OptimizationVerdict.REJECTED_BELOW_THRESHOLD


class MigrationMetrics:
    """Every series this system exports.

    Instances hold their own `CollectorRegistry` so tests can build one, observe
    a record and assert on the result without the process-global registry
    leaking state between them.
    """

    def __init__(self, registry: CollectorRegistry | None = None, *, dedup_size: int = 4096):
        self.registry = registry or CollectorRegistry()
        #: Migration ids already counted, newest last. Bounded so a long-lived
        #: worker cannot grow this without limit.
        self._seen: OrderedDict[str, None] = OrderedDict()
        self._dedup_size = dedup_size

        self.migrations = Counter(
            "etl_migrations_total",
            "Migrations observed, by final outcome and the risk the planner assessed.",
            ["outcome", "risk"],
            registry=self.registry,
        )
        self.migration_duration = Histogram(
            "etl_migration_duration_seconds",
            "Wall-clock duration of a whole migration.",
            ["outcome"],
            buckets=MIGRATION_BUCKETS,
            registry=self.registry,
        )
        self.stage_duration = Histogram(
            "etl_stage_duration_seconds",
            "Wall-clock duration of one migration stage.",
            ["stage"],
            buckets=STAGE_BUCKETS,
            registry=self.registry,
        )
        self.validation_checks = Counter(
            "etl_validation_checks_total",
            "Differ checks, by name and result. A skipped check is neither a "
            "pass nor a failure and is counted as itself.",
            ["check", "result"],
            registry=self.registry,
        )
        self.validations = Counter(
            "etl_validations_total",
            "Validation verdicts, computed by the differ from executed output.",
            ["status"],
            registry=self.registry,
        )
        self.repair_attempts = Counter(
            "etl_repair_attempts_total",
            "Repair attempts, including the ones the ledger refused before "
            "spending an execution.",
            ["outcome"],
            registry=self.registry,
        )
        self.repairs = Counter(
            "etl_repairs_total",
            "Repair loops that ran, by whether they recovered the migration.",
            ["outcome"],
            registry=self.registry,
        )
        self.optimization_attempts = Counter(
            "etl_optimization_attempts_total",
            "Optimisation attempts by verdict class. Every rejection reason is "
            "distinguishable, because 'nothing was kept' has several very "
            "different causes.",
            ["verdict"],
            registry=self.registry,
        )
        self.optimization_speedup = Histogram(
            "etl_optimization_speedup_ratio",
            "Measured speedup of accepted optimisations. Only accepted ones are "
            "observed: a rejected proposal's ratio is not a speedup, it is a "
            "number the system declined to believe.",
            buckets=SPEEDUP_BUCKETS,
            registry=self.registry,
        )
        self.benchmark_noise = Histogram(
            "etl_benchmark_noise_ratio",
            "Relative dispersion of a benchmark. Above the ceiling the "
            "comparison is refused, so this shows how close measurements run to "
            "being unusable.",
            ["role"],
            buckets=NOISE_BUCKETS,
            registry=self.registry,
        )
        self.execution_duration = Histogram(
            "etl_pipeline_execution_seconds",
            "One sandboxed pipeline execution.",
            ["engine"],
            buckets=EXECUTION_BUCKETS,
            registry=self.registry,
        )
        self.agent_duration = Histogram(
            "etl_agent_duration_seconds",
            "One agent invocation, end to end including its tool calls.",
            ["agent"],
            buckets=STAGE_BUCKETS,
            registry=self.registry,
        )
        self.agent_tool_calls = Counter(
            "etl_agent_tool_calls_total",
            "Tool invocations by agent. 'Did it actually look?' as a time series.",
            ["agent"],
            registry=self.registry,
        )
        self.gate_submissions = Counter(
            "etl_gate_submissions_total",
            "Static-gate submissions. More than one means the agent had to "
            "revise, which is the loop working rather than a failure.",
            registry=self.registry,
        )
        self.deliveries = Counter(
            "etl_deliveries_total",
            "Delivery outcomes: a refused PR is a result, not an absence.",
            ["disposition"],
            registry=self.registry,
        )
        self.claim_violations = Counter(
            "etl_pr_claim_violations_total",
            "Numeric claims in agent-written PR prose that the record did not "
            "support. Non-zero means the audit earned its keep.",
            registry=self.registry,
        )

    # -- observation -------------------------------------------------------
    def observe(self, record: MigrationRecord) -> bool:
        """Record every metric derivable from a finished migration.

        Returns False when this migration was already counted in this process,
        so a retried activity does not inflate the counters. See the module
        docstring for the exact guarantee this does and does not give.
        """
        if record.migration_id in self._seen:
            log.debug("metrics.duplicate_ignored", migration_id=record.migration_id)
            return False
        self._seen[record.migration_id] = None
        while len(self._seen) > self._dedup_size:
            self._seen.popitem(last=False)

        outcome = "failed" if record.failed else "succeeded"
        risk = record.risk.value if record.plan is not None else RiskLevel.LOW.value
        self.migrations.labels(outcome=outcome, risk=risk).inc()
        self.migration_duration.labels(outcome=outcome).observe(
            record.total_duration_seconds
        )

        for entry in record.stages:
            duration = entry.duration_seconds
            if duration is not None:
                self.stage_duration.labels(stage=entry.stage.value).observe(duration)

        if record.codegen is not None:
            self.gate_submissions.inc(record.codegen.gate_iterations)

        self._observe_validation(record)
        self._observe_repair(record)
        self._observe_optimization(record)
        self._observe_delivery(record)
        return True

    def _observe_validation(self, record: MigrationRecord) -> None:
        outcome = record.validation
        if outcome is None:
            return
        self.validations.labels(status=outcome.report.status.value).inc()
        for check in outcome.report.checks:
            result = "skip" if check.skipped else ("pass" if check.passed else "fail")
            self.validation_checks.labels(check=check.name, result=result).inc()
        for execution in (outcome.legacy_execution, outcome.spark_execution):
            if execution is not None and execution.succeeded:
                self.execution_duration.labels(engine=execution.engine).observe(
                    execution.duration_seconds
                )

    def _observe_repair(self, record: MigrationRecord) -> None:
        outcome = record.repair
        if outcome is None:
            return
        self.repairs.labels(outcome="succeeded" if outcome.succeeded else "exhausted").inc()
        for attempt in outcome.attempts:
            if not attempt.admitted:
                # The ledger refused it before spending an execution. Counting
                # these separately is the point: they are the loop's savings,
                # not its failures.
                result = "refused"
            else:
                result = "fixed" if attempt.succeeded else "did_not_fix"
            self.repair_attempts.labels(outcome=result).inc()

    def _observe_optimization(self, record: MigrationRecord) -> None:
        outcome = record.optimization
        if outcome is None:
            return
        for attempt in outcome.attempts:
            self.optimization_attempts.labels(verdict=classify_optimization(attempt)).inc()
            if attempt.comparison is not None:
                for role, side in (
                    ("baseline", attempt.comparison.baseline),
                    ("candidate", attempt.comparison.candidate),
                ):
                    if not side.failed and side.samples >= 2:
                        self.benchmark_noise.labels(role=role).observe(side.noise_ratio)
        if outcome.applied:
            # Only accepted speedups. A rejected ratio is a number the system
            # declined to believe, and averaging it in would report improvements
            # that were never kept.
            self.optimization_speedup.observe(outcome.speedup)

    def _observe_delivery(self, record: MigrationRecord) -> None:
        outcome = record.delivery
        if outcome is None:
            return
        disposition = (
            outcome.decision.disposition.value
            if outcome.decision is not None
            else DeliveryDisposition.REFUSED.value
        )
        self.deliveries.labels(disposition=disposition).inc()
        if outcome.audit is not None and outcome.audit.violations:
            self.claim_violations.inc(len(outcome.audit.violations))

    def observe_telemetry(self, telemetry: object) -> None:
        """Record one agent invocation.

        Separate from `observe` because agent telemetry is produced per activity
        and does not survive on the record; the caller already has it in hand.
        """
        agent = getattr(telemetry, "agent", None)
        if not isinstance(agent, str):
            return
        duration = getattr(telemetry, "duration_seconds", 0.0)
        self.agent_duration.labels(agent=agent).observe(float(duration))
        tools = getattr(telemetry, "tools_used", [])
        if tools:
            self.agent_tool_calls.labels(agent=agent).inc(len(tools))


#: Process-wide instance. A worker exports one registry; the CLI builds its own.
_metrics: MigrationMetrics | None = None


def get_metrics() -> MigrationMetrics:
    global _metrics
    if _metrics is None:
        _metrics = MigrationMetrics()
    return _metrics


def reset_metrics() -> None:
    """Drop the process-wide instance. For tests that need a clean registry."""
    global _metrics
    _metrics = None
