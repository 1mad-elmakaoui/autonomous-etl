"""Metrics, checked for the two ways an observability layer goes wrong.

A metric is a claim about the system, so it inherits the same rule everything
else here obeys: it must be derived from a measurement, not asserted at a call
site. Every series is read off a finished `MigrationRecord` — the same object
the PR body renders from — so a dashboard and a pull request cannot tell a
reviewer different stories. The tests below drive real records through
`observe()` and read the exposition output back.

The other failure is quieter and kills the Prometheus server rather than
misleading a human: **unbounded cardinality**. A `migration_id` label adds a
time series per migration for ever. `TestCardinality` walks every label value
this system can emit and asserts each comes from a bounded set.

No JVM, no network, no cluster.
"""

from __future__ import annotations

import urllib.request

import pytest
from prometheus_client import generate_latest
from tests.helpers_records import at, benchmark, migration_record, strategy

from etl_migrator.domain.artifacts import CodeGenResult, MigrationRecord, StageRecord
from etl_migrator.domain.code import GeneratedCode, StaticAnalysisReport
from etl_migrator.domain.delivery import (
    ClaimAudit,
    ClaimViolation,
    DeliveryDecision,
    DeliveryDisposition,
    DeliveryOutcome,
)
from etl_migrator.domain.enums import (
    MigrationStage,
    RiskCategory,
    RiskLevel,
    ValidationStatus,
)
from etl_migrator.domain.optimization import (
    BenchmarkComparison,
    OptimizationAttempt,
    OptimizationOutcome,
)
from etl_migrator.domain.repair import RepairAttempt, RepairOutcome, RepairStrategy
from etl_migrator.domain.validation import (
    CheckResult,
    ExecutionResult,
)
from etl_migrator.observability import (
    MetricsServer,
    MigrationMetrics,
    OptimizationVerdict,
    classify_optimization,
)
from etl_migrator.observability.metrics import SPEEDUP_BUCKETS


@pytest.fixture
def metrics() -> MigrationMetrics:
    """A private registry per test, so counters never leak between them."""
    return MigrationMetrics()


def series(metrics: MigrationMetrics, name: str) -> dict[frozenset[tuple[str, str]], float]:
    """Every sample of a metric, keyed by its label set."""
    found: dict[frozenset[tuple[str, str]], float] = {}
    for metric in metrics.registry.collect():
        for sample in metric.samples:
            if sample.name == name:
                found[frozenset(sample.labels.items())] = sample.value
    return found


def value(metrics: MigrationMetrics, name: str, **labels: str) -> float:
    return series(metrics, name).get(frozenset(labels.items()), 0.0)


class TestDerivedFromTheRecord:
    def test_a_successful_migration_is_counted_with_its_risk(
        self, metrics: MigrationMetrics
    ) -> None:
        metrics.observe(migration_record())
        assert value(metrics, "etl_migrations_total", outcome="succeeded", risk="high") == 1

    def test_a_failed_migration_is_counted_as_failed(self, metrics: MigrationMetrics) -> None:
        """A failure is a result, and the dashboard has to be able to see it."""
        metrics.observe(migration_record(failed=True))
        assert value(metrics, "etl_migrations_total", outcome="failed", risk="high") == 1
        assert value(metrics, "etl_migrations_total", outcome="succeeded", risk="high") == 0

    def test_stage_durations_come_from_the_recorded_timings(
        self, metrics: MigrationMetrics
    ) -> None:
        record = migration_record()
        record.stages = [
            StageRecord(
                stage=MigrationStage.VALIDATION,
                started_at=at(0),
                ended_at=at(42),
                succeeded=True,
            )
        ]
        metrics.observe(record)
        count = value(
            metrics, "etl_stage_duration_seconds_count", stage="validation"
        )
        total = value(metrics, "etl_stage_duration_seconds_sum", stage="validation")
        assert count == 1
        assert total == pytest.approx(42.0)

    def test_an_unfinished_stage_is_not_observed(self, metrics: MigrationMetrics) -> None:
        """A stage still running has no duration. Recording it as zero would
        drag every latency percentile down and make a stall look fast."""
        record = migration_record()
        record.stages = [
            StageRecord(stage=MigrationStage.VALIDATION, started_at=at(0), ended_at=None)
        ]
        metrics.observe(record)
        assert value(metrics, "etl_stage_duration_seconds_count", stage="validation") == 0

    def test_every_check_is_counted_by_name_and_result(
        self, metrics: MigrationMetrics
    ) -> None:
        record = migration_record(
            checks=[
                CheckResult(name="schema", passed=True),
                CheckResult(name="row_count", passed=False, detail="4 != 5"),
                CheckResult(name="null_counts", passed=False, skipped=True),
            ]
        )
        metrics.observe(record)
        assert value(metrics, "etl_validation_checks_total", check="schema", result="pass") == 1
        assert (
            value(metrics, "etl_validation_checks_total", check="row_count", result="fail") == 1
        )
        assert (
            value(metrics, "etl_validation_checks_total", check="null_counts", result="skip") == 1
        )

    def test_a_skipped_check_is_neither_a_pass_nor_a_failure(
        self, metrics: MigrationMetrics
    ) -> None:
        """The differ's own rule, carried into the metric: "we did not measure
        it" must not be aggregated as "it was fine"."""
        record = migration_record(
            checks=[CheckResult(name="null_counts", passed=False, skipped=True)]
        )
        metrics.observe(record)
        for result in ("pass", "fail"):
            counted = value(
                metrics, "etl_validation_checks_total", check="null_counts", result=result
            )
            assert counted == 0, f"a skipped check was counted as {result}"

    def test_pipeline_executions_are_timed_per_engine(
        self, metrics: MigrationMetrics
    ) -> None:
        record = migration_record()
        assert record.validation is not None
        record.validation.legacy_execution = ExecutionResult(
            engine="pandas", succeeded=True, duration_seconds=0.4
        )
        record.validation.spark_execution = ExecutionResult(
            engine="spark", succeeded=True, duration_seconds=11.2
        )
        metrics.observe(record)
        assert value(metrics, "etl_pipeline_execution_seconds_count", engine="pandas") == 1
        assert value(
            metrics, "etl_pipeline_execution_seconds_sum", engine="spark"
        ) == pytest.approx(11.2)

    def test_a_failed_execution_is_not_timed(self, metrics: MigrationMetrics) -> None:
        """Its duration is time-to-crash, which is not the distribution the
        histogram is describing."""
        record = migration_record()
        assert record.validation is not None
        record.validation.spark_execution = ExecutionResult(
            engine="spark", succeeded=False, duration_seconds=2.0, error="OOM"
        )
        metrics.observe(record)
        assert value(metrics, "etl_pipeline_execution_seconds_count", engine="spark") == 0

    def test_gate_submissions_count_the_agents_revisions(
        self, metrics: MigrationMetrics
    ) -> None:
        """More than one is the loop working, not a failure — and worth watching
        because a rising trend means generation quality is drifting."""
        record = migration_record()
        record.codegen = CodeGenResult(
            code=GeneratedCode(filename="p.py", content="x = 1\n"),
            static_analysis=StaticAnalysisReport(passed=True),
            gate_iterations=3,
        )
        metrics.observe(record)
        assert value(metrics, "etl_gate_submissions_total") == 3

    def test_a_refused_repair_attempt_is_distinguished_from_a_failed_one(
        self, metrics: MigrationMetrics
    ) -> None:
        """The ledger's refusals are the loop's savings. Counting them as
        failures would make the anti-oscillation guard look like a defect."""
        record = migration_record(
            repair=RepairOutcome(
                succeeded=True,
                attempts=[
                    RepairAttempt(attempt=1, validation_status=ValidationStatus.FAIL),
                    RepairAttempt(attempt=2, admitted=False, rejection_reason="repeat"),
                    RepairAttempt(attempt=3, validation_status=ValidationStatus.PASS),
                ],
            )
        )
        metrics.observe(record)
        assert value(metrics, "etl_repair_attempts_total", outcome="refused") == 1
        assert value(metrics, "etl_repair_attempts_total", outcome="did_not_fix") == 1
        assert value(metrics, "etl_repair_attempts_total", outcome="fixed") == 1
        assert value(metrics, "etl_repairs_total", outcome="succeeded") == 1

    def test_delivery_records_a_refusal_as_an_outcome(
        self, metrics: MigrationMetrics
    ) -> None:
        record = migration_record(
            delivery=DeliveryOutcome(
                decision=DeliveryDecision(
                    disposition=DeliveryDisposition.REFUSED, reason="never validated"
                )
            )
        )
        metrics.observe(record)
        assert value(metrics, "etl_deliveries_total", disposition="refused") == 1

    def test_claim_violations_are_counted(self, metrics: MigrationMetrics) -> None:
        """Non-zero means the PR audit earned its keep — a number in agent prose
        that the record did not support, caught before it was published."""
        record = migration_record(
            delivery=DeliveryOutcome(
                decision=DeliveryDecision(
                    disposition=DeliveryDisposition.READY, reason="validated"
                ),
                audit=ClaimAudit(
                    checked=3,
                    violations=[
                        ClaimViolation(
                            kind="speedup", claimed="3", supported=["1.25"], excerpt="3x faster"
                        )
                    ],
                ),
            )
        )
        metrics.observe(record)
        assert value(metrics, "etl_pr_claim_violations_total") == 1

    def test_agent_telemetry_is_observed_separately(
        self, metrics: MigrationMetrics
    ) -> None:
        from etl_migrator.domain.messages import AgentTelemetry

        metrics.observe_telemetry(
            AgentTelemetry(
                agent="spark_engineer",
                tools_used=["check_spark_code", "check_spark_code"],
                duration_seconds=2.5,
            )
        )
        assert value(metrics, "etl_agent_tool_calls_total", agent="spark_engineer") == 2
        assert value(
            metrics, "etl_agent_duration_seconds_sum", agent="spark_engineer"
        ) == pytest.approx(2.5)


class TestOnlyMeasuredSpeedupsAreExported:
    """The benchmarking rule, carried into the dashboard.

    A rejected optimisation's ratio is not a speedup — it is a number the system
    declined to believe. Averaging it into the histogram would report
    improvements that were never kept, which is precisely the benchmark theatre
    `evaluate_optimization` exists to prevent.
    """

    def test_an_accepted_optimisation_is_observed(self, metrics: MigrationMetrics) -> None:
        record = migration_record(
            optimization=OptimizationOutcome(
                applied=True,
                baseline=benchmark([10.0] * 4),
                final=benchmark([8.0] * 4),
                accepted_strategy=strategy("reduce_shuffle_partitions"),
                attempts=[
                    OptimizationAttempt(
                        attempt=1,
                        strategy=strategy("reduce_shuffle_partitions"),
                        accepted=True,
                        validation_status="PASS",
                        comparison=BenchmarkComparison(
                            baseline=benchmark([10.0] * 4), candidate=benchmark([8.0] * 4)
                        ),
                    )
                ],
            )
        )
        metrics.observe(record)
        assert value(metrics, "etl_optimization_speedup_ratio_count") == 1
        assert value(metrics, "etl_optimization_speedup_ratio_sum") == pytest.approx(1.25)
        assert value(metrics, "etl_optimization_attempts_total", verdict="accepted") == 1

    def test_a_rejected_optimisation_contributes_no_speedup(
        self, metrics: MigrationMetrics
    ) -> None:
        record = migration_record(
            optimization=OptimizationOutcome(
                applied=False,
                baseline=benchmark([10.0] * 4),
                final=benchmark([10.0] * 4),
                attempts=[
                    OptimizationAttempt(
                        attempt=1,
                        strategy=strategy("broadcast_small_side"),
                        accepted=False,
                        validation_status="PASS",
                        comparison=BenchmarkComparison(
                            baseline=benchmark([10.0] * 4),
                            candidate=benchmark([9.8] * 4),
                        ),
                    )
                ],
            )
        )
        metrics.observe(record)
        assert value(metrics, "etl_optimization_speedup_ratio_count") == 0

    def test_the_threshold_is_a_bucket_boundary(self) -> None:
        """So `rate` below it is the rejected population and above it the kept
        one, without anyone having to interpolate."""
        from etl_migrator.domain.optimization import DEFAULT_MIN_SPEEDUP

        assert DEFAULT_MIN_SPEEDUP in SPEEDUP_BUCKETS

    def test_benchmark_noise_is_exported_for_both_sides(
        self, metrics: MigrationMetrics
    ) -> None:
        """The measurement's own error bar, as a series. This is what tells you
        the fleet is drifting toward unmeasurable before verdicts start coming
        back inconclusive."""
        record = migration_record(
            optimization=OptimizationOutcome(
                attempts=[
                    OptimizationAttempt(
                        attempt=1,
                        strategy=strategy("reduce_shuffle_partitions"),
                        validation_status="PASS",
                        comparison=BenchmarkComparison(
                            baseline=benchmark([10.0, 10.1, 9.9, 10.0]),
                            candidate=benchmark([8.0, 8.2, 7.8, 8.0]),
                        ),
                    )
                ]
            )
        )
        metrics.observe(record)
        assert value(metrics, "etl_benchmark_noise_ratio_count", role="baseline") == 1
        assert value(metrics, "etl_benchmark_noise_ratio_count", role="candidate") == 1


class TestVerdictClassification:
    """Every rejection reason must stay distinguishable.

    "Nothing was kept" has several very different causes — the change broke the
    output, the measurement was unreadable, the gain was real but small — and a
    dashboard that collapsed them would hide the one that matters.
    """

    def attempt(self, **kwargs: object) -> OptimizationAttempt:
        return OptimizationAttempt(
            attempt=1, strategy=strategy("reduce_shuffle_partitions"), **kwargs  # type: ignore[arg-type]
        )

    def test_accepted(self) -> None:
        assert classify_optimization(self.attempt(accepted=True)) == OptimizationVerdict.ACCEPTED

    def test_the_agent_declining_is_not_a_rejection(self) -> None:
        """Declining is a valid, useful answer and should not read as a failure."""
        got = classify_optimization(
            self.attempt(
                admitted=False,
                rejection_reason="the optimizer found no grounded opportunity",
            )
        )
        assert got == OptimizationVerdict.DECLINED

    def test_a_repeated_approach_is_its_own_class(self) -> None:
        got = classify_optimization(
            self.attempt(admitted=False, rejection_reason="approach 'x' was already measured")
        )
        assert got == OptimizationVerdict.REFUSED_REPEAT

    def test_broken_correctness(self) -> None:
        got = classify_optimization(
            self.attempt(
                validation_status="FAIL",
                comparison=BenchmarkComparison(
                    baseline=benchmark([10.0] * 4), candidate=benchmark([1.0] * 4)
                ),
            )
        )
        assert got == OptimizationVerdict.REJECTED_CORRECTNESS

    def test_an_unreadable_measurement(self) -> None:
        got = classify_optimization(
            self.attempt(
                validation_status="PASS",
                comparison=BenchmarkComparison(
                    baseline=benchmark([10.0] * 4),
                    candidate=benchmark([2.0, 5.0, 5.0, 12.0]),
                ),
            )
        )
        assert got == OptimizationVerdict.REJECTED_INCONCLUSIVE

    def test_a_gain_that_rests_on_one_lucky_run(self) -> None:
        got = classify_optimization(
            self.attempt(
                validation_status="PASS",
                comparison=BenchmarkComparison(
                    baseline=benchmark([10.0] * 4),
                    candidate=benchmark([7.9, 8.1, 10.0, 10.2]),
                ),
            )
        )
        assert got == OptimizationVerdict.REJECTED_NOT_ROBUST

    def test_a_real_but_small_gain(self) -> None:
        got = classify_optimization(
            self.attempt(
                validation_status="PASS",
                comparison=BenchmarkComparison(
                    baseline=benchmark([10.0] * 4), candidate=benchmark([9.7] * 4)
                ),
            )
        )
        assert got == OptimizationVerdict.REJECTED_BELOW_THRESHOLD

    def test_the_class_is_derived_from_fields_not_parsed_from_prose(self) -> None:
        """The verdict string is for humans and will be reworded. A label that
        tracked it would fragment the time series the first time it was."""
        attempt = self.attempt(
            validation_status="PASS",
            verdict="some entirely different wording nobody planned for",
            comparison=BenchmarkComparison(
                baseline=benchmark([10.0] * 4), candidate=benchmark([9.7] * 4)
            ),
        )
        assert classify_optimization(attempt) == OptimizationVerdict.REJECTED_BELOW_THRESHOLD

    def test_every_class_is_declared(self) -> None:
        """So the dashboard can enumerate them without guessing."""
        declared = set(OptimizationVerdict.ALL)
        assert len(declared) == len(OptimizationVerdict.ALL), "duplicate verdict class"
        for name in declared:
            assert name.islower() and " " not in name


class TestCardinality:
    """The failure that takes down Prometheus rather than misleading a human."""

    def test_no_metric_carries_a_migration_id(self, metrics: MigrationMetrics) -> None:
        """One new time series per migration, for ever. The id belongs in the
        log line — where it already is — and in the artifact."""
        record = migration_record()
        metrics.observe(record)
        exposition = generate_latest(metrics.registry).decode()
        assert record.migration_id not in exposition

    def test_no_label_name_suggests_an_unbounded_value(
        self, metrics: MigrationMetrics
    ) -> None:
        forbidden = {"migration_id", "id", "path", "source", "url", "branch", "timestamp"}
        for metric in metrics.registry.collect():
            for sample in metric.samples:
                overlap = set(sample.labels) & forbidden
                assert not overlap, f"{sample.name} carries unbounded label(s) {overlap}"

    def test_every_emitted_label_value_comes_from_a_bounded_set(
        self, metrics: MigrationMetrics
    ) -> None:
        """Drive a maximal record through and check every value that appears.

        This is the test that would catch someone adding a label sourced from a
        filename, a scenario name or an error message.
        """
        bounded: dict[str, set[str]] = {
            "outcome": {"succeeded", "failed", "exhausted", "refused", "fixed", "did_not_fix"},
            "risk": {level.value for level in RiskLevel},
            "stage": {stage.value for stage in MigrationStage},
            "status": {status.value for status in ValidationStatus},
            "result": {"pass", "fail", "skip"},
            "verdict": set(OptimizationVerdict.ALL),
            "role": {"baseline", "candidate"},
            "engine": {"pandas", "spark"},
            "disposition": {d.value for d in DeliveryDisposition},
            "agent": {
                "discovery", "planner", "spark_engineer", "testing",
                "validation", "repair", "optimizer", "delivery",
            },
            # The differ's declared check names. Adding one is a deliberate act;
            # a label sourced from arbitrary text would not be.
            "check": {
                "schema", "row_count", "null_counts", "numeric_tolerance",
                "duplicate_counts", "aggregate_checksums", "column_statistics",
            },
        }
        metrics.observe(maximal_record())
        from etl_migrator.domain.messages import AgentTelemetry

        for agent in bounded["agent"]:
            metrics.observe_telemetry(AgentTelemetry(agent=agent, tools_used=["t"]))

        seen = 0
        for metric in metrics.registry.collect():
            for sample in metric.samples:
                for label, val in sample.labels.items():
                    if label == "le":
                        # Prometheus's own bucket-boundary label. Its cardinality
                        # is the length of the bucket list, which is fixed at
                        # definition time — checked separately below.
                        continue
                    assert label in bounded, f"{sample.name}: unknown label {label!r}"
                    assert val in bounded[label], (
                        f"{sample.name}: label {label}={val!r} is outside its bounded set"
                    )
                    seen += 1
        assert seen > 0, "no labelled samples were emitted, so this proved nothing"

    def test_histogram_buckets_are_declared_and_finite(self) -> None:
        """Buckets are the other multiplier on a histogram's series count, and
        the default set is wrong for everything this system measures — a
        migration takes minutes, so every observation would land in +Inf."""
        from etl_migrator.observability import metrics as module

        declared = [
            module.MIGRATION_BUCKETS,
            module.STAGE_BUCKETS,
            module.EXECUTION_BUCKETS,
            module.SPEEDUP_BUCKETS,
            module.NOISE_BUCKETS,
        ]
        for buckets in declared:
            assert buckets[-1] == float("inf"), "the last bucket must be +Inf"
            assert list(buckets) == sorted(buckets), "buckets must ascend"
            assert len(buckets) <= 12, "too many buckets multiplies the series count"


class TestDeduplication:
    def test_the_same_record_is_counted_once(self, metrics: MigrationMetrics) -> None:
        """Temporal retries `persist_artifacts`. Without this a retried activity
        inflates every counter it touches."""
        record = migration_record()
        assert metrics.observe(record) is True
        assert metrics.observe(record) is False
        assert value(metrics, "etl_migrations_total", outcome="succeeded", risk="high") == 1

    def test_different_migrations_are_counted_separately(
        self, metrics: MigrationMetrics
    ) -> None:
        metrics.observe(migration_record(migration_id="mig-a"))
        metrics.observe(migration_record(migration_id="mig-b"))
        assert value(metrics, "etl_migrations_total", outcome="succeeded", risk="high") == 2

    def test_the_dedup_memory_is_bounded(self) -> None:
        """A worker runs for weeks. An unbounded set of ids is a slow leak."""
        metrics = MigrationMetrics(dedup_size=8)
        for index in range(50):
            metrics.observe(migration_record(migration_id=f"mig-{index}"))
        assert len(metrics._seen) <= 8
        assert value(metrics, "etl_migrations_total", outcome="succeeded", risk="high") == 50


class TestExposition:
    def test_the_endpoint_serves_the_registry(self, metrics: MigrationMetrics) -> None:
        metrics.observe(migration_record())
        with MetricsServer(metrics.registry, port=0) as server:
            body = urllib.request.urlopen(
                f"http://127.0.0.1:{server.port}/metrics", timeout=10
            ).read().decode()
        assert "etl_migrations_total" in body
        assert "# HELP etl_migrations_total" in body

    def test_health_does_not_depend_on_temporal(self, metrics: MigrationMetrics) -> None:
        """A liveness probe that failed on a Temporal outage would turn one
        outage into a crash-loop across every replica simultaneously."""
        with MetricsServer(metrics.registry, port=0) as server:
            response = urllib.request.urlopen(
                f"http://127.0.0.1:{server.port}/healthz", timeout=10
            )
        assert response.status == 200

    def test_unknown_paths_are_not_served(self, metrics: MigrationMetrics) -> None:
        with (
            MetricsServer(metrics.registry, port=0) as server,
            pytest.raises(urllib.error.HTTPError) as excinfo,
        ):
            urllib.request.urlopen(
                f"http://127.0.0.1:{server.port}/../etc/passwd", timeout=10
            )
        assert excinfo.value.code == 404

    def test_every_metric_carries_help_text(self, metrics: MigrationMetrics) -> None:
        """A series nobody can interpret is a series nobody uses."""
        for metric in metrics.registry.collect():
            assert metric.documentation, f"{metric.name} has no HELP"
            assert len(metric.documentation) > 20, f"{metric.name}: HELP is too terse"


def maximal_record() -> MigrationRecord:
    """A record exercising every branch of `observe`, for the cardinality sweep."""
    record = migration_record(
        checks=[
            CheckResult(name=name, passed=True)
            for name in (
                "schema", "row_count", "null_counts", "numeric_tolerance",
                "duplicate_counts", "aggregate_checksums", "column_statistics",
            )
        ],
        repair=RepairOutcome(
            succeeded=False,
            exhausted=True,
            attempts=[
                RepairAttempt(attempt=1, validation_status=ValidationStatus.FAIL),
                RepairAttempt(attempt=2, admitted=False, rejection_reason="repeat"),
                RepairAttempt(
                    attempt=3,
                    strategy=RepairStrategy(
                        category=RiskCategory.NULL_SEMANTICS,
                        approach="filter_nulls",
                        description="Filter null group keys before aggregating.",
                    ),
                    validation_status=ValidationStatus.PASS,
                ),
            ],
        ),
        optimization=OptimizationOutcome(
            applied=True,
            baseline=benchmark([10.0] * 4),
            final=benchmark([8.0] * 4),
            accepted_strategy=strategy("reduce_shuffle_partitions"),
            attempts=[
                OptimizationAttempt(
                    attempt=1,
                    strategy=strategy("reduce_shuffle_partitions"),
                    accepted=True,
                    validation_status="PASS",
                    comparison=BenchmarkComparison(
                        baseline=benchmark([10.0] * 4), candidate=benchmark([8.0] * 4)
                    ),
                ),
            ],
        ),
        delivery=DeliveryOutcome(
            decision=DeliveryDecision(
                disposition=DeliveryDisposition.DRAFT, reason="validation failed"
            ),
            audit=ClaimAudit(checked=2),
        ),
    )
    record.stages = [
        StageRecord(stage=stage, started_at=at(0), ended_at=at(5), succeeded=True)
        for stage in MigrationStage
    ]
    assert record.validation is not None
    record.validation.legacy_execution = ExecutionResult(
        engine="pandas", succeeded=True, duration_seconds=0.3
    )
    record.validation.spark_execution = ExecutionResult(
        engine="spark", succeeded=True, duration_seconds=10.0
    )
    return record


class TestGrafanaDashboard:
    """A dashboard querying a metric nobody exports is a panel of "No data".

    That failure is silent and durable: the JSON is valid, Grafana renders it,
    and the panel simply never shows anything. Renaming a metric — or exporting
    one that was only ever planned — produces exactly that. So every metric name
    in every query is checked against the registry the code actually builds.
    """

    DASHBOARD = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "observability"
        / "grafana"
        / "etl-migration.json"
    )

    @classmethod
    def dashboard(cls) -> dict:
        import json

        return json.loads(cls.DASHBOARD.read_text("utf-8"))  # type: ignore[no-any-return]

    @staticmethod
    def exported_names(metrics: MigrationMetrics) -> set[str]:
        """Every series name the registry can produce, suffixes included."""
        names: set[str] = set()
        for metric in metrics.registry.collect():
            names.add(metric.name)
            for suffix in ("_total", "_count", "_sum", "_bucket", "_created"):
                names.add(f"{metric.name}{suffix}")
            for sample in metric.samples:
                names.add(sample.name)
        return names

    def test_the_dashboard_is_in_provisioning_form_not_export_form(self) -> None:
        """`__inputs` means "ask the user to pick a datasource on import".

        Grafana substitutes `${DS_...}` placeholders during the *import wizard*.
        File provisioning runs no wizard -- it loads the JSON verbatim -- so an
        exported-for-sharing dashboard dropped into a provisioning directory
        looks for a datasource whose uid is the literal string
        `${DS_PROMETHEUS}`, finds none, and errors every panel.

        Which renders as twelve panels of "No data": indistinguishable from
        having collected no metrics, and the reason this is asserted rather
        than assumed.
        """
        assert "__inputs" not in self.dashboard(), (
            "dashboard is in export form; provisioning cannot resolve its "
            "${DS_...} placeholders"
        )

    def test_every_datasource_it_references_is_one_provisioning_defines(self) -> None:
        import re

        provisioning = (
            self.DASHBOARD.parent / "provisioning" / "datasources" / "prometheus.yml"
        ).read_text("utf-8")
        defined = set(re.findall(r"^\s*uid:\s*(\S+)\s*$", provisioning, flags=re.M))
        assert defined, "no datasource uid is provisioned"

        referenced: set[str] = set()

        def walk(node: object) -> None:
            if isinstance(node, dict):
                source = node.get("datasource")
                if isinstance(source, dict) and isinstance(source.get("uid"), str):
                    referenced.add(source["uid"])
                for value in node.values():
                    walk(value)
            elif isinstance(node, list):
                for value in node:
                    walk(value)

        walk(self.dashboard().get("panels", []))
        assert referenced, "no panel names a datasource"
        assert referenced <= defined, (
            f"panels reference datasource uid(s) {sorted(referenced - defined)}, "
            f"but provisioning defines {sorted(defined)}"
        )

    def test_counters_are_totalled_over_the_window_not_rated_per_second(self) -> None:
        """`rate()` is the idiom for a counter that ticks constantly. This
        system completes a handful of migrations an hour, so a per-second rate
        of them rounds to zero at every sane time range -- the dashboard
        rendered correct queries against real data and displayed `declined
        0 req/s`, which reads exactly like a broken pipeline.

        `increase(...[$__range])` answers the question actually being asked:
        how many happened in the window I am looking at.
        """
        for panel in self.dashboard().get("panels", []):
            for target in panel.get("targets", []):
                expr = target.get("expr", "")
                assert "rate(" not in expr.replace("increase(", ""), (
                    f"{panel.get('title')!r} uses rate() on a low-frequency counter: {expr}"
                )
                assert "$__rate_interval" not in expr, (
                    f"{panel.get('title')!r} still windows on $__rate_interval: {expr}"
                )

    def test_no_panel_claims_its_numbers_are_requests(self) -> None:
        """`reqps` was the leftover default. Nothing here is a request, and the
        unit is what puts "req/s" under every number on the page."""
        for panel in self.dashboard().get("panels", []):
            unit = panel.get("fieldConfig", {}).get("defaults", {}).get("unit")
            assert unit != "reqps", f"{panel.get('title')!r} is labelled in requests per second"

    def test_the_dashboard_file_is_valid_json_with_panels(self) -> None:
        board = self.dashboard()
        assert board["panels"], "no panels — the tests below would prove nothing"
        assert board["uid"] == "etl-migration"

    def test_every_queried_metric_is_one_the_code_exports(
        self, metrics: MigrationMetrics
    ) -> None:
        import re

        # Populate the registry so histograms expose their _bucket/_sum/_count.
        metrics.observe(maximal_record())
        exported = self.exported_names(metrics)

        queried: set[str] = set()
        for panel in self.dashboard()["panels"]:
            for target in panel["targets"]:
                queried.update(re.findall(r"\betl_[a-z0-9_]+", target["expr"]))

        assert queried, "no etl_ metrics were queried, so this proved nothing"
        unknown = queried - exported
        assert not unknown, (
            f"the dashboard queries metric(s) the code does not export: {sorted(unknown)}. "
            "These panels would render 'No data' for ever."
        )

    def test_every_panel_explains_what_it_is_for(self) -> None:
        """A panel whose meaning lives only in the author's head gets
        misread at 3am, which is the only time anyone looks at it."""
        for panel in self.dashboard()["panels"]:
            description = panel.get("description", "")
            assert len(description) > 40, f"panel {panel['title']!r} has no real description"

    def test_the_noise_panel_marks_the_refusal_ceiling(self) -> None:
        """The number that decides whether a comparison is readable at all
        should be visible on the chart, not remembered."""
        from etl_migrator.domain.optimization import DEFAULT_MAX_NOISE_RATIO

        panel = next(
            p for p in self.dashboard()["panels"] if "noise" in p["title"].lower()
        )
        expressions = " ".join(t["expr"] for t in panel["targets"])
        assert f"vector({DEFAULT_MAX_NOISE_RATIO})" in expressions

    def test_rate_queries_use_the_dashboard_interval(self) -> None:
        """A hardcoded `[5m]` breaks whenever the scrape interval changes, and
        breaks quietly — the query still returns, just wrong."""
        import re

        for panel in self.dashboard()["panels"]:
            for target in panel["targets"]:
                for window in re.findall(r"rate\([^)]*\[([^\]]+)\]", target["expr"]):
                    assert window == "$__rate_interval", (
                        f"panel {panel['title']!r} hardcodes a [{window}] window"
                    )
