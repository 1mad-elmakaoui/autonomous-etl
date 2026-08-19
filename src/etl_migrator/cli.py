"""Command line entrypoint.

Local (no server):
    etl-migrator migrate examples/customer_pipeline/legacy_pipeline.py
    etl-migrator inspect  examples/customer_pipeline/legacy_pipeline.py
    etl-migrator profile  examples/customer_pipeline/input
    etl-migrator gate     .workspace/<id>/legacy_pipeline_spark.py
    etl-migrator patterns join

Durable (needs Temporal — `docker compose up -d`):
    etl-migrator worker
    etl-migrator submit  examples/customer_pipeline/legacy_pipeline.py
    etl-migrator status  <migration-id>
    etl-migrator approve <migration-id> --actor you --reason "checked the plan"
    etl-migrator abort   <migration-id> --reason "wrong source"
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Annotated, Any, TypeVar

import typer

from etl_migrator.config import get_settings
from etl_migrator.domain.artifacts import MigrationRecord
from etl_migrator.domain.delivery import DeliveryOutcome
from etl_migrator.domain.enums import TransformKind
from etl_migrator.domain.errors import MigrationError
from etl_migrator.domain.history import outcomes_by_category
from etl_migrator.domain.messages import ApprovalDecision
from etl_migrator.domain.optimization import (
    DEFAULT_MIN_SPEEDUP,
    DEFAULT_RUNS,
    OptimizationOutcome,
)
from etl_migrator.github import client_from_settings
from etl_migrator.knowledge.history import load_records
from etl_migrator.knowledge.patterns import DEFAULT_CATALOGUE
from etl_migrator.llm.factory import build_model_client_factory
from etl_migrator.observability import configure_logging
from etl_migrator.pipeline.local import (
    LocalMigrationPipeline,
    MigrationRequest,
    auto_approve,
    require_manual_approval,
)
from etl_migrator.tools.code_gate import analyze_generated_code
from etl_migrator.tools.data_profiler import profile_directory
from etl_migrator.tools.source_inspector import inspect_source

app = typer.Typer(
    add_completion=False,
    help="Autonomously migrate legacy ETL pipelines to validated PySpark.",
)

SourceArg = Annotated[Path, typer.Argument(help="Legacy pipeline source file.")]
InputDirOpt = Annotated[
    Path | None,
    typer.Option("--input-dir", "-i", help="Directory holding the pipeline's input data."),
]
ScenarioOpt = Annotated[
    str, typer.Option("--scenario", help="Fixture scenario when LLM_PROVIDER=scripted.")
]
MigrationIdArg = Annotated[str, typer.Argument(help="Migration id (also the workflow id).")]

R = TypeVar("R")


def _bootstrap() -> None:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)


def _resolve_input_dir(source: Path, input_dir: Path | None) -> Path:
    return input_dir or (source.parent / "input")


def _require_source(source: Path) -> None:
    if not source.is_file():
        typer.secho(f"source not found: {source}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)


def _print_optimization(outcome: OptimizationOutcome) -> None:
    """Show the measurement, not just the conclusion.

    Both timings and both error bars are printed whatever the verdict, so a
    reader can disagree with the decision on the evidence rather than having to
    trust it. "Nothing kept" with the numbers underneath is a result; "nothing
    kept" on its own is indistinguishable from a stage that failed to run.
    """
    colour = typer.colors.GREEN if outcome.applied else typer.colors.BRIGHT_BLACK
    typer.echo("  optimisation    : ", nl=False)
    typer.secho(
        f"{outcome.speedup:.2f}x applied" if outcome.applied else "nothing kept",
        fg=colour,
        bold=True,
    )
    if outcome.baseline is not None:
        typer.echo(f"      {outcome.baseline.render()}")
    if outcome.final is not None and outcome.final is not outcome.baseline:
        typer.echo(f"      {outcome.final.render()}")
    for attempt in outcome.attempts:
        mark = typer.colors.GREEN if attempt.accepted else typer.colors.YELLOW
        typer.secho(f"      {attempt.render()}", fg=mark)
    if outcome.accepted_strategy is not None:
        claimed = outcome.accepted_strategy.expected_speedup
        typer.echo(
            f"      agent expected {claimed:.2f}x, measured {outcome.speedup:.2f}x"
        )


def _print_delivery(outcome: DeliveryOutcome) -> None:
    """Show what was opened, or precisely why nothing was."""
    if outcome.skipped_reason is not None:
        typer.echo("  pull request    : ", nl=False)
        typer.secho("not opened", fg=typer.colors.BRIGHT_BLACK, bold=True)
        typer.echo(f"      {outcome.skipped_reason}")
        return

    pr = outcome.pull_request
    decision = outcome.decision
    colour = typer.colors.YELLOW if (pr and pr.draft) else typer.colors.GREEN
    typer.echo("  pull request    : ", nl=False)
    if pr is None:
        typer.secho("none", fg=typer.colors.BRIGHT_BLACK, bold=True)
    else:
        typer.secho(
            f"#{pr.number} {'(draft)' if pr.draft else ''}".strip(), fg=colour, bold=True
        )
        typer.echo(f"      {pr.url}")
        if pr.labels:
            typer.echo(f"      labels: {', '.join(pr.labels)}")
    if decision is not None:
        typer.echo(f"      {decision.reason}")
    if outcome.branch is not None:
        verb = "created" if outcome.branch.created else "reused"
        typer.echo(f"      branch {verb}: {outcome.branch.name}")
    for path in outcome.files:
        typer.echo(f"      file: {path}")
    if outcome.audit is not None:
        # Printed on the happy path too: "we checked the prose against the
        # measurements" is only reassuring if it is visible when it passes.
        mark = typer.colors.GREEN if outcome.audit.passed else typer.colors.RED
        typer.secho(f"      {outcome.audit.render()}", fg=mark)
    if outcome.narrative_revisions:
        typer.echo(
            f"      narrative revised {outcome.narrative_revisions} time(s) to match "
            "the record"
        )


def _print_record(record: MigrationRecord) -> None:
    typer.secho("\n" + "=" * 72, fg=typer.colors.BRIGHT_BLACK)
    colour = typer.colors.RED if record.failed else typer.colors.GREEN
    typer.secho(f"migration {record.migration_id}", fg=colour, bold=True)
    typer.secho("=" * 72, fg=typer.colors.BRIGHT_BLACK)
    typer.echo(f"  source          : {record.source_path}")
    typer.echo(f"  stage           : {record.stage.value}")
    if record.spec is not None:
        typer.echo(f"  transformations : {len(record.spec.transformations)}")
    if record.plan is not None:
        typer.echo(f"  plan steps      : {len(record.plan.steps)}")
        typer.echo(f"  semantic diffs  : {len(record.plan.all_semantic_differences)}")
        typer.echo(f"  overall risk    : {record.plan.overall_risk.value}")
        typer.echo(f"  approval needed : {record.plan.requires_human_approval}")
    if record.codegen is not None:
        typer.echo(
            f"  generated       : {record.codegen.code.filename} "
            f"({record.codegen.code.line_count} lines)"
        )
        verdict = "PASS" if record.codegen.static_analysis.passed else "FAIL"
        typer.echo(
            f"  gate            : {verdict} after "
            f"{record.codegen.gate_iterations} submission(s)"
        )
    if record.validation is not None:
        report = record.validation.report
        colour = (
            typer.colors.GREEN if report.status.value == "PASS" else typer.colors.RED
        )
        typer.echo("  validation      : ", nl=False)
        typer.secho(report.status.value, fg=colour, bold=True)
        for check in report.checks:
            mark = "skip" if check.skipped else ("ok" if check.passed else "FAIL")
            typer.echo(f"      [{mark:>4}] {check.name}: {check.detail}")
        if record.validation.test_run is not None:
            typer.echo(f"  generated tests : {record.validation.test_run.render()}")
        for difference in report.differences[:5]:
            typer.secho(f"      ! {difference.render()}", fg=typer.colors.YELLOW)
        if record.validation.diagnosis is not None:
            d = record.validation.diagnosis
            typer.secho(
                f"\n  diagnosis ({d.root_cause_category.value}, "
                f"confidence {d.confidence:.0%}): {d.summary}",
                fg=typer.colors.YELLOW,
            )
            typer.echo(f"  suggested fix   : {d.suggested_fix}")
    if record.repair is not None:
        repair = record.repair
        colour = typer.colors.GREEN if repair.succeeded else typer.colors.RED
        typer.echo("  repair          : ", nl=False)
        typer.secho(
            "SUCCEEDED" if repair.succeeded else "exhausted", fg=colour, bold=True
        )
        for attempt in repair.attempts:
            typer.echo(f"      {attempt.render()}")
        best = repair.best_attempt
        if not repair.succeeded and best is not None:
            typer.secho(
                f"      closest: attempt {best.attempt}, {best.differences} differences left",
                fg=typer.colors.YELLOW,
            )
    if record.optimization is not None:
        _print_optimization(record.optimization)
    if record.delivery is not None:
        _print_delivery(record.delivery)
    typer.echo(f"  artifacts       : {record.artifact_dir}")
    typer.echo(f"  duration        : {record.total_duration_seconds:.2f}s")
    if record.failed:
        typer.secho(f"\n  failure: {record.failure_reason}", fg=typer.colors.RED)
    elif record.validation is None and record.plan is not None:
        typer.echo("\n  validation was not run; required checks would be:")
        for name in record.plan.effective_required_checks():
            typer.echo(f"    - {name}")


# ---------------------------------------------------------------------------
# Local commands (no Temporal server needed)
# ---------------------------------------------------------------------------
@app.command()
def migrate(
    source: SourceArg,
    input_dir: InputDirOpt = None,
    scenario: ScenarioOpt = "customer_pipeline",
    migration_id: Annotated[
        str | None, typer.Option("--migration-id", help="Reuse a specific correlation id.")
    ] = None,
    require_approval: Annotated[
        bool,
        typer.Option(
            "--require-approval/--auto-approve",
            help="Stop at the approval gate instead of auto-approving. Submit through "
            "Temporal to wait durably for a real decision.",
        ),
    ] = False,
    validate: Annotated[
        bool,
        typer.Option(
            "--validate/--no-validate",
            help="Execute both pipelines and diff their outputs. Needs a JVM and adds "
            "roughly a minute; without it the migration is unproven.",
        ),
    ] = True,
    generated_tests: Annotated[
        bool,
        typer.Option("--tests/--no-tests", help="Generate and run a pytest suite."),
    ] = True,
    repair: Annotated[
        bool,
        typer.Option(
            "--repair/--no-repair",
            help="Attempt autonomous repair when validation fails.",
        ),
    ] = True,
    max_repair_attempts: Annotated[
        int,
        typer.Option(
            "--max-repair-attempts",
            min=1,
            max=10,
            help="Distinct strategies to try before escalating to a human.",
        ),
    ] = 3,
    optimize: Annotated[
        bool,
        typer.Option(
            "--optimize/--no-optimize",
            help="Benchmark the correct migration and try to make it faster. Each "
            "attempt costs a full re-validation and two benchmarks.",
        ),
    ] = True,
    max_optimization_attempts: Annotated[
        int, typer.Option("--max-optimization-attempts", min=1, max=5)
    ] = 2,
    benchmark_runs: Annotated[
        int,
        typer.Option(
            "--benchmark-runs",
            min=2,
            max=20,
            help="Timed executions per configuration, after a discarded warm-up. "
            "Fewer runs means a wider error bar, and a measurement too noisy to "
            "read is refused rather than rounded up.",
        ),
    ] = DEFAULT_RUNS,
    min_speedup: Annotated[
        float,
        typer.Option(
            "--min-speedup",
            min=1.001,
            help="Speedup an optimisation must measure to be kept.",
        ),
    ] = DEFAULT_MIN_SPEEDUP,
    pull_request: Annotated[
        bool,
        typer.Option(
            "--pr/--no-pr",
            help="Open a pull request when the migration finishes. Needs "
            "ETLM_GITHUB_TOKEN and ETLM_GITHUB_REPOSITORY; without them the stage "
            "reports that it was skipped rather than failing the migration.",
        ),
    ] = True,
) -> None:
    """Run a migration in-process, through to a measured correctness verdict."""
    _bootstrap()
    _require_source(source)
    settings = get_settings()
    github = client_from_settings(settings) if pull_request else None

    pipeline = LocalMigrationPipeline(
        settings,
        build_model_client_factory(settings, scenario=scenario),
        approval=require_manual_approval() if require_approval else auto_approve(),
        run_validation=validate,
        run_generated_tests=generated_tests,
        run_repair=repair,
        max_repair_attempts=max_repair_attempts,
        run_optimization=optimize and validate,
        max_optimization_attempts=max_optimization_attempts,
        benchmark_runs=benchmark_runs,
        min_speedup=min_speedup,
        github=github,
        run_delivery=pull_request,
    )
    request = MigrationRequest(
        source, _resolve_input_dir(source, input_dir), migration_id=migration_id,
        scenario=scenario,
    )

    try:
        record = asyncio.run(pipeline.run(request))
    except MigrationError as exc:
        typer.secho(f"\nmigration failed: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    _print_record(record)
    if record.failed:
        raise typer.Exit(code=1)


@app.command("history")
def history_cmd(
    limit: Annotated[
        int,
        typer.Option("--limit", min=1, max=1000, help="Most recent migrations to read."),
    ] = 200,
) -> None:
    """Show what past migrations established — and what they did not.

    Recomputed from the persisted records every time, so it cannot drift from
    what actually happened. A strategy with fewer than three attempts is
    reported without a success rate: two data points are not a rate, and a
    number would invite you to act as though they were.
    """
    _bootstrap()
    from etl_migrator.knowledge.history import load_history

    settings = get_settings()
    past = load_history(settings.workspace_dir, limit=limit)

    typer.secho("\n" + "=" * 72, fg=typer.colors.BRIGHT_BLACK)
    typer.secho("migration history", fg=typer.colors.CYAN, bold=True)
    typer.secho("=" * 72, fg=typer.colors.BRIGHT_BLACK)
    typer.echo(f"  workspace : {settings.workspace_dir}")
    typer.echo(f"  migrations: {past.migrations_observed} ({past.validated} validated)")

    if not past.sufficient:
        typer.secho(
            "\n  Too few migrations for any of this to be evidence. Treat it as "
            "anecdote.",
            fg=typer.colors.YELLOW,
        )

    for title, store in (
        ("repair strategies", past.repair_strategies),
        ("optimisation approaches", past.optimization_approaches),
    ):
        if not store:
            continue
        typer.secho(f"\n  {title}", bold=True)
        for key in sorted(store):
            evidence = store[key]
            colour = (
                typer.colors.RED
                if evidence.discouraged
                else typer.colors.BRIGHT_BLACK
                if not evidence.sufficient
                else typer.colors.GREEN
            )
            typer.secho(f"    {evidence.render()}", fg=colour)

    outcomes = outcomes_by_category(load_records(settings.workspace_dir, limit=limit))
    if outcomes:
        typer.secho("\n  validation outcome by declared risk category", bold=True)
        for category, (passed, total) in outcomes.items():
            typer.echo(f"    {category}: {passed}/{total} migrations passed")

    if past.migrations_observed == 0:
        typer.echo("\n  Nothing recorded yet. Run a migration and try again.")


@app.command("inspect")
def inspect_cmd(source: SourceArg) -> None:
    """Show the deterministic AST facts the Discovery agent sees. No LLM involved."""
    _bootstrap()
    typer.echo(inspect_source(source).render())


@app.command("profile")
def profile_cmd(
    directory: Annotated[Path, typer.Argument(help="Directory of input datasets.")],
) -> None:
    """Profile real input data: dtypes, null counts, cardinality, broadcast eligibility."""
    _bootstrap()
    profiles = profile_directory(directory)
    if not profiles:
        typer.secho(f"no readable datasets in {directory}", fg=typer.colors.YELLOW)
        raise typer.Exit(code=1)
    for profile in profiles:
        typer.echo(profile.render())
        typer.echo("")


@app.command("gate")
def gate_cmd(
    path: Annotated[Path, typer.Argument(help="Generated PySpark module to check.")],
) -> None:
    """Run the static security/quality gate against a generated module."""
    _bootstrap()
    report = analyze_generated_code(path.read_text(encoding="utf-8"))
    typer.echo(report.render())
    colour = typer.colors.GREEN if report.passed else typer.colors.RED
    typer.secho(f"\ngate: {'PASS' if report.passed else 'FAIL'}", fg=colour, bold=True)
    raise typer.Exit(code=0 if report.passed else 1)


@app.command("patterns")
def patterns_cmd(
    kind: Annotated[
        str | None, typer.Argument(help="Transformation kind, e.g. 'join'. Omit to list all.")
    ] = None,
) -> None:
    """Inspect the structured migration-pattern catalogue the planner queries."""
    _bootstrap()
    if kind is None:
        typer.echo("recorded patterns for: " + ", ".join(DEFAULT_CATALOGUE.kinds))
        return
    try:
        parsed = TransformKind(kind)
    except ValueError:
        typer.secho(
            f"unknown kind '{kind}'. Known: {', '.join(DEFAULT_CATALOGUE.kinds)}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2) from None
    typer.echo(DEFAULT_CATALOGUE.render(parsed))


# ---------------------------------------------------------------------------
# Durable commands (Temporal)
# ---------------------------------------------------------------------------
@app.command("worker")
def worker_cmd(
    role: Annotated[
        str,
        typer.Option(
            "--role",
            help="all: every activity on one queue (the default, right for a laptop). "
            "agent: the LLM and GitHub activities. execution: only the activities that "
            "run untrusted generated code — deployed with no internet egress.",
        ),
    ] = "all",
) -> None:
    """Run a Temporal worker serving ETLMigrationWorkflow and its activities."""
    _bootstrap()
    from etl_migrator.temporal.worker import (
        WorkerRole,
        activity_names_for,
        queue_for,
        run_worker,
    )

    try:
        worker_role = WorkerRole(role)
    except ValueError:
        typer.secho(
            f"unknown role {role!r}; expected one of "
            f"{', '.join(r.value for r in WorkerRole)}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2) from None

    settings = get_settings()
    served = activity_names_for(settings, worker_role)
    typer.secho(
        f"worker[{worker_role.value}] connecting to {settings.temporal_host} "
        f"(namespace={settings.temporal_namespace}, "
        f"queue={queue_for(settings, worker_role)})",
        fg=typer.colors.CYAN,
    )
    typer.echo(f"  serving {len(served)} activities: {', '.join(served)}")
    try:
        asyncio.run(run_worker(settings, role=worker_role))
    except KeyboardInterrupt:
        typer.echo("\nworker stopped")


@app.command("submit")
def submit_cmd(
    source: SourceArg,
    input_dir: InputDirOpt = None,
    scenario: ScenarioOpt = "customer_pipeline",
    migration_id: Annotated[
        str | None, typer.Option("--migration-id", help="Reuse a specific correlation id.")
    ] = None,
    wait: Annotated[
        bool, typer.Option("--wait/--no-wait", help="Block until the migration finishes.")
    ] = False,
    generated_tests: Annotated[
        bool,
        typer.Option("--tests/--no-tests", help="Generate and run a pytest suite."),
    ] = True,
    max_repair_attempts: Annotated[
        int, typer.Option("--max-repair-attempts", min=1, max=10)
    ] = 3,
    optimize: Annotated[
        bool,
        typer.Option(
            "--optimize/--no-optimize",
            help="Run the benchmark-and-optimise child workflow once correct.",
        ),
    ] = True,
    benchmark_runs: Annotated[
        int, typer.Option("--benchmark-runs", min=2, max=20)
    ] = DEFAULT_RUNS,
    min_speedup: Annotated[
        float, typer.Option("--min-speedup", min=1.001)
    ] = DEFAULT_MIN_SPEEDUP,
) -> None:
    """Submit a migration to Temporal. Survives worker restarts."""
    _bootstrap()
    _require_source(source)
    settings = get_settings()

    from etl_migrator.temporal import client as tc

    params = tc.build_input(
        source,
        _resolve_input_dir(source, input_dir),
        migration_id=migration_id,
        scenario=scenario,
    )
    params = params.model_copy(
        update={
            "run_generated_tests": generated_tests,
            "max_repair_attempts": max_repair_attempts,
            "optimize_enabled": optimize,
            "benchmark_runs": benchmark_runs,
            "min_speedup": min_speedup,
        }
    )

    async def _run() -> MigrationRecord | None:
        client = await tc.connect(settings)
        handle = await tc.submit(client, settings, params)
        typer.secho(f"submitted {params.migration_id}", fg=typer.colors.GREEN, bold=True)
        typer.echo(f"  run id : {handle.first_execution_run_id}")
        typer.echo(f"  status : etl-migrator status {params.migration_id}")
        typer.echo(f"  approve: etl-migrator approve {params.migration_id} --actor you")
        if not wait:
            return None
        typer.echo("\nwaiting for completion (Ctrl-C detaches; the migration keeps running)...")
        return await handle.result()

    record = _with_temporal(_run)
    if record is not None:
        _print_record(record)
        if record.failed:
            raise typer.Exit(code=1)


@app.command("status")
def status_cmd(migration_id: MigrationIdArg) -> None:
    """Query a running or finished migration without touching a database."""
    _bootstrap()
    settings = get_settings()

    from etl_migrator.temporal import client as tc

    async def _run() -> None:
        client = await tc.connect(settings)
        status = await tc.query_status(client, migration_id)
        typer.secho(f"\n{status.migration_id}", bold=True)
        typer.echo(f"  stage            : {status.stage.value}")
        typer.echo(f"  risk             : {status.risk.value}")
        typer.echo(f"  awaiting approval: {status.awaiting_approval}")
        if status.approval is not None:
            typer.echo(
                f"  decision         : "
                f"{'approved' if status.approval.approved else 'rejected'} "
                f"by {status.approval.actor}"
            )
        typer.echo(f"  finished         : {status.finished}")
        if status.failed:
            typer.secho(f"  failure          : {status.failure_reason}", fg=typer.colors.RED)
        typer.echo("\n  stages:")
        for stage in status.stages:
            mark = "?" if stage.succeeded is None else ("ok" if stage.succeeded else "FAIL")
            seconds = stage.duration_seconds
            took = f"{seconds:.2f}s" if seconds is not None else "running"
            typer.echo(f"    [{mark:>4}] {stage.stage.value:<18} {took:>10}  {stage.detail or ''}")
        if status.awaiting_approval:
            typer.secho(
                f"\n  waiting for: etl-migrator approve {migration_id} --actor you",
                fg=typer.colors.YELLOW,
            )

    _with_temporal(_run)


@app.command("approve")
def approve_cmd(
    migration_id: MigrationIdArg,
    actor: Annotated[str, typer.Option("--actor", help="Who is deciding. Recorded.")],
    reason: Annotated[str | None, typer.Option("--reason")] = None,
    reject: Annotated[
        bool, typer.Option("--reject", help="Reject instead of approving.")
    ] = False,
) -> None:
    """Release (or reject) a migration paused at the human-approval gate."""
    _bootstrap()
    settings = get_settings()

    from etl_migrator.temporal import client as tc

    decision = ApprovalDecision(approved=not reject, actor=actor, reason=reason)

    async def _run() -> None:
        client = await tc.connect(settings)
        await tc.send_approval(client, migration_id, decision)

    _with_temporal(_run)
    verb = "rejected" if reject else "approved"
    typer.secho(f"{migration_id} {verb} by {actor}", fg=typer.colors.GREEN)


@app.command("abort")
def abort_cmd(
    migration_id: MigrationIdArg,
    reason: Annotated[str, typer.Option("--reason", help="Why. Recorded on the migration.")],
) -> None:
    """Abort a running migration. Recorded distinctly from a failure."""
    _bootstrap()
    settings = get_settings()

    from etl_migrator.temporal import client as tc

    async def _run() -> None:
        client = await tc.connect(settings)
        await tc.send_abort(client, migration_id, reason)

    _with_temporal(_run)
    typer.secho(f"{migration_id} aborted", fg=typer.colors.YELLOW)


def _with_temporal(coro_factory: Callable[[], Coroutine[Any, Any, R]]) -> R:
    """Run a Temporal coroutine, turning a connection failure into advice."""
    try:
        return asyncio.run(coro_factory())
    except MigrationError as exc:
        typer.secho(f"error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    except RuntimeError as exc:
        if "Failed client connect" not in str(exc) and "connect" not in str(exc).lower():
            raise
        settings = get_settings()
        typer.secho(
            f"cannot reach Temporal at {settings.temporal_host}: {exc}\n"
            "start one with:  docker compose up -d temporal\n"
            "or run without a server:  etl-migrator migrate <source>",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1) from exc


def main() -> None:  # pragma: no cover - thin wrapper
    try:
        app()
    except MigrationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":  # pragma: no cover
    main()
