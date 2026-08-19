"""Temporal worker bootstrap.

Two things here are load-bearing.

The sandbox configuration. Temporal re-imports modules per workflow instance to
catch non-determinism, which is valuable for workflow code and pointless for the
domain models. The passthrough list is exactly the modules the workflow touches.
Passing through `etl_migrator.domain` is only safe because that package imports
nothing but pydantic and the stdlib and has no import-time side effects, which a
test enforces.

The worker role. A worker serves everything (`ALL`, right for a laptop or
`docker compose`) or one half of a split deployment:

* `AGENT` runs the LLM-backed activities and GitHub delivery. Needs egress to a
  model provider and api.github.com. Never executes generated code.
* `EXECUTION` runs untrusted generated code in the sandbox and the differ that
  reads its output. Needs nothing but Temporal.

That split is what makes the NetworkPolicy in `k8s/` mean anything. Co-located,
the pod needs internet egress for the model provider and the untrusted
subprocess inherits it.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from enum import StrEnum

from temporalio.activity import _Definition as ActivityDefinition
from temporalio.client import Client
from temporalio.worker import Worker
from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner, SandboxRestrictions

from etl_migrator.activities.migration import (
    DeliveryActivities,
    MigrationActivities,
    ValidationActivities,
)
from etl_migrator.config import Settings
from etl_migrator.github import client_from_settings
from etl_migrator.llm.factory import ModelClientFactory
from etl_migrator.observability import MetricsServer, get_logger, get_metrics
from etl_migrator.temporal.client import connect
from etl_migrator.workflows.delivery import DeliveryWorkflow
from etl_migrator.workflows.migration import ETLMigrationWorkflow
from etl_migrator.workflows.optimization import OptimizationWorkflow
from etl_migrator.workflows.repair import RepairWorkflow
from etl_migrator.workflows.validation import ValidationWorkflow

log = get_logger(__name__)

#: Registered for logging and for the CLI to display. Kept in sync with
#: `MigrationActivities.all()` by a test.
ACTIVITY_NAMES: tuple[str, ...] = (
    "analyze_legacy_pipeline",
    "generate_migration_plan",
    "generate_spark_code",
    "run_static_analysis",
    "persist_artifacts",
    "run_legacy_pipeline",
    "run_spark_pipeline",
    "validate_outputs",
    "generate_tests",
    "run_tests",
    "diagnose_validation_failure",
    "propose_repair",
    "benchmark_spark",
    "propose_optimization",
    "propose_pr_narrative",
    "deliver_pull_request",
)


class WorkerRole(StrEnum):
    """Which half of the deployment this worker is.

    `ALL` is the default because a single worker is correct for development and
    for compose; the split exists for Kubernetes, where the two halves get
    different network policies.
    """

    ALL = "all"
    AGENT = "agent"
    EXECUTION = "execution"


WORKFLOW_CLASSES = (
    ETLMigrationWorkflow,
    ValidationWorkflow,
    RepairWorkflow,
    OptimizationWorkflow,
    DeliveryWorkflow,
)

#: Modules the workflow imports that are safe and desirable to pass through.
PASSTHROUGH_MODULES: tuple[str, ...] = (
    "etl_migrator.domain",
    "etl_migrator.activities",
    "pydantic",
    "pydantic_core",
)


def build_sandbox_runner() -> SandboxedWorkflowRunner:
    return SandboxedWorkflowRunner(
        restrictions=SandboxRestrictions.default.with_passthrough_modules(
            *PASSTHROUGH_MODULES
        )
    )


def queue_for(settings: Settings, role: WorkerRole) -> str:
    """Which task queue a worker in this role should poll.

    An unset `temporal_execution_task_queue` means "one queue for everything",
    so an EXECUTION worker configured that way polls the main queue and the
    deployment is simply not split. That is the right default: the split is a
    Kubernetes concern, and a laptop should not have to opt out of it.
    """
    if role is WorkerRole.EXECUTION and settings.temporal_execution_task_queue:
        return settings.temporal_execution_task_queue
    return settings.temporal_task_queue


def activities_for(
    role: WorkerRole,
    *,
    migration: MigrationActivities,
    validation: ValidationActivities,
    delivery: DeliveryActivities,
) -> list[Callable[..., object]]:
    """The activities a worker in this role registers.

    A function rather than a branch inlined into `build_worker`, so the
    partition can be asserted directly. A test checks that nothing an EXECUTION
    worker registers reaches for a model client — which is the property the
    NetworkPolicy depends on, and the one that would rot silently.
    """
    if role is WorkerRole.EXECUTION:
        return list(validation.execution_activities())
    if role is WorkerRole.AGENT:
        return [*migration.all(), *validation.reasoning_activities(), *delivery.all()]
    return [*migration.all(), *validation.all(), *delivery.all()]


def activity_names_for(settings: Settings, role: WorkerRole) -> list[str]:
    """Registered activity names for a role. Shared by the startup log and tests."""
    definitions = [
        ActivityDefinition.must_from_callable(a)
        for a in activities_for(
            role,
            migration=MigrationActivities(settings),
            validation=ValidationActivities(settings),
            delivery=DeliveryActivities(settings),
        )
    ]
    # `name` is Optional on the definition only because a dynamic activity has
    # none; every activity here is declared with an explicit name.
    return [d.name for d in definitions if d.name is not None]


def build_worker(
    client: Client,
    settings: Settings,
    *,
    role: WorkerRole = WorkerRole.ALL,
    factory: ModelClientFactory | None = None,
    activities: MigrationActivities | None = None,
    validation_activities: ValidationActivities | None = None,
    delivery_activities: DeliveryActivities | None = None,
    task_queue: str | None = None,
) -> Worker:
    """Construct a worker for one role. Separated from `run_worker` so tests can
    build one without a server.

    An EXECUTION worker registers *only* the activities that run untrusted code.
    That is the point: it is deployed with no internet egress, so registering an
    LLM-backed activity on it would produce a task it can accept and then fail,
    which is worse than not offering to do the work at all.
    """
    acts = activities or MigrationActivities(settings, factory=factory)
    validation = validation_activities or ValidationActivities(settings, factory=factory)
    # The GitHub client is built once here rather than per activity: a token is
    # read exactly once per worker, and a worker without one still starts and
    # serves every other activity — delivery is the only thing it cannot do.
    delivery = delivery_activities or DeliveryActivities(
        settings, github=client_from_settings(settings), factory=factory
    )

    registered = activities_for(
        role, migration=acts, validation=validation, delivery=delivery
    )

    return Worker(
        client,
        task_queue=task_queue or queue_for(settings, role),
        # Workflow code is pure and cheap to host, so every worker can serve it.
        # Only the activities are partitioned.
        workflows=list(WORKFLOW_CLASSES),
        activities=registered,
        workflow_runner=build_sandbox_runner(),
    )


async def run_worker(
    settings: Settings,
    *,
    role: WorkerRole = WorkerRole.ALL,
    factory: ModelClientFactory | None = None,
    stop_event: asyncio.Event | None = None,
    serve_metrics: bool = True,
) -> None:
    """Connect and serve until interrupted.

    The metrics endpoint comes up *before* the Temporal connection, which is
    deliberate: a worker that cannot reach Temporal is exactly when you want to
    be able to scrape it, and a `/healthz` that depended on the connection would
    turn one Temporal outage into a crash-loop across every replica at once.
    """
    metrics_server = (
        MetricsServer(get_metrics().registry, port=settings.metrics_port).start()
        if serve_metrics
        else None
    )
    try:
        await _serve(settings, role=role, factory=factory, stop_event=stop_event)
    finally:
        if metrics_server is not None:
            metrics_server.stop()


async def _serve(
    settings: Settings,
    *,
    role: WorkerRole,
    factory: ModelClientFactory | None,
    stop_event: asyncio.Event | None,
) -> None:
    client = await connect(settings)
    worker = build_worker(client, settings, role=role, factory=factory)
    log.info(
        "worker.start",
        role=role.value,
        task_queue=queue_for(settings, role),
        namespace=settings.temporal_namespace,
        workflows=[w.__name__ for w in WORKFLOW_CLASSES],
        activities=activity_names_for(settings, role),
        provider=settings.llm_provider.value,
    )
    if stop_event is None:
        await worker.run()
        return
    async with worker:
        await stop_event.wait()
    log.info("worker.stopped")
