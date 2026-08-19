"""Logging, metrics and correlation ids.

The two halves answer different questions and are deliberately not merged.
Structured logs carry `migration_id` on every line, so one grep joins agent
reasoning, Spark execution and the pull request. Metrics carry no id at all —
that would be a new time series per migration — and instead aggregate what the
record already measured.
"""

from etl_migrator.observability.logging import (
    configure_logging,
    get_logger,
    migration_context,
    stage_context,
)
from etl_migrator.observability.metrics import (
    MigrationMetrics,
    OptimizationVerdict,
    classify_optimization,
    get_metrics,
    reset_metrics,
)
from etl_migrator.observability.server import HEALTH_PATH, METRICS_PATH, MetricsServer

__all__ = [
    "HEALTH_PATH",
    "METRICS_PATH",
    "MetricsServer",
    "MigrationMetrics",
    "OptimizationVerdict",
    "classify_optimization",
    "configure_logging",
    "get_logger",
    "get_metrics",
    "migration_context",
    "reset_metrics",
    "stage_context",
]
