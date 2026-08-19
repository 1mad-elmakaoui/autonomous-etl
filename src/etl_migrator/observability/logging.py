"""Structured logging with a correlation id bound for the whole migration.

Every log line emitted anywhere below `migration_context(...)` carries
`migration_id` and `stage` without any call site passing them. That is what
makes a distributed run greppable: one id joins agent reasoning, Spark
execution and the GitHub PR.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import structlog

_configured = False


def configure_logging(level: str = "INFO", fmt: str = "console") -> None:
    """Idempotent. Safe to call from CLI entrypoints and from Temporal workers."""
    global _configured
    if _configured:
        return

    renderer: Any = (
        structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())
        if fmt == "console"
        else structlog.processors.JSONRenderer()
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelNamesMapping().get(level.upper(), logging.INFO)
        ),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )
    _configured = True


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    configure_logging()
    return structlog.get_logger(name)  # type: ignore[no-any-return]


@contextmanager
def migration_context(migration_id: str, **extra: str) -> Iterator[None]:
    """Bind a correlation id for the duration of a block."""
    tokens = structlog.contextvars.bind_contextvars(migration_id=migration_id, **extra)
    try:
        yield
    finally:
        structlog.contextvars.reset_contextvars(**tokens)


@contextmanager
def stage_context(stage: str) -> Iterator[None]:
    tokens = structlog.contextvars.bind_contextvars(stage=stage)
    try:
        yield
    finally:
        structlog.contextvars.reset_contextvars(**tokens)
