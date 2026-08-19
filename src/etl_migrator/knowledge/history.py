"""Reading past migrations off disk.

The corpus is the artifacts the system already writes — one
`migration_record.json` per migration, the same file the PR ships and the
metrics are derived from. There is no separate learning database to fall out of
step with reality, and no ingestion step that could quietly stop running: if a
migration happened, its record is there, and `harvest` recomputes the evidence
from scratch every time.

That also makes the whole thing auditable. Every claim the history makes can be
traced to a named migration id and a file you can open.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from etl_migrator.domain.artifacts import MigrationRecord
from etl_migrator.domain.history import MigrationHistory, harvest
from etl_migrator.observability import get_logger

log = get_logger(__name__)

RECORD_FILENAME = "migration_record.json"

#: Newest-first cap. A worker that has run for months should not read ten
#: thousand files to answer one tool call, and the recent past is the relevant
#: one anyway: a strategy that stopped working six months ago should stop
#: counting in its favour.
DEFAULT_LIMIT = 200


def load_records(workspace: Path, *, limit: int = DEFAULT_LIMIT) -> list[MigrationRecord]:
    """Load persisted records, newest first, skipping anything unreadable.

    A corrupt or half-written record is logged and skipped rather than raised.
    The history is an advisory input to an agent; refusing to plan a migration
    because an unrelated one from last month has a truncated JSON file would be
    a poor trade.
    """
    if not workspace.is_dir():
        return []

    candidates = sorted(
        workspace.glob(f"*/{RECORD_FILENAME}"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    records: list[MigrationRecord] = []
    skipped = 0
    for path in candidates[:limit]:
        try:
            records.append(MigrationRecord.model_validate_json(path.read_text("utf-8")))
        except (ValidationError, json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
            skipped += 1
            log.warning("history.unreadable_record", path=str(path), error=str(exc))
    if skipped:
        log.info("history.loaded", records=len(records), skipped=skipped)
    return records


def load_history(workspace: Path, *, limit: int = DEFAULT_LIMIT) -> MigrationHistory:
    """Everything learned from the migrations on disk."""
    history = harvest(load_records(workspace, limit=limit))
    log.info(
        "history.harvested",
        migrations=history.migrations_observed,
        validated=history.validated,
        repair_strategies=len(history.repair_strategies),
        optimization_approaches=len(history.optimization_approaches),
    )
    return history
