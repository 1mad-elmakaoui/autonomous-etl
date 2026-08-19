"""Structured historical migration knowledge. Not RAG, a typed catalogue."""

from etl_migrator.knowledge.patterns import (
    DEFAULT_CATALOGUE,
    MigrationPattern,
    ObservedOutcome,
    PatternCatalogue,
)

__all__ = ["DEFAULT_CATALOGUE", "MigrationPattern", "ObservedOutcome", "PatternCatalogue"]
