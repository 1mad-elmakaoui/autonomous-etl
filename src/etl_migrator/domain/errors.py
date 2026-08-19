"""Typed errors.

The split matters for Temporal: `NonRetryableMigrationError` subclasses are
listed in `RetryPolicy.non_retryable_error_types`, so a malformed
agent contract fails fast instead of burning four LLM calls.
"""

from __future__ import annotations


class MigrationError(Exception):
    """Base class for every error raised by this system."""


class NonRetryableMigrationError(MigrationError):
    """Retrying cannot help: the input or configuration is wrong."""


class ConfigurationError(NonRetryableMigrationError):
    """Missing or contradictory configuration (e.g. provider set but no API key)."""


class UnsupportedSourceError(NonRetryableMigrationError):
    """The legacy source is not a language/dialect this system can migrate."""


class AgentContractError(MigrationError):
    """An agent returned something that does not satisfy its declared contract.

    Retryable: a second sample from the model often satisfies the schema.
    """


class ScriptExhaustedError(MigrationError):
    """The scripted (offline) model client ran out of recorded responses.

    This is deliberately loud. A silent fallback would let a broken agent loop
    look like a passing test, which is the failure mode this project exists to
    prevent.
    """


class StaticGateError(MigrationError):
    """Generated code failed the static gate after all repair iterations."""


class SandboxError(MigrationError):
    """Sandboxed execution of untrusted generated code failed."""
