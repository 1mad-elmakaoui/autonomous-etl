"""GitHub integration.

Three layers, for the same reason the LLM package has three: `GitHubClient`
holds the endpoint knowledge and the idempotency rules, `GitHubTransport` is the
only thing that touches a socket, and `InMemoryGitHub` substitutes the network
in tests without substituting any of the logic above it.

`client_from_settings` is the only place a token is read, and it returns `None`
rather than raising when GitHub is not configured — running a migration without
a repository is a normal thing to do, not an error.
"""

from __future__ import annotations

from etl_migrator.config import Settings
from etl_migrator.github.client import GitHubClient, RepositoryRef
from etl_migrator.github.fake import InMemoryGitHub
from etl_migrator.github.transport import (
    API_VERSION,
    GitHubError,
    GitHubTransport,
    HttpxTransport,
    Response,
)
from etl_migrator.observability import get_logger

log = get_logger(__name__)

__all__ = [
    "API_VERSION",
    "GitHubClient",
    "GitHubError",
    "GitHubTransport",
    "HttpxTransport",
    "InMemoryGitHub",
    "RepositoryRef",
    "Response",
    "client_from_settings",
]


def client_from_settings(settings: Settings) -> GitHubClient | None:
    """Build a live client, or None when GitHub is not configured.

    Returning None is deliberate. A missing token is not a broken configuration
    — it is the ordinary case for a local run — and the delivery stage reports
    the omission rather than failing a migration that has otherwise succeeded.
    Both values must be present: a token without a repository has nowhere to
    push, and a repository without a token cannot authenticate.
    """
    token = settings.github_token.get_secret_value() if settings.github_token else ""
    repository = settings.github_repository

    if not token or not repository:
        missing = [
            name
            for name, value in (
                ("ETLM_GITHUB_TOKEN", token),
                ("ETLM_GITHUB_REPOSITORY", repository),
            )
            if not value
        ]
        log.info("github.not_configured", missing=missing)
        return None

    return GitHubClient(HttpxTransport(token), repository)
