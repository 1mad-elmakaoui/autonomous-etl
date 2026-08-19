"""The seam between the GitHub client and the network.

Same shape as the LLM layer, for the same reason. `GitHubClient` speaks in
typed domain terms and knows the endpoint vocabulary; `GitHubTransport` is the
one thing under it that touches a socket. Swapping the transport swaps the
network, not the logic — so the tests exercise the real client, the real request
bodies and the real error handling, and only the last inch is substituted.

The alternative — mocking `GitHubClient` itself in tests — would leave every
line that actually builds a request untested, which is precisely where the bugs
in an API integration live.
"""

from __future__ import annotations

from typing import Any, Protocol

import httpx

from etl_migrator.domain.errors import ConfigurationError, NonRetryableMigrationError

#: Pinned per GitHub's guidance so a future default change cannot alter response
#: shapes underneath us. Verified against the live API.
API_VERSION = "2022-11-28"
API_ROOT = "https://api.github.com"
DEFAULT_TIMEOUT = 30.0


class GitHubError(NonRetryableMigrationError):
    """A GitHub request failed in a way worth surfacing verbatim.

    Carries the status and GitHub's own message: "Reference already exists" and
    "Resource not accessible by integration" call for completely different
    responses from a human, and collapsing both into "GitHub request failed"
    wastes that.
    """

    def __init__(self, status: int, message: str, *, url: str, body: Any = None) -> None:
        self.status = status
        self.github_message = message
        self.url = url
        self.body = body
        super().__init__(f"GitHub {status} on {url}: {message}")


class Response:
    """A transport-agnostic response. Deliberately tiny."""

    __slots__ = ("data", "status")

    def __init__(self, status: int, data: Any) -> None:
        self.status = status
        self.data = data

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300


class GitHubTransport(Protocol):
    """One method. Everything the client needs is a request/response pair."""

    def request(
        self, method: str, path: str, *, json: dict[str, Any] | None = None,
        params: dict[str, str] | None = None,
    ) -> Response:
        """Perform one API call. `path` is repo-relative to the API root."""
        ...


class HttpxTransport:
    """The real one.

    Auth is a bearer token read from settings, never a literal. The token is
    held as the raw string here because that is what an `Authorization` header
    needs, and it is never logged: request logging records method, path and
    status only.
    """

    def __init__(self, token: str, *, root: str = API_ROOT, timeout: float = DEFAULT_TIMEOUT):
        if not token:
            raise ConfigurationError(
                "ETLM_GITHUB_TOKEN is required to talk to GitHub. Set it in .env "
                "(see .env.example) or export it; it is never read from source."
            )
        self._root = root.rstrip("/")
        self._client = httpx.Client(
            timeout=timeout,
            headers={
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": API_VERSION,
                "Authorization": f"Bearer {token}",
                "User-Agent": "autonomous-etl-migration-agent",
            },
        )

    def request(
        self, method: str, path: str, *, json: dict[str, Any] | None = None,
        params: dict[str, str] | None = None,
    ) -> Response:
        response = self._client.request(
            method, f"{self._root}{path}", json=json, params=params
        )
        try:
            data = response.json() if response.content else None
        except ValueError:
            data = {"message": response.text[:400]}
        return Response(response.status_code, data)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> HttpxTransport:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
