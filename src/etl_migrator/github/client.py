"""A typed GitHub client, idempotent by construction.

Every write is lookup-then-create, because a Temporal activity can be retried
after its side effect has landed. The activity that creates a branch may run
twice: once succeeding, then the worker dies before recording it, then again. If
"create branch" were just `POST /git/refs`, the second run would 422 and take
the migration down having done nothing wrong.

So `ensure_branch` asks first and reports `created=False` when it found one.
`ensure_file` reads the existing blob sha and sends it, turning a create into an
update instead of a 409. `ensure_pull_request` searches for an open PR from the
same head before opening another.

Endpoint shapes were checked against the live API:
`GET /repos/{o}/{r}/git/ref/heads/{branch}` returns
`{"ref", "object": {"sha", ...}}` and 404s with `{"message", "status"}`;
`GET /repos/{o}/{r}/contents/{path}` returns `{"sha", "encoding", "content"}`.
"""

from __future__ import annotations

import base64
from typing import Any

from etl_migrator.domain.delivery import BranchRef, FileChange, PullRequestRef
from etl_migrator.github.transport import GitHubError, GitHubTransport, Response
from etl_migrator.observability import get_logger

log = get_logger(__name__)


class RepositoryRef:
    """`owner/repo`, validated once so every call site can stop worrying."""

    __slots__ = ("owner", "repo")

    def __init__(self, full_name: str) -> None:
        parts = full_name.strip().split("/")
        if len(parts) != 2 or not all(parts):
            raise GitHubError(
                0,
                f"repository must be 'owner/repo', got {full_name!r}",
                url="(local validation)",
            )
        self.owner, self.repo = parts

    @property
    def full_name(self) -> str:
        return f"{self.owner}/{self.repo}"

    @property
    def base(self) -> str:
        return f"/repos/{self.owner}/{self.repo}"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.full_name


class GitHubClient:
    """Typed operations over a `GitHubTransport`."""

    def __init__(self, transport: GitHubTransport, repository: str) -> None:
        self._transport = transport
        self.repo = RepositoryRef(repository)

    # -- plumbing ----------------------------------------------------------
    def _call(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        params: dict[str, str] | None = None,
        allow: tuple[int, ...] = (),
    ) -> Response:
        """Issue a request, raising on anything not explicitly allowed."""
        response = self._transport.request(method, path, json=json, params=params)
        log.info("github.request", method=method, path=path, status=response.status)
        if response.ok or response.status in allow:
            return response
        message = "unknown error"
        if isinstance(response.data, dict):
            message = str(response.data.get("message", message))
            errors = response.data.get("errors")
            if errors:
                message = f"{message} ({errors})"
        raise GitHubError(response.status, message, url=path, body=response.data)

    # -- repository --------------------------------------------------------
    def default_branch(self) -> str:
        """The repository's own default branch.

        Read rather than assumed. Hardcoding `main` breaks on every repository
        that still uses `master` or that branches from `develop`, and it fails
        by silently branching from the wrong commit rather than by erroring.
        """
        response = self._call("GET", self.repo.base)
        data = response.data if isinstance(response.data, dict) else {}
        branch = data.get("default_branch")
        if not isinstance(branch, str) or not branch:
            raise GitHubError(
                response.status,
                "repository response carried no default_branch",
                url=self.repo.base,
                body=response.data,
            )
        return branch

    # -- refs --------------------------------------------------------------
    def branch_sha(self, branch: str) -> str | None:
        """The commit a branch points at, or None when it does not exist."""
        response = self._call(
            "GET", f"{self.repo.base}/git/ref/heads/{branch}", allow=(404,)
        )
        if response.status == 404:
            return None
        data = response.data if isinstance(response.data, dict) else {}
        obj = data.get("object") or {}
        sha = obj.get("sha") if isinstance(obj, dict) else None
        return sha if isinstance(sha, str) else None

    def ensure_branch(self, branch: str, *, base: str | None = None) -> BranchRef:
        """Create `branch` off `base`, or return the existing one.

        Note what this deliberately does *not* do: if the branch already exists
        it is left exactly where it is, never force-moved onto the new base.
        A retry must not rewrite history that a human may already be reviewing.
        """
        existing = self.branch_sha(branch)
        if existing is not None:
            log.info("github.branch_exists", branch=branch, sha=existing[:8])
            return BranchRef(name=branch, sha=existing, created=False)

        base_branch = base or self.default_branch()
        base_sha = self.branch_sha(base_branch)
        if base_sha is None:
            raise GitHubError(
                404,
                f"base branch {base_branch!r} does not exist, so {branch!r} cannot "
                "be created from it",
                url=f"{self.repo.base}/git/ref/heads/{base_branch}",
            )

        response = self._call(
            "POST",
            f"{self.repo.base}/git/refs",
            json={"ref": f"refs/heads/{branch}", "sha": base_sha},
        )
        data = response.data if isinstance(response.data, dict) else {}
        obj = data.get("object") or {}
        sha = obj.get("sha") if isinstance(obj, dict) else base_sha
        log.info("github.branch_created", branch=branch, base=base_branch)
        return BranchRef(name=branch, sha=str(sha), created=True)

    # -- contents ----------------------------------------------------------
    def file_sha(self, path: str, *, ref: str) -> str | None:
        """The blob sha of a file on a ref, or None when it is not there."""
        response = self._call(
            "GET", f"{self.repo.base}/contents/{path}", params={"ref": ref}, allow=(404,)
        )
        if response.status == 404:
            return None
        data = response.data
        if isinstance(data, list):
            # The path is a directory. Not a file, so there is no blob sha.
            return None
        sha = data.get("sha") if isinstance(data, dict) else None
        return sha if isinstance(sha, str) else None

    def ensure_file(self, change: FileChange, *, branch: str) -> str:
        """Write a file, creating or updating as the branch requires.

        The `sha` field is what distinguishes the two: absent means "create, and
        fail if it exists"; present means "update, and fail if you are working
        from a stale version". Reading it immediately before the write is what
        makes a retried activity idempotent rather than a 409.
        """
        existing = self.file_sha(change.path, ref=branch)
        payload: dict[str, Any] = {
            "message": change.message,
            "content": base64.b64encode(change.content.encode("utf-8")).decode("ascii"),
            "branch": branch,
        }
        if existing is not None:
            payload["sha"] = existing

        response = self._call("PUT", f"{self.repo.base}/contents/{change.path}", json=payload)
        data = response.data if isinstance(response.data, dict) else {}
        content = data.get("content") or {}
        sha = content.get("sha") if isinstance(content, dict) else None
        log.info(
            "github.file_written",
            path=change.path,
            branch=branch,
            action="update" if existing is not None else "create",
        )
        return str(sha) if sha else ""

    # -- pull requests -----------------------------------------------------
    def find_open_pull_request(self, *, head: str) -> PullRequestRef | None:
        """An already-open PR from this head branch, if there is one."""
        response = self._call(
            "GET",
            f"{self.repo.base}/pulls",
            params={"state": "open", "head": f"{self.repo.owner}:{head}"},
        )
        items = response.data if isinstance(response.data, list) else []
        if not items:
            return None
        pr = items[0]
        return PullRequestRef(
            number=int(pr["number"]),
            url=str(pr.get("html_url", "")),
            draft=bool(pr.get("draft", False)),
            labels=[str(label["name"]) for label in pr.get("labels", [])],
            created=False,
        )

    def ensure_pull_request(
        self, *, head: str, base: str, title: str, body: str, draft: bool = False
    ) -> PullRequestRef:
        """Open a PR from `head`, or return the one already open.

        A second call does not open a duplicate, and does not rewrite the title
        or body of a PR a human may already have commented on.
        """
        existing = self.find_open_pull_request(head=head)
        if existing is not None:
            log.info("github.pr_exists", number=existing.number, head=head)
            return existing

        response = self._call(
            "POST",
            f"{self.repo.base}/pulls",
            json={
                "title": title,
                "body": body,
                "head": head,
                "base": base,
                "draft": draft,
            },
        )
        data = response.data if isinstance(response.data, dict) else {}
        log.info("github.pr_created", number=data.get("number"), head=head, draft=draft)
        return PullRequestRef(
            number=int(data["number"]),
            url=str(data.get("html_url", "")),
            draft=bool(data.get("draft", draft)),
            created=True,
        )

    def add_labels(self, number: int, labels: list[str]) -> list[str]:
        """Add labels to a PR. GitHub treats this as a set union, so it is
        naturally idempotent and needs no lookup first."""
        if not labels:
            return []
        response = self._call(
            "POST", f"{self.repo.base}/issues/{number}/labels", json={"labels": labels}
        )
        items = response.data if isinstance(response.data, list) else []
        return [str(item["name"]) for item in items]
