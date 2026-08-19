"""An in-memory GitHub that enforces GitHub's rules.

The counterpart of `ScriptedChatCompletionClient`: it replaces the outside world
without replacing the logic under test. A stub returning canned success would
let `ensure_branch` pass on a client that never checks whether the branch
exists, which is the bug the method exists to prevent.

So it models state and rejects what real GitHub rejects:

* creating a ref that already exists is 422 "Reference already exists"
* `PUT /contents` without a `sha` for an existing path is 422
* `PUT /contents` with a stale `sha` is 409
* writing to a branch that does not exist is 404
* opening a second PR from the same head is 422
* `content` must be valid base64

`requests` records the full call log, so a test can assert how a result was
reached: that a second `ensure_branch` did a lookup and no POST, for instance,
which is the difference between idempotent and lucky.
"""

from __future__ import annotations

import base64
import hashlib
from dataclasses import dataclass, field
from typing import Any

from etl_migrator.github.transport import Response


def _sha(*parts: str) -> str:
    return hashlib.sha1("\x00".join(parts).encode("utf-8")).hexdigest()


@dataclass
class _PullRequest:
    number: int
    title: str
    body: str
    head: str
    base: str
    draft: bool
    labels: list[str] = field(default_factory=list)
    state: str = "open"


@dataclass
class RecordedRequest:
    method: str
    path: str
    json: dict[str, Any] | None
    params: dict[str, str] | None
    status: int


class InMemoryGitHub:
    """A tiny GitHub. Enough of it to be unforgiving."""

    def __init__(
        self,
        *,
        repository: str = "acme/data-platform",
        default_branch: str = "main",
        seed_files: dict[str, str] | None = None,
    ) -> None:
        self.repository = repository
        self.default_branch = default_branch
        root = _sha("root", default_branch)
        #: branch -> head commit sha
        self.branches: dict[str, str] = {default_branch: root}
        #: (branch, path) -> content
        self.files: dict[tuple[str, str], str] = {
            (default_branch, path): content for path, content in (seed_files or {}).items()
        }
        self.pulls: list[_PullRequest] = []
        self.requests: list[RecordedRequest] = []
        self._next_number = 1
        #: Set to a status code to make the *next* matching call fail, so tests
        #: can drive the retry paths without waiting for real flakiness.
        self.fail_next: dict[str, int] = {}

    # -- helpers -----------------------------------------------------------
    def file_content(self, branch: str, path: str) -> str | None:
        return self.files.get((branch, path))

    def open_pull(self, head: str) -> _PullRequest | None:
        return next(
            (p for p in self.pulls if p.head == head and p.state == "open"), None
        )

    def calls(self, method: str, contains: str = "") -> list[RecordedRequest]:
        return [
            r for r in self.requests if r.method == method and contains in r.path
        ]

    def _blob_sha(self, branch: str, path: str) -> str:
        return _sha("blob", path, self.files[(branch, path)])

    # -- the transport interface -------------------------------------------
    def request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        params: dict[str, str] | None = None,
    ) -> Response:
        response = self._dispatch(method, path, json, params)
        self.requests.append(
            RecordedRequest(
                method=method, path=path, json=json, params=params, status=response.status
            )
        )
        return response

    def _dispatch(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None,
        params: dict[str, str] | None,
    ) -> Response:
        forced = self.fail_next.pop(f"{method} {path}", None)
        if forced is not None:
            return Response(forced, {"message": "injected failure", "status": str(forced)})

        prefix = f"/repos/{self.repository}"
        if not path.startswith(prefix):
            return Response(404, {"message": "Not Found", "status": "404"})
        rest = path[len(prefix) :]

        if method == "GET" and rest == "":
            return Response(
                200,
                {"full_name": self.repository, "default_branch": self.default_branch},
            )
        if method == "GET" and rest.startswith("/git/ref/heads/"):
            return self._get_ref(rest.removeprefix("/git/ref/heads/"))
        if method == "POST" and rest == "/git/refs":
            return self._create_ref(body or {})
        if method == "GET" and rest.startswith("/contents/"):
            return self._get_contents(rest.removeprefix("/contents/"), params or {})
        if method == "PUT" and rest.startswith("/contents/"):
            return self._put_contents(rest.removeprefix("/contents/"), body or {})
        if method == "GET" and rest.startswith("/pulls"):
            return self._list_pulls(params or {})
        if method == "POST" and rest == "/pulls":
            return self._create_pull(body or {})
        if method == "POST" and rest.startswith("/issues/") and rest.endswith("/labels"):
            number = rest.removeprefix("/issues/").removesuffix("/labels")
            return self._add_labels(number, body or {})

        return Response(404, {"message": "Not Found", "status": "404"})

    # -- refs --------------------------------------------------------------
    def _get_ref(self, branch: str) -> Response:
        sha = self.branches.get(branch)
        if sha is None:
            return Response(404, {"message": "Not Found", "status": "404"})
        return Response(
            200,
            {"ref": f"refs/heads/{branch}", "object": {"sha": sha, "type": "commit"}},
        )

    def _create_ref(self, body: dict[str, Any]) -> Response:
        ref = str(body.get("ref", ""))
        sha = str(body.get("sha", ""))
        if not ref.startswith("refs/heads/"):
            return Response(422, {"message": "Reference must be a valid ref name"})
        branch = ref.removeprefix("refs/heads/")
        if branch in self.branches:
            return Response(422, {"message": "Reference already exists"})
        if sha not in set(self.branches.values()):
            return Response(422, {"message": "Object does not exist"})

        self.branches[branch] = sha
        # A new branch inherits the tree of whatever commit it points at.
        source = next(b for b, head in self.branches.items() if head == sha and b != branch)
        for (existing_branch, path), content in list(self.files.items()):
            if existing_branch == source:
                self.files[(branch, path)] = content
        return Response(
            201, {"ref": ref, "object": {"sha": sha, "type": "commit"}}
        )

    # -- contents ----------------------------------------------------------
    def _get_contents(self, path: str, params: dict[str, str]) -> Response:
        branch = params.get("ref", self.default_branch)
        content = self.files.get((branch, path))
        if content is None:
            return Response(404, {"message": "Not Found", "status": "404"})
        return Response(
            200,
            {
                "path": path,
                "sha": self._blob_sha(branch, path),
                "encoding": "base64",
                "content": base64.b64encode(content.encode()).decode(),
            },
        )

    def _put_contents(self, path: str, body: dict[str, Any]) -> Response:
        branch = str(body.get("branch", self.default_branch))
        if branch not in self.branches:
            return Response(404, {"message": "Branch not found", "status": "404"})
        if not body.get("message"):
            return Response(422, {"message": "message wasn't supplied"})

        try:
            content = base64.b64decode(str(body.get("content", "")), validate=True).decode()
        except Exception:
            return Response(422, {"message": "content must be base64 encoded"})

        exists = (branch, path) in self.files
        supplied = body.get("sha")
        if exists and not supplied:
            return Response(
                422,
                {"message": f"Invalid request.\n\n{path} exists but no sha wasn't supplied."},
            )
        if exists and supplied != self._blob_sha(branch, path):
            return Response(
                409, {"message": f"{path} does not match {supplied}", "status": "409"}
            )
        if not exists and supplied:
            return Response(422, {"message": "sha supplied for a path that does not exist"})

        self.files[(branch, path)] = content
        self.branches[branch] = _sha("commit", branch, path, content)
        return Response(
            200 if exists else 201,
            {"content": {"path": path, "sha": self._blob_sha(branch, path)}},
        )

    # -- pulls -------------------------------------------------------------
    def _list_pulls(self, params: dict[str, str]) -> Response:
        head = params.get("head", "")
        wanted = head.split(":", 1)[1] if ":" in head else head
        found = [
            p
            for p in self.pulls
            if p.state == params.get("state", "open") and (not wanted or p.head == wanted)
        ]
        return Response(200, [self._render_pull(p) for p in found])

    def _create_pull(self, body: dict[str, Any]) -> Response:
        head = str(body.get("head", ""))
        base = str(body.get("base", ""))
        if head not in self.branches:
            return Response(422, {"message": f"head branch {head!r} does not exist"})
        if base not in self.branches:
            return Response(422, {"message": f"base branch {base!r} does not exist"})
        if self.open_pull(head) is not None:
            return Response(
                422, {"message": f"A pull request already exists for {head}."}
            )
        if self.branches[head] == self.branches[base]:
            return Response(422, {"message": "No commits between base and head"})
        if not body.get("title"):
            return Response(422, {"message": "title wasn't supplied"})

        pull = _PullRequest(
            number=self._next_number,
            title=str(body["title"]),
            body=str(body.get("body", "")),
            head=head,
            base=base,
            draft=bool(body.get("draft", False)),
        )
        self._next_number += 1
        self.pulls.append(pull)
        return Response(201, self._render_pull(pull))

    def _add_labels(self, number: str, body: dict[str, Any]) -> Response:
        pull = next((p for p in self.pulls if str(p.number) == number), None)
        if pull is None:
            return Response(404, {"message": "Not Found", "status": "404"})
        for label in body.get("labels", []):
            if label not in pull.labels:
                pull.labels.append(str(label))
        return Response(200, [{"name": name} for name in pull.labels])

    def _render_pull(self, pull: _PullRequest) -> dict[str, Any]:
        return {
            "number": pull.number,
            "html_url": f"https://github.com/{self.repository}/pull/{pull.number}",
            "draft": pull.draft,
            "title": pull.title,
            "body": pull.body,
            "head": {"ref": pull.head},
            "base": {"ref": pull.base},
            "labels": [{"name": name} for name in pull.labels],
        }
