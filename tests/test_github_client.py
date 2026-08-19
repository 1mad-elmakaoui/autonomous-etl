"""The GitHub client, driven against an in-memory GitHub that enforces the rules.

The point of these tests is idempotency, because that is the property Temporal
actually needs. An activity can be retried after its side effect has landed —
worker dies between the API call succeeding and the result being recorded — so
every write here must be safe to run twice. "Safe" specifically means: the
second run must not fail, must not duplicate anything, and must not overwrite
work a human may already be looking at.

`InMemoryGitHub` is what makes that testable rather than assumed. It returns 422
for a duplicate ref, 409 for a stale blob sha and 422 for a second PR from the
same head, exactly as the real API does, so a client that skipped its
lookup-then-create would fail these tests rather than sail through them. The
call log lets each test assert *how* the result was reached, which is the
difference between idempotent and merely lucky.

No network: nothing here resolves a hostname.
"""

from __future__ import annotations

import base64

import pytest

from etl_migrator.domain.delivery import FileChange
from etl_migrator.github import GitHubClient, GitHubError, InMemoryGitHub


@pytest.fixture
def hub() -> InMemoryGitHub:
    return InMemoryGitHub(
        repository="acme/data-platform",
        default_branch="main",
        seed_files={"README.md": "# platform\n"},
    )


@pytest.fixture
def client(hub: InMemoryGitHub) -> GitHubClient:
    return GitHubClient(hub, hub.repository)


def change(path: str = "migrations/m1/pipeline.py", content: str = "x = 1\n") -> FileChange:
    return FileChange(path=path, content=content, message="Add pipeline")


class TestRepositoryRef:
    def test_rejects_a_name_that_is_not_owner_slash_repo(self) -> None:
        with pytest.raises(GitHubError, match="owner/repo"):
            GitHubClient(InMemoryGitHub(), "not-a-repo")

    @pytest.mark.parametrize("bad", ["", "/repo", "owner/", "a/b/c"])
    def test_rejects_malformed_names(self, bad: str) -> None:
        with pytest.raises(GitHubError):
            GitHubClient(InMemoryGitHub(), bad)


class TestDefaultBranch:
    def test_is_read_from_the_repository_not_assumed(
        self, hub: InMemoryGitHub
    ) -> None:
        """Hardcoding `main` fails by branching from the wrong commit, silently."""
        hub.default_branch = "trunk"
        hub.branches = {"trunk": hub.branches["main"]}
        assert GitHubClient(hub, hub.repository).default_branch() == "trunk"


class TestEnsureBranch:
    def test_creates_a_branch_off_the_default(
        self, client: GitHubClient, hub: InMemoryGitHub
    ) -> None:
        ref = client.ensure_branch("etl-migration/m1")
        assert ref.created
        assert ref.name in hub.branches
        assert hub.branches[ref.name] == hub.branches["main"]

    def test_a_second_call_reuses_rather_than_failing(
        self, client: GitHubClient, hub: InMemoryGitHub
    ) -> None:
        """The retry case. Real GitHub answers 422 to the second POST."""
        first = client.ensure_branch("etl-migration/m1")
        second = client.ensure_branch("etl-migration/m1")

        assert first.created and not second.created
        assert first.sha == second.sha
        assert len(hub.calls("POST", "/git/refs")) == 1, "it tried to create it twice"

    def test_an_existing_branch_is_never_force_moved(
        self, client: GitHubClient, hub: InMemoryGitHub
    ) -> None:
        """A retry must not rewrite history a reviewer may already be reading.

        The branch has moved on since it was created — someone committed, or an
        earlier run of this activity wrote files. Re-running `ensure_branch`
        must leave it exactly where it is, not reset it to the base.
        """
        client.ensure_branch("etl-migration/m1")
        hub.branches["etl-migration/m1"] = "deadbeef" * 5
        moved = hub.branches["etl-migration/m1"]

        ref = client.ensure_branch("etl-migration/m1")
        assert ref.sha == moved
        assert hub.branches["etl-migration/m1"] == moved

    def test_a_missing_base_is_reported_clearly(self, client: GitHubClient) -> None:
        with pytest.raises(GitHubError, match="does not exist"):
            client.ensure_branch("etl-migration/m1", base="no-such-branch")

    def test_branch_sha_returns_none_rather_than_raising_on_404(
        self, client: GitHubClient
    ) -> None:
        """404 here is an answer, not a failure. Conflating them would make
        every first-time branch creation an error path."""
        assert client.branch_sha("never-existed") is None


class TestEnsureFile:
    def test_creates_a_file_with_base64_content(
        self, client: GitHubClient, hub: InMemoryGitHub
    ) -> None:
        client.ensure_branch("etl-migration/m1")
        client.ensure_file(change(content="print('hi')\n"), branch="etl-migration/m1")

        assert hub.file_content("etl-migration/m1", "migrations/m1/pipeline.py") == (
            "print('hi')\n"
        )
        put = hub.calls("PUT", "/contents/")[0]
        assert put.json is not None
        assert base64.b64decode(put.json["content"]).decode() == "print('hi')\n"

    def test_a_create_sends_no_sha_and_an_update_sends_one(
        self, client: GitHubClient, hub: InMemoryGitHub
    ) -> None:
        """The `sha` field is the whole create-versus-update distinction.

        Sending it on a create is a 422; omitting it on an update is also a 422.
        Reading the existing blob immediately before the write is what makes one
        method handle both.
        """
        client.ensure_branch("etl-migration/m1")
        client.ensure_file(change(content="v1\n"), branch="etl-migration/m1")
        client.ensure_file(change(content="v2\n"), branch="etl-migration/m1")

        writes = hub.calls("PUT", "/contents/")
        assert len(writes) == 2
        assert writes[0].json is not None and "sha" not in writes[0].json
        assert writes[1].json is not None and "sha" in writes[1].json
        assert all(w.status < 300 for w in writes)
        assert hub.file_content("etl-migration/m1", "migrations/m1/pipeline.py") == "v2\n"

    def test_rewriting_identical_content_is_harmless(
        self, client: GitHubClient, hub: InMemoryGitHub
    ) -> None:
        """The literal retry: same activity, same inputs, run twice."""
        client.ensure_branch("etl-migration/m1")
        client.ensure_file(change(), branch="etl-migration/m1")
        client.ensure_file(change(), branch="etl-migration/m1")

        assert all(w.status < 300 for w in hub.calls("PUT", "/contents/"))
        assert hub.file_content("etl-migration/m1", "migrations/m1/pipeline.py") == "x = 1\n"

    def test_writing_to_a_missing_branch_raises(self, client: GitHubClient) -> None:
        with pytest.raises(GitHubError) as excinfo:
            client.ensure_file(change(), branch="no-such-branch")
        assert excinfo.value.status == 404

    def test_a_stale_sha_surfaces_as_a_conflict(
        self, client: GitHubClient, hub: InMemoryGitHub
    ) -> None:
        """Proof the fake is actually enforcing the rule.

        If the fake accepted any sha, the create/update test above would pass on
        a client that sent a hardcoded string, and this suite would be theatre.
        """
        client.ensure_branch("etl-migration/m1")
        client.ensure_file(change(), branch="etl-migration/m1")
        response = hub.request(
            "PUT",
            "/repos/acme/data-platform/contents/migrations/m1/pipeline.py",
            json={
                "message": "m",
                "content": base64.b64encode(b"tampered").decode(),
                "branch": "etl-migration/m1",
                "sha": "0" * 40,
            },
        )
        assert response.status == 409

    def test_a_file_seen_on_the_base_branch_is_updated_not_recreated(
        self, client: GitHubClient, hub: InMemoryGitHub
    ) -> None:
        """A new branch inherits the base tree, so a path may already exist."""
        client.ensure_branch("etl-migration/m1")
        client.ensure_file(
            FileChange(path="README.md", content="# updated\n", message="edit"),
            branch="etl-migration/m1",
        )
        write = hub.calls("PUT", "/contents/README.md")[0]
        assert write.json is not None and "sha" in write.json
        assert hub.file_content("main", "README.md") == "# platform\n", "base was mutated"


class TestEnsurePullRequest:
    def prepare(self, client: GitHubClient) -> str:
        branch = "etl-migration/m1"
        client.ensure_branch(branch)
        client.ensure_file(change(), branch=branch)
        return branch

    def test_opens_a_pull_request(self, client: GitHubClient, hub: InMemoryGitHub) -> None:
        branch = self.prepare(client)
        pr = client.ensure_pull_request(
            head=branch, base="main", title="Migrate the pipeline", body="body"
        )
        assert pr.created
        assert pr.number == 1
        assert pr.url.endswith("/pull/1")
        assert hub.pulls[0].title == "Migrate the pipeline"

    def test_a_second_call_returns_the_open_one(
        self, client: GitHubClient, hub: InMemoryGitHub
    ) -> None:
        """Real GitHub 422s on a duplicate; a retried activity must not."""
        branch = self.prepare(client)
        first = client.ensure_pull_request(
            head=branch, base="main", title="Migrate", body="one"
        )
        second = client.ensure_pull_request(
            head=branch, base="main", title="Different title", body="two"
        )
        assert first.created and not second.created
        assert first.number == second.number
        assert len(hub.pulls) == 1
        assert len(hub.calls("POST", "/pulls")) == 1

    def test_a_retry_does_not_rewrite_the_body(
        self, client: GitHubClient, hub: InMemoryGitHub
    ) -> None:
        """A human may already have commented on what is written there."""
        branch = self.prepare(client)
        client.ensure_pull_request(head=branch, base="main", title="Migrate", body="one")
        client.ensure_pull_request(head=branch, base="main", title="Migrate", body="TWO")
        assert hub.pulls[0].body == "one"

    def test_drafts_are_opened_as_drafts(self, client: GitHubClient) -> None:
        branch = self.prepare(client)
        pr = client.ensure_pull_request(
            head=branch, base="main", title="Needs a human", body="b", draft=True
        )
        assert pr.draft

    def test_the_head_query_is_owner_qualified(
        self, client: GitHubClient, hub: InMemoryGitHub
    ) -> None:
        """GitHub ignores an unqualified `head`, silently returning every open
        PR — so the lookup would match a stranger's branch and skip the create."""
        branch = self.prepare(client)
        client.find_open_pull_request(head=branch)
        query = hub.calls("GET", "/pulls")[0]
        assert query.params is not None
        assert query.params["head"] == f"acme:{branch}"

    def test_labels_are_applied_and_are_a_set_union(
        self, client: GitHubClient, hub: InMemoryGitHub
    ) -> None:
        branch = self.prepare(client)
        pr = client.ensure_pull_request(head=branch, base="main", title="Migrate", body="b")
        client.add_labels(pr.number, ["autonomous-etl", "high-risk"])
        labels = client.add_labels(pr.number, ["autonomous-etl", "optimised"])
        assert sorted(labels) == ["autonomous-etl", "high-risk", "optimised"]
        assert sorted(hub.pulls[0].labels) == ["autonomous-etl", "high-risk", "optimised"]

    def test_no_labels_makes_no_request(
        self, client: GitHubClient, hub: InMemoryGitHub
    ) -> None:
        assert client.add_labels(1, []) == []
        assert hub.calls("POST", "/labels") == []


class TestErrorReporting:
    def test_github_message_is_preserved_verbatim(
        self, client: GitHubClient, hub: InMemoryGitHub
    ) -> None:
        """"Resource not accessible by integration" and "Reference already
        exists" call for different human responses; collapsing both into
        "request failed" wastes the only useful thing in the response."""
        hub.fail_next["GET /repos/acme/data-platform"] = 403
        with pytest.raises(GitHubError) as excinfo:
            client.default_branch()
        assert excinfo.value.status == 403
        assert "injected failure" in str(excinfo.value)

    def test_the_url_is_carried_on_the_error(self, client: GitHubClient) -> None:
        with pytest.raises(GitHubError) as excinfo:
            client.ensure_branch("m1", base="nope")
        assert "git/ref/heads/nope" in excinfo.value.url


class TestTheFakeIsUnforgiving:
    """Guard the guard.

    Every idempotency test above is only meaningful if the fake would have
    punished the naive implementation. These assert that it does.
    """

    def test_a_duplicate_ref_is_rejected(self, hub: InMemoryGitHub) -> None:
        body = {"ref": "refs/heads/dup", "sha": hub.branches["main"]}
        assert hub.request("POST", "/repos/acme/data-platform/git/refs", json=body).status == 201
        second = hub.request("POST", "/repos/acme/data-platform/git/refs", json=body)
        assert second.status == 422
        assert "already exists" in second.data["message"]

    def test_creating_over_an_existing_path_without_a_sha_is_rejected(
        self, hub: InMemoryGitHub
    ) -> None:
        response = hub.request(
            "PUT",
            "/repos/acme/data-platform/contents/README.md",
            json={
                "message": "m",
                "content": base64.b64encode(b"clobber").decode(),
                "branch": "main",
            },
        )
        assert response.status == 422
        assert hub.file_content("main", "README.md") == "# platform\n"

    def test_content_must_be_base64(self, hub: InMemoryGitHub) -> None:
        response = hub.request(
            "PUT",
            "/repos/acme/data-platform/contents/new.py",
            json={"message": "m", "content": "not base64!", "branch": "main"},
        )
        assert response.status == 422

    def test_a_second_pull_request_from_one_head_is_rejected(
        self, hub: InMemoryGitHub, client: GitHubClient
    ) -> None:
        client.ensure_branch("b1")
        client.ensure_file(change(), branch="b1")
        body = {"title": "t", "head": "b1", "base": "main", "body": ""}
        assert hub.request("POST", "/repos/acme/data-platform/pulls", json=body).status == 201
        assert hub.request("POST", "/repos/acme/data-platform/pulls", json=body).status == 422

    def test_a_pull_request_with_no_commits_is_rejected(
        self, hub: InMemoryGitHub, client: GitHubClient
    ) -> None:
        """An empty branch produces an unopenable PR, as on real GitHub."""
        client.ensure_branch("empty")
        response = hub.request(
            "POST",
            "/repos/acme/data-platform/pulls",
            json={"title": "t", "head": "empty", "base": "main", "body": ""},
        )
        assert response.status == 422
        assert "No commits" in response.data["message"]
