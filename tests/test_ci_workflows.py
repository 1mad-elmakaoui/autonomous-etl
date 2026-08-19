"""Policy checks on the CI configuration and the Dockerfile.

Neither GitHub Actions nor a Docker daemon is available where this suite runs,
which is the same problem the Temporal tier had and gets the same
answer: a large class of real mistakes is detectable statically, and refusing to
check anything because the full thing cannot run is how a workflow rots.

What is checked here is specifically the set of mistakes that are *silent* — a
green tick that means less than it appears to:

* CI billing a real LLM API instead of running against recorded fixtures;
* a secret pasted into a workflow file;
* a job granted write access it does not need;
* a test tier that stopped running because its marker was renamed;
* an image that quietly went back to running as root;
* a scanning step reintroducing a supply chain the project deliberately avoids.

What is *not* checked here: that the workflow runs. Only GitHub can answer that,
and this file does not pretend otherwise. What it does guarantee is that every
command CI runs has a local equivalent, so the first person to see a red build
can reproduce it.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

yaml = pytest.importorskip("yaml", reason="pyyaml is in the dev extra")

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_DIR = REPO_ROOT / ".github" / "workflows"
DOCKERFILE = REPO_ROOT / "Dockerfile"

#: Every third-party action this project is willing to depend on, and the
#: version it is pinned to. A new entry is a supply-chain decision, so the test
#: makes adding one deliberate rather than incidental.
ALLOWED_ACTIONS: dict[str, str] = {
    "actions/checkout": "v5",
    "actions/setup-python": "v6",
    "actions/setup-java": "v5",
    "actions/attest-build-provenance": "v3",
    "docker/setup-buildx-action": "v3",
    "docker/build-push-action": "v6",
    "docker/login-action": "v3",
    "docker/metadata-action": "v5",
}

#: In March 2026 an attacker force-pushed 75 of the 76 version tags in
#: `aquasecurity/trivy-action` so that trusted references served an infostealer
#: which harvested CI/CD secrets from Actions runners (GHSA-cxm3-wv7p-598c).
#: The scanner is still wanted; that delivery mechanism is not.
BANNED_ACTIONS: tuple[str, ...] = ("aquasecurity/trivy-action",)


def workflows() -> list[Path]:
    return sorted(WORKFLOW_DIR.glob("*.yml")) + sorted(WORKFLOW_DIR.glob("*.yaml"))


def load(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))  # type: ignore[no-any-return]


def steps_of(workflow: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        step
        for job in workflow.get("jobs", {}).values()
        for step in job.get("steps", [])
    ]


@pytest.fixture(scope="module")
def ci() -> dict[str, Any]:
    return load(WORKFLOW_DIR / "ci.yml")


@pytest.fixture(scope="module")
def release() -> dict[str, Any]:
    return load(WORKFLOW_DIR / "release.yml")


@pytest.fixture(scope="module")
def dockerfile() -> str:
    return DOCKERFILE.read_text(encoding="utf-8")


class TestWorkflowsAreWellFormed:
    def test_there_are_workflows_to_check(self) -> None:
        """Guard the guard: every test below vacuously passes on an empty glob."""
        assert workflows(), "no workflow files found — the rest of this file proves nothing"

    @pytest.mark.parametrize("path", workflows(), ids=lambda p: p.name)
    def test_each_workflow_parses(self, path: Path) -> None:
        workflow = load(path)
        assert isinstance(workflow, dict)
        assert workflow.get("jobs"), f"{path.name} declares no jobs"

    @pytest.mark.parametrize("path", workflows(), ids=lambda p: p.name)
    def test_each_workflow_has_a_trigger(self, path: Path) -> None:
        """`on` is the YAML 1.1 boolean `True` once parsed — a real trap when
        reading these programmatically, and the reason this helper exists."""
        workflow = load(path)
        triggers = workflow.get(True, workflow.get("on"))
        assert triggers, f"{path.name} has no trigger"

    @pytest.mark.parametrize("path", workflows(), ids=lambda p: p.name)
    def test_every_job_names_a_runner(self, path: Path) -> None:
        for name, job in load(path)["jobs"].items():
            assert job.get("runs-on"), f"{path.name}:{name} has no runs-on"


class TestSupplyChain:
    @pytest.mark.parametrize("path", workflows(), ids=lambda p: p.name)
    def test_every_action_is_allowlisted_and_pinned(self, path: Path) -> None:
        """An unpinned action is whatever its author pushed this morning.

        The trivy-action compromise is the argument: the tags did not change
        name, they changed *content*. An allowlist with an explicit version
        makes each dependency a decision someone made once, on purpose.
        """
        for step in steps_of(load(path)):
            uses = step.get("uses")
            if not uses:
                continue
            action, _, version = uses.partition("@")
            assert action in ALLOWED_ACTIONS, (
                f"{path.name} uses un-allowlisted action {action!r}. Adding one is a "
                "supply-chain decision: put it in ALLOWED_ACTIONS with a version."
            )
            assert version, f"{path.name}: {action} is not pinned to any version"
            assert version == ALLOWED_ACTIONS[action], (
                f"{path.name}: {action} pinned to {version}, allowlist says "
                f"{ALLOWED_ACTIONS[action]}"
            )

    @pytest.mark.parametrize("path", workflows(), ids=lambda p: p.name)
    def test_no_compromised_action_is_used(self, path: Path) -> None:
        text = path.read_text(encoding="utf-8")
        for banned in BANNED_ACTIONS:
            for line in text.splitlines():
                stripped = line.strip()
                if banned in stripped and not stripped.startswith("#"):
                    pytest.fail(
                        f"{path.name} references {banned}, which served an "
                        "infostealer from force-pushed tags in March 2026 "
                        "(GHSA-cxm3-wv7p-598c)"
                    )

    @pytest.mark.parametrize("path", workflows(), ids=lambda p: p.name)
    def test_the_scanner_version_is_pinned(self, path: Path) -> None:
        """`apt-get install trivy` unpinned is the same mutable-reference
        problem in a different package manager."""
        text = path.read_text(encoding="utf-8")
        if "trivy" not in text:
            return
        assert re.search(r'TRIVY_VERSION:\s*"\d+\.\d+\.\d+"', text), (
            f"{path.name} installs trivy without pinning TRIVY_VERSION"
        )
        assert "trivy=${TRIVY_VERSION}" in text


class TestSecrets:
    @pytest.mark.parametrize("path", workflows(), ids=lambda p: p.name)
    def test_no_credential_is_written_in_the_file(self, path: Path) -> None:
        """Every credential must arrive through `secrets.*`.

        A token pasted into a workflow is committed, mirrored to every fork and
        preserved in the reflog after it is "removed".
        """
        suspicious = re.compile(
            r"(ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|"
            r"sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16})"
        )
        found = suspicious.findall(path.read_text(encoding="utf-8"))
        assert not found, f"{path.name} appears to contain a literal credential"

    @pytest.mark.parametrize("path", workflows(), ids=lambda p: p.name)
    def test_secrets_are_only_referenced_through_the_secrets_context(
        self, path: Path
    ) -> None:
        """Anything credential-shaped must come from `secrets.*`.

        Two exemptions, both narrow and both stated rather than assumed:

        * `permissions:` uses `token`-shaped keys (`id-token: write`) whose
          values are access levels, not credentials;
        * a service container's password is a literal on purpose. The Postgres
          in the integration job exists for the length of one job, is reachable
          only from that job, and holds nothing. Parameterising it would imply
          there is something to protect.

        Anything else is a finding.
        """
        permission_levels = {"read", "write", "none"}
        ephemeral_service_credentials = {"temporal"}

        text = path.read_text(encoding="utf-8")
        for match in re.finditer(r"(?i)(password|token|api[_-]?key)\s*:\s*(\S+)", text):
            key, value = match.group(1), match.group(2)
            if value in permission_levels or value in ephemeral_service_credentials:
                continue
            assert value.startswith("${{"), (
                f"{path.name}: {key} is set to a literal, not a "
                f"secrets reference: {value!r}"
            )

    def test_the_service_container_credential_really_is_throwaway(
        self, ci: dict[str, Any]
    ) -> None:
        """Guard the exemption above.

        The literal is only acceptable because the database is ephemeral and
        unexposed. If that Postgres ever gained a volume or a published port,
        the exemption would stop being true while the test kept passing.
        """
        service = ci["jobs"]["integration"]["services"]["postgresql"]
        assert "volumes" not in service, "the CI database is no longer ephemeral"
        assert "ports" not in service, "the CI database is now reachable from outside"

    def test_ci_needs_no_credentials_at_all(self, ci: dict[str, Any]) -> None:
        """The point of the scripted provider.

        CI runs the real agent loops against recorded responses, so it needs no
        API key — which means a pull request from a fork can run the full suite,
        and a compromised runner has nothing to steal.
        """
        text = (WORKFLOW_DIR / "ci.yml").read_text(encoding="utf-8")
        assert "secrets." not in text, "CI should not need any secret"
        assert ci["env"]["ETLM_LLM_PROVIDER"] == "scripted"


class TestPermissions:
    @pytest.mark.parametrize("path", workflows(), ids=lambda p: p.name)
    def test_a_default_permission_block_is_declared(self, path: Path) -> None:
        """Without one, a job inherits whatever the repository default is —
        historically write-all, which is how a compromised step rewrites a
        branch."""
        assert "permissions" in load(path), f"{path.name} declares no permissions"

    def test_ci_is_read_only_throughout(self, ci: dict[str, Any]) -> None:
        assert ci["permissions"] == {"contents": "read"}
        for name, job in ci["jobs"].items():
            granted = job.get("permissions", {})
            assert "write" not in str(granted), f"ci.yml:{name} asks for write access"

    def test_only_the_publish_job_may_write_packages(
        self, release: dict[str, Any]
    ) -> None:
        assert release["permissions"] == {"contents": "read"}
        publish = release["jobs"]["publish"]["permissions"]
        assert publish["packages"] == "write"
        assert publish["contents"] == "read", "publishing must not write to the repo"


class TestEveryTierActuallyRuns:
    """The failure this guards against is a test tier silently not running.

    Renaming a marker, or splitting a suite into a file the selector no longer
    matches, turns a job green by making it empty. These assert the selectors
    used in CI still correspond to the markers the suite declares.
    """

    def test_the_markers_ci_selects_are_the_markers_pytest_knows(self) -> None:
        config = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        for marker in ("spark", "integration"):
            assert f'"{marker}:' in config, f"marker {marker!r} is not declared"

    def test_ci_runs_the_fast_tier_the_slow_tier_and_the_durable_tier(
        self, ci: dict[str, Any]
    ) -> None:
        commands = " ".join(
            str(step.get("run", "")) for step in steps_of(ci)
        )
        assert '-m "not spark and not integration"' in commands
        assert "-m spark" in commands
        assert "-m integration" in commands

    def test_no_marker_falls_outside_every_ci_selector(self) -> None:
        """The subtle version of an empty job.

        The three selectors are `not (spark or integration)`, `spark`, and
        `integration`, which cover the suite completely *provided* those are the
        only two markers in play. Add a third marker — say `slow` — and any test
        carrying it alongside `spark` still runs, but a test carrying it alone
        is excluded by the fast selector and matched by neither slow one. It
        stops running, and nothing goes red.

        So the property worth asserting is not about the selectors, which are
        checked above, but about the marker vocabulary they assume.
        """
        config = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        declared = set(re.findall(r'^\s+"(\w+):', config, flags=re.M))
        assert declared == {"spark", "integration"}, (
            f"marker(s) {declared - {'spark', 'integration'}} are declared but no CI "
            "job selects them, so tests carrying only those would never run"
        )

    def test_the_spark_job_installs_a_jvm_and_the_spark_extra(
        self, ci: dict[str, Any]
    ) -> None:
        job = ci["jobs"]["spark"]
        uses = [step.get("uses", "") for step in job["steps"]]
        runs = " ".join(str(step.get("run", "")) for step in job["steps"])
        assert any("setup-java" in u for u in uses), "the Spark job has no JVM"
        assert "[dev,spark]" in runs, "the Spark job does not install pyspark"

    def test_the_spark_job_generates_its_input_data(self, ci: dict[str, Any]) -> None:
        """A fresh checkout has no CSVs; without this the whole tier skips."""
        runs = " ".join(str(s.get("run", "")) for s in ci["jobs"]["spark"]["steps"])
        assert "generate_data.py" in runs

    def test_the_tested_python_versions_match_requires_python(
        self, ci: dict[str, Any]
    ) -> None:
        """Claiming support for a version nothing runs on is how a floor breaks."""
        config = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        declared = re.search(r'requires-python\s*=\s*">=(\d+\.\d+),<(\d+\.\d+)"', config)
        assert declared is not None
        floor, ceiling = declared.groups()
        tested = ci["jobs"]["test"]["strategy"]["matrix"]["python-version"]
        assert floor in tested, f"requires-python allows {floor} but CI never tests it"
        assert ceiling not in tested, f"CI tests {ceiling}, which requires-python excludes"


class TestEveryCiStepHasALocalEquivalent:
    """A red build nobody can reproduce locally is a red build nobody fixes."""

    def test_the_makefile_offers_every_check_ci_runs(self) -> None:
        makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
        for target in ("lint", "typecheck", "test-fast", "test", "test-integration", "audit"):
            assert f"\n{target}:" in makefile, f"no `make {target}` to reproduce CI"

    def test_the_lint_and_type_commands_are_the_same_ones(
        self, ci: dict[str, Any]
    ) -> None:
        makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
        runs = " ".join(str(s.get("run", "")) for s in ci["jobs"]["static"]["steps"])
        assert "ruff check ." in runs and "ruff check ." in makefile
        assert "mypy src" in runs and "mypy src" in makefile

    def test_make_test_fast_selects_what_cis_fast_job_selects(
        self, ci: dict[str, Any]
    ) -> None:
        """Same tier, same marker expression.

        These drifted once: CI excluded `integration` and the Makefile did not,
        so `make test-fast` silently pulled in tests needing a Temporal server
        and a JVM. On a machine with neither they skipped and nobody noticed;
        on a machine with both it ran for nine minutes and failed. "Reproduce
        it locally" is worth nothing if the local command is a different
        command.
        """
        makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
        target = re.search(r"\ntest-fast:.*?\n\t(.+)", makefile)
        assert target is not None, "no test-fast recipe found"

        selector = re.search(r'-m "([^"]+)"', target.group(1))
        assert selector is not None, f"test-fast runs no marker selector: {target.group(1)}"

        ci_runs = " ".join(str(s.get("run", "")) for s in steps_of(ci))
        assert f'-m "{selector.group(1)}"' in ci_runs, (
            f"make test-fast selects {selector.group(1)!r}, which no CI step uses"
        )


#: The default JDK in each Debian release's `main`. The worker image installs
#: from the pinned base image's own archive, so this is what is actually
#: obtainable -- not what the development machine happens to run.
DEBIAN_DEFAULT_JDK = {"bookworm": 17, "trixie": 21}

#: Java releases Spark 4 runs on.
SPARK4_JAVA = frozenset({17, 21})


class TestDockerfileJava:
    """The worker image is the only place a JVM is installed, and getting it
    wrong fails at `docker build` -- which no test here can run, and which CI
    has never executed. So these assert the three things that made it wrong:
    a version the base image cannot supply, a JAVA_HOME pointing somewhere the
    package did not install, and a version Spark will not accept.

    The original said `bookworm ships 21`. Bookworm ships 17; 21 arrives with
    trixie. The comment was confident and false, and nothing contradicted it.
    """

    @staticmethod
    def jdk_package(dockerfile: str) -> int:
        match = re.search(r"openjdk-(\d+)-jre-headless", dockerfile)
        assert match is not None, "the worker image installs no JRE"
        return int(match.group(1))

    @staticmethod
    def java_home(dockerfile: str) -> int:
        match = re.search(r"JAVA_HOME=/usr/lib/jvm/java-(\d+)-openjdk", dockerfile)
        assert match is not None, "no JAVA_HOME is declared"
        return int(match.group(1))

    def test_the_declared_java_home_matches_the_installed_jdk(self, dockerfile: str) -> None:
        installed = self.jdk_package(dockerfile)
        declared = self.java_home(dockerfile)
        assert installed == declared, (
            f"installs openjdk-{installed} but points JAVA_HOME at java-{declared}"
        )

    def test_the_jdk_is_one_the_pinned_base_image_can_actually_install(
        self, dockerfile: str
    ) -> None:
        release = re.search(r"FROM python:[\d.]+-slim-(\w+)", dockerfile)
        assert release is not None, "the base image names no Debian release"
        suite = release.group(1)
        assert suite in DEBIAN_DEFAULT_JDK, (
            f"unknown Debian suite {suite!r}; add its default JDK to DEBIAN_DEFAULT_JDK"
        )
        assert self.jdk_package(dockerfile) == DEBIAN_DEFAULT_JDK[suite], (
            f"{suite} ships openjdk-{DEBIAN_DEFAULT_JDK[suite]}, but the image asks for "
            f"openjdk-{self.jdk_package(dockerfile)} — apt will report no installation candidate"
        )

    def test_the_jdk_is_one_spark_4_runs_on(self, dockerfile: str) -> None:
        installed = self.jdk_package(dockerfile)
        assert installed in SPARK4_JAVA, (
            f"Spark 4 needs one of {sorted(SPARK4_JAVA)}, not {installed}"
        )


class TestDockerRunInvocations:
    """Both images set `ENTRYPOINT ["etl-migrator"]`, so a `docker run` that
    names any other binary must pass `--entrypoint` or the binary arrives as
    *arguments to the CLI*.

    `docker run --rm etl-migrator:worker java -version` did exactly that and
    died with `No such command 'java'` -- three lines above a uid check in the
    same recipe that got it right. The smoke target's whole job is to prove the
    images start, so a smoke target that cannot invoke them is worth a test.
    """

    @staticmethod
    def cli_commands() -> set[str]:
        """Derived from the app, so adding a command cannot stale this."""
        from etl_migrator.cli import app

        return {
            (c.name or c.callback.__name__.removesuffix("_cmd").replace("_", "-"))
            for c in app.registered_commands
            if c.callback is not None or c.name is not None
        }

    def test_every_docker_run_either_overrides_the_entrypoint_or_calls_the_cli(
        self,
    ) -> None:
        makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
        commands = self.cli_commands()

        for line in makefile.splitlines():
            if "docker run" not in line:
                continue
            if "--entrypoint" in line:
                continue  # a different binary, declared as such
            tokens = line.split()
            image = next(
                (i for i, t in enumerate(tokens) if "etl-migrator:" in t or t == "$$image"),
                None,
            )
            assert image is not None, f"cannot find the image in: {line.strip()}"
            args = tokens[image + 1 :]
            if not args:
                continue  # runs the image's own CMD, which is fine
            first = args[0]
            assert first.startswith("-") or first in commands, (
                f"`{first}` is not an etl-migrator option or command, so it will be passed "
                f"to the ENTRYPOINT rather than executed. Use --entrypoint.\n  {line.strip()}"
            )


class TestDockerfile:
    def test_it_exists_and_declares_both_targets(self, dockerfile: str) -> None:
        assert "AS cli" in dockerfile
        assert "AS worker" in dockerfile

    @staticmethod
    def final_user_per_stage(dockerfile: str) -> dict[str, list[str]]:
        """Every `USER` directive, in order, per stage."""
        stages: dict[str, list[str]] = {}
        current = ""
        for line in dockerfile.splitlines():
            stripped = line.strip()
            if stripped.upper().startswith("FROM "):
                current = stripped.split(" AS ")[-1] if " AS " in stripped else stripped
                stages[current] = []
            elif stripped.upper().startswith("USER ") and current:
                stages[current].append(stripped.split()[1])
        return stages

    def effective_user(self, dockerfile: str, stage: str) -> str | None:
        """A stage inherits its base's user, so walk the chain. Both runtime
        stages derive from `base`, which is where `etlm` is set."""
        stages = self.final_user_per_stage(dockerfile)
        chain = stages.get("base", []) + stages.get(stage, [])
        return chain[-1] if chain else None

    @pytest.mark.parametrize("stage", ["cli", "worker"])
    def test_the_final_user_is_never_root(self, dockerfile: str, stage: str) -> None:
        """`USER root` mid-file is fine and necessary — apt needs it. What must
        not happen is a stage *ending* as root, which is silent until something
        untrusted runs."""
        user = self.effective_user(dockerfile, stage)
        assert user is not None, f"stage {stage} never sets a user, so it inherits root"
        assert user != "root", f"stage {stage} ends as root"

    def test_the_root_detector_is_not_vacuous(self, dockerfile: str) -> None:
        """Guard the guard.

        The check above walks a stage chain and could pass by simply failing to
        find anything. This feeds it a Dockerfile that ends the worker stage as
        root and asserts it notices.
        """
        assert self.effective_user(dockerfile + "\nUSER root\n", "worker") == "root"

    def test_a_non_root_uid_is_created_explicitly(self, dockerfile: str) -> None:
        """A named uid is what a Kubernetes `runAsUser` can point at."""
        assert "useradd" in dockerfile
        assert "10001" in dockerfile

    def test_build_tooling_does_not_reach_the_runtime_images(
        self, dockerfile: str
    ) -> None:
        """`build-essential` belongs to the builder stage only. A compiler in an
        image that executes untrusted generated code is a gift."""
        builder, _, rest = dockerfile.partition("AS base")
        assert "build-essential" in builder
        assert "build-essential" not in rest

    def test_the_worker_gets_a_jre_and_the_cli_does_not(self, dockerfile: str) -> None:
        """Deliberately version-agnostic. This assertion used to read
        `"openjdk-21-jre" in worker`, which pinned the *wrong* version in place:
        bookworm has no openjdk-21, so the image could not build, and the test
        passed anyway because it only checked the Dockerfile agreed with itself.
        Which version is correct is asserted in TestDockerfileJava, against the
        base image rather than against a literal.
        """
        _, _, worker = dockerfile.partition("AS worker")
        cli_section = dockerfile.partition("AS cli")[2].partition("AS worker")[0]
        assert re.search(r"openjdk-\d+-jre", worker), "the worker image installs no JRE"
        assert "openjdk" not in cli_section

    def test_both_images_declare_a_healthcheck(self, dockerfile: str) -> None:
        assert dockerfile.count("HEALTHCHECK") == 2

    def test_dockerignore_excludes_the_env_file(self) -> None:
        """`.env` in the build context is one `COPY . .` from a secret baked
        into a published layer."""
        ignored = (REPO_ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
        entries = {line.strip() for line in ignored}
        assert ".env" in entries
        assert ".git" in entries
        assert ".venv" in entries

    def test_dockerignore_keeps_the_fixtures(self) -> None:
        """The scripted provider is what makes the image runnable with no API
        key, so the recorded fixtures are runtime data, not test data."""
        ignored = (REPO_ROOT / ".dockerignore").read_text(encoding="utf-8")
        assert "fixtures" not in ignored
        assert "COPY --chown=root:root fixtures/" in DOCKERFILE.read_text(encoding="utf-8")


class TestComposeStack:
    """The local stack, checked for the things that are silent when wrong."""

    @pytest.fixture(scope="class")
    @classmethod
    def compose(cls) -> dict[str, Any]:
        return load(REPO_ROOT / "docker-compose.yml")

    def test_the_worker_is_behind_a_profile(self, compose: dict[str, Any]) -> None:
        """`docker compose up` must stay fast and keep using the working tree.

        A worker that starts by default would silently serve the task queue from
        a stale image while you edit source and wonder why nothing changes.
        """
        assert compose["services"]["worker"]["profiles"] == ["worker"]

    def test_the_worker_holds_no_secret(self, compose: dict[str, Any]) -> None:
        """Every credential is passed through from the environment, defaulting
        to empty. A compose file with a key in it is a key in git history."""
        for key, value in compose["services"]["worker"]["environment"].items():
            if "KEY" in key or "TOKEN" in key:
                assert value.startswith("${"), f"{key} is a literal in compose"

    def test_the_worker_runs_with_a_read_only_root_filesystem(
        self, compose: dict[str, Any]
    ) -> None:
        """The counterpart of the sandbox: generated code runs in this
        container, so the blast radius of an escape should be a tmpfs and one
        volume rather than the whole filesystem."""
        worker = compose["services"]["worker"]
        assert worker["read_only"] is True
        assert worker["cap_drop"] == ["ALL"]
        assert "no-new-privileges:true" in worker["security_opt"]

    def test_every_path_the_image_writes_to_is_mounted_writable(
        self, compose: dict[str, Any], dockerfile: str
    ) -> None:
        """The failure a read-only root filesystem causes is a crash minutes in.

        The Dockerfile declares the workspace as the one writable path and puts
        Spark's scratch directory inside it; if compose ever stopped mounting
        that, the container would start cleanly and fail on the first Spark run.
        """
        worker = compose["services"]["worker"]
        assert "VOLUME [\"/var/lib/etl-migrator/workspace\"]" in dockerfile
        mounted = [m.split(":")[1] for m in worker["volumes"] if ":" in m]
        assert "/var/lib/etl-migrator/workspace" in mounted
        assert "/tmp" in worker["tmpfs"], "the JVM writes hsperfdata to /tmp"
        assert "SPARK_LOCAL_DIRS=/var/lib/etl-migrator/workspace" in dockerfile

    def test_the_examples_are_mounted_read_only(self, compose: dict[str, Any]) -> None:
        """The worker reads legacy sources and input data. Writing to either
        would corrupt the reference the validation is measured against."""
        volumes = compose["services"]["worker"]["volumes"]
        examples = next(v for v in volumes if v.startswith("./examples"))
        assert examples.endswith(":ro")

    def test_the_worker_points_at_the_compose_temporal(
        self, compose: dict[str, Any]
    ) -> None:
        worker = compose["services"]["worker"]
        assert worker["environment"]["ETLM_TEMPORAL_HOST"] == "temporal:7233"
        assert worker["depends_on"]["temporal"]["condition"] == "service_healthy"
