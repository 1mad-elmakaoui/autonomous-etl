"""Policy checks on the Kubernetes manifests.

No cluster is available where this suite runs, so this does what phases 2 and 7
did: check statically what a cluster would otherwise be the only judge of. Two
things make that more than box-ticking here.

First, the manifests encode a *security argument*, not just configuration. The
whole reason there are two worker deployments is that one of them executes
untrusted generated code and the other holds an API key, and keeping those apart
is what lets the executing pod have no internet egress at all. That argument is
only true while every one of its premises holds — the roles stay split, the
execution worker keeps no credentials, the default-deny policy stays default,
the token stays unmounted. Each premise is one edit away from being quietly
false, and none of them fails loudly. So each is a test.

Second, the schema is checkable for real: `kubeconform` validates against the
published Kubernetes JSON schemas, which catches the class of mistake that YAML
parsing cannot — a misspelled field, a value of the wrong type, an apiVersion
that does not exist. Those tests skip cleanly when the binary is absent rather
than pretending to have run.

What none of this establishes is that the manifests *apply*, that the policies
are enforced by the cluster's CNI, or that the pods start. `k8s/README.md` says
so, and says which commands answer it.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, ClassVar

import pytest

yaml = pytest.importorskip("yaml", reason="pyyaml is in the dev extra")

REPO_ROOT = Path(__file__).resolve().parents[1]
K8S_DIR = REPO_ROOT / "k8s" / "base"

#: Both worker roles, and the image each should be running. The execution
#: worker needs a JVM; the agent worker deliberately does not have one.
WORKER_ROLES: dict[str, str] = {
    "agent-worker": "cli",
    "execution-worker": "worker",
}


def manifests() -> list[Path]:
    return sorted(p for p in K8S_DIR.glob("*.yaml") if p.name != "kustomization.yaml")


def documents() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in manifests():
        out.extend(d for d in yaml.safe_load_all(path.read_text("utf-8")) if d)
    return out


def by_kind(kind: str) -> list[dict[str, Any]]:
    return [d for d in documents() if d.get("kind") == kind]


def deployment(component: str) -> dict[str, Any]:
    return next(
        d
        for d in by_kind("Deployment")
        if d["metadata"]["labels"]["app.kubernetes.io/component"] == component
    )


def pod_spec(component: str) -> dict[str, Any]:
    return deployment(component)["spec"]["template"]["spec"]  # type: ignore[no-any-return]


def container(component: str) -> dict[str, Any]:
    return pod_spec(component)["containers"][0]  # type: ignore[no-any-return]


class TestManifestsExist:
    def test_there_are_manifests_to_check(self) -> None:
        """Guard the guard: an empty glob would pass everything below."""
        assert manifests(), "no manifests found — the rest of this file proves nothing"

    def test_both_worker_roles_are_deployed(self) -> None:
        components = {
            d["metadata"]["labels"]["app.kubernetes.io/component"] for d in by_kind("Deployment")
        }
        assert components == set(WORKER_ROLES)

    @pytest.mark.parametrize("path", manifests(), ids=lambda p: p.name)
    def test_each_manifest_parses_and_is_namespaced(self, path: Path) -> None:
        for doc in yaml.safe_load_all(path.read_text("utf-8")):
            if not doc or doc["kind"] == "Namespace":
                continue
            assert doc["metadata"].get("namespace") == "etl-migration", (
                f"{path.name}: {doc['kind']} {doc['metadata']['name']} has no namespace, "
                "so it would land wherever kubectl's current context points"
            )


class TestSchema:
    """Real validation against the published Kubernetes schemas."""

    @staticmethod
    def kubeconform() -> str | None:
        return shutil.which("kubeconform")

    #: Custom resources have no published schema, so kubeconform cannot check
    #: them. Naming the ones we expect keeps `-ignore-missing-schemas` from
    #: quietly excusing a typo'd apiVersion on a core resource.
    KNOWN_CUSTOM_RESOURCES: ClassVar[set[str]] = {"ServiceMonitor"}

    def run_kubeconform(self, *extra: str) -> dict[str, Any]:
        binary = self.kubeconform()
        if binary is None:
            pytest.skip(
                "kubeconform not installed; install it and re-run to schema-validate: "
                "https://github.com/yannh/kubeconform"
            )
        result = subprocess.run(
            [binary, "-strict", "-output", "json", *extra, *[str(p) for p in manifests()]],
            capture_output=True,
            text=True,
            timeout=300,
        )
        return json.loads(result.stdout)  # type: ignore[no-any-return]

    def test_every_manifest_validates_strictly(self) -> None:
        report = self.run_kubeconform("-ignore-missing-schemas")
        invalid = [
            r
            for r in report.get("resources", [])
            if r.get("status") not in {"statusValid", "statusSkipped"}
        ]
        assert not invalid, json.dumps(invalid, indent=2)

    def test_only_known_custom_resources_go_unchecked(self) -> None:
        """`-ignore-missing-schemas` is necessary for CRDs and dangerous as a
        habit: it would also excuse `apiVersion: apps/v2`, which does not exist.

        So the skipped set is enumerated rather than trusted.
        """
        report = self.run_kubeconform("-ignore-missing-schemas")
        skipped = {
            r["kind"] for r in report.get("resources", []) if r.get("status") == "statusSkipped"
        }
        assert skipped <= self.KNOWN_CUSTOM_RESOURCES, (
            f"unvalidated resource kind(s) {skipped - self.KNOWN_CUSTOM_RESOURCES}; "
            "either a CRD nobody declared, or a misspelled apiVersion"
        )


class TestTheTwoRolesStaySeparate:
    """The premise the whole network story rests on.

    If these ever merged back into one deployment, the surviving pod would need
    the agent's internet egress *and* would be running untrusted code — and
    every NetworkPolicy in the directory would still be green.
    """

    def test_each_deployment_runs_exactly_one_role(self) -> None:
        for component in WORKER_ROLES:
            args = container(component)["args"]
            assert args[:2] == ["worker", "--role"], f"{component}: unexpected args {args}"
            assert args[2] == component.removesuffix("-worker")

    def test_the_roles_the_manifests_name_are_roles_the_code_has(self) -> None:
        """A typo here produces a pod that exits immediately, but only at runtime."""
        from etl_migrator.temporal.worker import WorkerRole

        valid = {r.value for r in WorkerRole}
        for component in WORKER_ROLES:
            assert container(component)["args"][2] in valid

    def test_the_execution_worker_serves_only_untrusted_code_activities(self) -> None:
        """The partition asserted against the code, not against a comment.

        This is what makes the NetworkPolicy correct: if an LLM-backed activity
        were ever registered on the execution worker, that worker would need
        internet egress, and the policy denying it would turn a security
        property into an outage.
        """
        from etl_migrator.config import Settings
        from etl_migrator.temporal.worker import WorkerRole, activity_names_for

        settings = Settings()
        execution = set(activity_names_for(settings, WorkerRole.EXECUTION))
        agent = set(activity_names_for(settings, WorkerRole.AGENT))

        assert execution == {
            "run_legacy_pipeline",
            "run_spark_pipeline",
            "validate_outputs",
            "run_tests",
            "benchmark_spark",
        }
        assert not (execution & agent), "an activity is served by both roles"

    def test_no_execution_activity_reaches_for_a_model_client(self) -> None:
        """Read from the source, because the property is about what the code does.

        Every LLM-backed activity builds a `StepContext` via `self._context(...)`.
        An execution-tier activity that started doing so would need a provider,
        a key and egress — none of which the execution worker has.
        """
        import ast
        import inspect

        from etl_migrator.activities.migration import ValidationActivities
        from etl_migrator.config import Settings

        source = ast.parse(
            Path(inspect.getfile(ValidationActivities)).read_text("utf-8")
        )
        methods = {
            node.name: node
            for node in ast.walk(source)
            if isinstance(node, ast.AsyncFunctionDef | ast.FunctionDef)
        }

        activities = ValidationActivities(Settings())
        for tier, expected in (
            (activities.execution_activities(), False),
            (activities.reasoning_activities(), True),
        ):
            for func in tier:
                node = methods[func.__name__]
                uses_model = "self._context(" in ast.unparse(node)
                assert uses_model is expected, (
                    f"{func.__name__}: uses a model client = {uses_model}, "
                    f"but it is in the {'reasoning' if expected else 'execution'} tier"
                )

    def test_the_execution_worker_carries_no_credentials(self) -> None:
        """No Secret reference at all, not merely an unused one.

        `sandbox/runner.py` scrubs the environment on the way into the
        subprocess as well; this is the outer of two independent layers, and
        neither is meant to be the only one.
        """
        spec = container("execution-worker")
        for entry in spec.get("env", []):
            assert "secretKeyRef" not in str(entry), (
                f"the execution worker reads {entry.get('name')} from a Secret"
            )
        for source in spec.get("envFrom", []):
            assert "secretRef" not in source, "the execution worker mounts a whole Secret"

    def test_the_execution_worker_needs_no_model_provider(self) -> None:
        env = {e["name"]: e.get("value") for e in container("execution-worker")["env"]}
        assert env.get("ETLM_LLM_PROVIDER") == "scripted"

    def test_the_agent_worker_reads_its_credentials_optionally(self) -> None:
        """A cluster without the Secret must still start.

        An unset GitHub token is an ordinary state — the delivery stage reports
        it and the migration carries on — so a missing Secret should not turn
        into a crash loop.
        """
        env = {e["name"]: e for e in container("agent-worker")["env"]}
        for key in ("ETLM_LLM_API_KEY", "ETLM_GITHUB_TOKEN"):
            ref = env[key]["valueFrom"]["secretKeyRef"]
            assert ref["optional"] is True, f"{key} is required, so a missing Secret crashes"

    def test_the_two_roles_run_the_image_that_suits_them(self) -> None:
        """The agent worker has no JVM; giving it the worker image would work
        and waste several hundred megabytes on every pod."""
        for component, suffix in WORKER_ROLES.items():
            image = container(component)["image"]
            assert image.endswith(f"-{suffix}"), f"{component} runs {image}"


class TestNetworkPolicy:
    """The isolation the subprocess sandbox cannot provide, checked here for real."""

    @staticmethod
    def policy(name: str) -> dict[str, Any]:
        return next(p for p in by_kind("NetworkPolicy") if p["metadata"]["name"] == name)

    def test_the_namespace_denies_everything_by_default(self) -> None:
        """An allowlist is auditable; a blocklist is unbounded. Same argument as
        the static gate's import policy."""
        deny = self.policy("default-deny-all")
        assert deny["spec"]["podSelector"] == {}, "the default policy does not select all pods"
        assert set(deny["spec"]["policyTypes"]) == {"Ingress", "Egress"}
        assert "egress" not in deny["spec"], "the default-deny policy allows some egress"
        assert "ingress" not in deny["spec"]

    def test_the_execution_worker_can_reach_temporal_and_nothing_else(self) -> None:
        """The headline property. Untrusted code has nowhere to send anything.

        Note what is asserted: not merely that there is no 443 rule, but that
        there is no `ipBlock` at all. A CIDR rule is how "just this one API"
        becomes "the whole internet" over a few well-meaning edits.
        """
        spec = self.policy("execution-worker-egress")["spec"]
        assert spec["podSelector"]["matchLabels"] == {
            "app.kubernetes.io/component": "execution-worker"
        }
        assert spec["policyTypes"] == ["Egress"]

        ports = {p["port"] for rule in spec["egress"] for p in rule.get("ports", [])}
        assert ports == {7233}, f"the execution worker may reach ports {ports}"
        assert not any("ipBlock" in dest for rule in spec["egress"] for dest in rule["to"]), (
            "the execution worker has a CIDR egress rule, so it can reach the internet"
        )

    def test_dns_is_allowed_or_nothing_resolves(self) -> None:
        """Omitting this is the classic default-deny mistake: every connection
        fails with an error that looks nothing like a firewall."""
        spec = self.policy("allow-dns")["spec"]
        assert spec["podSelector"] == {}
        ports = {(p["protocol"], p["port"]) for rule in spec["egress"] for p in rule["ports"]}
        assert ports == {("UDP", 53), ("TCP", 53)}

    def test_the_agent_worker_cannot_reach_the_cluster_or_cloud_metadata(self) -> None:
        """Its 443 rule is broad by necessity — a NetworkPolicy cannot select on
        hostnames — so the exclusions are what stop it being a route inwards.

        169.254.169.254 is the one that matters most: on every major cloud that
        address hands out instance credentials to anything that asks.
        """
        spec = self.policy("agent-worker-egress")["spec"]
        internet = next(
            rule for rule in spec["egress"] if any("ipBlock" in d for d in rule["to"])
        )
        block = next(d["ipBlock"] for d in internet["to"] if "ipBlock" in d)
        assert block["cidr"] == "0.0.0.0/0"
        excluded = set(block["except"])
        assert "169.254.0.0/16" in excluded, "cloud metadata is reachable"
        for private in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16"):
            assert private in excluded, f"{private} is reachable from the agent worker"

    def test_every_policy_explains_itself(self) -> None:
        """The two role policies encode a judgement, and a reader six months
        from now needs the judgement, not just the ports."""
        for name in ("execution-worker-egress", "agent-worker-egress"):
            annotations = self.policy(name)["metadata"].get("annotations", {})
            assert annotations.get("etl-migration/rationale")

    def test_the_policies_select_pods_that_exist(self) -> None:
        """A selector typo produces a policy that matches nothing, which fails
        open and looks exactly like a policy that is working."""
        deployed = {
            d["metadata"]["labels"]["app.kubernetes.io/component"] for d in by_kind("Deployment")
        }
        for name in ("execution-worker-egress", "agent-worker-egress"):
            selector = self.policy(name)["spec"]["podSelector"]["matchLabels"]
            component = selector["app.kubernetes.io/component"]
            assert component in deployed, f"{name} selects {component}, which is not deployed"


class TestPodHardening:
    @pytest.mark.parametrize("component", sorted(WORKER_ROLES))
    def test_pods_run_as_a_non_root_user(self, component: str) -> None:
        security = pod_spec(component)["securityContext"]
        assert security["runAsNonRoot"] is True
        # Matches the uid created in the Dockerfile, so a mounted volume has a
        # predictable owner.
        assert security["runAsUser"] == 10001

    @pytest.mark.parametrize("component", sorted(WORKER_ROLES))
    def test_containers_cannot_escalate_or_write_to_root(self, component: str) -> None:
        security = container(component)["securityContext"]
        assert security["allowPrivilegeEscalation"] is False
        assert security["readOnlyRootFilesystem"] is True
        assert security["capabilities"]["drop"] == ["ALL"]

    @pytest.mark.parametrize("component", sorted(WORKER_ROLES))
    def test_no_api_token_is_mounted(self, component: str) -> None:
        """Especially for the executing pod: a projected token is a credential
        sitting in the filesystem of the container running untrusted code."""
        spec = pod_spec(component)
        assert spec["automountServiceAccountToken"] is False
        assert spec["serviceAccountName"] != "default"

    @pytest.mark.parametrize("component", sorted(WORKER_ROLES))
    def test_the_writable_paths_are_mounted(self, component: str) -> None:
        """`readOnlyRootFilesystem` without these is a pod that starts and then
        fails the first time Spark or the JVM writes anything."""
        mounts = {m["mountPath"] for m in container(component)["volumeMounts"]}
        assert "/tmp" in mounts, "the JVM writes hsperfdata to /tmp"
        assert "/var/lib/etl-migrator/workspace" in mounts

    @pytest.mark.parametrize("component", sorted(WORKER_ROLES))
    def test_resources_are_bounded_at_both_ends(self, component: str) -> None:
        """A request without a limit lets one Spark run starve its neighbours;
        a limit without a request lets the scheduler overcommit the node."""
        resources = container(component)["resources"]
        for side in ("requests", "limits"):
            assert set(resources[side]) == {"cpu", "memory"}, f"{component}: {side} incomplete"

    def test_the_executing_pod_gets_the_memory_a_spark_driver_needs(self) -> None:
        """Under-requesting does not make it cheaper. It makes it OOMKill
        halfway through a benchmark and report a failed run."""
        limits = container("execution-worker")["resources"]["limits"]
        assert limits["memory"] == "4Gi"

    @pytest.mark.parametrize("component", sorted(WORKER_ROLES))
    def test_probes_are_defined(self, component: str) -> None:
        spec = container(component)
        assert "startupProbe" in spec
        assert "livenessProbe" in spec

    def test_the_namespace_enforces_the_restricted_pod_security_standard(self) -> None:
        """The backstop. Every securityContext above is a promise a manifest
        makes; this is the one thing that makes the API server hold it.
        """
        labels = by_kind("Namespace")[0]["metadata"]["labels"]
        assert labels["pod-security.kubernetes.io/enforce"] == "restricted"


class TestSecrets:
    def test_the_committed_secret_carries_no_values(self) -> None:
        """The template exists so the reference resolves and the shape is
        documented. A value in it would be a credential in git history."""
        secret = by_kind("Secret")[0]
        assert not secret.get("data")
        assert not secret.get("stringData")

    @pytest.mark.parametrize("path", manifests(), ids=lambda p: p.name)
    def test_no_manifest_contains_a_credential(self, path: Path) -> None:
        import re

        suspicious = re.compile(
            r"(ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|"
            r"sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16})"
        )
        assert not suspicious.findall(path.read_text("utf-8"))

    def test_the_configmap_holds_nothing_secret(self) -> None:
        """A ConfigMap is readable by anything that can read the namespace."""
        data = by_kind("ConfigMap")[0]["data"]
        for key in data:
            assert not any(word in key for word in ("TOKEN", "KEY", "PASSWORD", "SECRET")), (
                f"{key} is in a ConfigMap; it belongs in the Secret"
            )


class TestConfigMatchesTheCode:
    def test_every_configured_key_is_a_real_setting(self) -> None:
        """A typo'd ETLM_ key is silently ignored — `extra="ignore"` on Settings
        — so the pod starts with a default nobody intended."""
        from etl_migrator.config import Settings

        known = {f"ETLM_{name.upper()}" for name in Settings.model_fields}
        for key in by_kind("ConfigMap")[0]["data"]:
            assert key in known, f"{key} is not a Settings field, so it does nothing"

    def test_the_task_queue_split_is_actually_configured(self) -> None:
        """An empty execution queue collapses the split back to one queue —
        correct on a laptop, and the whole point of the deployment here."""
        data = by_kind("ConfigMap")[0]["data"]
        assert data["ETLM_TEMPORAL_EXECUTION_TASK_QUEUE"]
        assert data["ETLM_TEMPORAL_EXECUTION_TASK_QUEUE"] != data["ETLM_TEMPORAL_TASK_QUEUE"]

    def test_the_workspace_path_matches_the_image(self) -> None:
        data = by_kind("ConfigMap")[0]["data"]
        dockerfile = (REPO_ROOT / "Dockerfile").read_text("utf-8")
        assert data["ETLM_WORKSPACE_DIR"] in dockerfile

    def test_the_kind_cluster_disables_the_cni_that_ignores_policies(self) -> None:
        """kind's default CNI does not enforce NetworkPolicy.

        Applying these policies to a stock kind cluster yields a green
        `kubectl get netpol` and no enforcement whatsoever — a worse outcome
        than not applying them, because it looks verified.
        """
        cluster = yaml.safe_load((REPO_ROOT / "k8s" / "kind" / "cluster.yaml").read_text())
        assert cluster["networking"]["disableDefaultCNI"] is True


class TestMetricsWiring:
    """Adding a scrape endpoint interacts with the network policies.

    The default-deny policy blocks ingress as well as egress, so a `/metrics`
    port without a matching ingress rule produces a permanently unscrapeable pod
    and a dashboard of no data — with nothing anywhere reporting a problem.
    """

    @staticmethod
    def service() -> dict[str, Any]:
        return by_kind("Service")[0]

    def test_the_scrape_port_matches_the_configured_one(self) -> None:
        """Three places have to agree: the container port, the Service, and the
        port the process actually binds."""
        from etl_migrator.config import Settings

        configured = Settings().metrics_port
        for component in WORKER_ROLES:
            ports = {p["containerPort"] for p in container(component)["ports"]}
            assert configured in ports, f"{component} does not expose {configured}"
        assert self.service()["spec"]["ports"][0]["port"] == configured

    def test_ingress_is_allowed_for_the_scrape_and_only_the_scrape(self) -> None:
        policy = next(
            p
            for p in by_kind("NetworkPolicy")
            if p["metadata"]["name"] == "allow-metrics-scrape"
        )
        spec = policy["spec"]
        assert spec["policyTypes"] == ["Ingress"]
        ports = {p["port"] for rule in spec["ingress"] for p in rule["ports"]}
        assert ports == {9464}, f"more than the scrape port is reachable: {ports}"

    def test_the_scrape_rule_does_not_widen_egress(self) -> None:
        """Prometheus reaching in is not untrusted code reaching out. If this
        policy ever grew an egress section it would silently undo the isolation
        the executing worker depends on."""
        policy = next(
            p
            for p in by_kind("NetworkPolicy")
            if p["metadata"]["name"] == "allow-metrics-scrape"
        )
        assert "egress" not in policy["spec"]
        assert "Egress" not in policy["spec"]["policyTypes"]

    def test_the_metrics_service_is_headless(self) -> None:
        """A load-balanced Service would round-robin scrapes across replicas and
        stitch several processes into one incoherent series — counters that
        appear to go backwards."""
        assert self.service()["spec"]["clusterIP"] == "None"

    def test_the_service_selects_both_worker_roles(self) -> None:
        """Both are scraped: an execution worker that stops reporting is exactly
        what you want an alert on."""
        selector = self.service()["spec"]["selector"]
        assert selector == {"app.kubernetes.io/name": "etl-migrator"}
        for component in WORKER_ROLES:
            labels = deployment(component)["spec"]["template"]["metadata"]["labels"]
            assert labels["app.kubernetes.io/name"] == "etl-migrator"

    @pytest.mark.parametrize("component", sorted(WORKER_ROLES))
    def test_liveness_does_not_depend_on_temporal(self, component: str) -> None:
        """A probe that failed during a Temporal outage would restart every
        replica simultaneously and turn one outage into two."""
        probe = container(component)["livenessProbe"]
        assert probe["httpGet"]["path"] == "/healthz"
        assert probe["httpGet"]["port"] == "metrics"


class TestKustomizeDoesNotRewriteTheImages:
    """The two workers share a repository and differ only by tag: `-cli` has no
    JVM, `-worker` carries the JRE and PySpark. Only the execution worker runs
    Spark, so only it can use `-worker`.

    kustomize's `images:` transformer matches on image *name*, which for these
    two is the same string. A single entry with `newTag:` therefore rewrote
    both, quietly deploying the executing worker with the image that has no
    Java in it. Nothing in the Deployment files showed it -- they were correct;
    the transformer overrode them, and only the rendered output disagreed.

    These read `kustomization.yaml` directly so they run without kustomize
    installed.
    """

    @staticmethod
    def kustomization() -> dict[str, Any]:
        raw = (K8S_DIR / "kustomization.yaml").read_text("utf-8")
        return yaml.safe_load(raw)  # type: ignore[no-any-return]

    @staticmethod
    def declared_images() -> dict[str, str]:
        """component -> image reference, as the Deployment files declare it."""
        return {
            component: container(component)["image"] for component in WORKER_ROLES
        }

    def test_each_worker_declares_the_image_its_role_needs(self) -> None:
        for component, flavour in WORKER_ROLES.items():
            image = self.declared_images()[component]
            assert image.endswith(f"-{flavour}"), (
                f"{component} should run the {flavour!r} image, not {image!r}"
            )

    def test_no_image_transformer_can_collide_across_the_two_workers(self) -> None:
        entries = self.kustomization().get("images") or []
        if not entries:
            return  # nothing to rewrite, nothing to collide

        repositories = {ref.rsplit(":", 1)[0] for ref in self.declared_images().values()}
        for entry in entries:
            name = entry.get("name")
            rewrites_tag = "newTag" in entry or "digest" in entry
            assert not (rewrites_tag and name in repositories), (
                f"images: entry {name!r} sets a tag/digest, but both workers share that "
                "repository and differ only by tag -- it would rewrite both and give the "
                "execution worker an image with no JVM. Patch each Deployment by name in "
                "an overlay instead."
            )
