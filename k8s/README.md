# Kubernetes

What lives here is the part nobody else can write for this system: **two worker
deployments and the policies that keep them apart.**

Temporal, PostgreSQL and MinIO are deliberately absent. They have maintained
upstream Helm charts, and a hand-rolled copy in raw YAML would be a worse
version that silently drifts from the original. Install commands are below.

## The argument this directory encodes

The subprocess sandbox for generated code states plainly what it does *not*
give you:

> **Not guaranteed: network egress.** Blocking it needs a network namespace,
> which needs privileges the worker does not have.

This is where that closes, at the layer that actually owns it. The mechanism is
not just a NetworkPolicy, a policy alone would not have helped. It is the
**worker split**:

| | `agent-worker` | `execution-worker` |
|---|---|---|
| Runs generated code | never | always |
| Model provider + GitHub credentials | yes | **none** |
| Egress | Temporal, and 443 to the public internet | **Temporal only** |
| Service account token | not mounted | not mounted |
| Image | `…-cli` (no JVM) | `…-worker` (JRE 17) |

Co-located, one pod would need internet egress for the model provider *and*
would be running untrusted code, and the subprocess would inherit exactly that
egress. Separated, the executing pod has nowhere to send anything it finds and
nothing to download.

The split is real in the code, not just in the manifests: `WorkerRole.EXECUTION`
registers five activities and no others, and a test asserts that none of them
reaches for a model client. If that ever changed, the executing worker would
need egress it does not have, and the failure would be an outage rather than a
silent hole.

## Local cluster

```bash
kind create cluster --config k8s/kind/cluster.yaml

# kind's default CNI does NOT enforce NetworkPolicy, so the config above
# disables it. Without this step the policies apply cleanly and do nothing,
# which is worse than not applying them, because it looks verified.
kubectl apply -f https://raw.githubusercontent.com/projectcalico/calico/v3.30.0/manifests/calico.yaml
kubectl -n kube-system rollout status daemonset/calico-node --timeout=300s

# The images. The manifests name `ghcr.io/1mad-elmakaoui/autonomous-etl`,
# which is where the release workflow publishes, so on a cluster with no
# access to that registry (or before anything has ever been published) the pods
# sit in ImagePullBackOff. Build locally, tag as the manifests expect, and hand
# them to the cluster directly. `imagePullPolicy: IfNotPresent` is what makes
# this work; with `Always` the kubelet would go to the registry anyway.
make docker-build
for target in cli worker; do
  docker tag "etl-migrator:$target" "ghcr.io/1mad-elmakaoui/autonomous-etl:latest-$target"
  kind load docker-image --name etl-migration \
    "ghcr.io/1mad-elmakaoui/autonomous-etl:latest-$target"
done

# Dependencies, from their own charts.
helm repo add temporal https://charts.temporal.io
helm install temporal temporal/temporal -n etl-migration --create-namespace \
  --set server.replicaCount=1 --set cassandra.enabled=false \
  --set postgresql.enabled=true --set prometheus.enabled=false \
  --set grafana.enabled=false --set elasticsearch.enabled=false

# This directory.
kubectl apply -k k8s/base
```

Then create the Secret out of band, it is committed empty on purpose:

```bash
kubectl -n etl-migration create secret generic etl-migration-secrets \
  --from-literal=ETLM_LLM_API_KEY="$ANTHROPIC_API_KEY" \
  --from-literal=ETLM_GITHUB_TOKEN="$GITHUB_TOKEN" \
  --from-literal=ETLM_GITHUB_REPOSITORY=owner/repo
```

Every key is referenced with `optional: true`, so a cluster without the Secret
still starts and reports GitHub as unconfigured rather than crash-looping.

## Metrics

Both workers serve `/metrics` on 9464 and are scraped through a **headless**
Service, one series per pod. A load-balanced Service would round-robin scrapes
across replicas and stitch several processes into one incoherent series, with
counters that appear to go backwards.

`70-metrics.yaml` also carries a `ServiceMonitor`, which needs the Prometheus
Operator's CRD. On a cluster without it, drop that file from
`kustomization.yaml` and point Prometheus at the Service yourself.

Adding a scrape endpoint interacts with the policies above: the default-deny
rule blocks ingress as well as egress, so there is now exactly one ingress rule
,  port 9464, from the `monitoring` namespace only. It has no egress section, so
Prometheus reaching *in* does not widen what the executing worker can reach
*out* to.

The liveness probe hits `/healthz`, which deliberately does not check Temporal.
A probe that failed during a Temporal outage would restart every replica at once
and turn one outage into two.

## Verifying the isolation actually holds

The tests in `tests/test_kubernetes.py` check the manifests say the right thing.
They cannot check that the cluster *does* the right thing, only a cluster can.
This is the command that answers it:

```bash
# From the executing worker, the internet must be unreachable...
kubectl -n etl-migration exec deploy/etl-execution-worker -- \
  python -c "import socket; socket.create_connection(('1.1.1.1', 443), timeout=5)"
# expected: the connection times out

# ...while Temporal must be reachable, or the worker cannot poll for work.
kubectl -n etl-migration exec deploy/etl-execution-worker -- \
  python -c "import socket; socket.create_connection(('temporal-frontend', 7233), timeout=5)"
# expected: no error
```

If the first command succeeds, the CNI is not enforcing policy and the isolation
is theatre. Check that Calico is running before trusting anything here.

## What is verified, and what is not

`make k8s-validate` runs strict schema validation (kubeconform, against the
published Kubernetes schemas) plus the policy tests: non-root, no privilege
escalation, read-only root filesystem, capabilities dropped, default-deny
egress, no credentials on the executing worker, no service account token, every
`ETLM_` key a real setting, resource limits at both ends.

Those tests read the manifests. They cannot tell you what a cluster does with
them, and for a long time nothing had.

### Verified on a cluster, 2026-08-19

kind v1.36.1, three nodes, Calico v3.30.0, images loaded locally:

- **The manifests apply.** 15 of 16 resources accepted by a live API server.
  Both Deployments were admitted under the restricted Pod Security Standard, RBAC
  coherent, no field rejected. The 16th is the `ServiceMonitor`, which needs the
  Prometheus Operator's CRD and is expected to fail without it.
- **Both workers reach `Running`** and stay up without Temporal, restarting on
  the failed connection while `/healthz` keeps reporting them live, which is
  the intended behaviour, not a tolerated one.
- **The egress split is enforced, not just declared:**

  | Pod | `socket.create_connection(('1.1.1.1', 443), timeout=5)` |
  |---|---|
  | `etl-execution-worker` | `TimeoutError: timed out` |
  | `etl-agent-worker` | `reachable` |

A probe pod is a valid substitute for `exec` when the workers are down, but it
must carry **both** labels the policy selects on:

```yaml
labels:
  app.kubernetes.io/component: execution-worker
  app.kubernetes.io/part-of: autonomous-etl      # kustomize adds this to the
                                                 # NetworkPolicy selectors too
```

Omit `part-of` and the pod matches neither egress policy, falls through to
default-deny, and times out. Both probes then fail and the run looks like a
pass on the half you were watching. That happened; the control is what caught
it.

The second row is the part that matters. A timeout from the executing worker is
equally consistent with the policy working and with the cluster having no
outbound connectivity at all, and those look identical from that pod. Only a
pod that *can* get out separates them. Run both or neither.

This was a single manual run, not a continuous check: nothing in CI re-verifies
it, and it holds only on a cluster whose CNI enforces NetworkPolicy. On stock
kind it would not, which is why `k8s/kind/cluster.yaml` disables the default CNI
rather than leaving that to chance.
