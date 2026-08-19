# Architecture

The engineering document. The README explains what the system does; this
explains why it is shaped the way it is, what the interfaces are, and what will
break if they are ignored.

---

## 1. The central inversion

Most "AI migrates your code" systems are built as: *prompt → code → hope*. The
LLM's output is the product.

This system is built as: *hypothesis → proof → repair*. The LLM's output is a
**candidate**, and the product is the evidence that it is equivalent.

Everything else follows mechanically:

- If the generated code is a hypothesis, something deterministic must test it →
  **the differ, not the model, decides correctness**.
- If a hypothesis can be wrong, failure is a normal state → **the repair loop**.
- If repair is bounded and stateful across hours → **Temporal**.
- If the agent's own claims are not evidence → **every agent verdict is re-verified
  outside the agent**.

Three concrete places where this is enforced in code today:

| Enforcement | Location |
|---|---|
| The static gate is re-run *after* the agent returns; the agent's "it passed" is discarded | `pipeline/steps.py::verify_gate` |
| The approval requirement is recomputed from risk; an agent cannot set `requires_human_approval=False` on a HIGH-risk plan | `domain/lifecycle.py::enforce_approval_policy` |
| Discovery is given tools, not source text, so it cannot describe a column that does not exist | `DiscoveryAgent` |

---

## 2. Layer boundaries

```text
Orchestration (Temporal)     durable state, retries, approval, resumption
        │  no heavy I/O, no non-determinism
        ▼
Activities                   every side effect, every LLM call
        │
        ▼
Reasoning (AutoGen agents)   interpretation, decisions, strategies
        │  may not assert what it did not observe
        ▼
Perception (tools)           AST, profiling, gates, execution, diffing
                             no LLM, ever
```

The rule that keeps this honest: **an agent may only state a fact it obtained
from a tool.** Prompts enforce it; the recorded tool trace
(`artifacts/<id>/agent_trace.json`) makes violations auditable after the fact.

---

## 3. Interfaces between components

These are the contracts. Later phases add implementations behind them; they do
not change them. Activity and workflow wire formats live in
`domain/messages.py`, separate from the artifacts they carry: changing a wire
shape on a running workflow is a versioning event, changing a field on
`MigrationSpec` is not.

### 3.1 Data contracts (`etl_migrator.domain`)

```text
MigrationSpec     ← DiscoveryAgent      what the legacy pipeline does
MigrationPlan     ← PlannerAgent        how it maps to Spark + what could differ
GeneratedCode     ← SparkEngineerAgent  the candidate module
CodeGenResult                           code + independently verified gate report
MigrationRecord                         durable state of one migration
ValidationReport  ← the differ         computed verdict; no agent can build one
ValidationDiagnosis ← ValidationAgent  explanation of a failure, never a verdict
GeneratedTests    ← TestingAgent       a pytest module pinning the differences
```

`domain` imports nothing but pydantic and the stdlib, and has no import-time
side effects. That is a hard rule, and it is load-bearing: the package is on
Temporal's sandbox passthrough list, which is only safe under exactly that
constraint. Two tests assert it by parsing every module in the package rather
than trusting the comment.

Every model derives from `StrictModel` (`extra="forbid"`). When a model invents
a field, that must be a loud `ValidationError` inside a retryable activity, never
a silently dropped instruction.

### 3.2 The pipeline execution contract

Legacy and generated pipelines are interchangeable to the runner:

```python
# legacy
def run(input_dir: str, output_dir: str) -> None: ...

# generated Spark: the session is injected, never constructed
def run(spark: SparkSession, input_dir: str, output_dir: str) -> None: ...
```

The static gate enforces the Spark side exactly (`GATE004`). Injecting the
session rather than letting the module build one is what allows the Optimizer
to vary `spark.sql.shuffle.partitions`, AQE and broadcast thresholds
without touching the generated code, the same module is benchmarked under
different configurations.

### 3.3 Model provider contract

The abstraction is `autogen_core.models.ChatCompletionClient`. Agents receive a
`ModelClientFactory`, never a concrete client:

```python
class ModelClientFactory(Protocol):
    def client_for(self, agent: str) -> ChatCompletionClient: ...
```

Three implementations: Anthropic, OpenAI, and `ScriptedChatCompletionClient`.
The scripted one is a **real** implementation of the interface, not a stub, so
tests drive the genuine agent loop, tool dispatch and structured-output parsing.
Mocking the *agent* instead of the *model* would test nothing.

Wrapping `ChatCompletionClient` in a bespoke interface was considered and
rejected: it would have to be kept in sync with AutoGen's tool-calling and
structured-output plumbing for no benefit.

### 3.4 Agent contract

```python
class StructuredAgent(Generic[T]):
    async def run(self, task: str) -> AgentRun[T]
```

`AgentRun` carries the validated output **and the evidence**: which tools ran,
whether they errored, what they returned, how long it took. Those timings go to
Prometheus, and the history harvester mines them for the pattern catalogue.

---

## 4. Risk register

| # | Risk | Mitigation | Status |
|---|---|---|---|
| R1 | LLM claims success, output silently wrong | Correctness gated only by the executed-output diff; `ValidationReport.status` is computed, and the Validation agent cannot be constructed on a passing report | ✅ implemented + tested |
| R2 | pandas↔Spark semantic drift | Typed `SemanticDifference` per step → each becomes a required check *and* a generated test | ✅ implemented |
| R3 | Generated code is untrusted; executing it is RCE | Three layers: AST allowlist gate, a subprocess sandbox with a scrubbed environment, resource limits and a process-group kill, and a default-deny egress NetworkPolicy on a worker deployed separately *because* it executes untrusted code, so the policy can deny everything but Temporal | ✅ all three |
| R4 | Repair loop oscillates / burns tokens | `RepairLedger`: a repeated `(category, approach)` signature or a matching code fingerprint is refused by set membership, before any LLM call or Spark run | ✅ implemented + tested |
| R5 | Temporal determinism violations | 100% of side effects in Activities; sandboxed runner; passthrough limited to `domain`; AST guard + replay test | ✅ implemented |
| R6 | Optimizer trades correctness for speed | `evaluate_optimization` is a pure function over a `ValidationReport` and a `BenchmarkComparison`; correctness is settled before the candidate is timed, and a FAIL short-circuits the attempt. No agent field can set `applied` | ✅ implemented + tested |
| R7 | Spark JVM startup dominates CI | Session reuse per worker; `local[*]`; LLM fully scripted in CI | ✅ (`test_spark_equivalence` uses a module-scoped session) |
| R8 | Temporal retries re-run non-idempotent activities | Paths derived from `migration_id`, full overwrites; workflow id == migration id so resubmission dedupes | ✅ implemented + tested |
| R9 | Benchmark noise reported as speedup | Eight runs, warm-up discarded, median + p75 + a noise ceiling. A measurement too noisy to read is refused rather than rounded up, and the gain must hold at the p75 so one lucky run cannot carry it | ✅ implemented + tested |
| R10 | Agent talks its way past the approval gate | Approval recomputed from risk by the orchestrator; agent's field is advisory only | ✅ implemented + tested |
| R11 | PR body overstates what the migration achieved | The evidence half is rendered from the record with no LLM involved; `PullRequestNarrative` has no verdict field to fill in; numeric claims in the prose are audited against measurements and an unfixable violation means no PR | ✅ implemented + tested |
| R12 | A reviewer approves an unproven migration | `decide_delivery` refuses a PR for an unvalidated migration outright, and opens failures as labelled drafts carrying the failure in the body | ✅ implemented + tested |

---

## 5. Design decisions

### D1: `src/etl_migrator/` instead of flat top-level packages

The brief proposed top-level `agents/`, `workflows/`, `legacy/`, `api/`,
`domain/`. Those names are generic enough to collide on `sys.path`, which breaks
editable installs and makes Temporal's sandbox module-passthrough rules
ambiguous (`domain` in particular). One import root removes both problems. The
internal layout is otherwise exactly as specified.

### D2: Discovery gets tools, not source text

An LLM handed 200 lines of pandas will describe a column that is not there. An
LLM that must call `inspect_legacy_source()` gets line-numbered AST facts it
cannot invent, and `profile_input_data()` gets measured dtypes and null counts.
The cost is more round trips; the benefit is that `MigrationSpec.profiled`
distinguishes *observed* from *assumed*, which the planner then relies on when
deciding what to broadcast.

### D3: The planner is separate from the code generator

When a migration fails validation, the repair agent must determine whether the
*plan* was wrong or the *implementation* was. That is only answerable if the
plan was written down first, in machine-readable form, before any code existed.
The separation also means the acceptance criteria (`ValidationPlan`) are fixed
before the diff is visible, so they cannot be quietly relaxed to make a failure
go away.

### D4: Semantic differences carry their own validation check

`SemanticDifference.validation_check` names the differ check that proves the
mitigation worked, and `MigrationPlan.effective_required_checks()` folds those
into the required set. This creates the right incentive gradient: a planner that
stays silent about null semantics produces a *weaker* validation suite, and the
difference then shows up as a real diff. Silence is not free.

### D5: The static gate is an allowlist

A blocklist of dangerous imports is unbounded (`os`, `subprocess`, `socket`,
`ctypes`, `importlib`, `pty`, …). An allowlist of `pyspark` plus nine stdlib
modules is small enough to audit in one screen. Same reasoning for module-level
purity: rather than enumerate dangerous statements, permit only imports,
definitions and constants.

### D6: The gate is also an agent tool

`check_spark_code` is the same function the orchestrator runs afterwards. This
gives the Spark Engineer a real act → observe → correct loop inside a single
invocation, and `CodeGenResult.gate_iterations` records how many attempts it
needed, a quality signal in its own right. The orchestrator still re-runs the
gate independently, so a lying agent gains nothing.

### D7: Structured historical knowledge, not RAG

`knowledge/patterns.py` is a typed catalogue keyed by `TransformKind`. A lookup
is a dictionary access with a guaranteed shape, not a similarity search that may
return something adjacent. The harvester appends `ObservedOutcome` records from
completed migrations, at which point the catalogue becomes empirical, strategies
that repeatedly needed repair get demoted, and the planner sees that in the same
tool output. The type is defined now so the storage shape does not change when
the data starts arriving.

Today `success_rate` is `None` for every pattern and the renderer says
"curated guidance", which a test enforces. The system must not imply empirical
backing it does not have.

### D8: Two orchestrators, one implementation

`pipeline/steps.py` holds the work. `pipeline/local.py` runs it sequentially
in-process; `workflows/migration.py` runs the same steps durably through
Temporal. Neither owns an implementation of its own.

The local path is not a toy. It is what you run while developing an agent, it is
what keeps the whole system testable in CI where no Temporal server exists, and
running the two side by side is how you notice if the durable path has quietly
diverged. What legitimately differs between them is exactly what *should*:
durability, retries, and how the approval gate is satisfied, a callback locally,
a signal that survives a reboot in Temporal.

### D9: Workflow decisions live in pure functions

A Temporal workflow may not read a clock, generate a uuid or touch a disk. The
natural consequence is that the workflow body becomes glue. If the state
mutation and branching lived inline in that glue, it would only be testable by
running a workflow, which needs a server.

So it lives in `domain/lifecycle.py`, as pure functions that take `now`
explicitly. The workflow passes `workflow.now()`; the local pipeline passes
`datetime.now(UTC)`. Neither can introduce non-determinism, because there is no
clock in there to reach for. The payoff: every branch the workflow can take is
tested exhaustively in milliseconds without a server, and the workflow file
stays short enough to audit in one sitting.

This is also where the repair loop's bookkeeping lands, which is the
other reason it has its own home now.

### D10: A failing gate is data, not an exception

`run_static_analysis` returns a verdict rather than raising. A failing gate is a
legitimate outcome the workflow must reason about, since it routes to the
repair loop, and raising would hand that decision to a retry policy, which
would dutifully re-run a deterministic check three times and then give up.

The same reasoning sets the non-retryable list: `ConfigurationError` four times
is four timeouts, while `AgentContractError` genuinely often succeeds on a
second sample.

### D11: The verdict is a computed property, not a field

`ValidationReport.status` is a `@computed_field` derived from the measured
checks. There is no setter and no constructor path that takes it. That is a
stronger guarantee than a prompt instruction: an agent cannot emit a
`ValidationReport`, because producing one requires having run the differ.

The `ValidationDiagnosis` type is the mirror image, it exists only to explain a
failure, and a test asserts it has no field named `status`, `passed`, `valid`,
`approved` or `override`. The Validation agent's constructor raises on a passing
report, so there is no code path where a model is asked to opine on a migration
that already succeeded. An agent invited to review a pass will eventually find a
reason to bless something it should not.

### D12: Skipped is not passed

A check that could not run sets `skipped=True`, and `status` maps any skipped
check to ERROR rather than ignoring it. This covers two real cases: join keys
the plan declared that are absent from the actual output, and a check name no
differ implements. Both mean the plan demanded an assurance the system cannot
give, which is materially different from the assurance holding.

### D13: What the sandbox does and does not promise

The subprocess sandbox provides process isolation, an allowlisted environment
(so no secret the worker holds is visible to generated code), resource limits, a
wall-clock kill of the whole process group, a neutral working directory, and a
verdict read from a file rather than parsed from stdout.

It does **not** provide network isolation, and says so. Blocking egress requires
a network namespace and privileges the worker does not have; that control
belongs to a Kubernetes NetworkPolicy. Documenting the gap is worth
more than a comment claiming a guarantee that does not hold, someone would
eventually rely on it.

Two limit profiles exist because one does not fit: `RLIMIT_AS` and
`RLIMIT_NPROC` are correct for pandas and actively wrong for Spark, since the
JVM reserves a large virtual address space and its threads count against
NPROC on Linux. Applying them to a Spark run would not harden anything, it would
just stop Spark from starting.

### D14: The repair budget counts ideas, not executions

`max_repair_attempts` is meaningless on its own. An agent that alternates
between two wrong fixes will spend any budget you give it, and each spin costs a
full Spark execution.

`RepairLedger` makes the budget count *distinct ideas*. A strategy is
`(root cause category, approach slug)`; the slug is constrained to
`^[a-z][a-z0-9_]{2,48}$` precisely so two proposals can be compared
mechanically, "filter the null keys" and "filter out null country values" are
the same idea in prose and identical as `filter_null_group_keys`. Code is
additionally fingerprinted with comments and blank lines stripped, so a proposal
that only reworded its commentary is caught before it costs an execution.

Three consequences worth stating:

* A **rejected proposal still consumes an attempt.** An agent that cannot
  produce a distinct strategy has exhausted its ideas, and re-asking it is the
  bonfire the bound exists to prevent.
* The rejection is **fed into the next prompt**, which is the correction half of
  the loop, a repeat becomes a choice rather than an accident.
* The ledger is shared verbatim between `RepairWorkflow` and the local
  pipeline, so the rule cannot drift between the rehearsal and the real run.

### D15: Repair runs only on a measured difference

The parent skips the repair loop when `ValidationReport.error` is set. An ERROR
means a pipeline never produced an output, a crash, a timeout, a missing file.
There is no difference for a code change to act on. Repairing there would be
guessing, and it would look exactly like repairing something real.

### D16: Exhaustion returns the nearest miss

`RepairOutcome.best_attempt` is the admitted attempt with the fewest measured
differences, ties broken toward the earlier one. When a human inherits an
exhausted loop they need the closest attempt, not the last: the final attempt is
frequently a wilder guess than the second, and handing over the wrong one wastes
the human's first hour.

### D17: Fixtures are selected by request content

Recorded model responses are keyed by a `when` substring matched against the
request, not by a cursor that advances across invocations. The repair agent is
called once per attempt with a different prompt, and a real model would answer
differently because the prompt differs; matching on content reproduces that.

A cursor would also break under Temporal: a retried activity constructs a fresh
client, so a cursor would replay the wrong exchange. Content matching is
deterministic under retry by construction.

### D18: Tests run before the differ

Within `ValidationWorkflow`, the generated pytest suite executes before the
output comparison. A failing test says *which behaviour* broke; an output diff
says only that something did. Cheap, specific signal first, and a failing test
is recorded as a failing check even when the aggregate outputs happen to agree,
because that means a promised behaviour is unimplemented and the sample data
merely failed to exercise it.

---

## 6. Temporal design

### Workflows

| Workflow | Role | Status |
|---|---|---|
| `ETLMigrationWorkflow` | Parent; sole holder of durable state | ✅ shipped |
| `ValidationWorkflow` | Child; executes both pipelines, runs generated tests, diffs | ✅ shipped |
| `RepairWorkflow` | Child; bounded propose → admit → gate → re-validate loop | ✅ shipped |
| `OptimizationWorkflow` | Child; benchmark → propose → gate → re-validate → benchmark → keep or revert | ✅ shipped |
| `DeliveryWorkflow` | Child; decide → narrate → audit → branch/commit/PR/label | ✅ shipped |
| `DeploymentWorkflow` | Child; rollout → verify | not built: the migration's deliverable is a reviewed PR, and deploying the *agent* is a CD concern that `k8s/` and `release.yml` already cover |

Children rather than inline stages, so each gets its own retry budget, its own
timeouts, and its own row in the Temporal UI. The attachment points are marked
in `workflows/migration.py`.

### Signals and queries

```python
@workflow.signal
def approve(self, decision: ApprovalDecision) -> None: ...
@workflow.signal
def abort(self, reason: str) -> None: ...

@workflow.query
def status(self) -> MigrationStatus: ...     # cheap, safe to poll
@workflow.query
def report(self) -> MigrationRecord | None:  # the full durable record
```

Approval blocks on `workflow.wait_condition(..., timeout=...)`. A migration can
therefore sit pending for days and survive worker restarts, the approval is
durable, not an in-memory future.

One sharp edge worth recording, because it cost a bug: `wait_condition` returns
`None` and signals expiry by **raising `TimeoutError`**. Treating its return
value as "did the condition become true?" type-checks fine and is silently
wrong, every approval takes the timeout path. Caught by `mypy --strict`
reporting "function does not return a value", which is a good argument for
running strict typing on workflow code specifically.

### Retry policy per activity class

| Class | Policy |
|---|---|
| LLM activities | exponential backoff, `maximum_attempts=4`, `AgentContractError` retryable, `NonRetryableMigrationError` not |
| Spark execution | long `start_to_close_timeout`, heartbeats, few attempts |
| GitHub | idempotent by construction (lookup-then-create), safe to retry; `GitHubError` non-retryable because a 403/404/422 says the same thing on the fourth attempt |
| Kubernetes | idempotent by construction (lookup-then-create), safe to retry |

`NonRetryableMigrationError` subclasses go in
`RetryPolicy.non_retryable_error_types` so a malformed input fails fast instead
of burning four LLM calls. That split is why `domain/errors.py` exists.

### Determinism

Every LLM call, file read, Spark job and HTTP request is an Activity. Workflows
run in the sandboxed runner. `etl_migrator.domain` is the only passthrough
module, safe precisely because of the import rule in section 3.1.

---

## 7. Verification strategy without a Temporal server

CI cannot reach a Temporal server (the SDK's time-skipping test server is
downloaded on first use). The temptation is to mock the workflow and call it
tested. Instead the verification is tiered, and each tier is honest about what
it proves:

1. **Pure lifecycle** (`test_lifecycle.py`), every branch: approval, rejection,
   timeout, abort, resume point, stage timing. No server, no mocks, exhaustive.
2. **Real activities** (`test_activities.py`), the actual `@activity.defn`
   callables through Temporal's own `ActivityEnvironment`, including an
   idempotency test that runs `persist_artifacts` twice and compares bytes.
3. **Static workflow guards** (`test_workflow_definition.py`), signals and
   queries registered; retry policies bounded; the non-retryable list naming
   real exception classes; an AST scan proving no `datetime.now()`, `uuid4()`,
   `sleep()` or `open()` reaches workflow code. The scan has its own test
   feeding it code that *should* be rejected, a guard that cannot fire is worse
   than no guard, because it reads like coverage.
4. **Real execution** (`test_validation_e2e.py`, `test_repair_e2e.py`, `-m spark`), both pipelines
   run for real and their outputs are diffed. Half of this file is
   counterfactual: it removes each declared mitigation from the generated code
   and asserts the differ catches it, and it runs the generated pytest suite
   against a broken pipeline to prove the suite is not vacuous. The repair suite
   drives a gate-clean but semantically wrong pipeline through the loop to a
   measured PASS, including one refused attempt.
5. **Real server** (`test_temporal_integration.py`, `-m integration`), durable
   pause/approve/reject/abort, submit idempotency, and a `Replayer` check that
   a completed history replays against current code without divergence. Skips
   with the exact command needed to enable it.

Tiers 1–3 run everywhere. Tiers 4 and 5 need a JVM and a server respectively,
and neither is faked.

---

## 8. What is actually delivered

Implemented, tested, and verified by execution:

- Deterministic AST inspector, data profiler and static gate (no LLM)
- Discovery, Planner and Spark Engineer agents on real AutoGen, with real tools
- Provider abstraction with a deterministic offline provider
- Full artifact set with per-stage timings and a tool-invocation audit trail
- Approval-policy enforcement that overrules the agent
- `ETLMigrationWorkflow` with signals, queries, per-class retry policies, a
  durable human-approval gate and artifact persistence on every exit path
- Five Temporal activities, a sandboxed worker, a Pydantic data converter, and
  a CLI covering submit/status/approve/abort/worker
- `docker-compose.yml` for Temporal + Postgres + UI
- A subprocess sandbox with a scrubbed environment, resource limits and a
  process-group kill, used for both pipelines and for the generated test suite
- A deterministic output differ implementing seven checks, where a check that
  could not run is an error rather than a pass
- Testing and Data Validation agents, and `ValidationWorkflow` as a child
- A bounded autonomous repair loop with a ledger that refuses repeated
  strategies and cosmetically-identical code before spending an execution
- A benchmark-gated optimisation stage whose acceptance rule is a pure function
  over measurements: correctness re-verified in full before the candidate is
  timed, a noise ceiling that refuses unreadable measurements outright, and a
  p75-versus-median robustness test so one lucky run cannot carry a verdict
- A GitHub delivery stage where the PR body's evidence is rendered from the
  record, the agent's prose is audited against it before anything is pushed, an
  unvalidated migration is refused a PR entirely, and every write is
  lookup-then-create so a retried activity re-attaches rather than duplicating
- Two non-root container images, a seven-job CI pipeline that needs no
  credentials because the whole suite runs against recorded model responses, and
  a supply-chain policy enforced by tests: every action allowlisted and pinned,
  and the compromised `aquasecurity/trivy-action` banned outright
- Kubernetes manifests that split the workers by trust so the one executing
  generated code runs with default-deny egress, no credentials and no API token,
  closing the network gap the sandbox declares rather than leaving it
  declared
- A metrics layer derived entirely from the finished record, so a dashboard and
  a pull request cannot disagree, with bounded label cardinality enforced by a
  test and a Grafana dashboard whose every query is checked against the registry
  the code actually builds
- A learning loop keyed on typed strategy identifiers rather than embeddings,
  harvested from the artifacts the system already writes, which reports "not
  enough evidence" instead of a number below three observations, and which
  quantifies the optimiser's own historical over-optimism from the
  `expected_speedup` values the optimiser records
- 649 tests across five verification tiers, including counterfactuals that break
  the generated pipeline in each declared way and assert the system notices, a
  repair run that recovers a wrong pipeline to byte-identical correctness, and a
  null experiment that benchmarks a pipeline against itself to prove the harness
  is not biased toward finding improvements

All ten phases are delivered. Two capabilities are deliberately *absent* rather
than stubbed, and both are noted where they would otherwise be assumed: the
release workflow publishes images but does not deploy (a step that pretended to
would turn a missing capability into a green tick), and only the pandas legacy
inspector is implemented, the `SourceLanguage` enum and its dispatch exist, but
SQL and shell sources are rejected rather than half-parsed.
