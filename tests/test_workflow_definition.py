"""Structural checks on the Temporal workflow that need no server.

A running Temporal server is not available in CI (the SDK's test server is
fetched over the network on first use). That does not have to mean the workflow
is unverified: a large class of real mistakes — a signal that never registered,
a non-deterministic call sneaking into workflow code, a retry policy that
retries something unretryable — is detectable statically.

What genuinely requires a server lives in `test_temporal_integration.py` and
skips cleanly when there is none.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest
from temporalio.activity import _Definition as ActivityDefinition
from temporalio.workflow import _Definition as WorkflowDefinition

from etl_migrator.activities.migration import (
    DeliveryActivities,
    MigrationActivities,
    ValidationActivities,
)
from etl_migrator.config import Settings
from etl_migrator.temporal.worker import (
    ACTIVITY_NAMES,
    PASSTHROUGH_MODULES,
    build_sandbox_runner,
)
from etl_migrator.workflows.delivery import (
    DELIVERY_RETRY,
    DeliveryWorkflow,
)
from etl_migrator.workflows.delivery import (
    NON_RETRYABLE as DELIVERY_NON_RETRYABLE,
)
from etl_migrator.workflows.migration import (
    AGENT_RETRY,
    LOCAL_RETRY,
    NON_RETRYABLE,
    ETLMigrationWorkflow,
)
from etl_migrator.workflows.optimization import BENCHMARK_RETRY, OptimizationWorkflow
from etl_migrator.workflows.repair import RepairWorkflow
from etl_migrator.workflows.validation import (
    DIFF_RETRY,
    EXECUTION_RETRY,
    ValidationWorkflow,
)

#: Calls that make a workflow non-deterministic on replay, and the safe form.
BANNED_CALLS: dict[str, str] = {
    "now": "use workflow.now()",
    "utcnow": "use workflow.now()",
    "time": "use workflow.now()",
    "sleep": "use workflow.sleep()",
    "uuid4": "ids are supplied by the caller",
    "uuid1": "ids are supplied by the caller",
    "random": "use workflow.random()",
    "open": "file access belongs in an activity",
}


def nondeterministic_calls(tree: ast.Module) -> list[tuple[str, int]]:
    """Every banned call in `tree`, ignoring the `workflow.*` safe equivalents."""
    found: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name):
            name = func.id
        elif isinstance(func, ast.Attribute):
            if ast.unparse(func.value) == "workflow":
                continue  # workflow.now() / workflow.random() are the safe versions
            name = func.attr
        else:
            continue
        if name in BANNED_CALLS:
            found.append((name, node.lineno))
    return found


@pytest.fixture(scope="module")
def definition() -> WorkflowDefinition:
    return WorkflowDefinition.must_from_class(ETLMigrationWorkflow)


@pytest.fixture(scope="module")
def workflow_source() -> ast.Module:
    path = Path(inspect.getfile(ETLMigrationWorkflow))
    return ast.parse(path.read_text(encoding="utf-8"))


class TestRegistration:
    def test_workflow_name_is_stable(self, definition: WorkflowDefinition) -> None:
        """The name is a wire contract: the client starts workflows by string."""
        assert definition.name == "ETLMigrationWorkflow"

    def test_signals_are_registered(self, definition: WorkflowDefinition) -> None:
        assert sorted(definition.signals) == ["abort", "approve"]

    def test_queries_are_registered(self, definition: WorkflowDefinition) -> None:
        assert sorted(definition.queries) == ["report", "status"]

    def test_client_helper_targets_the_registered_name(self) -> None:
        from etl_migrator.temporal.client import WORKFLOW_NAME

        assert WorkflowDefinition.must_from_class(ETLMigrationWorkflow).name == WORKFLOW_NAME

    def test_activity_names_match_the_implementations(self) -> None:
        settings = Settings()
        registered = {
            ActivityDefinition.must_from_callable(a).name
            for a in [
                *MigrationActivities(settings).all(),
                *ValidationActivities(settings).all(),
                *DeliveryActivities(settings).all(),
            ]
        }
        assert registered == set(ACTIVITY_NAMES)


class TestDeterminism:
    """Static guards against the classic ways workflow code goes non-deterministic.

    These read the workflow's own AST. Cheaper than a replay test and they fail
    at the moment the bad line is written rather than the moment it diverges.
    """

    def test_the_detector_is_not_vacuous(self) -> None:
        """Guard the guard.

        A static check that can never fire is worse than none, because it reads
        like coverage. This feeds the detector code that *should* be rejected.
        """
        bad = ast.parse(
            "import datetime\n"
            "def run(self):\n"
            "    stamp = datetime.datetime.now()\n"
            "    token = uuid.uuid4()\n"
            "    time.sleep(1)\n"
        )
        assert {name for name, _ in nondeterministic_calls(bad)} == {"now", "uuid4", "sleep"}

    def test_workflow_now_is_not_flagged(self) -> None:
        good = ast.parse("def run(self):\n    return workflow.now()\n")
        assert nondeterministic_calls(good) == []

    def test_no_non_deterministic_calls_in_workflow_module(
        self, workflow_source: ast.Module
    ) -> None:
        offenders = nondeterministic_calls(workflow_source)
        assert not offenders, "\n".join(
            f"line {line}: {name}() — {BANNED_CALLS[name]}" for name, line in offenders
        )

    def test_workflow_reads_time_only_through_temporal(
        self, workflow_source: ast.Module
    ) -> None:
        source = ast.unparse(workflow_source)
        assert "workflow.now()" in source
        assert "datetime.now" not in source

    def test_workflow_does_no_direct_io(self, workflow_source: ast.Module) -> None:
        """Every side effect must be an activity. Imports of io-capable stdlib
        modules in workflow code are the smell that finds violations early."""
        imported = {
            alias.name.split(".")[0]
            for node in ast.walk(workflow_source)
            if isinstance(node, ast.Import)
            for alias in node.names
        } | {
            (node.module or "").split(".")[0]
            for node in ast.walk(workflow_source)
            if isinstance(node, ast.ImportFrom)
        }
        forbidden = {"os", "pathlib", "subprocess", "socket", "requests", "httpx", "pandas"}
        assert not (imported & forbidden), f"workflow imports {imported & forbidden}"

    def test_state_mutation_is_delegated_to_the_pure_lifecycle(
        self, workflow_source: ast.Module
    ) -> None:
        """The branching logic must stay in `domain.lifecycle`, where it is
        testable without a server. This is what keeps the workflow auditable."""
        source = ast.unparse(workflow_source)
        for helper in ("begin_stage", "complete_stage", "fail_stage", "enforce_approval_policy"):
            assert f"lifecycle.{helper}" in source


class TestRetryPolicies:
    def test_agent_activities_back_off_and_are_bounded(self) -> None:
        assert AGENT_RETRY.maximum_attempts == 4
        assert AGENT_RETRY.backoff_coefficient > 1
        assert AGENT_RETRY.maximum_interval is not None

    def test_configuration_errors_are_never_retried(self) -> None:
        """Retrying a missing API key four times just costs four timeouts."""
        assert AGENT_RETRY.non_retryable_error_types == NON_RETRYABLE
        assert LOCAL_RETRY.non_retryable_error_types == NON_RETRYABLE

    def test_non_retryable_list_names_real_exception_classes(self) -> None:
        """Temporal matches on the exception class name, so a typo here silently
        turns a fail-fast error into four retries."""
        import etl_migrator.domain.errors as errors

        for name in NON_RETRYABLE:
            assert hasattr(errors, name), f"{name} is not an exception in domain.errors"
            assert issubclass(getattr(errors, name), Exception)

    def test_agent_contract_errors_remain_retryable(self) -> None:
        """A malformed model response often succeeds on a second sample; that is
        exactly the case worth retrying."""
        assert "AgentContractError" not in NON_RETRYABLE


class TestSandbox:
    def test_runner_builds_with_the_declared_passthroughs(self) -> None:
        assert build_sandbox_runner() is not None

    def test_domain_is_passed_through(self) -> None:
        assert "etl_migrator.domain" in PASSTHROUGH_MODULES

    def test_domain_package_has_no_import_side_effects(self) -> None:
        """Passthrough is only safe because importing `domain` does nothing.

        If someone adds a module-level file read or network call in there, the
        sandbox stops protecting the workflow — so the constraint is asserted
        rather than left as a comment.
        """
        package = Path(inspect.getfile(__import__("etl_migrator.domain", fromlist=["x"]))).parent
        for module_path in sorted(package.glob("*.py")):
            tree = ast.parse(module_path.read_text(encoding="utf-8"))
            for stmt in tree.body:
                if isinstance(
                    stmt,
                    ast.Import
                    | ast.ImportFrom
                    | ast.FunctionDef
                    | ast.AsyncFunctionDef
                    | ast.ClassDef
                    | ast.AnnAssign
                    | ast.Assign,
                ):
                    continue
                if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
                    continue
                pytest.fail(
                    f"{module_path.name}: module-level {type(stmt).__name__} at line "
                    f"{getattr(stmt, 'lineno', '?')} breaks sandbox passthrough safety"
                )

    def test_domain_imports_only_stdlib_and_pydantic(self) -> None:
        package = Path(inspect.getfile(__import__("etl_migrator.domain", fromlist=["x"]))).parent
        allowed_third_party = {"pydantic"}
        for module_path in sorted(package.glob("*.py")):
            tree = ast.parse(module_path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                roots = []
                if isinstance(node, ast.Import):
                    roots = [a.name.split(".")[0] for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    roots = [node.module.split(".")[0]]
                for root in roots:
                    if root in {"etl_migrator", "__future__"}:
                        continue
                    assert (
                        root in allowed_third_party or root in _STDLIB
                    ), f"{module_path.name} imports third-party '{root}'"


_STDLIB = {
    "abc", "ast", "collections", "contextlib", "dataclasses", "datetime", "decimal",
    "enum", "functools", "hashlib", "itertools", "json", "math", "pathlib", "re",
    "statistics", "typing", "uuid",
}


class TestValidationWorkflow:
    """The child workflow gets the same structural scrutiny as the parent."""

    @pytest.fixture(scope="class")
    @classmethod
    def definition(cls) -> WorkflowDefinition:
        return WorkflowDefinition.must_from_class(ValidationWorkflow)

    @pytest.fixture(scope="class")
    @classmethod
    def source(cls) -> ast.Module:
        return ast.parse(
            Path(inspect.getfile(ValidationWorkflow)).read_text(encoding="utf-8")
        )

    def test_registered_with_a_queryable_outcome(
        self, definition: WorkflowDefinition
    ) -> None:
        assert definition.name == "ValidationWorkflow"
        assert sorted(definition.queries) == ["outcome"]

    def test_no_non_deterministic_calls(self, source: ast.Module) -> None:
        offenders = nondeterministic_calls(source)
        assert not offenders, "\n".join(
            f"line {line}: {name}() — {BANNED_CALLS[name]}" for name, line in offenders
        )

    def test_does_no_direct_io(self, source: ast.Module) -> None:
        """Execution and diffing are heavy I/O; every bit of it must be an
        activity, or the workflow cannot replay."""
        imported = {
            alias.name.split(".")[0]
            for node in ast.walk(source)
            if isinstance(node, ast.Import)
            for alias in node.names
        } | {
            (node.module or "").split(".")[0]
            for node in ast.walk(source)
            if isinstance(node, ast.ImportFrom)
        }
        forbidden = {"os", "pathlib", "subprocess", "socket", "pandas", "pyspark"}
        assert not (imported & forbidden), f"workflow imports {imported & forbidden}"

    def test_execution_is_retried_sparingly(self) -> None:
        """A Spark job that OOMs twice will OOM a third time; retrying it four
        times just burns thirty minutes."""
        assert EXECUTION_RETRY.maximum_attempts == 2

    def test_the_differ_is_barely_retried(self) -> None:
        """It is deterministic. A second failure is a bug, not bad luck."""
        assert DIFF_RETRY.maximum_attempts == 2

    def test_diagnosis_runs_only_after_a_failure(self, source: ast.Module) -> None:
        """Structural check that the agent is not invoked on a passing report."""
        rendered = ast.unparse(source)
        assert "ValidationStatus.PASS" in rendered
        assert "diagnose_validation_failure" in rendered


class TestParentInvokesTheChild:
    def test_parent_executes_the_validation_child_workflow(
        self, workflow_source: ast.Module
    ) -> None:
        rendered = ast.unparse(workflow_source)
        assert "execute_child_workflow" in rendered
        assert "ValidationWorkflow.run" in rendered

    def test_child_gets_a_deterministic_id(self, workflow_source: ast.Module) -> None:
        """Derived from the migration id, so a replayed parent addresses the same
        child rather than starting a second one."""
        rendered = ast.unparse(workflow_source)
        assert "-validation" in rendered


class TestRepairWorkflow:
    @pytest.fixture(scope="class")
    @classmethod
    def definition(cls) -> WorkflowDefinition:
        return WorkflowDefinition.must_from_class(RepairWorkflow)

    @pytest.fixture(scope="class")
    @classmethod
    def source(cls) -> ast.Module:
        return ast.parse(Path(inspect.getfile(RepairWorkflow)).read_text(encoding="utf-8"))

    def test_registered_with_a_queryable_outcome(
        self, definition: WorkflowDefinition
    ) -> None:
        assert definition.name == "RepairWorkflow"
        assert sorted(definition.queries) == ["outcome"]

    def test_no_non_deterministic_calls(self, source: ast.Module) -> None:
        offenders = nondeterministic_calls(source)
        assert not offenders, "\n".join(
            f"line {line}: {name}() — {BANNED_CALLS[name]}" for name, line in offenders
        )

    def test_the_loop_is_bounded_by_the_input(self, source: ast.Module) -> None:
        """An unbounded repair loop is a durable, restartable money fire."""
        rendered = ast.unparse(source)
        assert "range(1, params.max_attempts + 1)" in rendered

    def test_admissibility_is_decided_by_the_ledger(self, source: ast.Module) -> None:
        """Not by the agent, and not by a prompt: the check must be in workflow
        code where it is durable and costs nothing."""
        rendered = ast.unparse(source)
        assert "RepairLedger" in rendered
        assert "ledger.admits" in rendered
        assert "ledger.register_baseline" in rendered

    def test_a_rejected_proposal_skips_execution(self, source: ast.Module) -> None:
        """The saving is the point: refusing must not cost a Spark run."""
        rendered = ast.unparse(source)
        rejection = rendered.index("admitted=False")
        gate = rendered.index("MigrationActivities.run_static_analysis")
        assert rejection < gate, "the ledger check must precede the gate and execution"

    def test_exhaustion_is_returned_not_raised(self, source: ast.Module) -> None:
        rendered = ast.unparse(source)
        assert "exhausted = True" in rendered
        assert "raise" not in rendered.split("def run")[-1]


class TestParentInvokesRepair:
    def test_parent_runs_the_repair_child_on_a_validation_failure(
        self, workflow_source: ast.Module
    ) -> None:
        rendered = ast.unparse(workflow_source)
        assert "RepairWorkflow.run" in rendered
        assert "MigrationStage.REPAIR" in rendered

    def test_repair_is_skipped_when_nothing_was_measured(
        self, workflow_source: ast.Module
    ) -> None:
        """An ERROR report means a pipeline never produced output. There is no
        difference for a code change to act on, so repairing would be guessing."""
        rendered = ast.unparse(workflow_source)
        assert "outcome.report.error is not None" in rendered


class TestOptimizationWorkflow:
    @pytest.fixture(scope="class")
    @classmethod
    def definition(cls) -> WorkflowDefinition:
        return WorkflowDefinition.must_from_class(OptimizationWorkflow)

    @pytest.fixture(scope="class")
    @classmethod
    def source(cls) -> ast.Module:
        return ast.parse(
            Path(inspect.getfile(OptimizationWorkflow)).read_text(encoding="utf-8")
        )

    def test_registered_with_a_queryable_outcome(
        self, definition: WorkflowDefinition
    ) -> None:
        assert definition.name == "OptimizationWorkflow"
        assert sorted(definition.queries) == ["outcome"]

    def test_no_non_deterministic_calls(self, source: ast.Module) -> None:
        """Timing code is the obvious place for a stray `time.time()`.

        It belongs in the activity, where the clock is real; a workflow that
        reads it cannot replay.
        """
        offenders = nondeterministic_calls(source)
        assert not offenders, "\n".join(
            f"line {line}: {name}() — {BANNED_CALLS[name]}" for name, line in offenders
        )

    def test_does_no_direct_io(self, source: ast.Module) -> None:
        imported = {
            alias.name.split(".")[0]
            for node in ast.walk(source)
            if isinstance(node, ast.Import)
            for alias in node.names
        } | {
            (node.module or "").split(".")[0]
            for node in ast.walk(source)
            if isinstance(node, ast.ImportFrom)
        }
        forbidden = {"os", "pathlib", "subprocess", "socket", "pandas", "pyspark", "time"}
        assert not (imported & forbidden), f"workflow imports {imported & forbidden}"

    def test_correctness_is_re_verified_before_the_candidate_is_timed(
        self, source: ast.Module
    ) -> None:
        """The ordering claim, checked structurally.

        Benchmarking first and validating afterwards would produce the same
        verdicts, but it would spend a full benchmark on changes that are
        already known to be wrong — and it would invite someone to later report
        the speedup of a pipeline whose output does not match.
        """
        rendered = ast.unparse(source)
        validation = rendered.index("ValidationWorkflow.run")
        candidate_benchmark = rendered.rindex("ValidationActivities.benchmark_spark")
        assert validation < candidate_benchmark

    def test_a_failed_validation_short_circuits_the_attempt(
        self, source: ast.Module
    ) -> None:
        rendered = ast.unparse(source)
        assert "if not validation.passed" in rendered
        assert "regression" in rendered

    def test_the_verdict_comes_from_the_shared_pure_function(
        self, source: ast.Module
    ) -> None:
        """`applied` must be set from `evaluate_optimization` and nothing else.

        This is the assertion that would fail if someone later "simplified" the
        workflow by trusting `proposal.strategy.expected_speedup`.
        """
        rendered = ast.unparse(source)
        assert "evaluate_optimization(" in rendered

        # Read from the AST, not the text: the module docstring mentions the
        # field precisely to say it is never consulted, and a substring search
        # cannot tell an explanation from a use.
        read_in_code = [
            node.lineno
            for node in ast.walk(source)
            if isinstance(node, ast.Attribute) and node.attr == "expected_speedup"
        ]
        assert not read_in_code, (
            f"workflow reads the agent's own speedup claim at line(s) {read_in_code}"
        )

        applied = [
            line.strip() for line in rendered.splitlines() if "_outcome.applied = " in line
        ]
        assert applied == ["self._outcome.applied = True"], applied
        # ...and the only branch it sits in is the accepted one.
        accepted_at = rendered.index("if accepted:")
        assert accepted_at < rendered.index("self._outcome.applied = True")

    def test_the_loop_is_bounded_by_the_input(self, source: ast.Module) -> None:
        rendered = ast.unparse(source)
        assert "range(1, params.max_attempts + 1)" in rendered

    def test_a_repeated_approach_is_refused_before_it_is_measured(
        self, source: ast.Module
    ) -> None:
        """Same economy as the repair ledger: refusing must be free."""
        rendered = ast.unparse(source)
        refusal = rendered.index("already measured")
        benchmark = rendered.rindex("ValidationActivities.benchmark_spark")
        assert refusal < benchmark

    def test_a_failed_baseline_stops_the_stage(self, source: ast.Module) -> None:
        """With no baseline there is nothing to compare against, and every
        subsequent number would be meaningless."""
        rendered = ast.unparse(source)
        assert "if baseline.failed" in rendered

    def test_benchmarks_are_retried_sparingly(self) -> None:
        """A benchmark is several full executions; retrying one costs minutes."""
        assert BENCHMARK_RETRY.maximum_attempts == 2


class TestParentInvokesOptimization:
    def test_parent_runs_the_optimisation_child(
        self, workflow_source: ast.Module
    ) -> None:
        rendered = ast.unparse(workflow_source)
        assert "OptimizationWorkflow.run" in rendered
        assert "MigrationStage.OPTIMIZATION" in rendered

    def test_optimisation_runs_only_after_correctness_is_established(
        self, workflow_source: ast.Module
    ) -> None:
        """Optimising an incorrect migration is optimising the wrong thing."""
        rendered = ast.unparse(workflow_source)
        validation = rendered.index("ValidationWorkflow.run")
        optimization = rendered.index("OptimizationWorkflow.run")
        assert validation < optimization

    def test_an_optimisation_stage_that_keeps_nothing_is_not_a_failure(
        self, workflow_source: ast.Module
    ) -> None:
        """Reverting is the default outcome, not an error path: the migration
        was already correct and shippable before the stage ran."""
        rendered = ast.unparse(workflow_source)
        optimize = rendered.index("def _optimize")
        tail = rendered[optimize:]
        assert "fail_stage" not in tail


class TestDeliveryWorkflow:
    @pytest.fixture(scope="class")
    @classmethod
    def definition(cls) -> WorkflowDefinition:
        return WorkflowDefinition.must_from_class(DeliveryWorkflow)

    @pytest.fixture(scope="class")
    @classmethod
    def source(cls) -> ast.Module:
        return ast.parse(
            Path(inspect.getfile(DeliveryWorkflow)).read_text(encoding="utf-8")
        )

    def test_registered_with_a_queryable_outcome(
        self, definition: WorkflowDefinition
    ) -> None:
        assert definition.name == "DeliveryWorkflow"
        assert sorted(definition.queries) == ["outcome"]

    def test_no_non_deterministic_calls(self, source: ast.Module) -> None:
        offenders = nondeterministic_calls(source)
        assert not offenders, "\n".join(
            f"line {line}: {name}() — {BANNED_CALLS[name]}" for name, line in offenders
        )

    def test_does_no_direct_io(self, source: ast.Module) -> None:
        """Every HTTP call must be an activity. A workflow that talks to GitHub
        directly cannot replay, and would re-issue the request on every replay."""
        imported = {
            alias.name.split(".")[0]
            for node in ast.walk(source)
            if isinstance(node, ast.Import)
            for alias in node.names
        } | {
            (node.module or "").split(".")[0]
            for node in ast.walk(source)
            if isinstance(node, ast.ImportFrom)
        }
        forbidden = {"os", "pathlib", "subprocess", "socket", "httpx", "requests", "base64"}
        assert not (imported & forbidden), f"workflow imports {imported & forbidden}"

    def test_the_policy_decides_before_the_agent_is_called(
        self, source: ast.Module
    ) -> None:
        """A refused migration must not cost an LLM call.

        Same economy as the repair ledger: the cheap deterministic check runs
        first, and the expensive one only on what survives it.
        """
        rendered = ast.unparse(source)
        decision = rendered.index("decide_delivery(params.record)")
        agent = rendered.index("DeliveryActivities.propose_pr_narrative")
        assert decision < agent

    def test_a_refusal_returns_before_anything_reaches_github(
        self, source: ast.Module
    ) -> None:
        rendered = ast.unparse(source)
        refusal = rendered.index("if not decision.should_open")
        deliver = rendered.index("DeliveryActivities.deliver_pull_request")
        assert refusal < deliver

    def test_the_pull_request_is_opened_only_on_a_passing_audit(
        self, source: ast.Module
    ) -> None:
        """The structural form of the rule this stage exists for.

        `deliver_pull_request` must sit inside the `audit.passed` branch. If it
        ever moved outside, prose contradicting the measurements would be
        published, and no test of the audit itself would notice.
        """
        run = next(
            node
            for node in ast.walk(source)
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "run"
        )
        guarded = [
            node
            for node in ast.walk(run)
            if isinstance(node, ast.If)
            and "audit.passed" in ast.unparse(node.test)
            and "deliver_pull_request" in ast.unparse(node)
        ]
        assert len(guarded) == 1, "the delivery call is not guarded by the audit"

    def test_the_revision_loop_is_bounded_by_the_input(self, source: ast.Module) -> None:
        rendered = ast.unparse(source)
        assert "range(params.max_narrative_revisions + 1)" in rendered

    def test_exhausted_revisions_open_nothing(self, source: ast.Module) -> None:
        """Publishing with a disclaimer is not one of the options."""
        rendered = ast.unparse(source)
        assert "skipped_reason" in rendered
        tail = rendered[rendered.rindex("skipped_reason") :]
        assert "deliver_pull_request" not in tail

    def test_github_errors_are_not_retried(self) -> None:
        """A 403 or a 422 says the same thing on the fourth attempt."""
        assert "GitHubError" in DELIVERY_NON_RETRYABLE
        assert DELIVERY_RETRY.non_retryable_error_types == DELIVERY_NON_RETRYABLE

    def test_the_non_retryable_list_names_real_exception_classes(self) -> None:
        import etl_migrator.domain.errors as errors
        from etl_migrator.github.transport import GitHubError

        for name in DELIVERY_NON_RETRYABLE:
            found = getattr(errors, name, None) or (
                GitHubError if name == "GitHubError" else None
            )
            assert found is not None, f"{name} is not a real exception class"
            assert issubclass(found, Exception)


class TestParentInvokesDelivery:
    def test_parent_runs_the_delivery_child(self, workflow_source: ast.Module) -> None:
        rendered = ast.unparse(workflow_source)
        assert "DeliveryWorkflow.run" in rendered
        assert "MigrationStage.PULL_REQUEST" in rendered

    def test_delivery_runs_on_every_exit_path_including_failure(
        self, workflow_source: ast.Module
    ) -> None:
        """Unlike optimisation, delivery is not gated on success.

        A failed migration still produces work worth showing a human, as a
        labelled draft. Hanging delivery off `_finish` — which every return path
        goes through — is what guarantees that, so this asserts the wiring
        rather than trusting it.
        """
        rendered = ast.unparse(workflow_source)
        finish = next(
            node
            for node in ast.walk(workflow_source)
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "_finish"
        )
        assert "self._deliver(params, record)" in ast.unparse(finish)
        assert "_finish" in rendered

    def test_the_record_is_persisted_after_delivery_not_before(
        self, workflow_source: ast.Module
    ) -> None:
        """Otherwise the persisted artifacts would omit the PR that was opened."""
        finish = next(
            node
            for node in ast.walk(workflow_source)
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "_finish"
        )
        body = ast.unparse(finish)
        assert body.index("self._deliver") < body.index("persist_artifacts")

    def test_delivery_is_not_conditioned_on_the_migration_succeeding(
        self, workflow_source: ast.Module
    ) -> None:
        deliver = next(
            node
            for node in ast.walk(workflow_source)
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "_deliver"
        )
        rendered = ast.unparse(deliver)
        assert "record.failed" not in rendered
        assert "ValidationStatus" not in rendered


class TestUntrustedWorkRoutesToTheIsolatedWorker:
    """The workflow half of the network isolation.

    The manifests deny the executing worker any internet egress, and the worker
    registers only the activities that run untrusted code. Neither matters
    unless the workflows actually *dispatch* that work to the isolated queue —
    if an execution activity were left on the default queue it would be served
    by the agent worker, which has a model provider credential and a route to
    the internet, and nothing would look wrong.
    """

    #: Every activity that executes generated code, or reads what it produced.
    #: Kept in step with `ValidationActivities.execution_activities()` by
    #: `test_the_list_here_matches_the_workers_partition` below.
    UNTRUSTED = (
        "run_legacy_pipeline",
        "run_spark_pipeline",
        "run_tests",
        "validate_outputs",
        "benchmark_spark",
    )

    @staticmethod
    def sources() -> dict[str, ast.Module]:
        from etl_migrator.workflows import optimization, validation

        return {
            module.__name__: ast.parse(
                Path(inspect.getfile(module)).read_text(encoding="utf-8")
            )
            for module in (validation, optimization)
        }

    def test_the_list_here_matches_the_workers_partition(self) -> None:
        """Guard against this test drifting from the thing it guards."""
        from etl_migrator.config import Settings
        from etl_migrator.temporal.worker import WorkerRole, activity_names_for

        assert set(self.UNTRUSTED) == set(
            activity_names_for(Settings(), WorkerRole.EXECUTION)
        )

    def test_every_untrusted_activity_is_dispatched_with_a_task_queue(self) -> None:
        """Read the AST rather than the text: what matters is that the call
        carries a `task_queue` keyword, not that the file mentions one."""
        found: set[str] = set()
        for name, tree in self.sources().items():
            for call in (n for n in ast.walk(tree) if isinstance(n, ast.Call)):
                if not call.args:
                    continue
                target = ast.unparse(call.args[0])
                activity = target.rsplit(".", 1)[-1]
                if activity not in self.UNTRUSTED:
                    continue
                found.add(activity)
                keywords = {k.arg for k in call.keywords}
                assert "task_queue" in keywords, (
                    f"{name}: {activity} is dispatched without a task_queue, so it "
                    "would run on the agent worker — which holds credentials and "
                    "has internet egress"
                )
        assert found == set(self.UNTRUSTED), (
            f"not every untrusted activity was found at a call site: "
            f"missing {set(self.UNTRUSTED) - found}"
        )

    def test_the_queue_comes_from_the_workflow_input(self) -> None:
        """Hardcoding the queue name in workflow code would make the split
        impossible to turn off, and a single-worker deployment would hang
        waiting for a queue nobody polls."""
        for name, tree in self.sources().items():
            rendered = ast.unparse(tree)
            assert "execution_queue(params.execution_task_queue)" in rendered, name

    def test_an_unset_queue_means_the_workflows_own_queue(self) -> None:
        """The default has to be "same queue", or a laptop breaks.

        Temporal treats `None` as "inherit"; returning an empty string instead
        would dispatch to a queue named "" and the activity would never run.
        """
        from etl_migrator.workflows.validation import execution_queue

        assert execution_queue("") is None
        assert execution_queue("etl-execution") == "etl-execution"

    def test_the_parent_passes_the_queue_to_every_child(
        self, workflow_source: ast.Module
    ) -> None:
        """A child that does not receive it silently falls back to inheriting
        the parent's queue, undoing the split for that whole stage.

        Matched on the AST rather than by searching the text, because the
        obvious text search finds the module's *import* of the input class and
        passes without ever looking at a construction.
        """
        children = {
            "ValidationWorkflowInput",
            "RepairWorkflowInput",
            "OptimizationWorkflowInput",
        }
        constructed: set[str] = set()
        for call in (n for n in ast.walk(workflow_source) if isinstance(n, ast.Call)):
            name = call.func.id if isinstance(call.func, ast.Name) else ""
            if name not in children:
                continue
            constructed.add(name)
            keywords = {k.arg for k in call.keywords}
            assert "execution_task_queue" in keywords, (
                f"{name} is constructed without execution_task_queue, so that "
                "stage runs its untrusted work on the agent worker"
            )
        assert constructed == children, f"never constructed: {children - constructed}"
