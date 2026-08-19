"""Static analysis gate for LLM-generated PySpark code.

Two jobs, and it is worth being explicit that they are different:

1. **Security.** Generated code is untrusted input, full stop. This gate is the
   first of two layers, the second being the sandboxed subprocess. It rejects
   imports and calls that could touch the host before the code is ever handed
   to an executor.

2. **Feedback.** The Spark Engineer agent calls this gate *as a tool*, inside
   its own reasoning loop. So a violation is not a crash — it is an observation
   the agent gets to act on, which is the "act → observe → correct" cycle the
   brief asks for, happening within a single agent turn.

Design note: the gate is an allowlist, not a blocklist. A blocklist of dangerous
imports is unbounded; an allowlist of `pyspark` plus a handful of stdlib modules
is small and auditable.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

from etl_migrator.domain.artifacts import Finding, StaticAnalysisReport
from etl_migrator.domain.enums import Severity

#: Top-level modules a generated Spark pipeline may import. Nothing else.
ALLOWED_IMPORT_ROOTS: frozenset[str] = frozenset(
    {
        "pyspark",
        "__future__",
        "typing",
        "datetime",
        "decimal",
        "math",
        "dataclasses",
        "enum",
        "collections",
        "functools",
        "itertools",
    }
)

#: Callables that must never appear. Dynamic execution or host access.
BANNED_CALLS: frozenset[str] = frozenset(
    {
        "eval",
        "exec",
        "compile",
        "__import__",
        "open",
        "input",
        "breakpoint",
        "globals",
        "locals",
        "vars",
        "memoryview",
    }
)

#: Attribute names used for sandbox escape.
BANNED_ATTRIBUTES: frozenset[str] = frozenset(
    {
        "__globals__",
        "__builtins__",
        "__subclasses__",
        "__bases__",
        "__mro__",
        "__code__",
        "__loader__",
        "__reduce__",
        "__class__",
    }
)

#: Spark calls that pull an unbounded dataset to the driver.
DRIVER_COLLECTING_METHODS: frozenset[str] = frozenset({"collect", "toPandas", "toLocalIterator"})

#: The contract every generated pipeline must satisfy so the runner can execute
#: legacy and Spark implementations interchangeably.
REQUIRED_ENTRYPOINT = "run"
REQUIRED_PARAMETERS = ("spark", "input_dir", "output_dir")


@dataclass(frozen=True)
class GateOptions:
    """Tunable severities. Defaults are the production posture.

    `for_tests()` exists because generated *test* code is a different artifact
    with different rules: it may import pytest, and it may legitimately call
    `.collect()` — a test asserting on three rows is exactly the case where
    pulling data to the driver is correct. What does not relax is module-import
    purity, because pytest collects a module by importing it, so import-time
    side effects there are if anything more dangerous.
    """

    allow_udf: bool = False
    allow_driver_collect: bool = False
    require_entrypoint: bool = True
    extra_import_roots: frozenset[str] = frozenset()
    require_test_functions: bool = False

    @classmethod
    def for_pipeline(cls) -> GateOptions:
        return cls()

    @classmethod
    def for_tests(cls) -> GateOptions:
        return cls(
            allow_driver_collect=True,
            require_entrypoint=False,
            extra_import_roots=frozenset({"pytest"}),
            require_test_functions=True,
        )

    @property
    def allowed_import_roots(self) -> frozenset[str]:
        return ALLOWED_IMPORT_ROOTS | self.extra_import_roots


class _GateVisitor(ast.NodeVisitor):
    def __init__(self, options: GateOptions) -> None:
        self.options = options
        self.findings: list[Finding] = []
        self.imports: list[str] = []

    def _add(self, code: str, severity: Severity, message: str, line: int | None = None) -> None:
        self.findings.append(Finding(code=code, severity=severity, message=message, line=line))

    # -- imports -----------------------------------------------------------
    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._check_import(alias.name, node.lineno)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level:
            self._add(
                "GATE002",
                Severity.ERROR,
                "relative import is not allowed; generated modules are standalone",
                node.lineno,
            )
        elif node.module:
            self._check_import(node.module, node.lineno)
        self.generic_visit(node)

    def _check_import(self, module: str, line: int) -> None:
        self.imports.append(module)
        root = module.split(".")[0]
        allowed = self.options.allowed_import_roots
        if root not in allowed:
            self._add(
                "GATE002",
                Severity.ERROR,
                f"import of '{module}' is not permitted. Allowed roots: "
                f"{', '.join(sorted(allowed))}",
                line,
            )

    # -- calls -------------------------------------------------------------
    def visit_Call(self, node: ast.Call) -> None:
        name = self._callee_name(node.func)

        if name in BANNED_CALLS:
            self._add(
                "GATE003",
                Severity.ERROR,
                f"call to '{name}' is forbidden in generated code "
                "(dynamic execution / host access)",
                node.lineno,
            )

        if name in DRIVER_COLLECTING_METHODS and not self.options.allow_driver_collect:
            self._add(
                "GATE005",
                Severity.ERROR,
                f"'.{name}()' pulls the full dataset to the driver and will not scale. "
                "Use DataFrame operations, or aggregate first if a scalar is needed.",
                node.lineno,
            )

        if name in {"udf", "pandas_udf"} and not self.options.allow_udf:
            self._add(
                "GATE006",
                Severity.WARNING,
                f"'{name}' blocks Catalyst optimisation and serialises row-by-row. "
                "Prefer pyspark.sql.functions; keep the UDF only if no built-in exists.",
                node.lineno,
            )

        if name == "getOrCreate" and "SparkSession" in ast.unparse(node.func):
            self._add(
                "GATE008",
                Severity.ERROR,
                "generated code must not create its own SparkSession; the runner injects "
                "one via run(spark, input_dir, output_dir) so configuration stays under "
                "the optimizer's control",
                node.lineno,
            )

        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in BANNED_ATTRIBUTES:
            self._add(
                "GATE007",
                Severity.ERROR,
                f"access to '{node.attr}' is forbidden (sandbox escape vector)",
                node.lineno,
            )
        if node.attr == "rdd":
            self._add(
                "GATE009",
                Severity.WARNING,
                "dropping to the RDD API bypasses Catalyst; prefer the DataFrame API",
                node.lineno,
            )
        self.generic_visit(node)

    @staticmethod
    def _callee_name(func: ast.expr) -> str:
        if isinstance(func, ast.Name):
            return func.id
        if isinstance(func, ast.Attribute):
            return func.attr
        return ""


def _check_module_purity(tree: ast.Module) -> list[Finding]:
    """Importing a generated module must have no effect beyond defining names.

    The sandbox imports the module and then calls `run(...)`. If import time can
    do work, the security boundary is meaningless and the code is untestable.
    """
    findings: list[Finding] = []
    for stmt in tree.body:
        if isinstance(
            stmt,
            ast.Import
            | ast.ImportFrom
            | ast.FunctionDef
            | ast.AsyncFunctionDef
            | ast.ClassDef
            | ast.AnnAssign,
        ):
            continue
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
            continue  # module docstring
        if isinstance(stmt, ast.Assign) and all(
            isinstance(t, ast.Name) and t.id.isupper() for t in stmt.targets
        ):
            continue  # module constant
        if isinstance(stmt, ast.If):
            test = ast.unparse(stmt.test)
            if "__name__" in test:
                continue
        findings.append(
            Finding(
                code="GATE011",
                severity=Severity.ERROR,
                message=(
                    f"module-level statement '{type(stmt).__name__}' "
                    "executes on import. "
                    "All work must live inside run(spark, input_dir, output_dir)."
                ),
                line=getattr(stmt, "lineno", None),
            )
        )
    return findings


def _check_test_functions(tree: ast.Module) -> list[Finding]:
    """A test module with no tests passes vacuously, which is worse than failing."""
    tests = [
        stmt.name
        for stmt in tree.body
        if isinstance(stmt, ast.FunctionDef | ast.AsyncFunctionDef)
        and stmt.name.startswith("test_")
    ]
    if tests:
        return []
    return [
        Finding(
            code="GATE012",
            severity=Severity.ERROR,
            message="generated test module defines no test_* functions; an empty suite "
            "reports success without asserting anything",
        )
    ]


def _check_entrypoint(tree: ast.Module) -> list[Finding]:
    for stmt in tree.body:
        if isinstance(stmt, ast.FunctionDef) and stmt.name == REQUIRED_ENTRYPOINT:
            params = [a.arg for a in stmt.args.args]
            if tuple(params) != REQUIRED_PARAMETERS:
                return [
                    Finding(
                        code="GATE004",
                        severity=Severity.ERROR,
                        message=(
                            f"entrypoint must be exactly "
                            f"def {REQUIRED_ENTRYPOINT}({', '.join(REQUIRED_PARAMETERS)}), "
                            f"got ({', '.join(params)})"
                        ),
                        line=stmt.lineno,
                    )
                ]
            return []
    return [
        Finding(
            code="GATE004",
            severity=Severity.ERROR,
            message=(
                f"missing entrypoint: def {REQUIRED_ENTRYPOINT}"
                f"({', '.join(REQUIRED_PARAMETERS)}) -> None"
            ),
        )
    ]


def analyze_generated_code(code: str, options: GateOptions | None = None) -> StaticAnalysisReport:
    """Run the full gate. Never raises on bad code — it *reports* on it."""
    options = options or GateOptions()

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return StaticAnalysisReport(
            passed=False,
            findings=[
                Finding(
                    code="GATE001",
                    severity=Severity.ERROR,
                    message=f"generated code is not valid Python: {exc.msg}",
                    line=exc.lineno,
                )
            ],
        )

    visitor = _GateVisitor(options)
    visitor.visit(tree)
    findings = list(visitor.findings)
    findings += _check_module_purity(tree)
    if options.require_entrypoint:
        findings += _check_entrypoint(tree)
    if options.require_test_functions:
        findings += _check_test_functions(tree)

    if not any(i.split(".")[0] == "pyspark" for i in visitor.imports):
        findings.append(
            Finding(
                code="GATE010",
                severity=Severity.ERROR,
                message="generated code imports nothing from pyspark; this is not a Spark pipeline",
            )
        )

    findings.sort(key=lambda f: (f.line or 0, f.code))
    passed = not any(f.severity is Severity.ERROR for f in findings)
    return StaticAnalysisReport(passed=passed, findings=findings)
