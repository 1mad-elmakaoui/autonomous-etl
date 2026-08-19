"""Deterministic AST inspection of a legacy pandas pipeline.

This is the Discovery agent's eyes. It contains no LLM call and no heuristics
that guess intent — it reports only what is syntactically present, with line
numbers. The agent's job is to *interpret* these facts (what is the business
rule? what is risky?), not to rediscover them, because an LLM reading raw source
will hallucinate a column that is not there, whereas this cannot.

Everything it returns is traceable to a line number, which is what makes the
generated PR reviewable.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

from pydantic import Field

from etl_migrator.domain.enums import SourceLanguage
from etl_migrator.domain.errors import UnsupportedSourceError
from etl_migrator.domain.spec import StrictModel

# Pandas entrypoints that read data.
_READ_FUNCS = {
    "read_csv": "csv",
    "read_parquet": "parquet",
    "read_json": "json",
    "read_sql": "jdbc",
    "read_sql_query": "jdbc",
    "read_sql_table": "jdbc",
    "read_excel": "unknown",
    "read_table": "csv",
}

# Pandas DataFrame methods that write data.
_WRITE_METHODS = {
    "to_csv": "csv",
    "to_parquet": "parquet",
    "to_json": "json",
    "to_sql": "jdbc",
}

# DataFrame methods worth reporting, mapped to a coarse operation family.
_DATAFRAME_METHODS = {
    "merge": "join",
    "join": "join",
    "groupby": "aggregate",
    "agg": "aggregate",
    "sum": "aggregate",
    "mean": "aggregate",
    "count": "aggregate",
    "min": "aggregate",
    "max": "aggregate",
    "nunique": "aggregate",
    "sort_values": "sort",
    "drop_duplicates": "deduplicate",
    "rename": "rename",
    "fillna": "fillna",
    "dropna": "dropna",
    "astype": "astype",
    "reset_index": "reset_index",
    "set_index": "reset_index",
    "query": "filter",
    "apply": "apply_udf",
    "map": "apply_udf",
    "applymap": "apply_udf",
    "rolling": "window",
    "expanding": "window",
    "shift": "window",
    "rank": "window",
    "cumsum": "window",
    "pivot_table": "pivot",
    "pivot": "pivot",
    "concat": "union",
    "drop": "project",
    "head": "project",
    "assign": "derive_column",
    "round": "derive_column",
}


class SourceCall(StrictModel):
    line: int
    qualified: str = Field(description="e.g. 'pd.read_csv' or 'customers.merge'")
    receiver: str | None = Field(default=None, description="Variable or module alias called on.")
    callee: str
    family: str = Field(description="Coarse operation family, e.g. 'join', 'aggregate', 'read'.")
    args: list[str] = Field(default_factory=list, description="Unparsed positional arguments.")
    kwargs: dict[str, str] = Field(default_factory=dict, description="Unparsed keyword arguments.")
    assigned_to: str | None = None
    chain: list[str] = Field(
        default_factory=list, description="Method names chained after this call, in order."
    )


class BooleanFilter(StrictModel):
    """`df[df["age"] >= 18]` — a mask subscript, not a column selection."""

    line: int
    target: str
    expression: str
    columns: list[str] = Field(default_factory=list)
    assigned_to: str | None = None


class ColumnAssignment(StrictModel):
    """`orders["revenue"] = orders["quantity"] * orders["price"]`"""

    line: int
    target: str
    column: str
    expression: str
    columns_read: list[str] = Field(default_factory=list)


class SourceInspection(StrictModel):
    """Complete, deterministic view of a legacy source file."""

    path: str
    language: SourceLanguage
    sha256: str
    line_count: int
    imports: list[str] = Field(default_factory=list)
    import_aliases: dict[str, str] = Field(default_factory=dict)
    defined_functions: list[str] = Field(default_factory=list)
    reads: list[SourceCall] = Field(default_factory=list)
    writes: list[SourceCall] = Field(default_factory=list)
    calls: list[SourceCall] = Field(default_factory=list)
    boolean_filters: list[BooleanFilter] = Field(default_factory=list)
    column_assignments: list[ColumnAssignment] = Field(default_factory=list)
    referenced_columns: list[str] = Field(default_factory=list)
    dataframe_variables: list[str] = Field(default_factory=list)
    has_entrypoint: bool = False

    def render(self) -> str:
        """Compact, token-efficient rendering handed to the agent as tool output."""
        lines = [
            f"file: {self.path}  ({self.line_count} lines, sha256={self.sha256[:12]})",
            f"language: {self.language.value}",
            f"imports: {', '.join(self.imports) or 'none'}",
            f"functions: {', '.join(self.defined_functions) or 'none'}"
            f"   entrypoint_run_defined: {self.has_entrypoint}",
            f"dataframe variables: {', '.join(self.dataframe_variables) or 'none'}",
            f"referenced column literals: {', '.join(self.referenced_columns) or 'none'}",
            "",
            "READS:",
        ]
        lines += [
            f"  L{c.line}: {c.assigned_to or '?'} = {c.qualified}({', '.join(c.args)}"
            f"{', ' if c.args and c.kwargs else ''}"
            f"{', '.join(f'{k}={v}' for k, v in c.kwargs.items())})"
            for c in self.reads
        ] or ["  none"]
        lines += ["", "WRITES:"]
        lines += [
            f"  L{c.line}: {c.qualified}({', '.join(c.args)})" for c in self.writes
        ] or ["  none"]
        lines += ["", "BOOLEAN FILTERS:"]
        lines += [
            f"  L{f.line}: {f.assigned_to or '?'} = {f.target}[{f.expression}]  cols={f.columns}"
            for f in self.boolean_filters
        ] or ["  none"]
        lines += ["", "DERIVED COLUMNS:"]
        lines += [
            f"  L{a.line}: {a.target}[{a.column!r}] = {a.expression}  reads={a.columns_read}"
            for a in self.column_assignments
        ] or ["  none"]
        lines += ["", "DATAFRAME OPERATIONS:"]
        lines += [
            f"  L{c.line}: [{c.family}] {c.qualified}("
            f"{', '.join(c.args + [f'{k}={v}' for k, v in c.kwargs.items()])})"
            + (f" -> {c.assigned_to}" if c.assigned_to else "")
            + (f"  chained: .{'.'.join(c.chain)}" if c.chain else "")
            for c in self.calls
        ] or ["  none"]
        return "\n".join(lines)


def detect_language(path: Path, text: str) -> SourceLanguage:
    suffix = path.suffix.lower()
    if suffix == ".sql":
        return SourceLanguage.SQL
    if suffix in {".sh", ".bash"}:
        return SourceLanguage.SHELL
    if suffix == ".py":
        lowered = text.lower()
        if "import pandas" in lowered or "from pandas" in lowered:
            return SourceLanguage.PYTHON_PANDAS
        return SourceLanguage.PYTHON_PLAIN
    raise UnsupportedSourceError(f"cannot determine source language for {path}")


#: Keyword arguments whose string values name columns rather than files or modes.
_COLUMN_KWARGS = frozenset(
    {"on", "left_on", "right_on", "by", "subset", "columns", "index", "values", "level"}
)

#: Methods whose positional string arguments name columns.
_COLUMN_POSITIONAL_METHODS = frozenset(
    {"groupby", "sort_values", "drop_duplicates", "drop", "set_index", "agg", "pivot_table"}
)


def _dotted(node: ast.expr) -> tuple[str, str | None]:
    """Render a call target, returning `(qualified_name, root_identifier)`.

    The root identifier is `None` when the call is chained off an expression
    rather than a bare name — `df.groupby('c')['r'].sum` has no root. In that
    case the base is unparsed verbatim, which reads far better in tool output
    than a `<subscript>` placeholder and preserves which column was aggregated.
    """
    attrs: list[str] = []
    cur: ast.expr = node
    while isinstance(cur, ast.Attribute):
        attrs.append(cur.attr)
        cur = cur.value

    if isinstance(cur, ast.Name):
        prefix, root = cur.id, cur.id
    else:
        rendered = ast.unparse(cur)
        prefix = rendered if len(rendered) <= 60 else rendered[:57] + "..."
        root = None

    qualified = ".".join([prefix, *reversed(attrs)]) if attrs else prefix
    return qualified, root


def _column_literals_from_call(node: ast.Call, callee: str) -> list[str]:
    """String literals in this call that denote column names.

    Deliberately narrow. Harvesting every string in every call would sweep up
    file paths and write modes and present them to the agent as columns.
    """
    found: list[str] = []

    def collect(value: ast.expr) -> None:
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            found.append(value.value)
        elif isinstance(value, ast.List | ast.Tuple):
            for element in value.elts:
                if isinstance(element, ast.Constant) and isinstance(element.value, str):
                    found.append(element.value)
        elif isinstance(value, ast.Dict):
            for key in value.keys:
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    found.append(key.value)

    for kw in node.keywords:
        if kw.arg in _COLUMN_KWARGS:
            collect(kw.value)
    if callee in _COLUMN_POSITIONAL_METHODS:
        for arg in node.args:
            collect(arg)
    return found


def _string_columns(node: ast.AST) -> list[str]:
    """Every string literal used as a subscript key under `node`."""
    out: list[str] = []
    for sub in ast.walk(node):
        if isinstance(sub, ast.Subscript):
            key = sub.slice
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                out.append(key.value)
            elif isinstance(key, ast.List):
                out.extend(
                    e.value
                    for e in key.elts
                    if isinstance(e, ast.Constant) and isinstance(e.value, str)
                )
    return out


def _is_mask(node: ast.expr) -> bool:
    """True for `df[<boolean expression>]` rather than `df["col"]`."""
    return isinstance(node, ast.Compare | ast.BoolOp | ast.UnaryOp) or (
        isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitAnd | ast.BitOr)
    )


class _Visitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.imports: list[str] = []
        self.aliases: dict[str, str] = {}
        self.functions: list[str] = []
        self.reads: list[SourceCall] = []
        self.writes: list[SourceCall] = []
        self.calls: list[SourceCall] = []
        self.filters: list[BooleanFilter] = []
        self.assignments: list[ColumnAssignment] = []
        self.columns: set[str] = set()
        self.df_vars: set[str] = set()
        self._assign_target: str | None = None

    # -- imports -----------------------------------------------------------
    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.imports.append(alias.name)
            self.aliases[alias.asname or alias.name.split(".")[0]] = alias.name
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        self.imports.append(module)
        for alias in node.names:
            self.aliases[alias.asname or alias.name] = f"{module}.{alias.name}"
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.functions.append(node.name)
        self.generic_visit(node)

    # -- assignments -------------------------------------------------------
    def visit_Assign(self, node: ast.Assign) -> None:
        target = node.targets[0] if len(node.targets) == 1 else None

        # orders["revenue"] = <expr>
        if (
            isinstance(target, ast.Subscript)
            and isinstance(target.value, ast.Name)
            and isinstance(target.slice, ast.Constant)
            and isinstance(target.slice.value, str)
        ):
            self.df_vars.add(target.value.id)
            self.columns.add(target.slice.value)
            read = _string_columns(node.value)
            self.columns.update(read)
            self.assignments.append(
                ColumnAssignment(
                    line=node.lineno,
                    target=target.value.id,
                    column=target.slice.value,
                    expression=ast.unparse(node.value),
                    columns_read=sorted(set(read)),
                )
            )
            self.generic_visit(node.value)
            return

        name = target.id if isinstance(target, ast.Name) else None
        prev, self._assign_target = self._assign_target, name
        try:
            self.generic_visit(node)
        finally:
            self._assign_target = prev

    # -- subscripts (filter vs projection) ---------------------------------
    def visit_Subscript(self, node: ast.Subscript) -> None:
        if isinstance(node.value, ast.Name) and _is_mask(node.slice):
            self.df_vars.add(node.value.id)
            cols = _string_columns(node.slice)
            self.columns.update(cols)
            self.filters.append(
                BooleanFilter(
                    line=node.lineno,
                    target=node.value.id,
                    expression=ast.unparse(node.slice),
                    columns=sorted(set(cols)),
                    assigned_to=self._assign_target,
                )
            )
        self.columns.update(_string_columns(node))
        self.generic_visit(node)

    # -- calls -------------------------------------------------------------
    def visit_Call(self, node: ast.Call) -> None:
        qualified, root = _dotted(node.func)
        callee = qualified.rsplit(".", 1)[-1] if qualified else ""
        receiver = qualified.rsplit(".", 1)[0] if "." in qualified else None

        # `os.path.join(...)` is not a DataFrame join. Any call rooted at an
        # imported module that is not pandas is library plumbing, not a data
        # operation, and misreporting it would send the planner chasing a join
        # that does not exist.
        if (
            root is not None
            and root in self.aliases
            and self.aliases[root].split(".")[0] != "pandas"
        ):
            self.generic_visit(node)
            return

        args = [ast.unparse(a) for a in node.args]
        kwargs = {kw.arg: ast.unparse(kw.value) for kw in node.keywords if kw.arg}
        self.columns.update(_string_columns(node))

        record = SourceCall(
            line=node.lineno,
            qualified=qualified,
            receiver=receiver,
            callee=callee,
            family="unknown",
            args=args,
            kwargs=kwargs,
            assigned_to=self._assign_target,
            chain=self._chain_of(node),
        )

        if callee in _READ_FUNCS:
            record.family = "read"
            self.reads.append(record)
            if self._assign_target:
                self.df_vars.add(self._assign_target)
        elif callee in _WRITE_METHODS:
            record.family = "write"
            self.writes.append(record)
            if receiver:
                self.df_vars.add(receiver.split(".")[0])
        elif callee in _DATAFRAME_METHODS:
            record.family = _DATAFRAME_METHODS[callee]
            self.calls.append(record)
            self.columns.update(_column_literals_from_call(node, callee))
            if receiver and receiver.isidentifier():
                self.df_vars.add(receiver)
            if self._assign_target:
                self.df_vars.add(self._assign_target)

        self.generic_visit(node)

    @staticmethod
    def _chain_of(node: ast.Call) -> list[str]:
        """Method names applied to the *result* of this call, innermost first.

        `df.groupby("c")["r"].sum().reset_index()` records groupby with
        chain=['sum', 'reset_index'], which is what tells the planner this is a
        single aggregate rather than three unrelated steps.
        """
        chain: list[str] = []
        cur: ast.expr = node
        while True:
            parent = getattr(cur, "_etlm_parent", None)
            if isinstance(parent, ast.Attribute):
                grand = getattr(parent, "_etlm_parent", None)
                if isinstance(grand, ast.Call) and grand.func is parent:
                    chain.append(parent.attr)
                    cur = grand
                    continue
            if isinstance(parent, ast.Subscript):
                cur = parent
                continue
            break
        return chain


def _annotate_parents(tree: ast.AST) -> None:
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            child._etlm_parent = parent  # type: ignore[attr-defined]


def inspect_source(path: Path) -> SourceInspection:
    """Parse a legacy pipeline and report every fact needed to migrate it."""
    text = path.read_text(encoding="utf-8")
    language = detect_language(path, text)
    if language not in {SourceLanguage.PYTHON_PANDAS, SourceLanguage.PYTHON_PLAIN}:
        raise UnsupportedSourceError(
            f"{path} is {language.value}; phase 1 supports Python/pandas sources only. "
            "SQL and shell sources are handled by dedicated inspectors in a later phase."
        )

    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError as exc:  # pragma: no cover - exercised in tests
        raise UnsupportedSourceError(f"{path} is not valid Python: {exc}") from exc

    _annotate_parents(tree)
    visitor = _Visitor()
    visitor.visit(tree)

    return SourceInspection(
        path=str(path),
        language=language,
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        line_count=text.count("\n") + 1,
        imports=sorted(set(visitor.imports)),
        import_aliases=visitor.aliases,
        defined_functions=visitor.functions,
        reads=visitor.reads,
        writes=visitor.writes,
        calls=visitor.calls,
        boolean_filters=visitor.filters,
        column_assignments=visitor.assignments,
        referenced_columns=sorted(visitor.columns),
        dataframe_variables=sorted(visitor.df_vars),
        has_entrypoint="run" in visitor.functions,
    )
