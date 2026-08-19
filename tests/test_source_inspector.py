"""Tests for the deterministic AST inspector — the Discovery agent's eyes."""

from __future__ import annotations

from pathlib import Path

import pytest

from etl_migrator.domain.enums import SourceLanguage
from etl_migrator.domain.errors import UnsupportedSourceError
from etl_migrator.tools.source_inspector import detect_language, inspect_source


def _write(tmp_path: Path, body: str, name: str = "pipeline.py") -> Path:
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


class TestLanguageDetection:
    def test_pandas_source_detected(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "import pandas as pd\n")
        assert detect_language(path, path.read_text()) is SourceLanguage.PYTHON_PANDAS

    def test_plain_python_detected(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "import csv\n")
        assert detect_language(path, path.read_text()) is SourceLanguage.PYTHON_PLAIN

    def test_sql_rejected_with_actionable_message(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "SELECT 1;", name="q.sql")
        with pytest.raises(UnsupportedSourceError, match="phase 1 supports Python/pandas"):
            inspect_source(path)

    def test_invalid_python_is_not_a_crash(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "import pandas as pd\ndef (:\n")
        with pytest.raises(UnsupportedSourceError, match="not valid Python"):
            inspect_source(path)


class TestExampleInspection:
    """The example pipeline is the reference case; each assertion pins one planted trap."""

    @pytest.fixture(scope="class")
    @classmethod
    def inspection(cls, legacy_source: Path):
        return inspect_source(legacy_source)

    def test_reads_both_inputs(self, inspection) -> None:
        assert [r.assigned_to for r in inspection.reads] == ["customers", "orders"]
        assert all(r.callee == "read_csv" for r in inspection.reads)

    def test_detects_the_write(self, inspection) -> None:
        assert len(inspection.writes) == 1
        assert inspection.writes[0].callee == "to_csv"

    def test_boolean_filter_is_distinguished_from_projection(self, inspection) -> None:
        assert len(inspection.boolean_filters) == 1
        assert inspection.boolean_filters[0].columns == ["age"]
        assert inspection.boolean_filters[0].assigned_to == "customers"

    def test_derived_column_records_both_sides(self, inspection) -> None:
        assign = inspection.column_assignments[0]
        assert assign.column == "revenue"
        assert assign.columns_read == ["price", "quantity"]

    def test_merge_is_classified_as_a_join(self, inspection) -> None:
        joins = [c for c in inspection.calls if c.family == "join"]
        assert len(joins) == 1
        assert joins[0].kwargs["how"] == "'left'"

    def test_os_path_join_is_not_a_dataframe_join(self, inspection) -> None:
        """Regression: `os.path.join` was classified as a pandas merge, which would
        have sent the planner chasing a join that does not exist."""
        assert all("os.path.join" not in c.qualified for c in inspection.calls)

    def test_column_names_from_string_arguments_are_captured(self, inspection) -> None:
        """`groupby('country')` and `on='customer_id'` are columns, not literals to ignore."""
        assert {"country", "customer_id"} <= set(inspection.referenced_columns)

    def test_file_paths_are_not_mistaken_for_columns(self, inspection) -> None:
        assert not any(c.endswith(".csv") for c in inspection.referenced_columns)

    def test_chained_aggregate_is_linked(self, inspection) -> None:
        groupby = next(c for c in inspection.calls if c.callee == "groupby")
        assert "sum" in groupby.chain and "reset_index" in groupby.chain

    def test_entrypoint_and_dataframe_variables(self, inspection) -> None:
        assert inspection.has_entrypoint
        assert set(inspection.dataframe_variables) == {"customers", "orders", "result"}

    def test_render_is_stable_and_line_numbered(self, inspection) -> None:
        rendered = inspection.render()
        assert "BOOLEAN FILTERS:" in rendered
        assert "L34:" in rendered
