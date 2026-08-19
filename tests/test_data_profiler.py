"""Profiler tests. The profiler is what turns an inferred schema into a measured one."""

from __future__ import annotations

from pathlib import Path

from etl_migrator.domain.enums import DataFormat
from etl_migrator.tools.data_profiler import profile_dataset, profile_directory


class TestProfiling:
    def test_measures_nulls_and_cardinality(self, tmp_path: Path) -> None:
        csv = tmp_path / "d.csv"
        csv.write_text("a,b\n1,x\n2,\n2,y\n", encoding="utf-8")
        profile = profile_dataset(csv)

        assert profile.exists and profile.row_count == 3
        by_name = {c.name: c for c in profile.columns}
        assert by_name["b"].null_count == 1
        assert by_name["a"].distinct_count == 2
        assert by_name["a"].min_value == "1" and by_name["a"].max_value == "2"

    def test_detects_the_int_to_float_promotion_trap(self, tmp_path: Path) -> None:
        """pandas promotes an int column to float64 when it contains nulls. A
        migration that declares IntegerType on the Spark side fails schema
        comparison, so the profiler must report what pandas actually produced."""
        csv = tmp_path / "d.csv"
        csv.write_text("age,name\n30,a\n,b\n40,c\n", encoding="utf-8")
        column = next(c for c in profile_dataset(csv).columns if c.name == "age")
        assert column.dtype == "float64"
        assert column.null_count == 1

    def test_broadcast_eligibility_is_measured_not_guessed(self, tmp_path: Path) -> None:
        small = tmp_path / "small.csv"
        small.write_text("a\n1\n", encoding="utf-8")
        assert profile_dataset(small).broadcast_candidate is True

    def test_missing_file_is_reported_not_raised(self, tmp_path: Path) -> None:
        profile = profile_dataset(tmp_path / "nope.csv")
        assert not profile.exists and profile.error == "file not found"
        assert "NOT FOUND" in profile.render()

    def test_malformed_file_is_reported_not_raised(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.parquet"
        bad.write_bytes(b"not parquet at all")
        profile = profile_dataset(bad)
        assert not profile.exists and profile.error

    def test_unknown_format(self, tmp_path: Path) -> None:
        other = tmp_path / "f.bin"
        other.write_bytes(b"\x00")
        assert profile_dataset(other).format is DataFormat.UNKNOWN


class TestDirectoryProfiling:
    def test_skips_unreadable_formats_and_sorts_results(self, tmp_path: Path) -> None:
        (tmp_path / "b.csv").write_text("x\n1\n", encoding="utf-8")
        (tmp_path / "a.csv").write_text("y\n2\n", encoding="utf-8")
        (tmp_path / "notes.bin").write_bytes(b"\x00")
        profiles = profile_directory(tmp_path)
        assert [Path(p.path).name for p in profiles] == ["a.csv", "b.csv"]

    def test_missing_directory_is_empty_not_an_error(self, tmp_path: Path) -> None:
        assert profile_directory(tmp_path / "absent") == []

    def test_example_inputs_expose_the_planted_traps(self, example_input_dir: Path) -> None:
        profiles = {Path(p.path).name: p for p in profile_directory(example_input_dir)}
        customers = {c.name: c for c in profiles["customers.csv"].columns}
        assert customers["age"].null_count > 0, "null ages are the filter trap"
        assert customers["country"].null_count > 0, "null countries are the groupby trap"

        orders = {c.name: c for c in profiles["orders.csv"].columns}
        assert orders["customer_id"].distinct_count < profiles["customers.csv"].row_count, (
            "some customers must have no orders, or the left-join null trap is not exercised"
        )
