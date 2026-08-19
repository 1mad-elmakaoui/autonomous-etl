"""Differ tests — the component that decides whether a migration is correct.

These need no Spark and no LLM: two DataFrames in, a verdict out. That is
exactly why the differ is written this way. The most important assertions here
are the ones proving it *fails* when it should, because a differ that says PASS
too easily is worse than no validation at all.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from etl_migrator.domain.enums import RiskCategory, TransformKind, ValidationStatus
from etl_migrator.domain.plan import MigrationPlan, PlanStep, ValidationPlan
from etl_migrator.tools.differ import (
    check_duplicate_counts,
    check_null_counts,
    check_row_count,
    check_schema,
    compare_outputs,
    dtype_family,
    load_dataset,
)

REFERENCE = pd.DataFrame(
    {"country": ["DE", "ES", "FR"], "revenue": [100.0, 200.0, 300.0]}
)


def write_csv(path: Path, frame: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def write_spark_style(directory: Path, frame: pd.DataFrame, parts: int = 2) -> Path:
    """Mimic a Spark output: a directory of part files plus a _SUCCESS marker."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "_SUCCESS").write_text("", encoding="utf-8")
    chunks = [frame.iloc[i::parts] for i in range(parts)]
    for index, chunk in enumerate(chunks):
        chunk.to_csv(directory / f"part-{index:05d}-abc.csv", index=False)
    return directory


def plan_for(
    keys: list[str], tolerance: float = 1e-6, checks: list[str] | None = None
) -> MigrationPlan:
    return MigrationPlan(
        summary="s",
        steps=[
            PlanStep(
                id="s1", transformation_id="t1", kind=TransformKind.AGGREGATE,
                legacy_construct="groupby", spark_construct="groupBy", rationale="r",
            )
        ],
        validation_plan=ValidationPlan(
            join_keys=keys,
            numeric_tolerance=tolerance,
            # `is None` rather than a falsy test: an explicitly empty check list
            # is a case worth exercising, not a request for the defaults.
            required_checks=(
                checks
                if checks is not None
                else [
                    "schema", "row_count", "null_counts", "numeric_tolerance",
                    "duplicate_counts", "aggregate_checksums",
                ]
            ),
        ),
    )


class TestLoading:
    def test_reads_a_single_file(self, tmp_path: Path) -> None:
        path = write_csv(tmp_path / "out.csv", REFERENCE)
        assert len(load_dataset(path)) == 3

    def test_reassembles_a_directory_of_part_files(self, tmp_path: Path) -> None:
        """A Spark output is many files; the differ must see one table."""
        directory = write_spark_style(tmp_path / "out", REFERENCE, parts=3)
        loaded = load_dataset(directory)
        assert len(loaded) == 3
        assert set(loaded["country"]) == {"DE", "ES", "FR"}

    def test_ignores_success_markers(self, tmp_path: Path) -> None:
        directory = write_spark_style(tmp_path / "out", REFERENCE)
        assert list(load_dataset(directory).columns) == ["country", "revenue"]

    def test_missing_output_is_reported_not_raised(self, tmp_path: Path) -> None:
        report = compare_outputs(
            migration_id="m",
            reference_path=tmp_path / "nope.csv",
            candidate_path=tmp_path / "also-nope",
            plan=plan_for(["country"]),
        )
        assert report.status is ValidationStatus.ERROR
        assert report.error


class TestDtypeFamilies:
    @pytest.mark.parametrize(
        ("dtype", "family"),
        [("int64", "integer"), ("int32", "integer"), ("float64", "float"),
         ("object", "string"), ("bool", "boolean")],
    )
    def test_families(self, dtype: str, family: str) -> None:
        assert dtype_family(dtype) == family

    def test_int_and_float_are_different_families(self) -> None:
        """The pandas null-promotion trap: int64 becomes float64 when a null
        appears. That is a real difference and must not be normalised away."""
        assert dtype_family("int64") != dtype_family("float64")


class TestIndividualChecks:
    def test_schema_flags_a_missing_column(self) -> None:
        result = check_schema(REFERENCE, REFERENCE[["country"]])
        assert not result.passed
        assert result.differences[0].column == "revenue"

    def test_schema_flags_a_synthesised_index_column(self) -> None:
        candidate = REFERENCE.assign(index=[0, 1, 2])
        result = check_schema(REFERENCE, candidate)
        assert not result.passed
        assert result.differences[0].category is RiskCategory.INDEX_SEMANTICS

    def test_schema_flags_int_to_float_promotion(self) -> None:
        reference = pd.DataFrame({"n": [1, 2, 3]})
        candidate = pd.DataFrame({"n": [1.0, 2.0, 3.0]})
        result = check_schema(reference, candidate)
        assert not result.passed
        assert result.differences[0].category is RiskCategory.TYPE_COERCION

    def test_schema_flags_column_reordering(self) -> None:
        result = check_schema(REFERENCE, REFERENCE[["revenue", "country"]])
        assert not result.passed
        assert "different order" in result.differences[0].detail

    def test_row_count_difference_names_the_likely_cause(self) -> None:
        extra = pd.concat([REFERENCE, pd.DataFrame({"country": [None], "revenue": [9.0]})])
        result = check_row_count(REFERENCE, extra)
        assert not result.passed
        assert result.differences[0].category is RiskCategory.NULL_SEMANTICS
        assert "groupby drops them" in result.differences[0].detail

    def test_null_counts_flag_the_all_null_aggregate_trap(self) -> None:
        reference = pd.DataFrame({"revenue": [0.0, 1.0]})
        candidate = pd.DataFrame({"revenue": [None, 1.0]})
        result = check_null_counts(reference, candidate)
        assert not result.passed
        assert result.differences[0].category is RiskCategory.NULL_SEMANTICS

    def test_duplicate_counts_flag_a_join_explosion(self) -> None:
        candidate = pd.concat([REFERENCE, REFERENCE.iloc[[0]]], ignore_index=True)
        result = check_duplicate_counts(REFERENCE, candidate, ["country"])
        assert not result.passed
        assert result.differences[0].category is RiskCategory.DUPLICATE_EXPLOSION


class TestEndToEndVerdict:
    def test_identical_outputs_pass(self, tmp_path: Path) -> None:
        reference = write_csv(tmp_path / "ref.csv", REFERENCE)
        candidate = write_spark_style(tmp_path / "cand", REFERENCE)
        report = compare_outputs(
            migration_id="m", reference_path=reference, candidate_path=candidate,
            plan=plan_for(["country"]),
        )
        assert report.status is ValidationStatus.PASS
        assert report.schema_match and report.row_count_match
        assert report.numeric_tolerance_passed
        assert report.differences == []

    def test_float_noise_within_tolerance_passes(self, tmp_path: Path) -> None:
        """Double arithmetic differs in the last ulp between NumPy and the JVM.
        Failing on that would make every migration fail forever."""
        noisy = REFERENCE.assign(revenue=REFERENCE["revenue"] * (1 + 1e-13))
        report = compare_outputs(
            migration_id="m",
            reference_path=write_csv(tmp_path / "ref.csv", REFERENCE),
            candidate_path=write_csv(tmp_path / "cand.csv", noisy),
            plan=plan_for(["country"], tolerance=1e-9),
        )
        assert report.status is ValidationStatus.PASS

    def test_difference_beyond_tolerance_fails(self, tmp_path: Path) -> None:
        wrong = REFERENCE.assign(revenue=[100.0, 200.0, 301.0])
        report = compare_outputs(
            migration_id="m",
            reference_path=write_csv(tmp_path / "ref.csv", REFERENCE),
            candidate_path=write_csv(tmp_path / "cand.csv", wrong),
            plan=plan_for(["country"]),
        )
        assert report.status is ValidationStatus.FAIL
        assert not report.numeric_tolerance_passed

    def test_the_phantom_null_group_is_caught(self, tmp_path: Path) -> None:
        """The headline failure mode: a naive port keeps null group keys, adding
        a plausible-looking row with real revenue in it."""
        phantom = pd.concat(
            [REFERENCE, pd.DataFrame({"country": [None], "revenue": [143967.81]})],
            ignore_index=True,
        )
        report = compare_outputs(
            migration_id="m",
            reference_path=write_csv(tmp_path / "ref.csv", REFERENCE),
            candidate_path=write_spark_style(tmp_path / "cand", phantom),
            plan=plan_for(["country"]),
        )
        assert report.status is ValidationStatus.FAIL
        assert not report.row_count_match
        failed = {c.name for c in report.failed_checks}
        assert {"row_count", "aggregate_checksums"} <= failed

    def test_row_order_alone_is_not_a_failure(self, tmp_path: Path) -> None:
        """Spark join does not preserve pandas row order. Reporting that as a
        difference would make every migration fail for a non-reason."""
        shuffled = REFERENCE.iloc[::-1].reset_index(drop=True)
        report = compare_outputs(
            migration_id="m",
            reference_path=write_csv(tmp_path / "ref.csv", REFERENCE),
            candidate_path=write_csv(tmp_path / "cand.csv", shuffled),
            plan=plan_for(["country"]),
        )
        assert report.status is ValidationStatus.PASS

    def test_missing_row_is_located_by_key(self, tmp_path: Path) -> None:
        report = compare_outputs(
            migration_id="m",
            reference_path=write_csv(tmp_path / "ref.csv", REFERENCE),
            candidate_path=write_csv(tmp_path / "cand.csv", REFERENCE.iloc[:2]),
            plan=plan_for(["country"]),
        )
        assert report.status is ValidationStatus.FAIL
        assert any("FR" in d.reference for d in report.differences)


class TestUnmeasurableIsNotPassing:
    def test_declared_keys_absent_from_the_output_is_an_error(self, tmp_path: Path) -> None:
        """A check that could not run must never read as a pass."""
        report = compare_outputs(
            migration_id="m",
            reference_path=write_csv(tmp_path / "ref.csv", REFERENCE),
            candidate_path=write_csv(tmp_path / "cand.csv", REFERENCE),
            plan=plan_for(["nonexistent_key"]),
        )
        assert report.status is ValidationStatus.ERROR
        skipped = next(c for c in report.checks if c.name == "numeric_tolerance")
        assert skipped.skipped

    def test_unknown_check_name_is_an_error(self, tmp_path: Path) -> None:
        """A plan demanding an assurance the differ cannot provide must not
        silently receive a pass for it."""
        report = compare_outputs(
            migration_id="m",
            reference_path=write_csv(tmp_path / "ref.csv", REFERENCE),
            candidate_path=write_csv(tmp_path / "cand.csv", REFERENCE),
            plan=plan_for(["country"], checks=["schema", "vibes"]),
        )
        assert report.status is ValidationStatus.ERROR
        assert any(c.name == "vibes" and c.skipped for c in report.checks)

    def test_no_checks_at_all_is_an_error_not_a_pass(self, tmp_path: Path) -> None:
        report = compare_outputs(
            migration_id="m",
            reference_path=write_csv(tmp_path / "ref.csv", REFERENCE),
            candidate_path=write_csv(tmp_path / "cand.csv", REFERENCE),
            plan=plan_for(["country"], checks=[]),
        )
        assert report.status is ValidationStatus.ERROR


class TestFallbackComparison:
    def test_compares_sorted_rows_when_no_keys_are_declared(self, tmp_path: Path) -> None:
        report = compare_outputs(
            migration_id="m",
            reference_path=write_csv(tmp_path / "ref.csv", REFERENCE),
            candidate_path=write_csv(tmp_path / "cand.csv", REFERENCE.iloc[::-1]),
            plan=plan_for([]),
        )
        assert report.status is ValidationStatus.PASS
        check = next(c for c in report.checks if c.name == "numeric_tolerance")
        assert "sorted rows" in check.detail

    def test_falls_back_when_keys_are_not_unique(self, tmp_path: Path) -> None:
        """Joining on a non-unique key multiplies rows and would report a huge
        difference that is an artefact of the comparison, not of the data."""
        duplicated = pd.DataFrame(
            {"country": ["FR", "FR"], "revenue": [1.0, 2.0]}
        )
        report = compare_outputs(
            migration_id="m",
            reference_path=write_csv(tmp_path / "ref.csv", duplicated),
            candidate_path=write_csv(tmp_path / "cand.csv", duplicated),
            plan=plan_for(["country"]),
        )
        check = next(c for c in report.checks if c.name == "numeric_tolerance")
        assert "keys not unique" in check.detail
        assert report.status is ValidationStatus.PASS
