"""Tests for the static gate.

This is a security boundary, so the tests are adversarial: each one is an
attempt to get something past the gate that must not get past it.
"""

from __future__ import annotations

import pytest

from etl_migrator.domain.enums import Severity
from etl_migrator.tools.code_gate import GateOptions, analyze_generated_code

CLEAN = '''\
"""ok"""
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

THRESHOLD = 18


def run(spark: SparkSession, input_dir: str, output_dir: str) -> None:
    df = spark.read.csv(f"{input_dir}/a.csv", header=True)
    df.filter(F.col("age") >= THRESHOLD).write.mode("overwrite").csv(f"{output_dir}/out")
'''


def codes(source: str, options: GateOptions | None = None) -> set[str]:
    return {f.code for f in analyze_generated_code(source, options).findings}


class TestAcceptsValidCode:
    def test_clean_module_passes_with_no_findings(self) -> None:
        report = analyze_generated_code(CLEAN)
        assert report.passed
        assert report.findings == []


class TestRejectsDangerousCode:
    @pytest.mark.parametrize(
        ("snippet", "expected"),
        [
            ("import os\n", "GATE002"),
            ("import subprocess\n", "GATE002"),
            ("import requests\n", "GATE002"),
            ("from . import helper\n", "GATE002"),
        ],
    )
    def test_import_allowlist(self, snippet: str, expected: str) -> None:
        assert expected in codes(CLEAN.replace("THRESHOLD = 18", snippet + "THRESHOLD = 18"))

    @pytest.mark.parametrize("call", ["eval('1')", "exec('x=1')", "open('/etc/passwd')",
                                      "__import__('os')", "compile('1', '<s>', 'eval')"])
    def test_dynamic_execution_is_rejected(self, call: str) -> None:
        source = CLEAN.replace("    df = spark", f"    _ = {call}\n    df = spark")
        assert "GATE003" in codes(source)

    def test_sandbox_escape_attributes_are_rejected(self) -> None:
        source = CLEAN.replace("    df = spark", "    _ = ().__class__.__bases__\n    df = spark")
        assert "GATE007" in codes(source)

    def test_module_level_work_is_rejected(self) -> None:
        """Importing a generated module must have no effect; otherwise the
        sandbox boundary is meaningless."""
        source = CLEAN.replace("THRESHOLD = 18", "THRESHOLD = 18\nprint('side effect')")
        assert "GATE011" in codes(source)

    def test_dunder_main_block_is_allowed(self) -> None:
        source = CLEAN + '\n\nif __name__ == "__main__":\n    pass\n'
        assert "GATE011" not in codes(source)


class TestEnforcesRunnerContract:
    def test_missing_entrypoint(self) -> None:
        source = CLEAN.replace("def run(spark: SparkSession, input_dir: str, output_dir: str)",
                               "def main(spark, input_dir, output_dir)")
        assert "GATE004" in codes(source)

    def test_wrong_signature(self) -> None:
        source = CLEAN.replace("def run(spark: SparkSession, input_dir: str, output_dir: str)",
                               "def run(session, inp, out)")
        assert "GATE004" in codes(source)

    def test_self_created_session_is_rejected(self) -> None:
        source = CLEAN.replace(
            "    df = spark",
            "    spark = SparkSession.builder.getOrCreate()\n    df = spark",
        )
        assert "GATE008" in codes(source)

    def test_non_spark_module_is_rejected(self) -> None:
        source = 'def run(spark, input_dir, output_dir):\n    return None\n'
        assert "GATE010" in codes(source)


class TestScaleAndQuality:
    @pytest.mark.parametrize("method", ["collect", "toPandas", "toLocalIterator"])
    def test_driver_collecting_calls_are_errors(self, method: str) -> None:
        source = CLEAN.replace("    df = spark", "    df = spark")
        source = source.replace('.csv(f"{output_dir}/out")', f".{method}()")
        report = analyze_generated_code(source)
        assert "GATE005" in {f.code for f in report.findings}
        assert not report.passed

    def test_driver_collect_can_be_permitted_explicitly(self) -> None:
        source = CLEAN.replace('.csv(f"{output_dir}/out")', ".collect()")
        assert "GATE005" not in codes(source, GateOptions(allow_driver_collect=True))

    def test_udf_is_a_warning_not_a_failure(self) -> None:
        source = CLEAN.replace(
            "from pyspark.sql import functions as F",
            "from pyspark.sql import functions as F\nfrom pyspark.sql.functions import udf",
        ).replace("    df = spark", "    fn = udf(lambda x: x)\n    df = spark")
        report = analyze_generated_code(source)
        udf_findings = [f for f in report.findings if f.code == "GATE006"]
        assert udf_findings and udf_findings[0].severity is Severity.WARNING
        assert report.passed, "a UDF should be discouraged, not fatal"

    def test_rdd_access_is_a_warning(self) -> None:
        source = CLEAN.replace("    df.filter", "    _ = df.rdd\n    df.filter")
        assert "GATE009" in codes(source)


class TestReporting:
    def test_syntax_error_is_reported_not_raised(self) -> None:
        report = analyze_generated_code("def run(:\n")
        assert not report.passed
        assert report.findings[0].code == "GATE001"

    def test_findings_carry_line_numbers(self) -> None:
        source = CLEAN.replace("THRESHOLD = 18", "import os\nTHRESHOLD = 18")
        report = analyze_generated_code(source)
        finding = next(f for f in report.findings if f.code == "GATE002")
        assert finding.line == 5
        assert "line 5" in finding.render()
