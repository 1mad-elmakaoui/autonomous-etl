"""The test that actually matters: does the generated PySpark produce the same
answer as the legacy pandas pipeline?

Everything else in the suite checks that the machinery works. This checks that
the machinery was worth building. It runs the full generation pipeline, executes
both the reference and the candidate, and diffs the outputs — a hand-rolled
preview of the Data Validation agent.

Marked `spark` because it needs a JVM:  pytest -m "not spark"  skips it.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

from etl_migrator.config import Settings
from etl_migrator.llm.factory import ScriptedModelClientFactory
from etl_migrator.pipeline.local import LocalMigrationPipeline, MigrationRequest

pytestmark = pytest.mark.spark

pyspark = pytest.importorskip("pyspark", reason="pyspark extra not installed")
pandas = pytest.importorskip("pandas")


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def spark():
    from pyspark.sql import SparkSession

    session = (
        SparkSession.builder.master("local[2]")
        .appName("etlm-equivalence")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()


@pytest.fixture(scope="module")
def outputs(request: pytest.FixtureRequest, spark, tmp_path_factory) -> dict[str, Any]:
    """Generate the code, then execute both pipelines against the same input."""
    import asyncio

    import pandas as pd

    root = Path(__file__).resolve().parents[1]
    example = root / "examples" / "customer_pipeline"
    input_dir = example / "input"
    if not (input_dir / "customers.csv").is_file():
        pytest.skip("example data missing; run examples/customer_pipeline/generate_data.py")

    tmp = tmp_path_factory.mktemp("equivalence")
    settings = Settings(workspace_dir=tmp / "workspace", log_format="json")
    factory = ScriptedModelClientFactory.from_fixture(
        root / "fixtures" / "llm" / "customer_pipeline.json"
    )
    record = asyncio.run(
        LocalMigrationPipeline(settings, factory, run_validation=False).run(
            MigrationRequest(example / "legacy_pipeline.py", input_dir)
        )
    )
    assert record.codegen is not None and record.artifact_dir is not None

    legacy_out = tmp / "legacy"
    spark_out = tmp / "spark"
    legacy_out.mkdir()
    spark_out.mkdir()

    _load_module(example / "legacy_pipeline.py", "etlm_legacy_ref").run(
        str(input_dir), str(legacy_out)
    )
    reference = pd.read_csv(legacy_out / "revenue_by_country.csv")

    generated_path = Path(record.artifact_dir) / record.codegen.code.filename
    _load_module(generated_path, "etlm_generated").run(spark, str(input_dir), str(spark_out))
    parts = sorted((spark_out / "revenue_by_country").glob("*.csv"))
    candidate = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)

    return {
        "reference": reference.sort_values("country").reset_index(drop=True),
        "candidate": candidate.sort_values("country").reset_index(drop=True),
        "plan": record.plan,
    }


class TestGeneratedPipelineIsEquivalent:
    def test_it_actually_executes(self, outputs: dict[str, Any]) -> None:
        assert not outputs["candidate"].empty

    def test_schema_matches(self, outputs: dict[str, Any]) -> None:
        assert list(outputs["reference"].columns) == list(outputs["candidate"].columns)

    def test_row_count_matches(self, outputs: dict[str, Any]) -> None:
        """The load-bearing assertion. A naive translation emits one extra row —
        a phantom NULL country group — because pandas groupby drops null keys
        and Spark groupBy does not."""
        assert len(outputs["reference"]) == len(outputs["candidate"])

    def test_no_phantom_null_group(self, outputs: dict[str, Any]) -> None:
        assert outputs["candidate"]["country"].isna().sum() == 0

    def test_group_keys_match(self, outputs: dict[str, Any]) -> None:
        assert (
            outputs["reference"]["country"].tolist() == outputs["candidate"]["country"].tolist()
        )

    def test_numeric_values_match_within_the_declared_tolerance(
        self, outputs: dict[str, Any]
    ) -> None:
        tolerance = outputs["plan"].validation_plan.numeric_tolerance
        reference = outputs["reference"]["revenue"]
        candidate = outputs["candidate"]["revenue"]
        relative = (reference - candidate).abs() / reference.abs().clip(lower=1e-12)
        assert relative.max() < tolerance, f"max relative difference {relative.max():.3e}"

    def test_null_counts_match(self, outputs: dict[str, Any]) -> None:
        assert (
            outputs["reference"]["revenue"].isna().sum()
            == outputs["candidate"]["revenue"].isna().sum()
        )


class TestTheTrapIsReal:
    def test_naive_translation_would_disagree(self, spark, example_input_dir: Path) -> None:
        """Proves the null-key mitigation is load-bearing rather than decorative.

        Without the `dropna` emulation the Spark output gains a NULL country row
        carrying real revenue — a silent, plausible-looking wrong answer.
        """
        from pyspark.sql import functions as F

        customers = spark.read.csv(
            f"{example_input_dir}/customers.csv", header=True, inferSchema=True
        ).filter(F.col("age") >= 18)
        orders = spark.read.csv(
            f"{example_input_dir}/orders.csv", header=True, inferSchema=True
        ).withColumn("revenue", F.col("quantity") * F.col("price"))

        naive = (
            customers.join(orders, on="customer_id", how="left")
            .groupBy("country")
            .agg(F.sum("revenue").alias("revenue"))
        )
        null_groups = naive.filter(F.col("country").isNull())
        assert null_groups.count() == 1, "the null-country trap is no longer exercised"
