"""Domain-model invariants, and validation of the recorded fixtures.

The fixture check matters because the fixtures are hand-authored data. If a
model gains a required field, CI must fail here rather than at the first live
migration.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from etl_migrator.config import LLMProvider, Settings
from etl_migrator.domain.artifacts import GeneratedCode, MigrationRecord
from etl_migrator.domain.delivery import ClaimAudit, DeliveryOutcome
from etl_migrator.domain.enums import (
    RiskCategory,
    RiskLevel,
    TransformKind,
    ValidationStatus,
)
from etl_migrator.domain.errors import ConfigurationError
from etl_migrator.domain.optimization import BenchmarkResult, OptimizationOutcome
from etl_migrator.domain.plan import MigrationPlan, PlanStep, SemanticDifference, ValidationPlan
from etl_migrator.domain.spec import JoinDetail, MigrationSpec, StrictModel, Transformation
from etl_migrator.domain.validation import (
    CheckResult,
    ValidationOutcome,
    ValidationReport,
)
from etl_migrator.knowledge.patterns import DEFAULT_CATALOGUE


class TestFixtureIntegrity:
    def test_discovery_fixture_validates(self, fixture_payload: dict) -> None:
        spec = MigrationSpec.model_validate(fixture_payload["discovery"][-1]["content"])
        assert len(spec.transformations) == 9

    def test_planner_fixture_validates(self, fixture_payload: dict) -> None:
        plan = MigrationPlan.model_validate(fixture_payload["planner"][-1]["content"])
        assert len(plan.steps) == 9

    def test_codegen_fixture_validates(self, fixture_payload: dict) -> None:
        assert GeneratedCode.model_validate(
            fixture_payload["spark_engineer"][-1]["content"]
        ).entrypoint == "run"

    def test_every_plan_step_references_a_real_transformation(
        self, fixture_payload: dict
    ) -> None:
        spec = MigrationSpec.model_validate(fixture_payload["discovery"][-1]["content"])
        plan = MigrationPlan.model_validate(fixture_payload["planner"][-1]["content"])
        known = {t.id for t in spec.transformations}
        assert {s.transformation_id for s in plan.steps} <= known

    def test_every_semantic_difference_names_a_known_check(
        self, fixture_payload: dict
    ) -> None:
        plan = MigrationPlan.model_validate(fixture_payload["planner"][-1]["content"])
        valid = {
            "schema", "row_count", "null_counts", "numeric_tolerance",
            "duplicate_counts", "aggregate_checksums", "column_statistics",
        }
        assert {d.validation_check for d in plan.all_semantic_differences} <= valid


class TestStrictness:
    def test_unknown_fields_are_rejected(self) -> None:
        """A hallucinated field must be a loud error, not a dropped instruction."""
        with pytest.raises(ValidationError, match="Extra inputs"):
            Transformation(
                id="t1", kind=TransformKind.FILTER, description="d", output="o",
                invented_field="surprise",
            )

    def test_transformation_ids_must_be_ordinal(self) -> None:
        with pytest.raises(ValidationError):
            Transformation(id="first", kind=TransformKind.FILTER, description="d", output="o")

    def test_duplicate_transformation_ids_rejected(self) -> None:
        common = {"kind": TransformKind.FILTER, "description": "d", "output": "o"}
        with pytest.raises(ValidationError, match="duplicate transformation ids"):
            MigrationSpec(
                source_path="p.py", source_language="python_pandas", summary="s", confidence=0.5,
                transformations=[
                    Transformation(id="t1", **common),
                    Transformation(id="t1", **common),
                ],
            )

    def test_join_key_arity_must_match(self) -> None:
        with pytest.raises(ValidationError, match="arity mismatch"):
            JoinDetail(left="a", right="b", how="inner", left_keys=["x"], right_keys=["y", "z"])

    def test_confidence_is_bounded(self) -> None:
        with pytest.raises(ValidationError):
            MigrationSpec(
                source_path="p.py", source_language="python_pandas", summary="s", confidence=1.5
            )


class TestJsonRoundTrip:
    """Every model must survive its own serialisation.

    `MigrationRecord` crosses a JSON boundary twice on the durable path —
    Temporal's pydantic data converter, and the artifacts written to disk and
    read back — so a model that cannot be re-loaded from its own dump is a
    production bug, not a tidiness issue.

    The trap is specific and was live in this repository until delivery tripped
    it. Pydantic *writes* `@computed_field` values on dump and *refuses* them on
    load, and `StrictModel` sets `extra="forbid"`. Every verdict here is a
    computed field, by design, so no caller can set one — which meant the
    models carrying the system's conclusions were exactly the ones that could
    not round-trip.
    """

    @pytest.mark.parametrize(
        "model",
        [
            BenchmarkResult(label="b", durations=[1.0, 2.0, 5.0]),
            OptimizationOutcome(),
            ValidationReport(
                migration_id="m", checks=[CheckResult(name="row_count", passed=True)]
            ),
            DeliveryOutcome(),
            ClaimAudit(),
        ],
        ids=lambda m: type(m).__name__,
    )
    def test_a_model_can_be_loaded_from_its_own_dump(self, model: StrictModel) -> None:
        restored = type(model).model_validate_json(model.model_dump_json())
        assert restored == model

    def test_computed_values_are_recomputed_not_trusted(self) -> None:
        """The values are dropped on load and derived again from the real fields.

        So a tampered dump cannot smuggle in a verdict: rewriting `status` to
        PASS in the JSON changes nothing, because it is recomputed from the
        checks.
        """
        report = ValidationReport(
            migration_id="m",
            checks=[CheckResult(name="row_count", passed=False, detail="4 != 5")],
        )
        assert report.status is ValidationStatus.FAIL

        tampered = json.loads(report.model_dump_json())
        tampered["status"] = "PASS"
        assert ValidationReport.model_validate(tampered).status is ValidationStatus.FAIL

    def test_forbid_still_rejects_a_genuinely_unknown_field(self) -> None:
        """Guard the fix.

        Dropping computed keys must not have widened the door — a hallucinated
        field is still the loud error `extra="forbid"` exists to produce.
        """
        with pytest.raises(ValidationError, match="Extra inputs"):
            BenchmarkResult.model_validate(
                {"label": "b", "durations": [1.0], "invented_field": 1}
            )

    def test_the_whole_record_round_trips(self) -> None:
        """The one that actually crosses the wire."""
        record = MigrationRecord(migration_id="m-1", source_path="legacy.py")
        record.optimization = OptimizationOutcome()
        record.delivery = DeliveryOutcome()
        record.validation = ValidationOutcome(
            report=ValidationReport(
                migration_id="m-1", checks=[CheckResult(name="schema", passed=True)]
            )
        )
        restored = MigrationRecord.model_validate_json(record.model_dump_json())
        assert restored == record


class TestDerivedBehaviour:
    def test_max_risk_is_the_highest_declared_risk(self, fixture_payload: dict) -> None:
        spec = MigrationSpec.model_validate(fixture_payload["discovery"][-1]["content"])
        assert spec.max_risk is RiskLevel.HIGH

    def test_max_risk_of_a_riskless_spec_is_low(self) -> None:
        spec = MigrationSpec(
            source_path="p.py", source_language="python_pandas", summary="s", confidence=0.9
        )
        assert spec.max_risk is RiskLevel.LOW

    def test_declared_differences_add_required_checks(self) -> None:
        plan = MigrationPlan(
            summary="s",
            validation_plan=ValidationPlan(required_checks=["schema"]),
            steps=[
                PlanStep(
                    id="s1", transformation_id="t1", kind=TransformKind.AGGREGATE,
                    legacy_construct="groupby", spark_construct="groupBy", rationale="r",
                    semantic_differences=[
                        SemanticDifference(
                            category=RiskCategory.NULL_SEMANTICS, description="d",
                            mitigation="m", validation_check="null_counts",
                        )
                    ],
                )
            ],
        )
        assert plan.effective_required_checks() == ["schema", "null_counts"]

    def test_generated_code_is_content_addressed(self) -> None:
        a = GeneratedCode(filename="x.py", content="print(1)")
        b = GeneratedCode(filename="y.py", content="print(1)")
        assert a.sha256 == b.sha256

    def test_record_risk_falls_back_to_the_spec(self, fixture_payload: dict) -> None:
        spec = MigrationSpec.model_validate(fixture_payload["discovery"][-1]["content"])
        record = MigrationRecord(migration_id="m", source_path="p.py", spec=spec)
        assert record.risk is RiskLevel.HIGH


class TestPatternCatalogue:
    def test_covers_the_kinds_the_example_needs(self) -> None:
        for kind in (TransformKind.JOIN, TransformKind.AGGREGATE, TransformKind.RESET_INDEX):
            assert DEFAULT_CATALOGUE.lookup(kind)

    def test_aggregate_pattern_warns_about_the_dropna_default(self) -> None:
        rendered = DEFAULT_CATALOGUE.render(TransformKind.AGGREGATE)
        assert "dropna" in rendered

    def test_unknown_kind_renders_actionable_guidance(self) -> None:
        assert "first principles" in DEFAULT_CATALOGUE.render(TransformKind.PIVOT)

    def test_seed_patterns_have_no_observed_outcomes_yet(self) -> None:
        """Honesty check: the catalogue must not imply empirical backing it lacks."""
        for kind in DEFAULT_CATALOGUE.kinds:
            for pattern in DEFAULT_CATALOGUE.lookup(TransformKind(kind)):
                assert pattern.success_rate is None
                assert "curated guidance" in pattern.render()


class TestSettings:
    def test_scripted_provider_needs_no_credentials(self) -> None:
        settings = Settings(llm_provider=LLMProvider.SCRIPTED)
        with pytest.raises(ConfigurationError, match="no credentials"):
            settings.require_llm_credentials()

    def test_live_provider_without_a_key_explains_itself(self) -> None:
        settings = Settings(llm_provider=LLMProvider.ANTHROPIC, llm_api_key=None)
        with pytest.raises(ConfigurationError, match="ETLM_LLM_API_KEY is required"):
            settings.require_llm_credentials()

    def test_secrets_are_not_printed_in_repr(self) -> None:
        settings = Settings(llm_provider=LLMProvider.OPENAI, llm_api_key="sk-super-secret")
        assert "sk-super-secret" not in repr(settings)
        assert settings.require_llm_credentials() == "sk-super-secret"


class TestBrokenScenarioFixture:
    """The repair scenario has to be a *realistic* failure to be worth anything."""

    @pytest.fixture(scope="class")
    @classmethod
    def broken(cls) -> dict:
        import json
        from pathlib import Path

        path = Path(__file__).resolve().parents[1] / "fixtures" / "llm"
        return json.loads((path / "customer_pipeline_broken.json").read_text())

    def test_the_flawed_code_passes_the_static_gate(self, broken: dict) -> None:
        """The whole point: a gate failure never reaches validation, so a repair
        scenario built on one would exercise nothing."""
        from etl_migrator.tools.code_gate import analyze_generated_code

        code = GeneratedCode.model_validate(broken["spark_engineer"][-1]["content"])
        assert analyze_generated_code(code.content).passed

    def test_every_repair_proposal_validates_and_is_gate_clean(self, broken: dict) -> None:
        from etl_migrator.domain.repair import RepairProposal
        from etl_migrator.tools.code_gate import analyze_generated_code

        proposals = [
            RepairProposal.model_validate(turn["content"])
            for script in broken["repair"]
            for turn in script["turns"]
            if "content" in turn
        ]
        assert len(proposals) == 3
        for proposal in proposals:
            assert analyze_generated_code(proposal.code.content).passed

    def test_the_scenario_requires_two_distinct_strategies(self, broken: dict) -> None:
        """One defect would let a single lucky fix end the loop, and the ledger's
        rejection path would never run."""
        from etl_migrator.domain.repair import RepairProposal

        signatures = {
            RepairProposal.model_validate(turn["content"]).strategy.signature
            for script in broken["repair"]
            for turn in script["turns"]
            if "content" in turn
        }
        assert len(signatures) == 2

    def test_the_final_proposal_matches_the_known_good_implementation(
        self, broken: dict, fixture_payload: dict
    ) -> None:
        from etl_migrator.domain.repair import RepairProposal

        final = RepairProposal.model_validate(broken["repair"][-1]["turns"][-1]["content"])
        good = GeneratedCode.model_validate(fixture_payload["spark_engineer"][-1]["content"])
        assert final.code.content == good.content
