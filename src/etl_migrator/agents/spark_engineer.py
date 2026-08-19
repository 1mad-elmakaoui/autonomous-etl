"""Spark Engineer agent: `MigrationPlan` -> executable PySpark module.

This agent has a genuine closed loop inside a single invocation:

    draft code -> call check_spark_code() -> read real findings -> revise -> re-check

`check_spark_code` is the same `analyze_generated_code` gate the orchestrator
runs afterwards, so the agent is being judged by the tool it can query rather
than by its own opinion. The number of iterations it needed is recorded on
`CodeGenResult.gate_iterations` and is a useful quality signal in its own right:
an agent that needs five attempts on a simple pipeline is telling you something.

The gate result is re-verified by the caller. The agent's claim that the code
passed is never trusted.
"""

from __future__ import annotations

import json
from collections.abc import Callable

from autogen_core.models import ChatCompletionClient

from etl_migrator.agents.base import StructuredAgent
from etl_migrator.domain.artifacts import GeneratedCode
from etl_migrator.domain.plan import MigrationPlan
from etl_migrator.domain.spec import MigrationSpec
from etl_migrator.tools.code_gate import (
    ALLOWED_IMPORT_ROOTS,
    REQUIRED_ENTRYPOINT,
    REQUIRED_PARAMETERS,
    GateOptions,
    analyze_generated_code,
)

SYSTEM_MESSAGE = f"""\
You are a Spark Engineer Agent. You implement an approved migration plan as a
single, executable PySpark module.

Hard contract — the module is rejected automatically if it violates any of this:

* It defines exactly one entrypoint:
      def {REQUIRED_ENTRYPOINT}({", ".join(REQUIRED_PARAMETERS)}) -> None
  `spark` is an already-configured SparkSession supplied by the runner.
  `input_dir` and `output_dir` are directory paths as strings.
* It never creates a SparkSession. Configuration belongs to the runner and the
  optimizer, not to the pipeline.
* Importing the module does nothing. No work at module level, only imports,
  constants and function definitions.
* It imports only from: {", ".join(sorted(ALLOWED_IMPORT_ROOTS))}.
* It never calls eval, exec, compile, open, __import__, or touches the
  filesystem except through the Spark reader/writer.
* It never calls .collect() or .toPandas(). Those load the whole dataset into
  the driver.

Engineering rules:

* DataFrame API and pyspark.sql.functions only. Write a Python UDF only if no
  built-in expression can express the logic, and say so in `notes`.
* Implement every plan step, in order. Implement the declared mitigation for
  every semantic difference — those are not suggestions, each one is checked by
  a validation check after execution.
* Steps whose spark_construct is "(none ...)" produce no code. pandas index
  bookkeeping has no Spark equivalent; do not invent one.
* Alias every aggregate: F.sum("x").alias("x"), never a bare F.sum("x").
* Broadcast only the datasets the plan lists in execution_strategy.
* Read with explicit schema-affecting options (header, inferSchema) so column
  types are not accidentally strings.
* Write with the mode the specification's sink declares.

Process:

1. Draft the module.
2. Call `check_spark_code` with the complete source. It runs the same static
   gate that will judge you afterwards, and returns real findings.
3. If there are ERROR findings, fix them and call `check_spark_code` again.
   Address WARNING findings too unless you can justify them in `notes`.
4. Only once the gate passes, return the structured result. `content` must be
   the exact source you last validated.

Put every non-obvious decision in `notes` — that text goes into the pull
request body a human will review.
"""


class SparkEngineerAgent(StructuredAgent[GeneratedCode]):
    key = "spark_engineer"
    description = "Writes executable PySpark implementing an approved migration plan."

    def __init__(
        self,
        model_client: ChatCompletionClient,
        *,
        plan: MigrationPlan,
        spec: MigrationSpec,
        gate_options: GateOptions | None = None,
        max_tool_iterations: int = 8,
    ) -> None:
        self.plan = plan
        self.spec = spec
        self.gate_options = gate_options or GateOptions()
        #: Counts how many times the agent submitted code to the gate.
        self.gate_calls = 0
        super().__init__(
            model_client,
            GeneratedCode,
            system_message=SYSTEM_MESSAGE,
            tools=[self._make_gate_tool(), self._make_step_tool()],
            max_tool_iterations=max_tool_iterations,
        )

    def _make_gate_tool(self) -> Callable[[str], str]:
        options = self.gate_options
        agent = self

        def check_spark_code(code: str) -> str:
            """Run the static gate on candidate PySpark source and return every finding
            with severity, code and line number. Call this before returning a result,
            and again after each fix.

            Args:
                code: the complete module source to validate.
            """
            agent.gate_calls += 1
            report = analyze_generated_code(code, options)
            verdict = "PASS" if report.passed else "FAIL"
            return f"gate: {verdict} (submission #{agent.gate_calls})\n{report.render()}"

        return check_spark_code

    def _make_step_tool(self) -> Callable[[str], str]:
        plan = self.plan
        spec = self.spec

        def get_plan_step(step_id: str) -> str:
            """Return one plan step in full, together with the transformation it
            implements: legacy expression, source lines, join keys, aggregations and
            every declared semantic difference with its required mitigation.

            Args:
                step_id: a step id such as 's4'.
            """
            for step in plan.steps:
                if step.id == step_id:
                    payload = {"step": step.model_dump(mode="json")}
                    try:
                        payload["transformation"] = spec.transformation(
                            step.transformation_id
                        ).model_dump(mode="json")
                    except KeyError:
                        payload["transformation"] = {
                            "error": f"spec has no {step.transformation_id}"
                        }
                    return json.dumps(payload, indent=2)
            return f"no step '{step_id}'. Available: {', '.join(s.id for s in plan.steps)}"

        return get_plan_step


def codegen_task(spec: MigrationSpec, plan: MigrationPlan, filename: str) -> str:
    return (
        f"Implement the approved migration plan as a PySpark module named `{filename}`.\n\n"
        f"=== SPECIFICATION ===\n{spec.model_dump_json(indent=2)}\n\n"
        f"=== PLAN ===\n{plan.model_dump_json(indent=2)}\n\n"
        "Draft the module, validate it with check_spark_code until the gate passes, "
        "then return the validated source."
    )
