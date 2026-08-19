"""AutoGen agents. Each one owns a contract, a toolset, and nothing else.

Discovery, Planner, Spark Engineer, Testing, Data Validation and Repair are
implemented.
Optimization and Deployment agents arrive in phases 5-6 and reuse
`StructuredAgent` unchanged.
"""

from etl_migrator.agents.base import AgentRun, StructuredAgent, ToolInvocation
from etl_migrator.agents.discovery import DiscoveryAgent, discovery_task
from etl_migrator.agents.planner import PlannerAgent, planning_task
from etl_migrator.agents.repair import RepairAgent, repair_task
from etl_migrator.agents.spark_engineer import SparkEngineerAgent, codegen_task
from etl_migrator.agents.testing import TestingAgent, testing_task
from etl_migrator.agents.validation import ValidationAgent, diagnosis_task

__all__ = [
    "AgentRun",
    "DiscoveryAgent",
    "PlannerAgent",
    "RepairAgent",
    "SparkEngineerAgent",
    "StructuredAgent",
    "TestingAgent",
    "ToolInvocation",
    "ValidationAgent",
    "codegen_task",
    "diagnosis_task",
    "discovery_task",
    "planning_task",
    "repair_task",
    "testing_task",
]
