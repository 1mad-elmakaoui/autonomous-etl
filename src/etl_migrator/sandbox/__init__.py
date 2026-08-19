"""Sandboxed execution of untrusted generated code.

See `runner.py` for exactly what this layer guarantees and — just as important
— what it does not.
"""

from etl_migrator.sandbox.execute import (
    BASE_SPARK_CONF,
    run_legacy_pipeline,
    run_spark_pipeline,
    spark_conf_from,
)
from etl_migrator.sandbox.runner import (
    ENV_ALLOWLIST,
    ENV_DENYLIST_PREFIXES,
    SandboxLimits,
    SandboxResult,
    SandboxRunner,
    SubprocessSandbox,
    build_child_env,
)

__all__ = [
    "BASE_SPARK_CONF",
    "ENV_ALLOWLIST",
    "ENV_DENYLIST_PREFIXES",
    "SandboxLimits",
    "SandboxResult",
    "SandboxRunner",
    "SubprocessSandbox",
    "build_child_env",
    "run_legacy_pipeline",
    "run_spark_pipeline",
    "spark_conf_from",
]
