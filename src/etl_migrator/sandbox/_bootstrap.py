"""Child-process entrypoint for running untrusted generated code.

Executed as `python -m etl_migrator.sandbox._bootstrap <config.json>` in a fresh
process with a scrubbed environment. Never imported by the parent.

It applies its own resource limits rather than relying on `preexec_fn`, which is
documented as unsafe in a process that has threads — and the parent may well
have them.

The result is written to a JSON file rather than parsed out of stdout, because
the generated code is free to print whatever it likes to stdout and we must not
let it forge a verdict.
"""

from __future__ import annotations

import json
import resource
import sys
import time
import traceback
from pathlib import Path
from typing import Any


def apply_limits(limits: dict[str, int]) -> list[str]:
    """Apply rlimits, returning the ones that were actually set.

    Not every limit is safe for every engine, so the caller decides which to
    send; this only reports what landed.
    """
    applied: list[str] = []
    mapping = [
        ("cpu_seconds", resource.RLIMIT_CPU),
        ("memory_bytes", resource.RLIMIT_AS),
        ("file_size_bytes", resource.RLIMIT_FSIZE),
        ("open_files", resource.RLIMIT_NOFILE),
        ("processes", resource.RLIMIT_NPROC),
    ]
    for key, which in mapping:
        value = limits.get(key)
        if not value:
            continue
        try:
            _, hard = resource.getrlimit(which)
            ceiling = value if hard == resource.RLIM_INFINITY else min(value, hard)
            resource.setrlimit(which, (ceiling, hard))
            applied.append(f"{key}={ceiling}")
        except (ValueError, OSError) as exc:  # pragma: no cover - platform dependent
            applied.append(f"{key}=unset({exc})")
    return applied


def load_module(module_path: str, name: str) -> Any:
    """Import a module by path.

    The static gate has already established that importing it does nothing but
    define names — that check is what makes this line safe to write.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def run_legacy(cfg: dict[str, Any]) -> dict[str, str]:
    module = load_module(cfg["module_path"], "etlm_sandboxed_legacy")
    entrypoint = getattr(module, cfg.get("entrypoint", "run"))
    entrypoint(cfg["input_dir"], cfg["output_dir"])
    return {"engine": "pandas"}


def run_spark(cfg: dict[str, Any]) -> dict[str, str]:
    from pyspark.sql import SparkSession

    builder = SparkSession.builder.master(cfg.get("master", "local[2]")).appName(
        cfg.get("app_name", "etlm-sandbox")
    )
    for key, value in (cfg.get("spark_conf") or {}).items():
        builder = builder.config(key, str(value))
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel(cfg.get("log_level", "ERROR"))

    try:
        module = load_module(cfg["module_path"], "etlm_sandboxed_spark")
        entrypoint = getattr(module, cfg.get("entrypoint", "run"))
        entrypoint(spark, cfg["input_dir"], cfg["output_dir"])
        conf = spark.sparkContext.getConf()
        return {
            "engine": "spark",
            "spark_version": spark.version,
            "shuffle_partitions": str(spark.conf.get("spark.sql.shuffle.partitions", "?")),
            "adaptive_enabled": str(spark.conf.get("spark.sql.adaptive.enabled", "?")),
            "default_parallelism": str(spark.sparkContext.defaultParallelism),
            "app_id": str(conf.get("spark.app.id", "?")),
            **_job_metrics(spark),
        }
    finally:
        spark.stop()


def _job_metrics(spark: Any) -> dict[str, str]:
    """Job, stage and task counts, read from Spark's own status tracker.

    Deliberately limited to what PySpark exposes reliably after a run. Shuffle
    byte counts need a JVM-side listener; rather than estimate them and present
    the estimate as measured, they are simply absent.
    """
    try:
        tracker = spark.sparkContext.statusTracker()
        job_ids = list(tracker.getJobIdsForGroup())
        stage_ids: list[int] = []
        for job_id in job_ids:
            info = tracker.getJobInfo(job_id)
            if info is not None:
                stage_ids.extend(info.stageIds)
        tasks = 0
        for stage_id in set(stage_ids):
            stage = tracker.getStageInfo(stage_id)
            if stage is not None:
                tasks += int(stage.numTasks)
        return {
            "jobs": str(len(job_ids)),
            "stages": str(len(set(stage_ids))),
            "tasks": str(tasks),
        }
    except Exception as exc:
        return {"metrics_error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    config_path = Path(sys.argv[1])
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    result_path = Path(cfg["result_path"])

    applied = apply_limits(cfg.get("limits") or {})
    Path(cfg["output_dir"]).mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    payload: dict[str, Any] = {"ok": False, "limits_applied": applied}
    try:
        runner = {"legacy": run_legacy, "spark": run_spark}[cfg["kind"]]
        payload["metrics"] = runner(cfg)
        payload["ok"] = True
    except BaseException as exc:
        payload["error"] = f"{type(exc).__name__}: {exc}"
        payload["traceback"] = traceback.format_exc()[-4000:]
    finally:
        payload["duration_seconds"] = round(time.perf_counter() - started, 4)
        result_path.write_text(json.dumps(payload), encoding="utf-8")

    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
