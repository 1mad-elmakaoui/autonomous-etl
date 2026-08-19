"""Sandboxed execution of untrusted generated code.

What this layer guarantees, stated precisely because an oversold boundary is
worse than an understood one:

* Process isolation. The code runs in a separate interpreter, so it cannot
  corrupt the orchestrator or take down the worker with `sys.exit`.
* A scrubbed environment. The child gets an explicit allowlist.
  `ETLM_LLM_API_KEY`, `ETLM_GITHUB_TOKEN` and every other secret are not in it.
* Resource limits on CPU, address space, file size and open files, applied by
  the child itself.
* A wall-clock timeout enforced by the parent, killing the process group so a
  forked child cannot outlive it.
* A neutral working directory, so a relative path lands in a scratch dir rather
  than in the repository.

Not guaranteed: network egress. Blocking it needs a network namespace, which
needs privileges this process does not have. That belongs at the container
boundary (see the NetworkPolicy in `k8s/`), and the gate's import allowlist is
what keeps a socket out of the code to begin with.

The `SandboxRunner` protocol lets a container-backed implementation replace this
one without touching callers.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from etl_migrator.observability import get_logger

log = get_logger(__name__)

BOOTSTRAP_MODULE = "etl_migrator.sandbox._bootstrap"

#: Environment variables the child may see. Everything else is dropped — most
#: importantly every ETLM_* secret the worker was configured with.
ENV_ALLOWLIST: frozenset[str] = frozenset(
    {
        "PATH",
        "LANG",
        "LC_ALL",
        "TZ",
        "JAVA_HOME",
        "SPARK_HOME",
        "SPARK_LOCAL_IP",
        "PYSPARK_PYTHON",
        "PYSPARK_DRIVER_PYTHON",
        "HADOOP_HOME",
    }
)

#: Never forwarded, even if something adds them to the allowlist by accident.
ENV_DENYLIST_PREFIXES: tuple[str, ...] = ("ETLM_", "AWS_", "GITHUB_", "OPENAI_", "ANTHROPIC_")


@dataclass(frozen=True)
class SandboxLimits:
    """Resource ceilings for one sandboxed run.

    `memory_bytes` and `processes` are deliberately optional: the JVM reserves a
    large virtual address space and runs many threads, and on Linux threads
    count against RLIMIT_NPROC. Applying either to a Spark run does not harden
    it, it just makes it fail to start. So the Spark profile leaves them unset
    and bounds memory through Spark's own driver/executor settings instead.
    """

    wall_clock_seconds: int = 600
    cpu_seconds: int | None = 600
    memory_bytes: int | None = 4 * 1024**3
    file_size_bytes: int | None = 2 * 1024**3
    open_files: int | None = 4096
    processes: int | None = 256

    @classmethod
    def for_pandas(cls) -> SandboxLimits:
        return cls()

    @classmethod
    def for_spark(cls, wall_clock_seconds: int = 900) -> SandboxLimits:
        return cls(
            wall_clock_seconds=wall_clock_seconds,
            cpu_seconds=None,
            memory_bytes=None,
            file_size_bytes=8 * 1024**3,
            open_files=8192,
            processes=None,
        )

    def as_payload(self) -> dict[str, int]:
        return {
            key: value
            for key, value in (
                ("cpu_seconds", self.cpu_seconds),
                ("memory_bytes", self.memory_bytes),
                ("file_size_bytes", self.file_size_bytes),
                ("open_files", self.open_files),
                ("processes", self.processes),
            )
            if value is not None
        }


@dataclass
class SandboxResult:
    """Everything observed about a sandboxed run."""

    ok: bool
    duration_seconds: float
    exit_code: int | None = None
    timed_out: bool = False
    stdout: str = ""
    stderr: str = ""
    error: str | None = None
    traceback: str | None = None
    metrics: dict[str, str] = field(default_factory=dict)
    limits_applied: list[str] = field(default_factory=list)

    def tail(self, text: str, limit: int = 4000) -> str:
        return text if len(text) <= limit else "..." + text[-limit:]


class SandboxRunner(Protocol):
    """Substitutable execution backend."""

    def run(
        self,
        *,
        kind: str,
        module_path: Path,
        input_dir: Path,
        output_dir: Path,
        limits: SandboxLimits,
        extra: dict[str, Any] | None = None,
    ) -> SandboxResult: ...


def build_child_env() -> dict[str, str]:
    """Allowlisted environment for the child process."""
    env = {
        key: value
        for key, value in os.environ.items()
        if key in ENV_ALLOWLIST and not key.startswith(ENV_DENYLIST_PREFIXES)
    }
    env.setdefault("PATH", "/usr/local/bin:/usr/bin:/bin")
    # Keep the child on the same interpreter and importable package set as the
    # worker without inheriting anything else.
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p and Path(p).is_dir())
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    return env


class SubprocessSandbox:
    """Runs generated code in a separate interpreter with limits and a clean env."""

    def __init__(self, python_executable: str | None = None) -> None:
        self.python = python_executable or sys.executable

    def run(
        self,
        *,
        kind: str,
        module_path: Path,
        input_dir: Path,
        output_dir: Path,
        limits: SandboxLimits,
        extra: dict[str, Any] | None = None,
    ) -> SandboxResult:
        scratch = Path(tempfile.mkdtemp(prefix="etlm-sandbox-"))
        config_path = scratch / "config.json"
        result_path = scratch / "result.json"

        config: dict[str, Any] = {
            "kind": kind,
            "module_path": str(module_path.resolve()),
            "input_dir": str(input_dir.resolve()),
            "output_dir": str(output_dir.resolve()),
            "result_path": str(result_path),
            "limits": limits.as_payload(),
            **(extra or {}),
        }
        config_path.write_text(json.dumps(config), encoding="utf-8")

        log.info(
            "sandbox.start",
            kind=kind,
            module=module_path.name,
            wall_clock_s=limits.wall_clock_seconds,
        )
        started = time.perf_counter()
        timed_out = False
        stdout = stderr = ""
        exit_code: int | None = None

        try:
            process = subprocess.Popen(
                [self.python, "-m", BOOTSTRAP_MODULE, str(config_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=build_child_env(),
                cwd=scratch,  # neutral cwd: relative paths cannot reach the repo
                start_new_session=True,  # own process group, so the kill is total
                text=True,
            )
            try:
                stdout, stderr = process.communicate(timeout=limits.wall_clock_seconds)
                exit_code = process.returncode
            except subprocess.TimeoutExpired:
                timed_out = True
                self._terminate_group(process)
                stdout, stderr = process.communicate()
                exit_code = process.returncode

            duration = time.perf_counter() - started
            payload = self._read_result(result_path)

            if timed_out:
                result = SandboxResult(
                    ok=False,
                    duration_seconds=duration,
                    exit_code=exit_code,
                    timed_out=True,
                    error=f"exceeded the {limits.wall_clock_seconds}s wall-clock limit",
                )
            elif payload is None:
                # No result file means the child died before it could report —
                # OOM kill, segfault, or a hard rlimit. stderr is the only clue.
                result = SandboxResult(
                    ok=False,
                    duration_seconds=duration,
                    exit_code=exit_code,
                    error=(
                        f"sandboxed process exited with code {exit_code} without writing a "
                        "result; it was most likely killed by a resource limit"
                    ),
                )
            else:
                result = SandboxResult(
                    ok=bool(payload.get("ok")),
                    duration_seconds=float(payload.get("duration_seconds", duration)),
                    exit_code=exit_code,
                    error=payload.get("error"),
                    traceback=payload.get("traceback"),
                    metrics={k: str(v) for k, v in (payload.get("metrics") or {}).items()},
                    limits_applied=list(payload.get("limits_applied") or []),
                )

            result.stdout = result.tail(stdout or "")
            result.stderr = result.tail(stderr or "")
            log.info(
                "sandbox.done",
                kind=kind,
                ok=result.ok,
                duration_s=round(result.duration_seconds, 3),
                timed_out=result.timed_out,
            )
            return result
        finally:
            shutil.rmtree(scratch, ignore_errors=True)

    @staticmethod
    def _terminate_group(process: subprocess.Popen[str]) -> None:
        """Kill the whole process group.

        Spark forks executors; terminating only the parent leaves them running
        and holding the output directory.
        """
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(os.getpgid(process.pid), sig)
            except (ProcessLookupError, PermissionError):  # pragma: no cover
                return
            try:
                process.wait(timeout=5)
                return
            except subprocess.TimeoutExpired:
                continue

    @staticmethod
    def _read_result(path: Path) -> dict[str, Any] | None:
        if not path.is_file():
            return None
        try:
            data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:  # pragma: no cover - truncated write
            return None
        return data
