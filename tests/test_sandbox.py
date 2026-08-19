"""Sandbox tests.

Generated code is untrusted input, so these are adversarial: each one is a way
the sandbox could leak, hang, or let something out. The secret-scrubbing test in
particular is the one that matters most — a worker holds an LLM API key and a
GitHub token, and generated code runs in that worker's process tree.

Everything here uses real subprocesses. Mocking the sandbox would test the mock.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from etl_migrator.sandbox.runner import (
    ENV_ALLOWLIST,
    SandboxLimits,
    SubprocessSandbox,
    build_child_env,
)

LEGACY_SOURCE = """\
import os


def run(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "out.csv"), "w") as fh:
        fh.write("a\\n1\\n")
"""


def write_module(tmp_path: Path, source: str, name: str = "pipeline.py") -> Path:
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    return path


class TestEnvironmentScrubbing:
    def test_secrets_never_reach_the_child(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The threat is concrete: generated code reading the worker's API key
        and posting it somewhere. It cannot read what is not in its environment."""
        monkeypatch.setenv("ETLM_LLM_API_KEY", "sk-super-secret")
        monkeypatch.setenv("ETLM_GITHUB_TOKEN", "ghp_leak_me")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "aws-secret")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret")

        env = build_child_env()

        assert "ETLM_LLM_API_KEY" not in env
        assert "ETLM_GITHUB_TOKEN" not in env
        assert "AWS_SECRET_ACCESS_KEY" not in env
        assert "ANTHROPIC_API_KEY" not in env
        assert not any("secret" in value.lower() for value in env.values())

    def test_the_child_environment_is_an_allowlist_not_a_denylist(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A denylist only blocks the secrets someone remembered to name."""
        monkeypatch.setenv("SOME_FUTURE_CREDENTIAL", "oops")
        env = build_child_env()
        assert "SOME_FUTURE_CREDENTIAL" not in env
        assert set(env) <= ENV_ALLOWLIST | {
            "PATH", "PYTHONPATH", "PYTHONDONTWRITEBYTECODE", "PYTHONUNBUFFERED"
        }

    def test_the_child_can_still_find_python_and_java(self) -> None:
        env = build_child_env()
        assert env["PATH"]
        assert env["PYTHONPATH"]


class TestExecution:
    def test_runs_a_legacy_pipeline_and_produces_output(self, tmp_path: Path) -> None:
        module = write_module(tmp_path, LEGACY_SOURCE)
        result = SubprocessSandbox().run(
            kind="legacy",
            module_path=module,
            input_dir=tmp_path,
            output_dir=tmp_path / "out",
            limits=SandboxLimits.for_pandas(),
        )
        assert result.ok, result.error
        assert (tmp_path / "out" / "out.csv").is_file()
        assert result.metrics["engine"] == "pandas"

    def test_applies_resource_limits(self, tmp_path: Path) -> None:
        result = SubprocessSandbox().run(
            kind="legacy",
            module_path=write_module(tmp_path, LEGACY_SOURCE),
            input_dir=tmp_path,
            output_dir=tmp_path / "out",
            limits=SandboxLimits.for_pandas(),
        )
        assert any(limit.startswith("cpu_seconds=") for limit in result.limits_applied)
        assert any(limit.startswith("memory_bytes=") for limit in result.limits_applied)

    def test_a_crash_is_reported_not_raised(self, tmp_path: Path) -> None:
        module = write_module(
            tmp_path, "def run(input_dir, output_dir):\n    raise ValueError('boom')\n"
        )
        result = SubprocessSandbox().run(
            kind="legacy",
            module_path=module,
            input_dir=tmp_path,
            output_dir=tmp_path / "out",
            limits=SandboxLimits.for_pandas(),
        )
        assert not result.ok
        assert "ValueError: boom" in (result.error or "")
        assert "raise ValueError" in (result.traceback or "")

    def test_an_infinite_loop_is_killed_by_the_wall_clock(self, tmp_path: Path) -> None:
        """Generated code that never terminates must not pin a worker forever."""
        module = write_module(
            tmp_path, "def run(input_dir, output_dir):\n    while True:\n        pass\n"
        )
        result = SubprocessSandbox().run(
            kind="legacy",
            module_path=module,
            input_dir=tmp_path,
            output_dir=tmp_path / "out",
            limits=SandboxLimits(wall_clock_seconds=3, cpu_seconds=None, memory_bytes=None),
        )
        assert result.timed_out
        assert not result.ok
        assert "wall-clock" in (result.error or "")

    def test_the_child_cannot_crash_the_parent(self, tmp_path: Path) -> None:
        """`sys.exit` in generated code must not take down the orchestrator."""
        module = write_module(
            tmp_path,
            "import sys\n\n\ndef run(input_dir, output_dir):\n    sys.exit(3)\n",
        )
        result = SubprocessSandbox().run(
            kind="legacy",
            module_path=module,
            input_dir=tmp_path,
            output_dir=tmp_path / "out",
            limits=SandboxLimits.for_pandas(),
        )
        assert not result.ok
        assert os.getpid() > 0  # the parent is still here to make this assertion

    def test_stdout_cannot_forge_a_verdict(self, tmp_path: Path) -> None:
        """The result is read from a file, not parsed from stdout, so generated
        code printing a success payload changes nothing."""
        module = write_module(
            tmp_path,
            'def run(input_dir, output_dir):\n'
            '    print(\'{"ok": true, "metrics": {"engine": "forged"}}\')\n'
            "    raise RuntimeError('actually failed')\n",
        )
        result = SubprocessSandbox().run(
            kind="legacy",
            module_path=module,
            input_dir=tmp_path,
            output_dir=tmp_path / "out",
            limits=SandboxLimits.for_pandas(),
        )
        assert not result.ok
        assert result.metrics.get("engine") != "forged"

    def test_relative_paths_land_in_a_scratch_dir_not_the_repo(self, tmp_path: Path) -> None:
        """A neutral cwd means a stray relative write cannot touch the project."""
        module = write_module(
            tmp_path,
            "def run(input_dir, output_dir):\n"
            "    with open('escaped.txt', 'w') as fh:\n"
            "        fh.write('x')\n",
        )
        result = SubprocessSandbox().run(
            kind="legacy",
            module_path=module,
            input_dir=tmp_path,
            output_dir=tmp_path / "out",
            limits=SandboxLimits.for_pandas(),
        )
        assert result.ok
        assert not Path("escaped.txt").exists()
        assert not (tmp_path / "escaped.txt").exists()


class TestLimitProfiles:
    def test_pandas_profile_bounds_memory_and_processes(self) -> None:
        limits = SandboxLimits.for_pandas()
        assert limits.memory_bytes and limits.processes

    def test_spark_profile_leaves_memory_and_processes_unset(self) -> None:
        """The JVM reserves a huge virtual address space and runs many threads,
        which on Linux count against RLIMIT_NPROC. Applying either would stop
        Spark from starting without making anything safer — Spark's own driver
        memory setting is the real control."""
        limits = SandboxLimits.for_spark()
        assert limits.memory_bytes is None
        assert limits.processes is None
        assert limits.wall_clock_seconds >= 900

    def test_payload_omits_unset_limits(self) -> None:
        assert "memory_bytes" not in SandboxLimits.for_spark().as_payload()
        assert "cpu_seconds" in SandboxLimits.for_pandas().as_payload()
