"""Shared fixtures.

Note the deliberate choice throughout the suite: the *model* is mocked, the
agents are not. `ScriptedChatCompletionClient` is a real implementation of
AutoGen's `ChatCompletionClient`, so tests drive the genuine agent loop, tool
dispatch and structured-output parsing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from etl_migrator.config import LLMProvider, Settings
from etl_migrator.llm.factory import ScriptedModelClientFactory

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DIR = REPO_ROOT / "examples" / "customer_pipeline"
FIXTURE_PATH = REPO_ROOT / "fixtures" / "llm" / "customer_pipeline.json"


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def legacy_source() -> Path:
    return EXAMPLE_DIR / "legacy_pipeline.py"


@pytest.fixture(scope="session")
def example_input_dir() -> Path:
    path = EXAMPLE_DIR / "input"
    if not (path / "customers.csv").is_file():
        pytest.skip("example data missing; run examples/customer_pipeline/generate_data.py")
    return path


@pytest.fixture(scope="session")
def fixture_payload() -> dict[str, object]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        llm_provider=LLMProvider.SCRIPTED,
        llm_fixture_dir=FIXTURE_PATH.parent,
        workspace_dir=tmp_path / "workspace",
        log_format="json",
    )


@pytest.fixture
def factory() -> ScriptedModelClientFactory:
    return ScriptedModelClientFactory.from_fixture(FIXTURE_PATH)
