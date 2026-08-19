"""Provider abstraction.

The abstraction *is* `autogen_core.models.ChatCompletionClient`. Inventing a
second wrapper on top of it would only add a layer that has to be kept in sync
with AutoGen's tool-calling and structured-output plumbing. So the factory's
whole job is: read settings, hand back a client, and make the offline path a
first-class provider rather than a test hack.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from autogen_core.models import ChatCompletionClient

from etl_migrator.config import LLMProvider, Settings
from etl_migrator.domain.errors import ConfigurationError
from etl_migrator.llm.scripted import ScriptLibrary


class ModelClientFactory(Protocol):
    """Agents receive this, never a concrete client, so the same agent code runs
    against a live provider and against recorded fixtures."""

    def client_for(self, agent: str) -> ChatCompletionClient: ...


class LiveModelClientFactory:
    """Builds real provider clients from autogen-ext."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._api_key = settings.require_llm_credentials()

    def client_for(self, agent: str) -> ChatCompletionClient:
        s = self._settings
        common: dict[str, object] = {
            "model": s.llm_model,
            "api_key": self._api_key,
            "temperature": s.llm_temperature,
        }
        if s.llm_base_url:
            common["base_url"] = s.llm_base_url

        if s.llm_provider is LLMProvider.ANTHROPIC:
            from autogen_ext.models.anthropic import AnthropicChatCompletionClient

            return AnthropicChatCompletionClient(**common)  # type: ignore[arg-type]

        if s.llm_provider is LLMProvider.OPENAI:
            from autogen_ext.models.openai import OpenAIChatCompletionClient

            return OpenAIChatCompletionClient(**common)  # type: ignore[arg-type]

        raise ConfigurationError(f"provider {s.llm_provider} is not a live provider")


class ScriptedModelClientFactory:
    """Serves recorded turns. Deterministic, offline, and loud when incomplete."""

    def __init__(self, library: ScriptLibrary) -> None:
        self._library = library
        self.issued: dict[str, ChatCompletionClient] = {}

    @classmethod
    def from_fixture(cls, path: Path) -> ScriptedModelClientFactory:
        return cls(ScriptLibrary.from_file(path))

    def client_for(self, agent: str) -> ChatCompletionClient:
        client = self._library.client_for(agent)
        self.issued[agent] = client
        return client


def build_model_client_factory(
    settings: Settings, *, scenario: str = "customer_pipeline"
) -> ModelClientFactory:
    """Single place where provider choice is resolved."""
    if settings.llm_provider is LLMProvider.SCRIPTED:
        return ScriptedModelClientFactory.from_fixture(
            settings.llm_fixture_dir / f"{scenario}.json"
        )
    return LiveModelClientFactory(settings)
