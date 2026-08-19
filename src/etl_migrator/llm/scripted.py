"""A deterministic `ChatCompletionClient` backed by recorded responses.

Implementing the real interface rather than stubbing the agent means AutoGen
runs its genuine tool loop and structured-output parsing against fixtures, so
CI needs no API key and still exercises the code path that matters. A missing
fixture raises `ScriptExhaustedError` instead of inventing a response.

Fixtures live in `fixtures/llm/<scenario>.json`, keyed by agent name. Each
agent's value is either a flat list of turns, or a list of scripts selected by
a `when` substring matched against the request (one with no `when` is the
fallback). The `when` form covers agents invoked several times per migration
with different prompts, such as repair attempt 1 versus attempt 2. Selecting on
request content rather than advancing a cursor keeps it correct when Temporal
retries an activity.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncGenerator, Mapping, Sequence
from pathlib import Path
from typing import Any

from autogen_core import CancellationToken, FunctionCall
from autogen_core.models import (
    ChatCompletionClient,
    CreateResult,
    LLMMessage,
    ModelFamily,
    ModelInfo,
    RequestUsage,
)
from autogen_core.tools import Tool, ToolSchema
from pydantic import BaseModel, Field

from etl_migrator.domain.errors import ScriptExhaustedError

_MODEL_INFO = ModelInfo(
    vision=False,
    function_calling=True,
    json_output=True,
    family=ModelFamily.UNKNOWN,
    structured_output=True,
    multiple_system_messages=False,
)


def _call_id(tool_call: dict[str, Any]) -> str:
    """Stable id derived from the call itself, so replays are byte-reproducible."""
    digest = uuid.uuid5(uuid.NAMESPACE_OID, json.dumps(tool_call, sort_keys=True)).hex
    return f"call_{digest[:8]}"


class ScriptedTurn(BaseModel):
    """One recorded model response."""

    tool_calls: list[dict[str, Any]] | None = None
    content: Any = None

    def to_content(self) -> str | list[FunctionCall]:
        if self.tool_calls is not None:
            return [
                FunctionCall(
                    id=_call_id(tc),
                    name=str(tc["name"]),
                    arguments=json.dumps(tc.get("arguments", {})),
                )
                for tc in self.tool_calls
            ]
        if isinstance(self.content, str):
            return self.content
        return json.dumps(self.content)


class AgentScript(BaseModel):
    """One recorded exchange, optionally conditional on the request."""

    when: str | None = Field(
        default=None,
        description="Substring that must appear in the request for this script to "
        "apply. None means it is the fallback.",
    )
    turns: list[ScriptedTurn] = Field(default_factory=list)

    def matches(self, request: str) -> bool:
        return self.when is None or self.when in request


class ScriptedChatCompletionClient(ChatCompletionClient):
    """Replays a script. Selects which one on the first request, then stays on it.

    Selection happens once because every message in a single agent run must come
    from the same recorded exchange; a fresh client is created per agent
    invocation, which is what makes per-invocation scripts work.
    """

    def __init__(
        self,
        turns: Sequence[ScriptedTurn] | None = None,
        *,
        scripts: Sequence[AgentScript] | None = None,
        label: str = "scripted",
    ) -> None:
        if scripts is None:
            scripts = [AgentScript(turns=list(turns or []))]
        self._scripts = list(scripts)
        self._turns: list[ScriptedTurn] | None = None
        self._cursor = 0
        self._label = label
        self._usage = RequestUsage(prompt_tokens=0, completion_tokens=0)
        # Retained so tests can assert on what the agent actually sent.
        self.recorded_requests: list[list[LLMMessage]] = []

    async def create(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Tool | ToolSchema] = [],
        tool_choice: Tool | str = "auto",
        json_output: bool | type[BaseModel] | None = None,
        extra_create_args: Mapping[str, Any] = {},
        cancellation_token: CancellationToken | None = None,
    ) -> CreateResult:
        self.recorded_requests.append(list(messages))
        if self._turns is None:
            self._turns = self._select(messages)
        if self._cursor >= len(self._turns):
            raise ScriptExhaustedError(
                f"scripted client '{self._label}' exhausted after {len(self._turns)} turn(s); "
                "the agent asked for another completion. Either the agent loop changed or the "
                "fixture is incomplete — record the missing turn instead of loosening the client."
            )
        turn = self._turns[self._cursor]
        self._cursor += 1
        content = turn.to_content()
        return CreateResult(
            finish_reason="function_calls" if isinstance(content, list) else "stop",
            content=content,
            usage=self._usage,
            cached=False,
        )

    def _select(self, messages: Sequence[LLMMessage]) -> list[ScriptedTurn]:
        """Pick the script whose `when` appears in the request."""
        request = "\n".join(str(getattr(m, "content", "")) for m in messages)
        for script in self._scripts:
            if script.matches(request):
                return list(script.turns)
        raise ScriptExhaustedError(
            f"no scripted exchange for '{self._label}' matches this request; "
            f"recorded conditions: {[s.when for s in self._scripts]}. Record the "
            "missing exchange rather than loosening the matcher."
        )

    def create_stream(
        self, *args: Any, **kwargs: Any
    ) -> AsyncGenerator[str | CreateResult, None]:
        raise NotImplementedError("ScriptedChatCompletionClient does not stream")

    async def close(self) -> None:
        return None

    def actual_usage(self) -> RequestUsage:
        return self._usage

    def total_usage(self) -> RequestUsage:
        return self._usage

    def count_tokens(
        self, messages: Sequence[LLMMessage], *, tools: Sequence[Tool | ToolSchema] = []
    ) -> int:
        return sum(len(str(getattr(m, "content", ""))) for m in messages) // 4

    def remaining_tokens(
        self, messages: Sequence[LLMMessage], *, tools: Sequence[Tool | ToolSchema] = []
    ) -> int:
        return 128_000 - self.count_tokens(messages, tools=tools)

    @property
    def capabilities(self) -> ModelInfo:
        return _MODEL_INFO

    @property
    def model_info(self) -> ModelInfo:
        return _MODEL_INFO

    @property
    def turns_remaining(self) -> int:
        return 0 if self._turns is None else len(self._turns) - self._cursor


def _parse_scripts(entries: list[dict[str, Any]]) -> list[AgentScript]:
    """Accept either a flat turn list or a list of conditional scripts."""
    if entries and all("turns" in entry for entry in entries):
        return [AgentScript.model_validate(entry) for entry in entries]
    return [AgentScript(turns=[ScriptedTurn.model_validate(e) for e in entries])]


class ScriptLibrary:
    """Loads a fixture file and hands out one client per agent."""

    def __init__(self, scripts: Mapping[str, Sequence[AgentScript]], *, source: str) -> None:
        self._scripts = {k: list(v) for k, v in scripts.items()}
        self._source = source

    @classmethod
    def from_file(cls, path: Path) -> ScriptLibrary:
        if not path.is_file():
            raise ScriptExhaustedError(
                f"LLM fixture not found: {path}. Run with ETLM_LLM_PROVIDER=anthropic|openai, "
                f"or record fixtures with `etl-migrator record`."
            )
        raw = json.loads(path.read_text(encoding="utf-8"))
        scripts = {agent: _parse_scripts(entries) for agent, entries in raw.items()}
        return cls(scripts, source=str(path))

    def client_for(self, agent: str) -> ScriptedChatCompletionClient:
        if agent not in self._scripts:
            raise ScriptExhaustedError(
                f"no scripted turns for agent '{agent}' in {self._source}; "
                f"available: {sorted(self._scripts)}"
            )
        return ScriptedChatCompletionClient(
            scripts=self._scripts[agent], label=f"{agent}@{self._source}"
        )

    @property
    def agents(self) -> list[str]:
        return sorted(self._scripts)
