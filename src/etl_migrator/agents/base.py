"""Common scaffolding for every agent in the system.

The contract enforced here is the one from the brief:

    Input -> Reasoning -> Structured Decision -> Tool Invocation -> Observable Result

Concretely, `StructuredAgent` guarantees three things:

* the agent's answer is a validated Pydantic model, never free text;
* every tool the agent called is recorded, so "did it actually look?" is a
  question with an answer;
* a contract violation raises `AgentContractError`, which Temporal treats as a
  retryable failure rather than a silent degradation.

Note what is *not* here: no parsing of markdown code fences, no "extract the
JSON from the reply" regex. Structured output is a first-class model feature in
every provider used, and AutoGen surfaces it as `StructuredMessage[T]`.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any, Generic, TypeVar, cast

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import (
    BaseChatMessage,
    StructuredMessage,
    ToolCallExecutionEvent,
)
from autogen_core.models import ChatCompletionClient
from pydantic import BaseModel, ValidationError

from etl_migrator.domain.errors import AgentContractError
from etl_migrator.observability import get_logger

T = TypeVar("T", bound=BaseModel)

log = get_logger(__name__)


class ToolInvocation(BaseModel):
    """One observed tool call. The audit trail for 'the agent actually looked'."""

    name: str
    arguments: str
    is_error: bool
    result_preview: str


class AgentRun(BaseModel, Generic[T]):
    """Result of one agent execution, with the evidence attached."""

    agent: str
    output: T
    tool_invocations: list[ToolInvocation]
    llm_turns: int
    duration_seconds: float

    @property
    def tools_used(self) -> list[str]:
        return [t.name for t in self.tool_invocations]


class StructuredAgent(Generic[T]):
    """An AutoGen `AssistantAgent` bound to a Pydantic output contract."""

    #: Key used to look up this agent's model client (and its recorded fixtures).
    key: str = "base"
    #: One-line description used by AutoGen for agent selection.
    description: str = "A specialised migration agent."

    def __init__(
        self,
        model_client: ChatCompletionClient,
        output_type: type[T],
        *,
        system_message: str,
        tools: Sequence[Callable[..., Any]] = (),
        max_tool_iterations: int = 6,
    ) -> None:
        self._output_type = output_type
        self._model_client = model_client
        self._agent = AssistantAgent(
            name=self.key,
            model_client=model_client,
            description=self.description,
            system_message=system_message,
            tools=list(tools) or None,
            output_content_type=output_type,
            max_tool_iterations=max_tool_iterations,
            reflect_on_tool_use=True,
        )

    async def run(self, task: str) -> AgentRun[T]:
        started = time.perf_counter()
        log.info("agent.start", agent=self.key, output_type=self._output_type.__name__)

        try:
            result = await self._agent.run(task=task)
        except ValidationError as exc:
            # AutoGen parses structured output itself and surfaces a raw pydantic
            # error. Translating it here keeps "the model broke its contract" a
            # single typed failure, whether it was malformed JSON or a valid
            # object of the wrong shape. Temporal's retry policy names this one.
            raise AgentContractError(
                f"agent '{self.key}' did not produce a {self._output_type.__name__}: {exc}"
            ) from exc
        duration = time.perf_counter() - started

        invocations = _collect_tool_invocations(result.messages)
        output = self._extract(result.messages)

        log.info(
            "agent.done",
            agent=self.key,
            duration_s=round(duration, 3),
            tools_used=[i.name for i in invocations],
            messages=len(result.messages),
        )
        # Parametrise at runtime so pydantic validates `output` against the
        # agent's concrete contract rather than the TypeVar bound.
        run_model: Any = AgentRun[self._output_type]  # type: ignore[name-defined]
        return cast(
            "AgentRun[T]",
            run_model(
                agent=self.key,
                output=output,
                tool_invocations=invocations,
                llm_turns=len(result.messages),
                duration_seconds=duration,
            ),
        )

    def _extract(self, messages: Sequence[BaseChatMessage | Any]) -> T:
        for message in reversed(messages):
            if isinstance(message, StructuredMessage) and isinstance(
                message.content, self._output_type
            ):
                return message.content
        tail = messages[-1] if messages else None
        raise AgentContractError(
            f"agent '{self.key}' did not produce a {self._output_type.__name__}; "
            f"last message was {type(tail).__name__}: "
            f"{str(getattr(tail, 'content', ''))[:400]}"
        )

    async def close(self) -> None:
        await self._model_client.close()


def _collect_tool_invocations(messages: Sequence[Any]) -> list[ToolInvocation]:
    out: list[ToolInvocation] = []
    for message in messages:
        if isinstance(message, ToolCallExecutionEvent):
            for res in message.content:
                out.append(
                    ToolInvocation(
                        name=res.name,
                        arguments="",
                        is_error=bool(res.is_error),
                        result_preview=str(res.content)[:280],
                    )
                )
    return out
