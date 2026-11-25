"""Graph state.

Design rules:
- Keep state small and serialisable (it is checkpointed every step).
- Append-only channels use reducers (`add_messages`, `operator.add`) so
  parallel workers (Send fan-out) can write in the same super-step safely.
- Per-turn channels are reset at the start of each turn with `Overwrite(...)`,
  which bypasses the reducer (new in LangGraph 1.x) - no custom "reset" hacks.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Optional

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import NotRequired, TypedDict


class AgentResult(TypedDict):
    agent: str
    skill: Optional[str]
    query: str
    summary: str
    rows: list[dict[str, Any]]
    has_data: bool
    error: Optional[str]


class OrchestratorState(TypedDict):
    # Conversation (persists across turns in the same thread)
    messages: Annotated[list[AnyMessage], add_messages]

    # ---- per-turn working memory (reset by start_turn) ----
    question: NotRequired[str]                  # current question (+ clarifications)
    metadata: NotRequired[Optional[dict]]       # MetadataResult.model_dump()
    route: NotRequired[Optional[dict]]          # last RouteDecision.model_dump()
    plan: NotRequired[list[str]]                # remaining agents for sequential mode
    results: NotRequired[Annotated[list[AgentResult], operator.add]]
    attempted: NotRequired[Annotated[list[str], operator.add]]
    iterations: NotRequired[int]
    clarifications: NotRequired[int]


class WorkerTask(TypedDict):
    """Private input of a worker node, delivered via `Send(agent, WorkerTask)`.

    Workers don't read the whole graph state: they get exactly what they need.
    That keeps them reusable and makes parallel fan-out trivial.
    """

    question: str
    skill: Optional[str]
    metadata: dict
    prior_results: list[AgentResult]
