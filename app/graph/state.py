from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, Optional, TypedDict

from langchain_core.messages import AnyMessage


class MetadataContext(TypedDict, total=False):
    """Context gathered by the metadata agent."""

    tables: list[dict[str, Any]]
    fields: list[dict[str, Any]]
    source_system: str  # "snowflake" | "graphql" | "both"
    data_availability: dict[str, bool]  # {"snowflake": True, "graphql": False}
    lookup_query: str


class AgentResult(TypedDict, total=False):
    """Result from any data-fetching agent."""

    agent_name: str
    skill_used: Optional[str]
    query_generated: str
    data: Any
    error: Optional[str]
    has_data: bool


class HumanFeedbackRequest(TypedDict, total=False):
    """Structured interrupt payload."""

    feedback_type: Literal["classification", "clarification", "no_data"]
    question: str
    options: Optional[list[str]]
    context: str


class SupervisorState(TypedDict):
    """Top-level state flowing through the entire graph."""

    # Conversation
    messages: Annotated[list[AnyMessage], operator.add]

    # Routing
    current_agent: str
    route_reasoning: str
    active_skill: Optional[str]

    # Metadata context (populated by metadata agent)
    metadata_context: Optional[MetadataContext]

    # Agent results (accumulated across agents)
    agent_results: Annotated[list[AgentResult], operator.add]

    # Human feedback
    pending_feedback: Optional[HumanFeedbackRequest]
    human_response: Optional[str]

    # Control flow
    iteration_count: int
    is_complete: bool
