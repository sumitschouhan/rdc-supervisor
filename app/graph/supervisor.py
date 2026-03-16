from __future__ import annotations

import json
import os
from typing import Any, Optional

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from app.config import AppConfig, load_config
from app.graph.state import SupervisorState
from app.graph.routing import route_from_supervisor, route_after_data_agent
from app.prompts.loader import PromptLoader
from app.skills.registry import SkillRegistry
from app.agents.metadata_agent import MetadataAgent
from app.agents.mesh_agent import MeshAgent
from app.agents.graphql_agent import GraphQLAgent
from app.agents.cross_agent_dispatcher import CrossAgentDispatcher
from app.agents.human_feedback_agent import HumanFeedbackAgent
from app.agents.synthesizer import SynthesizerAgent


# Module-level singletons, initialized via init_graph()
_config: Optional[AppConfig] = None
_model: Optional[ChatAnthropic] = None
_prompt_loader: Optional[PromptLoader] = None
_skill_registry: Optional[SkillRegistry] = None
_graph: Any = None


def init_graph(
    config: AppConfig,
    model: ChatAnthropic,
    prompt_loader: PromptLoader,
    skill_registry: SkillRegistry,
) -> None:
    """Called once at startup to inject dependencies."""
    global _config, _model, _prompt_loader, _skill_registry
    _config = config
    _model = model
    _prompt_loader = prompt_loader
    _skill_registry = skill_registry


def set_graph(graph: Any) -> None:
    global _graph
    _graph = graph


def get_graph() -> Any:
    return _graph


# --- Node functions ---


def supervisor_node(state: SupervisorState) -> dict:
    """Analyzes user query + state, decides which agent to route to."""
    assert _model is not None and _prompt_loader is not None and _skill_registry is not None

    # Sequential cross-agent: dequeue the next pending agent without calling the LLM
    pending = state.get("pending_agents") or []
    if pending:
        next_agent = pending[0]
        remaining = pending[1:]
        return {
            "current_agent": next_agent,
            "pending_agents": remaining,
            "route_reasoning": f"Sequential cross-agent: running {next_agent} next",
            "iteration_count": state.get("iteration_count", 0) + 1,
        }

    user_msg = _get_latest_user_message(state)

    # Build context summary
    context_parts = [f"User query: {user_msg}"]

    metadata = state.get("metadata_context")
    if metadata:
        context_parts.append(
            f"Metadata available: {len(metadata.get('tables', []))} tables, "
            f"source_system: {metadata.get('source_system', 'unknown')}, "
            f"data_availability: {json.dumps(metadata.get('data_availability', {}))}"
        )
    else:
        context_parts.append("Metadata: NOT YET GATHERED")

    results = state.get("agent_results", [])
    if results:
        for r in results:
            context_parts.append(
                f"Agent '{r['agent_name']}' (skill: {r.get('skill_used', 'none')}): "
                f"has_data={r.get('has_data', False)}, error={r.get('error', 'none')}"
            )

    human_resp = state.get("human_response")
    if human_resp:
        context_parts.append(f"Human clarification: {human_resp}")

    # Build the supervisor prompt with skill catalog
    skill_catalog = _skill_registry.get_catalog()
    system = (
        _prompt_loader.supervisor_system(skill_catalog=skill_catalog)
        + "\n\n"
        + _prompt_loader.supervisor_output_format()
    )

    messages: list[Any] = [
        SystemMessage(content=system),
        HumanMessage(content="\n\n".join(context_parts)),
    ]

    response = _model.invoke(messages)

    # Parse JSON routing decision
    execution_mode = None
    cross_agent_targets = None
    try:
        decision = json.loads(response.content)
        agent = decision["agent"]
        skill = decision.get("skill")
        reasoning = decision.get("reasoning", "")
        # Cross-agent fields (only present when agent == "cross_agent_dispatcher")
        if agent == "cross_agent_dispatcher":
            execution_mode = decision.get("execution_mode", "parallel")
            cross_agent_targets = decision.get("agents", ["mesh_agent", "graphql_agent"])
    except (json.JSONDecodeError, KeyError, TypeError):
        # Fallback: try to extract agent name from text
        content = response.content if isinstance(response.content, str) else str(response.content)
        agent = "synthesizer"
        skill = None
        reasoning = f"Could not parse routing decision: {content[:100]}"

        for candidate in [
            "cross_agent_dispatcher", "metadata_agent", "mesh_agent",
            "graphql_agent", "human_feedback",
        ]:
            if candidate in content:
                agent = candidate
                break

    return {
        "current_agent": agent,
        "route_reasoning": reasoning,
        "active_skill": skill,
        "execution_mode": execution_mode,
        "cross_agent_targets": cross_agent_targets,
        "iteration_count": state.get("iteration_count", 0) + 1,
    }


def metadata_agent_node(state: SupervisorState) -> dict:
    assert _model is not None and _prompt_loader is not None
    agent = MetadataAgent(_model, _prompt_loader)
    return agent.invoke(state)


def mesh_agent_node(state: SupervisorState) -> dict:
    assert _model is not None and _prompt_loader is not None and _skill_registry is not None
    agent = MeshAgent(_model, _prompt_loader, _skill_registry)
    return agent.invoke(state)


def graphql_agent_node(state: SupervisorState) -> dict:
    assert _model is not None and _prompt_loader is not None and _skill_registry is not None
    agent = GraphQLAgent(_model, _prompt_loader, _skill_registry)
    return agent.invoke(state)


def cross_agent_dispatcher_node(state: SupervisorState) -> dict:
    assert _model is not None and _prompt_loader is not None and _skill_registry is not None
    agent = CrossAgentDispatcher(_model, _prompt_loader, _skill_registry)
    return agent.invoke(state)


def human_feedback_node(state: SupervisorState) -> dict:
    assert _prompt_loader is not None and _skill_registry is not None
    agent = HumanFeedbackAgent(_prompt_loader, _skill_registry)
    return agent.invoke(state)


def synthesizer_node(state: SupervisorState) -> dict:
    assert _model is not None and _prompt_loader is not None
    agent = SynthesizerAgent(_model, _prompt_loader)
    return agent.invoke(state)


# --- Graph construction ---


def build_graph() -> Any:
    """Build and compile the LangGraph supervisor graph."""
    builder = StateGraph(SupervisorState)

    # Add nodes
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("metadata_agent", metadata_agent_node)
    builder.add_node("mesh_agent", mesh_agent_node)
    builder.add_node("graphql_agent", graphql_agent_node)
    builder.add_node("cross_agent_dispatcher", cross_agent_dispatcher_node)
    builder.add_node("human_feedback", human_feedback_node)
    builder.add_node("synthesizer", synthesizer_node)

    # Entry: always start at supervisor
    builder.add_edge(START, "supervisor")

    # Supervisor routes conditionally
    builder.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "metadata_agent": "metadata_agent",
            "mesh_agent": "mesh_agent",
            "graphql_agent": "graphql_agent",
            "cross_agent_dispatcher": "cross_agent_dispatcher",
            "human_feedback": "human_feedback",
            "synthesizer": "synthesizer",
        },
    )

    # Dispatcher: if data found go to synthesizer, else back to supervisor
    builder.add_conditional_edges(
        "cross_agent_dispatcher",
        route_after_data_agent,
        {"synthesizer": "synthesizer", "supervisor": "supervisor"},
    )

    # Metadata always returns to supervisor for next decision
    builder.add_edge("metadata_agent", "supervisor")

    # Data agents: if data found -> synthesizer, else -> supervisor for fallback
    builder.add_conditional_edges(
        "mesh_agent",
        route_after_data_agent,
        {"synthesizer": "synthesizer", "supervisor": "supervisor"},
    )
    builder.add_conditional_edges(
        "graphql_agent",
        route_after_data_agent,
        {"synthesizer": "synthesizer", "supervisor": "supervisor"},
    )

    # Human feedback returns to supervisor with new context
    builder.add_edge("human_feedback", "supervisor")

    # Synthesizer terminates
    builder.add_edge("synthesizer", END)

    # Compile with in-memory checkpointer for interrupt support
    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)


# --- Helper ---


def _get_latest_user_message(state: SupervisorState) -> str:
    for msg in reversed(state.get("messages", [])):
        if hasattr(msg, "type") and msg.type == "human":
            return msg.content
    return ""
