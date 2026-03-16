from __future__ import annotations

from app.graph.state import SupervisorState


def route_from_supervisor(state: SupervisorState) -> str:
    """Conditional edge after supervisor node. Returns the next node name."""
    if state.get("is_complete"):
        return "synthesizer"

    # Safety valve: prevent infinite loops
    if state.get("iteration_count", 0) >= 5:
        return "synthesizer"

    current = state.get("current_agent", "")

    valid_agents = {
        "metadata_agent", "mesh_agent", "graphql_agent",
        "cross_agent_dispatcher", "human_feedback", "synthesizer",
    }
    if current in valid_agents:
        return current

    return "synthesizer"


def route_after_data_agent(state: SupervisorState) -> str:
    """Conditional edge after mesh_agent, graphql_agent, or cross_agent_dispatcher.

    - If sequential cross-agent has pending agents -> back to supervisor to dequeue next.
    - If any result has data -> synthesizer.
    - Otherwise -> supervisor for fallback / human feedback.
    """
    # Sequential cross-agent: more agents still waiting in the queue
    pending = state.get("pending_agents") or []
    if pending:
        return "supervisor"

    results = state.get("agent_results", [])
    if results and any(r.get("has_data") for r in results):
        return "synthesizer"

    # No data found - route back to supervisor to try fallback or human feedback
    return "supervisor"
