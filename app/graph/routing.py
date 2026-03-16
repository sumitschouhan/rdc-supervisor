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

    valid_agents = {"metadata_agent", "mesh_agent", "graphql_agent", "human_feedback", "synthesizer"}
    if current in valid_agents:
        return current

    return "synthesizer"


def route_after_data_agent(state: SupervisorState) -> str:
    """Conditional edge after mesh_agent or graphql_agent.
    If data was found -> synthesizer. Otherwise -> back to supervisor for fallback."""
    results = state.get("agent_results", [])
    if results and results[-1].get("has_data"):
        return "synthesizer"

    # No data found - route back to supervisor to try fallback or human feedback
    return "supervisor"
