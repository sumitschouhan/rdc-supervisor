"""Tests for routing logic (conditional edge functions)."""

from app.graph.routing import route_after_data_agent, route_from_supervisor


def _make_state(**overrides):
    base = {
        "messages": [],
        "current_agent": "",
        "route_reasoning": "",
        "active_skill": None,
        "metadata_context": None,
        "agent_results": [],
        "pending_feedback": None,
        "human_response": None,
        "iteration_count": 0,
        "is_complete": False,
    }
    base.update(overrides)
    return base


def test_route_to_metadata_agent():
    state = _make_state(current_agent="metadata_agent")
    assert route_from_supervisor(state) == "metadata_agent"


def test_route_to_mesh_agent():
    state = _make_state(current_agent="mesh_agent")
    assert route_from_supervisor(state) == "mesh_agent"


def test_route_to_graphql_agent():
    state = _make_state(current_agent="graphql_agent")
    assert route_from_supervisor(state) == "graphql_agent"


def test_route_to_human_feedback():
    state = _make_state(current_agent="human_feedback")
    assert route_from_supervisor(state) == "human_feedback"


def test_route_to_synthesizer_when_complete():
    state = _make_state(is_complete=True, current_agent="mesh_agent")
    assert route_from_supervisor(state) == "synthesizer"


def test_safety_valve_max_iterations():
    state = _make_state(iteration_count=5, current_agent="mesh_agent")
    assert route_from_supervisor(state) == "synthesizer"


def test_route_unknown_agent_defaults_to_synthesizer():
    state = _make_state(current_agent="unknown_agent")
    assert route_from_supervisor(state) == "synthesizer"


def test_route_after_data_agent_has_data():
    state = _make_state(
        agent_results=[{"agent_name": "mesh_agent", "has_data": True}]
    )
    assert route_after_data_agent(state) == "synthesizer"


def test_route_after_data_agent_no_data():
    state = _make_state(
        agent_results=[{"agent_name": "mesh_agent", "has_data": False}]
    )
    assert route_after_data_agent(state) == "supervisor"


def test_route_after_data_agent_empty_results():
    state = _make_state(agent_results=[])
    assert route_after_data_agent(state) == "supervisor"
