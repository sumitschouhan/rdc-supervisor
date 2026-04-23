"""Tests for the state schema."""

from app.graph.state import (
    AgentResult,
    HumanFeedbackRequest,
    MetadataContext,
    SupervisorState,
)


def test_metadata_context_creation():
    ctx: MetadataContext = {
        "tables": [{"table": "trades", "columns": []}],
        "fields": [],
        "source_system": "snowflake",
        "data_availability": {"snowflake": True, "graphql": False},
        "lookup_query": "show me trades",
    }
    assert ctx["source_system"] == "snowflake"
    assert ctx["data_availability"]["snowflake"] is True


def test_agent_result_creation():
    result: AgentResult = {
        "agent_name": "mesh_agent",
        "skill_used": "trade_query",
        "query_generated": "SELECT * FROM trades",
        "data": {"rows": [], "row_count": 0},
        "error": "No data",
        "has_data": False,
    }
    assert result["agent_name"] == "mesh_agent"
    assert result["has_data"] is False


def test_human_feedback_request():
    req: HumanFeedbackRequest = {
        "feedback_type": "classification",
        "title": "Classification Needed",
        "question": "Which industry?",
        "options": ["Tech", "Finance"],
        "suggestions": ["Show me Tech instruments"],
        "allow_free_text": False,
        "context": "Need classification",
        "progress": "Analyzed your query.",
    }
    assert req["feedback_type"] == "classification"
    assert req["title"] == "Classification Needed"
    assert len(req["options"]) == 2
    assert req["allow_free_text"] is False
    assert req["suggestions"] == ["Show me Tech instruments"]


def test_human_feedback_request_no_data():
    req: HumanFeedbackRequest = {
        "feedback_type": "no_data",
        "title": "No Results Found",
        "question": "I searched but found nothing...",
        "options": None,
        "suggestions": ["Try a different date range"],
        "allow_free_text": True,
        "context": "No records found.",
        "progress": "Searched Snowflake and GraphQL API.",
    }
    assert req["feedback_type"] == "no_data"
    assert req["options"] is None
    assert req["allow_free_text"] is True


def test_supervisor_state_partial():
    """Verify state can be created with minimal fields."""
    state: SupervisorState = {
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
        "hil_attempt_count": 0,
        "last_feedback_question": None,
        "execution_mode": None,
        "cross_agent_targets": None,
        "pending_agents": None,
    }
    assert state["iteration_count"] == 0
    assert state["is_complete"] is False
    assert state["hil_attempt_count"] == 0
    assert state["last_feedback_question"] is None
