"""Tests for the enhanced HumanFeedbackAgent and related schemas."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from app.agents.human_feedback_agent import HumanFeedbackAgent, _CLASSIFICATION_OPTIONS
from app.api.schemas import InterruptPayload, ResumeRequest, ChatResponse


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------

class TestInterruptPayload:
    def test_default_title(self):
        payload = InterruptPayload(
            feedback_type="no_data",
            question="Could you clarify?",
            context="No results found.",
        )
        assert payload.title == "Input Required"

    def test_custom_title(self):
        payload = InterruptPayload(
            feedback_type="classification",
            title="Classification Needed",
            question="Which sector?",
            options=["Technology", "Finance"],
            allow_free_text=False,
            context="Select a sector.",
        )
        assert payload.title == "Classification Needed"
        assert payload.allow_free_text is False

    def test_suggestions_and_progress(self):
        payload = InterruptPayload(
            feedback_type="no_data",
            title="No Results Found",
            question="Try refining your query.",
            suggestions=["Use a shorter date range", "Specify an instrument code"],
            allow_free_text=True,
            context="Searched all sources.",
            progress="Searched Snowflake and GraphQL API.",
        )
        assert len(payload.suggestions) == 2
        assert payload.progress is not None

    def test_options_none_by_default(self):
        payload = InterruptPayload(
            feedback_type="clarification",
            question="What do you mean?",
            context="Ambiguous query.",
        )
        assert payload.options is None
        assert payload.suggestions is None


class TestResumeRequest:
    def test_valid_response(self):
        req = ResumeRequest(thread_id="t1", response="Technology")
        assert req.response == "Technology"

    def test_strips_whitespace(self):
        req = ResumeRequest(thread_id="t1", response="  Finance  ")
        assert req.response == "Finance"

    def test_empty_response_rejected(self):
        with pytest.raises(Exception):
            ResumeRequest(thread_id="t1", response="")

    def test_whitespace_only_rejected(self):
        with pytest.raises(Exception):
            ResumeRequest(thread_id="t1", response="   ")


class TestChatResponse:
    def test_agent_summary_optional(self):
        resp = ChatResponse(thread_id="t1", response="Done.", is_complete=True)
        assert resp.agent_summary is None

    def test_agent_summary_populated(self):
        resp = ChatResponse(
            thread_id="t1",
            response="Done.",
            is_complete=True,
            agent_trace=["mesh_agent", "synthesizer"],
            agent_summary="Processed through: Snowflake query, and response generation.",
        )
        assert "Snowflake" in resp.agent_summary


# ---------------------------------------------------------------------------
# HumanFeedbackAgent unit tests
# ---------------------------------------------------------------------------

def _make_agent(config, skill_registry):
    from app.prompts.loader import PromptLoader
    loader = PromptLoader(config)
    return HumanFeedbackAgent(loader, skill_registry)


def _make_state(**overrides):
    from langchain_core.messages import HumanMessage
    base = {
        "messages": [HumanMessage(content="show me data")],
        "current_agent": "human_feedback",
        "route_reasoning": "",
        "active_skill": None,
        "metadata_context": None,
        "agent_results": [],
        "pending_feedback": None,
        "human_response": None,
        "iteration_count": 1,
        "is_complete": False,
        "hil_attempt_count": 0,
        "last_feedback_question": None,
        "execution_mode": None,
        "cross_agent_targets": None,
        "pending_agents": None,
    }
    base.update(overrides)
    return base


class TestHumanFeedbackAgentDetection:
    def test_needs_classification_with_sector(self, config, skill_registry):
        agent = _make_agent(config, skill_registry)
        assert agent._needs_classification("Show me instruments in the sector") is True

    def test_needs_classification_with_industry(self, config, skill_registry):
        agent = _make_agent(config, skill_registry)
        assert agent._needs_classification("List all instruments by industry") is True

    def test_no_classification_for_normal_query(self, config, skill_registry):
        agent = _make_agent(config, skill_registry)
        assert agent._needs_classification("Show me trades from last week") is False

    def test_is_ambiguous_with_vague_query(self, config, skill_registry):
        agent = _make_agent(config, skill_registry)
        assert agent._is_ambiguous_query("show me some data info", []) is True

    def test_not_ambiguous_when_results_exist(self, config, skill_registry):
        agent = _make_agent(config, skill_registry)
        results = [{"agent_name": "mesh_agent", "has_data": False}]
        assert agent._is_ambiguous_query("show me some info stuff", results) is False

    def test_not_ambiguous_specific_query(self, config, skill_registry):
        agent = _make_agent(config, skill_registry)
        assert agent._is_ambiguous_query("Show me AAPL trades from last week in USD", []) is False


class TestHumanFeedbackAgentBuilders:
    def test_classification_request_structure(self, config, skill_registry):
        agent = _make_agent(config, skill_registry)
        req = agent._build_classification_request("Show instruments in the sector")
        assert req["feedback_type"] == "classification"
        assert req["title"] == "Classification Needed"
        assert req["options"] == _CLASSIFICATION_OPTIONS
        assert req["allow_free_text"] is False
        assert req["suggestions"] is not None
        assert len(req["suggestions"]) > 0
        assert req["progress"] is not None

    def test_classification_request_options_list(self, config, skill_registry):
        agent = _make_agent(config, skill_registry)
        req = agent._build_classification_request("List by industry")
        assert "Technology" in req["options"]
        assert "Healthcare" in req["options"]
        assert "Finance" in req["options"]

    def test_no_data_request_structure(self, config, skill_registry):
        agent = _make_agent(config, skill_registry)
        results = [
            {"agent_name": "mesh_agent", "has_data": False, "error": "no records"},
            {"agent_name": "graphql_agent", "has_data": False, "error": "empty result"},
        ]
        req = agent._build_no_data_request("Show me trades from 2010", results)
        assert req["feedback_type"] == "no_data"
        assert req["title"] == "No Results Found"
        assert req["allow_free_text"] is True
        assert req["options"] is None
        assert "Snowflake" in req["question"]
        assert "GraphQL" in req["question"]

    def test_no_data_does_not_expose_agent_names(self, config, skill_registry):
        agent = _make_agent(config, skill_registry)
        results = [{"agent_name": "mesh_agent", "has_data": False, "error": "none"}]
        req = agent._build_no_data_request("query", results)
        assert "mesh_agent" not in req["question"]
        assert "graphql_agent" not in req["question"]

    def test_clarification_request_structure(self, config, skill_registry):
        agent = _make_agent(config, skill_registry)
        req = agent._build_clarification_request("show me some stuff", [])
        assert req["feedback_type"] == "clarification"
        assert req["title"] == "Could You Clarify?"
        assert req["allow_free_text"] is True
        assert req["suggestions"] is not None
        assert len(req["suggestions"]) >= 3

    def test_no_data_suggestions_trades(self, config, skill_registry):
        agent = _make_agent(config, skill_registry)
        suggestions = agent._generate_suggestions("Show me trades from 2010", [])
        assert any("date" in s.lower() or "recent" in s.lower() for s in suggestions)

    def test_no_data_suggestions_instruments(self, config, skill_registry):
        agent = _make_agent(config, skill_registry)
        suggestions = agent._generate_suggestions("Find all instruments", [])
        assert any("instrument" in s.lower() or "code" in s.lower() for s in suggestions)


class TestHumanFeedbackAgentNormalization:
    def test_normalize_exact_match(self, config, skill_registry):
        agent = _make_agent(config, skill_registry)
        result = agent._normalize_response("Technology", _CLASSIFICATION_OPTIONS)
        assert result == "Technology"

    def test_normalize_case_insensitive(self, config, skill_registry):
        agent = _make_agent(config, skill_registry)
        result = agent._normalize_response("technology", _CLASSIFICATION_OPTIONS)
        assert result == "Technology"

    def test_normalize_trims_whitespace(self, config, skill_registry):
        agent = _make_agent(config, skill_registry)
        result = agent._normalize_response("  Finance  ", _CLASSIFICATION_OPTIONS)
        assert result == "Finance"

    def test_normalize_free_text_no_options(self, config, skill_registry):
        agent = _make_agent(config, skill_registry)
        result = agent._normalize_response("Some custom text", None)
        assert result == "Some custom text"

    def test_normalize_unmatched_passes_through(self, config, skill_registry):
        agent = _make_agent(config, skill_registry)
        result = agent._normalize_response("Aerospace", _CLASSIFICATION_OPTIONS)
        assert result == "Aerospace"


class TestFriendlyProgress:
    def test_no_results_returns_default(self, config, skill_registry):
        agent = _make_agent(config, skill_registry)
        assert "No data sources" in agent._friendly_progress([])

    def test_single_agent_friendly_name(self, config, skill_registry):
        agent = _make_agent(config, skill_registry)
        results = [{"agent_name": "mesh_agent", "has_data": False}]
        progress = agent._friendly_progress(results)
        assert "Snowflake" in progress
        assert "mesh_agent" not in progress

    def test_multiple_agents_friendly_names(self, config, skill_registry):
        agent = _make_agent(config, skill_registry)
        results = [
            {"agent_name": "mesh_agent", "has_data": False},
            {"agent_name": "graphql_agent", "has_data": False},
        ]
        progress = agent._friendly_progress(results)
        assert "Snowflake" in progress
        assert "GraphQL" in progress


# ---------------------------------------------------------------------------
# PromptLoader tests for new methods
# ---------------------------------------------------------------------------

class TestPromptLoaderHIL:
    def test_classification_prompt(self, prompt_loader):
        result = prompt_loader.human_feedback_classification(
            query="Show me instruments in the sector",
            needed_fields="industry or sector",
        )
        assert "instruments in the sector" in result
        assert "industry or sector" in result

    def test_clarification_prompt(self, prompt_loader):
        result = prompt_loader.human_feedback_clarification(
            query="show me some data",
            ambiguity_reason="Your query is quite broad.",
        )
        assert "show me some data" in result
        assert "Your query is quite broad." in result

    def test_no_data_prompt(self, prompt_loader):
        result = prompt_loader.human_feedback_no_data(
            attempts_summary="- Snowflake: no records\n- GraphQL API: empty result",
        )
        assert "Snowflake" in result
        assert "GraphQL" in result
