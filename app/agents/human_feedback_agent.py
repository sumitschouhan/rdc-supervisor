from __future__ import annotations

from langchain_core.messages import AIMessage
from langgraph.types import interrupt

from app.agents.base import BaseAgent
from app.graph.state import SupervisorState
from app.prompts.loader import PromptLoader
from app.skills.registry import SkillRegistry

_AGENT_FRIENDLY_NAMES: dict[str, str] = {
    "mesh_agent": "Snowflake",
    "graphql_agent": "GraphQL API",
    "cross_agent_dispatcher": "Snowflake and GraphQL API",
    "metadata_agent": "data catalog",
}

_CLASSIFICATION_OPTIONS: list[str] = [
    "Technology",
    "Healthcare",
    "Finance",
    "Energy",
    "Consumer Goods",
    "Industrials",
    "Other",
]

_VAGUE_QUERY_SIGNALS: list[str] = [
    "some", "any", "something", "stuff", "things", "data", "info", "information",
    "details", "show me", "give me", "find", "get",
]


class HumanFeedbackAgent(BaseAgent):
    def __init__(self, prompt_loader: PromptLoader, skill_registry: SkillRegistry):
        self.prompt_loader = prompt_loader
        self.skill_registry = skill_registry

    def invoke(self, state: SupervisorState) -> dict:
        agent_results = state.get("agent_results", [])
        user_msg = self._get_latest_user_message(state)
        hil_count = state.get("hil_attempt_count", 0)

        if self._needs_classification(user_msg):
            feedback_request = self._build_classification_request(user_msg)
        elif self._is_ambiguous_query(user_msg, agent_results):
            feedback_request = self._build_clarification_request(user_msg, agent_results)
        else:
            feedback_request = self._build_no_data_request(user_msg, agent_results)

        human_response = interrupt(feedback_request)
        normalized = self._normalize_response(human_response, feedback_request.get("options"))

        return {
            "human_response": normalized,
            "pending_feedback": None,
            "last_feedback_question": feedback_request["question"],
            "hil_attempt_count": hil_count + 1,
            "messages": [
                AIMessage(content=f"[Human Feedback] Received: {normalized}")
            ],
            "iteration_count": state.get("iteration_count", 0) + 1,
        }

    # --- Detection helpers ---

    def _needs_classification(self, query: str) -> bool:
        """True when the query asks about a sector/category without specifying which one."""
        classification_keywords = [
            "industry", "sector", "instrument type", "classification", "category"
        ]
        query_lower = query.lower()
        if not any(kw in query_lower for kw in classification_keywords):
            return False
        skills = self.skill_registry.find_skills(query, domain="classification")
        return len(skills) > 0

    def _is_ambiguous_query(self, query: str, results: list) -> bool:
        """True when query succeeded in routing but came back with nothing AND query is vague."""
        if results:
            return False
        query_lower = query.lower()
        vague_count = sum(1 for s in _VAGUE_QUERY_SIGNALS if s in query_lower)
        return vague_count >= 2 or len(query.split()) <= 4

    # --- Request builders ---

    def _build_classification_request(self, query: str) -> dict:
        return {
            "feedback_type": "classification",
            "title": "Classification Needed",
            "question": self.prompt_loader.human_feedback_classification(
                query=query,
                needed_fields="industry or sector",
            ),
            "options": _CLASSIFICATION_OPTIONS,
            "suggestions": [
                f'Show me all {opt.lower()} instruments'
                for opt in _CLASSIFICATION_OPTIONS[:3]
            ],
            "allow_free_text": False,
            "context": (
                "Your query mentions a sector or classification category, "
                "but doesn't specify which one. Please select from the list."
            ),
            "progress": "Analyzed your query and identified a missing classification filter.",
        }

    def _build_clarification_request(self, query: str, results: list) -> dict:
        progress = self._friendly_progress(results)
        return {
            "feedback_type": "clarification",
            "title": "Could You Clarify?",
            "question": self.prompt_loader.human_feedback_clarification(
                query=query,
                ambiguity_reason=(
                    "Your query is quite broad and I'm not sure exactly what data you need."
                ),
            ),
            "options": None,
            "suggestions": [
                "Show me trades from the last 7 days",
                "List all positions for account ACC001",
                "Get the latest price for AAPL",
                "Show instruments in the Technology sector",
            ],
            "allow_free_text": True,
            "context": "A more specific query will help me find exactly what you need.",
            "progress": progress or "Reviewed your query but need more detail to proceed.",
        }

    def _build_no_data_request(self, query: str, results: list) -> dict:
        progress = self._friendly_progress(results)
        attempts_summary = self._friendly_attempts(results)
        return {
            "feedback_type": "no_data",
            "title": "No Results Found",
            "question": self.prompt_loader.human_feedback_no_data(
                attempts_summary=attempts_summary,
            ),
            "options": None,
            "suggestions": self._generate_suggestions(query, results),
            "allow_free_text": True,
            "context": (
                "I searched our available data sources but found no matching records. "
                "Please refine your query or try a different approach."
            ),
            "progress": progress,
        }

    # --- Formatting helpers ---

    def _friendly_progress(self, results: list) -> str:
        """Build a plain-English progress summary from agent results."""
        if not results:
            return "No data sources were searched yet."
        sources_checked = []
        for r in results:
            name = _AGENT_FRIENDLY_NAMES.get(r.get("agent_name", ""), r.get("agent_name", ""))
            if name and name not in sources_checked:
                sources_checked.append(name)
        if not sources_checked:
            return "Searched available data sources."
        sources_str = " and ".join(sources_checked)
        return f"Searched {sources_str} — no matching records found."

    def _friendly_attempts(self, results: list) -> str:
        """Build a user-friendly summary of what was tried (no internal agent names)."""
        if not results:
            return "- No data sources were queried."
        lines = []
        for r in results:
            source = _AGENT_FRIENDLY_NAMES.get(r.get("agent_name", ""), "Data source")
            error = r.get("error") or "no matching records"
            lines.append(f"- {source}: {error}")
        return "\n".join(lines)

    def _generate_suggestions(self, query: str, results: list) -> list[str]:
        """Suggest alternative query approaches based on what failed."""
        suggestions = []
        query_lower = query.lower()

        if "trade" in query_lower or "trades" in query_lower:
            suggestions.extend([
                "Try a more recent date range (e.g. last 30 days)",
                "Specify a particular instrument or account",
            ])
        elif "position" in query_lower or "holding" in query_lower:
            suggestions.extend([
                "Specify a portfolio or account ID",
                "Check if you mean current or historical positions",
            ])
        elif "instrument" in query_lower or "security" in query_lower:
            suggestions.extend([
                "Try searching by instrument code (e.g. AAPL, MSFT)",
                "Specify an industry sector to narrow results",
            ])
        else:
            suggestions.extend([
                "Add a specific time period (e.g. last week, Q1 2025)",
                "Include a filter like account ID, instrument code, or sector",
            ])

        suggestions.append("Rephrase the query with more specific criteria")
        return suggestions[:3]

    @staticmethod
    def _normalize_response(response: str, options: list[str] | None) -> str:
        """Trim whitespace and, if options exist, match case-insensitively."""
        cleaned = response.strip()
        if not options:
            return cleaned
        cleaned_lower = cleaned.lower()
        for opt in options:
            if opt.lower() == cleaned_lower:
                return opt
        return cleaned

    def _get_latest_user_message(self, state: SupervisorState) -> str:
        for msg in reversed(state.get("messages", [])):
            if hasattr(msg, "type") and msg.type == "human":
                return msg.content
        return ""
