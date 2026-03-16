from __future__ import annotations

from langchain_core.messages import AIMessage
from langgraph.types import interrupt

from app.agents.base import BaseAgent
from app.graph.state import SupervisorState
from app.prompts.loader import PromptLoader
from app.skills.registry import SkillRegistry


class HumanFeedbackAgent(BaseAgent):
    def __init__(self, prompt_loader: PromptLoader, skill_registry: SkillRegistry):
        self.prompt_loader = prompt_loader
        self.skill_registry = skill_registry

    def invoke(self, state: SupervisorState) -> dict:
        agent_results = state.get("agent_results", [])
        user_msg = self._get_latest_user_message(state)

        needs_classification = self._needs_classification(user_msg)

        if needs_classification:
            feedback_request = self._build_classification_request(user_msg)
        else:
            feedback_request = self._build_no_data_request(user_msg, agent_results)

        # Interrupt pauses the graph; resume value is returned here
        human_response = interrupt(feedback_request)

        return {
            "human_response": human_response,
            "pending_feedback": None,
            "messages": [
                AIMessage(content=f"[Human Feedback] User responded: {human_response}")
            ],
            "iteration_count": state.get("iteration_count", 0) + 1,
        }

    def _needs_classification(self, query: str) -> bool:
        """Check if query mentions classification concepts without specifics."""
        classification_keywords = ["industry", "sector", "instrument type", "classification", "category"]
        query_lower = query.lower()

        has_classification_concept = any(kw in query_lower for kw in classification_keywords)
        if not has_classification_concept:
            return False

        # Also check via classification skill
        skills = self.skill_registry.find_skills(query, domain="classification")
        return len(skills) > 0

    def _build_classification_request(self, query: str) -> dict:
        # Use classification skill for options if available
        skills = self.skill_registry.find_skills(query, domain="classification")
        options = ["Technology", "Healthcare", "Finance", "Energy", "Other"]

        return {
            "feedback_type": "classification",
            "question": self.prompt_loader.human_feedback_classification(
                query=query,
                needed_fields="industry classification or instrument type",
            ),
            "options": options,
            "context": "The query requires classification details to filter results.",
        }

    def _build_no_data_request(self, query: str, results: list) -> dict:
        attempts = "\n".join(
            f"- {r.get('agent_name', 'unknown')}: {r.get('error', 'no data')}"
            for r in results
        )
        return {
            "feedback_type": "no_data",
            "question": self.prompt_loader.human_feedback_no_data(
                attempts_summary=attempts,
            ),
            "options": None,
            "context": "All data agents returned no results.",
        }

    def _get_latest_user_message(self, state: SupervisorState) -> str:
        for msg in reversed(state.get("messages", [])):
            if hasattr(msg, "type") and msg.type == "human":
                return msg.content
        return ""
