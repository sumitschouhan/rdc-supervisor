from __future__ import annotations

import json
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from app.agents.base import BaseAgent
from app.graph.state import SupervisorState
from app.prompts.loader import PromptLoader


class SynthesizerAgent(BaseAgent):
    def __init__(self, model: ChatAnthropic, prompt_loader: PromptLoader):
        self.model = model
        self.system_prompt = prompt_loader.synthesizer_system()

    def invoke(self, state: SupervisorState) -> dict:
        user_msg = self._get_latest_user_message(state)
        results = state.get("agent_results", [])

        # Build a summary of all agent results for the synthesizer
        results_summary = []
        for r in results:
            if r.get("has_data"):
                results_summary.append(
                    f"Source: {r['agent_name']} (skill: {r.get('skill_used', 'none')})\n"
                    f"Query: {r.get('query_generated', 'N/A')}\n"
                    f"Data: {json.dumps(r.get('data', {}), indent=2)}"
                )

        if not results_summary:
            return {
                "messages": [AIMessage(content="I was unable to find any data for your query. Please try rephrasing your question.")],
                "is_complete": True,
            }

        messages: list[Any] = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(
                content=(
                    f"User query: {user_msg}\n\n"
                    f"Data results:\n{'---'.join(results_summary)}"
                )
            ),
        ]

        response = self.model.invoke(messages)

        return {
            "messages": [AIMessage(content=response.content)],
            "is_complete": True,
        }

    def _get_latest_user_message(self, state: SupervisorState) -> str:
        for msg in reversed(state.get("messages", [])):
            if hasattr(msg, "type") and msg.type == "human":
                return msg.content
        return ""
