from __future__ import annotations

import json
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

from app.agents.base import BaseAgent
from app.graph.state import AgentResult, SupervisorState
from app.prompts.loader import PromptLoader
from app.skills.registry import SkillRegistry
from app.tools.graphql_tools import execute_graphql_query


class GraphQLAgent(BaseAgent):
    def __init__(
        self,
        model: ChatAnthropic,
        prompt_loader: PromptLoader,
        skill_registry: SkillRegistry,
    ):
        self.tools = [execute_graphql_query]
        self.tool_map = {t.name: t for t in self.tools}
        self.model = model.bind_tools(self.tools)
        self.base_prompt = prompt_loader.graphql_agent_system()
        self.skill_registry = skill_registry

    def invoke(self, state: SupervisorState) -> dict:
        user_msg = self._get_latest_user_message(state)
        metadata = state.get("metadata_context") or {}
        active_skill_name = state.get("active_skill")

        # Build prompt: base + skill augmentation
        system_parts = [self.base_prompt]

        skill_used = None
        if active_skill_name:
            skill = self.skill_registry.get_skill(active_skill_name)
            if skill and skill.domain == "graphql":
                skill_used = skill.name
                tables_str = json.dumps(metadata.get("tables", []), indent=2)
                system_parts.append("\n--- Skill Context ---")
                system_parts.append(skill.render_prompt(tables=tables_str, query=user_msg))
                examples = skill.format_examples()
                if examples:
                    system_parts.append(examples)
        else:
            skills = self.skill_registry.find_skills(user_msg, domain="graphql")
            if skills:
                skill = skills[0]
                skill_used = skill.name
                tables_str = json.dumps(metadata.get("tables", []), indent=2)
                system_parts.append("\n--- Skill Context ---")
                system_parts.append(skill.render_prompt(tables=tables_str, query=user_msg))
                examples = skill.format_examples()
                if examples:
                    system_parts.append(examples)

        messages: list[Any] = [
            SystemMessage(content="\n".join(system_parts)),
            HumanMessage(
                content=(
                    f"User query: {user_msg}\n\n"
                    f"Available metadata:\n{json.dumps(metadata, indent=2)}"
                )
            ),
        ]

        response = self.model.invoke(messages)
        messages.append(response)

        query_generated = ""
        data = None
        has_data = False

        for tool_call in response.tool_calls:
            tool = self.tool_map[tool_call["name"]]
            query_generated = tool_call["args"].get("query", "")
            result = tool.invoke(tool_call["args"])

            messages.append(ToolMessage(
                content=json.dumps(result) if not isinstance(result, str) else result,
                tool_call_id=tool_call["id"],
            ))

            if isinstance(result, dict):
                result_data = result.get("data")
                if result_data and not result.get("errors"):
                    # Check if any field in data has actual records
                    for value in result_data.values():
                        if isinstance(value, list) and len(value) > 0:
                            data = result
                            has_data = True
                            break

        agent_result: AgentResult = {
            "agent_name": "graphql_agent",
            "skill_used": skill_used,
            "query_generated": query_generated,
            "data": data,
            "error": None if has_data else "No data returned",
            "has_data": has_data,
        }

        return {
            "agent_results": [agent_result],
            "messages": [AIMessage(content=f"[GraphQL Agent] Query: {query_generated} | Data found: {has_data}")],
            "iteration_count": state.get("iteration_count", 0) + 1,
        }

    def _get_latest_user_message(self, state: SupervisorState) -> str:
        for msg in reversed(state.get("messages", [])):
            if hasattr(msg, "type") and msg.type == "human":
                return msg.content
        return ""
