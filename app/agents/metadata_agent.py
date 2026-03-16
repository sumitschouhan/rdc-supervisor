from __future__ import annotations

import json
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

from app.agents.base import BaseAgent
from app.graph.state import MetadataContext, SupervisorState
from app.prompts.loader import PromptLoader
from app.tools.metadata_tools import lookup_table_schema, search_tables_by_keyword


class MetadataAgent(BaseAgent):
    def __init__(self, model: ChatAnthropic, prompt_loader: PromptLoader):
        self.tools = [lookup_table_schema, search_tables_by_keyword]
        self.tool_map = {t.name: t for t in self.tools}
        self.model = model.bind_tools(self.tools)
        self.system_prompt = prompt_loader.metadata_agent_system()

    def invoke(self, state: SupervisorState) -> dict:
        user_msg = self._get_latest_user_message(state)

        messages: list[Any] = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"User query: {user_msg}"),
        ]

        # Agentic tool-calling loop (max 3 rounds)
        tables: list[dict] = []
        fields: list[dict] = []
        source_systems: set[str] = set()
        data_availability: dict[str, bool] = {"snowflake": False, "graphql": False}

        for _ in range(3):
            response = self.model.invoke(messages)
            messages.append(response)

            if not response.tool_calls:
                break

            for tool_call in response.tool_calls:
                tool = self.tool_map[tool_call["name"]]
                result = tool.invoke(tool_call["args"])

                messages.append(ToolMessage(
                    content=json.dumps(result) if not isinstance(result, str) else result,
                    tool_call_id=tool_call["id"],
                ))

                # Process results
                if isinstance(result, dict) and "columns" in result:
                    tables.append(result)
                    fields.extend(result.get("columns", []))
                    src = result.get("source_system", "unknown")
                    source_systems.add(src)
                    if src in ("snowflake", "both"):
                        data_availability["snowflake"] = True
                    if src in ("graphql", "both"):
                        data_availability["graphql"] = True
                elif isinstance(result, list):
                    for item in result:
                        src = item.get("source", "unknown")
                        source_systems.add(src)
                        if src in ("snowflake", "both"):
                            data_availability["snowflake"] = True
                        if src in ("graphql", "both"):
                            data_availability["graphql"] = True

        # Determine overall source system
        if data_availability["snowflake"] and data_availability["graphql"]:
            overall_source = "both"
        elif data_availability["graphql"]:
            overall_source = "graphql"
        elif data_availability["snowflake"]:
            overall_source = "snowflake"
        else:
            overall_source = "unknown"

        metadata_context: MetadataContext = {
            "tables": tables,
            "fields": fields,
            "source_system": overall_source,
            "data_availability": data_availability,
            "lookup_query": user_msg,
        }

        return {
            "metadata_context": metadata_context,
            "iteration_count": state.get("iteration_count", 0) + 1,
        }

    def _get_latest_user_message(self, state: SupervisorState) -> str:
        for msg in reversed(state.get("messages", [])):
            if hasattr(msg, "type") and msg.type == "human":
                return msg.content
        return ""
