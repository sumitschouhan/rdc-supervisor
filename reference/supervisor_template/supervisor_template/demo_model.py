"""Offline, deterministic stand-in for Claude.

Lets you run the whole graph (CLI, API, tests) with no API key. It implements
the real chat-model contract (`bind_tools`, tool calls, structured output via
tool calls) so the graph, `create_agent`, middleware, Send fan-out, interrupts
and checkpointing are exercised exactly as in production - only the "thinking"
is replaced by keyword rules.

It understands the message formats produced in graph.py
("Question: ...", "Tables: ...", "Prior results: ...", "Results: ...").
Not needed in a real project once you have evals against the real model.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any, Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool

SF_WORDS = ["trade", "position", "holding", "portfolio", "issuer", "instrument", "security", "stock"]
GQL_WORDS = ["morningstar", "rating", "price", "pricing", "quote", "bid", "ask"]
SECTORS = ["technology", "healthcare", "finance", "energy"]
_STOP = {"SQL", "AND", "THE", "ETF", "FOR", "ALL", "TOP", "ID"}


def _field(text: str, name: str) -> str:
    m = re.search(rf"^{name}: (.*?)(?=^\w[\w ]*: |\Z)", text, re.M | re.S)
    return m.group(1).strip() if m else ""


def _first_human(messages: list[BaseMessage]) -> str:
    return next((m.text for m in messages if isinstance(m, HumanMessage)), "")


def _literals(question: str, prior: str) -> list[str]:
    lits = [w for w in re.findall(r"\b[A-Z]{2,5}\b", question) if w not in _STOP]
    lits += re.findall(r"clarification: (\w+)", question, re.I)
    if prior and prior != "none":
        for row in (r for res in json.loads(prior) for r in res.get("rows", [])):
            lits += [row[k] for k in ("instrument", "instrument_code", "code") if k in row]
    return list(dict.fromkeys(lits))


# ---------------------------------------------------------------- tool args
def _snowflake_args(messages: list[BaseMessage]) -> dict:
    text = _first_human(messages)
    q = _field(text, "Question")
    table = (_field(text, "Tables").split(",")[0] or "trades").strip()
    lits = _literals(q, _field(text, "Prior results"))
    where = f" WHERE instrument IN ({', '.join(repr(x) for x in lits)})" if lits else ""
    return {"sql": f"SELECT * FROM {table}{where} LIMIT 100"}


def _graphql_args(messages: list[BaseMessage]) -> dict:
    text = _first_human(messages)
    q = _field(text, "Question")
    root = (_field(text, "Tables").split(",")[0] or "pricing").strip()
    lits = _literals(q, _field(text, "Prior results"))
    flt = f'(filter: {{codes: [{", ".join(json.dumps(x) for x in lits)}]}})' if lits else ""
    return {"query": f"query {{ {root}{flt} {{ ... }} }}"}


def _catalog_args(messages: list[BaseMessage]) -> dict:
    return {"keywords": _field(_first_human(messages), "Question")}


ARG_BUILDERS: dict[str, Callable[[list[BaseMessage]], dict]] = {
    "run_snowflake_sql": _snowflake_args,
    "run_graphql_query": _graphql_args,
    "search_catalog": _catalog_args,
    # patterns/subagents_as_tools.py supervisor: first step is always metadata
    "lookup_metadata": lambda messages: {"question": _first_human(messages)},
}


# ------------------------------------------------------ structured outputs
def _route(messages: list[BaseMessage]) -> dict:
    q = _field(_first_human(messages), "Question").lower()
    sf, gql = any(w in q for w in SF_WORDS), any(w in q for w in GQL_WORDS)
    if "sector" in q and not any(s in q for s in SECTORS) and "clarification:" not in q:
        return {"needs_clarification": True, "clarification_question": "Which sector do you mean?",
                "clarification_options": [s.title() for s in SECTORS], "reasoning": "sector not specified"}
    agents = (["mesh_agent"] if sf else []) + (["graphql_agent"] if gql else [])
    if sf and gql:
        mode = "sequential" if re.search(r"\bthen\b|\bfor (my|those|these)\b", q) else "parallel"
    else:
        mode = "single"
    return {"agents": agents or ["mesh_agent"], "mode": mode, "reasoning": "demo keyword routing"}


def _metadata(messages: list[BaseMessage]) -> dict:
    tool_msgs = [m for m in messages if isinstance(m, ToolMessage)]
    tables = json.loads(tool_msgs[-1].text) if tool_msgs else []
    return {"tables": tables, "notes": "demo"}


def _worker_result(messages: list[BaseMessage]) -> dict:
    call = next((tc for m in reversed(messages) if isinstance(m, AIMessage) for tc in m.tool_calls), None)
    tool_msg = next((m for m in reversed(messages) if isinstance(m, ToolMessage)), None)
    n = int(re.match(r"(\d+)", tool_msg.text).group(1)) if tool_msg and tool_msg.text[:1].isdigit() else 0
    return {"query": next(iter(call["args"].values())) if call else "", "has_data": n > 0,
            "summary": f"Returned {n} row(s)." if tool_msg and tool_msg.status != "error" else tool_msg.text if tool_msg else ""}


SCHEMA_FILLERS: dict[str, Callable[[list[BaseMessage]], dict]] = {
    "RouteDecision": _route,
    "MetadataResult": _metadata,
    "WorkerResult": _worker_result,
}


class DemoChatModel(BaseChatModel):
    bound_tools: list[dict] = []

    @property
    def _llm_type(self) -> str:
        return "demo"

    def bind_tools(self, tools, *, tool_choice=None, **kwargs: Any):  # noqa: ARG002
        return self.model_copy(update={"bound_tools": [convert_to_openai_tool(t) for t in tools]})

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:  # noqa: ARG002
        names = [t["function"]["name"] for t in self.bound_tools]
        schemas = [n for n in names if n in SCHEMA_FILLERS]
        actions = [n for n in names if n in ARG_BUILDERS]
        last = messages[-1]

        if isinstance(last, HumanMessage) and actions:  # first step of an agent loop
            name, args = actions[0], ARG_BUILDERS[actions[0]](messages)
        elif schemas:  # time to return the structured result
            name, args = schemas[0], SCHEMA_FILLERS[schemas[0]](messages)
        else:  # plain text answer (synthesizer)
            results = _field(_first_human(messages), "Results")
            return self._text(f"[demo answer]\n{results}")
        call = {"name": name, "args": args, "id": f"call_{uuid.uuid4().hex[:8]}", "type": "tool_call"}
        return ChatResult(generations=[ChatGeneration(message=AIMessage("", tool_calls=[call]))])

    @staticmethod
    def _text(text: str) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage(text))])
