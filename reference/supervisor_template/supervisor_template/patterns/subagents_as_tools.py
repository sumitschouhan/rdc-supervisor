"""ALTERNATIVE pattern: tool-calling supervisor ("subagents as tools").

This is the pattern the LangChain docs now recommend for general multi-agent
supervisors (the `langgraph-supervisor` package / `create_supervisor` is no
longer actively maintained). The supervisor is itself a `create_agent`; each
specialist is wrapped as a tool. The model decides what to call, and calling
two tools in one turn runs them in parallel automatically.

When to choose which (see README "Choosing a pattern"):
- graph.py (explicit StateGraph): hard business rules, audited routing,
  fixed fallbacks, typed HITL, per-node retries/timeouts.  <- rdc-supervisor
- this file: open-ended assistants, fewer rules, fastest to extend; routing
  quality depends on the model + tool descriptions.

Both reuse the same worker agents, tools, skills and config.
"""

from __future__ import annotations

import json
from typing import Optional

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware, SummarizationMiddleware
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.types import interrupt

from supervisor_template.agents import build_metadata_agent, build_worker_agent
from supervisor_template.config import AppConfig
from supervisor_template.context import AppContext
from supervisor_template.llm import ModelSet
from supervisor_template.schemas import ClarificationAnswer, ClarificationRequest
from supervisor_template.skills import SkillRegistry

SUPERVISOR_PROMPT = """You coordinate data specialists to answer financial data questions.
1. Always call lookup_metadata first.
2. Call the specialist tool(s) whose source system appears in the metadata.
   Call several in the SAME turn when their parts are independent; call them
   one after another when a later one needs earlier results.
3. If the question is ambiguous, call ask_user.
4. Answer only from tool results and cite the source system."""


def build_tool_calling_supervisor(
    cfg: AppConfig, models: ModelSet, *, checkpointer: Optional[BaseCheckpointSaver] = None
):
    skills = SkillRegistry(cfg.skills)
    metadata_agent = build_metadata_agent(cfg, models)

    @tool
    async def lookup_metadata(question: str, runtime: ToolRuntime[AppContext]) -> str:
        """Find relevant tables/types and their source systems for a question."""
        out = await metadata_agent.ainvoke({"messages": [HumanMessage(f"Question: {question}")]},
                                           context=runtime.context)
        return out["structured_response"].model_dump_json()

    def wrap(name: str):
        agent = build_worker_agent(name, cfg, models)

        @tool(f"ask_{name}", description=cfg.agents[name].description)
        async def ask(question: str, runtime: ToolRuntime[AppContext]) -> str:
            skill = skills.match(question, name)
            out = await agent.ainvoke(
                {"messages": [HumanMessage(f"Question: {question}")], "skill_prompt": skills.render(skill)},
                context=runtime.context,
            )
            rows = [r for m in out["messages"] if isinstance(m, ToolMessage)
                    and isinstance(m.artifact, list) for r in m.artifact]
            sr = out.get("structured_response")
            return json.dumps({"agent": name, "query": sr.query if sr else "", "rows": rows[:50]}, default=str)

        return ask

    @tool
    def ask_user(question: str, options: list[str] | None = None) -> str:
        """Ask the user a clarifying question and return their answer."""
        req = ClarificationRequest(kind="clarification", question=question, options=options or [])
        return interrupt(req.model_dump(), response_schema=ClarificationAnswer).answer

    return create_agent(
        models.router,
        tools=[lookup_metadata, *(wrap(n) for n in cfg.agents), ask_user],
        system_prompt=SUPERVISOR_PROMPT,
        middleware=[
            ModelCallLimitMiddleware(run_limit=cfg.limits.max_supervisor_iterations * 2, exit_behavior="end"),
            # Keeps long chat threads within budget by summarising old turns.
            SummarizationMiddleware(model=models.synthesizer, trigger=("tokens", 100_000), keep=("messages", 20)),
        ],
        context_schema=AppContext,
        checkpointer=checkpointer,
        name="supervisor",
    )
