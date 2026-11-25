"""The supervisor graph (LangGraph 1.2).

    START -> start_turn -> supervisor ─┬─> metadata_agent ──────────┐
                              ▲        ├─> Send(mesh_agent, task) ──┤  (parallel fan-out
                              │        ├─> Send(graphql_agent, task)┤   or one-by-one)
                              │        ├─> human_review (interrupt) ┤
                              └────────┴────────────────────────────┘
                                       └─> synthesizer -> END

Why this shape ("deterministic shell, LLM core"):
- Business rules that must ALWAYS hold (metadata first, entitlements,
  instrument fallback, loop limits) live in code -> testable, auditable.
- The LLM only makes the judgement call (which agents? parallel or
  sequential? need clarification?) via a validated `RouteDecision`.
- Workers are `create_agent` agents invoked through `Send`, each with a
  private `WorkerTask` input. Parallel = several Sends in one Command.
- Every node returns `Command(goto=..., update=...)`, so routing lives next
  to the logic that decides it (no separate routing.py to keep in sync).

LangGraph 1.x features used: Command/Send, context_schema + Runtime,
input_schema per node, RetryPolicy, node `timeout`, node `error_handler`
(NodeError), interrupt(response_schema=...), Overwrite, durable checkpointer.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.errors import NodeError
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import Command, Overwrite, RetryPolicy, Send, interrupt

from supervisor_template.agents import build_metadata_agent, build_worker_agent
from supervisor_template.config import AppConfig
from supervisor_template.context import AppContext
from supervisor_template.llm import ModelSet, structured
from supervisor_template.schemas import (
    ClarificationAnswer,
    ClarificationRequest,
    MetadataResult,
    RouteDecision,
)
from supervisor_template.skills import SkillRegistry
from supervisor_template.state import AgentResult, OrchestratorState, WorkerTask


def _result(agent: str, *, skill=None, query="", summary="", rows=None, error=None) -> AgentResult:
    rows = rows or []
    return AgentResult(agent=agent, skill=skill, query=query, summary=summary,
                       rows=rows, has_data=bool(rows), error=error)


def build_graph(
    cfg: AppConfig,
    models: ModelSet,
    *,
    checkpointer: Optional[BaseCheckpointSaver] = None,
):
    skills = SkillRegistry(cfg.skills)
    metadata_agent = build_metadata_agent(cfg, models)
    workers = {name: build_worker_agent(name, cfg, models) for name in cfg.agents}
    router = structured(models.router, RouteDecision)
    limits = cfg.limits

    agent_catalog = "\n".join(
        f"- {n} (source_system={a.source_system}): {a.description}" for n, a in cfg.agents.items()
    )
    router_prompt = cfg.prompts.router.format(agent_catalog=agent_catalog, skill_catalog=skills.catalog())

    # ------------------------------------------------------------ helpers
    def task_for(state: OrchestratorState, agent: str, skill: Optional[str]) -> WorkerTask:
        return WorkerTask(
            question=state["question"],
            skill=skill if skill and skills.get(skill).agent == agent else skills.match(state["question"], agent),
            metadata=state["metadata"],
            prior_results=list(state.get("results", [])),
        )

    def available_agents(state: OrchestratorState, ctx: AppContext) -> list[str]:
        """Agents whose source is both in the metadata and allowed for this user."""
        sources = MetadataResult(**state["metadata"]).source_systems & ctx.allowed_sources
        return [n for n, a in cfg.agents.items() if a.source_system in sources]

    # -------------------------------------------------------------- nodes
    async def start_turn(state: OrchestratorState) -> dict:
        question = next(m.text for m in reversed(state["messages"]) if isinstance(m, HumanMessage))
        # Overwrite bypasses the operator.add reducer -> clean slate each turn,
        # while `messages` (conversation memory) is kept.
        return {"question": question, "metadata": None, "route": None, "plan": [],
                "results": Overwrite([]), "attempted": Overwrite([]),
                "iterations": 0, "clarifications": 0}

    async def supervisor(state: OrchestratorState, runtime: Runtime[AppContext]) -> Command:
        ctx = runtime.context or AppContext()
        it = state.get("iterations", 0) + 1
        results, attempted = state.get("results", []), state.get("attempted", [])
        route = state.get("route") or {}

        # 0. Hard loop guard.
        if it > limits.max_supervisor_iterations:
            return Command(goto="synthesizer", update={"iterations": it})

        # 1. Metadata first (rule lives in code, not in a prompt).
        if state.get("metadata") is None:
            return Command(goto="metadata_agent", update={"iterations": it})

        # 2. Sequential plan in progress -> dispatch the next agent with prior results.
        if state.get("plan"):
            nxt, *rest = state["plan"]
            return Command(goto=Send(nxt, task_for(state, nxt, route.get("skill"))),
                           update={"plan": rest, "iterations": it})

        # 3. Have data -> answer.
        if any(r["has_data"] for r in results):
            return Command(goto="synthesizer", update={"iterations": it})

        # 4. Ran agents but got nothing -> fallback agent, else ask the human.
        if results:
            allowed = set(available_agents(state, ctx))
            fallbacks = [
                fb for a in attempted
                if (fb := cfg.agents[a].fallback) and fb in allowed and fb not in attempted
                and skills.allows_fallback(route.get("skill") or skills.match(state["question"], a))
            ]
            if fallbacks:
                return Command(goto=[Send(fb, task_for(state, fb, None)) for fb in fallbacks],
                               update={"iterations": it})
            if state.get("clarifications", 0) < limits.max_clarifications:
                return Command(goto="human_review", update={"iterations": it, "route": None})
            return Command(goto="synthesizer", update={"iterations": it})

        # 5. Fresh question -> LLM routing decision (validated, then constrained).
        allowed = available_agents(state, ctx)
        if not allowed:
            return Command(goto="human_review", update={"iterations": it, "route": None})
        decision: RouteDecision = await router.ainvoke([
            SystemMessage(router_prompt),
            HumanMessage(
                f"Question: {state['question']}\n"
                f"Available agents: {', '.join(allowed)}\n"
                f"Metadata: {json.dumps(state['metadata'])}"
            ),
        ])
        update: dict[str, Any] = {"iterations": it, "route": decision.model_dump()}

        if decision.needs_clarification and state.get("clarifications", 0) < limits.max_clarifications:
            return Command(goto="human_review", update=update)

        targets = [a for a in dict.fromkeys(decision.agents) if a in allowed] or allowed[:1]
        if decision.mode == "sequential" and len(targets) > 1:
            first, *rest = targets
            update["plan"] = rest
            return Command(goto=Send(first, task_for(state, first, decision.skill)), update=update)
        if decision.mode == "single":
            targets = targets[:1]
        return Command(goto=[Send(t, task_for(state, t, decision.skill)) for t in targets], update=update)

    async def metadata_node(state: OrchestratorState, runtime: Runtime[AppContext]) -> dict:
        out = await metadata_agent.ainvoke(
            {"messages": [HumanMessage(f"Question: {state['question']}")]},
            context=runtime.context,
        )
        md: MetadataResult = out.get("structured_response") or MetadataResult()
        return {"metadata": md.model_dump()}

    def make_worker_node(name: str, agent):
        async def worker(task: WorkerTask, runtime: Runtime[AppContext]) -> dict:
            tables = [t["name"] for t in task["metadata"]["tables"]
                      if t["source_system"] == cfg.agents[name].source_system]
            prior = [r for r in task["prior_results"] if r["has_data"]]
            prompt = (
                f"Question: {task['question']}\n"
                f"Tables: {', '.join(tables)}\n"
                f"Metadata: {json.dumps(task['metadata'])}\n"
                f"Prior results: {json.dumps(prior) if prior else 'none'}"
            )
            out = await agent.ainvoke(
                {"messages": [HumanMessage(prompt)], "skill_prompt": skills.render(task["skill"])},
                context=runtime.context,
            )
            # Rows come from tool artifacts (ground truth), never from LLM text.
            rows = [row for m in out["messages"]
                    if isinstance(m, ToolMessage) and isinstance(m.artifact, list) for row in m.artifact]
            sr = out.get("structured_response")
            return {
                "results": [_result(name, skill=task["skill"], rows=rows,
                                    query=sr.query if sr else "", summary=sr.summary if sr else "")],
                "attempted": [name],
            }

        worker.__name__ = name
        return worker

    def worker_error_handler(state: Any, error: NodeError) -> Command:
        """Runs after the node's RetryPolicy is exhausted (or on timeout).
        Records a failed result instead of crashing the whole run."""
        msg = f"{type(error.error).__name__}: {error.error}"
        return Command(goto="supervisor", update={
            "results": [_result(error.node, error=msg, summary="agent failed")],
            "attempted": [error.node],
        })

    def human_review(state: OrchestratorState) -> Command:
        # Keep code BEFORE interrupt() side-effect free: the node re-runs on resume.
        route = state.get("route") or {}
        if route.get("needs_clarification"):
            req = ClarificationRequest(kind="clarification",
                                       question=route.get("clarification_question") or "Can you clarify?",
                                       options=route.get("clarification_options") or [])
        else:
            attempts = ", ".join(f"{r['agent']} ({r['error'] or 'no rows'})"
                                 for r in state.get("results", [])) or "metadata lookup"
            req = ClarificationRequest(kind="no_data", question=cfg.prompts.no_data_question.format(
                question=state["question"], attempts=attempts))

        # response_schema: clients get a JSON schema for the form, and the resume
        # value is validated and returned as a ClarificationAnswer instance.
        answer: ClarificationAnswer = interrupt(req.model_dump(), response_schema=ClarificationAnswer)

        return Command(goto="supervisor", update={
            "messages": [AIMessage(req.question, name="human_review"), HumanMessage(answer.answer)],
            "question": f"{state['question']} (clarification: {answer.answer})",
            "metadata": state.get("metadata") if req.kind == "clarification" else None,
            "route": None, "plan": [],
            "results": Overwrite([]), "attempted": Overwrite([]),
            "clarifications": state.get("clarifications", 0) + 1,
        })

    async def synthesizer(state: OrchestratorState) -> dict:
        results = state.get("results", [])
        if not any(r["has_data"] for r in results):
            text = "I couldn't find data for that question. Try adding identifiers or a time period."
        else:
            payload = [{k: r[k] for k in ("agent", "query", "summary", "rows", "error")} for r in results]
            resp = await models.synthesizer.ainvoke([
                SystemMessage(cfg.prompts.synthesizer),
                HumanMessage(f"Question: {state['question']}\n\nResults: {json.dumps(payload, default=str)}"),
            ])
            text = resp.text
        return {"messages": [AIMessage(text, name="synthesizer")]}

    # -------------------------------------------------------------- wiring
    worker_names = tuple(workers)
    retry = RetryPolicy(max_attempts=limits.worker_max_attempts)

    builder = StateGraph(OrchestratorState, context_schema=AppContext)
    builder.add_node("start_turn", start_turn)
    builder.add_node("supervisor", supervisor,
                     destinations=("metadata_agent", "human_review", "synthesizer", *worker_names))
    builder.add_node("metadata_agent", metadata_node, retry_policy=retry,
                     timeout=limits.worker_timeout_seconds)
    for name, agent in workers.items():
        builder.add_node(
            name,
            make_worker_node(name, agent),
            input_schema=WorkerTask,                 # private input via Send
            retry_policy=retry,                      # transient failures
            timeout=limits.worker_timeout_seconds,   # async nodes only
            error_handler=worker_error_handler,      # after retries are exhausted
        )
        builder.add_edge(name, "supervisor")
    builder.add_node("human_review", human_review, destinations=("supervisor",))
    builder.add_node("synthesizer", synthesizer)

    builder.add_edge(START, "start_turn")
    builder.add_edge("start_turn", "supervisor")
    builder.add_edge("metadata_agent", "supervisor")
    builder.add_edge("synthesizer", END)

    return builder.compile(checkpointer=checkpointer, name="rdc_supervisor")
