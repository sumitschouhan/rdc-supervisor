"""Specialist agents built with LangChain 1.x `create_agent`.

`create_agent` (langchain.agents) replaces the deprecated
`langgraph.prebuilt.create_react_agent`. It returns a compiled LangGraph
graph, so an agent can be (a) called from a node, (b) wrapped as a tool, or
(c) added directly as a subgraph node.

Cross-cutting behaviour is composed from middleware instead of hand-written
loops - each one replaces code you would otherwise own:
- ToolErrorMiddleware       tool exceptions -> error ToolMessage the model can fix
- ToolCallLimitMiddleware   cap tool calls per run (runaway-loop guard)
- ModelCallLimitMiddleware  cap model calls per run
- ModelFallbackMiddleware   retry the model call on a different model
- SkillPromptMiddleware     (ours) inject the selected skill into the prompt
Others worth knowing: HumanInTheLoopMiddleware (approve/edit tool calls),
SummarizationMiddleware (long threads), PIIMiddleware, ToolRetryMiddleware,
ModelRetryMiddleware, LLMToolSelectorMiddleware, ContextEditingMiddleware,
ModelRouterMiddleware (beta: pick a model per request).
"""

from __future__ import annotations

from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    ModelFallbackMiddleware,
    ToolCallLimitMiddleware,
    ToolErrorMiddleware,
)

from supervisor_template.config import AppConfig
from supervisor_template.context import AppContext
from supervisor_template.llm import ModelSet
from supervisor_template.schemas import MetadataResult, WorkerResult
from supervisor_template.skills import SkillPromptMiddleware
from supervisor_template.tools import TOOL_REGISTRY


def _tool_error_to_model(exc: Exception, request) -> str:  # noqa: ARG001
    # Surface only expected, safe error types to the model; anything else
    # propagates and is handled by the graph-level error_handler.
    if isinstance(exc, (ValueError, PermissionError)):
        return f"Tool error ({type(exc).__name__}): {exc}. Fix the query or report no data."
    return None


def _common_middleware(cfg: AppConfig, models: ModelSet) -> list:
    mw = [
        ToolErrorMiddleware(on_error=_tool_error_to_model),
        ToolCallLimitMiddleware(run_limit=cfg.limits.worker_tool_calls_per_run, exit_behavior="end"),
        ModelCallLimitMiddleware(run_limit=cfg.limits.worker_model_calls_per_run, exit_behavior="end"),
    ]
    if models.fallback is not None:
        mw.append(ModelFallbackMiddleware(models.fallback))
    return mw


def build_worker_agent(name: str, cfg: AppConfig, models: ModelSet):
    acfg = cfg.agents[name]
    return create_agent(
        models.worker,
        tools=[TOOL_REGISTRY[t] for t in acfg.tools],
        system_prompt=acfg.system_prompt,
        middleware=[SkillPromptMiddleware(), *_common_middleware(cfg, models)],
        # Pydantic class -> AutoStrategy: native structured output on Claude,
        # tool-call strategy on models without native support.
        response_format=WorkerResult,
        context_schema=AppContext,
        name=name,  # named agents show up as subagents in traces / v3 streams
    )


def build_metadata_agent(cfg: AppConfig, models: ModelSet):
    return create_agent(
        models.worker,
        tools=[TOOL_REGISTRY["search_catalog"]],
        system_prompt=cfg.prompts.metadata_agent,
        middleware=_common_middleware(cfg, models),
        response_format=MetadataResult,
        context_schema=AppContext,
        name="metadata_agent",
    )
