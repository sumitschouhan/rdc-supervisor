"""Deep Agents "analyst mode" (deepagents 0.7.x).

The supervisor graph stays the deterministic front door. For open-ended,
multi-step analytical questions ("compare...", "why did...", "analyse risk...")
the router picks mode="deep" and the `deep_analyst` node hands the question to
a Deep Agent that can plan, delegate in parallel, keep scratch files and
remember user preferences.

Deep Agents features used (and why):
- subagents (`task` tool)    mesh_analyst / graphql_analyst run in isolated context
                             windows and return a structured WorkerResult (JSON),
                             so the main agent's context stays small. Several
                             `task` calls in one turn run in parallel.
- skills (SKILL.md)          rendered from application.yml `skills:`. Progressive
                             disclosure: only name+description sit in the prompt;
                             the agent reads the full SKILL.md when it needs it.
                             Custom subagents do NOT inherit skills, so each gets
                             its own skills path.
- memory (AGENTS.md)         /guides/AGENTS.md = org data conventions (read-only)
                             /memories/AGENTS.md = per-user notes in a Store,
                             namespaced by user_id (never shared across users).
- CompositeBackend           StateBackend (thread scratch, checkpointed) by default,
                             StoreBackend for /memories/ (durable, cross-thread).
- permissions                deny writes to /skills/** and /guides/** (prompt-
                             injection hardening: the agent can't rewrite policy).
- interrupt_on (optional)    human approval of SQL before execution, with a
                             `when` predicate so only risky queries pause.
- context_schema             AppContext flows to subagents and tools unchanged.
- Built-in: summarization + large-tool-result offloading to files, Anthropic
  prompt caching, model-refusal handling, dangling tool-call repair.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from deepagents import FilesystemPermission, create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from deepagents.backends.utils import create_file_data
from langchain.agents.middleware import ToolCallLimitMiddleware, ToolErrorMiddleware
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

from supervisor_template.agents import _tool_error_to_model
from supervisor_template.config import AppConfig
from supervisor_template.context import AppContext
from supervisor_template.llm import ModelSet
from supervisor_template.schemas import WorkerResult
from supervisor_template.tools import TOOL_REGISTRY

SUBAGENT_SUFFIX = "_analyst"  # mesh_agent -> mesh_analyst


def skill_slug(name: str) -> str:
    """Agent Skills spec: lowercase letters, digits, hyphens; must equal dir name."""
    return re.sub(r"[^a-z0-9-]", "-", name.lower())


def render_skill_files(cfg: AppConfig) -> dict[str, str]:
    """application.yml skills -> /skills/<agent>/<skill>/SKILL.md (one source of truth)."""
    files: dict[str, str] = {}
    for name, s in cfg.skills.items():
        slug = skill_slug(name)
        examples = "\n".join(f"- Q: {ex['q']}\n  A: `{ex['a']}`" for ex in s.examples)
        files[f"/skills/{s.agent}/{slug}/SKILL.md"] = (
            f"---\nname: {slug}\n"
            f"description: {s.description}. Use for questions about {', '.join(s.keywords)}.\n---\n"
            f"# {slug}\n\n## Instructions\n{s.instructions or 'Follow the agent guidelines.'}\n\n"
            f"## Examples\n{examples or '- none'}\n"
        )
    return files


def _sql_needs_approval(request) -> bool:
    """Only pause for risky SQL: no LIMIT / no WHERE. Everything else auto-approves."""
    sql = str(request.tool_call["args"].get("sql", "")).lower()
    return " limit " not in f" {sql} " or " where " not in f" {sql} "


def build_deep_analyst(
    cfg: AppConfig,
    models: ModelSet,
    *,
    store: Optional[BaseStore] = None,
    checkpointer: Optional[BaseCheckpointSaver] = None,
    require_sql_approval: bool = False,
):
    store = store or InMemoryStore()  # use a persistent store (Postgres) in prod

    subagents = []
    for agent_name, acfg in cfg.agents.items():
        sub: dict[str, Any] = {
            "name": agent_name.replace("_agent", SUBAGENT_SUFFIX),
            "description": acfg.description + " Give it one focused data question.",
            "system_prompt": acfg.system_prompt
            + "\nRead the matching skill before writing the query. Return a short summary, not raw rows.",
            "tools": [TOOL_REGISTRY[t] for t in acfg.tools],
            "skills": [f"/skills/{agent_name}/"],     # NOT inherited from the parent
            "response_format": WorkerResult,          # parent receives compact JSON
            "middleware": [                           # NOT inherited from the parent
                ToolErrorMiddleware(on_error=_tool_error_to_model),
                ToolCallLimitMiddleware(run_limit=cfg.limits.worker_tool_calls_per_run, exit_behavior="end"),
            ],
        }
        if require_sql_approval and "run_snowflake_sql" in acfg.tools:
            sub["interrupt_on"] = {
                "run_snowflake_sql": {"allowed_decisions": ["approve", "edit", "reject"],
                                      "when": _sql_needs_approval},
            }
        subagents.append(sub)

    backend = CompositeBackend(
        default=StateBackend(),  # per-thread scratch files, checkpointed
        routes={
            # per-user durable memory; namespace from runtime context -> never shared
            "/memories/": StoreBackend(
                namespace=lambda rt: ("rdc-memory", (rt.context or AppContext()).user_id),
                store=store,
            ),
        },
    )

    return create_deep_agent(
        models.router,
        tools=[],  # main agent only plans/delegates; data access lives in subagents
        system_prompt=cfg.deep_analyst.system_prompt,
        subagents=subagents,
        memory=["/guides/AGENTS.md", "/memories/AGENTS.md"],
        backend=backend,
        permissions=[
            FilesystemPermission(operations=["write"], paths=["/skills/**", "/guides/**"], mode="deny"),
        ],
        context_schema=AppContext,
        checkpointer=checkpointer,
        name="deep_analyst",
    )


def deep_analyst_input(cfg: AppConfig, question: str, metadata: dict) -> dict:
    """Initial state: the question plus skill/guide files seeded into the StateBackend."""
    by_source: dict[str, list[str]] = {}
    for t in metadata.get("tables", []):
        by_source.setdefault(t["source_system"], []).append(t["name"])
    files = {**render_skill_files(cfg), "/guides/AGENTS.md": cfg.deep_analyst.guide}
    return {
        "messages": [{"role": "user", "content": (
            f"Question: {question}\n"
            + "".join(f"{src} tables: {', '.join(names)}\n" for src, names in by_source.items())
        )}],
        "files": {path: create_file_data(text) for path, text in files.items()},
    }
