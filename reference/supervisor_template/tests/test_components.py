import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from supervisor_template.config import AppConfig
from supervisor_template.context import AppContext
from supervisor_template.llm import build_models, structured
from supervisor_template.patterns.subagents_as_tools import build_tool_calling_supervisor
from supervisor_template.schemas import RouteDecision
from supervisor_template.skills import SkillRegistry
from supervisor_template.tools import run_snowflake_sql


def test_config_validates_cross_references(cfg):
    raw = cfg.model_dump()
    raw["skills"]["bad"] = {**raw["skills"]["trade_query"], "agent": "nope"}
    with pytest.raises(ValueError, match="unknown agent"):
        AppConfig(**raw)


def test_skill_matching_and_render(cfg):
    reg = SkillRegistry(cfg.skills)
    assert reg.match("total trade volume") == "trade_query"
    assert reg.match("morningstar rating", agent="graphql_agent") == "morningstar_query"
    assert reg.match("hello") is None
    assert reg.allows_fallback("instrument_query") and not reg.allows_fallback("trade_query")
    assert "## Skill: pricing_query" in reg.render("pricing_query")


def test_real_models_omit_sampling_params(cfg):
    m = build_models(cfg, demo=False)
    payload = m.worker._get_request_payload([HumanMessage("hi")])
    assert "temperature" not in payload
    assert payload["output_config"]["effort"] == cfg.models.worker.effort


def test_tool_artifacts_and_guardrails():
    # ToolRuntime is injected by the agent at run time; unit-test the function directly.
    rt = SimpleNamespace(context=AppContext())
    content, rows = run_snowflake_sql.func("SELECT * FROM trades WHERE instrument = 'AAPL'", rt)
    assert content.startswith("1 row(s)") and rows[0]["instrument"] == "AAPL"
    with pytest.raises(ValueError):
        run_snowflake_sql.func("DELETE FROM trades", rt)
    with pytest.raises(PermissionError):
        run_snowflake_sql.func("SELECT 1", SimpleNamespace(context=AppContext(allowed_sources=frozenset())))


async def test_router_structured_output(models):
    d = await structured(models.router, RouteDecision).ainvoke(
        [HumanMessage("Question: positions and morningstar ratings")])
    assert isinstance(d, RouteDecision) and d.mode == "parallel"


async def test_subagents_as_tools_pattern_smoke(cfg, models):
    agent = build_tool_calling_supervisor(cfg, models, checkpointer=InMemorySaver())
    out = await agent.ainvoke({"messages": [HumanMessage("Show me trades")]},
                              {"configurable": {"thread_id": "x"}}, context=AppContext())
    tool_msgs = [m for m in out["messages"] if m.type == "tool"]
    assert tool_msgs and json.loads(tool_msgs[0].content)["tables"]
