"""End-to-end graph behaviour with the offline demo model.

These run the real graph, real create_agent workers, real middleware,
Send fan-out, interrupts and checkpointing - only the LLM is scripted.
"""

import pytest
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from supervisor_template.context import AppContext


async def ask(graph, thread, text, ctx=None):
    return await graph.ainvoke({"messages": [HumanMessage(text)]}, thread,
                               context=ctx or AppContext(), version="v2")


def agents_run(out):
    return [(r["agent"], len(r["rows"])) for r in out.value["results"]]


async def test_single_agent(graph, thread):
    out = await ask(graph, thread, "Show me trades for AAPL")
    assert not out.interrupts
    assert agents_run(out) == [("mesh_agent", 1)]
    assert out.value["route"]["mode"] == "single"
    assert out.value["messages"][-1].name == "synthesizer"


async def test_parallel_cross_agent(graph, thread):
    out = await ask(graph, thread, "Show my positions and their Morningstar ratings")
    assert out.value["route"]["mode"] == "parallel"
    assert {a for a, _ in agents_run(out)} == {"mesh_agent", "graphql_agent"}


async def test_sequential_passes_prior_results(graph, thread):
    out = await ask(graph, thread, "For my top holdings then get current prices")
    assert out.value["route"]["mode"] == "sequential"
    mesh, gql = out.value["results"]
    assert (mesh["agent"], gql["agent"]) == ("mesh_agent", "graphql_agent")
    # graphql query was built from the instruments mesh returned
    for code in {r["instrument"] for r in mesh["rows"]}:
        assert code in gql["query"]


async def test_instrument_fallback_to_graphql(graph, thread):
    out = await ask(graph, thread, "Show instrument details for NVDA")
    assert agents_run(out) == [("mesh_agent", 0), ("graphql_agent", 1)]


async def test_no_fallback_when_skill_disallows(graph, thread):
    # trades skill has fallback: false -> no graphql attempt, ask the human instead
    out = await ask(graph, thread, "Show me trades for TSLA")
    assert out.interrupts and out.interrupts[0].value["kind"] == "no_data"
    state = (await graph.aget_state(thread)).values
    assert state["attempted"] == ["mesh_agent"]


async def test_clarification_interrupt_and_resume(graph, thread):
    out = await ask(graph, thread, "Show instruments in the sector")
    intr = out.interrupts[0]
    assert intr.value["kind"] == "clarification"
    assert "Technology" in intr.value["options"]
    assert intr.response_schema["required"] == ["answer"]  # typed form for the UI

    out = await graph.ainvoke(Command(resume={"answer": "Technology"}), thread,
                              context=AppContext(), version="v2")
    assert not out.interrupts
    assert agents_run(out) == [("mesh_agent", 2)]
    assert "clarification: Technology" in out.value["question"]


async def test_resume_value_is_validated(graph, thread):
    await ask(graph, thread, "Show instruments in the sector")
    with pytest.raises(Exception):
        await graph.ainvoke(Command(resume={"answer": ""}), thread, context=AppContext(), version="v2")


async def test_entitlements_restrict_routing(graph, thread):
    ctx = AppContext(allowed_sources=frozenset({"snowflake"}))
    out = await ask(graph, thread, "Show my positions and their Morningstar ratings", ctx)
    assert {a for a, _ in agents_run(out)} == {"mesh_agent"}


async def test_worker_crash_goes_to_error_handler(graph, thread, monkeypatch):
    import supervisor_template.tools._mock_data as data

    def boom(rows, query):
        raise RuntimeError("warehouse down")

    monkeypatch.setattr("supervisor_template.tools.snowflake.filter_rows", boom)
    out = await ask(graph, thread, "Show me trades for AAPL")
    # RetryPolicy retried, then error_handler recorded a failed result; the run
    # did not crash and the human is asked how to proceed.
    state = (await graph.aget_state(thread)).values
    assert state["results"][0]["error"].startswith("RuntimeError: warehouse down")
    assert out.interrupts[0].value["kind"] == "no_data"
    assert data.SNOWFLAKE  # untouched


async def test_multi_turn_memory_and_reset(graph, thread):
    await ask(graph, thread, "Show me trades for AAPL")
    out = await ask(graph, thread, "Show my positions")
    # per-turn channels were reset with Overwrite, conversation kept
    assert agents_run(out) == [("mesh_agent", 3)]
    humans = [m.text for m in out.value["messages"] if m.type == "human"]
    assert humans == ["Show me trades for AAPL", "Show my positions"]
