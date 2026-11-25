"""FastAPI surface.

- The graph is built once in `lifespan` (no module-level singletons).
- `ainvoke(..., version="v2")` returns a `GraphOutput` with `.value` and
  `.interrupts`, so detecting human-in-the-loop pauses is one attribute read.
- `/chat/stream` streams node progress + final answer as Server-Sent Events.
- Swap InMemorySaver for AsyncPostgresSaver in production (durable threads,
  survives restarts, enables resume across pods).

Run:  uvicorn supervisor_template.api:app --reload
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from pydantic import BaseModel

from supervisor_template.config import load_config
from supervisor_template.context import AppContext
from supervisor_template.graph import build_graph
from supervisor_template.llm import build_models


class SendRequest(BaseModel):
    thread_id: str
    message: str
    user_id: str = "anonymous"


class ResumeRequest(BaseModel):
    thread_id: str
    answer: str
    user_id: str = "anonymous"


class ChatResponse(BaseModel):
    thread_id: str
    status: str  # "complete" | "needs_input"
    answer: Optional[str] = None
    interrupt: Optional[dict[str, Any]] = None
    response_schema: Optional[dict[str, Any]] = None
    trace: list[dict[str, Any]] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config()
    # Production:
    #   from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    #   async with AsyncPostgresSaver.from_conn_string(DB_URI) as saver:
    #       await saver.setup(); app.state.graph = build_graph(cfg, models, checkpointer=saver); yield
    app.state.graph = build_graph(cfg, build_models(cfg), checkpointer=InMemorySaver())
    yield


app = FastAPI(title="Supervisor template", lifespan=lifespan)


def _context(user_id: str) -> AppContext:
    # In production derive this from the authenticated principal, not the body.
    return AppContext(user_id=user_id)


def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def _to_response(thread_id: str, out) -> ChatResponse:
    if out.interrupts:
        intr = out.interrupts[0]
        return ChatResponse(thread_id=thread_id, status="needs_input",
                            interrupt=intr.value, response_schema=intr.response_schema)
    state = out.value
    answer = next((m.text for m in reversed(state["messages"]) if isinstance(m, AIMessage)), None)
    trace = [{"agent": r["agent"], "query": r["query"], "rows": len(r["rows"]), "error": r["error"]}
             for r in state.get("results", [])]
    return ChatResponse(thread_id=thread_id, status="complete", answer=answer, trace=trace)


@app.post("/chat/send", response_model=ChatResponse)
async def send(req: SendRequest, request: Request) -> ChatResponse:
    graph = request.app.state.graph
    out = await graph.ainvoke({"messages": [HumanMessage(req.message)]}, _config(req.thread_id),
                              context=_context(req.user_id), version="v2")
    return _to_response(req.thread_id, out)


@app.post("/chat/resume", response_model=ChatResponse)
async def resume(req: ResumeRequest, request: Request) -> ChatResponse:
    graph = request.app.state.graph
    snapshot = await graph.aget_state(_config(req.thread_id))
    if not snapshot.interrupts:
        raise HTTPException(409, "thread is not waiting for input")
    out = await graph.ainvoke(Command(resume={"answer": req.answer}), _config(req.thread_id),
                              context=_context(req.user_id), version="v2")
    return _to_response(req.thread_id, out)


@app.post("/chat/stream")
async def stream(req: SendRequest, request: Request) -> StreamingResponse:
    graph = request.app.state.graph

    async def events():
        # version="v2": every chunk is a typed StreamPart dict {"type", "ns", "data"}.
        async for part in graph.astream({"messages": [HumanMessage(req.message)]}, _config(req.thread_id),
                                        context=_context(req.user_id), stream_mode=["updates"],
                                        version="v2"):
            for node, update in (part["data"] or {}).items():
                if node == "__interrupt__":
                    payload = {"type": "interrupt", "value": update[0].value}
                else:
                    payload = {"type": "node", "node": node}
                    if node == "synthesizer":
                        payload["answer"] = update["messages"][-1].text
                yield f"data: {json.dumps(payload, default=str)}\n\n"
        yield "data: {\"type\": \"done\"}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


@app.get("/chat/history/{thread_id}")
async def history(thread_id: str, request: Request) -> dict:
    snapshot = await request.app.state.graph.aget_state(_config(thread_id))
    msgs = snapshot.values.get("messages", []) if snapshot.values else []
    return {"thread_id": thread_id, "messages": [{"role": m.type, "content": m.text} for m in msgs]}


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
