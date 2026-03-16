from __future__ import annotations

from fastapi import APIRouter, HTTPException
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from app.api.schemas import (
    ChatRequest,
    ChatResponse,
    HistoryResponse,
    InterruptPayload,
    ResumeRequest,
)
from app.graph.supervisor import get_graph

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/send", response_model=ChatResponse)
async def send_message(req: ChatRequest) -> ChatResponse:
    """Send a new user message. May return a response or an interrupt."""
    graph = get_graph()
    if graph is None:
        raise HTTPException(status_code=503, detail="Graph not initialized")

    config = {"configurable": {"thread_id": req.thread_id}}

    result = graph.invoke(
        {
            "messages": [HumanMessage(content=req.message)],
            "iteration_count": 0,
            "is_complete": False,
            "agent_results": [],
            "metadata_context": None,
            "current_agent": "",
            "route_reasoning": "",
            "active_skill": None,
            "pending_feedback": None,
            "human_response": None,
        },
        config=config,
    )

    return _build_response(result, req.thread_id, graph, config)


@router.post("/resume", response_model=ChatResponse)
async def resume_after_interrupt(req: ResumeRequest) -> ChatResponse:
    """Resume execution after a human-in-the-loop interrupt."""
    graph = get_graph()
    if graph is None:
        raise HTTPException(status_code=503, detail="Graph not initialized")

    config = {"configurable": {"thread_id": req.thread_id}}

    result = graph.invoke(
        Command(resume=req.response),
        config=config,
    )

    return _build_response(result, req.thread_id, graph, config)


@router.get("/history/{thread_id}", response_model=HistoryResponse)
async def get_history(thread_id: str) -> HistoryResponse:
    """Retrieve conversation history for a thread."""
    graph = get_graph()
    if graph is None:
        raise HTTPException(status_code=503, detail="Graph not initialized")

    config = {"configurable": {"thread_id": thread_id}}
    state = graph.get_state(config)

    messages = []
    for msg in state.values.get("messages", []):
        messages.append({
            "role": getattr(msg, "type", "unknown"),
            "content": getattr(msg, "content", ""),
        })

    return HistoryResponse(thread_id=thread_id, messages=messages)


def _build_response(result: dict, thread_id: str, graph, config: dict) -> ChatResponse:
    """Convert graph result to API response, detecting interrupts."""

    # Check for interrupt via graph state
    state = graph.get_state(config)

    if state.tasks:
        # There are pending tasks (interrupt happened)
        for task in state.tasks:
            if hasattr(task, "interrupts") and task.interrupts:
                interrupt_val = task.interrupts[0].value
                if isinstance(interrupt_val, dict):
                    return ChatResponse(
                        thread_id=thread_id,
                        interrupt=InterruptPayload(**interrupt_val),
                        is_complete=False,
                    )

    # Normal completion
    messages = result.get("messages", [])
    last_ai = None
    for msg in reversed(messages):
        if hasattr(msg, "type") and msg.type == "ai":
            content = msg.content
            # Skip internal agent trace messages
            if not content.startswith("["):
                last_ai = content
                break

    if last_ai is None:
        # Fall back to any AI message
        for msg in reversed(messages):
            if hasattr(msg, "type") and msg.type == "ai":
                last_ai = msg.content
                break

    agent_trace = [
        r.get("agent_name", "unknown")
        for r in result.get("agent_results", [])
    ]

    return ChatResponse(
        thread_id=thread_id,
        response=last_ai or "No response generated.",
        is_complete=True,
        agent_trace=agent_trace,
    )
