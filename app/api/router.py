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

_AGENT_LABELS: dict[str, str] = {
    "metadata_agent": "data catalog lookup",
    "mesh_agent": "Snowflake query",
    "graphql_agent": "GraphQL query",
    "cross_agent_dispatcher": "multi-source query",
    "human_feedback": "user input",
    "synthesizer": "response generation",
}

_INITIAL_STATE: dict = {
    "iteration_count": 0,
    "is_complete": False,
    "agent_results": [],
    "metadata_context": None,
    "current_agent": "",
    "route_reasoning": "",
    "active_skill": None,
    "pending_feedback": None,
    "human_response": None,
    "hil_attempt_count": 0,
    "last_feedback_question": None,
    "execution_mode": None,
    "cross_agent_targets": None,
    "pending_agents": None,
}


@router.post("/send", response_model=ChatResponse)
async def send_message(req: ChatRequest) -> ChatResponse:
    """Send a new user message. May return a response or an interrupt."""
    graph = get_graph()
    if graph is None:
        raise HTTPException(status_code=503, detail="Graph not initialized")

    config = {"configurable": {"thread_id": req.thread_id}}

    result = graph.invoke(
        {**_INITIAL_STATE, "messages": [HumanMessage(content=req.message)]},
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

    # Validate that there is actually a pending interrupt for this thread
    state = graph.get_state(config)
    if not state.tasks or not any(
        hasattr(t, "interrupts") and t.interrupts for t in state.tasks
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "No pending interrupt found for this thread. "
                "Use POST /chat/send to start a new conversation."
            ),
        )

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
        role = getattr(msg, "type", "unknown")
        content = getattr(msg, "content", "")
        # Skip internal agent trace messages
        if role == "ai" and isinstance(content, str) and content.startswith("["):
            continue
        messages.append({"role": role, "content": content})

    return HistoryResponse(thread_id=thread_id, messages=messages)


def _build_response(result: dict, thread_id: str, graph, config: dict) -> ChatResponse:
    """Convert graph result to API response, detecting interrupts."""
    state = graph.get_state(config)

    if state.tasks:
        for task in state.tasks:
            if hasattr(task, "interrupts") and task.interrupts:
                interrupt_val = task.interrupts[0].value
                if isinstance(interrupt_val, dict):
                    payload = InterruptPayload(**interrupt_val)
                    return ChatResponse(
                        thread_id=thread_id,
                        interrupt=payload,
                        is_complete=False,
                    )

    # Normal completion — extract the last meaningful AI message
    messages = result.get("messages", [])
    last_ai = _extract_final_response(messages)

    agent_results = result.get("agent_results", [])
    agent_trace = [r.get("agent_name", "unknown") for r in agent_results]
    agent_summary = _build_agent_summary(agent_trace)

    return ChatResponse(
        thread_id=thread_id,
        response=last_ai or "No response generated.",
        is_complete=True,
        agent_trace=agent_trace,
        agent_summary=agent_summary,
    )


def _extract_final_response(messages: list) -> str | None:
    """Return the last AI message that is not an internal trace."""
    for msg in reversed(messages):
        if hasattr(msg, "type") and msg.type == "ai":
            content = msg.content
            if isinstance(content, str) and not content.startswith("["):
                return content
    # Fall back to any AI message
    for msg in reversed(messages):
        if hasattr(msg, "type") and msg.type == "ai":
            return msg.content
    return None


def _build_agent_summary(agent_trace: list[str]) -> str | None:
    """Build a human-readable sentence summarising what the system did."""
    if not agent_trace:
        return None
    labels = [_AGENT_LABELS.get(a, a) for a in agent_trace]
    unique_labels = list(dict.fromkeys(labels))  # preserve order, deduplicate
    if len(unique_labels) == 1:
        return f"Completed via {unique_labels[0]}."
    steps = ", ".join(unique_labels[:-1]) + f", and {unique_labels[-1]}"
    return f"Processed through: {steps}."
