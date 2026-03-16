from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Any, Optional


class ChatRequest(BaseModel):
    message: str = Field(..., description="User message text")
    thread_id: str = Field(..., description="Conversation thread identifier")


class ResumeRequest(BaseModel):
    thread_id: str = Field(..., description="Thread to resume")
    response: str = Field(..., description="User's response to the interrupt question")


class InterruptPayload(BaseModel):
    feedback_type: str
    question: str
    options: Optional[list[str]] = None
    context: str


class ChatResponse(BaseModel):
    thread_id: str
    response: Optional[str] = None
    interrupt: Optional[InterruptPayload] = None
    is_complete: bool = False
    agent_trace: list[str] = Field(
        default_factory=list,
        description="Ordered list of agents that executed",
    )


class HistoryResponse(BaseModel):
    thread_id: str
    messages: list[dict[str, Any]]
