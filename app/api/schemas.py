from __future__ import annotations

from pydantic import BaseModel, Field, field_validator
from typing import Any, Optional


class ChatRequest(BaseModel):
    message: str = Field(..., description="User message text")
    thread_id: str = Field(..., description="Conversation thread identifier")


class ResumeRequest(BaseModel):
    thread_id: str = Field(..., description="Thread to resume")
    response: str = Field(..., description="User's response to the interrupt question")

    @field_validator("response")
    @classmethod
    def response_not_empty(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("Response cannot be empty. Please provide a meaningful answer.")
        return stripped


class InterruptPayload(BaseModel):
    feedback_type: str  # "classification" | "clarification" | "no_data"
    title: str = Field(default="Input Required", description="Short, user-facing title")
    question: str
    options: Optional[list[str]] = None
    suggestions: Optional[list[str]] = None
    allow_free_text: bool = True
    context: str
    progress: Optional[str] = None


class ChatResponse(BaseModel):
    thread_id: str
    response: Optional[str] = None
    interrupt: Optional[InterruptPayload] = None
    is_complete: bool = False
    agent_trace: list[str] = Field(
        default_factory=list,
        description="Ordered list of agents that executed",
    )
    agent_summary: Optional[str] = Field(
        default=None,
        description="Human-readable description of what the system did",
    )


class HistoryResponse(BaseModel):
    thread_id: str
    messages: list[dict[str, Any]]
