"""Pydantic schemas for every LLM decision and every human interaction.

Rule of thumb for the template: an LLM never returns free text that code has
to parse. Routing, metadata and worker outcomes are structured outputs
(`with_structured_output` / `create_agent(response_format=...)`), and the
human-in-the-loop answer is validated with `interrupt(..., response_schema=...)`.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

SourceSystem = Literal["snowflake", "graphql"]


# ---------------------------------------------------------------- metadata
class TableInfo(BaseModel):
    name: str
    source_system: SourceSystem
    description: str = ""
    columns: list[str] = Field(default_factory=list)


class MetadataResult(BaseModel):
    """Structured output of the metadata agent."""

    tables: list[TableInfo] = Field(default_factory=list)
    notes: str = ""

    @property
    def source_systems(self) -> set[str]:
        return {t.source_system for t in self.tables}


# ----------------------------------------------------------------- routing
class RouteDecision(BaseModel):
    """Structured output of the router LLM."""

    agents: list[str] = Field(
        default_factory=list,
        description="Agent names to run, in execution order for sequential mode.",
    )
    mode: Literal["single", "parallel", "sequential", "deep"] = "single"
    skill: Optional[str] = Field(None, description="Best matching skill name, if any.")
    needs_clarification: bool = False
    clarification_question: Optional[str] = None
    clarification_options: list[str] = Field(default_factory=list)
    reasoning: str = Field("", description="One sentence explaining the decision.")


# ------------------------------------------------------------------ workers
class WorkerResult(BaseModel):
    """Structured output every data agent returns (rows travel as tool artifacts)."""

    query: str = Field(description="The exact SQL/GraphQL that was executed.")
    summary: str = Field(description="One or two sentences describing what was found.")
    has_data: bool


# ------------------------------------------------------- human-in-the-loop
class ClarificationRequest(BaseModel):
    """Payload surfaced to the client by interrupt()."""

    kind: Literal["clarification", "no_data"]
    question: str
    options: list[str] = Field(default_factory=list)


class ClarificationAnswer(BaseModel):
    """Expected resume value. Passed as interrupt(response_schema=...), so the
    client gets a JSON schema to render a form and LangGraph validates the resume."""

    answer: str = Field(min_length=1)
