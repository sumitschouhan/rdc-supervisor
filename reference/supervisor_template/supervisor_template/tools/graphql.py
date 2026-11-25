from __future__ import annotations

import re
from typing import Any

from langchain.tools import ToolRuntime, tool

from supervisor_template.context import AppContext
from supervisor_template.tools._mock_data import GRAPHQL, filter_rows


@tool(response_format="content_and_artifact")
def run_graphql_query(query: str, runtime: ToolRuntime[AppContext]) -> tuple[str, list[dict[str, Any]]]:
    """Execute a GraphQL query against the market-data API and return the rows."""
    if runtime.context and "graphql" not in runtime.context.allowed_sources:
        raise PermissionError("user is not entitled to the GraphQL API")
    if re.search(r"\bmutation\b", query, re.I):
        raise ValueError("mutations are not allowed")

    # --- mock execution: replace with httpx.AsyncClient / gql client ---
    root = next((k for k in GRAPHQL if re.search(rf"\b{k}\b", query)), None)
    rows = filter_rows(GRAPHQL[root], query) if root else []
    return f"{len(rows)} row(s). Preview: {rows[:5]}", rows
