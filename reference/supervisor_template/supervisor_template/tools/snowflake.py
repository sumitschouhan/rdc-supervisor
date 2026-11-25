from __future__ import annotations

import re
from typing import Any

from langchain.tools import ToolRuntime, tool

from supervisor_template.context import AppContext
from supervisor_template.tools._mock_data import SNOWFLAKE, filter_rows


@tool(response_format="content_and_artifact")
def run_snowflake_sql(sql: str, runtime: ToolRuntime[AppContext]) -> tuple[str, list[dict[str, Any]]]:
    """Execute a read-only Snowflake SQL query and return the rows."""
    # Defence in depth: the supervisor already filters by entitlement.
    if runtime.context and "snowflake" not in runtime.context.allowed_sources:
        raise PermissionError("user is not entitled to Snowflake")
    if not re.match(r"^\s*(select|with)\b", sql, re.I):
        raise ValueError("only read-only SELECT queries are allowed")

    # --- mock execution: replace with snowflake.connector / Snowpark ---
    table = next((t for t in SNOWFLAKE if re.search(rf"\b{t}\b", sql, re.I)), None)
    rows = filter_rows(SNOWFLAKE[table], sql) if table else []

    # `content` goes to the LLM (keep it short); the artifact (full rows) does NOT
    # go to the LLM - the worker node collects it from the ToolMessage.
    preview = rows[:5]
    return f"{len(rows)} row(s). Preview: {preview}", rows
