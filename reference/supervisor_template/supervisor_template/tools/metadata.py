from __future__ import annotations

import json
import re

from langchain.tools import ToolRuntime, tool

from supervisor_template.context import AppContext
from supervisor_template.tools._mock_data import CATALOG


@tool
def search_catalog(keywords: str, runtime: ToolRuntime[AppContext]) -> str:
    """Search the data catalog. Returns matching tables/types with their source
    system and columns. `keywords` is a space-separated list of business terms."""
    words = set(re.findall(r"[a-z]+", keywords.lower()))
    allowed = runtime.context.allowed_sources if runtime.context else None
    hits = [
        {k: t[k] for k in ("name", "source_system", "description", "columns")}
        for t in CATALOG
        if words & set(t["keywords"]) and (allowed is None or t["source_system"] in allowed)
    ]
    return json.dumps(hits)
