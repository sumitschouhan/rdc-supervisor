"""Tool registry: application.yml refers to tools by name."""

from supervisor_template.tools.graphql import run_graphql_query
from supervisor_template.tools.metadata import search_catalog
from supervisor_template.tools.snowflake import run_snowflake_sql

TOOL_REGISTRY = {t.name: t for t in (run_snowflake_sql, run_graphql_query, search_catalog)}

__all__ = ["TOOL_REGISTRY", "run_graphql_query", "run_snowflake_sql", "search_catalog"]
