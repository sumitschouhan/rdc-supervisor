"""Tests for mock tools."""

from app.tools.metadata_tools import lookup_table_schema, search_tables_by_keyword
from app.tools.mesh_tools import execute_snowflake_query
from app.tools.graphql_tools import execute_graphql_query


def test_lookup_table_schema_found():
    result = lookup_table_schema.invoke({"table_name": "trades"})
    assert result["table"] == "trades"
    assert len(result["columns"]) > 0
    assert result["source_system"] == "snowflake"


def test_lookup_table_schema_not_found():
    result = lookup_table_schema.invoke({"table_name": "nonexistent"})
    assert "error" in result


def test_search_tables_by_keyword():
    result = search_tables_by_keyword.invoke({"keyword": "trade"})
    assert len(result) > 0
    assert result[0]["name"] == "trades"


def test_search_tables_by_keyword_no_match():
    result = search_tables_by_keyword.invoke({"keyword": "xyz123"})
    assert len(result) == 0


def test_execute_snowflake_trades():
    result = execute_snowflake_query.invoke({"sql": "SELECT * FROM trades"})
    assert result["row_count"] == 3


def test_execute_snowflake_positions():
    result = execute_snowflake_query.invoke({"sql": "SELECT * FROM positions"})
    assert result["row_count"] == 2


def test_execute_snowflake_no_data():
    result = execute_snowflake_query.invoke({"sql": "SELECT * FROM unknown_table"})
    assert result["row_count"] == 0


def test_execute_graphql_instruments():
    result = execute_graphql_query.invoke({"query": "{ instruments { code } }"})
    assert result["data"] is not None
    assert len(result["data"]["instruments"]) > 0


def test_execute_graphql_morningstar():
    result = execute_graphql_query.invoke({"query": "{ morningstarRatings { fundCode } }"})
    assert result["data"]["morningstarRatings"] is not None


def test_execute_graphql_pricing():
    result = execute_graphql_query.invoke({"query": "{ pricing { closePrice } }"})
    assert result["data"]["pricing"] is not None


def test_execute_graphql_no_data():
    result = execute_graphql_query.invoke({"query": "{ unknown { field } }"})
    assert result.get("errors") is not None


def test_instruments_in_both_sources():
    """Instruments should be available in both Snowflake and GraphQL."""
    schema = lookup_table_schema.invoke({"table_name": "instruments"})
    assert schema["source_system"] == "both"

    sf_result = execute_snowflake_query.invoke({"sql": "SELECT * FROM instruments"})
    assert sf_result["row_count"] > 0

    gql_result = execute_graphql_query.invoke({"query": "{ instruments { code } }"})
    assert gql_result["data"] is not None
