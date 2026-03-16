from langchain_core.tools import tool


MOCK_SCHEMAS = {
    "trades": {
        "table": "trades",
        "columns": [
            {"name": "trade_id", "type": "VARCHAR", "description": "Unique trade identifier"},
            {"name": "instrument", "type": "VARCHAR", "description": "Financial instrument code"},
            {"name": "trade_date", "type": "DATE", "description": "Date of trade execution"},
            {"name": "quantity", "type": "NUMBER", "description": "Number of units traded"},
            {"name": "price", "type": "DECIMAL(18,4)", "description": "Price per unit"},
            {"name": "side", "type": "VARCHAR", "description": "BUY or SELL"},
            {"name": "account_id", "type": "VARCHAR", "description": "Account identifier"},
        ],
        "source_system": "snowflake",
    },
    "positions": {
        "table": "positions",
        "columns": [
            {"name": "position_id", "type": "VARCHAR", "description": "Unique position identifier"},
            {"name": "account_id", "type": "VARCHAR", "description": "Account holding the position"},
            {"name": "instrument", "type": "VARCHAR", "description": "Financial instrument code"},
            {"name": "quantity", "type": "NUMBER", "description": "Current position size"},
            {"name": "market_value", "type": "DECIMAL(18,4)", "description": "Current market value"},
        ],
        "source_system": "snowflake",
    },
    "instruments": {
        "table": "instruments",
        "columns": [
            {"name": "instrument_code", "type": "VARCHAR", "description": "Unique instrument code"},
            {"name": "name", "type": "VARCHAR", "description": "Instrument name"},
            {"name": "type", "type": "VARCHAR", "description": "equity, bond, fund, etc."},
            {"name": "industry", "type": "VARCHAR", "description": "Industry classification"},
            {"name": "exchange", "type": "VARCHAR", "description": "Listed exchange"},
        ],
        "source_system": "both",  # available in Snowflake (primary) and GraphQL (fallback)
    },
    "issuers": {
        "table": "issuers",
        "columns": [
            {"name": "issuer_id", "type": "VARCHAR", "description": "Unique issuer identifier"},
            {"name": "name", "type": "VARCHAR", "description": "Issuer/company name"},
            {"name": "sector", "type": "VARCHAR", "description": "Business sector"},
            {"name": "country", "type": "VARCHAR", "description": "Country of incorporation"},
        ],
        "source_system": "snowflake",
    },
    "morningstar_ratings": {
        "table": "morningstar_ratings",
        "columns": [
            {"name": "fund_code", "type": "String", "description": "Fund identifier"},
            {"name": "fund_name", "type": "String", "description": "Fund name"},
            {"name": "overall_rating", "type": "Int", "description": "Overall Morningstar rating 1-5"},
            {"name": "category_rank", "type": "Int", "description": "Rank within category"},
        ],
        "source_system": "graphql",
    },
    "pricing": {
        "table": "pricing",
        "columns": [
            {"name": "instrument_code", "type": "String", "description": "Instrument identifier"},
            {"name": "date", "type": "Date", "description": "Price date"},
            {"name": "close_price", "type": "Float", "description": "Closing price"},
            {"name": "open_price", "type": "Float", "description": "Opening price"},
            {"name": "volume", "type": "Int", "description": "Trading volume"},
        ],
        "source_system": "graphql",
    },
}

ALL_TABLES = [
    {"name": "trades", "description": "Trade execution records", "source": "snowflake"},
    {"name": "positions", "description": "Current portfolio positions", "source": "snowflake"},
    {"name": "instruments", "description": "Instrument reference data", "source": "both"},
    {"name": "issuers", "description": "Issuer/company reference data", "source": "snowflake"},
    {"name": "morningstar_ratings", "description": "Morningstar fund ratings and analytics", "source": "graphql"},
    {"name": "pricing", "description": "Market pricing and quote data", "source": "graphql"},
]


@tool
def lookup_table_schema(table_name: str) -> dict:
    """Look up the schema for a given table name. Returns columns, types, and source system."""
    return MOCK_SCHEMAS.get(table_name, {"error": f"Table '{table_name}' not found"})


@tool
def search_tables_by_keyword(keyword: str) -> list[dict]:
    """Search for tables whose name or description matches a keyword."""
    keyword_lower = keyword.lower()
    return [
        t for t in ALL_TABLES
        if keyword_lower in t["name"] or keyword_lower in t["description"].lower()
    ]
