from langchain_core.tools import tool


@tool
def execute_snowflake_query(sql: str) -> dict:
    """Execute a Snowflake SQL query and return results. (Mock implementation)"""
    sql_lower = sql.lower()

    if "trades" in sql_lower:
        return {
            "columns": ["trade_id", "instrument", "trade_date", "quantity", "price", "side"],
            "rows": [
                ["T001", "AAPL", "2025-01-15", 100, 198.50, "BUY"],
                ["T002", "GOOGL", "2025-01-15", 50, 175.20, "SELL"],
                ["T003", "MSFT", "2025-01-16", 200, 420.75, "BUY"],
            ],
            "row_count": 3,
        }
    elif "positions" in sql_lower:
        return {
            "columns": ["position_id", "account_id", "instrument", "quantity", "market_value"],
            "rows": [
                ["P001", "ACC001", "AAPL", 500, 99250.00],
                ["P002", "ACC001", "GOOGL", 200, 35040.00],
            ],
            "row_count": 2,
        }
    elif "instruments" in sql_lower:
        return {
            "columns": ["instrument_code", "name", "type", "industry", "exchange"],
            "rows": [
                ["AAPL", "Apple Inc.", "equity", "Technology", "NASDAQ"],
                ["MSFT", "Microsoft Corp.", "equity", "Technology", "NASDAQ"],
            ],
            "row_count": 2,
        }
    elif "issuers" in sql_lower:
        return {
            "columns": ["issuer_id", "name", "sector", "country"],
            "rows": [
                ["ISS001", "Apple Inc.", "Technology", "US"],
                ["ISS002", "JPMorgan Chase", "Finance", "US"],
            ],
            "row_count": 2,
        }

    return {"columns": [], "rows": [], "row_count": 0}
