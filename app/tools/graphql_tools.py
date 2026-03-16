from langchain_core.tools import tool


@tool
def execute_graphql_query(query: str) -> dict:
    """Execute a GraphQL query and return results. (Mock implementation)"""
    query_lower = query.lower()

    if "instrument" in query_lower:
        return {
            "data": {
                "instruments": [
                    {"code": "AAPL", "name": "Apple Inc.", "type": "equity", "industry": "Technology", "exchange": "NASDAQ"},
                    {"code": "GOOGL", "name": "Alphabet Inc.", "type": "equity", "industry": "Technology", "exchange": "NASDAQ"},
                ]
            }
        }
    elif "morningstar" in query_lower or "rating" in query_lower:
        return {
            "data": {
                "morningstarRatings": [
                    {"fundCode": "VFINX", "fundName": "Vanguard 500 Index", "overallRating": 4, "categoryRank": 12},
                    {"fundCode": "FXAIX", "fundName": "Fidelity 500 Index", "overallRating": 5, "categoryRank": 3},
                ]
            }
        }
    elif "pricing" in query_lower or "price" in query_lower:
        return {
            "data": {
                "pricing": [
                    {"instrumentCode": "AAPL", "date": "2025-01-16", "closePrice": 198.50, "openPrice": 196.00, "volume": 45000000},
                ]
            }
        }

    return {"data": None, "errors": [{"message": "No data found"}]}
