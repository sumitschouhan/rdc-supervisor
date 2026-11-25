"""In-memory stand-ins for Snowflake, the GraphQL API and the metadata catalog.
Replace the functions in snowflake.py / graphql.py / metadata.py with real clients."""

from __future__ import annotations

import re
from typing import Any

CATALOG: list[dict[str, Any]] = [
    {"name": "trades", "source_system": "snowflake", "description": "Trade executions",
     "columns": ["trade_id", "instrument", "trade_date", "quantity", "price", "side"],
     "keywords": ["trade", "trades", "execution", "buy", "sell", "volume"]},
    {"name": "positions", "source_system": "snowflake", "description": "Account positions / holdings",
     "columns": ["account_id", "instrument", "quantity", "market_value"],
     "keywords": ["position", "positions", "holding", "holdings", "portfolio", "exposure"]},
    {"name": "instruments", "source_system": "snowflake", "description": "Instrument reference data",
     "columns": ["instrument_code", "name", "type", "industry", "exchange"],
     "keywords": ["instrument", "instruments", "security", "securities", "stock", "equity", "bond"]},
    {"name": "issuers", "source_system": "snowflake", "description": "Issuer reference data",
     "columns": ["issuer_id", "name", "sector", "country"],
     "keywords": ["issuer", "issuers", "company", "companies", "counterparty"]},
    {"name": "instrumentsGql", "source_system": "graphql", "description": "Instrument data (fallback)",
     "columns": ["code", "name", "type", "industry", "exchange"],
     "keywords": ["instrument", "instruments", "security", "securities", "stock", "equity"]},
    {"name": "morningstarRatings", "source_system": "graphql", "description": "Morningstar ratings",
     "columns": ["fundCode", "fundName", "overallRating", "starRating", "categoryRank"],
     "keywords": ["morningstar", "rating", "ratings", "star", "fund", "etf"]},
    {"name": "pricing", "source_system": "graphql", "description": "Prices / market data",
     "columns": ["instrumentCode", "date", "closePrice", "bid", "ask"],
     "keywords": ["price", "prices", "pricing", "quote", "bid", "ask", "close", "nav"]},
]

SNOWFLAKE: dict[str, list[dict[str, Any]]] = {
    "trades": [
        {"trade_id": "T001", "instrument": "AAPL", "trade_date": "2026-11-16", "quantity": 100, "price": 241.5, "side": "BUY"},
        {"trade_id": "T002", "instrument": "GOOGL", "trade_date": "2026-11-17", "quantity": 50, "price": 198.2, "side": "SELL"},
        {"trade_id": "T003", "instrument": "MSFT", "trade_date": "2026-11-18", "quantity": 200, "price": 512.7, "side": "BUY"},
    ],
    "positions": [
        {"account_id": "ACC001", "instrument": "AAPL", "quantity": 500, "market_value": 120750.0},
        {"account_id": "ACC001", "instrument": "MSFT", "quantity": 120, "market_value": 61524.0},
        {"account_id": "ACC001", "instrument": "VFINX", "quantity": 300, "market_value": 182400.0},
    ],
    "instruments": [
        {"instrument_code": "AAPL", "name": "Apple Inc.", "type": "equity", "industry": "Technology", "exchange": "NASDAQ"},
        {"instrument_code": "MSFT", "name": "Microsoft Corp.", "type": "equity", "industry": "Technology", "exchange": "NASDAQ"},
    ],
    "issuers": [
        {"issuer_id": "ISS001", "name": "Apple Inc.", "sector": "Technology", "country": "US"},
        {"issuer_id": "ISS002", "name": "JPMorgan Chase", "sector": "Finance", "country": "US"},
    ],
}

GRAPHQL: dict[str, list[dict[str, Any]]] = {
    "instrumentsGql": [
        {"code": "AAPL", "name": "Apple Inc.", "type": "equity", "industry": "Technology", "exchange": "NASDAQ"},
        {"code": "NVDA", "name": "NVIDIA Corp.", "type": "equity", "industry": "Technology", "exchange": "NASDAQ"},
    ],
    "morningstarRatings": [
        {"fundCode": "VFINX", "fundName": "Vanguard 500 Index", "overallRating": 4, "starRating": 4, "categoryRank": 12},
        {"fundCode": "FXAIX", "fundName": "Fidelity 500 Index", "overallRating": 5, "starRating": 5, "categoryRank": 3},
    ],
    "pricing": [
        {"instrumentCode": "AAPL", "date": "2026-11-24", "closePrice": 241.5, "bid": 241.4, "ask": 241.6},
        {"instrumentCode": "MSFT", "date": "2026-11-24", "closePrice": 512.7, "bid": 512.5, "ask": 512.9},
        {"instrumentCode": "VFINX", "date": "2026-11-24", "closePrice": 608.0, "bid": 607.9, "ask": 608.1},
    ],
}

_LITERAL = re.compile(r"""['"]([A-Za-z0-9_.\- ]+)['"]""")


def filter_rows(rows: list[dict[str, Any]], query: str) -> list[dict[str, Any]]:
    """Naive mock filter: keep rows matching ANY quoted literal in the query."""
    literals = {lit.lower() for lit in _LITERAL.findall(query)}
    if not literals:
        return rows
    return [r for r in rows if any(str(v).lower() in literals for v in r.values())]
