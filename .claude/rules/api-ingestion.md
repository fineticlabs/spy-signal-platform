---
description: API and data ingestion rules — loaded for ingestion and API code
globs:
  - src/ingestion/**
  - src/api/**
---

# API & Ingestion Rules

## Alpaca Data
- Use `TimeFrame` enum (not strings) for all database queries: `db.query_bars(timeframe=TimeFrame.ONE_MIN)`.
- Use `DataFeed` enum (not strings) for Alpaca websocket: `StockDataStream(feed=DataFeed.IEX)`.
- The `_FEED_MAP` dict in `websocket.py` converts config strings ("iex"/"sip") to `DataFeed` enums.
- Handle websocket disconnects with exponential backoff (base 1s, max 60s).
- Alpaca REST API: respect rate limits. Use `time.sleep()` between batch requests in backfill scripts only (not in async code — use `asyncio.sleep()` there).
- Skip bars with zero or negative prices during ingestion.
- All bars stored with UTC timestamps. Convert to ET only for display/time-of-day logic.

## Database
- SQLite with WAL mode for concurrent reads during live scanning.
- `insert_bars()` accepts `list[Bar]` — always pass Pydantic models, not raw dicts.
- `query_bars()` returns `list[Bar]` — timeframe parameter must be `TimeFrame` enum.

## FastAPI
- Internal API only (serves dashboard). Not exposed to internet.
- All endpoints return Pydantic models for serialization.
- Use dependency injection for database connections.
