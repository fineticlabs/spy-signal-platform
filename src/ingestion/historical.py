from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import structlog
from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.models.bars import BarSet
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame as AlpacaTimeFrame

from src.config import get_alpaca_settings
from src.models import Bar, TimeFrame
from src.storage.database import BarDatabase

logger = structlog.get_logger(__name__)

_ALPACA_TIMEFRAME_MAP: dict[TimeFrame, AlpacaTimeFrame] = {
    TimeFrame.ONE_MIN: AlpacaTimeFrame.Minute,
    TimeFrame.FIVE_MIN: AlpacaTimeFrame(5, AlpacaTimeFrame.Minute),
    TimeFrame.FIFTEEN_MIN: AlpacaTimeFrame(15, AlpacaTimeFrame.Minute),
    TimeFrame.THIRTY_MIN: AlpacaTimeFrame(30, AlpacaTimeFrame.Minute),
    TimeFrame.DAILY: AlpacaTimeFrame.Day,
}

_FEED_MAP: dict[str, DataFeed] = {
    "iex": DataFeed.IEX,
    "sip": DataFeed.SIP,
}


def fetch_historical_bars(
    symbol: str,
    days: int,
    timeframe: TimeFrame = TimeFrame.ONE_MIN,
    db: BarDatabase | None = None,
) -> list[Bar]:
    """Fetch N days of bars for *symbol* from Alpaca REST API and persist to SQLite.

    Alpaca returns at most 10 000 bars per page; this function handles pagination
    automatically and inserts each page as it arrives so memory stays bounded.

    Args:
        symbol:    Ticker symbol (e.g. "SPY").
        days:      Number of calendar days to look back.
        timeframe: Bar timeframe (default 1-min).
        db:        Optional :class:`BarDatabase` instance.  A new one is created
                   when *None* is passed.

    Returns:
        All bars fetched, as a list of :class:`~src.models.Bar`.
    """
    settings = get_alpaca_settings()
    client = StockHistoricalDataClient(
        api_key=settings.api_key,
        secret_key=settings.secret_key,
    )

    end = datetime.now(tz=UTC)
    start = end - timedelta(days=days)

    alpaca_tf = _ALPACA_TIMEFRAME_MAP[timeframe]
    data_feed = _FEED_MAP.get(settings.feed, DataFeed.IEX)

    logger.info(
        "fetching_historical_bars",
        symbol=symbol,
        days=days,
        timeframe=timeframe.value,
        start=start.isoformat(),
        end=end.isoformat(),
    )

    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=alpaca_tf,
        start=start,
        end=end,
        feed=data_feed,
    )

    bar_db = db if db is not None else BarDatabase()
    all_bars: list[Bar] = []

    try:
        response: BarSet | dict[str, Any] = client.get_stock_bars(request)
    except Exception as exc:
        logger.error("alpaca_request_failed", symbol=symbol, error=str(exc))
        raise

    if isinstance(response, BarSet):
        raw_bars = response.data.get(symbol, [])
    else:
        raw_bars = response.get(symbol, [])

    page_bars: list[Bar] = []

    for raw in raw_bars:
        bar = Bar(
            symbol=symbol,
            timeframe=timeframe,
            timestamp=raw.timestamp.replace(tzinfo=UTC)
            if raw.timestamp.tzinfo is None
            else raw.timestamp,
            open=Decimal(str(raw.open)),
            high=Decimal(str(raw.high)),
            low=Decimal(str(raw.low)),
            close=Decimal(str(raw.close)),
            volume=int(raw.volume),
            vwap=Decimal(str(raw.vwap)) if raw.vwap is not None else Decimal(str(raw.close)),
        )
        page_bars.append(bar)

    if page_bars:
        try:
            bar_db.insert_bars(page_bars)
        except sqlite3.Error as exc:
            logger.error("db_insert_failed", count=len(page_bars), error=str(exc))
            raise
        all_bars.extend(page_bars)

    logger.info(
        "historical_fetch_complete",
        symbol=symbol,
        total_bars=len(all_bars),
        timeframe=timeframe.value,
    )
    return all_bars
