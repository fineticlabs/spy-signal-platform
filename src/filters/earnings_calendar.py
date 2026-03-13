"""Earnings calendar filter — block ORB trades around earnings announcements.

On earnings day and the day after, the affected ticker is blocked from ORB
trading.  The post-earnings gap and volatility make the opening range
unreliable for breakout setups.

For backtesting: fetches earnings dates from yfinance and caches them locally
in ``data/earnings_cache.json``.
For live trading: checks earnings dates daily at startup.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)

_CACHE_PATH = Path("data/earnings_cache.json")


def _fetch_earnings_dates(symbol: str, limit: int = 100) -> list[date]:
    """Fetch historical earnings announcement dates from yfinance.

    Args:
        symbol: Ticker symbol (e.g. ``"TSLA"``).
        limit:  Maximum number of earnings dates to request.

    Returns:
        Sorted list of :class:`~datetime.date` objects (ascending).
    """
    import yfinance as yf

    ticker = yf.Ticker(symbol)
    try:
        df = ticker.get_earnings_dates(limit=limit)
    except Exception:
        logger.warning("earnings_fetch_failed", symbol=symbol)
        return []

    if df is None or df.empty:
        return []

    # Index is tz-aware Timestamp in America/New_York
    dates: list[date] = sorted({ts.date() for ts in df.index})
    return dates


def load_earnings_cache() -> dict[str, list[str]]:
    """Load the local earnings date cache from disk.

    Returns:
        Dict mapping symbol → list of ISO date strings.
        Empty dict if cache file does not exist.
    """
    if not _CACHE_PATH.exists():
        return {}
    with _CACHE_PATH.open() as f:
        data: dict[str, list[str]] = json.load(f)
    return data


def save_earnings_cache(cache: dict[str, list[str]]) -> None:
    """Persist the earnings date cache to disk.

    Args:
        cache: Dict mapping symbol → list of ISO date strings.
    """
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _CACHE_PATH.open("w") as f:
        json.dump(cache, f, indent=2, sort_keys=True)
    logger.info("earnings_cache_saved", path=str(_CACHE_PATH), symbols=len(cache))


def get_earnings_dates(
    symbol: str,
    *,
    use_cache: bool = True,
    refresh: bool = False,
) -> set[date]:
    """Get earnings announcement dates for a symbol.

    Loads from cache if available; fetches from yfinance and updates cache
    if not cached or ``refresh=True``.

    Args:
        symbol:    Ticker symbol.
        use_cache: Whether to use the local cache (default ``True``).
        refresh:   Force re-fetch even if cached (default ``False``).

    Returns:
        Set of :class:`~datetime.date` for all known earnings dates.
    """
    cache = load_earnings_cache() if use_cache else {}

    if symbol in cache and not refresh:
        return {date.fromisoformat(d) for d in cache[symbol]}

    logger.info("fetching_earnings_dates", symbol=symbol)
    dates = _fetch_earnings_dates(symbol)

    if dates:
        cache[symbol] = [d.isoformat() for d in dates]
        save_earnings_cache(cache)

    return set(dates)


def get_earnings_blackout_dates(
    symbol: str,
    *,
    use_cache: bool = True,
    refresh: bool = False,
) -> set[date]:
    """Get the full blackout window: earnings day + day after.

    Args:
        symbol:    Ticker symbol.
        use_cache: Whether to use cache.
        refresh:   Force re-fetch.

    Returns:
        Set of dates where the ticker should be blocked from ORB trading.
    """
    earnings = get_earnings_dates(symbol, use_cache=use_cache, refresh=refresh)
    blackout: set[date] = set()
    for d in earnings:
        blackout.add(d)
        blackout.add(d + timedelta(days=1))
        # If earnings is on Friday, day-after is Monday
        if d.weekday() == 4:  # Friday
            blackout.add(d + timedelta(days=3))  # Monday
    return blackout


def is_earnings_blackout(symbol: str, d: date) -> bool:
    """Check if a specific date is in the earnings blackout window for a ticker.

    Args:
        symbol: Ticker symbol.
        d:      Calendar date to check.

    Returns:
        ``True`` if the ticker should be blocked on this date.
    """
    return d in get_earnings_blackout_dates(symbol)


def compute_earnings_blocked_array(
    index: pd.DatetimeIndex,
    symbol: str,
) -> np.ndarray[Any, np.dtype[np.bool_]]:
    """Pre-compute a boolean array marking bars in earnings blackout windows.

    Used by the backtest engine to block ORB trades for a specific ticker
    on earnings day and the day after.

    Args:
        index:  UTC DatetimeIndex aligned with 1-min bar data.
        symbol: Ticker symbol to check earnings for.

    Returns:
        Boolean numpy array of length ``len(index)``.  ``True`` means the
        bar falls in an earnings blackout window.
    """
    from zoneinfo import ZoneInfo

    et_tz = ZoneInfo("America/New_York")
    n = len(index)
    blocked = np.zeros(n, dtype=bool)

    blackout_dates = get_earnings_blackout_dates(symbol)
    if not blackout_dates:
        return blocked

    idx = index.tz_localize("UTC") if index.tzinfo is None else index
    et_index = idx.tz_convert(et_tz)

    for i, ts in enumerate(et_index):
        if ts.date() in blackout_dates:
            blocked[i] = True

    return blocked


def prefetch_earnings_cache(symbols: list[str]) -> None:
    """Pre-fetch and cache earnings dates for a list of symbols.

    Skips symbols that are already cached.  Call this before running
    a multi-ticker backtest to avoid repeated yfinance requests.

    Args:
        symbols: List of ticker symbols to fetch.
    """
    cache = load_earnings_cache()
    updated = False

    for sym in symbols:
        if sym in cache:
            logger.debug("earnings_cache_hit", symbol=sym)
            continue
        logger.info("fetching_earnings_dates", symbol=sym)
        dates = _fetch_earnings_dates(sym)
        if dates:
            cache[sym] = [d.isoformat() for d in dates]
            updated = True

    if updated:
        save_earnings_cache(cache)
