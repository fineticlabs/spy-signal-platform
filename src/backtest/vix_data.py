"""VIX daily data loader — downloads ^VIX from yfinance and caches in SQLite.

Provides a per-bar VIX array for the backtest engine: each 1-min bar gets the
*previous trading day's* VIX close (no lookahead).
"""

from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import structlog
import yfinance as yf

logger = structlog.get_logger(__name__)

_VIX_TABLE = """
CREATE TABLE IF NOT EXISTS vix_daily (
    date   TEXT PRIMARY KEY,  -- YYYY-MM-DD
    close  REAL NOT NULL
);
"""

_DB_PATH = "data/spy_signals.db"


def _get_conn(db_path: str = _DB_PATH) -> sqlite3.Connection:
    """Open WAL-mode SQLite connection and ensure VIX table exists."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute(_VIX_TABLE)
    conn.commit()
    return conn


def download_vix(
    start: str = "2019-01-01",
    end: str | None = None,
    db_path: str = _DB_PATH,
) -> int:
    """Download daily ^VIX close from yfinance and upsert into SQLite.

    Args:
        start: Start date string (YYYY-MM-DD).
        end:   End date string (defaults to today).
        db_path: SQLite database path.

    Returns:
        Number of rows upserted.
    """
    logger.info("vix_download_start", start=start, end=end)
    ticker = yf.Ticker("^VIX")
    hist = ticker.history(start=start, end=end, auto_adjust=True)

    if hist.empty:
        logger.warning("vix_download_empty")
        return 0

    conn = _get_conn(db_path)
    rows = 0
    for ts, row in hist.iterrows():
        d = ts.strftime("%Y-%m-%d")  # type: ignore[union-attr]
        close = float(row["Close"])
        conn.execute(
            "INSERT OR REPLACE INTO vix_daily (date, close) VALUES (?, ?)",
            (d, close),
        )
        rows += 1

    conn.commit()
    conn.close()
    logger.info("vix_download_done", rows=rows)
    return rows


def load_vix_series(db_path: str = _DB_PATH) -> pd.Series:
    """Load the full VIX daily close series from SQLite.

    Returns:
        pd.Series indexed by ``datetime.date`` with float VIX close values.
    """
    conn = _get_conn(db_path)
    df = pd.read_sql("SELECT date, close FROM vix_daily ORDER BY date", conn)
    conn.close()

    if df.empty:
        return pd.Series(dtype=float)

    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df.set_index("date")["close"]


def compute_vix_for_bars(
    index: pd.DatetimeIndex,
    vix_series: pd.Series,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Map previous-day VIX close to each 1-min bar (no lookahead).

    For each trading day D in the bar index, the VIX value assigned is the
    close from day D-1 (the most recent prior trading day with VIX data).

    Args:
        index:      UTC DatetimeIndex of 1-min bars.
        vix_series: pd.Series indexed by ``datetime.date`` with VIX closes.

    Returns:
        Float64 numpy array of length ``len(index)``.  NaN where no prior
        VIX data is available.
    """
    from zoneinfo import ZoneInfo

    et_tz = ZoneInfo("America/New_York")
    n = len(index)
    vix_out = np.full(n, np.nan)

    if vix_series.empty:
        return vix_out

    # Convert VIX series to a sorted list of (date, close) for binary search
    vix_dates = sorted(vix_series.index)
    vix_dict: dict[date, float] = dict(vix_series.items())

    # Group bars by ET trading day
    et_index = (
        index.tz_convert(et_tz)
        if index.tzinfo is not None
        else index.tz_localize("UTC").tz_convert(et_tz)
    )

    date_to_positions: dict[date, list[int]] = {}
    dates_ordered: list[date] = []
    for i, ts in enumerate(et_index):
        d = ts.date()
        if d not in date_to_positions:
            dates_ordered.append(d)
            date_to_positions[d] = []
        date_to_positions[d].append(i)

    # For each trading day, find the previous day's VIX close
    for d in dates_ordered:
        # Find the most recent VIX date strictly before d
        prev_vix = _find_prev_vix(d, vix_dates, vix_dict)
        if prev_vix is not None:
            for pos in date_to_positions[d]:
                vix_out[pos] = prev_vix

    return vix_out


def _find_prev_vix(
    target: date,
    sorted_dates: list[date],
    vix_dict: dict[date, float],
) -> float | None:
    """Binary search for the most recent VIX close strictly before target date."""
    import bisect

    idx = bisect.bisect_left(sorted_dates, target) - 1
    if idx < 0:
        return None
    return vix_dict[sorted_dates[idx]]
