"""Historical bar loader for backtesting.

Responsibilities
----------------
- Load 1-min bars from SQLite into a pandas DataFrame with a UTC DatetimeIndex.
- Resample to 5-min and 15-min bars using proper OHLCV aggregation
  (open=first, high=max, low=min, close=last, volume=sum).
- Split the full date range into walk-forward windows (in-sample / out-of-sample).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING

import pandas as pd
import structlog

from src.models import TimeFrame

if TYPE_CHECKING:
    from src.storage.database import BarDatabase

logger = structlog.get_logger(__name__)

# OHLCV aggregation rules for pandas .resample()
_OHLCV_AGG: dict[str, str] = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
    "vwap": "mean",
}

# Resample offset aliases
_TF_OFFSET: dict[TimeFrame, str] = {
    TimeFrame.ONE_MIN: "1min",
    TimeFrame.FIVE_MIN: "5min",
    TimeFrame.FIFTEEN_MIN: "15min",
    TimeFrame.THIRTY_MIN: "30min",
}


@dataclass(frozen=True)
class WalkForwardWindow:
    """A single in-sample / out-of-sample split."""

    in_sample_start: date
    in_sample_end: date
    out_of_sample_start: date
    out_of_sample_end: date

    def __str__(self) -> str:
        return (
            f"IS={self.in_sample_start}..{self.in_sample_end}  "
            f"OOS={self.out_of_sample_start}..{self.out_of_sample_end}"
        )


def load_bars(
    db: BarDatabase,
    symbol: str = "SPY",
    timeframe: TimeFrame = TimeFrame.ONE_MIN,
    start: datetime | None = None,
    end: datetime | None = None,
) -> pd.DataFrame:
    """Load bars from SQLite and return a DataFrame with a UTC DatetimeIndex.

    Columns: open, high, low, close, volume, vwap (all float64).
    Rows are ordered by timestamp ascending; incomplete final bars are dropped.

    Args:
        db:        Open :class:`~src.storage.database.BarDatabase` instance.
        symbol:    Ticker to load (default ``"SPY"``).
        timeframe: Bar timeframe to load (default ``1Min``).
        start:     Inclusive start (UTC).  ``None`` = all data.
        end:       Inclusive end (UTC).   ``None`` = all data.

    Returns:
        DataFrame indexed by UTC timestamp.  Empty if no data found.
    """
    bars = db.query_bars(symbol=symbol, timeframe=timeframe, start=start, end=end)
    if not bars:
        logger.warning("load_bars_empty", symbol=symbol, timeframe=timeframe.value)
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "vwap"])

    records = [
        {
            "timestamp": b.timestamp,
            "open": float(b.open),
            "high": float(b.high),
            "low": float(b.low),
            "close": float(b.close),
            "volume": int(b.volume),
            "vwap": float(b.vwap),
        }
        for b in bars
    ]
    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()

    logger.info(
        "load_bars_done",
        symbol=symbol,
        timeframe=timeframe.value,
        rows=len(df),
        start=str(df.index[0]),
        end=str(df.index[-1]),
    )
    return df


def resample(df: pd.DataFrame, target: TimeFrame) -> pd.DataFrame:
    """Resample a 1-min DataFrame to *target* timeframe.

    Args:
        df:     1-min OHLCV DataFrame (DatetimeIndex, UTC).
        target: Target timeframe (must be >= 1-min).

    Returns:
        Resampled DataFrame.  Incomplete final bar is dropped (``dropna``).
    """
    if target == TimeFrame.ONE_MIN:
        return df.copy()

    offset = _TF_OFFSET.get(target)
    if offset is None:
        raise ValueError(f"Unsupported resample target: {target}")

    resampled = (
        df.resample(offset, label="left", closed="left").agg(_OHLCV_AGG).dropna(subset=["close"])
    )
    logger.debug("resample_done", target=target.value, rows=len(resampled))
    return resampled


def make_walk_forward_windows(
    df: pd.DataFrame,
    in_sample_days: int = 60,
    out_of_sample_days: int = 20,
) -> list[WalkForwardWindow]:
    """Slice the data into sequential in-sample / out-of-sample windows.

    Windows are non-overlapping.  The last OOS window is truncated to whatever
    data remains (may be shorter than *out_of_sample_days*).

    Args:
        df:                 DataFrame with UTC DatetimeIndex.
        in_sample_days:     Calendar days per in-sample period (default 60).
        out_of_sample_days: Calendar days per out-of-sample period (default 20).

    Returns:
        List of :class:`WalkForwardWindow`.  Empty if there is not enough data
        for even one full window.
    """
    if df.empty:
        return []

    total_days = in_sample_days + out_of_sample_days
    first_date: date = df.index[0].date()
    last_date: date = df.index[-1].date()

    windows: list[WalkForwardWindow] = []
    window_start = first_date

    while True:
        is_start = window_start
        is_end = is_start + timedelta(days=in_sample_days - 1)
        oos_start = is_end + timedelta(days=1)
        oos_end = oos_start + timedelta(days=out_of_sample_days - 1)

        if is_end > last_date:
            break  # not enough data for even a full IS window

        # Clamp OOS end to actual data
        oos_end = min(oos_end, last_date)

        windows.append(
            WalkForwardWindow(
                in_sample_start=is_start,
                in_sample_end=is_end,
                out_of_sample_start=oos_start,
                out_of_sample_end=oos_end,
            )
        )
        window_start = is_start + timedelta(days=total_days)

    logger.info(
        "walk_forward_windows",
        count=len(windows),
        is_days=in_sample_days,
        oos_days=out_of_sample_days,
    )
    return windows


def slice_window(df: pd.DataFrame, window: WalkForwardWindow) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (in_sample_df, out_of_sample_df) for the given window.

    Args:
        df:     Full DataFrame with UTC DatetimeIndex.
        window: :class:`WalkForwardWindow` defining the date ranges.

    Returns:
        Tuple of (in_sample, out_of_sample) DataFrames.
    """
    tz = UTC

    is_start = datetime(
        window.in_sample_start.year,
        window.in_sample_start.month,
        window.in_sample_start.day,
        tzinfo=tz,
    )
    is_end = datetime(
        window.in_sample_end.year,
        window.in_sample_end.month,
        window.in_sample_end.day,
        23,
        59,
        59,
        tzinfo=tz,
    )
    oos_start = datetime(
        window.out_of_sample_start.year,
        window.out_of_sample_start.month,
        window.out_of_sample_start.day,
        tzinfo=tz,
    )
    oos_end = datetime(
        window.out_of_sample_end.year,
        window.out_of_sample_end.month,
        window.out_of_sample_end.day,
        23,
        59,
        59,
        tzinfo=tz,
    )

    in_sample = df.loc[is_start:is_end].copy()  # type: ignore[misc]
    out_of_sample = df.loc[oos_start:oos_end].copy()  # type: ignore[misc]
    return in_sample, out_of_sample
