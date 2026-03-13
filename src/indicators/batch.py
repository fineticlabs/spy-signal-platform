"""Batch indicator computation using TA-Lib.

All functions operate on a completed pandas DataFrame of bars (columns: open,
high, low, close, volume) and return a pandas Series aligned to the same index.
Leading NaN values are expected until enough bars have been consumed by each
indicator's lookback window.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import structlog
import talib

logger = structlog.get_logger(__name__)

# Column name constants expected in the input DataFrame
_OPEN = "open"
_HIGH = "high"
_LOW = "low"
_CLOSE = "close"
_VOLUME = "volume"


def _to_float64(series: pd.Series) -> np.ndarray[int, np.dtype[np.float64]]:
    """Convert a Series of Decimal/object values to a float64 NumPy array."""
    return series.astype(float).to_numpy(dtype=np.float64)  # type: ignore[no-any-return]


def calculate_ema(df: pd.DataFrame, period: int) -> pd.Series:
    """Exponential Moving Average of close prices.

    Weights more recent prices exponentially more than older ones.
    Reacts faster to price changes than a simple moving average.
    Leading ``NaN`` values fill the first ``period - 1`` bars.

    Args:
        df:     DataFrame with at least a ``close`` column.
        period: Smoothing window length (e.g. 9, 20, 50).

    Returns:
        Series of EMA values aligned to ``df.index``.
    """
    close = _to_float64(df[_CLOSE])
    result = talib.EMA(close, timeperiod=period)
    return pd.Series(result, index=df.index, name=f"ema_{period}")


def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Relative Strength Index measuring momentum.

    Oscillates between 0 and 100.  Values above 70 are commonly considered
    overbought; values below 30 are oversold.  Leading ``NaN`` fills the first
    ``period`` bars.

    Args:
        df:     DataFrame with at least a ``close`` column.
        period: Look-back window (default 14).

    Returns:
        Series of RSI values (0-100) aligned to ``df.index``.
    """
    close = _to_float64(df[_CLOSE])
    result = talib.RSI(close, timeperiod=period)
    return pd.Series(result, index=df.index, name=f"rsi_{period}")


def calculate_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Moving Average Convergence/Divergence.

    Returns three series: the MACD line (fast EMA - slow EMA), the signal line
    (EMA of MACD), and the histogram (MACD - signal).  A positive histogram
    suggests bullish momentum; negative suggests bearish.

    Args:
        df:     DataFrame with at least a ``close`` column.
        fast:   Fast EMA period (default 12).
        slow:   Slow EMA period (default 26).
        signal: Signal EMA period (default 9).

    Returns:
        Tuple of (macd, signal, histogram) Series aligned to ``df.index``.
    """
    close = _to_float64(df[_CLOSE])
    macd_line, signal_line, histogram = talib.MACD(
        close, fastperiod=fast, slowperiod=slow, signalperiod=signal
    )
    idx = df.index
    return (
        pd.Series(macd_line, index=idx, name="macd"),
        pd.Series(signal_line, index=idx, name="macd_signal"),
        pd.Series(histogram, index=idx, name="macd_histogram"),
    )


def calculate_bollinger(
    df: pd.DataFrame,
    period: int = 20,
    std: int = 2,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands around a simple moving average.

    Upper and lower bands are placed ``std`` standard deviations from the
    middle SMA.  Price touching the upper band may indicate overbought
    conditions; touching the lower band may indicate oversold.

    Args:
        df:     DataFrame with at least a ``close`` column.
        period: SMA window (default 20).
        std:    Number of standard deviations for the bands (default 2).

    Returns:
        Tuple of (upper, middle, lower) Series aligned to ``df.index``.
    """
    close = _to_float64(df[_CLOSE])
    upper, middle, lower = talib.BBANDS(
        close,
        timeperiod=period,
        nbdevup=std,
        nbdevdn=std,
        matype=0,  # type: ignore[arg-type]  # 0 = SMA, standard for BB
    )
    idx = df.index
    return (
        pd.Series(upper, index=idx, name="bb_upper"),
        pd.Series(middle, index=idx, name="bb_middle"),
        pd.Series(lower, index=idx, name="bb_lower"),
    )


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range -- a measure of volatility.

    True range is ``max(high - low, |high - prev_close|, |low - prev_close|)``.
    ATR is the rolling average of true ranges.  Higher values indicate higher
    volatility; used for position sizing and stop placement.

    Args:
        df:     DataFrame with ``high``, ``low``, and ``close`` columns.
        period: Smoothing window (default 14).

    Returns:
        Series of ATR values aligned to ``df.index``.
    """
    high = _to_float64(df[_HIGH])
    low = _to_float64(df[_LOW])
    close = _to_float64(df[_CLOSE])
    result = talib.ATR(high, low, close, timeperiod=period)
    return pd.Series(result, index=df.index, name=f"atr_{period}")


def calculate_vwap(df: pd.DataFrame) -> pd.Series:
    """Cumulative intraday VWAP from the first bar in *df*.

    VWAP = cumulative_sum(typical_price * volume) / cumulative_sum(volume)
    where typical_price = (high + low + close) / 3.

    The calculation resets at the start of each trading session.  Passing only
    the current day's bars ensures correct session-scoped VWAP.

    Args:
        df: DataFrame with ``high``, ``low``, ``close``, and ``volume`` columns.

    Returns:
        Series of VWAP values aligned to ``df.index``.
    """
    typical_price = (
        df[_HIGH].astype(float) + df[_LOW].astype(float) + df[_CLOSE].astype(float)
    ) / 3
    volume = df[_VOLUME].astype(float)
    cum_tp_vol = (typical_price * volume).cumsum()
    cum_vol = volume.cumsum()
    result = cum_tp_vol / cum_vol
    return pd.Series(result.to_numpy(), index=df.index, name="vwap")
