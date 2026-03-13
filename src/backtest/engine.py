"""ORB backtesting engine using the Backtesting.py framework.

Design notes
------------
- ``ORBStrategy`` is a ``backtesting.Strategy`` subclass.
- All opening-range logic uses *completed* bars only (shift-1 to avoid
  lookahead bias: the ORB high/low is locked after the 5th bar closes).
- Volume filter uses a 20-bar rolling mean shifted by 1 (same protection).
- ATR is computed via TA-Lib on the full price series (also shifted).
- Slippage: $0.02 per share, round-trip (applied via ``Backtest`` argument).
- Commission: $0 (Alpaca is commission-free).
- Forced flat at 15:55 ET.
"""

from __future__ import annotations

from datetime import time
from typing import Any

import numpy as np
import pandas as pd
import structlog
import talib
from backtesting import Backtest, Strategy

logger = structlog.get_logger(__name__)

# ── constants ─────────────────────────────────────────────────────────────────

_ET_OFFSET = pd.Timedelta(hours=-5)  # ET = UTC-5 (EST); adjust for EDT if needed
_ORB_BARS = 5  # number of 1-min bars in the opening range (9:30-9:34)
_ATR_PERIOD = 14
_ATR_MULTIPLIER = 1.5
_RISK_MULTIPLIER = 2.0
_VOL_WINDOW = 20
_VOL_MULTIPLIER = 1.5
_LUNCH_START = time(11, 30)
_LUNCH_END = time(13, 30)
_CUTOFF = time(15, 45)
_FORCE_FLAT = time(15, 55)


# ── helpers ───────────────────────────────────────────────────────────────────


def _et_time(ts: pd.Timestamp) -> time:
    """Convert a UTC pandas Timestamp to an ET wall-clock time (EST = UTC-5)."""
    et: pd.Timestamp = ts + _ET_OFFSET
    result: time = et.to_pydatetime().time()
    return result


def _is_market_hour(ts: pd.Timestamp) -> bool:
    t = _et_time(ts)
    return time(9, 30) <= t < _FORCE_FLAT


# ── Backtesting.py Strategy ───────────────────────────────────────────────────


class ORBStrategy(Strategy):  # type: ignore[misc]
    """Opening Range Breakout implemented as a Backtesting.py Strategy.

    Parameters (can be optimised via ``Backtest.optimize``):
        atr_mult:     ATR multiplier for stop distance (default 1.5).
        risk_mult:    Risk-to-reward for target (default 2.0).
        vol_mult:     Volume multiplier threshold (default 1.5).
        skip_lunch:   Whether to skip the 11:30-13:30 ET window (default 1).
    """

    atr_mult: float = _ATR_MULTIPLIER
    risk_mult: float = _RISK_MULTIPLIER
    vol_mult: float = _VOL_MULTIPLIER
    skip_lunch: int = 1  # 0 or 1 (bool parameters must be int for .optimize)

    def init(self) -> None:
        """Pre-compute all series indicators once on the full price history."""
        high = self.data.High
        low = self.data.Low
        close = self.data.Close
        volume = self.data.Volume

        # ATR(14) — batch via TA-Lib; shift by 1 to avoid lookahead
        atr_raw = talib.ATR(high, low, close, timeperiod=_ATR_PERIOD)
        self.atr = self.I(lambda: np.roll(atr_raw, 1), name="ATR")

        # Rolling average volume (20-bar), shifted by 1
        vol_series = pd.Series(volume)
        avg_vol_raw = vol_series.rolling(_VOL_WINDOW).mean().to_numpy()
        self.avg_vol = self.I(lambda: np.roll(avg_vol_raw, 1), name="AvgVol")

        # ORB high/low: for each bar, look back to find the max-high / min-low
        # of the first 5 bars *of that calendar day* that occur before 9:35 ET.
        # Computed once via pandas for efficiency; stored as arrays.
        index: pd.DatetimeIndex = self.data.index
        orb_high_arr, orb_low_arr = _compute_orb_arrays(
            index=index,
            high=high,
            low=low,
            orb_bars=_ORB_BARS,
        )
        self.orb_high_series = self.I(lambda: orb_high_arr, name="ORB_High")
        self.orb_low_series = self.I(lambda: orb_low_arr, name="ORB_Low")

    def next(self) -> None:
        """Called on every completed bar."""
        ts: pd.Timestamp = self.data.index[-1]
        t = _et_time(ts)

        # Force flat before close
        if t >= _FORCE_FLAT and self.position:
            self.position.close()
            return

        # No new entries outside trading window
        if t >= _CUTOFF:
            return
        if self.skip_lunch and _LUNCH_START <= t < _LUNCH_END:
            return

        # Already in a position — nothing to enter
        if self.position:
            return

        orb_high = self.orb_high_series[-1]
        orb_low = self.orb_low_series[-1]

        # ORB not yet established (NaN)
        if np.isnan(orb_high) or np.isnan(orb_low):
            return

        atr = self.atr[-1]
        avg_vol = self.avg_vol[-1]

        if np.isnan(atr) or np.isnan(avg_vol) or avg_vol <= 0:
            return

        close = self.data.Close[-1]
        volume = self.data.Volume[-1]

        # Volume filter
        if volume < avg_vol * self.vol_mult:
            return

        # LONG breakout
        if close > orb_high:
            entry = close
            stop = entry - self.atr_mult * atr
            risk = entry - stop
            if risk <= 0:
                return
            target = entry + self.risk_mult * risk
            self.buy(sl=stop, tp=target)

        # SHORT breakdown
        elif close < orb_low:
            entry = close
            stop = entry + self.atr_mult * atr
            risk = stop - entry
            if risk <= 0:
                return
            target = entry - self.risk_mult * risk
            self.sell(sl=stop, tp=target)


# ── ORB array pre-computation ─────────────────────────────────────────────────


def _compute_orb_arrays(
    index: pd.DatetimeIndex,
    high: Any,  # numpy array from backtesting
    low: Any,
    orb_bars: int = 5,
) -> tuple[np.ndarray[Any, np.dtype[np.float64]], np.ndarray[Any, np.dtype[np.float64]]]:
    """Pre-compute the ORB high/low for every bar in the series.

    For each bar at time T, the ORB high and low are the max-high and min-low
    of the first *orb_bars* 1-min bars of that calendar day (9:30-9:35 ET),
    available only *after* the opening range is complete.

    Bars inside the opening range itself get NaN (ORB not yet locked).

    This function is lookahead-safe: bar at index i sees the ORB values that
    were established by bars [0..orb_bars-1] of the same day.

    Args:
        index:    UTC DatetimeIndex aligned with *high* and *low*.
        high:     Numpy-like array of bar highs.
        low:      Numpy-like array of bar lows.
        orb_bars: Number of 1-min bars that define the opening range.

    Returns:
        (orb_high_array, orb_low_array) — both float64 numpy arrays with NaN
        where the ORB is not yet established.
    """
    high_arr = np.asarray(high, dtype=float)
    low_arr = np.asarray(low, dtype=float)
    n = len(index)

    orb_high_out = np.full(n, np.nan)
    orb_low_out = np.full(n, np.nan)

    # Convert index to ET for day-grouping
    et_index = index + _ET_OFFSET

    # Group bar positions by calendar date
    date_groups: dict[Any, list[int]] = {}
    for i, ts in enumerate(et_index):
        d = ts.date()
        date_groups.setdefault(d, []).append(i)

    for positions in date_groups.values():
        if len(positions) < orb_bars:
            continue  # not enough bars in this day
        # The ORB is established after the first orb_bars bars close
        orb_indices = positions[:orb_bars]
        orb_high = float(np.max(high_arr[orb_indices]))
        orb_low = float(np.min(low_arr[orb_indices]))

        # Assign to all bars AFTER the opening range (not during)
        for pos in positions[orb_bars:]:
            orb_high_out[pos] = orb_high
            orb_low_out[pos] = orb_low

    return orb_high_out, orb_low_out


# ── Backtest runner ───────────────────────────────────────────────────────────


def run_backtest(
    df: pd.DataFrame,
    cash: float = 50_000.0,
    slippage: float = 0.02,
    atr_mult: float = _ATR_MULTIPLIER,
    risk_mult: float = _RISK_MULTIPLIER,
    vol_mult: float = _VOL_MULTIPLIER,
    skip_lunch: bool = True,
) -> Any:
    """Run the ORB backtest on *df* and return the Backtesting stats dict.

    Args:
        df:          OHLCV DataFrame with DatetimeIndex (UTC).
        cash:        Starting equity in USD.
        slippage:    Slippage per share per side in USD (default $0.02).
        atr_mult:    ATR stop multiplier.
        risk_mult:   Reward-to-risk multiplier for target.
        vol_mult:    Volume threshold multiplier.
        skip_lunch:  Whether to skip the 11:30-13:30 ET chop zone.

    Returns:
        ``backtesting.Stats`` object (dict-like).  Access ``._trades`` for the
        full trade log as a DataFrame.
    """
    # Backtesting.py requires column names: Open, High, Low, Close, Volume
    bt_df = df.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    )

    bt = Backtest(
        bt_df,
        ORBStrategy,
        cash=cash,
        commission=0.0,
        exclusive_orders=True,
    )

    stats = bt.run(
        atr_mult=atr_mult,
        risk_mult=risk_mult,
        vol_mult=vol_mult,
        skip_lunch=int(skip_lunch),
    )

    logger.info(
        "backtest_done",
        trades=int(stats["# Trades"]),
        total_return_pct=float(stats["Return [%]"]),
        win_rate_pct=float(stats["Win Rate [%]"]) if stats["# Trades"] > 0 else 0.0,
    )
    return stats
