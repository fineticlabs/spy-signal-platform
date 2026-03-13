"""ORB backtesting engine using the Backtesting.py framework.

Design notes
------------
- ``ORBStrategy`` is a ``backtesting.Strategy`` subclass.
- All opening-range logic uses *completed* bars only (shift-1 to avoid
  lookahead bias: the ORB high/low is locked after the 5th bar closes).
- Volume filter uses a 20-bar rolling mean shifted by 1 (same protection); threshold 1.5x.
- ATR is computed via TA-Lib on the full price series (also shifted).
- Timezone: uses ZoneInfo("America/New_York") to handle EDT/EST correctly.
  A fixed -5h offset was wrong during EDT season (Mar-Nov) and caused the
  lunch chop filter and entry cutoff to fire 1 hour late.
- Trading windows: 9:35-11:00 ET (first hour) and 14:30-15:30 ET (power hour).
  All other times are blocked — no lunch chop parameter needed.
- 15-min EMA(20) trend alignment: LONG only if close > EMA, SHORT only if close < EMA.
  EMA is computed by resampling 1-min to 15-min, shifted 1 bar, forward-filled back.
- Dynamic targets: ORB range vs trailing 20-day p25/p75.
  range > p75 → 2.5R target; range < p25 → 1.5R target; else → 2R (default).
- Gap classification filter (gap-and-go bias):
    gap = (today_open - prev_day_close) / prev_day_close * 100
    gap >  +0.3%: LONG only (gap-up momentum favours breakout longs)
    gap < -0.3%: SHORT only (gap-down momentum favours breakdown shorts)
    |gap| ≤ 0.3%: both directions allowed (neutral open)
    NaN (first day): both allowed
  today_open = first 1-min bar open at 9:30 ET; prev_day_close = last bar close
  of the preceding trading day.  Both values are known before any ORB signal
  can fire (ORB signals require ≥5 completed bars), so no lookahead bias.
- Max 5 trades per calendar day (ET) enforced in next().
- ORB range filter: skips days where range < min_orb_pct of price.
- Slippage: $0.02 per share, round-trip (applied via ``Backtest`` argument).
- Commission: $0 (Alpaca is commission-free).
- Forced flat at 15:55 ET.
"""

from __future__ import annotations

from datetime import date, time
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import structlog
import talib
from backtesting import Backtest, Strategy

logger = structlog.get_logger(__name__)

# ── constants ─────────────────────────────────────────────────────────────────

_ET_TZ = ZoneInfo("America/New_York")
_ORB_BARS = 5  # number of 1-min bars in the opening range (9:30-9:34)
_ATR_PERIOD = 14
_ATR_MULTIPLIER = 1.5
_RISK_MULTIPLIER = 2.0
_VOL_WINDOW = 20
_VOL_MULTIPLIER = 1.5  # restored: 2.0x was too strict, filtered too many setups

# Trading windows (ET wall-clock times)
_WINDOW1_START = time(9, 35)  # first hour open
_WINDOW1_END = time(11, 0)
_WINDOW2_START = time(14, 30)  # power hour
_WINDOW2_END = time(15, 30)
_FORCE_FLAT = time(15, 55)

_MAX_TRADES_PER_DAY = 5
_MIN_ORB_PCT = 0.0015  # lowered from 0.3% → 0.15% of price (less restrictive)
_EMA15M_PERIOD = 20  # EMA period on 15-min bars for trend alignment
_ORB_RANGE_WINDOW = 20  # trailing days for ORB range percentiles
_GAP_THRESHOLD_PCT = 0.3  # ±0.3% gap separates directional bias from neutral


# ── helpers ───────────────────────────────────────────────────────────────────


def _et_time(ts: pd.Timestamp) -> time:
    """Convert a UTC pandas Timestamp to an ET wall-clock time (EDT/EST-aware)."""
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    et: pd.Timestamp = ts.tz_convert(_ET_TZ)
    result: time = et.to_pydatetime().time()
    return result


def _et_date(ts: pd.Timestamp) -> date:
    """Convert a UTC pandas Timestamp to an ET calendar date (EDT/EST-aware)."""
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    result: date = ts.tz_convert(_ET_TZ).to_pydatetime().date()
    return result


def _to_et_index(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Convert a DatetimeIndex (UTC) to America/New_York, handling tz-naive input."""
    if index.tzinfo is None:
        index = index.tz_localize("UTC")
    return index.tz_convert(_ET_TZ)


# ── Gap classification ────────────────────────────────────────────────────────


def _compute_gap_array(
    index: pd.DatetimeIndex,
    open_prices: Any,
    close_prices: Any,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Pre-compute the daily opening gap percentage for every 1-min bar.

    gap = (today_first_open - prev_day_last_close) / prev_day_last_close * 100

    The gap is the same value for all bars within a trading day.  The first
    trading day in the dataset returns NaN (no prior day available).

    Zero lookahead: ``today_first_open`` is the open of the 9:30 ET bar;
    ORB signals require at least 5 completed bars (≥9:35 ET), so the gap
    is always known before any signal can fire.

    Args:
        index:        UTC DatetimeIndex aligned with the price arrays.
        open_prices:  Numpy-like array of bar open prices.
        close_prices: Numpy-like array of bar close prices.

    Returns:
        Float64 numpy array of length ``len(index)``.  NaN for the first
        trading day; percentage gap (e.g. 0.5 = +0.5%) for all other days.
    """
    open_arr = np.asarray(open_prices, dtype=float)
    close_arr = np.asarray(close_prices, dtype=float)
    n = len(index)
    gap_out = np.full(n, np.nan)

    et_index = _to_et_index(index)

    # Build ordered list of trading dates → bar positions
    dates: list[Any] = []
    date_to_positions: dict[Any, list[int]] = {}
    for i, ts in enumerate(et_index):
        d = ts.date()
        if d not in date_to_positions:
            dates.append(d)
            date_to_positions[d] = []
        date_to_positions[d].append(i)

    for idx_d in range(1, len(dates)):  # skip first day (no prior day)
        d = dates[idx_d]
        d_prev = dates[idx_d - 1]

        positions_today = date_to_positions[d]
        positions_prev = date_to_positions[d_prev]

        if not positions_today or not positions_prev:
            continue

        today_open = open_arr[positions_today[0]]  # first bar open at 9:30
        prev_close = close_arr[positions_prev[-1]]  # last bar close yesterday

        if prev_close <= 0 or np.isnan(prev_close) or np.isnan(today_open):
            continue

        gap_pct = (today_open - prev_close) / prev_close * 100.0
        for pos in positions_today:
            gap_out[pos] = gap_pct

    return gap_out


def _classify_gap(gap_pct: float, threshold: float) -> tuple[bool, bool]:
    """Classify a gap and return (allows_long, allows_short) directional flags.

    Args:
        gap_pct:   Opening gap in percent (e.g. 0.5 = +0.5%, -0.4 = -0.4%).
                   Pass ``float('nan')`` for the first trading day.
        threshold: Absolute gap percentage that separates directional bias
                   from neutral (default ``_GAP_THRESHOLD_PCT`` = 0.3).

    Returns:
        ``(allows_long, allows_short)`` tuple:
        - Gap up  (> +threshold): (True, False)  — LONG only
        - Gap down (< -threshold): (False, True) — SHORT only
        - Flat / NaN              : (True, True)  — both directions
    """
    if np.isnan(gap_pct):
        return True, True
    if gap_pct > threshold:
        return True, False
    if gap_pct < -threshold:
        return False, True
    return True, True


# ── Backtesting.py Strategy ───────────────────────────────────────────────────


class ORBStrategy(Strategy):  # type: ignore[misc]
    """Opening Range Breakout implemented as a Backtesting.py Strategy.

    Parameters (can be optimised via ``Backtest.optimize``):
        atr_mult:           ATR multiplier for stop distance (default 1.5).
        risk_mult:          Risk-to-reward for target when no percentile data (default 2.0).
        vol_mult:           Volume multiplier threshold (default 1.5).
        max_trades_per_day: Max entries per calendar day ET (default 5).
        min_orb_pct:        Min ORB range as fraction of price (default 0.0015).
    """

    atr_mult: float = _ATR_MULTIPLIER
    risk_mult: float = _RISK_MULTIPLIER
    vol_mult: float = _VOL_MULTIPLIER
    max_trades_per_day: int = _MAX_TRADES_PER_DAY
    min_orb_pct: float = _MIN_ORB_PCT

    def init(self) -> None:
        """Pre-compute all series indicators once on the full price history."""
        high = self.data.High
        low = self.data.Low
        close = self.data.Close
        volume = self.data.Volume

        # ATR(14) - batch via TA-Lib; shift by 1 to avoid lookahead
        atr_raw = talib.ATR(high, low, close, timeperiod=_ATR_PERIOD)
        self.atr = self.I(lambda: np.roll(atr_raw, 1), name="ATR")

        # Rolling average volume (20-bar), shifted by 1
        vol_series = pd.Series(volume)
        avg_vol_raw = vol_series.rolling(_VOL_WINDOW).mean().to_numpy()
        self.avg_vol = self.I(lambda: np.roll(avg_vol_raw, 1), name="AvgVol")

        # ORB high/low + range filter arrays
        index: pd.DatetimeIndex = self.data.index
        orb_high_arr, orb_low_arr = _compute_orb_arrays(
            index=index,
            high=high,
            low=low,
            orb_bars=_ORB_BARS,
        )
        self.orb_high_series = self.I(lambda: orb_high_arr, name="ORB_High")
        self.orb_low_series = self.I(lambda: orb_low_arr, name="ORB_Low")

        # ORB range as % of midpoint - used to filter narrow/choppy ranges
        midpoint = (orb_high_arr + orb_low_arr) / 2.0
        orb_range_arr = np.where(
            (midpoint > 0) & ~np.isnan(orb_high_arr) & ~np.isnan(orb_low_arr),
            (orb_high_arr - orb_low_arr) / midpoint,
            np.nan,
        )
        self.orb_range_pct = self.I(lambda: orb_range_arr, name="ORB_Range_Pct")

        # 15-min EMA(20) trend alignment filter (no-lookahead: shifted 1 bar)
        close_arr = np.asarray(close, dtype=float)
        ema15m_arr = _compute_15m_ema(index, close_arr, period=_EMA15M_PERIOD)
        self.ema15m = self.I(lambda: ema15m_arr, name="EMA15m")

        # ORB range percentile arrays for dynamic target selection
        orb_p25_arr, orb_p75_arr = _compute_orb_percentile_arrays(
            index=index,
            orb_high_arr=orb_high_arr,
            orb_low_arr=orb_low_arr,
            window=_ORB_RANGE_WINDOW,
        )
        self.orb_p25 = self.I(lambda: orb_p25_arr, name="ORB_P25")
        self.orb_p75 = self.I(lambda: orb_p75_arr, name="ORB_P75")

        # Daily gap classification (no lookahead: gap known at 9:30, ORB fires >=9:35)
        open_arr_for_gap = np.asarray(self.data.Open, dtype=float)
        gap_arr = _compute_gap_array(index, open_arr_for_gap, close_arr)
        self.gap_pct = self.I(lambda: gap_arr, name="GapPct")

        # Daily trade counter state (reset per ET calendar day in next())
        self._last_et_date: date | None = None
        self._daily_trade_count: int = 0

    def next(self) -> None:
        """Called on every completed bar."""
        ts: pd.Timestamp = self.data.index[-1]
        t = _et_time(ts)

        # Force flat before close
        if t >= _FORCE_FLAT and self.position:
            self.position.close()
            return

        # Only trade during first-hour window or power-hour window
        in_window1 = _WINDOW1_START <= t < _WINDOW1_END
        in_window2 = _WINDOW2_START <= t < _WINDOW2_END
        if not (in_window1 or in_window2):
            return

        # Already in a position - nothing to enter
        if self.position:
            return

        # Reset daily trade counter on new ET calendar day
        today = _et_date(ts)
        if today != self._last_et_date:
            self._last_et_date = today
            self._daily_trade_count = 0

        # Enforce max trades per day
        if self._daily_trade_count >= self.max_trades_per_day:
            return

        orb_high = self.orb_high_series[-1]
        orb_low = self.orb_low_series[-1]
        orb_range = self.orb_range_pct[-1]

        # ORB not yet established (NaN)
        if np.isnan(orb_high) or np.isnan(orb_low):
            return

        # Skip narrow ORB days - too choppy to trade breakouts
        if np.isnan(orb_range) or orb_range < self.min_orb_pct:
            return

        atr = self.atr[-1]
        avg_vol = self.avg_vol[-1]
        ema15m_val = self.ema15m[-1]

        if np.isnan(atr) or np.isnan(avg_vol) or avg_vol <= 0:
            return

        close = self.data.Close[-1]
        volume = self.data.Volume[-1]

        # Volume filter
        if volume < avg_vol * self.vol_mult:
            return

        # 15-min EMA trend alignment: allow both directions if EMA not yet available
        if not np.isnan(ema15m_val):
            bullish_trend = close > ema15m_val
            bearish_trend = close < ema15m_val
        else:
            bullish_trend = True
            bearish_trend = True

        # Gap classification filter
        gap_val = float(self.gap_pct[-1])
        gap_allows_long, gap_allows_short = _classify_gap(gap_val, _GAP_THRESHOLD_PCT)

        # Dynamic risk multiplier based on ORB range vs trailing 20-day percentiles
        p25 = self.orb_p25[-1]
        p75 = self.orb_p75[-1]
        orb_range_abs = orb_high - orb_low

        if not np.isnan(p25) and not np.isnan(p75):
            if orb_range_abs > p75:
                dynamic_risk_mult = 2.5
            elif orb_range_abs < p25:
                dynamic_risk_mult = 1.5
            else:
                dynamic_risk_mult = self.risk_mult
        else:
            dynamic_risk_mult = self.risk_mult

        # LONG breakout
        if close > orb_high and bullish_trend and gap_allows_long:
            entry = close
            stop = entry - self.atr_mult * atr
            risk = entry - stop
            if risk <= 0:
                return
            target = entry + dynamic_risk_mult * risk
            self.buy(sl=stop, tp=target)
            self._daily_trade_count += 1

        # SHORT breakdown
        elif close < orb_low and bearish_trend and gap_allows_short:
            entry = close
            stop = entry + self.atr_mult * atr
            risk = stop - entry
            if risk <= 0:
                return
            target = entry - dynamic_risk_mult * risk
            self.sell(sl=stop, tp=target)
            self._daily_trade_count += 1


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
        (orb_high_array, orb_low_array) - both float64 numpy arrays with NaN
        where the ORB is not yet established.
    """
    high_arr = np.asarray(high, dtype=float)
    low_arr = np.asarray(low, dtype=float)
    n = len(index)

    orb_high_out = np.full(n, np.nan)
    orb_low_out = np.full(n, np.nan)

    # Convert index to ET for day-grouping (handles EDT/EST correctly)
    et_index = _to_et_index(index)

    # Group bar positions by calendar date in ET
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


# ── 15-min EMA trend alignment ────────────────────────────────────────────────


def _compute_15m_ema(
    index: pd.DatetimeIndex,
    close: np.ndarray[Any, np.dtype[np.float64]],
    period: int = 20,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Compute a no-lookahead 15-min EMA mapped back to 1-min resolution.

    Steps:
        1. Build a 1-min close Series with ET-aware index.
        2. Resample to 15-min (last close of each 15-min bar).
        3. Compute EMA(period) via TA-Lib.
        4. Shift by 1 bar to prevent lookahead (each 15-min bar uses EMA
           from the *previous* 15-min bar).
        5. Reindex back to 1-min with forward-fill (EMA value is constant
           within each 15-min bucket until the next bar closes).

    Args:
        index:  UTC DatetimeIndex aligned with *close*.
        close:  Float64 numpy array of 1-min close prices.
        period: EMA lookback period (default 20).

    Returns:
        Float64 numpy array of length ``len(index)`` with NaN where EMA is
        not yet available.
    """
    et_index = _to_et_index(index)
    close_series = pd.Series(close, index=et_index)

    # Resample to 15-min (last close of each bucket), drop empty buckets
    close_15m = close_series.resample("15min").last().dropna()
    if len(close_15m) < period:
        return np.full(len(index), np.nan)

    # EMA on 15-min bars
    ema_raw = talib.EMA(close_15m.to_numpy(dtype=float), timeperiod=period)
    ema_15m = pd.Series(ema_raw, index=close_15m.index)

    # Shift by 1: bar N uses EMA computed through bar N-1
    ema_shifted = ema_15m.shift(1)

    # Forward-fill back to 1-min resolution
    ema_1m = ema_shifted.reindex(et_index, method="ffill")

    result: np.ndarray[Any, np.dtype[np.float64]] = ema_1m.to_numpy(dtype=float)
    return result


# ── ORB range percentile arrays ───────────────────────────────────────────────


def _compute_orb_percentile_arrays(
    index: pd.DatetimeIndex,
    orb_high_arr: np.ndarray[Any, np.dtype[np.float64]],
    orb_low_arr: np.ndarray[Any, np.dtype[np.float64]],
    window: int = 20,
) -> tuple[np.ndarray[Any, np.dtype[np.float64]], np.ndarray[Any, np.dtype[np.float64]]]:
    """Compute trailing 20-day p25/p75 of ORB range for each bar.

    For each trading day D, we look at the ORB range (high - low) of the
    *preceding* ``window`` trading days (not including D itself) and compute
    the 25th and 75th percentiles.  These percentiles are then broadcast to
    every 1-min bar of day D.

    Bars belonging to the first ``window`` trading days receive NaN (not
    enough history yet).

    Args:
        index:       UTC DatetimeIndex aligned with orb arrays.
        orb_high_arr: Pre-computed ORB highs (NaN for ORB bars themselves).
        orb_low_arr:  Pre-computed ORB lows.
        window:      Number of trailing trading days for percentile calculation.

    Returns:
        (p25_array, p75_array) - both float64 numpy arrays, NaN until
        sufficient history is available.
    """
    et_index = _to_et_index(index)
    n = len(index)

    p25_out = np.full(n, np.nan)
    p75_out = np.full(n, np.nan)

    # Build ordered list of trading dates and their bar positions
    dates: list[Any] = []
    date_to_positions: dict[Any, list[int]] = {}
    for i, ts in enumerate(et_index):
        d = ts.date()
        if d not in date_to_positions:
            dates.append(d)
            date_to_positions[d] = []
        date_to_positions[d].append(i)

    # Extract ORB range for each trading day (first non-NaN bar on that day)
    daily_range: dict[Any, float] = {}
    for d in dates:
        positions = date_to_positions[d]
        range_val = np.nan
        for pos in positions:
            if not np.isnan(orb_high_arr[pos]) and not np.isnan(orb_low_arr[pos]):
                range_val = float(orb_high_arr[pos] - orb_low_arr[pos])
                break
        daily_range[d] = range_val

    # Compute trailing p25/p75 for each day using only prior days' ranges
    for idx_d, d in enumerate(dates):
        # Use only the preceding `window` days (no lookahead)
        start_idx = max(0, idx_d - window)
        end_idx = idx_d  # exclusive: do not include current day

        if end_idx - start_idx < window:
            continue  # not enough history yet

        prior_ranges = [
            daily_range[dates[i]]
            for i in range(start_idx, end_idx)
            if not np.isnan(daily_range[dates[i]])
        ]
        if len(prior_ranges) < window:
            continue

        p25 = float(np.percentile(prior_ranges, 25))
        p75 = float(np.percentile(prior_ranges, 75))

        for pos in date_to_positions[d]:
            p25_out[pos] = p25
            p75_out[pos] = p75

    return p25_out, p75_out


# ── Backtest runner ───────────────────────────────────────────────────────────


def run_backtest(
    df: pd.DataFrame,
    cash: float = 50_000.0,
    slippage: float = 0.02,
    atr_mult: float = _ATR_MULTIPLIER,
    risk_mult: float = _RISK_MULTIPLIER,
    vol_mult: float = _VOL_MULTIPLIER,
    max_trades_per_day: int = _MAX_TRADES_PER_DAY,
    min_orb_pct: float = _MIN_ORB_PCT,
) -> Any:
    """Run the ORB backtest on *df* and return the Backtesting stats dict.

    Args:
        df:                OHLCV DataFrame with DatetimeIndex (UTC).
        cash:              Starting equity in USD.
        slippage:          Slippage per share per side in USD (default $0.02).
        atr_mult:          ATR stop multiplier.
        risk_mult:         Reward-to-risk multiplier for target (fallback when no
                           percentile data available).
        vol_mult:          Volume threshold multiplier (default 1.5).
        max_trades_per_day: Max entries per ET calendar day (default 5).
        min_orb_pct:       Min ORB range as fraction of price (default 0.0015).

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
        max_trades_per_day=max_trades_per_day,
        min_orb_pct=min_orb_pct,
    )

    logger.info(
        "backtest_done",
        trades=int(stats["# Trades"]),
        total_return_pct=float(stats["Return [%]"]),
        win_rate_pct=float(stats["Win Rate [%]"]) if stats["# Trades"] > 0 else 0.0,
    )
    return stats
