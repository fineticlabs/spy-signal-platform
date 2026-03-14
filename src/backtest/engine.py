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
- Trading window: 9:35-10:00 ET only.  ORB momentum exhausts within the first 25 min.
  10:xx WR=27% and power-hour WR=34% across 5.6 years under all regime conditions —
  structural time-of-day decay that no filter can fix.
- 15-min EMA(20) trend alignment: LONG only if close > EMA, SHORT only if close < EMA.
  EMA is computed by resampling 1-min to 15-min, shifted 1 bar, forward-filled back.
- Realized volatility filter (VIX proxy): 20-day rolling annualized std of daily SPY
  returns.  Only trade when realized vol < 18% (~VIX 20).  Computed from prior day's
  window (no lookahead).  Filters out bear/high-vol regimes where ORB breakouts fail.
- Daily ADX(14) trending filter: resampled to daily OHLCV, shifted 1 day.  Only trade
  when daily ADX > 25.  Confirms the broader market is trending not ranging.  Falls back
  to both-allowed if ADX not yet available (first 14 trading days).
- Consecutive-candle confirmation: requires TWO consecutive 1-min closes above ORB high
  (longs) or below ORB low (shorts) before entry.  Filters single-candle fakeouts that
  immediately reverse — primary failure mode in 2020-2022.
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
- First-5-min relative volume (RVOL): sum volume of first 5 bars (9:30-9:34 ET),
  divide by 20-day rolling average of same.  Informational only in backtest
  (no blocking).  Live strategy uses RVOL for confidence adjustment:
  RVOL < 0.5 → demote (-2 confidence, LOW_RVOL tag);
  RVOL >= 1.5 → boost (+1 confidence, HIGH_RVOL tag).
- Economic calendar filter: blocks all ORB trades on days with FOMC, NFP,
  CPI, or PPI releases.  The opening range on these days is unreliable due
  to pre-release positioning and post-release volatility spikes.
- Broad market direction confirmation: informational only (no blocking).
  For non-SPY/QQQ tickers, checks SPY's position vs its intraday VWAP.
  Live strategy tags signals with [SPY_ALIGNED] (+1 confidence) or
  [SPY_CONFLICT] (-2 confidence) for manual discretion.
- Earnings proximity filter: informational only (no blocking).  Tracks
  earnings day + day after per ticker via yfinance dates cached locally
  in data/earnings_cache.json.  Live strategy tags signals with [EARNINGS].
- Prior-day volume profile: informational only (no blocking).  Computes
  POC, Value Area (VAH/VAL), HVN, LVN from prior day's intraday bars.
  Tags trades: VP_LVN_TARGET (target beyond VA, +1 confidence in live),
  VP_HVN_TARGET (target inside VA → BLOCKED), VP_POC_CROSS (path
  crosses POC, -1 confidence).
- VIX term structure: ratio = VIX / VIX3M from prior day.
  BACKWARDATION (ratio > 1.00) → BLOCKED (19% WR, PF 0.65 in backtest).
  CONTANGO (ratio < 0.85) → informational tag only (+1 confidence in live).
  Normal (0.85-1.00) → no tag.
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

from src.backtest.volume_profile import compute_volume_profile
from src.filters.earnings_calendar import compute_earnings_blocked_array
from src.filters.economic_calendar import compute_econ_blocked_array
from src.filters.vix_term_structure import (
    BACKWARDATION_THRESHOLD,
    classify_term_structure,
)

logger = structlog.get_logger(__name__)

# ── constants ─────────────────────────────────────────────────────────────────

_ET_TZ = ZoneInfo("America/New_York")
_ORB_BARS = 5  # number of 1-min bars in the opening range (9:30-9:34)
_ATR_PERIOD = 14
_ATR_MULTIPLIER = 1.5
_RISK_MULTIPLIER = 2.0
_VOL_WINDOW = 20
_VOL_MULTIPLIER = 1.5  # restored: 2.0x was too strict, filtered too many setups

# Trading window — 9:35-10:00 ET only.
# ORB momentum exhausts within the first 25 min after the range is set.
# 10:xx WR=27% and power-hour WR=34% across all regime conditions tested —
# structural time-of-day decay that filters cannot fix.
_WINDOW1_START = time(9, 35)
_WINDOW1_END = time(10, 0)
_FORCE_FLAT = time(15, 55)

_MAX_TRADES_PER_DAY = 5
_MIN_ORB_PCT = 0.0015
_EMA15M_PERIOD = 20  # EMA period on 15-min bars for trend alignment
_ORB_RANGE_WINDOW = 20  # trailing days for ORB range percentiles
_GAP_THRESHOLD_PCT = 0.3  # ±0.3% gap separates directional bias from neutral

# Realized volatility filter (VIX proxy)
_REALIZED_VOL_WINDOW = 20  # trading days for rolling HV
_REALIZED_VOL_MAX = 0.18  # 18% annualized ~ VIX 20; blocks high-vol regimes

# Daily ADX trending filter
_DAILY_ADX_PERIOD = 14
_DAILY_ADX_MIN = 25.0  # daily ADX > 25 confirms trending market

# First-5-min relative volume (RVOL) — informational only (no blocking)
_RVOL_WINDOW = 20  # trailing days for rolling average of first-5-min volume

# Market direction confirmation — exempt tickers (they ARE the market)
_MARKET_DIRECTION_EXEMPT = {"SPY", "QQQ"}


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


# ── Realized volatility (VIX proxy) ──────────────────────────────────────────


def _compute_daily_realized_vol(
    index: pd.DatetimeIndex,
    close_prices: Any,
    window: int = _REALIZED_VOL_WINDOW,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Rolling annualized realized volatility broadcast to 1-min bars.

    For each trading day D the value is the rolling ``window``-day std of daily
    returns computed through day D-1 (no lookahead).  NaN until enough history.

    Approximates VIX: annualized HV ~18% ≈ VIX ~20, so filtering at 18%
    blocks the high-volatility bear-market regimes where ORB breakouts fail.

    Args:
        index:        UTC DatetimeIndex aligned with close_prices.
        close_prices: Numpy-like array of 1-min close prices.
        window:       Rolling window in trading days (default 20).

    Returns:
        Float64 numpy array of length ``len(index)``.
    """
    close_arr = np.asarray(close_prices, dtype=float)
    et_index = _to_et_index(index)
    n = len(index)

    # Build ordered list of trading dates → bar positions
    dates: list[Any] = []
    date_to_positions: dict[Any, list[int]] = {}
    for i, ts in enumerate(et_index):
        d = ts.date()
        if d not in date_to_positions:
            dates.append(d)
            date_to_positions[d] = []
        date_to_positions[d].append(i)

    # Daily close = last bar close of each ET date
    daily_closes = [float(close_arr[date_to_positions[d][-1]]) for d in dates]
    daily_close_series = pd.Series(daily_closes, dtype=float)

    # Daily log returns, rolling std, annualize, shift 1 day (no lookahead)
    daily_ret = daily_close_series.pct_change()
    rolling_vol = daily_ret.rolling(window).std() * float(np.sqrt(252))
    vol_shifted = rolling_vol.shift(1)  # use prior day's computed vol

    # Broadcast shifted vol to 1-min bars
    vol_out = np.full(n, np.nan)
    for idx_d, d in enumerate(dates):
        v = vol_shifted.iloc[idx_d]
        if not np.isnan(float(v)):
            for pos in date_to_positions[d]:
                vol_out[pos] = float(v)

    return vol_out


# ── Daily ADX ─────────────────────────────────────────────────────────────────


def _compute_daily_adx(
    index: pd.DatetimeIndex,
    high: Any,
    low: Any,
    close: Any,
    period: int = _DAILY_ADX_PERIOD,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Daily ADX(14) broadcast to 1-min bars, shifted 1 day for no lookahead.

    Resamples 1-min bars to daily OHLCV (high=max, low=min, close=last),
    computes TA-Lib ADX, shifts by 1 trading day, then forward-fills each
    day's value to all 1-min bars of that calendar date.

    Args:
        index:  UTC DatetimeIndex aligned with high/low/close.
        high:   Numpy-like array of 1-min bar highs.
        low:    Numpy-like array of 1-min bar lows.
        close:  Numpy-like array of 1-min bar closes.
        period: ADX lookback period (default 14).

    Returns:
        Float64 numpy array of length ``len(index)``.  NaN until enough
        daily history is available.
    """
    high_arr = np.asarray(high, dtype=float)
    low_arr = np.asarray(low, dtype=float)
    close_arr = np.asarray(close, dtype=float)
    et_index = _to_et_index(index)
    n = len(index)

    # Group 1-min bar positions by ET calendar date
    dates: list[Any] = []
    date_to_positions: dict[Any, list[int]] = {}
    for i, ts in enumerate(et_index):
        d = ts.date()
        if d not in date_to_positions:
            dates.append(d)
            date_to_positions[d] = []
        date_to_positions[d].append(i)

    # Build daily OHLCV arrays
    d_high = np.array([float(np.max(high_arr[date_to_positions[d]])) for d in dates])
    d_low = np.array([float(np.min(low_arr[date_to_positions[d]])) for d in dates])
    d_close = np.array([float(close_arr[date_to_positions[d][-1]]) for d in dates])

    if len(dates) < period + 1:
        return np.full(n, np.nan)

    # ADX on daily bars via TA-Lib
    adx_raw = talib.ADX(d_high, d_low, d_close, timeperiod=period)

    # Shift by 1 day — bar D uses ADX known through day D-1
    adx_series = pd.Series(adx_raw, dtype=float)
    adx_shifted = adx_series.shift(1).to_numpy()

    # Broadcast to 1-min bars
    adx_out = np.full(n, np.nan)
    for idx_d, d in enumerate(dates):
        v = adx_shifted[idx_d]
        if not np.isnan(float(v)):
            for pos in date_to_positions[d]:
                adx_out[pos] = float(v)

    return adx_out


# ── First-5-min relative volume (RVOL) ───────────────────────────────────────


def _compute_first5min_rvol(
    index: pd.DatetimeIndex,
    volume: Any,
    orb_bars: int = _ORB_BARS,
    window: int = _RVOL_WINDOW,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Rolling RVOL of the first 5-min session volume, broadcast to 1-min bars.

    For each trading day D:
        first5_vol = sum of volume for the first ``orb_bars`` bars (9:30-9:34 ET)
        avg_first5  = mean of first5_vol over the preceding ``window`` trading days
        RVOL        = first5_vol / avg_first5

    The RVOL value is constant for all bars of that trading day.  Days without
    enough history (fewer than ``window`` prior days) get NaN (default-allow).

    No lookahead: the first-5-min volume is fully observed by 9:35 ET, before
    any ORB signal can fire.

    Args:
        index:    UTC DatetimeIndex aligned with *volume*.
        volume:   Numpy-like array of 1-min bar volumes.
        orb_bars: Number of opening bars to sum (default 5).
        window:   Rolling average lookback in trading days (default 20).

    Returns:
        Float64 numpy array of length ``len(index)`` with RVOL per bar.
        NaN where insufficient history or data.
    """
    vol_arr = np.asarray(volume, dtype=float)
    et_index = _to_et_index(index)
    n = len(index)

    # Group bar positions by ET calendar date (ordered)
    dates: list[Any] = []
    date_to_positions: dict[Any, list[int]] = {}
    for i, ts in enumerate(et_index):
        d = ts.date()
        if d not in date_to_positions:
            dates.append(d)
            date_to_positions[d] = []
        date_to_positions[d].append(i)

    # Compute first-5-min volume for each trading day
    daily_first5_vol: list[float] = []
    for d in dates:
        positions = date_to_positions[d]
        if len(positions) >= orb_bars:
            first5 = float(np.sum(vol_arr[positions[:orb_bars]]))
        else:
            first5 = float("nan")
        daily_first5_vol.append(first5)

    # Rolling mean of prior ``window`` days, shifted by 1 (no lookahead)
    first5_series = pd.Series(daily_first5_vol, dtype=float)
    rolling_avg = first5_series.rolling(window).mean().shift(1)

    # RVOL = today's first-5-min vol / trailing average
    rvol_daily = first5_series / rolling_avg

    # Broadcast to 1-min bars
    rvol_out = np.full(n, np.nan)
    for idx_d, d in enumerate(dates):
        v = rvol_daily.iloc[idx_d]
        if not np.isnan(float(v)):
            for pos in date_to_positions[d]:
                rvol_out[pos] = float(v)

    return rvol_out


# ── Intraday VWAP computation ────────────────────────────────────────────────


def compute_intraday_vwap(
    index: pd.DatetimeIndex,
    high: Any,
    low: Any,
    close: Any,
    volume: Any,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Compute intraday VWAP, resetting at the start of each trading day.

    VWAP = cumulative(typical_price * volume) / cumulative(volume)
    where typical_price = (high + low + close) / 3.

    Args:
        index:  UTC DatetimeIndex aligned with price/volume arrays.
        high:   Numpy-like array of bar highs.
        low:    Numpy-like array of bar lows.
        close:  Numpy-like array of bar closes.
        volume: Numpy-like array of bar volumes.

    Returns:
        Float64 numpy array of VWAP values, same length as *index*.
    """
    h = np.asarray(high, dtype=float)
    l_ = np.asarray(low, dtype=float)
    c = np.asarray(close, dtype=float)
    v = np.asarray(volume, dtype=float)
    n = len(index)

    typical = (h + l_ + c) / 3.0
    tp_vol = typical * v

    et_index = _to_et_index(index)
    vwap_out = np.full(n, np.nan)

    # Group by ET calendar date
    date_groups: dict[Any, list[int]] = {}
    for i, ts in enumerate(et_index):
        d = ts.date()
        date_groups.setdefault(d, []).append(i)

    for positions in date_groups.values():
        cum_tpv = 0.0
        cum_vol = 0.0
        for pos in positions:
            cum_tpv += tp_vol[pos]
            cum_vol += v[pos]
            if cum_vol > 0:
                vwap_out[pos] = cum_tpv / cum_vol

    return vwap_out


# ── Prior-day Volume Profile ─────────────────────────────────────────────────


def _compute_prior_day_vp_arrays(
    index: pd.DatetimeIndex,
    high: Any,
    low: Any,
    close: Any,
    volume: Any,
) -> tuple[
    np.ndarray[Any, np.dtype[np.float64]],
    np.ndarray[Any, np.dtype[np.float64]],
    np.ndarray[Any, np.dtype[np.float64]],
    np.ndarray[Any, np.dtype[np.float64]],
    np.ndarray[Any, np.dtype[np.float64]],
]:
    """Compute prior-day volume profile levels broadcast to each trading day.

    For each trading day D, computes the volume profile from day D-1's
    intraday bars and broadcasts the resulting POC, VAH, VAL, nearest-HVN,
    and nearest-LVN to all 1-min bars of day D.

    No lookahead: day D only sees day D-1's completed volume profile.

    Args:
        index:  UTC DatetimeIndex aligned with price/volume arrays.
        high:   Numpy-like array of bar highs.
        low:    Numpy-like array of bar lows.
        close:  Numpy-like array of bar closes.
        volume: Numpy-like array of bar volumes.

    Returns:
        Tuple of 5 float64 numpy arrays (length = len(index)):
        (vp_poc, vp_vah, vp_val, vp_hvn_nearest, vp_lvn_nearest).
        NaN where no prior-day VP is available or no HVN/LVN exists.
    """
    high_arr = np.asarray(high, dtype=float)
    low_arr = np.asarray(low, dtype=float)
    close_arr = np.asarray(close, dtype=float)
    vol_arr = np.asarray(volume, dtype=float)
    et_index = _to_et_index(index)
    n = len(index)

    # Output arrays
    poc_out = np.full(n, np.nan)
    vah_out = np.full(n, np.nan)
    val_out = np.full(n, np.nan)
    hvn_out = np.full(n, np.nan)
    lvn_out = np.full(n, np.nan)

    # Group bar positions by ET calendar date (ordered)
    dates: list[Any] = []
    date_to_positions: dict[Any, list[int]] = {}
    for i, ts in enumerate(et_index):
        d = ts.date()
        if d not in date_to_positions:
            dates.append(d)
            date_to_positions[d] = []
        date_to_positions[d].append(i)

    # For each day D (starting from day 2), compute VP from day D-1
    for idx_d in range(1, len(dates)):
        d = dates[idx_d]
        d_prev = dates[idx_d - 1]

        prev_positions = date_to_positions[d_prev]
        today_positions = date_to_positions[d]

        if len(prev_positions) < 5:
            continue  # not enough bars for a meaningful VP

        # Extract prior day's OHLCV
        prev_h = high_arr[prev_positions]
        prev_l = low_arr[prev_positions]
        prev_c = close_arr[prev_positions]
        prev_v = vol_arr[prev_positions]

        vp = compute_volume_profile(prev_h, prev_l, prev_c, prev_v)
        if vp is None:
            continue

        # Compute today's midpoint for nearest-HVN/LVN calculation
        # Use the ORB midpoint (first 5 bars) as reference price
        if len(today_positions) >= 5:
            orb_high = float(np.max(high_arr[today_positions[:5]]))
            orb_low = float(np.min(low_arr[today_positions[:5]]))
            ref_price = (orb_high + orb_low) / 2.0
        else:
            ref_price = float(close_arr[today_positions[0]])

        # Find nearest HVN to reference price
        nearest_hvn = float("nan")
        if vp.hvn:
            nearest_hvn = min(vp.hvn, key=lambda x: abs(x - ref_price))

        # Find nearest LVN to reference price
        nearest_lvn = float("nan")
        if vp.lvn:
            nearest_lvn = min(vp.lvn, key=lambda x: abs(x - ref_price))

        # Broadcast to all bars of day D
        for pos in today_positions:
            poc_out[pos] = vp.poc
            vah_out[pos] = vp.vah
            val_out[pos] = vp.val_
            hvn_out[pos] = nearest_hvn
            lvn_out[pos] = nearest_lvn

    return poc_out, vah_out, val_out, hvn_out, lvn_out


# ── Backtesting.py Strategy ───────────────────────────────────────────────────


class ORBStrategy(Strategy):  # type: ignore[misc]
    """Opening Range Breakout implemented as a Backtesting.py Strategy.

    Parameters (can be optimised via ``Backtest.optimize``):
        atr_mult:           ATR multiplier for stop distance (default 1.5).
        risk_mult:          Risk-to-reward for target when no percentile data (default 2.0).
        vol_mult:           Volume multiplier threshold (default 1.5).
        max_trades_per_day: Max entries per calendar day ET (default 5).
        min_orb_pct:        Min ORB range as fraction of price (default 0.0015).
        symbol:             Ticker symbol for per-ticker filters (default "SPY").
    """

    atr_mult: float = _ATR_MULTIPLIER
    risk_mult: float = _RISK_MULTIPLIER
    vol_mult: float = _VOL_MULTIPLIER
    max_trades_per_day: int = _MAX_TRADES_PER_DAY
    min_orb_pct: float = _MIN_ORB_PCT
    symbol: str = "SPY"

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

        # Realized vol (VIX proxy) — 20-day rolling HV, shifted 1 day, < 18% to trade
        realized_vol_arr = _compute_daily_realized_vol(index, close_arr)
        self.realized_vol = self.I(lambda: realized_vol_arr, name="RealizedVol")

        # Daily ADX(14) — shifted 1 day, > 25 required to confirm trending regime
        daily_adx_arr = _compute_daily_adx(index, high, low, close)
        self.daily_adx = self.I(lambda: daily_adx_arr, name="DailyADX")

        # First-5-min RVOL — pre-computed on full dataset (passed as column) or
        # computed locally (only useful when full history is available)
        if hasattr(self.data, "RVOL"):
            rvol_arr = np.asarray(self.data.RVOL, dtype=float)
        else:
            rvol_arr = _compute_first5min_rvol(index, volume)
        self.rvol = self.I(lambda: rvol_arr, name="RVOL")

        # Economic calendar filter: block ORB on FOMC/NFP/CPI/PPI days
        econ_blocked_arr = compute_econ_blocked_array(index)
        self.econ_blocked = self.I(
            lambda: econ_blocked_arr.astype(float),
            name="EconBlocked",
        )

        # Earnings proximity filter: block ticker on earnings day + day after
        earnings_blocked_arr = compute_earnings_blocked_array(index, self.symbol)
        self.earnings_blocked = self.I(
            lambda: earnings_blocked_arr.astype(float),
            name="EarningsBlocked",
        )

        # Market direction confirmation: SPY VWAP + close for cross-ticker filter
        # SPY and QQQ are exempt (they ARE the market direction).
        self._needs_market_confirm = self.symbol not in _MARKET_DIRECTION_EXEMPT
        if self._needs_market_confirm and hasattr(self.data, "SPY_VWAP"):
            spy_vwap_arr = np.asarray(self.data.SPY_VWAP, dtype=float)
            spy_close_arr = np.asarray(self.data.SPY_CLOSE, dtype=float)
            self.spy_vwap = self.I(lambda: spy_vwap_arr, name="SPY_VWAP")
            self.spy_close = self.I(lambda: spy_close_arr, name="SPY_CLOSE")
        else:
            self.spy_vwap = None
            self.spy_close = None

        # Prior-day volume profile: informational only (no blocking)
        vp_poc, vp_vah, vp_val, vp_hvn, vp_lvn = _compute_prior_day_vp_arrays(
            index,
            high,
            low,
            close,
            volume,
        )
        self.vp_poc = self.I(lambda: vp_poc, name="VP_POC")
        self.vp_vah = self.I(lambda: vp_vah, name="VP_VAH")
        self.vp_val = self.I(lambda: vp_val, name="VP_VAL")
        self.vp_hvn = self.I(lambda: vp_hvn, name="VP_HVN")
        self.vp_lvn = self.I(lambda: vp_lvn, name="VP_LVN")

        # VIX term structure: informational only (no blocking)
        # Ratio = VIX / VIX3M, shifted 1 day (no lookahead).
        # Passed as a column from run_backtest.py (pre-loaded from cache/yfinance).
        if hasattr(self.data, "VIX_TERM_RATIO"):
            vts_arr = np.asarray(self.data.VIX_TERM_RATIO, dtype=float)
        else:
            vts_arr = np.full(len(index), np.nan)
        self.vix_term_ratio = self.I(lambda: vts_arr, name="VIX_Term_Ratio")

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

        # Only trade 9:35-10:00 ET — first 25 min after ORB completes
        if not (_WINDOW1_START <= t < _WINDOW1_END):
            return

        # Already in a position - nothing to enter
        if self.position:
            return

        # Reset daily trade counter on new ET calendar day
        today = _et_date(ts)
        if today != self._last_et_date:
            self._last_et_date = today
            self._daily_trade_count = 0

        # Skip Mondays — consistently worst WR and expectancy across all regimes
        if ts.tz_convert(_ET_TZ).dayofweek == 0:
            return

        # Skip high-impact economic event days (FOMC, NFP, CPI, PPI)
        if self.econ_blocked[-1] > 0.5:
            return

        # Skip backwardation days (VIX/VIX3M > 1.00 = near-term fear spike)
        vts_ratio_val = float(self.vix_term_ratio[-1])
        if not np.isnan(vts_ratio_val) and vts_ratio_val > BACKWARDATION_THRESHOLD:
            return

        # Earnings proximity: informational only (no blocking in backtest)
        # Live strategy tags signals with [EARNINGS] for manual discretion.

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

        # Realized vol filter: skip high-volatility regimes (~VIX > 20)
        realized_vol = self.realized_vol[-1]
        if not np.isnan(realized_vol) and realized_vol >= _REALIZED_VOL_MAX:
            return

        # Daily ADX filter: skip ranging markets (ADX <= 25)
        d_adx = self.daily_adx[-1]
        if not np.isnan(d_adx) and d_adx <= _DAILY_ADX_MIN:
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

        # Market direction confirmation: informational only (no blocking)
        # Live strategy uses SPY VWAP alignment for confidence adjustment.

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

        # Consecutive-candle confirmation: require prev bar also outside ORB
        prev_close = float(self.data.Close[-2]) if len(self.data.Close) >= 2 else float("nan")

        # ── VP intersection check ─────────────────────────────────────
        def _vp_check(target_price: float, entry_price: float) -> tuple[bool, str]:
            """Check VP intersection and return (blocked, tag_string).

            Returns:
                (blocked, tag): blocked=True if target is inside Value Area
                (VP_HVN_TARGET), meaning high-volume resistance will likely
                prevent target from being reached.  tag is a comma-separated
                string of VP tags for the trade log.
            """
            parts: list[str] = []
            vp_poc_val = float(self.vp_poc[-1])
            vp_vah_val = float(self.vp_vah[-1])
            vp_val_val = float(self.vp_val[-1])

            if np.isnan(vp_poc_val):
                return False, ""

            blocked = False

            # Target beyond Value Area → moving through LVN (low resistance)
            if target_price > vp_vah_val or target_price < vp_val_val:
                parts.append("VP_LVN_TARGET")

            # Target inside Value Area → HVN resistance → BLOCK
            if vp_val_val < target_price < vp_vah_val:
                parts.append("VP_HVN_TARGET")
                blocked = True

            # POC cross: entry-to-target path crosses POC
            if (entry_price < vp_poc_val < target_price) or (
                target_price < vp_poc_val < entry_price
            ):
                parts.append("VP_POC_CROSS")

            return blocked, ",".join(parts)

        # ── VIX term structure tag ────────────────────────────────────
        vts_ratio = float(self.vix_term_ratio[-1])
        vts_label = classify_term_structure(vts_ratio) if not np.isnan(vts_ratio) else None

        def _build_tag(vp_tag: str) -> str | None:
            """Combine VP tag and VTS tag into a single comma-separated string."""
            parts = [vp_tag] if vp_tag else []
            if vts_label:
                parts.append(vts_label)
            return ",".join(parts) if parts else None

        # LONG breakout: two consecutive closes above ORB high
        if (
            close > orb_high
            and not np.isnan(prev_close)
            and prev_close > orb_high
            and bullish_trend
            and gap_allows_long
        ):
            entry = close
            stop = entry - self.atr_mult * atr
            risk = entry - stop
            if risk <= 0:
                return
            target = entry + dynamic_risk_mult * risk
            vp_blocked, vp_tag = _vp_check(target, entry)
            if vp_blocked:
                return
            self.buy(sl=stop, tp=target, tag=_build_tag(vp_tag))
            self._daily_trade_count += 1

        # SHORT breakdown: two consecutive closes below ORB low
        elif (
            close < orb_low
            and not np.isnan(prev_close)
            and prev_close < orb_low
            and bearish_trend
            and gap_allows_short
        ):
            entry = close
            stop = entry + self.atr_mult * atr
            risk = stop - entry
            if risk <= 0:
                return
            target = entry - dynamic_risk_mult * risk
            vp_blocked, vp_tag = _vp_check(target, entry)
            if vp_blocked:
                return
            self.sell(sl=stop, tp=target, tag=_build_tag(vp_tag))
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
    symbol: str = "SPY",
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
        symbol:            Ticker symbol for per-ticker filters (default "SPY").

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
        symbol=symbol,
    )

    logger.info(
        "backtest_done",
        trades=int(stats["# Trades"]),
        total_return_pct=float(stats["Return [%]"]),
        win_rate_pct=float(stats["Win Rate [%]"]) if stats["# Trades"] > 0 else 0.0,
    )
    return stats
