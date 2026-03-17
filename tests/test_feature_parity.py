"""Feature parity tests: backtest (TA-Lib batch) vs live scanner (talipp streaming).

Validates that the two computation paths produce identical indicator values
for the same input data, within a tight tolerance.  Any divergence here means
live signals will not match backtest expectations.

Findings
--------
MATCH (< 0.01% relative error):
  - EMA(9), EMA(20), EMA(50): TA-Lib and talipp use identical EMA formula.
  - RSI(14): Both use Wilder's smoothing — perfect parity.
  - VWAP: Both use cumulative TP*V / cumulative V with daily reset — identical.
  - Volume average (20-bar rolling): pandas rolling vs deque — identical.
  - ORB high/low: numpy max/min vs streaming tracker — identical.
  - Realized volatility (20-day): Engine function matches manual pandas.
  - Daily ADX(14): Engine function matches manual TA-Lib on daily OHLC.
  - 15-min EMA(20): Engine function matches manual resample + TA-Lib.

DIVERGENCE — ATR(14):
  TA-Lib seeds ATR with a simple average of the first 14 true ranges, then
  applies Wilder's smoothing (EMA with alpha=1/14).  talipp uses a different
  initialization (SMA of first N true ranges with N = period - 1 or period).
  This causes ~0.12% error at bar 15 that decays exponentially to <0.01% by
  bar ~82.  In production the live scanner loads 390+ historical bars at
  startup before any signal fires, so this warmup divergence has zero impact
  on actual trading signals.  The test documents the decay pattern and asserts
  convergence below 0.01% after 100 bars.

DIVERGENCE — Gap %:
  The batch path computes gap from raw float64 OHLC.  The streaming path
  computes gap from Decimal-rounded prices (4 dp, matching Alpaca feed).
  This causes a ~0.015% relative error — below the 0.3% gap threshold used
  for directional filtering, so it has no practical impact.  The test uses
  a 0.05% tolerance for this specific case.
"""

from __future__ import annotations

from collections import deque
from decimal import Decimal
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import talib

from src.backtest.engine import (
    _compute_15m_ema,
    _compute_daily_adx,
    _compute_daily_realized_vol,
    _compute_gap_array,
    _compute_orb_arrays,
    compute_intraday_vwap,
)
from src.indicators.streaming import StreamingATR, StreamingEMA, StreamingRSI
from src.levels.opening_range import OpeningRangeTracker
from src.levels.vwap import SessionVWAP
from src.models import Bar, TimeFrame
from tests.conftest import make_1min_df

# ── Tolerance ────────────────────────────────────────────────────────────────
# 0.01% relative tolerance — tighter than any trading-relevant precision.
_REL_TOL = 1e-4
# Looser tolerance for float-vs-Decimal boundary tests (gap %).
_DECIMAL_REL_TOL = 5e-4

_ET = ZoneInfo("America/New_York")


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_bars_from_df(df: pd.DataFrame) -> list[Bar]:
    """Convert a DataFrame (from make_1min_df) to a list of Bar models."""
    bars: list[Bar] = []
    for ts, row in df.iterrows():
        bars.append(
            Bar(
                symbol="SPY",
                timeframe=TimeFrame.ONE_MIN,
                timestamp=ts.to_pydatetime(),
                open=Decimal(str(round(row["open"], 4))),
                high=Decimal(str(round(row["high"], 4))),
                low=Decimal(str(round(row["low"], 4))),
                close=Decimal(str(round(row["close"], 4))),
                volume=int(row["volume"]),
                vwap=Decimal(str(round((row["high"] + row["low"] + row["close"]) / 3, 4))),
            )
        )
    return bars


def _relative_error(a: float, b: float) -> float:
    """Relative error between two values.  Returns 0.0 when both are zero."""
    denom = max(abs(a), abs(b), 1e-12)
    return abs(a - b) / denom


def _collect_mismatches(
    batch: np.ndarray,
    streaming: list[float | None],
    warmup: int = 0,
    tol: float = _REL_TOL,
) -> list[tuple[int, float, float, float]]:
    """Return (index, batch_val, stream_val, rel_error) for divergent bars."""
    mismatches = []
    for i in range(warmup, len(streaming)):
        b = batch[i]
        s = streaming[i]
        if np.isnan(b) or s is None:
            continue
        err = _relative_error(b, s)
        if err > tol:
            mismatches.append((i, float(b), s, err))
    return mismatches


# ── ATR(14) ──────────────────────────────────────────────────────────────────


class TestATRParity:
    """ATR(14): TA-Lib batch vs talipp streaming.

    Known divergence: different SMA-seed initialization causes ~0.12% error
    near warmup that decays exponentially.  Converges to <0.01% by bar ~82.
    """

    def test_atr14_converges_after_warmup(self):
        """After 100 bars, TA-Lib and talipp ATR(14) agree within 0.01%."""
        df = make_1min_df(n_days=3, bars_per_day=390)
        bars = _make_bars_from_df(df)

        batch_atr = talib.ATR(
            df["high"].to_numpy(dtype=float),
            df["low"].to_numpy(dtype=float),
            df["close"].to_numpy(dtype=float),
            timeperiod=14,
        )

        streamer = StreamingATR(14)
        streaming: list[float | None] = []
        for bar in bars:
            streamer.update(bar)
            v = streamer.value
            streaming.append(float(v) if v is not None else None)

        # After 100 bars both should have converged
        mismatches = _collect_mismatches(batch_atr, streaming, warmup=100)
        assert len(mismatches) == 0, (
            f"ATR(14) divergence after bar 100: {len(mismatches)} bars. "
            f"First 3: {mismatches[:3]}"
        )

    def test_atr14_initialization_divergence_documented(self):
        """The early-bar divergence is real but decays below 0.01% by bar ~82.

        This test documents the divergence pattern: max error at the start of
        the post-warmup window, monotonically decaying as Wilder's smoothing
        washes out the seed difference.
        """
        df = make_1min_df(n_days=1, bars_per_day=390)
        bars = _make_bars_from_df(df)

        batch_atr = talib.ATR(
            df["high"].to_numpy(dtype=float),
            df["low"].to_numpy(dtype=float),
            df["close"].to_numpy(dtype=float),
            timeperiod=14,
        )

        streamer = StreamingATR(14)
        streaming: list[float | None] = []
        for bar in bars:
            streamer.update(bar)
            v = streamer.value
            streaming.append(float(v) if v is not None else None)

        # Collect errors for bars 15-100 (the divergence zone)
        errors: list[float] = []
        for i in range(15, 100):
            b = batch_atr[i]
            s = streaming[i]
            if np.isnan(b) or s is None:
                continue
            errors.append(_relative_error(b, s))

        # The divergence should exist (peak > 0.01%)
        assert max(errors) > _REL_TOL, "Expected initialization divergence not found"

        # The divergence should decay (later errors smaller than earlier)
        first_half_max = max(errors[: len(errors) // 2])
        second_half_max = max(errors[len(errors) // 2 :])
        assert second_half_max < first_half_max, "ATR divergence should decay over time"

        # By bar 100 it should be below tolerance
        assert (
            errors[-1] < _REL_TOL
        ), f"ATR divergence should be <0.01% by bar 100, got {errors[-1]:.6e}"

    def test_atr14_both_produce_values(self):
        """Both paths produce non-NaN/non-None values after warmup."""
        df = make_1min_df(n_days=1, bars_per_day=390)
        bars = _make_bars_from_df(df)

        batch_atr = talib.ATR(
            df["high"].to_numpy(dtype=float),
            df["low"].to_numpy(dtype=float),
            df["close"].to_numpy(dtype=float),
            timeperiod=14,
        )

        streamer = StreamingATR(14)
        for bar in bars:
            streamer.update(bar)

        assert not np.isnan(batch_atr[-1])
        assert streamer.value is not None


# ── EMA ──────────────────────────────────────────────────────────────────────


class TestEMAParity:
    """EMA: TA-Lib batch vs talipp streaming.  Perfect parity confirmed."""

    def test_ema20_values_match(self):
        """EMA(20) on 1-min close prices: TA-Lib vs talipp."""
        df = make_1min_df(n_days=3, bars_per_day=390)
        bars = _make_bars_from_df(df)
        close = df["close"].to_numpy(dtype=float)
        batch_ema = talib.EMA(close, timeperiod=20)

        streamer = StreamingEMA(20)
        streaming: list[float | None] = []
        for bar in bars:
            streamer.update(bar)
            v = streamer.value
            streaming.append(float(v) if v is not None else None)

        mismatches = _collect_mismatches(batch_ema, streaming, warmup=50)
        assert (
            len(mismatches) == 0
        ), f"EMA(20) divergence at {len(mismatches)} bars. First 3: {mismatches[:3]}"

    def test_ema9_values_match(self):
        """EMA(9) on 1-min close prices: TA-Lib vs talipp."""
        df = make_1min_df(n_days=2, bars_per_day=390)
        bars = _make_bars_from_df(df)
        close = df["close"].to_numpy(dtype=float)
        batch_ema = talib.EMA(close, timeperiod=9)

        streamer = StreamingEMA(9)
        streaming: list[float | None] = []
        for bar in bars:
            streamer.update(bar)
            v = streamer.value
            streaming.append(float(v) if v is not None else None)

        mismatches = _collect_mismatches(batch_ema, streaming, warmup=30)
        assert (
            len(mismatches) == 0
        ), f"EMA(9) divergence at {len(mismatches)} bars. First 3: {mismatches[:3]}"

    def test_ema50_values_match(self):
        """EMA(50) on 1-min close prices: TA-Lib vs talipp."""
        df = make_1min_df(n_days=3, bars_per_day=390)
        bars = _make_bars_from_df(df)
        close = df["close"].to_numpy(dtype=float)
        batch_ema = talib.EMA(close, timeperiod=50)

        streamer = StreamingEMA(50)
        streaming: list[float | None] = []
        for bar in bars:
            streamer.update(bar)
            v = streamer.value
            streaming.append(float(v) if v is not None else None)

        mismatches = _collect_mismatches(batch_ema, streaming, warmup=80)
        assert (
            len(mismatches) == 0
        ), f"EMA(50) divergence at {len(mismatches)} bars. First 3: {mismatches[:3]}"


# ── RSI(14) ──────────────────────────────────────────────────────────────────


class TestRSIParity:
    """RSI(14): TA-Lib batch vs talipp streaming.  Perfect parity confirmed."""

    def test_rsi14_values_match(self):
        """RSI(14) on 1-min close prices: TA-Lib vs talipp."""
        df = make_1min_df(n_days=3, bars_per_day=390)
        bars = _make_bars_from_df(df)
        close = df["close"].to_numpy(dtype=float)
        batch_rsi = talib.RSI(close, timeperiod=14)

        streamer = StreamingRSI(14)
        streaming: list[float | None] = []
        for bar in bars:
            streamer.update(bar)
            v = streamer.value
            streaming.append(float(v) if v is not None else None)

        mismatches = _collect_mismatches(batch_rsi, streaming, warmup=50)
        assert (
            len(mismatches) == 0
        ), f"RSI(14) divergence at {len(mismatches)} bars. First 3: {mismatches[:3]}"


# ── VWAP ─────────────────────────────────────────────────────────────────────


class TestVWAPParity:
    """Intraday VWAP: backtest batch vs live SessionVWAP streaming."""

    def test_vwap_values_match(self):
        """VWAP computed batch vs streaming bar-by-bar."""
        df = make_1min_df(n_days=2, bars_per_day=390)
        bars = _make_bars_from_df(df)

        batch_vwap = compute_intraday_vwap(
            index=df.index,
            high=df["high"].to_numpy(dtype=float),
            low=df["low"].to_numpy(dtype=float),
            close=df["close"].to_numpy(dtype=float),
            volume=df["volume"].to_numpy(dtype=float),
        )

        streamer = SessionVWAP()
        streaming: list[float | None] = []
        for bar in bars:
            streamer.update(bar)
            v = streamer.vwap
            streaming.append(float(v) if v is not None else None)

        mismatches = _collect_mismatches(batch_vwap, streaming)
        assert (
            len(mismatches) == 0
        ), f"VWAP divergence at {len(mismatches)} bars. First 3: {mismatches[:3]}"

    def test_vwap_resets_daily(self):
        """Both paths reset VWAP at the start of each trading day."""
        df = make_1min_df(n_days=3, bars_per_day=390)
        bars = _make_bars_from_df(df)

        batch_vwap = compute_intraday_vwap(
            index=df.index,
            high=df["high"].to_numpy(dtype=float),
            low=df["low"].to_numpy(dtype=float),
            close=df["close"].to_numpy(dtype=float),
            volume=df["volume"].to_numpy(dtype=float),
        )

        streamer = SessionVWAP()
        streaming: list[float | None] = []
        for bar in bars:
            streamer.update(bar)
            v = streamer.vwap
            streaming.append(float(v) if v is not None else None)

        # Check day boundary parity
        for start_idx in [390, 780]:
            b = batch_vwap[start_idx]
            s = streaming[start_idx]
            if not np.isnan(b) and s is not None:
                err = _relative_error(b, s)
                assert err <= _REL_TOL, f"VWAP mismatch at day boundary bar {start_idx}"


# ── Volume Average (20-bar rolling) ─────────────────────────────────────────


class TestVolumeAverageParity:
    """20-bar rolling volume average: pandas batch vs manual streaming."""

    def test_volume_avg_values_match(self):
        """Rolling 20-bar volume average matches between batch and streaming."""
        df = make_1min_df(n_days=2, bars_per_day=390)

        # Batch: pandas rolling (same as backtest engine)
        vol_series = pd.Series(df["volume"].to_numpy(dtype=float))
        batch_avg = vol_series.rolling(20).mean().to_numpy()

        # Streaming: manual deque (same approach as live ORB strategy)
        window: deque[float] = deque(maxlen=20)
        streaming: list[float | None] = []
        for _, row in df.iterrows():
            window.append(float(row["volume"]))
            if len(window) == 20:
                streaming.append(sum(window) / len(window))
            else:
                streaming.append(None)

        mismatches = _collect_mismatches(batch_avg, streaming, warmup=20)
        assert (
            len(mismatches) == 0
        ), f"Volume avg divergence at {len(mismatches)} bars. First 3: {mismatches[:3]}"


# ── Opening Range High/Low ──────────────────────────────────────────────────


class TestORBParity:
    """ORB high/low: backtest numpy vs live OpeningRangeTracker."""

    def test_orb_high_low_match(self):
        """ORB high/low from batch arrays vs streaming tracker."""
        df = make_1min_df(n_days=3, bars_per_day=390)
        bars = _make_bars_from_df(df)

        orb_high_arr, orb_low_arr = _compute_orb_arrays(
            index=df.index,
            high=df["high"].to_numpy(dtype=float),
            low=df["low"].to_numpy(dtype=float),
            orb_bars=5,
        )

        tracker = OpeningRangeTracker()
        streaming_highs: list[float | None] = []
        streaming_lows: list[float | None] = []
        for bar in bars:
            tracker.update(bar)
            if tracker.is_complete:
                h = tracker.orb_high
                lo = tracker.orb_low
                streaming_highs.append(float(h) if h is not None else None)
                streaming_lows.append(float(lo) if lo is not None else None)
            else:
                streaming_highs.append(None)
                streaming_lows.append(None)

        high_mismatches = []
        low_mismatches = []
        for i in range(len(bars)):
            bh = orb_high_arr[i]
            sh = streaming_highs[i]
            bl = orb_low_arr[i]
            sl = streaming_lows[i]

            if np.isnan(bh) and sh is None:
                continue
            if np.isnan(bh) or sh is None:
                high_mismatches.append((i, bh, sh, "one_nan"))
                continue
            err = _relative_error(bh, sh)
            if err > _REL_TOL:
                high_mismatches.append((i, bh, sh, err))

            if np.isnan(bl) or sl is None:
                low_mismatches.append((i, bl, sl, "one_nan"))
                continue
            err = _relative_error(bl, sl)
            if err > _REL_TOL:
                low_mismatches.append((i, bl, sl, err))

        assert (
            len(high_mismatches) == 0
        ), f"ORB high divergence: {len(high_mismatches)} bars. First 3: {high_mismatches[:3]}"
        assert (
            len(low_mismatches) == 0
        ), f"ORB low divergence: {len(low_mismatches)} bars. First 3: {low_mismatches[:3]}"


# ── Gap Percentage ───────────────────────────────────────────────────────────


class TestGapParity:
    """Gap %: backtest batch vs manual live computation.

    Known divergence: batch uses raw float64, live uses Decimal-rounded prices.
    Max error ~0.015% — far below the 0.3% gap threshold for directional bias.
    """

    def test_gap_pct_matches_within_decimal_tolerance(self):
        """Gap % agrees within 0.05% (accounts for float/Decimal rounding)."""
        df = make_1min_df(n_days=5, bars_per_day=390)
        bars = _make_bars_from_df(df)

        batch_gap = _compute_gap_array(
            index=df.index,
            open_prices=df["open"].to_numpy(dtype=float),
            close_prices=df["close"].to_numpy(dtype=float),
        )

        # Streaming: compute gap from Bar models (Decimal prices, like live)
        prev_day_close: Decimal | None = None
        current_date_str: str | None = None
        current_gap: float | None = None
        streaming_gap: list[float | None] = []

        for bar in bars:
            bar_date = bar.timestamp.astimezone(_ET).date()
            bar_str = str(bar_date)

            if current_date_str is None:
                current_date_str = bar_str
                streaming_gap.append(None)
                continue

            if bar_str != current_date_str:
                # New day boundary
                if prev_day_close is not None and prev_day_close > 0:
                    current_gap = float((bar.open - prev_day_close) / prev_day_close * 100)
                else:
                    current_gap = None
                current_date_str = bar_str

            streaming_gap.append(current_gap)
            prev_day_close = bar.close

        mismatches = []
        for i in range(len(bars)):
            b = batch_gap[i]
            s = streaming_gap[i]
            if np.isnan(b) and s is None:
                continue
            if np.isnan(b) or s is None:
                continue
            err = _relative_error(b, s)
            if err > _DECIMAL_REL_TOL:
                mismatches.append((i, b, s, err))

        assert len(mismatches) == 0, (
            f"Gap % divergence beyond 0.05%: {len(mismatches)} bars. " f"First 3: {mismatches[:3]}"
        )

    def test_gap_direction_always_agrees(self):
        """Gap directional classification (long-only, short-only, both) always matches.

        Even with the float/Decimal rounding difference, the gap direction
        should never flip because the 0.3% threshold is 20x larger than
        the rounding error.
        """
        df = make_1min_df(n_days=5, bars_per_day=390)
        bars = _make_bars_from_df(df)
        threshold = 0.3

        batch_gap = _compute_gap_array(
            index=df.index,
            open_prices=df["open"].to_numpy(dtype=float),
            close_prices=df["close"].to_numpy(dtype=float),
        )

        # Streaming gap (same logic as test above)
        prev_day_close: Decimal | None = None
        current_date_str: str | None = None
        current_gap: float | None = None
        streaming_gap: list[float | None] = []

        for bar in bars:
            bar_date = bar.timestamp.astimezone(_ET).date()
            bar_str = str(bar_date)

            if current_date_str is None:
                current_date_str = bar_str
                streaming_gap.append(None)
                continue

            if bar_str != current_date_str:
                if prev_day_close is not None and prev_day_close > 0:
                    current_gap = float((bar.open - prev_day_close) / prev_day_close * 100)
                else:
                    current_gap = None
                current_date_str = bar_str

            streaming_gap.append(current_gap)
            prev_day_close = bar.close

        def _classify(gap: float | None) -> str:
            if gap is None or np.isnan(gap):
                return "both"
            if gap > threshold:
                return "long_only"
            if gap < -threshold:
                return "short_only"
            return "both"

        direction_mismatches = []
        for i in range(len(bars)):
            b = batch_gap[i]
            s = streaming_gap[i]
            b_val = None if np.isnan(b) else float(b)
            batch_dir = _classify(b_val)
            stream_dir = _classify(s)
            if batch_dir != stream_dir:
                direction_mismatches.append((i, b_val, s, batch_dir, stream_dir))

        assert len(direction_mismatches) == 0, f"Gap direction mismatch: {direction_mismatches[:5]}"


# ── Realized Volatility (20-day) ────────────────────────────────────────────


class TestRealizedVolParity:
    """Realized vol: backtest batch computation vs manual reference."""

    def test_realized_vol_formula(self):
        """Engine function matches manual pandas: rolling std * sqrt(252)."""
        df = make_1min_df(n_days=30, bars_per_day=390)

        batch_vol = _compute_daily_realized_vol(
            index=df.index,
            close_prices=df["close"].to_numpy(dtype=float),
            window=20,
        )

        # Manual reference
        et_index = df.index.tz_convert(_ET)
        date_groups: dict = {}
        for i, ts in enumerate(et_index):
            d = ts.date()
            date_groups.setdefault(d, []).append(i)

        dates = list(date_groups.keys())
        close_arr = df["close"].to_numpy(dtype=float)
        daily_closes = [close_arr[date_groups[d][-1]] for d in dates]
        daily_series = pd.Series(daily_closes, dtype=float)
        daily_ret = daily_series.pct_change()
        rolling_vol = daily_ret.rolling(20).std() * float(np.sqrt(252))
        vol_shifted = rolling_vol.shift(1)

        mismatches = []
        for idx_d, d in enumerate(dates):
            first_bar = date_groups[d][0]
            manual = vol_shifted.iloc[idx_d]
            engine = batch_vol[first_bar]

            if pd.isna(manual) and np.isnan(engine):
                continue
            if pd.isna(manual) or np.isnan(engine):
                mismatches.append((idx_d, float(manual), engine, "one_nan"))
                continue
            err = _relative_error(float(manual), engine)
            if err > _REL_TOL:
                mismatches.append((idx_d, float(manual), engine, err))

        assert (
            len(mismatches) == 0
        ), f"Realized vol divergence: {len(mismatches)} days. First 3: {mismatches[:3]}"


# ── Daily ADX(14) ───────────────────────────────────────────────────────────


class TestDailyADXParity:
    """Daily ADX(14): engine function vs manual TA-Lib on daily OHLC."""

    def test_daily_adx_matches_manual(self):
        """Engine _compute_daily_adx matches raw TA-Lib on daily bars."""
        df = make_1min_df(n_days=30, bars_per_day=390)

        batch_adx = _compute_daily_adx(
            index=df.index,
            high=df["high"].to_numpy(dtype=float),
            low=df["low"].to_numpy(dtype=float),
            close=df["close"].to_numpy(dtype=float),
            period=14,
        )

        # Manual: resample to daily, run TA-Lib ADX, shift 1 day
        et_index = df.index.tz_convert(_ET)
        date_groups: dict = {}
        ordered_dates: list = []
        for i, ts in enumerate(et_index):
            d = ts.date()
            if d not in date_groups:
                ordered_dates.append(d)
                date_groups[d] = []
            date_groups[d].append(i)

        high_arr = df["high"].to_numpy(dtype=float)
        low_arr = df["low"].to_numpy(dtype=float)
        close_arr = df["close"].to_numpy(dtype=float)

        d_high = np.array([np.max(high_arr[date_groups[d]]) for d in ordered_dates])
        d_low = np.array([np.min(low_arr[date_groups[d]]) for d in ordered_dates])
        d_close = np.array([close_arr[date_groups[d][-1]] for d in ordered_dates])

        manual_adx_raw = talib.ADX(d_high, d_low, d_close, timeperiod=14)
        manual_shifted = pd.Series(manual_adx_raw, dtype=float).shift(1).to_numpy()

        mismatches = []
        for idx_d, d in enumerate(ordered_dates):
            first_bar = date_groups[d][0]
            manual = manual_shifted[idx_d]
            engine = batch_adx[first_bar]

            if np.isnan(manual) and np.isnan(engine):
                continue
            if np.isnan(manual) or np.isnan(engine):
                mismatches.append((idx_d, manual, engine, "one_nan"))
                continue
            err = _relative_error(manual, engine)
            if err > _REL_TOL:
                mismatches.append((idx_d, manual, engine, err))

        assert (
            len(mismatches) == 0
        ), f"Daily ADX divergence: {len(mismatches)} days. First 3: {mismatches[:3]}"


# ── 15-min EMA(20) ──────────────────────────────────────────────────────────


class TestEMA15mParity:
    """15-min EMA(20): engine function vs manual TA-Lib on 15-min resampled."""

    def test_15m_ema_matches_manual(self):
        """_compute_15m_ema matches raw TA-Lib on 15-min bars."""
        df = make_1min_df(n_days=10, bars_per_day=390)

        batch_ema = _compute_15m_ema(
            index=df.index,
            close=df["close"].to_numpy(dtype=float),
            period=20,
        )

        # Manual: resample, run TA-Lib, shift, forward-fill
        et_index = df.index.tz_convert(_ET)
        close_series = pd.Series(df["close"].to_numpy(dtype=float), index=et_index)
        close_15m = close_series.resample("15min").last().dropna()
        ema_raw = talib.EMA(close_15m.to_numpy(dtype=float), timeperiod=20)
        ema_15m = pd.Series(ema_raw, index=close_15m.index)
        ema_shifted = ema_15m.shift(1)
        ema_1m = ema_shifted.reindex(et_index, method="ffill").to_numpy(dtype=float)

        mismatches = []
        for i in range(len(df)):
            b = batch_ema[i]
            m = ema_1m[i]
            if np.isnan(b) and np.isnan(m):
                continue
            if np.isnan(b) or np.isnan(m):
                mismatches.append((i, b, m, "one_nan"))
                continue
            err = _relative_error(b, m)
            if err > _REL_TOL:
                mismatches.append((i, b, m, err))

        assert (
            len(mismatches) == 0
        ), f"15-min EMA divergence: {len(mismatches)} bars. First 3: {mismatches[:3]}"


# ── Cross-library summary ───────────────────────────────────────────────────


class TestCrossLibrarySummary:
    """Comprehensive cross-library comparison on 780 bars.

    Runs ATR, EMA, RSI, VWAP and asserts all within tolerance.
    ATR uses warmup=100 to skip the known initialization divergence zone.
    """

    def test_all_indicators_within_tolerance(self):
        """All shared indicators on 780 bars within 0.01% after warmup."""
        df = make_1min_df(n_days=2, bars_per_day=390)
        bars = _make_bars_from_df(df)

        high = df["high"].to_numpy(dtype=float)
        low = df["low"].to_numpy(dtype=float)
        close = df["close"].to_numpy(dtype=float)
        volume = df["volume"].to_numpy(dtype=float)

        results: dict[str, dict[str, float]] = {}

        # ATR(14) — use warmup=100 to skip known init divergence
        batch_atr = talib.ATR(high, low, close, timeperiod=14)
        streamer_atr = StreamingATR(14)
        max_err_atr = 0.0
        compared_atr = 0
        for i, bar in enumerate(bars):
            streamer_atr.update(bar)
            v = streamer_atr.value
            if v is not None and not np.isnan(batch_atr[i]) and i >= 100:
                err = _relative_error(batch_atr[i], float(v))
                max_err_atr = max(max_err_atr, err)
                compared_atr += 1
        results["ATR(14)"] = {"max_error": max_err_atr, "compared": compared_atr}

        # EMA(20)
        batch_ema = talib.EMA(close, timeperiod=20)
        streamer_ema = StreamingEMA(20)
        max_err_ema = 0.0
        compared_ema = 0
        for i, bar in enumerate(bars):
            streamer_ema.update(bar)
            v = streamer_ema.value
            if v is not None and not np.isnan(batch_ema[i]) and i >= 30:
                err = _relative_error(batch_ema[i], float(v))
                max_err_ema = max(max_err_ema, err)
                compared_ema += 1
        results["EMA(20)"] = {"max_error": max_err_ema, "compared": compared_ema}

        # RSI(14)
        batch_rsi = talib.RSI(close, timeperiod=14)
        streamer_rsi = StreamingRSI(14)
        max_err_rsi = 0.0
        compared_rsi = 0
        for i, bar in enumerate(bars):
            streamer_rsi.update(bar)
            v = streamer_rsi.value
            if v is not None and not np.isnan(batch_rsi[i]) and i >= 30:
                err = _relative_error(batch_rsi[i], float(v))
                max_err_rsi = max(max_err_rsi, err)
                compared_rsi += 1
        results["RSI(14)"] = {"max_error": max_err_rsi, "compared": compared_rsi}

        # VWAP
        batch_vwap = compute_intraday_vwap(
            index=df.index,
            high=high,
            low=low,
            close=close,
            volume=volume,
        )
        streamer_vwap = SessionVWAP()
        max_err_vwap = 0.0
        compared_vwap = 0
        for i, bar in enumerate(bars):
            streamer_vwap.update(bar)
            v = streamer_vwap.vwap
            if v is not None and not np.isnan(batch_vwap[i]):
                err = _relative_error(batch_vwap[i], float(v))
                max_err_vwap = max(max_err_vwap, err)
                compared_vwap += 1
        results["VWAP"] = {"max_error": max_err_vwap, "compared": compared_vwap}

        failures = []
        for name, stats in results.items():
            if stats["max_error"] > _REL_TOL:
                failures.append(
                    f"{name}: max_error={stats['max_error']:.6e} "
                    f"(compared {int(stats['compared'])} bars)"
                )

        assert len(failures) == 0, "Indicators exceeding 0.01% tolerance:\n" + "\n".join(failures)
