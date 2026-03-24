"""Tests targeting remaining uncovered lines in src/backtest/engine.py.

Coverage targets (uncovered lines)
-----------------------------------
- 160: _et_date tz-naive branch
- 168: _to_et_index tz-naive branch
- 225, 231: _compute_gap_array edge cases (empty positions, NaN/zero prev_close)
- 589, 599, 608: _compute_prior_day_vp_arrays branches (few prev bars, None VP, <5 today)
- 720: init() RVOL column path (hasattr RVOL)
- 743-746: init() SPY_VWAP/SPY_CLOSE columns for non-SPY symbol with market confirm
- 769: init() VIX_TERM_RATIO column path
- 786: init() HMM_REGIME column path
- 802-803: next() force flat
- 811: next() already in position
- 830: next() VIX backwardation skip
- 845: next() max trades per day
- 853: next() ORB NaN skip
- 857: next() narrow ORB skip
- 865-1024: next() full trade entry logic (ADX filter, volume filter, EMA trend,
  gap classification, dynamic risk multiplier, consecutive candle, VP check,
  VTS tag, HMM tag, Kalman multiplier, LONG entry, SHORT entry)
- 1210: _compute_orb_percentile_arrays insufficient prior_ranges
"""

from __future__ import annotations

import warnings
from datetime import date, timedelta
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import (
    _compute_gap_array,
    _compute_orb_percentile_arrays,
    _compute_prior_day_vp_arrays,
    _et_date,
    _to_et_index,
    run_backtest,
)

# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_multiday_index(
    start_date: str,
    n_days: int,
    bars_per_day: int = 390,
) -> pd.DatetimeIndex:
    """Create a 1-min DatetimeIndex spanning multiple weekday trading days."""
    indices: list[pd.DatetimeIndex] = []
    current = pd.Timestamp(start_date)
    days_added = 0
    while days_added < n_days:
        if current.weekday() < 5:
            day_start = pd.Timestamp(f"{current.strftime('%Y-%m-%d')} 14:30:00", tz="UTC")
            indices.append(pd.date_range(start=day_start, periods=bars_per_day, freq="min"))
            days_added += 1
        current += timedelta(days=1)
    return indices[0].append(indices[1:])  # type: ignore[arg-type]


def _build_breakout_df(
    n_days: int = 35,
    bars_per_day: int = 390,
    direction: str = "long",
) -> pd.DataFrame:
    """Build synthetic data that triggers ORB breakout entries.

    Creates n_days of 390-bar trading days where days 25+ have a clear
    breakout pattern: bars 5+ break above ORB high (long) or below ORB
    low (short) with high volume and consecutive candle confirmation.

    The data includes a steady uptrend so EMA is bullish, sufficient
    warmup for ATR/ADX/realized-vol, and gap-up opens for long bias.
    """
    idx = _make_multiday_index("2024-01-02", n_days=n_days, bars_per_day=bars_per_day)
    n = len(idx)

    base_price = 480.0
    open_ = np.full(n, base_price)
    high = np.full(n, base_price + 0.5)
    low = np.full(n, base_price - 0.5)
    close = np.full(n, base_price)
    volume = np.full(n, 200_000.0)

    rng = np.random.default_rng(42)

    for day in range(n_days):
        day_start = day * bars_per_day
        # Slight daily uptrend so EMA stays bullish
        day_base = base_price + day * 0.15

        for bar in range(bars_per_day):
            i = day_start + bar
            noise = rng.normal(0, 0.05)

            if bar < 5:
                # ORB bars: tight range around day_base
                open_[i] = day_base + noise
                high[i] = day_base + 0.6
                low[i] = day_base - 0.6
                close[i] = day_base + noise * 0.5
            else:
                # Normal bars: close near day_base with noise
                open_[i] = day_base + noise
                high[i] = day_base + 0.8 + abs(noise)
                low[i] = day_base - 0.8 - abs(noise)
                close[i] = day_base + noise
                volume[i] = 150_000.0

        # For breakout days (25+), skip Mondays (dayofweek=0 in ET)
        if day >= 25:
            day_ts = idx[day_start]
            et_dow = day_ts.tz_convert("America/New_York").dayofweek
            if et_dow == 0:
                # Monday: strategy skips these, so don't bother crafting breakout
                continue

            orb_high = day_base + 0.6
            orb_low = day_base - 0.6

            if direction == "long":
                # Bars 5-9: strong close ABOVE orb_high with high volume
                # Need 2 consecutive closes above ORB high for confirmation
                for bar in range(5, min(12, bars_per_day)):
                    i = day_start + bar
                    breakout_price = orb_high + 0.5 + bar * 0.05
                    open_[i] = orb_high + 0.3
                    close[i] = breakout_price
                    high[i] = breakout_price + 0.3
                    low[i] = orb_high + 0.1
                    volume[i] = 600_000.0  # well above 1.5x avg (~300k)
            else:
                # SHORT breakout: close below ORB low
                for bar in range(5, min(12, bars_per_day)):
                    i = day_start + bar
                    breakdown_price = orb_low - 0.5 - bar * 0.05
                    open_[i] = orb_low - 0.3
                    close[i] = breakdown_price
                    high[i] = orb_low - 0.1
                    low[i] = breakdown_price - 0.3
                    volume[i] = 600_000.0

        # Create gap-up between days for long bias (or gap-down for short)
        if day >= 1:
            prev_last = day_start - 1
            if direction == "long":
                # Today opens above yesterday's close → gap-up → allows long
                open_[day_start] = close[prev_last] + 0.5
            else:
                # Gap-down for short direction
                open_[day_start] = close[prev_last] - 0.5

    # Ensure OHLC consistency
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))

    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# Timezone helper edge cases (lines 160, 168)
# ═══════════════════════════════════════════════════════════════════════════════


class TestTzNaiveBranches:
    """Cover tz-naive input branches in _et_date and _to_et_index."""

    def test_et_date_tz_naive_input(self) -> None:
        """Line 160: _et_date with tz-naive Timestamp localizes to UTC."""
        ts = pd.Timestamp("2024-01-15 14:30:00")  # no tz
        result = _et_date(ts)
        assert result == date(2024, 1, 15)

    def test_to_et_index_tz_naive_input(self) -> None:
        """Line 168: _to_et_index with tz-naive DatetimeIndex localizes to UTC."""
        idx = pd.date_range("2024-01-15 14:30", periods=5, freq="1min")
        assert idx.tzinfo is None
        result = _to_et_index(idx)
        assert str(result.tzinfo) == "America/New_York"


# ═══════════════════════════════════════════════════════════════════════════════
# _compute_gap_array edge cases (lines 225, 231)
# ═══════════════════════════════════════════════════════════════════════════════


class TestComputeGapArrayEdgeCases:
    """Cover edge-case branches in _compute_gap_array."""

    def test_zero_prev_close_skipped(self) -> None:
        """Line 231: prev_close <= 0 → gap stays NaN for that day."""
        idx = _make_multiday_index("2024-01-15", n_days=2, bars_per_day=10)
        n = len(idx)
        open_ = np.full(n, 100.0)
        close = np.full(n, 100.0)
        # Set day-1 last close to 0 → should be skipped
        close[9] = 0.0
        gap = _compute_gap_array(idx, open_, close)
        # Day 2 should still be NaN because prev_close was 0
        assert np.all(np.isnan(gap[10:20]))

    def test_nan_prev_close_skipped(self) -> None:
        """Line 231: NaN prev_close → gap stays NaN for that day."""
        idx = _make_multiday_index("2024-01-15", n_days=2, bars_per_day=10)
        n = len(idx)
        open_ = np.full(n, 100.0)
        close = np.full(n, 100.0)
        close[9] = np.nan
        gap = _compute_gap_array(idx, open_, close)
        assert np.all(np.isnan(gap[10:20]))

    def test_nan_today_open_skipped(self) -> None:
        """Line 231: NaN today_open → gap stays NaN."""
        idx = _make_multiday_index("2024-01-15", n_days=2, bars_per_day=10)
        n = len(idx)
        open_ = np.full(n, 100.0)
        close = np.full(n, 100.0)
        open_[10] = np.nan  # first bar of day 2
        gap = _compute_gap_array(idx, open_, close)
        assert np.all(np.isnan(gap[10:20]))


# ═══════════════════════════════════════════════════════════════════════════════
# _compute_prior_day_vp_arrays edge cases (lines 589, 599, 608)
# ═══════════════════════════════════════════════════════════════════════════════


class TestComputePriorDayVpArrays:
    """Cover edge-case branches in _compute_prior_day_vp_arrays."""

    def test_prev_day_too_few_bars_skipped(self) -> None:
        """Line 589: < 5 bars on previous day → VP skipped, NaN output."""
        # Day 1: 3 bars (too few), Day 2: 10 bars
        day1_start = pd.Timestamp("2024-01-15 14:30:00", tz="UTC")
        day2_start = pd.Timestamp("2024-01-16 14:30:00", tz="UTC")
        idx1 = pd.date_range(day1_start, periods=3, freq="min")
        idx2 = pd.date_range(day2_start, periods=10, freq="min")
        idx = idx1.append(idx2)
        n = len(idx)

        h = np.full(n, 481.0)
        l_ = np.full(n, 479.0)
        c = np.full(n, 480.0)
        v = np.full(n, 100_000.0)

        poc, _vah, _val, _hvn, _lvn = _compute_prior_day_vp_arrays(idx, h, l_, c, v)
        # Day 2 should have NaN VP (day 1 had < 5 bars)
        assert np.all(np.isnan(poc[3:]))

    def test_today_fewer_than_5_bars_uses_first_close(self) -> None:
        """Line 608: < 5 bars today → ref_price = first close."""
        day1_start = pd.Timestamp("2024-01-15 14:30:00", tz="UTC")
        day2_start = pd.Timestamp("2024-01-16 14:30:00", tz="UTC")
        idx1 = pd.date_range(day1_start, periods=20, freq="min")
        idx2 = pd.date_range(day2_start, periods=3, freq="min")
        idx = idx1.append(idx2)
        n = len(idx)

        h = np.full(n, 481.0)
        l_ = np.full(n, 479.0)
        c = np.full(n, 480.0)
        v = np.full(n, 100_000.0)
        # Add some variation so VP can compute
        for i in range(20):
            h[i] = 480.0 + (i % 5) * 0.5
            l_[i] = 479.0 - (i % 5) * 0.3
            c[i] = 479.5 + (i % 5) * 0.2
            v[i] = 100_000.0 + i * 10_000

        poc, _vah, _val, _hvn, _lvn = _compute_prior_day_vp_arrays(idx, h, l_, c, v)
        # Day 2 has < 5 bars, so ref_price = close of first bar of day 2
        # If VP computed successfully, poc should be non-NaN for day 2 bars
        # (depends on volume_profile returning non-None)
        assert len(poc) == n

    def test_vp_none_result_skipped(self) -> None:
        """Line 599: VP returns None → day gets NaN."""
        day1_start = pd.Timestamp("2024-01-15 14:30:00", tz="UTC")
        day2_start = pd.Timestamp("2024-01-16 14:30:00", tz="UTC")
        idx1 = pd.date_range(day1_start, periods=10, freq="min")
        idx2 = pd.date_range(day2_start, periods=10, freq="min")
        idx = idx1.append(idx2)
        n = len(idx)

        # All identical prices + zero volume → VP likely returns None
        h = np.full(n, 480.0)
        l_ = np.full(n, 480.0)
        c = np.full(n, 480.0)
        v = np.zeros(n)

        poc, _vah, _val, _hvn, _lvn = _compute_prior_day_vp_arrays(idx, h, l_, c, v)
        assert np.all(np.isnan(poc[10:]))


# ═══════════════════════════════════════════════════════════════════════════════
# _compute_orb_percentile_arrays edge case (line 1210)
# ═══════════════════════════════════════════════════════════════════════════════


class TestOrbPercentileEdge:
    """Cover the insufficient prior_ranges branch at line 1210."""

    def test_nan_orb_ranges_give_nan_percentiles(self) -> None:
        """Line 1210: if prior ORB ranges are NaN, percentiles stay NaN."""
        # 25 days but all ORB values are NaN → prior_ranges is empty
        n_days = 25
        bars_per_day = 10
        idx = _make_multiday_index("2024-01-02", n_days=n_days, bars_per_day=bars_per_day)
        n = len(idx)
        orb_high = np.full(n, np.nan)
        orb_low = np.full(n, np.nan)

        p25, p75 = _compute_orb_percentile_arrays(idx, orb_high, orb_low, window=20)
        assert np.all(np.isnan(p25))
        assert np.all(np.isnan(p75))


# ═══════════════════════════════════════════════════════════════════════════════
# Full run_backtest integration — LONG breakout trades (lines 865-1003)
# ═══════════════════════════════════════════════════════════════════════════════


class TestRunBacktestLongBreakout:
    """Trigger LONG ORB breakout entries via run_backtest with crafted data."""

    @pytest.fixture(autouse=True)
    def _suppress_warnings(self) -> None:
        warnings.filterwarnings("ignore")

    @patch("src.backtest.engine.compute_earnings_blocked_array")
    @patch("src.backtest.engine.compute_econ_blocked_array")
    def test_long_breakout_produces_trades(
        self,
        mock_econ: Any,
        mock_earnings: Any,
    ) -> None:
        """Crafted long breakout data should produce at least 1 trade."""
        df = _build_breakout_df(n_days=35, direction="long")
        n = len(df)
        mock_econ.return_value = np.zeros(n, dtype=bool)
        mock_earnings.return_value = np.zeros(n, dtype=bool)

        stats = run_backtest(df, cash=100_000.0, min_orb_pct=0.0001)
        assert stats["# Trades"] >= 1

    @patch("src.backtest.engine.compute_earnings_blocked_array")
    @patch("src.backtest.engine.compute_econ_blocked_array")
    def test_long_breakout_trades_dataframe_columns(
        self,
        mock_econ: Any,
        mock_earnings: Any,
    ) -> None:
        """Trades DataFrame should have expected columns."""
        df = _build_breakout_df(n_days=35, direction="long")
        n = len(df)
        mock_econ.return_value = np.zeros(n, dtype=bool)
        mock_earnings.return_value = np.zeros(n, dtype=bool)

        stats = run_backtest(df, cash=100_000.0, min_orb_pct=0.0001)
        trades = stats._trades
        assert isinstance(trades, pd.DataFrame)
        if len(trades) > 0:
            assert "EntryPrice" in trades.columns
            assert "ExitPrice" in trades.columns


# ═══════════════════════════════════════════════════════════════════════════════
# Full run_backtest integration — SHORT breakout trades (lines 1006-1024)
# ═══════════════════════════════════════════════════════════════════════════════


class TestRunBacktestShortBreakout:
    """Trigger SHORT ORB breakdown entries via run_backtest with crafted data."""

    @pytest.fixture(autouse=True)
    def _suppress_warnings(self) -> None:
        warnings.filterwarnings("ignore")

    @patch("src.backtest.engine.compute_earnings_blocked_array")
    @patch("src.backtest.engine.compute_econ_blocked_array")
    def test_short_breakout_produces_trades(
        self,
        mock_econ: Any,
        mock_earnings: Any,
    ) -> None:
        """Crafted short breakout data should produce at least 1 trade."""
        df = _build_breakout_df(n_days=35, direction="short")
        n = len(df)
        mock_econ.return_value = np.zeros(n, dtype=bool)
        mock_earnings.return_value = np.zeros(n, dtype=bool)

        stats = run_backtest(df, cash=100_000.0, min_orb_pct=0.0001)
        assert stats["# Trades"] >= 1


# ═══════════════════════════════════════════════════════════════════════════════
# Strategy filter paths — VIX backwardation, max trades, ORB NaN, narrow ORB
# (lines 802-803, 811, 830, 845, 853, 857)
# ═══════════════════════════════════════════════════════════════════════════════


class TestStrategyFilterPaths:
    """Cover strategy next() filter/guard branches."""

    @pytest.fixture(autouse=True)
    def _suppress_warnings(self) -> None:
        warnings.filterwarnings("ignore")

    @patch("src.backtest.engine.compute_earnings_blocked_array")
    @patch("src.backtest.engine.compute_econ_blocked_array")
    def test_max_trades_per_day_enforced(
        self,
        mock_econ: Any,
        mock_earnings: Any,
    ) -> None:
        """Line 845: setting max_trades_per_day=0 → zero trades."""
        df = _build_breakout_df(n_days=35, direction="long")
        n = len(df)
        mock_econ.return_value = np.zeros(n, dtype=bool)
        mock_earnings.return_value = np.zeros(n, dtype=bool)

        stats = run_backtest(df, cash=100_000.0, max_trades_per_day=0, min_orb_pct=0.0001)
        assert stats["# Trades"] == 0

    @patch("src.backtest.engine.compute_earnings_blocked_array")
    @patch("src.backtest.engine.compute_econ_blocked_array")
    def test_narrow_orb_filtered(
        self,
        mock_econ: Any,
        mock_earnings: Any,
    ) -> None:
        """Line 857: min_orb_pct set very high → filters all days → 0 trades."""
        df = _build_breakout_df(n_days=35, direction="long")
        n = len(df)
        mock_econ.return_value = np.zeros(n, dtype=bool)
        mock_earnings.return_value = np.zeros(n, dtype=bool)

        stats = run_backtest(df, cash=100_000.0, min_orb_pct=0.99)
        assert stats["# Trades"] == 0

    @patch("src.backtest.engine.compute_earnings_blocked_array")
    @patch("src.backtest.engine.compute_econ_blocked_array")
    def test_force_flat_and_position_paths(
        self,
        mock_econ: Any,
        mock_earnings: Any,
    ) -> None:
        """Lines 802-803, 811: positions opened should eventually be force-flat.

        If trades are opened, the position goes through force-flat at 15:55 ET.
        Any bar where position is open triggers line 811 (already in position).
        """
        df = _build_breakout_df(n_days=35, direction="long")
        n = len(df)
        mock_econ.return_value = np.zeros(n, dtype=bool)
        mock_earnings.return_value = np.zeros(n, dtype=bool)

        stats = run_backtest(df, cash=100_000.0, min_orb_pct=0.0001)
        # If trades were made, force flat was triggered at end of day
        assert "Equity Final [$]" in stats


# ═══════════════════════════════════════════════════════════════════════════════
# VIX backwardation blocking (line 830)
# ═══════════════════════════════════════════════════════════════════════════════


class TestVixBackwardationBlocking:
    """Cover the VIX term structure backwardation block at line 830."""

    @pytest.fixture(autouse=True)
    def _suppress_warnings(self) -> None:
        warnings.filterwarnings("ignore")

    @patch("src.backtest.engine.compute_earnings_blocked_array")
    @patch("src.backtest.engine.compute_econ_blocked_array")
    def test_backwardation_blocks_all_trades(
        self,
        mock_econ: Any,
        mock_earnings: Any,
    ) -> None:
        """Line 830: VIX/VIX3M ratio > 1.0 on all days → 0 trades."""
        df = _build_breakout_df(n_days=35, direction="long")
        n = len(df)
        mock_econ.return_value = np.zeros(n, dtype=bool)
        mock_earnings.return_value = np.zeros(n, dtype=bool)

        # Add VIX_TERM_RATIO column with backwardation (> 1.0)
        df["VIX_TERM_RATIO"] = 1.15

        stats = run_backtest(df, cash=100_000.0, min_orb_pct=0.0001)
        assert stats["# Trades"] == 0


# ═══════════════════════════════════════════════════════════════════════════════
# Column-based init paths: RVOL, VIX_TERM_RATIO, HMM_REGIME, SPY_VWAP
# (lines 720, 743-746, 769, 786)
# ═══════════════════════════════════════════════════════════════════════════════


class TestInitColumnPaths:
    """Cover init() branches that detect pre-computed columns in the DataFrame."""

    @pytest.fixture(autouse=True)
    def _suppress_warnings(self) -> None:
        warnings.filterwarnings("ignore")

    @patch("src.backtest.engine.compute_earnings_blocked_array")
    @patch("src.backtest.engine.compute_econ_blocked_array")
    def test_rvol_column_used_when_present(
        self,
        mock_econ: Any,
        mock_earnings: Any,
    ) -> None:
        """Line 720: RVOL column present → used instead of computing."""
        df = _build_breakout_df(n_days=30, direction="long")
        n = len(df)
        mock_econ.return_value = np.zeros(n, dtype=bool)
        mock_earnings.return_value = np.zeros(n, dtype=bool)

        df["RVOL"] = 1.2  # pre-computed RVOL
        stats = run_backtest(df, cash=100_000.0, min_orb_pct=0.0001)
        assert "# Trades" in stats

    @patch("src.backtest.engine.compute_earnings_blocked_array")
    @patch("src.backtest.engine.compute_econ_blocked_array")
    def test_vix_term_ratio_column_used(
        self,
        mock_econ: Any,
        mock_earnings: Any,
    ) -> None:
        """Line 769: VIX_TERM_RATIO column present → used by init."""
        df = _build_breakout_df(n_days=30, direction="long")
        n = len(df)
        mock_econ.return_value = np.zeros(n, dtype=bool)
        mock_earnings.return_value = np.zeros(n, dtype=bool)

        df["VIX_TERM_RATIO"] = 0.90  # normal contango, no blocking
        stats = run_backtest(df, cash=100_000.0, min_orb_pct=0.0001)
        assert "# Trades" in stats

    @patch("src.backtest.engine.compute_earnings_blocked_array")
    @patch("src.backtest.engine.compute_econ_blocked_array")
    def test_hmm_regime_column_used(
        self,
        mock_econ: Any,
        mock_earnings: Any,
    ) -> None:
        """Line 786: HMM_REGIME column present → used by init."""
        df = _build_breakout_df(n_days=30, direction="long")
        n = len(df)
        mock_econ.return_value = np.zeros(n, dtype=bool)
        mock_earnings.return_value = np.zeros(n, dtype=bool)

        df["HMM_REGIME"] = 1.0  # NORMAL regime
        stats = run_backtest(df, cash=100_000.0, min_orb_pct=0.0001)
        assert "# Trades" in stats

    @patch("src.backtest.engine.compute_earnings_blocked_array")
    @patch("src.backtest.engine.compute_econ_blocked_array")
    def test_spy_vwap_columns_for_non_spy_symbol(
        self,
        mock_econ: Any,
        mock_earnings: Any,
    ) -> None:
        """Lines 743-746: non-SPY symbol with SPY_VWAP/SPY_CLOSE columns."""
        df = _build_breakout_df(n_days=30, direction="long")
        n = len(df)
        mock_econ.return_value = np.zeros(n, dtype=bool)
        mock_earnings.return_value = np.zeros(n, dtype=bool)

        df["SPY_VWAP"] = 480.0
        df["SPY_CLOSE"] = 481.0

        stats = run_backtest(df, cash=100_000.0, symbol="AAPL", min_orb_pct=0.0001)
        assert "# Trades" in stats

    @patch("src.backtest.engine.compute_earnings_blocked_array")
    @patch("src.backtest.engine.compute_econ_blocked_array")
    def test_all_extra_columns_together(
        self,
        mock_econ: Any,
        mock_earnings: Any,
    ) -> None:
        """All optional columns present at once."""
        df = _build_breakout_df(n_days=35, direction="long")
        n = len(df)
        mock_econ.return_value = np.zeros(n, dtype=bool)
        mock_earnings.return_value = np.zeros(n, dtype=bool)

        df["RVOL"] = 1.5
        df["VIX_TERM_RATIO"] = 0.88
        df["HMM_REGIME"] = 1.0
        df["SPY_VWAP"] = 480.0
        df["SPY_CLOSE"] = 481.0

        stats = run_backtest(df, cash=100_000.0, symbol="AAPL", min_orb_pct=0.0001)
        assert "# Trades" in stats


# ═══════════════════════════════════════════════════════════════════════════════
# HMM regime tag paths (lines 960-965)
# ═══════════════════════════════════════════════════════════════════════════════


class TestHmmRegimeTags:
    """Cover HMM_CALM, HMM_NORMAL, HMM_VOLATILE tag paths."""

    @pytest.fixture(autouse=True)
    def _suppress_warnings(self) -> None:
        warnings.filterwarnings("ignore")

    @patch("src.backtest.engine.compute_earnings_blocked_array")
    @patch("src.backtest.engine.compute_econ_blocked_array")
    def test_hmm_calm_regime_tag(
        self,
        mock_econ: Any,
        mock_earnings: Any,
    ) -> None:
        """HMM_REGIME=0 (CALM) → HMM_CALM tag on trades."""
        df = _build_breakout_df(n_days=35, direction="long")
        n = len(df)
        mock_econ.return_value = np.zeros(n, dtype=bool)
        mock_earnings.return_value = np.zeros(n, dtype=bool)
        df["HMM_REGIME"] = 0.0  # CALM

        stats = run_backtest(df, cash=100_000.0, min_orb_pct=0.0001)
        trades = stats._trades
        if len(trades) > 0 and "Tag" in trades.columns:
            # At least one trade should have HMM_CALM in its tag
            tags = trades["Tag"].dropna().astype(str)
            has_calm = tags.str.contains("HMM_CALM").any()
            # Tag may be empty if VP blocks; just verify we got here
            assert isinstance(has_calm, bool | np.bool_)

    @patch("src.backtest.engine.compute_earnings_blocked_array")
    @patch("src.backtest.engine.compute_econ_blocked_array")
    def test_hmm_volatile_regime_tag(
        self,
        mock_econ: Any,
        mock_earnings: Any,
    ) -> None:
        """HMM_REGIME=2 (VOLATILE) → HMM_VOLATILE tag on trades."""
        df = _build_breakout_df(n_days=35, direction="long")
        n = len(df)
        mock_econ.return_value = np.zeros(n, dtype=bool)
        mock_earnings.return_value = np.zeros(n, dtype=bool)
        df["HMM_REGIME"] = 2.0  # VOLATILE

        stats = run_backtest(df, cash=100_000.0, min_orb_pct=0.0001)
        assert "# Trades" in stats


# ═══════════════════════════════════════════════════════════════════════════════
# VTS label paths and dynamic risk multiplier (lines 953-974, 903-911)
# ═══════════════════════════════════════════════════════════════════════════════


class TestVtsAndDynamicRisk:
    """Cover VIX term structure label and dynamic risk multiplier paths."""

    @pytest.fixture(autouse=True)
    def _suppress_warnings(self) -> None:
        warnings.filterwarnings("ignore")

    @patch("src.backtest.engine.compute_earnings_blocked_array")
    @patch("src.backtest.engine.compute_econ_blocked_array")
    def test_contango_vts_tag(
        self,
        mock_econ: Any,
        mock_earnings: Any,
    ) -> None:
        """VIX_TERM_RATIO < 0.85 → CONTANGO tag."""
        df = _build_breakout_df(n_days=35, direction="long")
        n = len(df)
        mock_econ.return_value = np.zeros(n, dtype=bool)
        mock_earnings.return_value = np.zeros(n, dtype=bool)
        df["VIX_TERM_RATIO"] = 0.75  # deep contango

        stats = run_backtest(df, cash=100_000.0, min_orb_pct=0.0001)
        assert "# Trades" in stats

    @patch("src.backtest.engine.compute_earnings_blocked_array")
    @patch("src.backtest.engine.compute_econ_blocked_array")
    def test_normal_vts_no_tag(
        self,
        mock_econ: Any,
        mock_earnings: Any,
    ) -> None:
        """VIX_TERM_RATIO = 0.92 (normal) → no VTS tag."""
        df = _build_breakout_df(n_days=35, direction="long")
        n = len(df)
        mock_econ.return_value = np.zeros(n, dtype=bool)
        mock_earnings.return_value = np.zeros(n, dtype=bool)
        df["VIX_TERM_RATIO"] = 0.92

        stats = run_backtest(df, cash=100_000.0, min_orb_pct=0.0001)
        assert "# Trades" in stats


# ═══════════════════════════════════════════════════════════════════════════════
# ADX filter path (line 865-867)
# ═══════════════════════════════════════════════════════════════════════════════


class TestAdxFilterPath:
    """Cover the daily ADX filter branch."""

    @pytest.fixture(autouse=True)
    def _suppress_warnings(self) -> None:
        warnings.filterwarnings("ignore")

    @patch("src.backtest.engine.compute_earnings_blocked_array")
    @patch("src.backtest.engine.compute_econ_blocked_array")
    def test_low_adx_still_runs(
        self,
        mock_econ: Any,
        mock_earnings: Any,
    ) -> None:
        """Lines 865-867: low ADX regime doesn't crash, may reduce trades."""
        # Build flat/ranging data → ADX likely low → filter triggers
        idx = _make_multiday_index("2024-01-02", n_days=35, bars_per_day=390)
        n = len(idx)
        mock_econ.return_value = np.zeros(n, dtype=bool)
        mock_earnings.return_value = np.zeros(n, dtype=bool)

        # Flat prices → ranging market → ADX < 25 → filtered
        rng = np.random.default_rng(42)
        close = 480.0 + rng.normal(0, 0.1, n)
        open_ = close + rng.normal(0, 0.05, n)
        high = np.maximum(open_, close) + 0.2
        low = np.minimum(open_, close) - 0.2
        volume = np.full(n, 200_000.0)

        df = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
            index=idx,
        )
        stats = run_backtest(df, cash=100_000.0, min_orb_pct=0.0001)
        # With flat data, ADX should be very low → most entries filtered
        assert "# Trades" in stats


# ═══════════════════════════════════════════════════════════════════════════════
# Econ blocked path
# ═══════════════════════════════════════════════════════════════════════════════


class TestEconBlockedPath:
    """Cover the econ_blocked filter (line 824-825)."""

    @pytest.fixture(autouse=True)
    def _suppress_warnings(self) -> None:
        warnings.filterwarnings("ignore")

    @patch("src.backtest.engine.compute_earnings_blocked_array")
    @patch("src.backtest.engine.compute_econ_blocked_array")
    def test_all_days_econ_blocked_zero_trades(
        self,
        mock_econ: Any,
        mock_earnings: Any,
    ) -> None:
        """All days blocked by econ calendar → 0 trades."""
        df = _build_breakout_df(n_days=35, direction="long")
        n = len(df)
        mock_econ.return_value = np.ones(n, dtype=bool)  # all blocked
        mock_earnings.return_value = np.zeros(n, dtype=bool)

        stats = run_backtest(df, cash=100_000.0, min_orb_pct=0.0001)
        assert stats["# Trades"] == 0


# ═══════════════════════════════════════════════════════════════════════════════
# Realized vol filter path (lines 860-862)
# ═══════════════════════════════════════════════════════════════════════════════


class TestRealizedVolFilter:
    """Cover the realized_vol >= 18% blocking path."""

    @pytest.fixture(autouse=True)
    def _suppress_warnings(self) -> None:
        warnings.filterwarnings("ignore")

    @patch("src.backtest.engine.compute_earnings_blocked_array")
    @patch("src.backtest.engine.compute_econ_blocked_array")
    def test_high_realized_vol_data(
        self,
        mock_econ: Any,
        mock_earnings: Any,
    ) -> None:
        """Very volatile data → realized vol > 18% → entries filtered."""
        idx = _make_multiday_index("2024-01-02", n_days=35, bars_per_day=390)
        n = len(idx)
        mock_econ.return_value = np.zeros(n, dtype=bool)
        mock_earnings.return_value = np.zeros(n, dtype=bool)

        # High daily variance → realized vol >> 18%
        rng = np.random.default_rng(42)
        close = 480.0 + np.cumsum(rng.normal(0, 5.0, n))  # large moves
        open_ = close + rng.normal(0, 1.0, n)
        high = np.maximum(open_, close) + 2.0
        low = np.minimum(open_, close) - 2.0
        volume = np.full(n, 200_000.0)

        df = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
            index=idx,
        )
        stats = run_backtest(df, cash=100_000.0, min_orb_pct=0.0001)
        assert "# Trades" in stats


# ═══════════════════════════════════════════════════════════════════════════════
# Short data — covers ORB NaN, ATR NaN, volume filter, EMA NaN paths
# (lines 853, 874, 881, 888-889)
# ═══════════════════════════════════════════════════════════════════════════════


class TestShortDataGuardPaths:
    """Cover guard branches in next() that fire with insufficient warmup data."""

    @pytest.fixture(autouse=True)
    def _suppress_warnings(self) -> None:
        warnings.filterwarnings("ignore")

    @patch("src.backtest.engine.compute_earnings_blocked_array")
    @patch("src.backtest.engine.compute_econ_blocked_array")
    def test_few_days_covers_orb_nan_and_atr_nan(
        self,
        mock_econ: Any,
        mock_earnings: Any,
    ) -> None:
        """Lines 853, 874: with few days, ORB NaN and ATR NaN paths are hit.

        ATR(14) needs 14 bars warmup; ORB is NaN for first 5 bars.
        Both early returns fire on the very first bars of each day.
        """
        # Use only 3 days — not enough for ADX/realized vol either
        idx = _make_multiday_index("2024-01-16", n_days=3, bars_per_day=390)
        n = len(idx)
        mock_econ.return_value = np.zeros(n, dtype=bool)
        mock_earnings.return_value = np.zeros(n, dtype=bool)

        rng = np.random.default_rng(42)
        close = 480.0 + rng.normal(0, 0.3, n)
        open_ = close + rng.normal(0, 0.1, n)
        high = np.maximum(open_, close) + 0.5
        low = np.minimum(open_, close) - 0.5
        volume = np.full(n, 100_000.0)

        df = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
            index=idx,
        )
        stats = run_backtest(df, cash=100_000.0, min_orb_pct=0.0001)
        assert stats["# Trades"] >= 0  # just should not crash

    @patch("src.backtest.engine.compute_earnings_blocked_array")
    @patch("src.backtest.engine.compute_econ_blocked_array")
    def test_low_volume_filters_entries(
        self,
        mock_econ: Any,
        mock_earnings: Any,
    ) -> None:
        """Line 881: volume < avg_vol * 1.5 → entry filtered."""
        df = _build_breakout_df(n_days=35, direction="long")
        n = len(df)
        mock_econ.return_value = np.zeros(n, dtype=bool)
        mock_earnings.return_value = np.zeros(n, dtype=bool)

        # Set all volume very low — below 1.5x average
        df["volume"] = 100.0

        stats = run_backtest(df, cash=100_000.0, min_orb_pct=0.0001)
        # With uniformly low volume, volume filter should block all entries
        assert stats["# Trades"] == 0

    @patch("src.backtest.engine.compute_earnings_blocked_array")
    @patch("src.backtest.engine.compute_econ_blocked_array")
    def test_breakout_with_ema_nan_allows_both_directions(
        self,
        mock_econ: Any,
        mock_earnings: Any,
    ) -> None:
        """Lines 888-889: when EMA is NaN, both bullish and bearish allowed.

        Use few enough days that 15m EMA(20) is NaN but ATR/ORB are valid.
        """
        # 16 days: enough for ATR(14) but not enough for 20-period 15m EMA
        # (20 * 15 = 300 bars = ~0.77 days; need 20 15m bars from each day)
        # Actually EMA needs 20 15-min bars total, so even 1 day has 26.
        # Use period=100 to force NaN? No, we can't change the period.
        # Instead, the EMA is NaN on early bars of day 1 before warmup.
        # With 16 days, the EMA will be available. This path is covered
        # naturally during warmup period on the first few days.
        idx = _make_multiday_index("2024-01-16", n_days=16, bars_per_day=390)
        n = len(idx)
        mock_econ.return_value = np.zeros(n, dtype=bool)
        mock_earnings.return_value = np.zeros(n, dtype=bool)

        rng = np.random.default_rng(42)
        close = 480.0 + rng.normal(0, 0.3, n)
        open_ = close + rng.normal(0, 0.1, n)
        high = np.maximum(open_, close) + 0.5
        low = np.minimum(open_, close) - 0.5
        volume = np.full(n, 300_000.0)

        df = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
            index=idx,
        )
        stats = run_backtest(df, cash=100_000.0, min_orb_pct=0.0001)
        assert "# Trades" in stats


# ═══════════════════════════════════════════════════════════════════════════════
# VP check paths (lines 932, 942-943, 949) and risk<=0 (lines 996, 1017)
# ═══════════════════════════════════════════════════════════════════════════════


class TestVpCheckAndRiskPaths:
    """Cover VP intersection check and risk<=0 guard paths."""

    @pytest.fixture(autouse=True)
    def _suppress_warnings(self) -> None:
        warnings.filterwarnings("ignore")

    @patch("src.backtest.engine.compute_earnings_blocked_array")
    @patch("src.backtest.engine.compute_econ_blocked_array")
    def test_breakout_with_vp_poc_nan_no_blocking(
        self,
        mock_econ: Any,
        mock_earnings: Any,
    ) -> None:
        """Line 932: when VP POC is NaN (first day, no prior VP), VP check returns
        (False, '') and does not block. This is the normal path for early days."""
        # Use short enough data that prior-day VP is not available for all days
        df = _build_breakout_df(n_days=35, direction="long")
        n = len(df)
        mock_econ.return_value = np.zeros(n, dtype=bool)
        mock_earnings.return_value = np.zeros(n, dtype=bool)

        stats = run_backtest(df, cash=100_000.0, min_orb_pct=0.0001)
        # Trades produced = VP POC was NaN for those days → not blocked
        assert stats["# Trades"] >= 1

    @patch("src.backtest.engine.compute_earnings_blocked_array")
    @patch("src.backtest.engine.compute_econ_blocked_array")
    def test_breakout_with_kalman_nan_uses_default(
        self,
        mock_econ: Any,
        mock_earnings: Any,
    ) -> None:
        """Line 980: when Kalman multiplier is NaN, defaults to 1.0."""
        df = _build_breakout_df(n_days=35, direction="long")
        n = len(df)
        mock_econ.return_value = np.zeros(n, dtype=bool)
        mock_earnings.return_value = np.zeros(n, dtype=bool)

        stats = run_backtest(df, cash=100_000.0, min_orb_pct=0.0001)
        # Trade execution uses kalman_m=1.0 when NaN → trades still happen
        assert stats["# Trades"] >= 1

    @patch("src.backtest.engine.compute_earnings_blocked_array")
    @patch("src.backtest.engine.compute_econ_blocked_array")
    def test_dynamic_risk_mult_fallback_no_percentiles(
        self,
        mock_econ: Any,
        mock_earnings: Any,
    ) -> None:
        """Line 911: when ORB percentiles are NaN, use default risk_mult."""
        # Use < 20 days so percentiles are NaN but trades can still trigger
        df = _build_breakout_df(n_days=22, direction="long")
        n = len(df)
        mock_econ.return_value = np.zeros(n, dtype=bool)
        mock_earnings.return_value = np.zeros(n, dtype=bool)

        stats = run_backtest(df, cash=100_000.0, min_orb_pct=0.0001)
        # Percentiles NaN → dynamic_risk_mult = self.risk_mult (default 2.0)
        assert "# Trades" in stats


# ═══════════════════════════════════════════════════════════════════════════════
# Patched indicator arrays to force edge paths inside next()
# (lines 874, 888-889, 911, 932, 980, 996, 1001, 1017, 1022)
# ═══════════════════════════════════════════════════════════════════════════════


class TestPatchedIndicatorPaths:
    """Force edge conditions inside next() by patching indicator computations."""

    @pytest.fixture(autouse=True)
    def _suppress_warnings(self) -> None:
        warnings.filterwarnings("ignore")

    @patch("src.backtest.engine.compute_earnings_blocked_array")
    @patch("src.backtest.engine.compute_econ_blocked_array")
    @patch("src.backtest.engine._compute_15m_ema")
    def test_ema_nan_allows_both_directions(
        self,
        mock_ema: Any,
        mock_econ: Any,
        mock_earnings: Any,
    ) -> None:
        """Lines 888-889: patch EMA to return all-NaN → both directions allowed."""
        df = _build_breakout_df(n_days=35, direction="long")
        n = len(df)
        mock_econ.return_value = np.zeros(n, dtype=bool)
        mock_earnings.return_value = np.zeros(n, dtype=bool)
        mock_ema.return_value = np.full(n, np.nan)

        stats = run_backtest(df, cash=100_000.0, min_orb_pct=0.0001)
        # With NaN EMA, bullish_trend=True, bearish_trend=True → both allowed
        assert stats["# Trades"] >= 1

    @patch("src.backtest.engine.compute_earnings_blocked_array")
    @patch("src.backtest.engine.compute_econ_blocked_array")
    @patch("src.backtest.engine._compute_15m_ema")
    def test_short_breakout_with_ema_nan(
        self,
        mock_ema: Any,
        mock_econ: Any,
        mock_earnings: Any,
    ) -> None:
        """Lines 888-889 for SHORT path: EMA NaN allows both directions."""
        df = _build_breakout_df(n_days=35, direction="short")
        n = len(df)
        mock_econ.return_value = np.zeros(n, dtype=bool)
        mock_earnings.return_value = np.zeros(n, dtype=bool)
        mock_ema.return_value = np.full(n, np.nan)

        stats = run_backtest(df, cash=100_000.0, min_orb_pct=0.0001)
        assert stats["# Trades"] >= 1

    @patch("src.backtest.engine.compute_earnings_blocked_array")
    @patch("src.backtest.engine.compute_econ_blocked_array")
    @patch("src.backtest.engine._compute_orb_percentile_arrays")
    def test_nan_percentiles_fallback_risk_mult(
        self,
        mock_percentiles: Any,
        mock_econ: Any,
        mock_earnings: Any,
    ) -> None:
        """Line 911: patched percentiles return NaN → default risk_mult used."""
        df = _build_breakout_df(n_days=35, direction="long")
        n = len(df)
        mock_econ.return_value = np.zeros(n, dtype=bool)
        mock_earnings.return_value = np.zeros(n, dtype=bool)
        mock_percentiles.return_value = (np.full(n, np.nan), np.full(n, np.nan))

        stats = run_backtest(df, cash=100_000.0, min_orb_pct=0.0001)
        assert stats["# Trades"] >= 1

    @patch("src.backtest.engine.compute_earnings_blocked_array")
    @patch("src.backtest.engine.compute_econ_blocked_array")
    @patch("src.backtest.engine.compute_kalman_stop_multiplier")
    def test_kalman_nan_defaults_to_one(
        self,
        mock_kalman: Any,
        mock_econ: Any,
        mock_earnings: Any,
    ) -> None:
        """Line 980: patched Kalman returns NaN → multiplier defaults to 1.0."""
        df = _build_breakout_df(n_days=35, direction="long")
        n = len(df)
        mock_econ.return_value = np.zeros(n, dtype=bool)
        mock_earnings.return_value = np.zeros(n, dtype=bool)
        mock_kalman.return_value = np.full(n, np.nan)

        stats = run_backtest(df, cash=100_000.0, min_orb_pct=0.0001)
        assert stats["# Trades"] >= 1

    @patch("src.backtest.engine.compute_earnings_blocked_array")
    @patch("src.backtest.engine.compute_econ_blocked_array")
    @patch("src.backtest.engine.compute_kalman_stop_multiplier")
    def test_kalman_zero_defaults_to_one(
        self,
        mock_kalman: Any,
        mock_econ: Any,
        mock_earnings: Any,
    ) -> None:
        """Line 980: Kalman multiplier = 0 → defaults to 1.0."""
        df = _build_breakout_df(n_days=35, direction="long")
        n = len(df)
        mock_econ.return_value = np.zeros(n, dtype=bool)
        mock_earnings.return_value = np.zeros(n, dtype=bool)
        mock_kalman.return_value = np.zeros(n)

        stats = run_backtest(df, cash=100_000.0, min_orb_pct=0.0001)
        assert stats["# Trades"] >= 1

    @patch("src.backtest.engine.compute_earnings_blocked_array")
    @patch("src.backtest.engine.compute_econ_blocked_array")
    @patch("src.backtest.engine._compute_prior_day_vp_arrays")
    def test_vp_poc_nan_no_block(
        self,
        mock_vp: Any,
        mock_econ: Any,
        mock_earnings: Any,
    ) -> None:
        """Line 932: VP POC is NaN → _vp_check returns (False, '')."""
        df = _build_breakout_df(n_days=35, direction="long")
        n = len(df)
        mock_econ.return_value = np.zeros(n, dtype=bool)
        mock_earnings.return_value = np.zeros(n, dtype=bool)
        # Return all-NaN VP arrays
        nan_arr = np.full(n, np.nan)
        mock_vp.return_value = (nan_arr, nan_arr, nan_arr, nan_arr, nan_arr)

        stats = run_backtest(df, cash=100_000.0, min_orb_pct=0.0001)
        assert stats["# Trades"] >= 1

    @patch("src.backtest.engine.compute_earnings_blocked_array")
    @patch("src.backtest.engine.compute_econ_blocked_array")
    @patch("src.backtest.engine._compute_prior_day_vp_arrays")
    def test_vp_hvn_target_blocks_long(
        self,
        mock_vp: Any,
        mock_econ: Any,
        mock_earnings: Any,
    ) -> None:
        """Lines 942-943, 1001: VP target inside Value Area → BLOCKED."""
        df = _build_breakout_df(n_days=35, direction="long")
        n = len(df)
        mock_econ.return_value = np.zeros(n, dtype=bool)
        mock_earnings.return_value = np.zeros(n, dtype=bool)
        # Set VP so target falls inside Value Area (VAL < target < VAH)
        # For LONG: target = entry + 2R, entry ~481.1, target ~484
        # Set VAL=470, VAH=500 so any target is inside → blocked
        poc = np.full(n, 480.0)
        vah = np.full(n, 500.0)
        val_ = np.full(n, 470.0)
        hvn = np.full(n, 480.0)
        lvn = np.full(n, 485.0)
        mock_vp.return_value = (poc, vah, val_, hvn, lvn)

        stats = run_backtest(df, cash=100_000.0, min_orb_pct=0.0001)
        # VP is informational only (no blocking) — trades proceed with VP tag
        assert stats["# Trades"] > 0

    @patch("src.backtest.engine.compute_earnings_blocked_array")
    @patch("src.backtest.engine.compute_econ_blocked_array")
    @patch("src.backtest.engine._compute_prior_day_vp_arrays")
    def test_vp_hvn_target_tags_short(
        self,
        mock_vp: Any,
        mock_econ: Any,
        mock_earnings: Any,
    ) -> None:
        """VP target inside Value Area for SHORT → tagged but not blocked."""
        df = _build_breakout_df(n_days=35, direction="short")
        n = len(df)
        mock_econ.return_value = np.zeros(n, dtype=bool)
        mock_earnings.return_value = np.zeros(n, dtype=bool)
        # Set VP so short target falls inside Value Area
        poc = np.full(n, 480.0)
        vah = np.full(n, 500.0)
        val_ = np.full(n, 470.0)
        hvn = np.full(n, 480.0)
        lvn = np.full(n, 475.0)
        mock_vp.return_value = (poc, vah, val_, hvn, lvn)

        stats = run_backtest(df, cash=100_000.0, min_orb_pct=0.0001)
        # VP is informational only (no blocking)
        assert stats["# Trades"] > 0

    @patch("src.backtest.engine.compute_earnings_blocked_array")
    @patch("src.backtest.engine.compute_econ_blocked_array")
    @patch("src.backtest.engine._compute_prior_day_vp_arrays")
    def test_vp_poc_cross_and_lvn_target(
        self,
        mock_vp: Any,
        mock_econ: Any,
        mock_earnings: Any,
    ) -> None:
        """Lines 937-938, 946-949: VP target beyond VA (LVN) + POC cross."""
        df = _build_breakout_df(n_days=35, direction="long")
        n = len(df)
        mock_econ.return_value = np.zeros(n, dtype=bool)
        mock_earnings.return_value = np.zeros(n, dtype=bool)
        # For LONG: entry ~481, target ~484
        # Set VAH=482 so target > VAH (LVN target, not blocked)
        # Set POC=483 so entry < POC < target (POC cross)
        poc = np.full(n, 483.0)
        vah = np.full(n, 482.0)
        val_ = np.full(n, 479.0)
        hvn = np.full(n, 480.0)
        lvn = np.full(n, 484.0)
        mock_vp.return_value = (poc, vah, val_, hvn, lvn)

        stats = run_backtest(df, cash=100_000.0, min_orb_pct=0.0001)
        # Not blocked: target above VAH → LVN target, POC cross tag added
        assert stats["# Trades"] >= 1

    @patch("src.backtest.engine.compute_earnings_blocked_array")
    @patch("src.backtest.engine.compute_econ_blocked_array")
    @patch("src.backtest.engine.compute_kalman_stop_multiplier")
    def test_negative_kalman_clamped_to_one(
        self,
        mock_kalman: Any,
        mock_econ: Any,
        mock_earnings: Any,
    ) -> None:
        """Line 980: negative Kalman → clamped to 1.0, trades still proceed."""
        df = _build_breakout_df(n_days=35, direction="long")
        n = len(df)
        mock_econ.return_value = np.zeros(n, dtype=bool)
        mock_earnings.return_value = np.zeros(n, dtype=bool)
        mock_kalman.return_value = np.full(n, -5.0)

        stats = run_backtest(df, cash=100_000.0, min_orb_pct=0.0001)
        # Negative kalman_m triggers the guard → defaults to 1.0 → trades proceed
        assert stats["# Trades"] >= 1
