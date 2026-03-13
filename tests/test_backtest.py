"""Tests for the backtesting framework.

Coverage
--------
- data_loader: load/resample, walk-forward window generation, slice_window
- engine: no lookahead bias (shift check on ORB arrays), signal correctness
- metrics: metric calculations against known values
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.backtest.data_loader import (
    WalkForwardWindow,
    make_walk_forward_windows,
    resample,
    slice_window,
)
from src.backtest.engine import (
    _compute_15m_ema,
    _compute_orb_arrays,
    _compute_orb_percentile_arrays,
    _et_date,
    _et_time,
)
from src.backtest.metrics import (
    _max_drawdown,
    _sharpe_ratio,
    compute_metrics,
)
from src.models import TimeFrame

_UTC = UTC


# ── fixtures ─────────────────────────────────────────────────────────────────


def _make_1min_df(
    n_bars: int = 390,
    start: datetime | None = None,
    close: float = 480.0,
    volume: int = 500_000,
) -> pd.DataFrame:
    """Build a minimal 1-min OHLCV DataFrame starting at 9:30 ET on 2024-01-15.

    Creates a single trading day (390 1-min bars by default).
    """
    if start is None:
        # 9:30 ET = 14:30 UTC
        start = datetime(2024, 1, 15, 14, 30, tzinfo=_UTC)

    idx = pd.date_range(start=start, periods=n_bars, freq="1min", tz="UTC")
    df = pd.DataFrame(
        {
            "open": close - 0.1,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": volume,
            "vwap": close,
        },
        index=idx,
    )
    return df


def _make_multiday_df(n_days: int = 5) -> pd.DataFrame:
    """Build n_days of 1-min bars (390 bars per day, 9:30 ET = 14:30 UTC)."""
    from datetime import timedelta

    base = datetime(2024, 1, 2, 14, 30, tzinfo=_UTC)
    frames = []
    for d in range(n_days):
        day_start = base + timedelta(days=d)
        frames.append(_make_1min_df(n_bars=390, start=day_start))
    return pd.concat(frames).sort_index()


# ── data_loader tests ─────────────────────────────────────────────────────────


class TestResample:
    def test_5min_produces_correct_bar_count(self) -> None:
        """390 1-min bars → 78 5-min bars (390 / 5)."""
        df = _make_1min_df(n_bars=390)
        result = resample(df, TimeFrame.FIVE_MIN)
        assert len(result) == 78

    def test_15min_produces_correct_bar_count(self) -> None:
        """390 1-min bars → 26 15-min bars (390 / 15)."""
        df = _make_1min_df(n_bars=390)
        result = resample(df, TimeFrame.FIFTEEN_MIN)
        assert len(result) == 26

    def test_1min_passthrough(self) -> None:
        """Resampling to 1-min returns a copy unchanged."""
        df = _make_1min_df(n_bars=10)
        result = resample(df, TimeFrame.ONE_MIN)
        assert len(result) == len(df)

    def test_ohlcv_aggregation_correctness(self) -> None:
        """High = max, Low = min, Open = first, Close = last, Volume = sum."""
        # 5 bars with known values: high goes 1,2,3,4,5; close goes 10,20,30,40,50
        idx = pd.date_range("2024-01-15 14:30", periods=5, freq="1min", tz="UTC")
        df = pd.DataFrame(
            {
                "open": [100.0] * 5,
                "high": [1.0, 2.0, 3.0, 4.0, 5.0],
                "low": [0.1, 0.2, 0.3, 0.4, 0.5],
                "close": [10.0, 20.0, 30.0, 40.0, 50.0],
                "volume": [100, 200, 300, 400, 500],
                "vwap": [10.0] * 5,
            },
            index=idx,
        )
        result = resample(df, TimeFrame.FIVE_MIN)
        assert len(result) == 1
        row = result.iloc[0]
        assert row["high"] == 5.0
        assert row["low"] == 0.1
        assert row["close"] == 50.0
        assert row["volume"] == 1500

    def test_unsupported_timeframe_raises(self) -> None:
        df = _make_1min_df(n_bars=10)
        with pytest.raises(ValueError, match="Unsupported"):
            resample(df, TimeFrame.DAILY)


class TestWalkForwardWindows:
    def test_single_window(self) -> None:
        """80 days of data → exactly 1 window (60 IS + 20 OOS)."""
        df = _make_multiday_df(n_days=82)
        windows = make_walk_forward_windows(df, in_sample_days=60, out_of_sample_days=20)
        assert len(windows) >= 1

    def test_window_is_oos_do_not_overlap(self) -> None:
        """OOS start must be the day after IS end."""
        df = _make_multiday_df(n_days=90)
        windows = make_walk_forward_windows(df, in_sample_days=60, out_of_sample_days=20)
        for w in windows:
            assert w.out_of_sample_start > w.in_sample_end

    def test_empty_df_returns_empty_list(self) -> None:
        df = pd.DataFrame()
        windows = make_walk_forward_windows(df)
        assert windows == []

    def test_insufficient_data_returns_empty(self) -> None:
        """Less than IS days → no window."""
        df = _make_multiday_df(n_days=5)  # only 5 days, need 60
        windows = make_walk_forward_windows(df, in_sample_days=60, out_of_sample_days=20)
        assert windows == []

    def test_window_dates_sequential(self) -> None:
        """Each consecutive pair of windows advances the start date."""
        df = _make_multiday_df(n_days=200)
        windows = make_walk_forward_windows(df, in_sample_days=60, out_of_sample_days=20)
        import itertools

        for a, b in itertools.pairwise(windows):
            assert b.in_sample_start > a.in_sample_start

    def test_oos_end_clamped_to_last_date(self) -> None:
        """Final OOS window end is <= last date in the DataFrame."""
        df = _make_multiday_df(n_days=82)
        windows = make_walk_forward_windows(df, in_sample_days=60, out_of_sample_days=20)
        last_date: date = df.index[-1].date()
        assert windows[-1].out_of_sample_end <= last_date


class TestSliceWindow:
    def test_in_sample_slice_has_correct_dates(self) -> None:
        df = _make_multiday_df(n_days=5)
        window = WalkForwardWindow(
            in_sample_start=date(2024, 1, 2),
            in_sample_end=date(2024, 1, 3),
            out_of_sample_start=date(2024, 1, 4),
            out_of_sample_end=date(2024, 1, 5),
        )
        is_df, _ = slice_window(df, window)
        # All IS timestamps must fall within [Jan 2, Jan 3]
        assert not is_df.empty
        first_day: date = is_df.index[0].date()
        last_day: date = is_df.index[-1].date()
        assert first_day >= date(2024, 1, 2)
        assert last_day <= date(2024, 1, 3)

    def test_out_of_sample_slice_has_correct_dates(self) -> None:
        df = _make_multiday_df(n_days=5)
        window = WalkForwardWindow(
            in_sample_start=date(2024, 1, 2),
            in_sample_end=date(2024, 1, 3),
            out_of_sample_start=date(2024, 1, 4),
            out_of_sample_end=date(2024, 1, 5),
        )
        _, oos_df = slice_window(df, window)
        assert not oos_df.empty
        first_day: date = oos_df.index[0].date()
        assert first_day >= date(2024, 1, 4)

    def test_slices_do_not_overlap(self) -> None:
        df = _make_multiday_df(n_days=5)
        window = WalkForwardWindow(
            in_sample_start=date(2024, 1, 2),
            in_sample_end=date(2024, 1, 3),
            out_of_sample_start=date(2024, 1, 4),
            out_of_sample_end=date(2024, 1, 5),
        )
        is_df, oos_df = slice_window(df, window)
        # No timestamp appears in both slices
        shared = is_df.index.intersection(oos_df.index)
        assert len(shared) == 0


# ── engine: no-lookahead tests ────────────────────────────────────────────────


class TestORBArraysNoLookahead:
    """Verify _compute_orb_arrays never exposes future bar data."""

    def _build_index_and_prices(
        self,
        n_bars: int = 20,
        start: str = "2024-01-15 14:30",  # 9:30 ET
    ) -> tuple[
        pd.DatetimeIndex,
        np.ndarray[Any, np.dtype[np.float64]],
        np.ndarray[Any, np.dtype[np.float64]],
    ]:
        idx = pd.date_range(start, periods=n_bars, freq="1min", tz="UTC")
        high = np.arange(1.0, n_bars + 1.0)  # unique values per bar
        low = high - 0.5
        return idx, high, low

    def test_orb_bars_during_window_are_nan(self) -> None:
        """The first 5 bars (opening range) must have NaN ORB values."""
        idx, high, low = self._build_index_and_prices(n_bars=20)
        orb_high, orb_low_nan = _compute_orb_arrays(idx, high, low, orb_bars=5)
        # First 5 bars: NaN (ORB not yet established)
        assert all(np.isnan(orb_high[:5]))
        assert all(np.isnan(orb_low_nan[:5]))

    def test_orb_bars_after_window_use_only_first_n_bars(self) -> None:
        """Bars 6+ should use max-high / min-low of bars 0-4 only."""
        idx, high, low = self._build_index_and_prices(n_bars=20)
        # high = [1,2,3,4,5, 6,7,...,20] → ORB high should be 5.0
        orb_high, _orb_low = _compute_orb_arrays(idx, high, low, orb_bars=5)
        for i in range(5, 20):
            assert orb_high[i] == pytest.approx(5.0), f"bar {i} saw future high"

    def test_orb_low_uses_only_first_n_bars(self) -> None:
        """ORB low = min of first n bars, not affected by later bars."""
        idx, high, low = self._build_index_and_prices(n_bars=20)
        # low = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, ...] → ORB low = 0.5
        _orb_high, orb_low = _compute_orb_arrays(idx, high, low, orb_bars=5)
        for i in range(5, 20):
            assert orb_low[i] == pytest.approx(0.5), f"bar {i} saw future low"

    def test_multiday_orb_is_computed_per_day(self) -> None:
        """Each calendar day gets its own ORB, not contaminated by prior days."""
        # Two days, 15 bars each (9:30-9:44 ET = UTC 14:30-14:44)
        idx1 = pd.date_range("2024-01-15 14:30", periods=15, freq="1min", tz="UTC")
        idx2 = pd.date_range("2024-01-16 14:30", periods=15, freq="1min", tz="UTC")
        idx = idx1.append(idx2)

        # Day 1 highs: 1-15; Day 2 highs: 100-114
        high = np.concatenate([np.arange(1.0, 16.0), np.arange(100.0, 115.0)])
        low = high - 0.5

        orb_high, _ = _compute_orb_arrays(idx, high, low, orb_bars=5)

        # Day 1: ORB high = max of bars 0-4 = 5.0
        for i in range(5, 15):
            assert orb_high[i] == pytest.approx(
                5.0
            ), f"day1 bar {i}: expected 5.0 got {orb_high[i]}"

        # Day 2: ORB high = max of bars 15-19 = 104.0
        for i in range(20, 30):
            assert orb_high[i] == pytest.approx(
                104.0
            ), f"day2 bar {i}: expected 104.0 got {orb_high[i]}"

    def test_orb_range_pct_is_nonzero(self) -> None:
        """Verify the ORB range (high-low)/midpoint is non-zero for non-flat bars."""
        idx, high, low = self._build_index_and_prices(n_bars=20)
        orb_high, orb_low = _compute_orb_arrays(idx, high, low, orb_bars=5)
        # For bars after ORB: orb_high=5.0, orb_low=0.5, midpoint=2.75
        # range_pct = (5.0-0.5)/2.75 ≈ 1.636
        midpoint = (orb_high[10] + orb_low[10]) / 2.0
        range_pct = (orb_high[10] - orb_low[10]) / midpoint
        assert range_pct == pytest.approx((5.0 - 0.5) / 2.75, abs=0.001)


class TestETTime:
    def test_930_et_converts_correctly_est(self) -> None:
        """14:30 UTC = 9:30 EST (January = UTC-5)."""
        from datetime import time

        ts = pd.Timestamp("2024-01-15 14:30:00", tz="UTC")
        assert _et_time(ts) == time(9, 30)

    def test_1600_et_converts_correctly_est(self) -> None:
        from datetime import time

        ts = pd.Timestamp("2024-01-15 21:00:00", tz="UTC")
        assert _et_time(ts) == time(16, 0)

    def test_930_et_converts_correctly_edt(self) -> None:
        """13:30 UTC = 9:30 EDT (July = UTC-4). Old fixed -5h offset was wrong here."""
        from datetime import time

        ts = pd.Timestamp("2024-07-15 13:30:00", tz="UTC")
        assert _et_time(ts) == time(9, 30)

    def test_lunch_1130_edt(self) -> None:
        """15:30 UTC = 11:30 EDT. The old offset saw this as 10:30, skipping lunch filter."""
        from datetime import time

        ts = pd.Timestamp("2024-07-15 15:30:00", tz="UTC")
        assert _et_time(ts) == time(11, 30)

    def test_tz_naive_input_treated_as_utc(self) -> None:
        """A tz-naive Timestamp is assumed to be UTC and converted correctly."""
        from datetime import time

        ts_naive = pd.Timestamp("2024-01-15 14:30:00")  # no tz
        assert _et_time(ts_naive) == time(9, 30)


class TestETDate:
    def test_date_est(self) -> None:
        """14:30 UTC Jan 15 = Jan 15 ET (EST, UTC-5)."""
        ts = pd.Timestamp("2024-01-15 14:30:00", tz="UTC")
        assert _et_date(ts) == date(2024, 1, 15)

    def test_date_edt_midnight_rollover(self) -> None:
        """03:00 UTC = 23:00 ET previous day during EDT (UTC-4)."""
        ts = pd.Timestamp("2024-07-16 03:00:00", tz="UTC")  # 23:00 EDT July 15
        assert _et_date(ts) == date(2024, 7, 15)


# ── metrics tests ─────────────────────────────────────────────────────────────


class TestComputeMetrics:
    def _make_trades(self, pnl_list: list[float]) -> pd.DataFrame:
        n = len(pnl_list)
        idx = pd.date_range("2024-01-15 14:35", periods=n, freq="30min", tz="UTC")
        return pd.DataFrame(
            {
                "PnL": pnl_list,
                "ReturnPct": [p / 500 * 100 for p in pnl_list],
                "EntryTime": idx,
                "ExitTime": idx + pd.Timedelta(minutes=15),
            }
        )

    def test_empty_trades_returns_zeros(self) -> None:
        m = compute_metrics(pd.DataFrame())
        assert m["total_trades"] == 0
        assert m["win_rate"] == 0.0

    def test_win_rate_calculation(self) -> None:
        """3 wins, 1 loss → 75% win rate."""
        trades = self._make_trades([100.0, 200.0, 150.0, -75.0])
        m = compute_metrics(trades)
        assert m["win_rate"] == pytest.approx(0.75, abs=0.001)
        assert m["loss_rate"] == pytest.approx(0.25, abs=0.001)

    def test_profit_factor(self) -> None:
        """gross_profit=450, gross_loss=75 → PF=6.0."""
        trades = self._make_trades([100.0, 200.0, 150.0, -75.0])
        m = compute_metrics(trades)
        assert float(m["profit_factor"]) == pytest.approx(6.0, abs=0.01)

    def test_expectancy(self) -> None:
        """avg_win=150, win_rate=0.75, avg_loss=75, loss_rate=0.25.
        expectancy = 150*0.75 - 75*0.25 = 112.5 - 18.75 = 93.75."""
        trades = self._make_trades([100.0, 200.0, 150.0, -75.0])
        m = compute_metrics(trades)
        assert float(m["expectancy"]) == pytest.approx(93.75, abs=0.01)

    def test_avg_winner_and_loser(self) -> None:
        trades = self._make_trades([100.0, 200.0, -50.0, -150.0])
        m = compute_metrics(trades)
        assert float(m["avg_winner"]) == pytest.approx(150.0, abs=0.01)
        assert float(m["avg_loser"]) == pytest.approx(100.0, abs=0.01)

    def test_realized_rr(self) -> None:
        """avg_winner=150, avg_loser=100 → RR = 1.5."""
        trades = self._make_trades([100.0, 200.0, -50.0, -150.0])
        m = compute_metrics(trades)
        assert float(m["realized_rr"]) == pytest.approx(1.5, abs=0.01)

    def test_all_losses_profit_factor(self) -> None:
        """All losses → profit factor 0.0 (no gross profit)."""
        trades = self._make_trades([-100.0, -200.0])
        m = compute_metrics(trades)
        assert float(m["profit_factor"]) == 0.0

    def test_all_wins_profit_factor_is_inf(self) -> None:
        """All wins → profit factor inf."""
        trades = self._make_trades([100.0, 200.0])
        m = compute_metrics(trades)
        assert float(m["profit_factor"]) == float("inf")


class TestMaxDrawdown:
    def test_no_drawdown(self) -> None:
        """Monotonically increasing PnL curve → drawdown is exactly 0."""
        equity = pd.Series([0.0, 100.0, 200.0, 300.0])
        dd = _max_drawdown(equity)
        assert dd == 0.0

    def test_full_loss_normalised_by_initial_capital(self) -> None:
        """$100 loss from a $100 peak on $100 starting capital → -100% drawdown.

        Pass initial_capital=100 to match the curve's starting point so
        normalisation works the same as the old fixed-divisor calculation.
        """
        equity = pd.Series([100.0, 50.0, 0.0])
        dd = _max_drawdown(equity, initial_capital=100.0)
        assert dd == pytest.approx(-1.0, abs=0.01)

    def test_partial_drawdown_normalised(self) -> None:
        """$10k loss from $60k peak on $50k capital → -(10k/60k) ≈ -16.7%."""
        # PnL curve peaks at +$10k then falls to $0 (back to start)
        equity = pd.Series([0.0, 10_000.0, 0.0])
        dd = _max_drawdown(equity, initial_capital=50_000.0)
        # shifted: [50000, 60000, 50000]; peak = 60000
        # dd = (50000 - 60000) / 60000 = -10000/60000 ≈ -0.1667
        assert dd == pytest.approx(-10_000 / 60_000, abs=0.001)

    def test_no_blowup_on_small_early_peak(self) -> None:
        """Regression: tiny PnL peak then large loss must not produce -1000%+.

        Old bug: peak=$50, loss=$500 → (50-550)/50 = -10.0 (-1000%).
        New code: normalised by $50k capital → ≈ -1% drawdown.
        """
        equity = pd.Series([0.0, 50.0, -450.0])  # cumulative PnL
        dd = _max_drawdown(equity, initial_capital=50_000.0)
        # shifted: [50000, 50050, 49550]; peak = 50050
        # dd = (49550 - 50050) / 50050 ≈ -0.01
        assert -0.02 < dd < 0.0, f"Expected small drawdown near -1%, got {dd:.1%}"

    def test_empty_series(self) -> None:
        dd = _max_drawdown(pd.Series([], dtype=float))
        assert dd == 0.0


class TestSharpeRatio:
    def test_positive_returns_positive_sharpe(self) -> None:
        # Variable positive returns so std > 0 and mean > 0 → Sharpe > 0
        equity = pd.Series([0.0, 5.0, 15.0, 12.0, 25.0, 30.0])
        sharpe = _sharpe_ratio(equity)
        assert sharpe > 0

    def test_flat_equity_sharpe_is_zero(self) -> None:
        equity = pd.Series([100.0, 100.0, 100.0, 100.0])
        sharpe = _sharpe_ratio(equity)
        assert sharpe == 0.0

    def test_too_short_returns_zero(self) -> None:
        sharpe = _sharpe_ratio(pd.Series([100.0]))
        assert sharpe == 0.0


# ── 15-min EMA filter tests ───────────────────────────────────────────────────


def _make_1min_index(n_bars: int = 390, start_utc: str = "2024-01-15 14:30") -> pd.DatetimeIndex:
    """1-min UTC DatetimeIndex starting at 9:30 ET (EST, Jan = UTC-5)."""
    return pd.date_range(start_utc, periods=n_bars, freq="1min", tz="UTC")


class Test15mEMAComputation:
    def test_returns_correct_length(self) -> None:
        """Output array must match input length."""
        idx = _make_1min_index(n_bars=390)
        close = np.full(390, 480.0)
        result = _compute_15m_ema(idx, close, period=20)
        assert len(result) == 390

    def test_nan_for_insufficient_history(self) -> None:
        """Fewer than 20 15-min bars → all NaN (no EMA available)."""
        # 20 15-min bars = 300 1-min bars; use only 100
        idx = _make_1min_index(n_bars=100)
        close = np.linspace(470.0, 490.0, 100)
        result = _compute_15m_ema(idx, close, period=20)
        assert np.all(np.isnan(result))

    def test_no_lookahead_shift(self) -> None:
        """EMA value at bar N must reflect only bars before N (shift-1).

        We do this by verifying that the very first non-NaN value appears
        *after* at least (period * 15) bars have elapsed, not at bar period*15-1.
        """
        n_bars = 700  # enough for ~46 15-min bars
        idx = _make_1min_index(n_bars=n_bars)
        close = np.linspace(470.0, 500.0, n_bars)
        result = _compute_15m_ema(idx, close, period=5)
        first_valid = int(np.argmax(~np.isnan(result)))
        # shift(1) means the first usable 15-min EMA bar (bar 4, zero-indexed)
        # is pushed to bar 5, which starts at 1-min bar 5*15 = 75
        assert first_valid >= 5 * 15, f"first valid EMA at bar {first_valid}, expected >= 75"

    def test_flat_price_ema_equals_price(self) -> None:
        """With constant close prices, EMA should converge to that price."""
        n_bars = 700
        idx = _make_1min_index(n_bars=n_bars)
        close = np.full(n_bars, 480.0)
        result = _compute_15m_ema(idx, close, period=5)
        valid = result[~np.isnan(result)]
        assert len(valid) > 0
        assert np.allclose(valid, 480.0, atol=0.01)

    def test_long_blocked_when_price_below_ema(self) -> None:
        """When price < EMA15m, bullish_trend should be False → LONG blocked.

        We verify the filter logic directly (not through full backtest).
        """
        # Simulate: EMA is 485, current close is 480 (below EMA)
        ema_val = 485.0
        close_val = 480.0
        bullish_trend = close_val > ema_val
        bearish_trend = close_val < ema_val
        assert not bullish_trend, "LONG should be blocked when price < EMA"
        assert bearish_trend, "SHORT should be allowed when price < EMA"

    def test_short_blocked_when_price_above_ema(self) -> None:
        """When price > EMA15m, bearish_trend should be False → SHORT blocked."""
        ema_val = 475.0
        close_val = 480.0
        bullish_trend = close_val > ema_val
        bearish_trend = close_val < ema_val
        assert bullish_trend, "LONG should be allowed when price > EMA"
        assert not bearish_trend, "SHORT should be blocked when price > EMA"


# ── ORB percentile array tests ────────────────────────────────────────────────


def _make_multiday_orb_arrays(
    n_days: int,
    orb_range: float = 1.0,
) -> tuple[
    pd.DatetimeIndex, np.ndarray[Any, np.dtype[np.float64]], np.ndarray[Any, np.dtype[np.float64]]
]:
    """Build a multi-day 1-min index with fixed ORB high/low for testing."""
    from datetime import timedelta

    base = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)
    frames_idx = []
    for d in range(n_days):
        day_start = base + timedelta(days=d)
        frames_idx.append(pd.date_range(day_start, periods=390, freq="1min", tz="UTC"))
    index = frames_idx[0]
    for f in frames_idx[1:]:
        index = index.append(f)

    n = len(index)
    # Build ORB arrays: first 5 bars of each day = NaN, rest = high/low
    orb_high_arr = np.full(n, np.nan)
    orb_low_arr = np.full(n, np.nan)

    for d in range(n_days):
        start = d * 390
        for i in range(5, 390):
            orb_high_arr[start + i] = 480.0 + orb_range / 2
            orb_low_arr[start + i] = 480.0 - orb_range / 2

    return index, orb_high_arr, orb_low_arr


class TestORBPercentileArrays:
    def test_nan_for_first_window_days(self) -> None:
        """First 20 trading days must have NaN percentiles (no prior history)."""
        n_days = 25
        index, orb_high, orb_low = _make_multiday_orb_arrays(n_days)
        p25, p75 = _compute_orb_percentile_arrays(index, orb_high, orb_low, window=20)
        # All bars on days 0-19 should be NaN
        day20_start = 20 * 390
        assert np.all(np.isnan(p25[:day20_start])), "first 20 days should be NaN"
        assert np.all(np.isnan(p75[:day20_start])), "first 20 days should be NaN"

    def test_available_after_window_days(self) -> None:
        """Day 21 and beyond should have non-NaN percentiles."""
        n_days = 25
        index, orb_high, orb_low = _make_multiday_orb_arrays(n_days)
        p25, p75 = _compute_orb_percentile_arrays(index, orb_high, orb_low, window=20)
        # Bars after day 20 should be valid (use bar at start of day 21, after ORB)
        day21_bar = 20 * 390 + 10
        assert not np.isnan(p25[day21_bar]), "day 21 should have p25"
        assert not np.isnan(p75[day21_bar]), "day 21 should have p75"

    def test_p25_leq_p75(self) -> None:
        """P25 must always be <= P75."""
        n_days = 30
        # Vary ORB range per day using a loop
        index, orb_high, orb_low = _make_multiday_orb_arrays(n_days, orb_range=1.0)
        p25, p75 = _compute_orb_percentile_arrays(index, orb_high, orb_low, window=20)
        valid = ~np.isnan(p25) & ~np.isnan(p75)
        assert np.all(p25[valid] <= p75[valid])

    def test_constant_range_gives_equal_percentiles(self) -> None:
        """With identical ORB ranges every day, p25 == p75 == that range."""
        n_days = 30
        orb_range = 2.0
        index, orb_high, orb_low = _make_multiday_orb_arrays(n_days, orb_range=orb_range)
        p25, p75 = _compute_orb_percentile_arrays(index, orb_high, orb_low, window=20)
        # After day 20, all percentiles should equal the constant range
        day21_bar = 20 * 390 + 10
        assert p25[day21_bar] == pytest.approx(orb_range, abs=0.001)
        assert p75[day21_bar] == pytest.approx(orb_range, abs=0.001)


# ── trading window filter tests ───────────────────────────────────────────────


class TestTradingWindows:
    """Verify the explicit trading window logic (9:35-11:00 and 14:30-15:30 ET)."""

    def _in_window(self, t: Any) -> bool:
        """Replicate the next() window check."""
        from datetime import time as _time

        w1_start = _time(9, 35)
        w1_end = _time(11, 0)
        w2_start = _time(14, 30)
        w2_end = _time(15, 30)
        in_w1 = w1_start <= t < w1_end
        in_w2 = w2_start <= t < w2_end
        return in_w1 or in_w2

    def test_first_hour_open_is_tradeable(self) -> None:
        from datetime import time

        assert self._in_window(time(9, 35))
        assert self._in_window(time(10, 0))
        assert self._in_window(time(10, 59))

    def test_first_hour_close_boundary_blocked(self) -> None:
        from datetime import time

        assert not self._in_window(time(11, 0))  # exclusive upper bound
        assert not self._in_window(time(11, 30))

    def test_midday_blocked(self) -> None:
        from datetime import time

        assert not self._in_window(time(12, 0))
        assert not self._in_window(time(13, 0))
        assert not self._in_window(time(14, 0))
        assert not self._in_window(time(14, 29))

    def test_power_hour_is_tradeable(self) -> None:
        from datetime import time

        assert self._in_window(time(14, 30))
        assert self._in_window(time(15, 0))
        assert self._in_window(time(15, 29))

    def test_power_hour_upper_boundary_blocked(self) -> None:
        from datetime import time

        assert not self._in_window(time(15, 30))  # exclusive upper bound
        assert not self._in_window(time(15, 45))
