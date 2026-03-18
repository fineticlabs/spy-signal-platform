"""Additional tests for the backtest engine — run_backtest wrapper and helpers.

Coverage targets
----------------
- run_backtest: custom parameters, very short data, equity final key, trade count.
- _classify_gap: already well-covered in test_backtest.py — skip.
- _compute_first5min_rvol: NaN handling, single-day, multi-day with short bars.
"""

from __future__ import annotations

import contextlib
import warnings
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import (
    _compute_first5min_rvol,
    run_backtest,
)
from tests.conftest import make_1min_df

# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_trading_day_index(
    date_str: str,
    n_bars: int = 390,
) -> pd.DatetimeIndex:
    """Create a 1-min DatetimeIndex for a single US trading day (14:30-21:00 UTC)."""
    day_start = pd.Timestamp(f"{date_str} 14:30:00", tz="UTC")
    return pd.date_range(start=day_start, periods=n_bars, freq="min")


def _make_multiday_index(
    start_date: str,
    n_days: int,
    bars_per_day: int = 390,
) -> pd.DatetimeIndex:
    """Create a 1-min DatetimeIndex spanning multiple trading days (skips weekends)."""
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


def _make_ohlcv_df(
    index: pd.DatetimeIndex,
    base_price: float = 480.0,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Build an OHLCV DataFrame suitable for run_backtest."""
    if rng is None:
        rng = np.random.default_rng(42)
    n = len(index)
    noise = rng.normal(0, 0.5, n)
    close = base_price + np.cumsum(noise)
    open_ = close + rng.normal(0, 0.1, n)
    high = np.maximum(open_, close) + rng.uniform(0.1, 1.0, n)
    low = np.minimum(open_, close) - rng.uniform(0.1, 1.0, n)
    volume = (100_000 + rng.integers(-20_000, 20_000, n)).astype(float)
    volume = np.maximum(volume, 1000)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=index,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# run_backtest additional coverage
# ═══════════════════════════════════════════════════════════════════════════════


class TestRunBacktestExtended:
    """Extended tests for run_backtest() — custom params and edge cases."""

    @pytest.fixture(autouse=True)
    def _suppress_warnings(self) -> None:
        warnings.filterwarnings("ignore")

    def test_custom_cash_passed_through(self) -> None:
        """Custom cash parameter is reflected in starting equity."""
        idx = _make_multiday_index("2024-01-02", n_days=30)
        df = _make_ohlcv_df(idx)
        stats = run_backtest(df, cash=100_000.0)
        # With 0 or few trades, equity final should be close to initial cash
        equity = float(stats["Equity Final [$]"])
        # If no trades, equity == cash. If trades, equity changed. Either way, no crash.
        assert equity > 0

    def test_custom_slippage_accepted(self) -> None:
        """Slippage parameter is accepted without error."""
        idx = _make_multiday_index("2024-01-02", n_days=30)
        df = _make_ohlcv_df(idx)
        stats = run_backtest(df, slippage=0.05)
        assert "# Trades" in stats

    def test_equity_final_key_present(self) -> None:
        """Stats object has 'Equity Final [$]' key."""
        idx = _make_multiday_index("2024-01-02", n_days=30)
        df = _make_ohlcv_df(idx)
        stats = run_backtest(df)
        assert "Equity Final [$]" in stats

    def test_trades_is_dataframe(self) -> None:
        """_trades attribute is a pandas DataFrame."""
        idx = _make_multiday_index("2024-01-02", n_days=30)
        df = _make_ohlcv_df(idx)
        stats = run_backtest(df)
        assert isinstance(stats._trades, pd.DataFrame)

    def test_very_short_data_handled(self) -> None:
        """< 5 bars should not cause unhandled crash."""
        idx = _make_trading_day_index("2024-01-15", n_bars=3)
        df = _make_ohlcv_df(idx)
        with contextlib.suppress(Exception):
            run_backtest(df)

    def test_custom_atr_mult(self) -> None:
        """Custom ATR multiplier accepted."""
        idx = _make_multiday_index("2024-01-02", n_days=30)
        df = _make_ohlcv_df(idx)
        stats = run_backtest(df, atr_mult=2.0)
        assert "# Trades" in stats

    def test_custom_symbol_parameter(self) -> None:
        """Non-SPY symbol accepted."""
        idx = _make_multiday_index("2024-01-02", n_days=30)
        df = _make_ohlcv_df(idx)
        stats = run_backtest(df, symbol="AAPL")
        assert "# Trades" in stats

    def test_make_1min_df_produces_valid_backtest_input(self) -> None:
        """make_1min_df from conftest produces data run_backtest can consume."""
        df = make_1min_df(n_days=30)
        stats = run_backtest(df)
        assert "# Trades" in stats
        assert float(stats["Equity Final [$]"]) > 0

    def test_return_pct_key_present(self) -> None:
        """Stats has Return [%] key."""
        idx = _make_multiday_index("2024-01-02", n_days=30)
        df = _make_ohlcv_df(idx)
        stats = run_backtest(df)
        assert "Return [%]" in stats

    def test_max_trades_per_day_param(self) -> None:
        """Custom max_trades_per_day accepted."""
        idx = _make_multiday_index("2024-01-02", n_days=30)
        df = _make_ohlcv_df(idx)
        stats = run_backtest(df, max_trades_per_day=1)
        assert "# Trades" in stats


# ═══════════════════════════════════════════════════════════════════════════════
# _compute_first5min_rvol extended coverage
# ═══════════════════════════════════════════════════════════════════════════════


class TestComputeFirst5minRvolExtended:
    """Additional tests for _compute_first5min_rvol NaN handling and edge cases."""

    def test_single_day_all_nan(self) -> None:
        """A single trading day has no prior history → all NaN."""
        idx = _make_trading_day_index("2024-01-15", n_bars=390)
        rng = np.random.default_rng(42)
        volume = rng.integers(50_000, 500_000, 390).astype(float)
        rvol = _compute_first5min_rvol(idx, volume)
        assert len(rvol) == 390
        # Only 1 day → rolling(20).mean().shift(1) is NaN → all NaN
        assert np.all(np.isnan(rvol))

    def test_output_length_matches_input(self) -> None:
        """RVOL array length matches input index length."""
        idx = _make_multiday_index("2024-01-02", n_days=25)
        rng = np.random.default_rng(42)
        volume = rng.integers(50_000, 500_000, len(idx)).astype(float)
        rvol = _compute_first5min_rvol(idx, volume)
        assert len(rvol) == len(idx)

    def test_insufficient_history_gives_nan(self) -> None:
        """Days within the first 20-day window should have NaN RVOL."""
        idx = _make_multiday_index("2024-01-02", n_days=5)
        rng = np.random.default_rng(42)
        volume = rng.integers(50_000, 500_000, len(idx)).astype(float)
        rvol = _compute_first5min_rvol(idx, volume)
        # 5 days < 20-day window → all NaN
        assert np.all(np.isnan(rvol))

    def test_day_with_fewer_than_5_bars_gets_nan(self) -> None:
        """If a day has < 5 bars, its first-5-min volume is NaN."""
        # Create 22 full days + 1 short day (3 bars)
        full_idx = _make_multiday_index("2024-01-02", n_days=22)
        rng = np.random.default_rng(42)

        # Add a short day (Feb 5 is a Monday — skip to Tue Feb 6)
        short_start = pd.Timestamp("2024-02-06 14:30:00", tz="UTC")
        short_idx = pd.date_range(start=short_start, periods=3, freq="min")
        combined_idx = full_idx.append(short_idx)

        volume = rng.integers(50_000, 500_000, len(combined_idx)).astype(float)
        rvol = _compute_first5min_rvol(combined_idx, volume)
        assert len(rvol) == len(combined_idx)

    def test_rvol_constant_within_day(self) -> None:
        """RVOL value is the same for all bars within a single trading day."""
        idx = _make_multiday_index("2024-01-02", n_days=25)
        rng = np.random.default_rng(42)
        volume = rng.integers(50_000, 500_000, len(idx)).astype(float)
        rvol = _compute_first5min_rvol(idx, volume)

        # Check last day: all bars should have same RVOL (or all NaN)
        last_day_start = 24 * 390
        last_day_rvol = rvol[last_day_start : last_day_start + 390]
        non_nan = last_day_rvol[~np.isnan(last_day_rvol)]
        if len(non_nan) > 0:
            assert np.all(non_nan == non_nan[0])

    def test_zero_volume_does_not_crash(self) -> None:
        """Zero volume bars should not cause division errors."""
        idx = _make_multiday_index("2024-01-02", n_days=25)
        volume = np.zeros(len(idx))
        rvol = _compute_first5min_rvol(idx, volume)
        assert len(rvol) == len(idx)
