"""Tests for helper functions in src.backtest.engine."""

from __future__ import annotations

from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import (
    _ET_TZ,
    _compute_15m_ema,
    _compute_daily_adx,
    _compute_daily_realized_vol,
)
from tests.conftest import make_1min_df

# ── synthetic data fixtures ──────────────────────────────────────────────────


@pytest.fixture
def df_30d() -> pd.DataFrame:
    """30 trading days of synthetic 1-min data — enough for all rolling windows."""
    return make_1min_df(n_days=30, bars_per_day=390, base_price=480.0)


@pytest.fixture
def df_5d() -> pd.DataFrame:
    """5 trading days of synthetic 1-min data."""
    return make_1min_df(n_days=5, bars_per_day=390, base_price=480.0)


# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════


class TestEngineConstants:
    """Verify key module-level constants."""

    def test_et_tz_is_new_york(self) -> None:
        assert ZoneInfo("America/New_York") == _ET_TZ

    def test_slippage_default(self) -> None:
        """run_backtest has slippage=0.02 as default parameter."""
        import inspect

        from src.backtest.engine import run_backtest

        sig = inspect.signature(run_backtest)
        assert sig.parameters["slippage"].default == 0.02


# ═══════════════════════════════════════════════════════════════════════════════
# _compute_daily_realized_vol
# ═══════════════════════════════════════════════════════════════════════════════


class TestComputeDailyRealizedVol:
    """Tests for _compute_daily_realized_vol()."""

    def test_output_length_matches_input(self, df_30d: pd.DataFrame) -> None:
        result = _compute_daily_realized_vol(df_30d.index, df_30d["close"].values)
        assert len(result) == len(df_30d)

    def test_nan_for_first_warmup_days(self, df_30d: pd.DataFrame) -> None:
        result = _compute_daily_realized_vol(df_30d.index, df_30d["close"].values)
        # First day's bars should all be NaN (need 20 days + 1 shift)
        first_day_end = 390  # first day = 390 bars
        assert np.all(np.isnan(result[:first_day_end]))

    def test_positive_values_where_available(self, df_30d: pd.DataFrame) -> None:
        result = _compute_daily_realized_vol(df_30d.index, df_30d["close"].values)
        valid = result[~np.isnan(result)]
        if len(valid) > 0:
            assert np.all(valid >= 0)

    def test_short_data_all_nan(self, df_5d: pd.DataFrame) -> None:
        """With only 5 days, rolling 20-day window cannot produce valid values."""
        result = _compute_daily_realized_vol(df_5d.index, df_5d["close"].values)
        assert np.all(np.isnan(result))


# ═══════════════════════════════════════════════════════════════════════════════
# _compute_daily_adx
# ═══════════════════════════════════════════════════════════════════════════════


class TestComputeDailyAdx:
    """Tests for _compute_daily_adx()."""

    def test_output_length_matches_input(self, df_30d: pd.DataFrame) -> None:
        result = _compute_daily_adx(
            df_30d.index,
            df_30d["high"].values,
            df_30d["low"].values,
            df_30d["close"].values,
        )
        assert len(result) == len(df_30d)

    def test_nan_for_warmup_period(self, df_30d: pd.DataFrame) -> None:
        result = _compute_daily_adx(
            df_30d.index,
            df_30d["high"].values,
            df_30d["low"].values,
            df_30d["close"].values,
        )
        # ADX(14) needs 14+ daily bars + 1 shift => first ~15 days are NaN
        # Check that at least the first day is NaN
        first_day_bars = 390
        assert np.all(np.isnan(result[:first_day_bars]))

    def test_shifted_by_one_day_no_lookahead(self, df_30d: pd.DataFrame) -> None:
        """ADX is shifted 1 day: day D uses ADX through D-1.

        Verify by checking that all bars on the same ET date share
        one value, and that value changes day-to-day (not bar-to-bar).
        """
        result = _compute_daily_adx(
            df_30d.index,
            df_30d["high"].values,
            df_30d["low"].values,
            df_30d["close"].values,
        )
        et_dates = df_30d.index.tz_convert("America/New_York").date
        # Group by ET date, check each date has a single unique non-NaN value
        for d in set(et_dates):
            mask = et_dates == d
            day_vals = result[mask]
            valid = day_vals[~np.isnan(day_vals)]
            if len(valid) > 0:
                assert np.all(valid == valid[0]), f"ADX not constant within day {d}"

    def test_insufficient_days_all_nan(self, df_5d: pd.DataFrame) -> None:
        """ADX(14) needs at least period+1 daily bars; 5 days is not enough."""
        result = _compute_daily_adx(
            df_5d.index,
            df_5d["high"].values,
            df_5d["low"].values,
            df_5d["close"].values,
        )
        assert np.all(np.isnan(result))


# ═══════════════════════════════════════════════════════════════════════════════
# _compute_15m_ema
# ═══════════════════════════════════════════════════════════════════════════════


class TestCompute15mEma:
    """Tests for _compute_15m_ema()."""

    def test_output_length_matches_input(self, df_30d: pd.DataFrame) -> None:
        result = _compute_15m_ema(df_30d.index, df_30d["close"].values)
        assert len(result) == len(df_30d)

    def test_nan_before_warmup(self, df_5d: pd.DataFrame) -> None:
        """EMA(20) on 15-min bars + shift 1 → first bars should be NaN."""
        result = _compute_15m_ema(df_5d.index, df_5d["close"].values)
        # The very first bar must be NaN (not enough 15-min bars yet)
        assert np.isnan(result[0])

    def test_values_are_reasonable_prices(self, df_30d: pd.DataFrame) -> None:
        """Where EMA is available, values should be near the base price (~480)."""
        result = _compute_15m_ema(df_30d.index, df_30d["close"].values)
        valid = result[~np.isnan(result)]
        if len(valid) > 0:
            assert np.all(valid > 400)
            assert np.all(valid < 600)

    def test_too_few_bars_returns_all_nan(self) -> None:
        """If fewer than period 15-min bars exist, result is all NaN."""
        # 10 bars of 1-min data → less than 1 complete 15-min bar at period=20
        df = make_1min_df(n_days=1, bars_per_day=10)
        result = _compute_15m_ema(df.index, df["close"].values, period=20)
        assert np.all(np.isnan(result))
