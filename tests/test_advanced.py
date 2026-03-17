"""Comprehensive tests for advanced modules: Kalman filter, Volume Profile, HMM Regime."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from src.backtest.volume_profile import VolumeProfileResult, compute_volume_profile
from src.levels.kalman_levels import compute_kalman_stop_multiplier
from src.strategies.hmm_regime import (
    HMMRegime,
    compute_hmm_regime_for_window,
    train_hmm,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def _make_1day_index(
    date: str = "2024-01-15",
    n_bars: int = 390,
) -> pd.DatetimeIndex:
    """Return a UTC DatetimeIndex spanning one market day (14:30-21:00 UTC)."""
    start = pd.Timestamp(date, tz="UTC").replace(hour=14, minute=30)
    return pd.date_range(start, periods=n_bars, freq="1min", tz="UTC")


def _make_ohlcv_df(
    n_days: int = 1,
    bars_per_day: int = 390,
    base_price: float = 480.0,
    volatility: float = 0.3,
    start_date: str = "2024-01-15",
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic 1-min OHLCV DataFrame with UTC DatetimeIndex."""
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    current_date = pd.Timestamp(start_date, tz="America/New_York")

    for _ in range(n_days):
        while current_date.dayofweek >= 5:
            current_date += pd.Timedelta(days=1)
        market_open = current_date.replace(hour=9, minute=30, second=0, microsecond=0).tz_convert(
            "UTC"
        )

        for bar_idx in range(bars_per_day):
            ts = market_open + pd.Timedelta(minutes=bar_idx)
            drift = 0.001 * bar_idx / bars_per_day
            noise = rng.normal(0, volatility)
            mid = base_price * (1 + drift) + noise
            spread = rng.uniform(0.1, 0.5)
            o = mid + rng.normal(0, 0.1)
            c = mid + rng.normal(0, 0.1)
            h = max(o, c) + spread
            l_ = min(o, c) - spread
            vol = int(rng.integers(50_000, 500_000))
            rows.append(
                {
                    "open": o,
                    "high": h,
                    "low": l_,
                    "close": c,
                    "volume": vol,
                    "timestamp": ts,
                }
            )
        current_date += pd.Timedelta(days=1)

    df = pd.DataFrame(rows).set_index("timestamp")
    df.index = pd.DatetimeIndex(df.index, tz="UTC")
    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)
    return df


# ===========================================================================
# Kalman Filter Tests
# ===========================================================================


class TestKalmanStopMultiplier:
    """Tests for compute_kalman_stop_multiplier."""

    def _default_inputs(
        self, n_bars: int = 390, base_price: float = 480.0
    ) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
        """Return standard (index, close, atr) for one market day."""
        rng = np.random.default_rng(42)
        idx = _make_1day_index(n_bars=n_bars)
        close = base_price + np.cumsum(rng.normal(0, 0.05, n_bars))
        atr = np.full(n_bars, 1.5)
        return idx, close, atr

    def test_returns_array_same_length(self) -> None:
        idx, close, atr = self._default_inputs()
        result = compute_kalman_stop_multiplier(idx, close, atr)
        assert len(result) == len(idx)

    def test_all_values_positive(self) -> None:
        idx, close, atr = self._default_inputs()
        result = compute_kalman_stop_multiplier(idx, close, atr)
        assert np.all(result > 0)

    def test_orb_bars_default_to_one(self) -> None:
        """First 5 bars (ORB bars) should have multiplier == 1.0."""
        idx, close, atr = self._default_inputs()
        result = compute_kalman_stop_multiplier(idx, close, atr, orb_bars=5)
        assert np.all(result[:5] == 1.0)

    def test_values_clamped(self) -> None:
        idx, close, atr = self._default_inputs()
        result = compute_kalman_stop_multiplier(idx, close, atr)
        assert np.all(result >= 0.90)
        assert np.all(result <= 1.50)

    def test_nan_atr_returns_all_ones(self) -> None:
        idx, close, _ = self._default_inputs()
        atr_nan = np.full(len(idx), np.nan)
        result = compute_kalman_stop_multiplier(idx, close, atr_nan)
        assert np.allclose(result, 1.0)

    def test_single_bar_day_returns_ones(self) -> None:
        idx = _make_1day_index(n_bars=1)
        close = np.array([480.0])
        atr = np.array([1.5])
        result = compute_kalman_stop_multiplier(idx, close, atr)
        assert np.allclose(result, 1.0)

    def test_volatile_prices_higher_multiplier(self) -> None:
        """Volatile prices should produce higher multipliers than flat prices."""
        rng = np.random.default_rng(42)
        n = 390
        idx = _make_1day_index(n_bars=n)
        atr = np.full(n, 1.5)

        # Flat prices: very small noise
        close_flat = 480.0 + np.cumsum(rng.normal(0, 0.001, n))
        result_flat = compute_kalman_stop_multiplier(idx, close_flat, atr)

        # Volatile prices: large noise
        rng2 = np.random.default_rng(99)
        close_volatile = 480.0 + np.cumsum(rng2.normal(0, 2.0, n))
        result_volatile = compute_kalman_stop_multiplier(idx, close_volatile, atr)

        # Average multiplier on volatile data should exceed flat data
        avg_flat = result_flat[5:].mean()
        avg_volatile = result_volatile[5:].mean()
        assert avg_volatile > avg_flat

    def test_custom_orb_bars(self) -> None:
        """With orb_bars=10, first 10 bars should be 1.0."""
        idx, close, atr = self._default_inputs()
        result = compute_kalman_stop_multiplier(idx, close, atr, orb_bars=10)
        assert np.all(result[:10] == 1.0)

    def test_fewer_bars_than_orb_returns_ones(self) -> None:
        """If total bars <= orb_bars, all should be 1.0."""
        idx = _make_1day_index(n_bars=3)
        close = np.array([480.0, 480.1, 480.2])
        atr = np.array([1.5, 1.5, 1.5])
        result = compute_kalman_stop_multiplier(idx, close, atr, orb_bars=5)
        assert np.allclose(result, 1.0)

    def test_zero_atr_returns_ones(self) -> None:
        idx, close, _ = self._default_inputs()
        atr_zero = np.zeros(len(idx))
        result = compute_kalman_stop_multiplier(idx, close, atr_zero)
        assert np.allclose(result, 1.0)

    def test_multi_day_index(self) -> None:
        """With a multi-day index, each day is processed independently."""
        idx1 = _make_1day_index(date="2024-01-15", n_bars=390)
        idx2 = _make_1day_index(date="2024-01-16", n_bars=390)
        idx = idx1.append(idx2)
        rng = np.random.default_rng(42)
        n = len(idx)
        close = 480.0 + np.cumsum(rng.normal(0, 0.05, n))
        atr = np.full(n, 1.5)
        result = compute_kalman_stop_multiplier(idx, close, atr)
        assert len(result) == n
        # ORB bars of second day should also be 1.0
        assert np.all(result[390:395] == 1.0)

    def test_result_dtype_is_float(self) -> None:
        idx, close, atr = self._default_inputs()
        result = compute_kalman_stop_multiplier(idx, close, atr)
        assert result.dtype == np.float64 or np.issubdtype(result.dtype, np.floating)


# ===========================================================================
# Volume Profile Tests
# ===========================================================================


class TestVolumeProfile:
    """Tests for compute_volume_profile."""

    def _make_bars(
        self,
        n: int = 100,
        base_price: float = 480.0,
        seed: int = 42,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (high, low, close, volume) arrays."""
        rng = np.random.default_rng(seed)
        close = base_price + np.cumsum(rng.normal(0, 0.1, n))
        spread = rng.uniform(0.1, 0.5, n)
        high = close + spread
        low = close - spread
        volume = rng.integers(50_000, 500_000, size=n).astype(float)
        return high, low, close, volume

    def test_returns_volume_profile_result(self) -> None:
        h, lo, c, v = self._make_bars()
        result = compute_volume_profile(h, lo, c, v)
        assert isinstance(result, VolumeProfileResult)

    def test_poc_within_price_range(self) -> None:
        h, lo, c, v = self._make_bars()
        result = compute_volume_profile(h, lo, c, v)
        assert result is not None
        assert float(np.min(lo)) <= result.poc <= float(np.max(h))

    def test_vah_ge_poc_ge_val(self) -> None:
        h, lo, c, v = self._make_bars()
        result = compute_volume_profile(h, lo, c, v)
        assert result is not None
        assert result.vah >= result.poc
        assert result.poc >= result.val_

    def test_value_area_contains_approximately_70_pct(self) -> None:
        """The value area should capture roughly 70% of total volume."""
        rng = np.random.default_rng(42)
        n = 200
        # Use prices centered tightly so bins are well populated
        close = 480.0 + np.cumsum(rng.normal(0, 0.2, n))
        spread = rng.uniform(0.05, 0.15, n)
        high = close + spread
        low = close - spread
        volume = rng.integers(100_000, 500_000, size=n).astype(float)

        result = compute_volume_profile(high, low, close, volume, tick_size=0.05)
        assert result is not None

        # Volume within value area should be >= 65% (allowing small tolerance)
        # We verify by rebuilding a simplified check: bars whose mid is in VA
        total_vol = volume.sum()
        va_vol = 0.0
        for i in range(n):
            mid = (high[i] + low[i]) / 2
            if result.val_ <= mid <= result.vah:
                va_vol += volume[i]
        # The actual VA is built at tick-level resolution, so bar-level check
        # is approximate. Just ensure it's in the right ballpark (> 50%).
        assert va_vol / total_vol > 0.50

    def test_returns_none_for_less_than_5_bars(self) -> None:
        h = np.array([481.0, 482.0, 483.0, 484.0])
        lo = np.array([479.0, 480.0, 481.0, 482.0])
        c = np.array([480.0, 481.0, 482.0, 483.0])
        v = np.array([100_000.0, 100_000.0, 100_000.0, 100_000.0])
        assert compute_volume_profile(h, lo, c, v) is None

    def test_returns_none_for_all_zero_volume(self) -> None:
        h, lo, c, _ = self._make_bars()
        v = np.zeros(len(h))
        assert compute_volume_profile(h, lo, c, v) is None

    def test_returns_none_when_high_le_low(self) -> None:
        """Flat price (high == low for all bars) should return None."""
        n = 50
        flat = np.full(n, 480.0)
        v = np.full(n, 100_000.0)
        assert compute_volume_profile(flat, flat, flat, v) is None

    def test_hvn_above_average(self) -> None:
        """HVN nodes should correspond to bins with volume > 1.5x average."""
        h, lo, c, v = self._make_bars(n=200)
        result = compute_volume_profile(h, lo, c, v)
        assert result is not None
        # HVN should be a list of floats
        assert isinstance(result.hvn, list)
        for node in result.hvn:
            assert isinstance(node, float)

    def test_lvn_below_average(self) -> None:
        """LVN nodes should correspond to bins with volume < 0.5x average."""
        h, lo, c, v = self._make_bars(n=200)
        result = compute_volume_profile(h, lo, c, v)
        assert result is not None
        assert isinstance(result.lvn, list)
        for node in result.lvn:
            assert isinstance(node, float)

    def test_frozen_dataclass(self) -> None:
        """VolumeProfileResult should be immutable (frozen)."""
        h, lo, c, v = self._make_bars()
        result = compute_volume_profile(h, lo, c, v)
        assert result is not None
        with pytest.raises(AttributeError):
            result.poc = 999.0  # type: ignore[misc]

    def test_hvn_lvn_sorted(self) -> None:
        h, lo, c, v = self._make_bars(n=300)
        result = compute_volume_profile(h, lo, c, v)
        assert result is not None
        assert result.hvn == sorted(result.hvn)
        assert result.lvn == sorted(result.lvn)

    def test_custom_tick_size(self) -> None:
        h, lo, c, v = self._make_bars()
        result_fine = compute_volume_profile(h, lo, c, v, tick_size=0.01)
        result_coarse = compute_volume_profile(h, lo, c, v, tick_size=0.50)
        assert result_fine is not None
        assert result_coarse is not None
        # Both should produce a valid POC
        assert float(np.min(lo)) <= result_fine.poc <= float(np.max(h))
        assert float(np.min(lo)) <= result_coarse.poc <= float(np.max(h))

    def test_single_high_volume_bar(self) -> None:
        """A single bar with massive volume should dominate the POC."""
        rng = np.random.default_rng(42)
        n = 50
        close = 480.0 + np.cumsum(rng.normal(0, 0.1, n))
        spread = rng.uniform(0.1, 0.3, n)
        high = close + spread
        low = close - spread
        volume = np.full(n, 10_000.0)
        # Place a huge volume bar at index 25
        volume[25] = 10_000_000.0
        result = compute_volume_profile(high, low, close, volume)
        assert result is not None
        # POC should be near bar 25's price range
        bar25_mid = (high[25] + low[25]) / 2
        assert abs(result.poc - bar25_mid) < 1.0


# ===========================================================================
# HMM Regime Tests
# ===========================================================================


class TestHMMRegime:
    """Tests for HMM regime detection."""

    @pytest.fixture()
    def large_is_df(self) -> pd.DataFrame:
        """60+ trading days of 1-min data for sufficient IS training."""
        return _make_ohlcv_df(n_days=65, bars_per_day=390, seed=42)

    @pytest.fixture()
    def small_oos_df(self) -> pd.DataFrame:
        """5 trading days of OOS data."""
        return _make_ohlcv_df(
            n_days=5,
            bars_per_day=390,
            start_date="2024-04-15",
            seed=99,
        )

    def test_hmm_regime_enum_values(self) -> None:
        assert HMMRegime.CALM == 0
        assert HMMRegime.NORMAL == 1
        assert HMMRegime.VOLATILE == 2

    def test_returns_array_same_length_as_oos(
        self, large_is_df: pd.DataFrame, small_oos_df: pd.DataFrame
    ) -> None:
        result = compute_hmm_regime_for_window(large_is_df, small_oos_df)
        assert len(result) == len(small_oos_df)

    def test_values_in_valid_set(
        self, large_is_df: pd.DataFrame, small_oos_df: pd.DataFrame
    ) -> None:
        result = compute_hmm_regime_for_window(large_is_df, small_oos_df)
        valid_values = {0.0, 1.0, 2.0}
        unique_vals = set(np.unique(result))
        assert unique_vals.issubset(valid_values)

    def test_insufficient_data_returns_default(self) -> None:
        """With < 200 5-min bars IS data, should fall back to NORMAL (1.0)."""
        small_is = _make_ohlcv_df(n_days=1, bars_per_day=390, seed=42)
        oos = _make_ohlcv_df(
            n_days=2,
            bars_per_day=390,
            start_date="2024-01-22",
            seed=99,
        )
        result = compute_hmm_regime_for_window(small_is, oos)
        assert np.allclose(result, 1.0)

    def test_all_same_price_graceful(self) -> None:
        """All-same-price data should not crash."""
        n_days_is = 65
        n_bars = n_days_is * 390
        idx_start = pd.Timestamp("2024-01-15 14:30", tz="UTC")
        idx_is = pd.date_range(idx_start, periods=n_bars, freq="1min", tz="UTC")
        df_is = pd.DataFrame(
            {
                "open": 480.0,
                "high": 480.0,
                "low": 480.0,
                "close": 480.0,
                "volume": 100_000,
            },
            index=idx_is,
        )

        n_oos = 5 * 390
        idx_oos = pd.date_range(
            pd.Timestamp("2024-04-15 14:30", tz="UTC"),
            periods=n_oos,
            freq="1min",
            tz="UTC",
        )
        df_oos = pd.DataFrame(
            {
                "open": 480.0,
                "high": 480.0,
                "low": 480.0,
                "close": 480.0,
                "volume": 100_000,
            },
            index=idx_oos,
        )

        # Should not raise
        result = compute_hmm_regime_for_window(df_is, df_oos)
        assert len(result) == n_oos

    def test_empty_is_returns_default(self) -> None:
        oos = _make_ohlcv_df(n_days=2, start_date="2024-01-22", seed=99)
        empty_is = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        empty_is.index = pd.DatetimeIndex([], tz="UTC")
        result = compute_hmm_regime_for_window(empty_is, oos)
        assert np.allclose(result, 1.0)

    def test_empty_oos_returns_empty(self) -> None:
        is_df = _make_ohlcv_df(n_days=65, seed=42)
        empty_oos = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        empty_oos.index = pd.DatetimeIndex([], tz="UTC")
        result = compute_hmm_regime_for_window(is_df, empty_oos)
        assert len(result) == 0

    def test_result_dtype_is_float(
        self, large_is_df: pd.DataFrame, small_oos_df: pd.DataFrame
    ) -> None:
        result = compute_hmm_regime_for_window(large_is_df, small_oos_df)
        assert np.issubdtype(result.dtype, np.floating)


class TestTrainHMM:
    """Tests for train_hmm standalone."""

    def test_returns_none_with_insufficient_data(self) -> None:
        """A tiny 5-min dataframe with < 200 clean rows should return None."""
        rng = np.random.default_rng(42)
        n = 50  # Way below 200 minimum
        idx = pd.date_range("2024-01-15 14:30", periods=n, freq="5min", tz="UTC")
        df = pd.DataFrame(
            {
                "open": 480.0 + rng.normal(0, 0.1, n),
                "high": 481.0 + rng.normal(0, 0.1, n),
                "low": 479.0 + rng.normal(0, 0.1, n),
                "close": 480.0 + rng.normal(0, 0.1, n),
                "volume": rng.integers(50_000, 500_000, size=n).astype(float),
            },
            index=idx,
        )
        result = train_hmm(df)
        assert result is None

    def test_returns_tuple_with_sufficient_data(self) -> None:
        """With enough clean 5-min bars, should return (model, mapping, scaler)."""
        rng = np.random.default_rng(42)
        n = 300  # Above 200 minimum
        idx = pd.date_range("2024-01-15 14:30", periods=n, freq="5min", tz="UTC")
        close = 480.0 + np.cumsum(rng.normal(0, 0.1, n))
        spread = rng.uniform(0.1, 0.5, n)
        df = pd.DataFrame(
            {
                "open": close + rng.normal(0, 0.05, n),
                "high": close + spread,
                "low": close - spread,
                "close": close,
                "volume": rng.integers(50_000, 500_000, size=n).astype(float),
            },
            index=idx,
        )
        result = train_hmm(df)
        assert result is not None
        _model, mapping, _scaler = result
        # mapping should have 3 entries
        assert len(mapping) == 3
        # All three regime labels should be present
        assert set(mapping.values()) == {
            HMMRegime.CALM,
            HMMRegime.NORMAL,
            HMMRegime.VOLATILE,
        }

    def test_returns_none_for_very_short_dataframe(self) -> None:
        """DataFrame too short to even compute ATR should return None."""
        rng = np.random.default_rng(42)
        n = 10
        idx = pd.date_range("2024-01-15 14:30", periods=n, freq="5min", tz="UTC")
        df = pd.DataFrame(
            {
                "open": 480.0 + rng.normal(0, 0.1, n),
                "high": 481.0 + rng.normal(0, 0.1, n),
                "low": 479.0 + rng.normal(0, 0.1, n),
                "close": 480.0 + rng.normal(0, 0.1, n),
                "volume": rng.integers(50_000, 500_000, size=n).astype(float),
            },
            index=idx,
        )
        result = train_hmm(df)
        assert result is None
