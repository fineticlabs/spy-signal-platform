"""Extended tests for src.indicators.batch — covering calculate_bollinger and calculate_vwap edge cases."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.indicators.batch import calculate_bollinger, calculate_vwap


def _make_df(
    closes: list[float],
    *,
    highs: list[float] | None = None,
    lows: list[float] | None = None,
    volumes: list[int] | None = None,
    dates: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame from close prices."""
    n = len(closes)
    h = highs if highs is not None else [c + 0.5 for c in closes]
    l_ = lows if lows is not None else [c - 0.5 for c in closes]
    v = volumes if volumes is not None else [100_000] * n
    idx = (
        dates
        if dates is not None
        else pd.date_range("2024-01-15 14:30", periods=n, freq="1min", tz="UTC")
    )
    return pd.DataFrame(
        {"open": closes, "high": h, "low": l_, "close": closes, "volume": v},
        index=idx,
    )


# ── calculate_bollinger ──────────────────────────────────────────────────────


class TestCalculateBollinger:
    """Cover calculate_bollinger (lines 90-95): returns upper/middle/lower, SMA relationship, leading NaN."""

    def test_returns_three_series(self) -> None:
        df = _make_df([float(i) for i in range(30)])
        upper, middle, lower = calculate_bollinger(df, period=20, std=2)
        assert isinstance(upper, pd.Series)
        assert isinstance(middle, pd.Series)
        assert isinstance(lower, pd.Series)
        assert len(upper) == len(df)

    def test_middle_is_sma(self) -> None:
        closes = [float(480 + i * 0.1) for i in range(30)]
        df = _make_df(closes)
        _, middle, _ = calculate_bollinger(df, period=20, std=2)
        # Middle at index 19 should be the SMA of the first 20 closes
        expected_sma = sum(closes[:20]) / 20
        assert middle.iloc[19] == pytest.approx(expected_sma, rel=1e-6)

    def test_upper_middle_lower_relationship(self) -> None:
        """After warmup, upper > middle > lower always holds."""
        closes = [float(480 + np.sin(i / 3)) for i in range(50)]
        df = _make_df(closes)
        upper, middle, lower = calculate_bollinger(df, period=20, std=2)
        # Check only after warmup (period-1 bars are NaN)
        valid = ~np.isnan(upper.to_numpy())
        assert all(upper[valid] > middle[valid])
        assert all(middle[valid] > lower[valid])

    def test_leading_nan_count(self) -> None:
        """First period-1 bars should be NaN."""
        period = 20
        df = _make_df([float(480 + i) for i in range(40)])
        upper, middle, lower = calculate_bollinger(df, period=period, std=2)
        assert np.isnan(upper.iloc[0])
        assert np.isnan(middle.iloc[0])
        assert np.isnan(lower.iloc[0])
        # Value at index period-1 should be valid
        assert not np.isnan(middle.iloc[period - 1])

    def test_bands_width_proportional_to_std(self) -> None:
        """Upper - middle should equal middle - lower (symmetric bands)."""
        closes = [float(480 + np.sin(i / 3)) for i in range(50)]
        df = _make_df(closes)
        upper, middle, lower = calculate_bollinger(df, period=20, std=2)
        idx = 30  # well past warmup
        upper_spread = float(upper.iloc[idx] - middle.iloc[idx])
        lower_spread = float(middle.iloc[idx] - lower.iloc[idx])
        assert upper_spread == pytest.approx(lower_spread, rel=1e-6)


# ── calculate_vwap ───────────────────────────────────────────────────────────


class TestCalculateVwap:
    """Cover calculate_vwap edge cases (lines 121-130): single bar, multi-bar accumulation."""

    def test_single_bar_equals_typical_price(self) -> None:
        h, l_, c = 481.0, 479.0, 480.0
        df = _make_df([c], highs=[h], lows=[l_], volumes=[100_000])
        result = calculate_vwap(df)
        typical = (h + l_ + c) / 3
        assert result.iloc[0] == pytest.approx(typical, rel=1e-6)

    def test_multi_bar_cumulative(self) -> None:
        """VWAP should converge toward the volume-weighted typical price."""
        highs = [481.0, 482.0, 483.0]
        lows = [479.0, 480.0, 481.0]
        closes = [480.0, 481.0, 482.0]
        volumes = [100_000, 200_000, 100_000]
        df = _make_df(closes, highs=highs, lows=lows, volumes=volumes)
        result = calculate_vwap(df)

        # Manual calculation
        tp = [(hi + lo + cl) / 3 for hi, lo, cl in zip(highs, lows, closes, strict=True)]
        cum_tp_vol = sum(t * v for t, v in zip(tp, volumes, strict=True))
        cum_vol = sum(volumes)
        expected = cum_tp_vol / cum_vol
        assert result.iloc[2] == pytest.approx(expected, rel=1e-6)

    def test_vwap_series_name(self) -> None:
        df = _make_df([480.0, 481.0])
        result = calculate_vwap(df)
        assert result.name == "vwap"

    def test_vwap_length_matches_input(self) -> None:
        df = _make_df([480.0] * 10)
        result = calculate_vwap(df)
        assert len(result) == 10

    def test_equal_volume_bars_vwap_is_mean_typical(self) -> None:
        """When all volumes are equal, VWAP = running average of typical prices."""
        highs = [481.0, 482.0]
        lows = [479.0, 480.0]
        closes = [480.0, 481.0]
        vol = 100_000
        df = _make_df(closes, highs=highs, lows=lows, volumes=[vol, vol])
        result = calculate_vwap(df)

        tp0 = (481.0 + 479.0 + 480.0) / 3
        tp1 = (482.0 + 480.0 + 481.0) / 3
        expected_bar1 = (tp0 + tp1) / 2
        assert result.iloc[1] == pytest.approx(expected_bar1, rel=1e-6)
