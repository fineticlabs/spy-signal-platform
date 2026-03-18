"""Tests for src.backtest.ml_features — ML feature extraction for ORB trades."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.backtest.ml_features import (
    FEATURE_COLS,
    TICKER_LIST,
    _build_feature_row,
    _count_consecutive,
    encode_ticker,
)

# ── encode_ticker ────────────────────────────────────────────────────────────


class TestEncodeTicker:
    def test_spy_maps_to_zero(self) -> None:
        assert encode_ticker("SPY") == 0

    def test_qqq_maps_to_one(self) -> None:
        assert encode_ticker("QQQ") == 1

    def test_meta_maps_to_five(self) -> None:
        assert encode_ticker("META") == 5

    def test_amzn_maps_to_six(self) -> None:
        assert encode_ticker("AMZN") == 6

    def test_unknown_ticker_returns_len_ticker_list(self) -> None:
        assert encode_ticker("AAPL") == len(TICKER_LIST)
        assert encode_ticker("GOOG") == len(TICKER_LIST)


# ── constants ────────────────────────────────────────────────────────────────


class TestConstants:
    def test_feature_cols_has_13_features(self) -> None:
        assert len(FEATURE_COLS) == 13

    def test_ticker_list_has_7_tickers(self) -> None:
        assert len(TICKER_LIST) == 7

    def test_feature_cols_contains_expected_names(self) -> None:
        expected = {
            "orb_range_pct",
            "gap_pct",
            "volume_ratio",
            "atr_pct",
            "rsi_14",
            "ema_distance_pct",
            "time_minutes",
            "day_of_week",
            "vix_proxy",
            "adx_value",
            "consecutive_candles",
            "ticker_encoded",
            "direction",
        }
        assert set(FEATURE_COLS) == expected


# ── _count_consecutive ───────────────────────────────────────────────────────


class TestCountConsecutive:
    def test_three_closes_above_orb_high(self) -> None:
        close_arr = np.array([100.0, 101.0, 102.0, 103.0])
        orb_high_arr = np.array([99.0, 99.0, 99.0, 99.0])
        orb_low_arr = np.array([95.0, 95.0, 95.0, 95.0])
        result = _count_consecutive(3, close_arr, orb_high_arr, orb_low_arr, direction=1)
        # Bars 3, 2, 1, 0 all above 99 -> 4
        assert result == 4

    def test_exactly_three_consecutive(self) -> None:
        close_arr = np.array([98.0, 101.0, 102.0, 103.0])
        orb_high_arr = np.array([99.0, 99.0, 99.0, 99.0])
        orb_low_arr = np.array([95.0, 95.0, 95.0, 95.0])
        # Bar 0 is below ORB high, so counting from bar 3 backwards: 3, 2, 1 -> 3
        result = _count_consecutive(3, close_arr, orb_high_arr, orb_low_arr, direction=1)
        assert result == 3

    def test_first_bar_below_orb_returns_zero(self) -> None:
        close_arr = np.array([98.0, 101.0, 102.0, 97.0])
        orb_high_arr = np.array([99.0, 99.0, 99.0, 99.0])
        orb_low_arr = np.array([95.0, 95.0, 95.0, 95.0])
        # bar_idx=3, close=97 < orb_high=99 -> 0
        result = _count_consecutive(3, close_arr, orb_high_arr, orb_low_arr, direction=1)
        assert result == 0

    def test_nan_in_close_stops_counting(self) -> None:
        close_arr = np.array([101.0, np.nan, 102.0, 103.0])
        orb_high_arr = np.array([99.0, 99.0, 99.0, 99.0])
        orb_low_arr = np.array([95.0, 95.0, 95.0, 95.0])
        # Bar 3=103>99 (1), bar 2=102>99 (2), bar 1=NaN -> stop
        result = _count_consecutive(3, close_arr, orb_high_arr, orb_low_arr, direction=1)
        assert result == 2

    def test_short_direction_counts_below_orb_low(self) -> None:
        close_arr = np.array([90.0, 93.0, 94.0, 93.5])
        orb_high_arr = np.array([99.0, 99.0, 99.0, 99.0])
        orb_low_arr = np.array([95.0, 95.0, 95.0, 95.0])
        # All < 95 -> 4 consecutive
        result = _count_consecutive(3, close_arr, orb_high_arr, orb_low_arr, direction=0)
        assert result == 4

    def test_nan_in_orb_high_stops_counting(self) -> None:
        close_arr = np.array([101.0, 102.0, 103.0])
        orb_high_arr = np.array([np.nan, 99.0, 99.0])
        orb_low_arr = np.array([95.0, 95.0, 95.0])
        # Bar 2=103>99 (1), bar 1=102>99 (2), bar 0 orb_high=NaN -> stop
        result = _count_consecutive(2, close_arr, orb_high_arr, orb_low_arr, direction=1)
        assert result == 2


# ── _build_feature_row ───────────────────────────────────────────────────────


def _make_combined_df_and_arrays(n: int = 10) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Helper: build a minimal combined_df and arrays dict for _build_feature_row."""
    idx = pd.date_range("2024-01-15 14:35", periods=n, freq="min", tz="UTC")
    df = pd.DataFrame(
        {
            "open": np.full(n, 480.0),
            "high": np.full(n, 481.0),
            "low": np.full(n, 479.0),
            "close": np.full(n, 480.5),
            "volume": np.full(n, 100_000.0),
        },
        index=idx,
    )
    arrays = {
        "atr": np.full(n, 1.5),
        "avg_vol": np.full(n, 90_000.0),
        "rsi14": np.full(n, 55.0),
        "orb_range_pct": np.full(n, 0.005),
        "orb_high": np.full(n, 481.0),
        "orb_low": np.full(n, 479.0),
        "ema15m": np.full(n, 480.0),
        "gap_pct": np.full(n, 0.1),
        "realized_vol": np.full(n, 15.0),
        "adx": np.full(n, 25.0),
    }
    return df, arrays


class TestBuildFeatureRow:
    def test_out_of_bounds_index_returns_none(self) -> None:
        df, arrays = _make_combined_df_and_arrays(10)
        assert _build_feature_row(-1, df, arrays, "SPY", 1, 50.0, 0) is None
        assert _build_feature_row(10, df, arrays, "SPY", 1, 50.0, 0) is None

    def test_zero_close_returns_none(self) -> None:
        df, arrays = _make_combined_df_and_arrays(10)
        df.iloc[5, df.columns.get_loc("close")] = 0.0
        result = _build_feature_row(5, df, arrays, "SPY", 1, 50.0, 0)
        assert result is None

    def test_valid_row_has_all_expected_keys(self) -> None:
        df, arrays = _make_combined_df_and_arrays(10)
        result = _build_feature_row(5, df, arrays, "SPY", 1, 50.0, 0)
        assert result is not None
        expected_keys = {
            "symbol",
            "entry_time",
            "window_idx",
            "direction",
            "orb_range_pct",
            "gap_pct",
            "volume_ratio",
            "atr_pct",
            "rsi_14",
            "ema_distance_pct",
            "time_minutes",
            "day_of_week",
            "vix_proxy",
            "adx_value",
            "consecutive_candles",
            "pnl",
            "label",
        }
        assert set(result.keys()) == expected_keys

    def test_label_is_one_for_positive_pnl(self) -> None:
        df, arrays = _make_combined_df_and_arrays(10)
        result = _build_feature_row(5, df, arrays, "SPY", 1, 100.0, 0)
        assert result is not None
        assert result["label"] == 1

    def test_label_is_zero_for_negative_pnl(self) -> None:
        df, arrays = _make_combined_df_and_arrays(10)
        result = _build_feature_row(5, df, arrays, "SPY", 1, -50.0, 0)
        assert result is not None
        assert result["label"] == 0

    def test_direction_preserved(self) -> None:
        df, arrays = _make_combined_df_and_arrays(10)
        result = _build_feature_row(5, df, arrays, "SPY", 0, 50.0, 0)
        assert result is not None
        assert result["direction"] == 0


# ── _compute_indicator_arrays ────────────────────────────────────────────────


class TestComputeIndicatorArrays:
    """Tests for _compute_indicator_arrays with TA-Lib mocked."""

    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:
        """30-bar OHLCV DataFrame for indicator tests."""
        n = 30
        idx = pd.date_range("2024-01-15 14:30", periods=n, freq="min", tz="UTC")
        rng = np.random.default_rng(42)
        base = 480.0
        close = base + np.cumsum(rng.normal(0, 0.2, n))
        return pd.DataFrame(
            {
                "open": close + rng.normal(0, 0.05, n),
                "high": close + np.abs(rng.normal(0.3, 0.1, n)),
                "low": close - np.abs(rng.normal(0.3, 0.1, n)),
                "close": close,
                "volume": rng.integers(50_000, 200_000, n).astype(float),
            },
            index=idx,
        )

    @patch("src.backtest.ml_features._compute_daily_adx")
    @patch("src.backtest.ml_features._compute_daily_realized_vol")
    @patch("src.backtest.ml_features._compute_gap_array")
    @patch("src.backtest.ml_features._compute_15m_ema")
    @patch("src.backtest.ml_features._compute_orb_arrays")
    @patch("src.backtest.ml_features.talib")
    def test_output_length_matches_input(
        self,
        mock_talib: object,
        mock_orb: object,
        mock_ema: object,
        mock_gap: object,
        mock_rvol: object,
        mock_adx: object,
        sample_df: pd.DataFrame,
    ) -> None:
        from src.backtest.ml_features import _compute_indicator_arrays

        n = len(sample_df)
        mock_talib.ATR.return_value = np.full(n, 1.0)  # type: ignore[union-attr]
        mock_talib.RSI.return_value = np.full(n, 50.0)  # type: ignore[union-attr]
        mock_orb.return_value = (np.full(n, 481.0), np.full(n, 479.0))  # type: ignore[union-attr]
        mock_ema.return_value = np.full(n, 480.0)  # type: ignore[union-attr]
        mock_gap.return_value = np.full(n, 0.1)  # type: ignore[union-attr]
        mock_rvol.return_value = np.full(n, 15.0)  # type: ignore[union-attr]
        mock_adx.return_value = np.full(n, 25.0)  # type: ignore[union-attr]

        arrays = _compute_indicator_arrays(sample_df)
        for key, arr in arrays.items():
            assert len(arr) == n, f"{key} length mismatch"

    @patch("src.backtest.ml_features._compute_daily_adx")
    @patch("src.backtest.ml_features._compute_daily_realized_vol")
    @patch("src.backtest.ml_features._compute_gap_array")
    @patch("src.backtest.ml_features._compute_15m_ema")
    @patch("src.backtest.ml_features._compute_orb_arrays")
    @patch("src.backtest.ml_features.talib")
    def test_atr_shifted_by_one_first_is_nan(
        self,
        mock_talib: object,
        mock_orb: object,
        mock_ema: object,
        mock_gap: object,
        mock_rvol: object,
        mock_adx: object,
        sample_df: pd.DataFrame,
    ) -> None:
        from src.backtest.ml_features import _compute_indicator_arrays

        n = len(sample_df)
        atr_raw = np.arange(1.0, n + 1.0)
        mock_talib.ATR.return_value = atr_raw  # type: ignore[union-attr]
        mock_talib.RSI.return_value = np.full(n, 50.0)  # type: ignore[union-attr]
        mock_orb.return_value = (np.full(n, 481.0), np.full(n, 479.0))  # type: ignore[union-attr]
        mock_ema.return_value = np.full(n, 480.0)  # type: ignore[union-attr]
        mock_gap.return_value = np.full(n, 0.1)  # type: ignore[union-attr]
        mock_rvol.return_value = np.full(n, 15.0)  # type: ignore[union-attr]
        mock_adx.return_value = np.full(n, 25.0)  # type: ignore[union-attr]

        arrays = _compute_indicator_arrays(sample_df)
        assert np.isnan(arrays["atr"][0]), "ATR[0] should be NaN after shift"
        # ATR[1] should be original ATR[0] = 1.0
        assert arrays["atr"][1] == pytest.approx(1.0)

    @patch("src.backtest.ml_features._compute_daily_adx")
    @patch("src.backtest.ml_features._compute_daily_realized_vol")
    @patch("src.backtest.ml_features._compute_gap_array")
    @patch("src.backtest.ml_features._compute_15m_ema")
    @patch("src.backtest.ml_features._compute_orb_arrays")
    @patch("src.backtest.ml_features.talib")
    def test_rsi_shifted_by_one(
        self,
        mock_talib: object,
        mock_orb: object,
        mock_ema: object,
        mock_gap: object,
        mock_rvol: object,
        mock_adx: object,
        sample_df: pd.DataFrame,
    ) -> None:
        from src.backtest.ml_features import _compute_indicator_arrays

        n = len(sample_df)
        rsi_raw = np.linspace(30.0, 70.0, n)
        mock_talib.ATR.return_value = np.full(n, 1.0)  # type: ignore[union-attr]
        mock_talib.RSI.return_value = rsi_raw  # type: ignore[union-attr]
        mock_orb.return_value = (np.full(n, 481.0), np.full(n, 479.0))  # type: ignore[union-attr]
        mock_ema.return_value = np.full(n, 480.0)  # type: ignore[union-attr]
        mock_gap.return_value = np.full(n, 0.1)  # type: ignore[union-attr]
        mock_rvol.return_value = np.full(n, 15.0)  # type: ignore[union-attr]
        mock_adx.return_value = np.full(n, 25.0)  # type: ignore[union-attr]

        arrays = _compute_indicator_arrays(sample_df)
        assert np.isnan(arrays["rsi14"][0]), "RSI[0] should be NaN after shift"
        # RSI[1] should be original RSI[0] = 30.0
        assert arrays["rsi14"][1] == pytest.approx(30.0)
