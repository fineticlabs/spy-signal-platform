"""Extended tests for src.backtest.ml_features — covers extract_features_for_symbol and extract_all_features."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from src.backtest.ml_features import (
    FEATURE_COLS,
    TICKER_LIST,
    extract_all_features,
    extract_features_for_symbol,
)

# ── helpers ──────────────────────────────────────────────────────────────────


def _make_1min_df(n_bars: int = 400) -> pd.DataFrame:
    """Synthetic 1-min OHLCV with UTC DatetimeIndex."""
    rng = np.random.default_rng(42)
    idx = pd.date_range("2024-01-15 14:30", periods=n_bars, freq="min", tz="UTC")
    base = 480.0
    close = base + np.cumsum(rng.normal(0, 0.2, n_bars))
    return pd.DataFrame(
        {
            "open": close + rng.normal(0, 0.05, n_bars),
            "high": close + np.abs(rng.normal(0.3, 0.1, n_bars)),
            "low": close - np.abs(rng.normal(0.3, 0.1, n_bars)),
            "close": close,
            "volume": rng.integers(50_000, 200_000, n_bars).astype(float),
        },
        index=idx,
    )


def _make_trades_df(n_trades: int = 3) -> pd.DataFrame:
    """Minimal _trades DataFrame as returned by run_backtest stats."""
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "EntryBar": rng.integers(10, 50, n_trades),
            "Size": rng.choice([100, -100], n_trades),
            "PnL": rng.normal(50, 100, n_trades),
            "EntryPrice": rng.normal(480, 2, n_trades),
            "ExitPrice": rng.normal(481, 2, n_trades),
        }
    )


def _make_feature_row_dict(symbol: str = "SPY", win_idx: int = 0) -> dict[str, Any]:
    """A single feature row dict matching the schema."""
    rng = np.random.default_rng(42)
    row: dict[str, Any] = {
        "symbol": symbol,
        "entry_time": pd.Timestamp("2024-01-15 14:45", tz="UTC"),
        "window_idx": win_idx,
        "direction": 1,
        "pnl": 50.0,
        "label": 1,
    }
    for col in FEATURE_COLS:
        if col not in row:
            row[col] = rng.normal(0, 1)
    return row


# ── extract_features_for_symbol ──────────────────────────────────────────────


class TestExtractFeaturesForSymbol:
    """Tests for extract_features_for_symbol (lines 304-363)."""

    @patch("src.backtest.ml_features.run_backtest")
    @patch("src.backtest.ml_features._compute_indicator_arrays")
    @patch("src.backtest.ml_features.slice_window")
    @patch("src.backtest.ml_features.make_walk_forward_windows")
    @patch("src.backtest.ml_features.load_bars")
    def test_returns_dataframe_with_trades(
        self,
        mock_load_bars: MagicMock,
        mock_windows: MagicMock,
        mock_slice: MagicMock,
        mock_indicators: MagicMock,
        mock_backtest: MagicMock,
    ) -> None:
        df_1min = _make_1min_df(400)
        mock_load_bars.return_value = df_1min

        # 1 window
        mock_windows.return_value = [MagicMock()]

        is_df = df_1min.iloc[:200]
        oos_df = df_1min.iloc[200:]
        mock_slice.return_value = (is_df, oos_df)

        n_combined = len(is_df) + len(oos_df)
        mock_indicators.return_value = {
            "atr": np.full(n_combined, 1.5),
            "avg_vol": np.full(n_combined, 90_000.0),
            "rsi14": np.full(n_combined, 55.0),
            "orb_range_pct": np.full(n_combined, 0.005),
            "orb_high": np.full(n_combined, 481.0),
            "orb_low": np.full(n_combined, 479.0),
            "ema15m": np.full(n_combined, 480.0),
            "gap_pct": np.full(n_combined, 0.1),
            "realized_vol": np.full(n_combined, 15.0),
            "adx": np.full(n_combined, 25.0),
        }

        trades_df = _make_trades_df(3)
        mock_backtest.return_value = {"_trades": trades_df}

        db = MagicMock()
        result = extract_features_for_symbol(db, "SPY")

        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert "label" in result.columns
        assert "symbol" in result.columns

    @patch("src.backtest.ml_features.load_bars")
    def test_empty_bars_returns_empty_df(self, mock_load_bars: MagicMock) -> None:
        mock_load_bars.return_value = pd.DataFrame()
        db = MagicMock()
        result = extract_features_for_symbol(db, "SPY")
        assert result.empty

    @patch("src.backtest.ml_features.make_walk_forward_windows")
    @patch("src.backtest.ml_features.load_bars")
    def test_no_windows_returns_empty_df(
        self, mock_load_bars: MagicMock, mock_windows: MagicMock
    ) -> None:
        mock_load_bars.return_value = _make_1min_df(400)
        mock_windows.return_value = []
        db = MagicMock()
        result = extract_features_for_symbol(db, "SPY")
        assert result.empty

    @patch("src.backtest.ml_features.run_backtest")
    @patch("src.backtest.ml_features._compute_indicator_arrays")
    @patch("src.backtest.ml_features.slice_window")
    @patch("src.backtest.ml_features.make_walk_forward_windows")
    @patch("src.backtest.ml_features.load_bars")
    def test_backtest_exception_skips_window(
        self,
        mock_load_bars: MagicMock,
        mock_windows: MagicMock,
        mock_slice: MagicMock,
        mock_indicators: MagicMock,
        mock_backtest: MagicMock,
    ) -> None:
        df_1min = _make_1min_df(400)
        mock_load_bars.return_value = df_1min
        mock_windows.return_value = [MagicMock()]

        is_df = df_1min.iloc[:200]
        oos_df = df_1min.iloc[200:]
        mock_slice.return_value = (is_df, oos_df)

        n_combined = len(is_df) + len(oos_df)
        mock_indicators.return_value = {
            k: np.full(n_combined, 1.0)
            for k in [
                "atr",
                "avg_vol",
                "rsi14",
                "orb_range_pct",
                "orb_high",
                "orb_low",
                "ema15m",
                "gap_pct",
                "realized_vol",
                "adx",
            ]
        }

        mock_backtest.side_effect = RuntimeError("backtest failed")

        db = MagicMock()
        result = extract_features_for_symbol(db, "SPY")
        assert result.empty

    @patch("src.backtest.ml_features.run_backtest")
    @patch("src.backtest.ml_features._compute_indicator_arrays")
    @patch("src.backtest.ml_features.slice_window")
    @patch("src.backtest.ml_features.make_walk_forward_windows")
    @patch("src.backtest.ml_features.load_bars")
    def test_empty_trades_returns_empty_df(
        self,
        mock_load_bars: MagicMock,
        mock_windows: MagicMock,
        mock_slice: MagicMock,
        mock_indicators: MagicMock,
        mock_backtest: MagicMock,
    ) -> None:
        df_1min = _make_1min_df(400)
        mock_load_bars.return_value = df_1min
        mock_windows.return_value = [MagicMock()]

        is_df = df_1min.iloc[:200]
        oos_df = df_1min.iloc[200:]
        mock_slice.return_value = (is_df, oos_df)

        n_combined = len(is_df) + len(oos_df)
        mock_indicators.return_value = {
            k: np.full(n_combined, 1.0)
            for k in [
                "atr",
                "avg_vol",
                "rsi14",
                "orb_range_pct",
                "orb_high",
                "orb_low",
                "ema15m",
                "gap_pct",
                "realized_vol",
                "adx",
            ]
        }

        mock_backtest.return_value = {"_trades": pd.DataFrame()}

        db = MagicMock()
        result = extract_features_for_symbol(db, "SPY")
        assert result.empty

    @patch("src.backtest.ml_features.run_backtest")
    @patch("src.backtest.ml_features._compute_indicator_arrays")
    @patch("src.backtest.ml_features.slice_window")
    @patch("src.backtest.ml_features.make_walk_forward_windows")
    @patch("src.backtest.ml_features.load_bars")
    def test_short_oos_window_skipped(
        self,
        mock_load_bars: MagicMock,
        mock_windows: MagicMock,
        mock_slice: MagicMock,
        mock_indicators: MagicMock,
        mock_backtest: MagicMock,
    ) -> None:
        """OOS window with < 30 bars is skipped."""
        df_1min = _make_1min_df(400)
        mock_load_bars.return_value = df_1min
        mock_windows.return_value = [MagicMock()]

        is_df = df_1min.iloc[:380]
        oos_df = df_1min.iloc[380:395]  # Only 15 bars
        mock_slice.return_value = (is_df, oos_df)

        db = MagicMock()
        result = extract_features_for_symbol(db, "SPY")
        assert result.empty
        mock_backtest.assert_not_called()


# ── extract_all_features ─────────────────────────────────────────────────────


class TestExtractAllFeatures:
    """Tests for extract_all_features (lines 423-446)."""

    @patch("src.backtest.ml_features.extract_features_for_symbol")
    def test_combines_two_symbols(self, mock_extract: MagicMock) -> None:
        rows_spy = [_make_feature_row_dict("SPY", 0), _make_feature_row_dict("SPY", 1)]
        rows_qqq = [_make_feature_row_dict("QQQ", 0)]
        # Offset QQQ entry_time so sort is deterministic
        rows_qqq[0]["entry_time"] = pd.Timestamp("2024-01-15 15:00", tz="UTC")

        mock_extract.side_effect = [
            pd.DataFrame(rows_spy),
            pd.DataFrame(rows_qqq),
        ]

        db = MagicMock()
        result = extract_all_features(db, ["SPY", "QQQ"])

        assert len(result) == 3
        assert set(result["symbol"].unique()) == {"SPY", "QQQ"}

    @patch("src.backtest.ml_features.extract_features_for_symbol")
    def test_empty_results_returns_empty_df(self, mock_extract: MagicMock) -> None:
        mock_extract.return_value = pd.DataFrame()

        db = MagicMock()
        result = extract_all_features(db, ["SPY", "QQQ"])
        assert result.empty

    @patch("src.backtest.ml_features.extract_features_for_symbol")
    def test_ticker_encoded_column_added(self, mock_extract: MagicMock) -> None:
        rows = [_make_feature_row_dict("SPY", 0)]
        mock_extract.return_value = pd.DataFrame(rows)

        db = MagicMock()
        result = extract_all_features(db, ["SPY"])

        assert "ticker_encoded" in result.columns
        # SPY -> 0
        assert result["ticker_encoded"].iloc[0] == 0

    @patch("src.backtest.ml_features.extract_features_for_symbol")
    def test_sorted_by_entry_time(self, mock_extract: MagicMock) -> None:
        rows = [
            _make_feature_row_dict("SPY", 0),
            _make_feature_row_dict("SPY", 1),
        ]
        # Make second row earlier than first
        rows[0]["entry_time"] = pd.Timestamp("2024-01-15 16:00", tz="UTC")
        rows[1]["entry_time"] = pd.Timestamp("2024-01-15 14:00", tz="UTC")

        mock_extract.return_value = pd.DataFrame(rows)

        db = MagicMock()
        result = extract_all_features(db, ["SPY"])

        times = result["entry_time"].tolist()
        assert times == sorted(times)

    @patch("src.backtest.ml_features.extract_features_for_symbol")
    def test_unknown_ticker_encoded_correctly(self, mock_extract: MagicMock) -> None:
        rows = [_make_feature_row_dict("AAPL", 0)]
        mock_extract.return_value = pd.DataFrame(rows)

        db = MagicMock()
        result = extract_all_features(db, ["AAPL"])

        # Unknown ticker -> len(TICKER_LIST)
        assert result["ticker_encoded"].iloc[0] == len(TICKER_LIST)

    @patch("src.backtest.ml_features.extract_features_for_symbol")
    def test_one_symbol_empty_other_has_data(self, mock_extract: MagicMock) -> None:
        """When one symbol returns empty, result has only the other symbol."""
        rows = [_make_feature_row_dict("QQQ", 0)]

        mock_extract.side_effect = [
            pd.DataFrame(),  # SPY empty
            pd.DataFrame(rows),  # QQQ has data
        ]

        db = MagicMock()
        result = extract_all_features(db, ["SPY", "QQQ"])

        assert len(result) == 1
        assert result["symbol"].iloc[0] == "QQQ"

    @patch("src.backtest.ml_features.extract_features_for_symbol")
    def test_index_reset_after_concat(self, mock_extract: MagicMock) -> None:
        rows_spy = [_make_feature_row_dict("SPY", 0)]
        rows_qqq = [_make_feature_row_dict("QQQ", 0)]
        rows_qqq[0]["entry_time"] = pd.Timestamp("2024-01-15 15:00", tz="UTC")

        mock_extract.side_effect = [
            pd.DataFrame(rows_spy),
            pd.DataFrame(rows_qqq),
        ]

        db = MagicMock()
        result = extract_all_features(db, ["SPY", "QQQ"])

        # Index should be 0, 1 (reset)
        assert list(result.index) == [0, 1]
