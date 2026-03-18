"""Tests for src.backtest.ml_backtest module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.backtest.ml_backtest import (
    _metrics_from_pnl_frame,
    _print_ml_summary,
    _score_with_purged_cv,
)
from src.backtest.ml_features import FEATURE_COLS

# ── helpers ──────────────────────────────────────────────────────────────────


def _make_features_df(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """Build a minimal features DataFrame matching expected schema."""
    rng = np.random.default_rng(seed)
    entry_times = pd.date_range("2024-01-02", periods=n, freq="4h", tz="UTC")

    data: dict[str, object] = {}
    for col in FEATURE_COLS:
        if col == "ticker_encoded":
            data[col] = rng.integers(0, 5, n).astype(int)
        elif col == "direction":
            data[col] = rng.choice([0, 1], n).astype(int)
        elif col == "day_of_week":
            data[col] = rng.integers(0, 5, n)
        else:
            data[col] = rng.normal(0, 1, n)

    data["label"] = rng.choice([0, 1], n)
    data["entry_time"] = entry_times
    data["window_idx"] = np.repeat(np.arange(n // 10), 10)[:n]
    data["pnl"] = rng.normal(50, 200, n)
    data["symbol"] = "SPY"

    return pd.DataFrame(data)


def _make_pnl_df(n: int = 50, seed: int = 42) -> pd.DataFrame:
    """Minimal DataFrame with pnl and entry_time columns."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "pnl": rng.normal(50, 200, n),
            "entry_time": pd.date_range("2024-01-02", periods=n, freq="4h", tz="UTC"),
        }
    )


# ── _metrics_from_pnl_frame ─────────────────────────────────────────────────


class TestMetricsFromPnlFrame:
    def test_empty_df_zero_trades(self) -> None:
        result = _metrics_from_pnl_frame(pd.DataFrame())
        assert result.get("total_trades", 0) == 0

    def test_valid_df_total_trades(self) -> None:
        df = _make_pnl_df(n=50)
        result = _metrics_from_pnl_frame(df)
        assert result["total_trades"] == 50

    def test_valid_df_win_rate_range(self) -> None:
        df = _make_pnl_df(n=50)
        result = _metrics_from_pnl_frame(df)
        wr = float(result["win_rate"])
        assert 0.0 <= wr <= 1.0

    def test_all_winners(self) -> None:
        df = pd.DataFrame(
            {
                "pnl": [100.0, 50.0, 75.0],
                "entry_time": pd.date_range("2024-01-02", periods=3, freq="4h", tz="UTC"),
            }
        )
        result = _metrics_from_pnl_frame(df)
        assert float(result["win_rate"]) == pytest.approx(1.0)

    def test_all_losers(self) -> None:
        df = pd.DataFrame(
            {
                "pnl": [-100.0, -50.0, -75.0],
                "entry_time": pd.date_range("2024-01-02", periods=3, freq="4h", tz="UTC"),
            }
        )
        result = _metrics_from_pnl_frame(df)
        assert float(result["win_rate"]) == pytest.approx(0.0)


# ── _print_ml_summary ───────────────────────────────────────────────────────


class TestPrintMlSummary:
    def test_empty_df_prints_zero_trades(self, capsys: pytest.CaptureFixture[str]) -> None:
        _print_ml_summary("Test", pd.DataFrame())
        captured = capsys.readouterr()
        assert "0 trades" in captured.out

    def test_valid_df_prints_stats(self, capsys: pytest.CaptureFixture[str]) -> None:
        df = _make_pnl_df(n=50)
        _print_ml_summary("Baseline", df)
        captured = capsys.readouterr()
        assert "Baseline" in captured.out
        assert "Trades:" in captured.out
        assert "Win rate:" in captured.out
        assert "Profit factor:" in captured.out

    def test_label_appears_in_output(self, capsys: pytest.CaptureFixture[str]) -> None:
        df = _make_pnl_df(n=10)
        _print_ml_summary("ML >60%", df)
        captured = capsys.readouterr()
        assert "ML >60%" in captured.out


# ── _score_with_purged_cv ────────────────────────────────────────────────────


class TestScoreWithPurgedCv:
    def test_adds_confidence_column(self) -> None:
        """Mock LightGBM so the scorer adds a confidence column."""
        df = _make_features_df(n=100)

        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.column_stack([np.zeros(10), np.full(10, 0.75)])

        with patch("src.backtest.ml_backtest.lgb.LGBMClassifier", return_value=mock_model):
            result = _score_with_purged_cv(df)

        assert "confidence" in result.columns

    def test_early_windows_get_nan(self) -> None:
        """Windows with < MIN_TRAIN_TRADES training rows get NaN confidence."""
        # 3 windows of 10 rows each = 30 total; MIN_TRAIN_TRADES=50
        df = _make_features_df(n=30)
        df["window_idx"] = np.repeat([0, 1, 2], 10)

        with patch("src.backtest.ml_backtest.lgb.LGBMClassifier") as mock_cls:
            result = _score_with_purged_cv(df)
            # With only 30 rows total, no window has 50 prior training rows
            mock_cls.assert_not_called()

        assert result["confidence"].isna().all()

    def test_later_windows_scored_when_enough_training(self) -> None:
        """With enough training data, later windows get scored."""
        # 120 rows across 12 windows of 10 — window 6+ should have >= 50 train rows
        df = _make_features_df(n=120, seed=99)
        df["window_idx"] = np.repeat(np.arange(12), 10)

        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.column_stack([np.zeros(10), np.full(10, 0.65)])

        with patch("src.backtest.ml_backtest.lgb.LGBMClassifier", return_value=mock_model):
            result = _score_with_purged_cv(df)

        # Some early windows should be NaN, some later should be scored
        assert result["confidence"].isna().any()
        assert (~result["confidence"].isna()).any()
