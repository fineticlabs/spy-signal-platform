"""Extended tests for ml_backtest module — edge cases and run_ml_backtest.

Coverage targets
----------------
- _score_with_purged_cv: empty test_df, too few train rows, LightGBM failure,
  x_test all NaN after dropna.
- run_ml_backtest: CSV loading path, empty features early-return, full flow
  with mocked scoring + summaries, comparison table output.
- _metrics_from_pnl_frame: empty frame, normal frame.
- _print_ml_summary: empty frame, populated frame.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.backtest.ml_backtest import (
    _metrics_from_pnl_frame,
    _print_ml_summary,
    _score_with_purged_cv,
    run_ml_backtest,
)
from src.backtest.ml_features import FEATURE_COLS

# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_features_df(
    n: int = 100,
    n_windows: int = 5,
    *,
    seed: int = 42,
) -> pd.DataFrame:
    """Build a synthetic features DataFrame matching extract_all_features output."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({col: rng.normal(0, 1, n) for col in FEATURE_COLS})
    df["ticker_encoded"] = rng.integers(0, 7, n)
    df["label"] = rng.integers(0, 2, n)
    df["entry_time"] = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
    df["pnl"] = rng.normal(50, 200, n)
    df["symbol"] = "SPY"
    df["window_idx"] = np.repeat(range(n_windows), n // n_windows)
    df["confidence"] = np.nan
    return df


# ── _score_with_purged_cv edge cases ────────────────────────────────────────


class TestScoreWithPurgedCvEdgeCases:
    """Edge-case tests for the purged walk-forward scorer."""

    def test_empty_test_df_window_skipped(self) -> None:
        """Window with no rows for a given window_idx is silently skipped."""
        df = _make_features_df(n=60, n_windows=3)
        # Remove all rows for window 1 → empty test_df for that window
        df = df[df["window_idx"] != 1].reset_index(drop=True)
        result = _score_with_purged_cv(df)
        assert "confidence" in result.columns
        assert len(result) == len(df)

    def test_too_few_train_rows_gives_nan(self) -> None:
        """Windows with < MIN_TRAIN_TRADES in training set stay NaN."""
        # Only 10 rows per window → never reaches 50 training trades
        df = _make_features_df(n=30, n_windows=3)
        result = _score_with_purged_cv(df)
        assert result["confidence"].isna().all()

    def test_x_test_all_nan_skipped(self) -> None:
        """When all feature cols in test set are NaN, window is skipped."""
        df = _make_features_df(n=200, n_windows=4)
        # Set all feature values to NaN for the last window only
        last_w = df["window_idx"].max()
        mask = df["window_idx"] == last_w
        for col in FEATURE_COLS:
            df.loc[mask, col] = np.nan
        result = _score_with_purged_cv(df)
        # Last window should still be NaN (skipped due to empty x_test after dropna)
        assert result.loc[mask, "confidence"].isna().all()

    @patch("src.backtest.ml_backtest.lgb.LGBMClassifier")
    def test_lgbm_fit_failure_caught(self, mock_cls: MagicMock) -> None:
        """If LightGBM.fit raises, confidence stays NaN for that window."""
        instance = MagicMock()
        instance.fit.side_effect = ValueError("singular matrix")
        mock_cls.return_value = instance

        df = _make_features_df(n=200, n_windows=4)
        result = _score_with_purged_cv(df)
        # All windows should be NaN since fit always fails
        assert result["confidence"].isna().all()

    def test_single_window_all_nan(self) -> None:
        """A single-window DataFrame has no training data → all NaN."""
        df = _make_features_df(n=50, n_windows=1)
        result = _score_with_purged_cv(df)
        assert result["confidence"].isna().all()

    def test_returns_same_length(self) -> None:
        """Output DataFrame has same number of rows as input."""
        df = _make_features_df(n=100, n_windows=5)
        result = _score_with_purged_cv(df)
        assert len(result) == 100

    def test_confidence_column_added(self) -> None:
        """Output always has a 'confidence' column."""
        df = _make_features_df(n=20, n_windows=2)
        result = _score_with_purged_cv(df)
        assert "confidence" in result.columns


# ── _metrics_from_pnl_frame ─────────────────────────────────────────────────


class TestMetricsFromPnlFrame:
    """Tests for the metrics helper."""

    def test_empty_frame_returns_metrics(self) -> None:
        result = _metrics_from_pnl_frame(pd.DataFrame())
        assert isinstance(result, dict)
        assert result.get("total_trades", -1) == 0

    def test_populated_frame_returns_correct_total(self) -> None:
        df = _make_features_df(n=50, n_windows=1)
        result = _metrics_from_pnl_frame(df)
        assert result["total_trades"] == 50

    def test_net_profit_is_sum_of_pnl(self) -> None:
        df = pd.DataFrame(
            {
                "pnl": [100.0, -50.0, 200.0],
                "entry_time": pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC"),
            }
        )
        result = _metrics_from_pnl_frame(df)
        assert abs(float(result["net_profit"]) - 250.0) < 0.01


# ── _print_ml_summary ───────────────────────────────────────────────────────


class TestPrintMlSummary:
    """Tests for the summary printer."""

    def test_empty_frame_prints_zero_trades(self, capsys: pytest.CaptureFixture[str]) -> None:
        _print_ml_summary("Test", pd.DataFrame())
        captured = capsys.readouterr()
        assert "0 trades" in captured.out

    def test_populated_frame_prints_stats(self, capsys: pytest.CaptureFixture[str]) -> None:
        df = _make_features_df(n=50, n_windows=1)
        _print_ml_summary("Baseline", df)
        captured = capsys.readouterr()
        assert "Baseline" in captured.out
        assert "Trades:" in captured.out
        assert "Win rate:" in captured.out


# ── run_ml_backtest ──────────────────────────────────────────────────────────


class TestRunMlBacktest:
    """Integration tests for the main entry point (heavily mocked)."""

    def _make_scored_df(self) -> pd.DataFrame:
        """Features DF with confidence pre-populated."""
        df = _make_features_df(n=100, n_windows=5)
        rng = np.random.default_rng(99)
        df["confidence"] = rng.uniform(0.3, 0.95, len(df))
        return df

    @patch("src.backtest.ml_backtest.save_equity_curve")
    @patch("src.backtest.ml_backtest._print_ml_summary")
    @patch("src.backtest.ml_backtest._score_with_purged_cv")
    @patch("src.backtest.ml_backtest.extract_all_features")
    @patch("src.backtest.ml_backtest.BarDatabase")
    def test_full_flow_extract_features(
        self,
        mock_db_cls: MagicMock,
        mock_extract: MagicMock,
        mock_score: MagicMock,
        mock_summary: MagicMock,
        mock_save_eq: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Happy path: extract features → score → print summaries → save curve."""
        scored = self._make_scored_df()
        mock_extract.return_value = scored.drop(columns=["confidence"])
        mock_score.return_value = scored
        mock_db = MagicMock()
        mock_db_cls.return_value = mock_db

        run_ml_backtest(
            symbols=["SPY"],
            features_csv=None,
            out_dir=tmp_path,
        )

        mock_db.connect.assert_called_once()
        mock_db.close.assert_called_once()
        mock_extract.assert_called_once()
        mock_score.assert_called_once()
        assert mock_summary.call_count == 3  # Baseline, >60%, >70%
        mock_save_eq.assert_called_once()

    @patch("src.backtest.ml_backtest.save_equity_curve")
    @patch("src.backtest.ml_backtest._print_ml_summary")
    @patch("src.backtest.ml_backtest._score_with_purged_cv")
    def test_load_from_csv(
        self,
        mock_score: MagicMock,
        mock_summary: MagicMock,
        mock_save_eq: MagicMock,
        tmp_path: Path,
    ) -> None:
        """When features_csv exists, loads from file instead of extracting."""
        scored = self._make_scored_df()
        csv_path = tmp_path / "features.csv"
        scored.to_csv(csv_path, index=False)

        mock_score.return_value = scored

        run_ml_backtest(
            symbols=["SPY"],
            features_csv=csv_path,
            out_dir=tmp_path,
        )

        mock_score.assert_called_once()
        mock_save_eq.assert_called_once()

    @patch("src.backtest.ml_backtest.extract_all_features")
    @patch("src.backtest.ml_backtest.BarDatabase")
    def test_empty_features_returns_early(
        self,
        mock_db_cls: MagicMock,
        mock_extract: MagicMock,
        capsys: pytest.CaptureFixture[str],
        tmp_path: Path,
    ) -> None:
        """When extract_all_features returns empty DF, prints message and returns."""
        mock_extract.return_value = pd.DataFrame()
        mock_db = MagicMock()
        mock_db_cls.return_value = mock_db

        run_ml_backtest(
            symbols=["SPY"],
            features_csv=None,
            out_dir=tmp_path,
        )

        captured = capsys.readouterr()
        assert "No features extracted" in captured.out

    @patch("src.backtest.ml_backtest.save_equity_curve")
    @patch("src.backtest.ml_backtest._score_with_purged_cv")
    @patch("src.backtest.ml_backtest.extract_all_features")
    @patch("src.backtest.ml_backtest.BarDatabase")
    def test_comparison_table_printed(
        self,
        mock_db_cls: MagicMock,
        mock_extract: MagicMock,
        mock_score: MagicMock,
        mock_save_eq: MagicMock,
        capsys: pytest.CaptureFixture[str],
        tmp_path: Path,
    ) -> None:
        """Comparison table is printed to stdout."""
        scored = self._make_scored_df()
        mock_extract.return_value = scored.drop(columns=["confidence"])
        mock_score.return_value = scored
        mock_db = MagicMock()
        mock_db_cls.return_value = mock_db

        run_ml_backtest(
            symbols=["SPY"],
            features_csv=None,
            out_dir=tmp_path,
        )

        captured = capsys.readouterr()
        assert "Three-Way Comparison" in captured.out
        assert "No ML" in captured.out

    @patch("src.backtest.ml_backtest.save_equity_curve")
    @patch("src.backtest.ml_backtest._score_with_purged_cv")
    @patch("src.backtest.ml_backtest.extract_all_features")
    @patch("src.backtest.ml_backtest.BarDatabase")
    def test_equity_curve_saved_to_out_dir(
        self,
        mock_db_cls: MagicMock,
        mock_extract: MagicMock,
        mock_score: MagicMock,
        mock_save_eq: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Equity curve PNG saved to the specified output directory."""
        scored = self._make_scored_df()
        mock_extract.return_value = scored.drop(columns=["confidence"])
        mock_score.return_value = scored
        mock_db = MagicMock()
        mock_db_cls.return_value = mock_db

        run_ml_backtest(
            symbols=["SPY"],
            features_csv=None,
            out_dir=tmp_path,
        )

        call_args = mock_save_eq.call_args
        curve_path = call_args[0][1]
        assert str(curve_path).startswith(str(tmp_path))

    @patch("src.backtest.ml_backtest.save_equity_curve")
    @patch("src.backtest.ml_backtest._score_with_purged_cv")
    @patch("src.backtest.ml_backtest.extract_all_features")
    @patch("src.backtest.ml_backtest.BarDatabase")
    def test_features_cached_to_csv_when_path_given(
        self,
        mock_db_cls: MagicMock,
        mock_extract: MagicMock,
        mock_score: MagicMock,
        mock_save_eq: MagicMock,
        tmp_path: Path,
    ) -> None:
        """When features_csv is given but doesn't exist, features are cached."""
        features = _make_features_df(n=100, n_windows=5)
        scored = features.copy()
        rng = np.random.default_rng(99)
        scored["confidence"] = rng.uniform(0.3, 0.95, len(scored))

        mock_extract.return_value = features
        mock_score.return_value = scored
        mock_db = MagicMock()
        mock_db_cls.return_value = mock_db

        csv_path = tmp_path / "sub" / "cached.csv"
        run_ml_backtest(
            symbols=["SPY"],
            features_csv=csv_path,
            out_dir=tmp_path,
        )

        assert csv_path.exists()

    @patch("src.backtest.ml_backtest.save_equity_curve")
    @patch("src.backtest.ml_backtest._score_with_purged_cv")
    @patch("src.backtest.ml_backtest.extract_all_features")
    @patch("src.backtest.ml_backtest.BarDatabase")
    def test_all_low_confidence_produces_empty_ml_sets(
        self,
        mock_db_cls: MagicMock,
        mock_extract: MagicMock,
        mock_score: MagicMock,
        mock_save_eq: MagicMock,
        capsys: pytest.CaptureFixture[str],
        tmp_path: Path,
    ) -> None:
        """When all confidence < 0.60, ML filtered sets are empty."""
        scored = _make_features_df(n=100, n_windows=5)
        scored["confidence"] = 0.3  # All below 60%

        mock_extract.return_value = scored.drop(columns=["confidence"])
        mock_score.return_value = scored
        mock_db = MagicMock()
        mock_db_cls.return_value = mock_db

        run_ml_backtest(
            symbols=["SPY"],
            features_csv=None,
            out_dir=tmp_path,
        )

        captured = capsys.readouterr()
        # The comparison table should still print
        assert "Three-Way Comparison" in captured.out
