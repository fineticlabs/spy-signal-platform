"""Extended tests for src.backtest.ml_scorer — covers uncovered lines 76-306."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.backtest.ml_features import FEATURE_COLS

# ── synthetic data helper ────────────────────────────────────────────────────


def _make_features_df(
    n: int = 200,
    seed: int = 42,
    label_values: np.ndarray | None = None,
) -> pd.DataFrame:
    """Build a synthetic features DataFrame matching the training schema."""
    rng = np.random.default_rng(seed)
    data: dict[str, Any] = {}
    for col in FEATURE_COLS:
        if col == "ticker_encoded":
            data[col] = rng.integers(0, 7, n)
        elif col == "direction":
            data[col] = rng.choice([0, 1], n)
        elif col == "day_of_week":
            data[col] = rng.integers(0, 5, n)
        else:
            data[col] = rng.normal(0, 1, n)

    if label_values is not None:
        data["label"] = label_values
    else:
        data["label"] = rng.integers(0, 2, n)
    data["entry_time"] = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
    data["pnl"] = rng.normal(50, 200, n)
    data["symbol"] = "SPY"
    n_windows = max((n + 19) // 20, 1)
    data["window_idx"] = np.repeat(range(n_windows), 20)[:n]
    return pd.DataFrame(data)


# ── _purged_cv_score ─────────────────────────────────────────────────────────


class TestPurgedCvScore:
    """Tests for _purged_cv_score (lines 76-114)."""

    def test_returns_float_between_zero_and_one(self) -> None:
        from src.backtest.ml_scorer import _purged_cv_score

        df = _make_features_df(n=200, seed=42)
        x_data = df[FEATURE_COLS].copy()
        x_data["ticker_encoded"] = x_data["ticker_encoded"].astype(int)
        y = df["label"].astype(int)
        entry_times = df["entry_time"]

        params = {"n_estimators": 50, "max_depth": 3, "learning_rate": 0.1}
        score = _purged_cv_score(x_data, y, entry_times, params)

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_all_same_label_returns_fallback(self) -> None:
        """When all labels are identical, degenerate folds are skipped -> 0.5."""
        from src.backtest.ml_scorer import _purged_cv_score

        n = 200
        df = _make_features_df(n=n, seed=42, label_values=np.ones(n, dtype=int))
        x_data = df[FEATURE_COLS].copy()
        x_data["ticker_encoded"] = x_data["ticker_encoded"].astype(int)
        y = df["label"].astype(int)
        entry_times = df["entry_time"]

        params = {"n_estimators": 50, "max_depth": 3, "learning_rate": 0.1}
        score = _purged_cv_score(x_data, y, entry_times, params)

        assert score == pytest.approx(0.5)

    def test_too_few_samples_returns_fallback(self) -> None:
        """With very few samples, all folds are skipped -> 0.5."""
        from src.backtest.ml_scorer import _purged_cv_score

        n = 15  # too few for 5-fold + purge
        df = _make_features_df(n=n, seed=42)
        x_data = df[FEATURE_COLS].copy()
        x_data["ticker_encoded"] = x_data["ticker_encoded"].astype(int)
        y = df["label"].astype(int)
        entry_times = df["entry_time"]

        params = {"n_estimators": 50, "max_depth": 3, "learning_rate": 0.1}
        score = _purged_cv_score(x_data, y, entry_times, params)

        assert score == pytest.approx(0.5)

    def test_purge_gap_reduces_train_set(self) -> None:
        """Verify purge works by using tightly-spaced timestamps."""
        from src.backtest.ml_scorer import _purged_cv_score

        n = 200
        # Timestamps all within 2 days -- purge of 5 days should remove most train data
        df = _make_features_df(n=n, seed=42)
        df["entry_time"] = pd.date_range("2024-01-01", periods=n, freq="10min", tz="UTC")
        x_data = df[FEATURE_COLS].copy()
        x_data["ticker_encoded"] = x_data["ticker_encoded"].astype(int)
        y = df["label"].astype(int)
        entry_times = df["entry_time"]

        params = {"n_estimators": 50, "max_depth": 3, "learning_rate": 0.1}
        score = _purged_cv_score(x_data, y, entry_times, params)

        # With tight timestamps, purge removes most train rows -> fallback 0.5
        assert score == pytest.approx(0.5)


# ── _make_objective ──────────────────────────────────────────────────────────


class TestMakeObjective:
    """Tests for _make_objective (lines 136-150)."""

    def test_returns_callable(self) -> None:
        from src.backtest.ml_scorer import _make_objective

        df = _make_features_df(n=200)
        x_data = df[FEATURE_COLS].copy()
        x_data["ticker_encoded"] = x_data["ticker_encoded"].astype(int)
        y = df["label"].astype(int)
        entry_times = df["entry_time"]

        objective = _make_objective(x_data, y, entry_times)
        assert callable(objective)

    def test_objective_with_mock_trial_returns_float(self) -> None:
        from src.backtest.ml_scorer import _make_objective

        df = _make_features_df(n=200)
        x_data = df[FEATURE_COLS].copy()
        x_data["ticker_encoded"] = x_data["ticker_encoded"].astype(int)
        y = df["label"].astype(int)
        entry_times = df["entry_time"]

        objective = _make_objective(x_data, y, entry_times)

        trial = MagicMock()
        trial.suggest_int.side_effect = lambda name, lo, hi: {
            "n_estimators": 50,
            "max_depth": 3,
            "num_leaves": 31,
            "min_child_samples": 10,
        }[name]
        trial.suggest_float.side_effect = lambda name, lo, hi, **kw: {
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 1.0,
            "reg_lambda": 1.0,
        }[name]

        result = objective(trial)
        assert isinstance(result, float)

    def test_objective_calls_suggest_methods(self) -> None:
        from src.backtest.ml_scorer import _make_objective

        df = _make_features_df(n=200)
        x_data = df[FEATURE_COLS].copy()
        x_data["ticker_encoded"] = x_data["ticker_encoded"].astype(int)
        y = df["label"].astype(int)
        entry_times = df["entry_time"]

        objective = _make_objective(x_data, y, entry_times)

        trial = MagicMock()
        trial.suggest_int.side_effect = lambda name, lo, hi: {
            "n_estimators": 50,
            "max_depth": 3,
            "num_leaves": 31,
            "min_child_samples": 10,
        }[name]
        trial.suggest_float.side_effect = lambda name, lo, hi, **kw: {
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 1.0,
            "reg_lambda": 1.0,
        }[name]

        objective(trial)

        assert trial.suggest_int.call_count == 4
        assert trial.suggest_float.call_count == 5


# ── tune_and_train ───────────────────────────────────────────────────────────


class TestTuneAndTrain:
    """Tests for tune_and_train (lines 174-238)."""

    def test_empty_dataframe_raises(self) -> None:
        from src.backtest.ml_scorer import tune_and_train

        with pytest.raises(ValueError, match="empty"):
            tune_and_train(pd.DataFrame())

    def test_fewer_than_50_rows_raises(self) -> None:
        from src.backtest.ml_scorer import tune_and_train

        df = _make_features_df(n=30)
        with pytest.raises(ValueError, match="need"):
            tune_and_train(df)

    def test_rows_with_nans_below_threshold_raises(self) -> None:
        """If NaN-dropping leaves < 50 rows, should raise."""
        from src.backtest.ml_scorer import tune_and_train

        df = _make_features_df(n=60)
        # Inject NaN into enough rows to drop below 50
        df.loc[df.index[:20], "orb_range_pct"] = np.nan
        with pytest.raises(ValueError, match="need"):
            tune_and_train(df)

    @pytest.mark.slow
    def test_valid_data_returns_classifier(self, tmp_path: Path) -> None:
        from src.backtest.ml_scorer import tune_and_train

        df = _make_features_df(n=200, seed=42)
        model_path = tmp_path / "model.pkl"
        shap_path = tmp_path / "shap.png"

        with patch("src.backtest.ml_scorer._save_shap_plot"):
            model = tune_and_train(
                df,
                n_trials=2,
                model_path=model_path,
                shap_path=shap_path,
            )

        import lightgbm as lgb

        assert isinstance(model, lgb.LGBMClassifier)

    @pytest.mark.slow
    def test_saves_model_to_disk(self, tmp_path: Path) -> None:
        from src.backtest.ml_scorer import tune_and_train

        df = _make_features_df(n=200, seed=42)
        model_path = tmp_path / "model.pkl"
        shap_path = tmp_path / "shap.png"

        with patch("src.backtest.ml_scorer._save_shap_plot"):
            tune_and_train(
                df,
                n_trials=2,
                model_path=model_path,
                shap_path=shap_path,
            )

        assert model_path.exists()
        with open(model_path, "rb") as f:
            loaded = pickle.load(f)  # noqa: S301
        import lightgbm as lgb

        assert isinstance(loaded, lgb.LGBMClassifier)

    @pytest.mark.slow
    def test_calls_save_shap_plot(self, tmp_path: Path) -> None:
        from src.backtest.ml_scorer import tune_and_train

        df = _make_features_df(n=200, seed=42)
        model_path = tmp_path / "model.pkl"
        shap_path = tmp_path / "shap.png"

        with patch("src.backtest.ml_scorer._save_shap_plot") as mock_shap:
            tune_and_train(
                df,
                n_trials=2,
                model_path=model_path,
                shap_path=shap_path,
            )
            mock_shap.assert_called_once()


# ── _save_shap_plot ──────────────────────────────────────────────────────────


class TestSaveShapPlot:
    """Tests for _save_shap_plot (lines 256-281)."""

    def test_calls_shap_and_matplotlib(self, tmp_path: Path) -> None:
        import sys

        import src.backtest.ml_scorer as scorer_mod

        mock_model = MagicMock()
        n = 50
        x_data = pd.DataFrame(
            {col: np.random.default_rng(42).normal(0, 1, n) for col in FEATURE_COLS}
        )
        shap_path = tmp_path / "shap.png"

        mock_explainer_instance = MagicMock()
        mock_explainer_instance.shap_values.return_value = np.random.default_rng(42).normal(
            0, 1, (n, len(FEATURE_COLS))
        )

        mock_shap_mod = MagicMock()
        mock_shap_mod.TreeExplainer.return_value = mock_explainer_instance

        with (
            patch.dict(sys.modules, {"shap": mock_shap_mod}),
            patch("matplotlib.pyplot.figure"),
            patch("matplotlib.pyplot.savefig") as mock_savefig,
            patch("matplotlib.pyplot.title"),
            patch("matplotlib.pyplot.tight_layout"),
            patch("matplotlib.pyplot.close"),
        ):
            scorer_mod._save_shap_plot(mock_model, x_data, shap_path)
            mock_shap_mod.TreeExplainer.assert_called_once_with(mock_model)
            mock_shap_mod.summary_plot.assert_called_once()
            mock_savefig.assert_called_once()

    def test_shap_exception_handled_gracefully(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from src.backtest.ml_scorer import _save_shap_plot

        mock_model = MagicMock()
        n = 50
        x_data = pd.DataFrame(
            {col: np.random.default_rng(42).normal(0, 1, n) for col in FEATURE_COLS}
        )
        shap_path = tmp_path / "shap.png"

        with patch.dict("sys.modules", {"shap": None}):
            # Importing shap will fail, caught by except block
            _save_shap_plot(mock_model, x_data, shap_path)

        captured = capsys.readouterr()
        assert "skipped" in captured.out.lower() or "skip" in captured.out.lower()

    def test_shap_values_list_uses_index_one(self, tmp_path: Path) -> None:
        """When shap_values returns a list [class0, class1], use index 1."""
        import sys

        import src.backtest.ml_scorer as scorer_mod

        mock_model = MagicMock()
        n = 50
        x_data = pd.DataFrame(
            {col: np.random.default_rng(42).normal(0, 1, n) for col in FEATURE_COLS}
        )
        shap_path = tmp_path / "shap.png"

        rng = np.random.default_rng(42)
        class0_vals = rng.normal(0, 1, (n, len(FEATURE_COLS)))
        class1_vals = rng.normal(0, 1, (n, len(FEATURE_COLS)))

        mock_explainer_instance = MagicMock()
        mock_explainer_instance.shap_values.return_value = [class0_vals, class1_vals]

        mock_shap_mod = MagicMock()
        mock_shap_mod.TreeExplainer.return_value = mock_explainer_instance
        mock_plt = MagicMock()

        orig_shap = sys.modules.get("shap")
        orig_plt = sys.modules.get("matplotlib.pyplot")
        sys.modules["shap"] = mock_shap_mod
        sys.modules["matplotlib.pyplot"] = mock_plt
        try:
            scorer_mod._save_shap_plot(mock_model, x_data, shap_path)

            # summary_plot should be called with class1_vals
            call_args = mock_shap_mod.summary_plot.call_args
            np.testing.assert_array_equal(call_args[0][0], class1_vals)
        finally:
            if orig_shap is not None:
                sys.modules["shap"] = orig_shap
            if orig_plt is not None:
                sys.modules["matplotlib.pyplot"] = orig_plt


# ── load_model ───────────────────────────────────────────────────────────────


class TestLoadModel:
    """Tests for load_model (lines 299-306)."""

    def test_missing_file_raises_file_not_found(self, tmp_path: Path) -> None:
        from src.backtest.ml_scorer import load_model

        with pytest.raises(FileNotFoundError, match="Model not found"):
            load_model(tmp_path / "nonexistent.pkl")

    def test_valid_pickle_returns_model(self, tmp_path: Path) -> None:
        from src.backtest.ml_scorer import load_model

        model_path = tmp_path / "model.pkl"
        fake_model = {"type": "fake_lgbm", "n_estimators": 10}
        with open(model_path, "wb") as f:
            pickle.dump(fake_model, f)

        loaded = load_model(model_path)
        assert loaded == {"type": "fake_lgbm", "n_estimators": 10}

    def test_load_preserves_model_type(self, tmp_path: Path) -> None:
        """Pickle round-trip preserves LGBMClassifier type."""
        import lightgbm as lgb

        from src.backtest.ml_scorer import load_model

        model = lgb.LGBMClassifier(n_estimators=10, verbosity=-1)
        model_path = tmp_path / "model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        loaded = load_model(model_path)
        assert isinstance(loaded, lgb.LGBMClassifier)
