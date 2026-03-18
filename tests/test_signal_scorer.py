"""Tests for src.signals.ml_scorer — SignalScorer class."""

from __future__ import annotations

import pickle
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.backtest.ml_features import FEATURE_COLS, TICKER_LIST, encode_ticker
from src.signals.ml_scorer import SignalScorer

# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture()
def mock_model() -> MagicMock:
    """Return a MagicMock mimicking LGBMClassifier."""
    model = MagicMock()
    model.predict_proba.return_value = np.array([[0.3, 0.7]])
    return model


@pytest.fixture()
def model_path(tmp_path: Path) -> Path:
    """Create a dummy pickle file so SignalScorer.__init__ can open it."""
    path = tmp_path / "model.pkl"
    with open(path, "wb") as f:
        pickle.dump("placeholder", f)
    return path


@pytest.fixture()
def scorer(model_path: Path, mock_model: MagicMock) -> SignalScorer:
    """Return a SignalScorer with the model replaced by a MagicMock."""
    s = SignalScorer(model_path=model_path)
    s._model = mock_model
    return s


@pytest.fixture()
def full_features() -> dict[str, float]:
    """Feature dict with all FEATURE_COLS populated."""
    return {
        "orb_range_pct": 0.5,
        "gap_pct": 0.1,
        "volume_ratio": 1.8,
        "atr_pct": 0.3,
        "rsi_14": 55.0,
        "ema_distance_pct": 0.02,
        "time_minutes": 15.0,
        "day_of_week": 2.0,
        "vix_proxy": 18.5,
        "adx_value": 30.0,
        "consecutive_candles": 3.0,
        "ticker_encoded": 0,
        "direction": 1.0,
    }


# ── __init__ ─────────────────────────────────────────────────────────────────


class TestInit:
    """Tests for SignalScorer.__init__."""

    def test_init_raises_file_not_found_when_model_missing(self, tmp_path: Path) -> None:
        """FileNotFoundError raised when model_path does not exist."""
        missing = tmp_path / "nonexistent.pkl"
        with pytest.raises(FileNotFoundError, match="No ML model"):
            SignalScorer(model_path=missing)

    def test_init_loads_model_from_pickle(self, model_path: Path) -> None:
        """Model is loaded and stored as _model attribute."""
        scorer = SignalScorer(model_path=model_path)
        assert scorer._model is not None
        # The placeholder string was pickled, so it should be loaded back
        assert scorer._model == "placeholder"

    def test_init_explainer_is_none(self, scorer: SignalScorer) -> None:
        """_explainer starts as None (lazy-initialized)."""
        assert scorer._explainer is None


# ── _build_df ────────────────────────────────────────────────────────────────


class TestBuildDf:
    """Tests for SignalScorer._build_df."""

    def test_build_df_all_features_correct_columns(
        self, scorer: SignalScorer, full_features: dict[str, float]
    ) -> None:
        """DataFrame columns match FEATURE_COLS in order."""
        df = scorer._build_df(full_features)
        assert list(df.columns) == FEATURE_COLS

    def test_build_df_all_features_correct_values(
        self, scorer: SignalScorer, full_features: dict[str, float]
    ) -> None:
        """Values from features dict are preserved in the DataFrame."""
        df = scorer._build_df(full_features)
        for col in FEATURE_COLS:
            assert df[col].iloc[0] == full_features[col]

    def test_build_df_single_row(
        self, scorer: SignalScorer, full_features: dict[str, float]
    ) -> None:
        """Output is always a 1-row DataFrame."""
        df = scorer._build_df(full_features)
        assert len(df) == 1

    def test_build_df_missing_features_fills_nan(self, scorer: SignalScorer) -> None:
        """Missing feature keys produce NaN in the DataFrame."""
        df = scorer._build_df({"direction": 1.0, "ticker_encoded": 0})
        for col in FEATURE_COLS:
            if col not in ("direction", "ticker_encoded"):
                assert np.isnan(df[col].iloc[0]), f"{col} should be NaN"

    def test_build_df_symbol_key_auto_encodes(self, scorer: SignalScorer) -> None:
        """'symbol' key auto-encodes to ticker_encoded."""
        df = scorer._build_df({"symbol": "SPY"})
        assert df["ticker_encoded"].iloc[0] == encode_ticker("SPY")

    def test_build_df_ticker_key_auto_encodes(self, scorer: SignalScorer) -> None:
        """'ticker' key auto-encodes to ticker_encoded."""
        df = scorer._build_df({"ticker": "QQQ"})
        assert df["ticker_encoded"].iloc[0] == encode_ticker("QQQ")

    def test_build_df_symbol_takes_precedence_over_ticker(self, scorer: SignalScorer) -> None:
        """When both 'symbol' and 'ticker' present, 'symbol' is used."""
        df = scorer._build_df({"symbol": "SPY", "ticker": "QQQ"})
        assert df["ticker_encoded"].iloc[0] == encode_ticker("SPY")

    def test_build_df_ticker_encoded_is_int_dtype(
        self, scorer: SignalScorer, full_features: dict[str, float]
    ) -> None:
        """ticker_encoded column has integer dtype."""
        df = scorer._build_df(full_features)
        assert pd.api.types.is_integer_dtype(df["ticker_encoded"])

    def test_build_df_ticker_encoded_int_with_symbol(self, scorer: SignalScorer) -> None:
        """ticker_encoded is int even when auto-encoded from symbol."""
        df = scorer._build_df({"symbol": "AMD"})
        assert pd.api.types.is_integer_dtype(df["ticker_encoded"])

    def test_build_df_unknown_symbol_encoded_as_len_ticker_list(self, scorer: SignalScorer) -> None:
        """Unknown symbol is encoded as len(TICKER_LIST)."""
        df = scorer._build_df({"symbol": "ZZZZZ"})
        assert df["ticker_encoded"].iloc[0] == len(TICKER_LIST)

    def test_build_df_no_symbol_no_ticker_encodes_unknown(self, scorer: SignalScorer) -> None:
        """When neither symbol nor ticker provided, uses UNKNOWN encoding."""
        df = scorer._build_df({"direction": 1.0})
        assert df["ticker_encoded"].iloc[0] == len(TICKER_LIST)

    def test_build_df_explicit_ticker_encoded_not_overridden(self, scorer: SignalScorer) -> None:
        """Explicit ticker_encoded in features is used directly."""
        df = scorer._build_df({"ticker_encoded": 2, "symbol": "SPY"})
        # ticker_encoded key present → used directly, symbol ignored for that col
        assert df["ticker_encoded"].iloc[0] == 2


# ── score ────────────────────────────────────────────────────────────────────


class TestScore:
    """Tests for SignalScorer.score."""

    def test_score_returns_float(
        self, scorer: SignalScorer, full_features: dict[str, float]
    ) -> None:
        """score() returns a Python float."""
        result = scorer.score(full_features)
        assert isinstance(result, float)

    def test_score_returns_value_between_0_and_1(
        self, scorer: SignalScorer, full_features: dict[str, float]
    ) -> None:
        """score() returns value in [0, 1] (mock returns 0.7)."""
        result = scorer.score(full_features)
        assert 0.0 <= result <= 1.0

    def test_score_returns_correct_proba(
        self, scorer: SignalScorer, full_features: dict[str, float]
    ) -> None:
        """score() returns class-1 probability from predict_proba."""
        result = scorer.score(full_features)
        assert result == pytest.approx(0.7)

    def test_score_calls_predict_proba(
        self, scorer: SignalScorer, full_features: dict[str, float]
    ) -> None:
        """score() invokes model.predict_proba on _build_df output."""
        scorer.score(full_features)
        scorer._model.predict_proba.assert_called_once()

    def test_score_passes_build_df_output_to_model(
        self, scorer: SignalScorer, full_features: dict[str, float]
    ) -> None:
        """The DataFrame passed to predict_proba matches _build_df result."""
        expected_df = scorer._build_df(full_features)
        scorer.score(full_features)
        actual_arg = scorer._model.predict_proba.call_args[0][0]
        pd.testing.assert_frame_equal(actual_arg, expected_df)


# ── explain ──────────────────────────────────────────────────────────────────


class TestExplain:
    """Tests for SignalScorer.explain."""

    @staticmethod
    def _make_mock_shap(
        shap_values_return: np.ndarray | list[np.ndarray],
    ) -> tuple[MagicMock, MagicMock]:
        """Create mock shap module and explainer.

        Returns:
            (mock_shap_module, mock_explainer)
        """
        mock_explainer = MagicMock()
        mock_explainer.shap_values.return_value = shap_values_return
        mock_shap = MagicMock()
        mock_shap.TreeExplainer.return_value = mock_explainer
        return mock_shap, mock_explainer

    @staticmethod
    def _patch_shap(mock_shap: MagicMock):
        """Patch shap in sys.modules (lazy import inside explain())."""
        return patch.dict("sys.modules", {"shap": mock_shap})

    def test_explain_returns_dict(
        self, scorer: SignalScorer, full_features: dict[str, float]
    ) -> None:
        """explain() returns a dict."""
        shap_vals = np.random.default_rng(42).random(len(FEATURE_COLS))
        mock_shap, _ = self._make_mock_shap(
            [
                np.zeros((1, len(FEATURE_COLS))),
                shap_vals.reshape(1, -1),
            ]
        )

        with self._patch_shap(mock_shap):
            result = scorer.explain(full_features)

        assert isinstance(result, dict)

    def test_explain_keys_match_feature_cols(
        self, scorer: SignalScorer, full_features: dict[str, float]
    ) -> None:
        """explain() dict keys are exactly FEATURE_COLS."""
        shap_vals = np.random.default_rng(42).random(len(FEATURE_COLS))
        mock_shap, _ = self._make_mock_shap(
            [
                np.zeros((1, len(FEATURE_COLS))),
                shap_vals.reshape(1, -1),
            ]
        )

        with self._patch_shap(mock_shap):
            result = scorer.explain(full_features)

        assert list(result.keys()) == FEATURE_COLS

    def test_explain_values_match_shap_output(
        self, scorer: SignalScorer, full_features: dict[str, float]
    ) -> None:
        """explain() values correspond to class-1 SHAP values."""
        rng = np.random.default_rng(42)
        shap_vals = rng.random(len(FEATURE_COLS))
        mock_shap, _ = self._make_mock_shap(
            [
                np.zeros((1, len(FEATURE_COLS))),
                shap_vals.reshape(1, -1),
            ]
        )

        with self._patch_shap(mock_shap):
            result = scorer.explain(full_features)

        for col, expected in zip(FEATURE_COLS, shap_vals.tolist(), strict=True):
            assert result[col] == pytest.approx(expected)

    def test_explain_lazy_initializes_explainer(
        self, scorer: SignalScorer, full_features: dict[str, float]
    ) -> None:
        """_explainer is None before first explain(), set after."""
        assert scorer._explainer is None

        mock_shap, _ = self._make_mock_shap(
            [
                np.zeros((1, len(FEATURE_COLS))),
                np.zeros((1, len(FEATURE_COLS))),
            ]
        )

        with self._patch_shap(mock_shap):
            scorer.explain(full_features)

        assert scorer._explainer is not None

    def test_explain_reuses_explainer_on_second_call(
        self, scorer: SignalScorer, full_features: dict[str, float]
    ) -> None:
        """TreeExplainer is created only once across multiple explain() calls."""
        mock_shap, _ = self._make_mock_shap(
            [
                np.zeros((1, len(FEATURE_COLS))),
                np.zeros((1, len(FEATURE_COLS))),
            ]
        )

        with self._patch_shap(mock_shap):
            scorer.explain(full_features)
            scorer.explain(full_features)
            mock_shap.TreeExplainer.assert_called_once()

    def test_explain_handles_single_array_shap_values(
        self, scorer: SignalScorer, full_features: dict[str, float]
    ) -> None:
        """When shap_values returns a single array (not list), it still works."""
        shap_vals = np.random.default_rng(42).random(len(FEATURE_COLS))
        mock_shap, _ = self._make_mock_shap(shap_vals.reshape(1, -1))

        with self._patch_shap(mock_shap):
            result = scorer.explain(full_features)

        for col, expected in zip(FEATURE_COLS, shap_vals.tolist(), strict=True):
            assert result[col] == pytest.approx(expected)
