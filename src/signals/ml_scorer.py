"""Live ML signal confidence scorer for the production orchestrator.

Usage
-----
    scorer = SignalScorer()           # loads model from default path
    conf   = scorer.score(features)  # 0.0 - 1.0
    shap   = scorer.explain(features) # dict[feature_name, shap_value]

The ``features`` dict uses the same key names as the backtest training
features (see ml_features.FEATURE_COLS).  Missing keys default to NaN,
which LightGBM handles gracefully via its native NaN split logic.

This class is intentionally NOT wired into main.py yet.  It is ready to be
injected into the signal pipeline once paper-trading results validate the
confidence threshold in production.

Thread safety
-------------
``score()`` and ``explain()`` are read-only after __init__ and are
safe to call from the asyncio event loop (no shared mutable state).
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import structlog

from src.backtest.ml_features import FEATURE_COLS, encode_ticker

logger = structlog.get_logger(__name__)

_DEFAULT_MODEL_PATH = Path("models/lgbm_signal_scorer.pkl")


class SignalScorer:
    """Load a trained LightGBM model and score live signals.

    Args:
        model_path: Path to the pickled LGBMClassifier produced by
                    ``ml_scorer.tune_and_train()``.
    """

    def __init__(self, model_path: Path = _DEFAULT_MODEL_PATH) -> None:
        if not model_path.exists():
            raise FileNotFoundError(
                f"No ML model at {model_path}. " "Run: python scripts/train_ml_scorer.py"
            )
        with open(model_path, "rb") as f:
            self._model = pickle.load(f)  # noqa: S301

        # SHAP explainer is lazy-initialised on first .explain() call
        self._explainer: Any | None = None

        logger.info("signal_scorer_loaded", path=str(model_path))

    # ── public API ────────────────────────────────────────────────────────────

    def score(self, features: dict[str, float]) -> float:
        """Return the ML-estimated win probability for a signal.

        Args:
            features: Dict with keys matching FEATURE_COLS.  The 'ticker'
                      key (string) is automatically encoded; pass
                      ``ticker_encoded`` directly if you prefer.

        Returns:
            Float in [0.0, 1.0] where 1.0 = high confidence winner.
        """
        x_data = self._build_df(features)
        proba: float = float(self._model.predict_proba(x_data)[0, 1])
        return proba

    def explain(self, features: dict[str, float]) -> dict[str, float]:
        """Return SHAP contribution values per feature for one signal.

        Positive SHAP = pushes prediction toward 'winner'.
        Negative SHAP = pushes prediction toward 'loser'.

        Args:
            features: Same dict as passed to ``score()``.

        Returns:
            Dict mapping feature name → SHAP value.
        """
        import shap

        if self._explainer is None:
            self._explainer = shap.TreeExplainer(self._model)

        x_data = self._build_df(features)
        shap_values = self._explainer.shap_values(x_data)

        # LightGBM binary: shap_values is [class0_arr, class1_arr] or single arr
        sv = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]

        return dict(zip(FEATURE_COLS, sv.tolist(), strict=False))

    # ── helpers ───────────────────────────────────────────────────────────────

    def _build_df(self, features: dict[str, float]) -> pd.DataFrame:
        """Convert a features dict to a model-ready single-row DataFrame.

        Handles:
        - Missing keys → NaN (LightGBM handles natively).
        - 'symbol' or 'ticker' keys → auto-encoded to 'ticker_encoded'.
        - ticker_encoded cast to int for LightGBM categorical.

        Args:
            features: Raw feature dict from the signal pipeline.

        Returns:
            1-row DataFrame with exactly FEATURE_COLS columns in order.
        """
        row: dict[str, Any] = {}
        for col in FEATURE_COLS:
            if col in features:
                row[col] = [features[col]]
            elif col == "ticker_encoded":
                # Accept 'symbol' or 'ticker' as the string key
                sym = features.get("symbol", features.get("ticker", "UNKNOWN"))
                row[col] = [encode_ticker(str(sym))]
            else:
                row[col] = [np.nan]

        x_data = pd.DataFrame(row)
        x_data["ticker_encoded"] = x_data["ticker_encoded"].astype(int)
        return x_data
