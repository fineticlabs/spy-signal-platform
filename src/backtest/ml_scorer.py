"""LightGBM signal confidence scorer — training pipeline.

Pipeline
--------
1. Accept a features DataFrame (from ml_features.extract_all_features).
2. Use TimeSeriesSplit (5 folds) with a 5-day purge gap for Optuna tuning.
3. Optimize ROC-AUC with 200 Optuna trials.
4. Train final LightGBM model on all data with best hyperparameters.
5. Save model to models/lgbm_signal_scorer.pkl.
6. Generate SHAP bar plot to docs/shap_feature_importance.png.

Design choices
--------------
- class_weight='balanced' to handle win/loss imbalance without SMOTE,
  which cannot be applied to time-series data without leakage.
- TimeSeriesSplit with manual purge: train fold only uses rows whose
  entry_time < first_val_entry_time - 5 days.
- LightGBM categorical_feature for ticker_encoded to let the tree treat
  ticker as a nominal variable.
- Optuna's MedianPruner to discard poor trials early.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import structlog
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from src.backtest.ml_features import FEATURE_COLS

logger = structlog.get_logger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── constants ─────────────────────────────────────────────────────────────────

_MODEL_PATH = Path("models/lgbm_signal_scorer.pkl")
_SHAP_PATH = Path("docs/shap_feature_importance.png")
_N_SPLITS = 5
_PURGE_DAYS = 5
_N_TRIALS = 200
_CATEGORICAL_FEATURES = ["ticker_encoded"]


# ── purged time-series CV ─────────────────────────────────────────────────────


def _purged_cv_score(
    x_data: pd.DataFrame,
    y: pd.Series,
    entry_times: pd.Series,
    params: dict[str, Any],
) -> float:
    """5-fold purged time-series cross-validation ROC-AUC.

    For each fold, removes train rows within _PURGE_DAYS of the first
    validation timestamp to eliminate autocorrelation leakage.

    Args:
        x_data:      Feature DataFrame (n_samples, n_features).
        y:           Binary label Series (0/1).
        entry_times: Series of entry timestamps (same index as x_data/y).
        params:      LightGBM hyperparameters.

    Returns:
        Mean ROC-AUC across all folds (NaN folds skipped).
    """
    tscv = TimeSeriesSplit(n_splits=_N_SPLITS)
    scores: list[float] = []

    for train_idx, val_idx in tscv.split(x_data):
        val_start = entry_times.iloc[val_idx].min()
        purge_cutoff = val_start - pd.Timedelta(days=_PURGE_DAYS)

        # Apply purge: drop train rows too close to validation
        train_mask = entry_times.iloc[train_idx] < purge_cutoff
        purged_train_idx = np.array(train_idx)[train_mask.to_numpy()]

        if len(purged_train_idx) < 30 or len(val_idx) < 10:
            continue

        x_train = x_data.iloc[purged_train_idx]
        y_train = y.iloc[purged_train_idx]
        x_val = x_data.iloc[val_idx]
        y_val = y.iloc[val_idx]

        model = lgb.LGBMClassifier(
            **params,
            class_weight="balanced",
            random_state=42,
            verbosity=-1,
            n_jobs=1,
        )
        model.fit(
            x_train,
            y_train,
            categorical_feature=_CATEGORICAL_FEATURES,
        )

        proba = model.predict_proba(x_val)[:, 1]
        if len(np.unique(y_val)) < 2:
            continue  # skip degenerate fold

        scores.append(roc_auc_score(y_val, proba))

    return float(np.mean(scores)) if scores else 0.5


# ── Optuna objective ───────────────────────────────────────────────────────────


def _make_objective(
    x_data: pd.DataFrame,
    y: pd.Series,
    entry_times: pd.Series,
) -> Any:
    """Return an Optuna objective closure.

    Args:
        x_data:      Feature DataFrame.
        y:           Label Series.
        entry_times: Entry timestamp Series.

    Returns:
        Callable ``objective(trial) -> float`` for optuna.create_study.
    """

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
        }
        return _purged_cv_score(x_data, y, entry_times, params)

    return objective


# ── model training ────────────────────────────────────────────────────────────


def tune_and_train(
    features_df: pd.DataFrame,
    n_trials: int = _N_TRIALS,
    model_path: Path = _MODEL_PATH,
    shap_path: Path = _SHAP_PATH,
) -> lgb.LGBMClassifier:
    """Tune hyperparameters with Optuna, train final model, save + SHAP.

    Args:
        features_df: Output of extract_all_features(); must contain
                     FEATURE_COLS, 'label', and 'entry_time' columns.
        n_trials:    Number of Optuna trials (default 200).
        model_path:  Where to save the trained model .pkl.
        shap_path:   Where to save the SHAP summary plot PNG.

    Returns:
        Trained LGBMClassifier.
    """
    if features_df.empty:
        raise ValueError("features_df is empty — run feature extraction first")

    df = features_df.dropna(subset=[*FEATURE_COLS, "label"]).copy()
    if len(df) < 50:
        raise ValueError(f"Only {len(df)} clean rows — need ≥50 to train")

    x_data = df[FEATURE_COLS].copy()
    # Cast ticker_encoded to int for LightGBM categorical support
    x_data["ticker_encoded"] = x_data["ticker_encoded"].astype(int)
    y: pd.Series = df["label"].astype(int)
    entry_times: pd.Series = df["entry_time"]

    logger.info(
        "ml_training_start",
        samples=len(df),
        win_rate=round(float(y.mean()), 3),
        n_trials=n_trials,
    )

    # ── Optuna hyperparameter search ──────────────────────────────────────
    sampler = optuna.samplers.TPESampler(seed=42)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=20)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )
    study.optimize(
        _make_objective(x_data, y, entry_times),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    best_params = study.best_params
    best_auc = study.best_value
    logger.info("optuna_done", best_auc=round(best_auc, 4), best_params=best_params)
    print(f"\n  Best AUC: {best_auc:.4f}")
    print(f"  Best params: {best_params}")

    # ── Final model on all data ───────────────────────────────────────────
    final_model = lgb.LGBMClassifier(
        **best_params,
        class_weight="balanced",
        random_state=42,
        verbosity=-1,
        n_jobs=-1,
    )
    final_model.fit(
        x_data,
        y,
        categorical_feature=_CATEGORICAL_FEATURES,
    )

    # ── Save model ────────────────────────────────────────────────────────
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(final_model, f)
    logger.info("model_saved", path=str(model_path))
    print(f"\n  Model saved → {model_path}")

    # ── SHAP feature importance plot ──────────────────────────────────────
    _save_shap_plot(final_model, x_data, shap_path)

    return final_model


# ── SHAP plot ─────────────────────────────────────────────────────────────────


def _save_shap_plot(
    model: lgb.LGBMClassifier,
    x_data: pd.DataFrame,
    shap_path: Path,
) -> None:
    """Compute SHAP values and save a bar-chart summary plot.

    Args:
        model:     Trained LGBMClassifier.
        x_data:    Feature DataFrame (used as background sample).
        shap_path: Output PNG path.
    """
    try:
        import matplotlib.pyplot as plt
        import shap

        explainer = shap.TreeExplainer(model)
        # Sample up to 500 rows for speed (SHAP is O(n*trees))
        sample = x_data.sample(min(len(x_data), 500), random_state=42)
        shap_values = explainer.shap_values(sample)

        # For binary LightGBM, shap_values may be a list [class0_vals, class1_vals]
        # or a single 2D array depending on the SHAP version.  Normalise:
        sv = shap_values[1] if isinstance(shap_values, list) else shap_values

        shap_path.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(10, 7))
        shap.summary_plot(sv, sample, plot_type="bar", show=False)
        plt.title("LightGBM Signal Scorer — SHAP Feature Importance")
        plt.tight_layout()
        plt.savefig(shap_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("shap_plot_saved", path=str(shap_path))
        print(f"  SHAP plot saved → {shap_path}")

    except Exception as exc:
        logger.warning("shap_plot_failed", error=str(exc))
        print(f"  SHAP plot skipped: {exc}")


# ── model loading ─────────────────────────────────────────────────────────────


def load_model(model_path: Path = _MODEL_PATH) -> lgb.LGBMClassifier:
    """Load a previously saved LGBMClassifier.

    Args:
        model_path: Path to the .pkl file.

    Returns:
        Loaded LGBMClassifier.

    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. " "Run scripts/train_ml_scorer.py first."
        )
    with open(model_path, "rb") as f:
        model: lgb.LGBMClassifier = pickle.load(f)  # noqa: S301
    logger.info("model_loaded", path=str(model_path))
    return model
