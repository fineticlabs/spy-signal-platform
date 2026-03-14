"""Hidden Markov Model regime detection for walk-forward backtesting.

Uses ``hmmlearn.GaussianHMM`` with 3 states trained on in-sample data to
classify each out-of-sample bar into one of three regimes:

    - **CALM**: lowest return variance — tight-range, low-vol environment.
    - **NORMAL**: mid-range variance — standard market conditions.
    - **VOLATILE**: highest return variance — stressed/chaotic environment.

Design
------
- Features: 5-min bar returns, volume ratio (bar vol / 20-bar avg), ATR
  change (first difference of 14-period ATR).  All features are z-score
  standardized using IS statistics (no OOS leakage).
- Trained on IS period 5-min bars, predicts on OOS 5-min bars.
- Predictions are mapped back to 1-min resolution via forward-fill.
- No lookahead: each bar's regime is known only after the IS training
  period ends and the OOS prediction is made.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import numpy as np
import pandas as pd
import structlog
import talib
from hmmlearn.hmm import GaussianHMM

logger = structlog.get_logger(__name__)

_HMM_N_COMPONENTS = 3
_HMM_N_ITER = 100
_HMM_COVARIANCE_TYPE = "full"
_VOLUME_AVG_WINDOW = 20
_ATR_PERIOD = 14
_MIN_TRAINING_BARS = 200  # minimum 5-min bars needed to fit HMM


class HMMRegime(IntEnum):
    """Regime labels ordered by return variance."""

    CALM = 0
    NORMAL = 1
    VOLATILE = 2


@dataclass(frozen=True)
class HMMScaler:
    """Z-score standardization parameters learned from IS data."""

    means: np.ndarray[Any, np.dtype[np.float64]]
    stds: np.ndarray[Any, np.dtype[np.float64]]

    def transform(
        self, features: np.ndarray[Any, np.dtype[np.float64]]
    ) -> np.ndarray[Any, np.dtype[np.float64]]:
        """Apply the IS-learned standardization to new features."""
        return (features - self.means) / self.stds


def _compute_raw_features(
    df_5min: pd.DataFrame,
) -> tuple[np.ndarray[Any, np.dtype[np.float64]], np.ndarray[Any, np.dtype[np.bool_]]] | None:
    """Build the raw 3-feature matrix from 5-min OHLCV bars.

    Returns:
        (features_all, valid_mask) where features_all is (N, 3) and
        valid_mask indicates rows without NaN.  Returns None if ATR
        cannot be computed.
    """
    close = df_5min["close"].values.astype(float)
    high = df_5min["high"].values.astype(float)
    low = df_5min["low"].values.astype(float)
    volume = df_5min["volume"].values.astype(float)

    # Feature 1: log returns
    log_ret = np.diff(np.log(np.maximum(close, 1e-10)), prepend=np.nan)

    # Feature 2: volume ratio (bar vol / 20-bar rolling mean)
    vol_series = pd.Series(volume)
    vol_avg = vol_series.rolling(_VOLUME_AVG_WINDOW).mean().values
    vol_ratio = np.where(vol_avg > 0, volume / vol_avg, np.nan)

    # Feature 3: ATR change (first difference of ATR)
    if len(high) < _ATR_PERIOD + 1:
        return None
    atr = talib.ATR(high, low, close, timeperiod=_ATR_PERIOD)
    atr_change = np.diff(atr, prepend=np.nan)

    features = np.column_stack([log_ret, vol_ratio, atr_change])
    valid_mask = ~np.any(np.isnan(features), axis=1)

    return features, valid_mask


def _label_states(
    model: GaussianHMM,
) -> dict[int, HMMRegime]:
    """Map HMM hidden states to regime labels by return variance.

    The state with the highest variance in the returns feature (column 0)
    is labelled VOLATILE; the lowest is CALM; the remaining is NORMAL.
    """
    covars = model.covars_
    return_variances = np.array([covars[s][0, 0] for s in range(_HMM_N_COMPONENTS)])

    sorted_indices = np.argsort(return_variances)

    return {
        int(sorted_indices[0]): HMMRegime.CALM,
        int(sorted_indices[1]): HMMRegime.NORMAL,
        int(sorted_indices[2]): HMMRegime.VOLATILE,
    }


def train_hmm(
    df_5min_is: pd.DataFrame,
) -> tuple[GaussianHMM, dict[int, HMMRegime], HMMScaler] | None:
    """Train a GaussianHMM on in-sample 5-min bars.

    Returns:
        (fitted_model, state_label_mapping, scaler) tuple, or None if
        training fails.
    """
    result = _compute_raw_features(df_5min_is)
    if result is None:
        logger.warning("hmm_insufficient_data", n_bars=len(df_5min_is))
        return None

    features_all, valid_mask = result
    clean = features_all[valid_mask]

    if len(clean) < _MIN_TRAINING_BARS:
        logger.warning("hmm_insufficient_data", n_clean=len(clean))
        return None

    # Z-score standardize using IS statistics
    means = clean.mean(axis=0)
    stds = clean.std(axis=0)
    stds[stds == 0] = 1.0
    scaler = HMMScaler(means=means, stds=stds)
    scaled = scaler.transform(clean)

    model = GaussianHMM(
        n_components=_HMM_N_COMPONENTS,
        covariance_type=_HMM_COVARIANCE_TYPE,
        n_iter=_HMM_N_ITER,
        random_state=42,
    )

    try:
        model.fit(scaled)
    except Exception as exc:
        logger.error("hmm_training_failed", error=str(exc))
        return None

    if not model.monitor_.converged:
        logger.warning("hmm_not_converged", n_iter=model.monitor_.n_iter)

    mapping = _label_states(model)
    logger.info(
        "hmm_trained",
        n_bars=len(clean),
        converged=model.monitor_.converged,
        state_mapping={str(k): v.name for k, v in mapping.items()},
    )
    return model, mapping, scaler


def predict_regime(
    model: GaussianHMM,
    mapping: dict[int, HMMRegime],
    scaler: HMMScaler,
    df_5min_oos: pd.DataFrame,
) -> pd.Series:
    """Predict regime labels for out-of-sample 5-min bars.

    Uses the IS-learned scaler to standardize OOS features before
    prediction (no OOS leakage).
    """
    n = len(df_5min_oos)
    default = pd.Series(
        [HMMRegime.NORMAL] * n,
        index=df_5min_oos.index,
        dtype=int,
    )

    result = _compute_raw_features(df_5min_oos)
    if result is None:
        return default

    features_all, valid_mask = result

    if valid_mask.sum() < 2:
        return default

    # Apply IS standardization to OOS features
    valid_features = scaler.transform(features_all[valid_mask])

    try:
        raw_states = model.predict(valid_features)
    except Exception as exc:
        logger.error("hmm_prediction_failed", error=str(exc))
        return default

    regimes = np.full(n, HMMRegime.NORMAL, dtype=int)
    valid_indices = np.where(valid_mask)[0]
    for i, state in zip(valid_indices, raw_states, strict=False):
        regimes[i] = mapping.get(int(state), HMMRegime.NORMAL)

    return pd.Series(regimes, index=df_5min_oos.index, dtype=int)


def compute_hmm_regime_for_window(
    df_1min_is: pd.DataFrame,
    df_1min_oos: pd.DataFrame,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Train HMM on IS data and predict regime for OOS 1-min bars.

    This is the main entry point called from the walk-forward loop.
    It handles:
        1. Resampling IS and OOS 1-min bars to 5-min.
        2. Training HMM on IS 5-min features (with z-score standardization).
        3. Predicting regime on OOS 5-min bars (using IS scaler).
        4. Mapping 5-min predictions back to 1-min resolution.

    Returns:
        Float64 numpy array of length ``len(df_1min_oos)`` with values
        from HMMRegime (0=CALM, 1=NORMAL, 2=VOLATILE).
        Falls back to all-NORMAL if training/prediction fails.
    """
    n_oos = len(df_1min_oos)
    default = np.full(n_oos, float(HMMRegime.NORMAL))

    if df_1min_is.empty or df_1min_oos.empty:
        return default

    # Resample to 5-min
    ohlcv_agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }

    df_5min_is = (
        df_1min_is[["open", "high", "low", "close", "volume"]]
        .resample("5min")
        .agg(ohlcv_agg)
        .dropna()
    )
    df_5min_oos = (
        df_1min_oos[["open", "high", "low", "close", "volume"]]
        .resample("5min")
        .agg(ohlcv_agg)
        .dropna()
    )

    if len(df_5min_is) < _MIN_TRAINING_BARS or df_5min_oos.empty:
        logger.info(
            "hmm_skipped",
            reason="insufficient_5min_bars",
            is_bars=len(df_5min_is),
            oos_bars=len(df_5min_oos),
        )
        return default

    # Train on IS
    result = train_hmm(df_5min_is)
    if result is None:
        return default

    model, mapping, scaler = result

    # Predict on OOS (5-min resolution) using IS scaler
    regime_5min = predict_regime(model, mapping, scaler, df_5min_oos)

    # Map back to 1-min resolution via forward-fill
    regime_series = regime_5min.reindex(df_1min_oos.index, method="ffill")
    regime_series = regime_series.fillna(float(HMMRegime.NORMAL))

    return regime_series.values.astype(float)
