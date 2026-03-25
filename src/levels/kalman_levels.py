"""Kalman filter for adaptive ORB stop sizing.

Uses ``filterpy.kalman.KalmanFilter`` with a constant-velocity model to
track price evolution after the opening range.  The filter's innovation
sequence (prediction errors) measures how "surprised" the filter is by
each new bar.  Large innovations = unexpected price movement = widen stops.

Design
------
- State vector: [price_level, trend_velocity]  (dim_x=2)
- Measurement: observed bar close price         (dim_z=1)
- Innovation-based scaling:
    innovation = observed_price - predicted_price
    normalized_innovation = |innovation| / ATR
    rolling_avg_innovation over last 5 bars
    multiplier = 1.0 + (rolling_avg_innovation - 1.0) * sensitivity
  Clamped to [0.85, 1.8].
- Asymmetric design: default 1.0, widens on volatile price action,
  tightens slightly on calm/trending action.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import structlog
from filterpy.kalman import KalmanFilter

logger = structlog.get_logger(__name__)

# Kalman stop multiplier clamps
_KALMAN_MULT_MIN = 0.90  # floor: slightly tighter on very calm days
_KALMAN_MULT_MAX = 1.5  # ceiling: up to 1.5x on volatile days
_INNOVATION_WINDOW = 5  # rolling window for innovation averaging
_SENSITIVITY = 0.5  # how aggressively to scale from innovation


def _run_kalman_for_day(
    close_prices: np.ndarray[Any, np.dtype[np.float64]],
    atr_value: float,
    orb_bars: int = 5,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Run Kalman filter on one day's closes and return innovation-based multipliers.

    Args:
        close_prices: Array of close prices for one trading day.
        atr_value:    ATR(14) value at the start of the day.
        orb_bars:     Number of bars in the opening range (default 5).

    Returns:
        Float64 array of stop multipliers, same length as close_prices.
    """
    n = len(close_prices)
    mult_out = np.ones(n)

    if n <= orb_bars or atr_value <= 0 or np.isnan(atr_value):
        return mult_out

    # Initialize Kalman filter
    kf = KalmanFilter(dim_x=2, dim_z=1)

    kf.F = np.array([[1.0, 1.0], [0.0, 1.0]])
    kf.H = np.array([[1.0, 0.0]])

    # Initial state: ORB midpoint with zero velocity
    orb_close_mean = float(np.mean(close_prices[:orb_bars]))
    kf.x = np.array([[orb_close_mean], [0.0]])

    # Moderate initial covariance
    kf.P = np.array([[atr_value**2, 0.0], [0.0, (atr_value * 0.1) ** 2]])

    # Process noise: expect price drift ~20% of ATR per bar
    q_price = (atr_value * 0.2) ** 2
    q_velocity = (atr_value * 0.05) ** 2
    kf.Q = np.array([[q_price, 0.0], [0.0, q_velocity]])

    # Measurement noise
    kf.R = np.array([[(atr_value * 0.1) ** 2]])

    # Collect normalized innovations
    innovations: list[float] = []

    for i in range(orb_bars, n):
        z = np.array([[close_prices[i]]])

        kf.predict()

        # Innovation = measurement - predicted measurement
        predicted = float((kf.H @ kf.x)[0, 0])
        innovation = abs(close_prices[i] - predicted)
        normalized_innovation = innovation / atr_value

        kf.update(z)

        innovations.append(normalized_innovation)

        # Rolling average of normalized innovations
        window = innovations[-_INNOVATION_WINDOW:]
        avg_innovation = sum(window) / len(window)

        # Scale: avg_innovation ~0.2-0.3 on calm days, ~0.8-1.5 on volatile days
        # Map to multiplier: 1.0 + (avg_innovation - baseline) * sensitivity
        # Baseline ~0.3 (expected normalized innovation under calm conditions)
        raw_mult = 1.0 + (avg_innovation - 0.3) * _SENSITIVITY
        mult_out[i] = float(np.clip(raw_mult, _KALMAN_MULT_MIN, _KALMAN_MULT_MAX))

    return mult_out


class StreamingKalman:
    """Incremental Kalman stop multiplier — bar-by-bar update for live trading.

    Matches the batch ``_run_kalman_for_day`` logic: after the ORB window
    (first ``orb_bars`` bars of the session), a constant-velocity Kalman
    filter tracks price evolution.  Innovation (prediction error) drives
    the stop multiplier.

    Usage::

        km = StreamingKalman()
        km.reset(orb_bars_closes, atr_at_orb)  # at 9:35 ET
        for bar in post_orb_bars:
            km.update(float(bar.close))
            stop_mult = km.multiplier   # 0.90 -1.50
    """

    def __init__(self, orb_bars: int = 5) -> None:
        self._orb_bars = orb_bars
        self._kf: KalmanFilter | None = None
        self._innovations: list[float] = []
        self._multiplier: float = 1.0
        self._bar_count: int = 0
        self._ready: bool = False

    def reset(self, orb_closes: list[float], atr_value: float) -> None:
        """Initialize the filter at the end of the ORB window.

        Args:
            orb_closes: Close prices of the ORB bars (length ``orb_bars``).
            atr_value:  ATR(14) at the first post-ORB bar (already shifted).
        """
        self._innovations.clear()
        self._multiplier = 1.0
        self._bar_count = 0
        self._ready = False

        if atr_value <= 0 or np.isnan(atr_value) or len(orb_closes) < self._orb_bars:
            return

        self._atr = atr_value
        kf = KalmanFilter(dim_x=2, dim_z=1)
        kf.F = np.array([[1.0, 1.0], [0.0, 1.0]])
        kf.H = np.array([[1.0, 0.0]])
        orb_mean = float(np.mean(orb_closes))
        kf.x = np.array([[orb_mean], [0.0]])
        kf.P = np.array([[atr_value**2, 0.0], [0.0, (atr_value * 0.1) ** 2]])
        q_price = (atr_value * 0.2) ** 2
        q_velocity = (atr_value * 0.05) ** 2
        kf.Q = np.array([[q_price, 0.0], [0.0, q_velocity]])
        kf.R = np.array([[(atr_value * 0.1) ** 2]])
        self._kf = kf
        self._ready = True

    def update(self, close: float) -> float:
        """Feed one post-ORB close price and return the updated multiplier."""
        if not self._ready or self._kf is None:
            return self._multiplier

        z = np.array([[close]])
        self._kf.predict()
        predicted = float((self._kf.H @ self._kf.x)[0, 0])
        innovation = abs(close - predicted) / self._atr
        self._kf.update(z)

        self._innovations.append(innovation)
        window = self._innovations[-_INNOVATION_WINDOW:]
        avg_innovation = sum(window) / len(window)
        raw = 1.0 + (avg_innovation - 0.3) * _SENSITIVITY
        self._multiplier = float(np.clip(raw, _KALMAN_MULT_MIN, _KALMAN_MULT_MAX))
        self._bar_count += 1
        return self._multiplier

    @property
    def multiplier(self) -> float:
        """Current stop multiplier (0.90 -1.50, default 1.0)."""
        return self._multiplier

    @property
    def ready(self) -> bool:
        """True once reset() has been called with valid data."""
        return self._ready


def compute_kalman_stop_multiplier(
    index: pd.DatetimeIndex,
    close_prices: Any,
    atr_values: Any,
    orb_bars: int = 5,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Compute per-bar Kalman innovation-based multipliers for adaptive stop sizing.

    For each trading day, runs a Kalman filter on post-ORB close prices
    and tracks the innovation sequence (prediction errors).  Large
    innovations signal unexpected price movement, warranting wider stops.

    No lookahead: each bar's multiplier depends only on bars up to and
    including that bar.

    Args:
        index:        UTC DatetimeIndex aligned with price/ATR arrays.
        close_prices: Numpy-like array of 1-min close prices.
        atr_values:   Numpy-like array of ATR(14) values (already shifted
                      by 1 bar for no-lookahead).
        orb_bars:     Number of bars in the opening range (default 5).

    Returns:
        Float64 numpy array of multipliers, same length as ``index``.
        Default 1.0 where Kalman filter cannot run.
    """
    from zoneinfo import ZoneInfo

    _et_tz = ZoneInfo("America/New_York")

    close_arr = np.asarray(close_prices, dtype=float)
    atr_arr = np.asarray(atr_values, dtype=float)
    n = len(index)
    mult_out = np.ones(n)

    # Convert to ET for day grouping
    if index.tzinfo is None:
        et_index = index.tz_localize("UTC").tz_convert(_et_tz)
    else:
        et_index = index.tz_convert(_et_tz)

    # Group bar positions by ET calendar date
    dates: list[Any] = []
    date_to_positions: dict[Any, list[int]] = {}
    for i, ts in enumerate(et_index):
        d = ts.date()
        if d not in date_to_positions:
            dates.append(d)
            date_to_positions[d] = []
        date_to_positions[d].append(i)

    for d in dates:
        positions = date_to_positions[d]
        if len(positions) <= orb_bars:
            continue

        day_close = close_arr[positions]

        # Use ATR from the first post-ORB bar (already shifted, no lookahead)
        atr_val = float(atr_arr[positions[orb_bars]])
        if np.isnan(atr_val) or atr_val <= 0:
            continue

        day_mult = _run_kalman_for_day(day_close, atr_val, orb_bars)

        for j, pos in enumerate(positions):
            mult_out[pos] = day_mult[j]

    return mult_out
