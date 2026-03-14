"""VIX term structure filter — contango/backwardation regime detection.

Computes the ratio VIX / VIX3M (CBOE 3-month volatility index) to classify
the volatility term structure:

  ratio < 0.85  → Strong contango (calm, complacent markets)
  0.85 ≤ ratio ≤ 1.00 → Normal contango (no adjustment)
  ratio > 1.00  → Backwardation (fear/panic, near-term vol > long-term)

For backtesting: downloads daily ^VIX and ^VIX3M from yfinance and caches
locally in ``data/vix_term_structure_cache.parquet``.
For live trading: fetches current ratio at startup.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)

_CACHE_PATH = Path("data/vix_term_structure_cache.parquet")

# ── Term structure thresholds ────────────────────────────────────────────────

CONTANGO_THRESHOLD = 0.85  # ratio < 0.85 → strong contango
BACKWARDATION_THRESHOLD = 1.00  # ratio > 1.00 → backwardation


def fetch_vix_term_structure(
    start: str = "2019-01-01",
    end: str | None = None,
) -> pd.DataFrame:
    """Download daily ^VIX and ^VIX3M closes from yfinance.

    Args:
        start: Start date ISO string (default "2019-01-01" for buffer before
               2020 backtest start).
        end:   End date ISO string (default None → today).

    Returns:
        DataFrame with columns ``["vix", "vix3m", "ratio"]`` indexed by date.
        ``ratio`` = VIX / VIX3M.  Rows with missing data are dropped.
    """
    import yfinance as yf

    logger.info("fetching_vix_term_structure", start=start, end=end)

    vix = yf.download("^VIX", start=start, end=end, progress=False)
    vix3m = yf.download("^VIX3M", start=start, end=end, progress=False)

    if vix.empty or vix3m.empty:
        logger.warning("vix_term_structure_fetch_failed", vix_rows=len(vix), vix3m_rows=len(vix3m))
        return pd.DataFrame(columns=["vix", "vix3m", "ratio"])

    # yfinance returns MultiIndex columns for single ticker: (col, ticker)
    # Flatten to just the column name
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)
    if isinstance(vix3m.columns, pd.MultiIndex):
        vix3m.columns = vix3m.columns.get_level_values(0)

    df = pd.DataFrame(
        {
            "vix": vix["Close"],
            "vix3m": vix3m["Close"],
        }
    ).dropna()

    # Ensure index is tz-naive date for clean joining
    df.index = pd.DatetimeIndex(df.index).tz_localize(None)

    df["ratio"] = df["vix"] / df["vix3m"]

    logger.info(
        "vix_term_structure_loaded",
        rows=len(df),
        start=str(df.index[0].date()) if len(df) > 0 else "N/A",
        end=str(df.index[-1].date()) if len(df) > 0 else "N/A",
    )
    return df


def load_vix_term_structure_cache() -> pd.DataFrame:
    """Load cached VIX term structure data from disk.

    Returns:
        DataFrame with columns ``["vix", "vix3m", "ratio"]``, or empty
        DataFrame if cache does not exist.
    """
    if not _CACHE_PATH.exists():
        return pd.DataFrame(columns=["vix", "vix3m", "ratio"])
    df: pd.DataFrame = pd.read_parquet(_CACHE_PATH)
    return df


def save_vix_term_structure_cache(df: pd.DataFrame) -> None:
    """Persist VIX term structure data to disk as Parquet.

    Args:
        df: DataFrame with columns ``["vix", "vix3m", "ratio"]``.
    """
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(_CACHE_PATH)
    logger.info("vix_term_structure_cache_saved", path=str(_CACHE_PATH), rows=len(df))


def get_vix_term_structure(
    *,
    use_cache: bool = True,
    refresh: bool = False,
) -> pd.DataFrame:
    """Get VIX term structure data, using cache if available.

    Args:
        use_cache: Whether to use local cache (default ``True``).
        refresh:   Force re-fetch even if cached (default ``False``).

    Returns:
        DataFrame with columns ``["vix", "vix3m", "ratio"]`` indexed by date.
    """
    if use_cache and not refresh:
        cached = load_vix_term_structure_cache()
        if not cached.empty:
            return cached

    df = fetch_vix_term_structure()
    if not df.empty:
        save_vix_term_structure_cache(df)
    return df


def classify_term_structure(ratio: float) -> str | None:
    """Classify a VIX/VIX3M ratio into a regime label.

    Args:
        ratio: VIX / VIX3M value.

    Returns:
        ``"CONTANGO"`` if ratio < 0.85, ``"BACKWARDATION"`` if ratio > 1.00,
        or ``None`` for normal contango (0.85-1.00, no tag).
    """
    if np.isnan(ratio):
        return None
    if ratio < CONTANGO_THRESHOLD:
        return "CONTANGO"
    if ratio > BACKWARDATION_THRESHOLD:
        return "BACKWARDATION"
    return None


def compute_vix_term_structure_array(
    index: pd.DatetimeIndex,
    vts_df: pd.DataFrame,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Map daily VIX/VIX3M ratio to 1-min bar resolution, shifted 1 day.

    For each trading day D, the ratio is from day D-1 (no lookahead).
    This matches the pattern used by realized vol and daily ADX.

    Args:
        index:  UTC DatetimeIndex aligned with 1-min bar data.
        vts_df: DataFrame from :func:`get_vix_term_structure` with a
                ``ratio`` column indexed by date.

    Returns:
        Float64 numpy array of length ``len(index)``.  NaN where no
        prior-day ratio is available.
    """
    from zoneinfo import ZoneInfo

    et_tz = ZoneInfo("America/New_York")
    n = len(index)
    ratio_out = np.full(n, np.nan)

    if vts_df.empty or "ratio" not in vts_df.columns:
        return ratio_out

    # Build a date → ratio lookup from the VTS DataFrame
    # Shift by 1: bar on day D uses ratio from day D-1
    ratio_series = vts_df["ratio"].copy()
    ratio_series.index = pd.DatetimeIndex(ratio_series.index).date  # type: ignore[assignment]
    ratio_shifted = ratio_series.shift(1)
    ratio_lookup: dict[date, float] = {
        d: float(v) for d, v in ratio_shifted.items() if not np.isnan(float(v))
    }

    # Convert 1-min index to ET dates and map
    idx = index.tz_localize("UTC") if index.tzinfo is None else index
    et_index = idx.tz_convert(et_tz)

    for i, ts in enumerate(et_index):
        d = ts.date()
        if d in ratio_lookup:
            ratio_out[i] = ratio_lookup[d]

    return ratio_out
