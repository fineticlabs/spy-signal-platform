"""Feature extraction for ML signal confidence scoring.

For every ORB trade that fired in the walk-forward backtest, extract a
feature vector from the bar data at signal time plus meta-context features.
The label is 1 if the trade was profitable (PnL > 0) else 0.

Features extracted
------------------
orb_range_pct    : ORB width / midpoint  (range compactness)
gap_pct          : (today_open - prev_close) / prev_close * 100
volume_ratio     : breakout bar volume / 20-bar avg volume
atr_pct          : ATR(14) / close price  (normalised volatility)
rsi_14           : RSI(14) at signal bar (shifted by 1 — no lookahead)
ema_distance_pct : (close - 15m EMA20) / close
time_minutes     : minutes since 9:30 ET (9:35→5, 9:55→25)
day_of_week      : 0=Mon … 4=Fri  (Mon always 0 though it's filtered)
vix_proxy        : realized vol (20-day rolling HV, annualized)
adx_value        : daily ADX(14) at signal date (shifted 1 day)
consecutive_candles: # consecutive closes past ORB before entry (≥2 always)
ticker_encoded   : integer encoding of ticker symbol
direction        : 1=LONG, 0=SHORT
label            : 1=winner (PnL>0), 0=loser

Lookahead safety
----------------
All indicator values use the same shift-1 convention as the live engine:
the indicator value seen by bar N was computed through bar N-1.

Signal bar vs fill bar
----------------------
Backtesting.py's _trades ``EntryBar`` is the bar at which the order FILLED
(bar N+1), not the bar at which ``next()`` fired and conditions were checked
(bar N).  We use ``signal_bar = EntryBar - 1`` for all feature lookups.

Lookback for vix_proxy / adx_value
------------------------------------
These indicators need 20 / 14 trading days of history respectively.  A
20-day OOS window does not provide enough lookback.  We compute indicators
on the IS+OOS combined slice so the IS period provides the warm-up.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import structlog
import talib

from src.backtest.data_loader import (
    load_bars,
    make_walk_forward_windows,
    slice_window,
)
from src.backtest.engine import (
    _ET_TZ,
    _compute_15m_ema,
    _compute_daily_adx,
    _compute_daily_realized_vol,
    _compute_gap_array,
    _compute_orb_arrays,
    run_backtest,
)
from src.models import TimeFrame

logger = structlog.get_logger(__name__)

# ── constants shared with engine ──────────────────────────────────────────────
_ATR_PERIOD = 14
_RSI_PERIOD = 14
_VOL_WINDOW = 20


# ── indicator array computation ───────────────────────────────────────────────


def _compute_indicator_arrays(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Compute all feature arrays for a bar DataFrame.

    All arrays are shift-1 safe (no lookahead).

    Args:
        df: OHLCV DataFrame with UTC DatetimeIndex (lowercase columns).

    Returns:
        Dict of name → numpy array, same length as df.
    """
    index: pd.DatetimeIndex = df.index  # type: ignore[assignment]
    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    volume = df["volume"].to_numpy(dtype=float)
    open_ = df["open"].to_numpy(dtype=float)

    # ATR(14) — shift 1 to avoid lookahead
    atr_raw = talib.ATR(high, low, close, timeperiod=_ATR_PERIOD)
    atr_arr = np.roll(atr_raw, 1)
    atr_arr[0] = np.nan

    # Rolling volume mean (20-bar) — shift 1
    avg_vol_raw = pd.Series(volume).rolling(_VOL_WINDOW).mean().to_numpy()
    avg_vol_arr = np.roll(avg_vol_raw, 1)
    avg_vol_arr[0] = np.nan

    # RSI(14) — shift 1
    rsi_raw = talib.RSI(close, timeperiod=_RSI_PERIOD)
    rsi_arr = np.roll(rsi_raw, 1)
    rsi_arr[0] = np.nan

    # ORB arrays (NaN inside ORB bars, valid after)
    orb_high_arr, orb_low_arr = _compute_orb_arrays(index, high, low)

    # ORB range as % of midpoint
    midpoint = (orb_high_arr + orb_low_arr) / 2.0
    orb_range_pct_arr = np.where(
        (midpoint > 0) & ~np.isnan(orb_high_arr) & ~np.isnan(orb_low_arr),
        (orb_high_arr - orb_low_arr) / midpoint,
        np.nan,
    )

    # 15-min EMA(20) — shift 1 inside _compute_15m_ema
    ema15m_arr = _compute_15m_ema(index, close)

    # Daily gap pct (no lookahead: gap known at 9:30, signal fires ≥9:35)
    gap_pct_arr = _compute_gap_array(index, open_, close)

    # Realized vol (20-day rolling HV, shifted 1 day)
    realized_vol_arr = _compute_daily_realized_vol(index, close)

    # Daily ADX(14) — shifted 1 day
    adx_arr = _compute_daily_adx(index, high, low, close)

    return {
        "atr": atr_arr,
        "avg_vol": avg_vol_arr,
        "rsi14": rsi_arr,
        "orb_high": orb_high_arr,
        "orb_low": orb_low_arr,
        "orb_range_pct": orb_range_pct_arr,
        "ema15m": ema15m_arr,
        "gap_pct": gap_pct_arr,
        "realized_vol": realized_vol_arr,
        "adx": adx_arr,
    }


# ── consecutive-candle counter ────────────────────────────────────────────────


def _count_consecutive(
    bar_idx: int,
    close_arr: np.ndarray,
    orb_high_arr: np.ndarray,
    orb_low_arr: np.ndarray,
    direction: int,
) -> int:
    """Count consecutive closes past the ORB level ending at bar_idx.

    Uses the ORB arrays (not a scalar) so the ORB value is looked up per bar,
    correctly handling day-boundary transitions within the window.

    Args:
        bar_idx:    Index of the signal bar.
        close_arr:  Close price numpy array.
        orb_high_arr: Pre-computed ORB high array (same length as close_arr).
        orb_low_arr:  Pre-computed ORB low array.
        direction:  1=LONG (count > orb_high), 0=SHORT (count < orb_low).

    Returns:
        Number of consecutive closes past ORB.  0 if the condition fails
        immediately at bar_idx.
    """
    count = 0
    idx = bar_idx
    while idx >= 0:
        c = close_arr[idx]
        orb_h = orb_high_arr[idx]
        orb_l = orb_low_arr[idx]
        if np.isnan(c) or np.isnan(orb_h) or np.isnan(orb_l):
            break
        if direction == 1 and c > orb_h or direction == 0 and c < orb_l:
            count += 1
        else:
            break
        idx -= 1
    return count


# ── single-trade feature builder ──────────────────────────────────────────────


def _build_feature_row(
    signal_bar_idx: int,
    combined_df: pd.DataFrame,
    arrays: dict[str, np.ndarray],
    symbol: str,
    direction: int,
    pnl: float,
    window_idx: int,
) -> dict[str, Any] | None:
    """Build one feature dict from the signal bar.

    Args:
        signal_bar_idx: Position in combined_df where the signal fired
                        (= EntryBar - 1, i.e. the bar where next() was called).
        combined_df:    IS+OOS combined DataFrame (provides close history
                        for consecutive-candle counting).
        arrays:         Pre-computed indicator arrays from combined_df.
        symbol:         Ticker symbol string.
        direction:      1=LONG, 0=SHORT.
        pnl:            Realized P&L for the trade.
        window_idx:     Walk-forward window index (for purged CV).

    Returns:
        Feature dict or None if critical values are missing.
    """
    if signal_bar_idx < 0 or signal_bar_idx >= len(combined_df):
        return None

    close = float(combined_df["close"].iloc[signal_bar_idx])
    volume = float(combined_df["volume"].iloc[signal_bar_idx])

    atr = float(arrays["atr"][signal_bar_idx])
    avg_vol = float(arrays["avg_vol"][signal_bar_idx])
    rsi14 = float(arrays["rsi14"][signal_bar_idx])
    orb_range_pct = float(arrays["orb_range_pct"][signal_bar_idx])
    ema15m = float(arrays["ema15m"][signal_bar_idx])
    gap_pct = float(arrays["gap_pct"][signal_bar_idx])
    realized_vol = float(arrays["realized_vol"][signal_bar_idx])
    adx = float(arrays["adx"][signal_bar_idx])

    if close <= 0:
        return None

    # Derived features
    volume_ratio = volume / avg_vol if (avg_vol > 0 and not np.isnan(avg_vol)) else np.nan
    atr_pct = atr / close if (not np.isnan(atr) and close > 0) else np.nan
    ema_distance_pct = (close - ema15m) / close if (not np.isnan(ema15m) and close > 0) else np.nan

    # Time features from the signal bar's timestamp
    ts: pd.Timestamp = combined_df.index[signal_bar_idx]  # type: ignore[assignment]
    ts_et = ts.tz_convert(_ET_TZ)
    time_minutes = int(ts_et.hour * 60 + ts_et.minute - (9 * 60 + 30))
    day_of_week = int(ts_et.dayofweek)  # 0=Mon … 4=Fri

    # Consecutive candle count (uses ORB arrays to handle day boundaries correctly)
    close_arr = combined_df["close"].to_numpy(dtype=float)
    consecutive = _count_consecutive(
        signal_bar_idx,
        close_arr,
        arrays["orb_high"],
        arrays["orb_low"],
        direction,
    )

    return {
        "symbol": symbol,
        "entry_time": ts,
        "window_idx": window_idx,
        "direction": direction,
        "orb_range_pct": orb_range_pct,
        "gap_pct": gap_pct,
        "volume_ratio": volume_ratio,
        "atr_pct": atr_pct,
        "rsi_14": rsi14,
        "ema_distance_pct": ema_distance_pct,
        "time_minutes": time_minutes,
        "day_of_week": day_of_week,
        "vix_proxy": realized_vol,
        "adx_value": adx,
        "consecutive_candles": float(consecutive),
        "pnl": pnl,
        "label": 1 if pnl > 0 else 0,
    }


# ── per-symbol feature extraction ─────────────────────────────────────────────


def extract_features_for_symbol(
    db: Any,  # BarDatabase — typed as Any to avoid circular import
    symbol: str,
    is_days: int = 60,
    oos_days: int = 20,
) -> pd.DataFrame:
    """Extract ML features for all trades on one ticker.

    Runs the same walk-forward split as the main backtest.  For each window:
    - Indicator arrays are computed on the IS+OOS combined slice so that the
      IS period provides lookback for vix_proxy (20-day) and adx (14-day).
    - The backtest runs on the OOS slice only.
    - EntryBar (fill bar N+1) is mapped back to the signal bar (N) for features.

    Args:
        db:       Open BarDatabase instance.
        symbol:   Ticker symbol.
        is_days:  In-sample window size in calendar days.
        oos_days: Out-of-sample window size in calendar days.

    Returns:
        DataFrame of feature rows, one per trade.  Empty if no data/trades.
    """
    logger.info("extracting_features", symbol=symbol)
    df_1min = load_bars(db, symbol=symbol, timeframe=TimeFrame.ONE_MIN)
    if df_1min.empty:
        logger.warning("no_bars_for_symbol", symbol=symbol)
        return pd.DataFrame()

    windows = make_walk_forward_windows(df_1min, is_days, oos_days)
    if not windows:
        return pd.DataFrame()

    all_rows: list[dict[str, Any]] = []

    for win_idx, window in enumerate(windows):
        is_df, oos_df = slice_window(df_1min, window)
        if oos_df.empty or len(oos_df) < 30:
            continue

        # Combine IS+OOS for indicator computation — IS provides lookback warmup
        combined_df = pd.concat([is_df, oos_df]).sort_index()
        is_len = len(is_df)

        # Pre-compute all indicator arrays on the combined IS+OOS window
        arrays = _compute_indicator_arrays(combined_df)

        # Run the backtest on OOS only
        try:
            stats = run_backtest(oos_df)
        except Exception as exc:
            logger.warning("feature_backtest_failed", symbol=symbol, window=win_idx, error=str(exc))
            continue

        trades: pd.DataFrame = stats["_trades"]
        if len(trades) == 0:
            continue

        for _, trade in trades.iterrows():
            fill_bar_in_oos = int(trade["EntryBar"])
            # Signal bar is one before the fill bar (Backtesting.py fills at N+1)
            signal_bar_in_oos = fill_bar_in_oos - 1
            # Map to position in the combined IS+OOS array
            signal_bar_in_combined = is_len + signal_bar_in_oos

            # Size > 0 → long, Size < 0 → short
            direction = 1 if float(trade["Size"]) > 0 else 0
            pnl = float(trade["PnL"])

            row = _build_feature_row(
                signal_bar_idx=signal_bar_in_combined,
                combined_df=combined_df,
                arrays=arrays,
                symbol=symbol,
                direction=direction,
                pnl=pnl,
                window_idx=win_idx,
            )
            if row is not None:
                all_rows.append(row)

    logger.info("features_extracted", symbol=symbol, trades=len(all_rows))
    return pd.DataFrame(all_rows)


# ── multi-symbol feature extraction ──────────────────────────────────────────


# Canonical list of ticker symbols; used for label encoding consistency
TICKER_LIST: list[str] = ["SPY", "QQQ", "MSFT", "AMD", "TSLA", "META", "AMZN"]

# Feature columns in canonical order (matches training + inference)
FEATURE_COLS: list[str] = [
    "orb_range_pct",
    "gap_pct",
    "volume_ratio",
    "atr_pct",
    "rsi_14",
    "ema_distance_pct",
    "time_minutes",
    "day_of_week",
    "vix_proxy",
    "adx_value",
    "consecutive_candles",
    "ticker_encoded",
    "direction",
]


def encode_ticker(symbol: str) -> int:
    """Map ticker symbol to a stable integer index.

    Args:
        symbol: Uppercase ticker string.

    Returns:
        Integer in [0, len(TICKER_LIST)-1], or len(TICKER_LIST) for unknowns.
    """
    try:
        return TICKER_LIST.index(symbol)
    except ValueError:
        return len(TICKER_LIST)  # unknown ticker


def extract_all_features(
    db: Any,
    symbols: list[str],
    is_days: int = 60,
    oos_days: int = 20,
) -> pd.DataFrame:
    """Extract features for all symbols, combine, and encode categoricals.

    Args:
        db:       Open BarDatabase instance.
        symbols:  List of ticker symbols.
        is_days:  In-sample window size.
        oos_days: Out-of-sample window size.

    Returns:
        Combined DataFrame with all feature columns plus label, symbol,
        entry_time, window_idx, pnl.  Sorted by entry_time ascending.
    """
    frames: list[pd.DataFrame] = []
    for sym in symbols:
        df = extract_features_for_symbol(db, sym, is_days, oos_days)
        if not df.empty:
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)

    # Encode ticker symbol as integer
    combined["ticker_encoded"] = combined["symbol"].apply(encode_ticker)

    # Sort by entry time (critical for time-series CV)
    combined = combined.sort_values("entry_time").reset_index(drop=True)

    logger.info(
        "all_features_extracted",
        total_trades=len(combined),
        symbols=symbols,
        win_rate=round(float(combined["label"].mean()), 3),
    )
    return combined
