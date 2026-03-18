#!/usr/bin/env python
"""Replay a single trading day against the current ORB strategy pipeline.

Usage
-----
    python scripts/replay_day.py [--date 2026-03-13] [--symbols SPY,QQQ,...] [--cash 50000]

Shows every signal that would have fired on the target date, with entry/stop/target
prices and the eventual outcome (hit target, hit stop, or EOD exit with P&L).

Uses the exact same pipeline as ``run_backtest.py``: 60-day IS training window
for HMM regime, Kalman filter stops, all filters (time windows, EMA trend, gap,
Monday, consecutive candle, realized vol, ADX, economic calendar, Volume Profile
HVN, VIX term structure).
"""

from __future__ import annotations

import argparse
import sys
import warnings
from datetime import date, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import structlog

# ── path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtest.data_loader import WalkForwardWindow, load_bars, slice_window
from src.backtest.engine import (
    _compute_first5min_rvol,
    compute_intraday_vwap,
    run_backtest,
)
from src.filters.vix_term_structure import (
    compute_vix_term_structure_array,
    get_vix_term_structure,
)
from src.models import TimeFrame
from src.storage.database import BarDatabase
from src.strategies.hmm_regime import compute_hmm_regime_for_window

# Suppress sklearn/hmmlearn RuntimeWarnings (convergence, matmul overflow)
warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"sklearn\..*")
warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"hmmlearn\..*")

logger = structlog.get_logger(__name__)

_ET_TZ = ZoneInfo("America/New_York")
_IS_DAYS = 60


def _get_default_symbols() -> list[str]:
    """Return the canonical symbol list from AppSettings, with hardcoded fallback."""
    try:
        from src.config import get_app_settings

        return get_app_settings().symbols
    except Exception:
        return [
            "SPY",
            "QQQ",
            "MSFT",
            "AMD",
            "TSLA",
            "AMZN",
            "UBER",
            "SMCI",
            "SHOP",
            "PLTR",
            "NFLX",
            "MSTR",
            "SNOW",
            "ARM",
            "DASH",
            "PYPL",
            "INTC",
            "MU",
            "HOOD",
            "DKNG",
            "SOXL",
            "ROKU",
            "TQQQ",
            "BA",
            "MRVL",
            "META",
        ]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay a single trading day through the ORB pipeline"
    )
    parser.add_argument(
        "--date",
        default="2026-03-13",
        help="Target date to replay (YYYY-MM-DD, default: 2026-03-13)",
    )
    default_syms = _get_default_symbols()
    default_syms_str = ",".join(default_syms)
    parser.add_argument(
        "--symbols",
        default=default_syms_str,
        help=f"Comma-separated ticker symbols (default: {default_syms_str})",
    )
    parser.add_argument(
        "--cash", type=float, default=50_000.0, help="Starting cash (default: 50000)"
    )
    return parser.parse_args()


def _outcome_label(trade: pd.Series) -> str:
    """Determine if the trade hit target, hit stop, or was closed at EOD."""
    exit_price = float(trade["ExitPrice"])
    sl = trade["SL"]
    tp = trade["TP"]
    # Check if exit price matches SL or TP (within slippage tolerance)
    tol = 0.05
    if pd.notna(tp) and abs(exit_price - float(tp)) < tol:
        return "TARGET"
    if pd.notna(sl) and abs(exit_price - float(sl)) < tol:
        return "STOP"
    # If neither, it was a time-based exit (EOD forced flat)
    return "EOD"


def _replay_symbol(
    db: BarDatabase,
    sym: str,
    target_date: date,
    cash: float,
    spy_ref: pd.DataFrame | None,
    vts_df: pd.DataFrame | None,
) -> pd.DataFrame | None:
    """Replay a single symbol on the target date.

    Returns a DataFrame of trades, or None if no data / no trades.
    """
    df_1min = load_bars(db, symbol=sym, timeframe=TimeFrame.ONE_MIN)
    if df_1min.empty:
        return None

    # Pre-compute RVOL on full history
    rvol_arr = _compute_first5min_rvol(df_1min.index, df_1min["volume"].values)
    df_1min["RVOL"] = rvol_arr

    # Merge SPY reference for market direction
    if spy_ref is not None and sym not in {"SPY", "QQQ"}:
        df_1min = df_1min.join(spy_ref[["SPY_VWAP", "SPY_CLOSE"]], how="left")
        df_1min["SPY_VWAP"] = df_1min["SPY_VWAP"].ffill()
        df_1min["SPY_CLOSE"] = df_1min["SPY_CLOSE"].ffill()

    # Merge VIX term structure
    if vts_df is not None and not vts_df.empty:
        vts_arr = compute_vix_term_structure_array(df_1min.index, vts_df)
        df_1min["VIX_TERM_RATIO"] = vts_arr

    # Build the walk-forward window.
    # OOS needs ~20 warmup days before the target for ATR/EMA/volume indicators
    # to be primed.  We include extra days in OOS and filter trades to target date.
    warmup_days = 20
    is_start = target_date - timedelta(days=_IS_DAYS + warmup_days)
    is_end = target_date - timedelta(days=warmup_days + 1)
    oos_start = target_date - timedelta(days=warmup_days)

    window = WalkForwardWindow(
        in_sample_start=is_start,
        in_sample_end=is_end,
        out_of_sample_start=oos_start,
        out_of_sample_end=target_date,
    )

    is_1min, oos_1min = slice_window(df_1min, window)
    _, oos_df = slice_window(df_1min, window)

    if oos_df.empty:
        return None

    # Train HMM on IS, predict on OOS
    hmm_regime_arr = compute_hmm_regime_for_window(is_1min, oos_1min)
    hmm_series = pd.Series(hmm_regime_arr, index=oos_1min.index, dtype=float)
    oos_df = oos_df.copy()
    oos_df["HMM_REGIME"] = hmm_series.reindex(oos_df.index, method="ffill").fillna(1.0)

    try:
        stats = run_backtest(oos_df, cash=cash, symbol=sym)
    except Exception as exc:
        logger.error("replay_failed", symbol=sym, error=str(exc))
        return None

    trades: pd.DataFrame = stats["_trades"]
    if trades.empty:
        return None

    # Filter trades to only those entered on the target date
    trades = trades.copy()
    trades["_entry_date"] = pd.to_datetime(trades["EntryTime"]).dt.tz_convert(_ET_TZ).dt.date
    trades = trades[trades["_entry_date"] == target_date].drop(columns=["_entry_date"])

    if trades.empty:
        return None

    trades["ticker"] = sym
    return trades


def main() -> None:
    """Entry point: replay target date across all symbols."""
    args = _parse_args()
    target_date = date.fromisoformat(args.date)
    symbols: list[str] = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    cash = args.cash

    print(f"\n{'=' * 90}")
    print(f"  ORB Day Replay — {target_date.strftime('%A, %B %d, %Y')}")
    print(
        f"  IS training window: {target_date - timedelta(days=_IS_DAYS)} to {target_date - timedelta(days=1)}"
    )
    print(f"  Symbols: {len(symbols)}")
    print(f"{'=' * 90}")

    db = BarDatabase()
    db.connect()

    # Load SPY reference
    spy_ref: pd.DataFrame | None = None
    spy_1min = load_bars(db, symbol="SPY", timeframe=TimeFrame.ONE_MIN)
    if not spy_1min.empty:
        spy_vwap = compute_intraday_vwap(
            spy_1min.index,
            spy_1min["high"].values,
            spy_1min["low"].values,
            spy_1min["close"].values,
            spy_1min["volume"].values,
        )
        spy_ref = pd.DataFrame(
            {"SPY_VWAP": spy_vwap, "SPY_CLOSE": spy_1min["close"].values},
            index=spy_1min.index,
        )

    # Load VIX term structure
    vts_df = get_vix_term_structure()

    all_trades: list[pd.DataFrame] = []

    for sym in symbols:
        trades = _replay_symbol(db, sym, target_date, cash, spy_ref, vts_df)
        if trades is not None and not trades.empty:
            all_trades.append(trades)

    db.close()

    if not all_trades:
        print(f"\n  No signals fired on {target_date}.\n")
        return

    combined = pd.concat(all_trades, ignore_index=True)
    combined = combined.sort_values("EntryTime").reset_index(drop=True)

    _print_trade_table(combined)


def _format_duration(td: pd.Timedelta) -> str:
    """Format a timedelta as MM:SS."""
    total_secs = int(td.total_seconds())
    minutes, seconds = divmod(abs(total_secs), 60)
    return f"{minutes:>3d}:{seconds:02d}"


def _compute_peak_concurrent(combined: pd.DataFrame) -> int:
    """Compute the peak number of overlapping positions from entry/exit times."""
    events: list[tuple[pd.Timestamp, int]] = []
    for _, trade in combined.iterrows():
        entry_ts = pd.Timestamp(trade["EntryTime"])
        exit_ts = pd.Timestamp(trade["ExitTime"])
        events.append((entry_ts, 1))
        events.append((exit_ts, -1))
    # Sort by time; on ties, exits (-1) before entries (+1) so a slot freed
    # at the same bar as a new entry doesn't double-count.
    events.sort(key=lambda e: (e[0], e[1]))
    peak = 0
    current = 0
    for _, delta in events:
        current += delta
        peak = max(peak, current)
    return peak


def _print_trade_table(combined: pd.DataFrame) -> None:
    """Print the formatted trade table with entry/exit times and duration."""
    sep = "-" * 130
    print(
        f"\n  {'#':>2}  {'Entry':>8}  {'Exit':>8}  {'Ticker':<6}  {'Dir':<5}  "
        f"{'Entry$':>9}  {'Stop$':>9}  {'Target$':>9}  {'Exit$':>9}  "
        f"{'Outcome':<7}  {'Dur':>6}  {'P&L':>10}  {'Shares':>6}  {'Position$':>10}"
    )
    print(f"  {sep}")

    total_pnl = 0.0
    total_position = 0.0
    wins = 0
    losses = 0

    for i, (_, trade) in enumerate(combined.iterrows(), start=1):
        entry_time = pd.Timestamp(trade["EntryTime"]).astimezone(_ET_TZ)
        exit_time = pd.Timestamp(trade["ExitTime"]).astimezone(_ET_TZ)
        entry_str = entry_time.strftime("%H:%M:%S")
        exit_str = exit_time.strftime("%H:%M:%S")
        duration = (
            pd.Timedelta(trade["Duration"])
            if pd.notna(trade.get("Duration"))
            else exit_time - entry_time
        )
        dur_str = _format_duration(duration)
        ticker = str(trade["ticker"])
        size = int(trade["Size"])
        shares = abs(size)
        direction = "LONG" if size > 0 else "SHORT"
        entry_price = float(trade["EntryPrice"])
        exit_price = float(trade["ExitPrice"])
        position_val = shares * entry_price
        sl_raw = trade["SL"]
        tp_raw = trade["TP"]
        sl_str = f"${float(sl_raw):>8.2f}" if pd.notna(sl_raw) else f"{'N/A':>9}"
        tp_str = f"${float(tp_raw):>8.2f}" if pd.notna(tp_raw) else f"{'N/A':>9}"
        pnl = float(trade["PnL"])
        outcome = _outcome_label(trade)

        total_pnl += pnl
        total_position += position_val
        if pnl >= 0:
            wins += 1
        else:
            losses += 1

        pnl_str = f"${pnl:>+9.2f}"
        print(
            f"  {i:>2}  {entry_str:>8}  {exit_str:>8}  {ticker:<6}  {direction:<5}  "
            f"${entry_price:>8.2f}  {sl_str}  {tp_str}  ${exit_price:>8.2f}  "
            f"{outcome:<7}  {dur_str}  {pnl_str}  {shares:>6}  ${position_val:>9,.0f}"
        )

    # Summary
    total = wins + losses
    wr = wins / total * 100 if total > 0 else 0.0
    pnl_sign = "+" if total_pnl >= 0 else ""
    peak_concurrent = _compute_peak_concurrent(combined)
    avg_position = total_position / total if total > 0 else 0.0

    print(f"  {sep}")
    print("\n  Day Summary")
    print(f"  {'-' * 40}")
    print(f"  Signals:              {total}")
    print(f"  Wins:                 {wins}")
    print(f"  Losses:               {losses}")
    print(f"  Win Rate:             {wr:.0f}%")
    print(f"  Net P&L:              {pnl_sign}${total_pnl:.2f}")
    print(f"  Max concurrent:       {peak_concurrent}")
    print(f"  Total capital deployed: ${total_position:,.0f}")
    print(f"  Avg position size:    ${avg_position:,.0f}")
    print()


if __name__ == "__main__":
    main()
