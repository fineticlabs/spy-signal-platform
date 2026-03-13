#!/usr/bin/env python
"""CLI walk-forward backtest runner.

Usage
-----
    python scripts/run_backtest.py [--symbol SPY] [--is-days 60] [--oos-days 20]
                                   [--cash 50000] [--out-dir docs]

The script:
  1. Loads 1-min bars from the SQLite database (configured via .env).
  2. Splits the date range into walk-forward IS/OOS windows.
  3. Runs the ORB backtest on each OOS window (parameters fit on IS).
  4. Prints a combined metric summary.
  5. Saves a combined equity curve to ``docs/equity_curve.png``.
  6. Saves the full trade log to ``docs/backtest_results.csv``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import structlog

# ── path setup ────────────────────────────────────────────────────────────────
# Allow running from the repo root without installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtest.data_loader import (
    load_bars,
    make_walk_forward_windows,
    resample,
    slice_window,
)
from src.backtest.engine import run_backtest
from src.backtest.metrics import compute_metrics, print_summary, save_equity_curve
from src.models import TimeFrame
from src.storage.database import BarDatabase

logger = structlog.get_logger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Walk-forward ORB backtest runner")
    parser.add_argument("--symbol", default="SPY", help="Ticker symbol (default: SPY)")
    parser.add_argument(
        "--is-days", type=int, default=60, help="In-sample window days (default: 60)"
    )
    parser.add_argument(
        "--oos-days", type=int, default=20, help="Out-of-sample window days (default: 20)"
    )
    parser.add_argument(
        "--cash", type=float, default=50_000.0, help="Starting cash (default: 50000)"
    )
    parser.add_argument("--out-dir", default="docs", help="Output directory (default: docs)")
    parser.add_argument("--timeframe", default="1Min", help="Bar timeframe (default: 1Min)")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── load data ─────────────────────────────────────────────────────────────
    db = BarDatabase()
    db.connect()

    try:
        tf = TimeFrame(args.timeframe)
    except ValueError:
        logger.error("invalid_timeframe", value=args.timeframe)
        sys.exit(1)

    logger.info("loading_bars", symbol=args.symbol, timeframe=tf.value)
    df_1min = load_bars(db, symbol=args.symbol, timeframe=TimeFrame.ONE_MIN)
    db.close()

    if df_1min.empty:
        print(f"No 1-min bars found for {args.symbol}. Run scripts/backfill_data.py first.")
        sys.exit(1)

    # Resample to the requested timeframe for the engine
    df = resample(df_1min, tf) if tf != TimeFrame.ONE_MIN else df_1min

    # ── walk-forward windows ───────────────────────────────────────────────────
    windows = make_walk_forward_windows(
        df_1min,
        in_sample_days=args.is_days,
        out_of_sample_days=args.oos_days,
    )

    if not windows:
        print(
            "Not enough data for walk-forward splits. Need at least "
            f"{args.is_days + args.oos_days} calendar days of data."
        )
        sys.exit(1)

    print(f"\nRunning {len(windows)} walk-forward window(s) on {args.symbol}...")

    all_trades: list[pd.DataFrame] = []
    combined_pnl: list[float] = []

    for i, window in enumerate(windows, start=1):
        print(f"\n  [{i}/{len(windows)}] {window}")

        _, oos_df = slice_window(df, window)
        if oos_df.empty:
            logger.warning("empty_oos_window", window=str(window))
            continue

        try:
            stats = run_backtest(oos_df, cash=args.cash)
            trades: pd.DataFrame = stats["_trades"]

            if len(trades) > 0:
                trades = trades.copy()
                trades["window"] = i
                all_trades.append(trades)
                combined_pnl.extend(trades["PnL"].tolist())

            print(
                f"    Trades={stats['# Trades']}, "
                f"Return={stats['Return [%]']:.1f}%, "
                f"WinRate={stats['Win Rate [%]']:.0f}%"
            )

        except Exception as exc:
            logger.error("window_backtest_failed", window=str(window), error=str(exc))
            print(f"    ERROR: {exc}")

    if not all_trades:
        print("\nNo trades generated across all windows.")
        sys.exit(0)

    # ── combined metrics ───────────────────────────────────────────────────────
    combined_trades = pd.concat(all_trades, ignore_index=True)
    equity = pd.Series(combined_pnl).cumsum()
    trading_days = sum((w.out_of_sample_end - w.out_of_sample_start).days for w in windows)

    metrics = compute_metrics(
        trades=combined_trades,
        equity_curve=equity,
        trading_days=trading_days,
    )
    print_summary(metrics, label=f"Walk-Forward Results — {args.symbol}")

    # ── save outputs ───────────────────────────────────────────────────────────
    curve_path = out_dir / "equity_curve.png"
    save_equity_curve(equity, curve_path, title=f"{args.symbol} ORB Walk-Forward Equity Curve")
    print(f"\nEquity curve saved to {curve_path}")

    csv_path = out_dir / "backtest_results.csv"
    combined_trades.to_csv(csv_path, index=False)
    print(f"Trade log saved to {csv_path}")


if __name__ == "__main__":
    main()
