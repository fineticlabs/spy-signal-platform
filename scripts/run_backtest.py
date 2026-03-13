#!/usr/bin/env python
"""CLI walk-forward backtest runner — multi-symbol ORB.

Usage
-----
    python scripts/run_backtest.py [--symbols SPY,QQQ] [--is-days 60] [--oos-days 20]
                                   [--cash 50000] [--out-dir docs] [--timeframe 1Min]

The script:
  1. Loads 1-min bars from the SQLite database (configured via .env) for each symbol.
  2. Splits the date range into walk-forward IS/OOS windows per symbol.
  3. Runs the ORB backtest on each OOS window.
  4. Prints a per-ticker summary table flagging weak symbols.
  5. Computes combined metrics across all tickers.
  6. Saves a combined equity curve to ``docs/equity_curve.png``.
  7. Saves the full combined trade log to ``docs/backtest_results.csv``.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
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
from src.backtest.engine import _compute_first5min_rvol, run_backtest
from src.backtest.metrics import compute_metrics, print_summary, save_equity_curve
from src.models import TimeFrame
from src.storage.database import BarDatabase

logger = structlog.get_logger(__name__)

# ── flag thresholds ───────────────────────────────────────────────────────────
_MIN_TRADES_THRESHOLD: int = 10
_MIN_EXPECTANCY_THRESHOLD: float = 0.0


# ── data classes ──────────────────────────────────────────────────────────────


@dataclass
class TickerResult:
    """Aggregated backtest result for a single ticker symbol."""

    symbol: str
    trades: pd.DataFrame
    pnl: list[float] = field(default_factory=list)
    windows_count: int = 0


# ── argument parsing ──────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Walk-forward ORB backtest runner (multi-symbol)")
    parser.add_argument(
        "--symbols",
        default="SPY,QQQ,MSFT,AMD,TSLA,AMZN,UBER,SMCI,SHOP,PLTR,NFLX,MSTR,SNOW,ARM,DASH",
        help="Comma-separated list of ticker symbols (default: 15-ticker portfolio)",
    )
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


# ── per-ticker logic ──────────────────────────────────────────────────────────


def _run_symbol(
    db: BarDatabase,
    sym: str,
    tf: TimeFrame,
    is_days: int,
    oos_days: int,
    cash: float,
) -> TickerResult | None:
    """Load bars, walk-forward split, and run backtest for a single symbol.

    Args:
        db:       Open :class:`~src.storage.database.BarDatabase` instance.
        sym:      Ticker symbol string.
        tf:       Target bar timeframe.
        is_days:  In-sample window size in calendar days.
        oos_days: Out-of-sample window size in calendar days.
        cash:     Starting cash for the backtest engine.

    Returns:
        A :class:`TickerResult` with aggregated trades, or ``None`` if the
        symbol has no data or produces no valid windows.
    """
    logger.info("loading_bars", symbol=sym, timeframe=tf.value)
    df_1min = load_bars(db, symbol=sym, timeframe=TimeFrame.ONE_MIN)

    if df_1min.empty:
        print(f"\n[{sym}] No 1-min bars found. Run scripts/backfill_data.py first.")
        return None

    # Pre-compute first-5-min RVOL on full history so OOS slices inherit values
    rvol_arr = _compute_first5min_rvol(df_1min.index, df_1min["volume"].values)
    df_1min["RVOL"] = rvol_arr

    df = resample(df_1min, tf) if tf != TimeFrame.ONE_MIN else df_1min

    windows = make_walk_forward_windows(
        df_1min,
        in_sample_days=is_days,
        out_of_sample_days=oos_days,
    )

    if not windows:
        print(
            f"\n[{sym}] Not enough data for walk-forward splits. "
            f"Need at least {is_days + oos_days} calendar days of data."
        )
        return None

    print(f"\nRunning {len(windows)} walk-forward window(s) on {sym}...")

    all_trades: list[pd.DataFrame] = []
    combined_pnl: list[float] = []

    for i, window in enumerate(windows, start=1):
        print(f"  [{i}/{len(windows)}] {window}")

        _, oos_df = slice_window(df, window)
        if oos_df.empty:
            logger.warning("empty_oos_window", symbol=sym, window=str(window))
            continue

        try:
            stats = run_backtest(oos_df, cash=cash, symbol=sym)
            trades: pd.DataFrame = stats["_trades"]

            if len(trades) > 0:
                trades = trades.copy()
                trades["window"] = i
                trades["ticker"] = sym
                all_trades.append(trades)
                combined_pnl.extend(trades["PnL"].tolist())

            print(
                f"    Trades={stats['# Trades']}, "
                f"Return={stats['Return [%]']:.1f}%, "
                f"WinRate={stats['Win Rate [%]']:.0f}%"
            )

        except Exception as exc:
            logger.error("window_backtest_failed", symbol=sym, window=str(window), error=str(exc))
            print(f"    ERROR: {exc}")

    if not all_trades:
        print(f"\n[{sym}] No trades generated across all windows.")
        return TickerResult(
            symbol=sym,
            trades=pd.DataFrame(),
            pnl=[],
            windows_count=len(windows),
        )

    combined_trades = pd.concat(all_trades, ignore_index=True)
    return TickerResult(
        symbol=sym,
        trades=combined_trades,
        pnl=combined_pnl,
        windows_count=len(windows),
    )


# ── per-ticker summary table ──────────────────────────────────────────────────


def _compute_ticker_row(result: TickerResult) -> dict[str, object]:
    """Compute display metrics for one ticker's summary row.

    Args:
        result: A :class:`TickerResult` produced by :func:`_run_symbol`.

    Returns:
        Dict with keys: symbol, total_trades, win_rate, profit_factor,
        expectancy, net_pnl, flag.
    """
    trades = result.trades
    if trades.empty:
        return {
            "symbol": result.symbol,
            "total_trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "expectancy": 0.0,
            "net_pnl": 0.0,
            "flag": "**REMOVE**",
        }

    pnl: pd.Series = trades["PnL"]
    total = len(pnl)
    winners = pnl[pnl > 0]
    losers = pnl[pnl < 0]

    win_rate = len(winners) / total if total > 0 else 0.0
    loss_rate = len(losers) / total if total > 0 else 0.0

    gross_profit = float(winners.sum()) if len(winners) > 0 else 0.0
    gross_loss = abs(float(losers.sum())) if len(losers) > 0 else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    avg_winner = float(winners.mean()) if len(winners) > 0 else 0.0
    avg_loser = float(abs(losers.mean())) if len(losers) > 0 else 0.0
    expectancy = avg_winner * win_rate - avg_loser * loss_rate
    net_pnl = gross_profit - gross_loss

    should_remove = total < _MIN_TRADES_THRESHOLD or expectancy <= _MIN_EXPECTANCY_THRESHOLD
    flag = "**REMOVE**" if should_remove else ""

    return {
        "symbol": result.symbol,
        "total_trades": total,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "net_pnl": net_pnl,
        "flag": flag,
    }


def _print_ticker_summary(results: list[TickerResult]) -> None:
    """Print the per-ticker summary table to stdout.

    Rows with fewer than ``_MIN_TRADES_THRESHOLD`` trades or non-positive
    expectancy are flagged with ``**REMOVE**``.

    Args:
        results: List of :class:`TickerResult` objects from all symbols.
    """
    rows = [_compute_ticker_row(r) for r in results]

    sep = "-" * 80
    header = (
        f"{'Symbol':<8}  {'Trades':>6}  {'WinRate':>8}  "
        f"{'PF':>6}  {'Expect':>9}  {'NetPnL':>11}  {'Flag'}"
    )
    print(f"\n{'=' * 80}")
    print("  Per-Ticker Summary")
    print(sep)
    print(f"  {header}")
    print(sep)

    for row in rows:
        win_pct = f"{float(row['win_rate']) * 100:.1f}%"
        pf_val = float(row["profit_factor"])
        pf_str = f"{pf_val:.3f}" if pf_val != float("inf") else "inf"
        print(
            f"  {row['symbol']!s:<8}  "
            f"{int(row['total_trades']):>6}  "
            f"{win_pct:>8}  "
            f"{pf_str:>6}  "
            f"${float(row['expectancy']):>8.2f}  "
            f"${float(row['net_pnl']):>10.2f}  "
            f"{row['flag']!s}"
        )

    print("=" * 80)


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    """Entry point: parse args, run per-symbol backtests, print results, save outputs."""
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    symbols: list[str] = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    if not symbols:
        print("No symbols provided. Use --symbols SPY,QQQ")
        sys.exit(1)

    try:
        tf = TimeFrame(args.timeframe)
    except ValueError:
        logger.error("invalid_timeframe", value=args.timeframe)
        sys.exit(1)

    db = BarDatabase()
    db.connect()

    ticker_results: list[TickerResult] = []

    for sym in symbols:
        result = _run_symbol(
            db=db,
            sym=sym,
            tf=tf,
            is_days=args.is_days,
            oos_days=args.oos_days,
            cash=args.cash,
        )
        if result is not None:
            ticker_results.append(result)

    db.close()

    if not ticker_results:
        print("\nNo results produced for any symbol.")
        sys.exit(0)

    # ── per-ticker summary table ───────────────────────────────────────────────
    _print_ticker_summary(ticker_results)

    # ── combined metrics across all tickers ───────────────────────────────────
    results_with_trades = [r for r in ticker_results if not r.trades.empty]

    if not results_with_trades:
        print("\nNo trades generated across all symbols.")
        sys.exit(0)

    all_trade_frames: list[pd.DataFrame] = [r.trades for r in results_with_trades]
    combined_trades = pd.concat(all_trade_frames, ignore_index=True)

    combined_pnl_all: list[float] = []
    for r in results_with_trades:
        combined_pnl_all.extend(r.pnl)

    equity = pd.Series(combined_pnl_all).cumsum()

    metrics = compute_metrics(
        trades=combined_trades,
        equity_curve=equity,
    )
    label = "Walk-Forward Results — " + ", ".join(r.symbol for r in results_with_trades)
    print_summary(metrics, label=label)

    # ── save outputs ───────────────────────────────────────────────────────────
    curve_path = out_dir / "equity_curve.png"
    title = "ORB Walk-Forward Equity Curve (" + ", ".join(symbols) + ")"
    save_equity_curve(equity, curve_path, title=title)
    print(f"\nEquity curve saved to {curve_path}")

    csv_path = out_dir / "backtest_results.csv"
    combined_trades.to_csv(csv_path, index=False)
    print(f"Trade log saved to {csv_path}")


if __name__ == "__main__":
    main()
