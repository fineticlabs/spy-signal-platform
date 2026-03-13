#!/usr/bin/env python
"""CLI runner for the Monte Carlo permutation test on ORB backtest results.

Usage
-----
    python scripts/run_monte_carlo.py [--csv docs/combined_15/backtest_results.csv]
                                       [--permutations 10000]
                                       [--out-dir docs]
                                       [--cash 50000]

Loads the trade log CSV, extracts the PnL column, runs the permutation test,
prints a summary, and saves the distribution plot.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtest.monte_carlo import (
    print_monte_carlo_summary,
    run_monte_carlo,
    save_monte_carlo_plot,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monte Carlo permutation test for ORB backtest")
    parser.add_argument(
        "--csv",
        default="docs/combined_15/backtest_results.csv",
        help="Path to backtest trade log CSV (default: docs/combined_15/backtest_results.csv)",
    )
    parser.add_argument(
        "--permutations",
        type=int,
        default=10_000,
        help="Number of permutations (default: 10000)",
    )
    parser.add_argument("--out-dir", default="docs", help="Output directory (default: docs)")
    parser.add_argument(
        "--cash", type=float, default=50_000.0, help="Starting cash for drawdown (default: 50000)"
    )
    return parser.parse_args()


def main() -> None:
    """Entry point: load CSV, run Monte Carlo, print + save results."""
    args = _parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"Trade log not found: {csv_path}")
        print("Run the backtest first: python scripts/run_backtest.py")
        sys.exit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load trade PnL
    trades = pd.read_csv(csv_path)
    if "PnL" not in trades.columns:
        print(f"CSV missing 'PnL' column. Columns found: {list(trades.columns)}")
        sys.exit(1)

    pnl = trades["PnL"].dropna().values.astype(np.float64)
    print(f"Loaded {len(pnl):,} trades from {csv_path}")

    # Run permutation test
    result = run_monte_carlo(
        pnl=pnl,
        n_permutations=args.permutations,
        initial_capital=args.cash,
    )

    # Print summary
    print_monte_carlo_summary(result)

    # Save plot
    plot_path = out_dir / "monte_carlo_pf_distribution.png"
    save_monte_carlo_plot(result, plot_path)
    print(f"\nPlot saved to {plot_path}")


if __name__ == "__main__":
    main()
