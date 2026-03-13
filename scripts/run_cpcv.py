#!/usr/bin/env python
"""CLI runner for Combinatorial Purged Cross-Validation on ORB backtest results.

Usage
-----
    python scripts/run_cpcv.py [--csv docs/combined_15/backtest_results.csv]
                                [--n-splits 6] [--n-test-splits 2]
                                [--embargo-hours 24] [--out-dir docs]

Loads the trade log CSV and runs CPCV with purging and embargo to produce
C(n_splits, n_test_splits) unique test paths.  Prints per-path PF table
and saves a bar chart.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# ── path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtest.cpcv_validation import (
    print_cpcv_summary,
    run_cpcv,
    save_cpcv_plot,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combinatorial Purged Cross-Validation for ORB backtest"
    )
    parser.add_argument(
        "--csv",
        default="docs/combined_15/backtest_results.csv",
        help="Path to backtest trade log CSV (default: docs/combined_15/backtest_results.csv)",
    )
    parser.add_argument("--n-splits", type=int, default=6, help="Number of folds (default: 6)")
    parser.add_argument(
        "--n-test-splits", type=int, default=2, help="Test folds per combo (default: 2)"
    )
    parser.add_argument(
        "--embargo-hours", type=int, default=24, help="Embargo period in hours (default: 24)"
    )
    parser.add_argument("--out-dir", default="docs", help="Output directory (default: docs)")
    return parser.parse_args()


def main() -> None:
    """Entry point: load CSV, run CPCV, print + save results."""
    args = _parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"Trade log not found: {csv_path}")
        print("Run the backtest first: python scripts/run_backtest.py")
        sys.exit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    trades = pd.read_csv(csv_path)
    required = {"PnL", "EntryTime", "ExitTime"}
    missing = required - set(trades.columns)
    if missing:
        print(f"CSV missing columns: {missing}. Columns found: {list(trades.columns)}")
        sys.exit(1)

    print(f"Loaded {len(trades):,} trades from {csv_path}")

    result = run_cpcv(
        trades=trades,
        n_splits=args.n_splits,
        n_test_splits=args.n_test_splits,
        embargo_hours=args.embargo_hours,
    )

    print_cpcv_summary(result)

    plot_path = out_dir / "cpcv_profit_factor.png"
    save_cpcv_plot(result, plot_path)
    print(f"\nPlot saved to {plot_path}")


if __name__ == "__main__":
    main()
