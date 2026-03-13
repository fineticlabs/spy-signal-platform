"""Combinatorial Purged Cross-Validation (CPCV) for ORB backtest robustness.

Uses ``CombPurgedKFoldCV`` from the *timeseriescv* package (Marcos Lopez de
Prado, *Advances in Financial Machine Learning*, 2018) to generate all
C(n_splits, n_test_splits) unique train/test combinations with purging and
embargo so that train and test intervals never overlap.

With n_splits=6, n_test_splits=2 this produces C(6,2) = 15 unique test
paths.  For each path we compute profit factor on the held-out test trades.

Interpretation:
- Median PF > 1.2 across all 15 paths  =>  edge is robust
- Any path with PF < 1.0               =>  note which period failed
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import structlog
from timeseriescv.cross_validation import CombPurgedKFoldCV

if TYPE_CHECKING:
    from pathlib import Path

logger = structlog.get_logger(__name__)

_DEFAULT_N_SPLITS = 6
_DEFAULT_N_TEST_SPLITS = 2


@dataclass(frozen=True)
class PathResult:
    """Metrics for a single CPCV test path."""

    path_id: int
    test_folds: tuple[int, ...]
    n_trades: int
    win_rate: float
    profit_factor: float
    net_pnl: float
    date_range: str


@dataclass
class CPCVResult:
    """Aggregated results across all CPCV paths."""

    n_splits: int
    n_test_splits: int
    n_paths: int
    total_trades: int
    paths: list[PathResult] = field(default_factory=list)

    @property
    def pfs(self) -> np.ndarray:
        return np.array([p.profit_factor for p in self.paths])

    @property
    def min_pf(self) -> float:
        return float(self.pfs.min())

    @property
    def max_pf(self) -> float:
        return float(self.pfs.max())

    @property
    def median_pf(self) -> float:
        return float(np.median(self.pfs))

    @property
    def mean_pf(self) -> float:
        return float(self.pfs.mean())

    @property
    def failing_paths(self) -> list[PathResult]:
        return [p for p in self.paths if p.profit_factor < 1.0]


def _profit_factor(pnl: pd.Series) -> float:
    """Gross profit / gross loss.  Returns inf when no losers."""
    winners = pnl[pnl > 0]
    losers = pnl[pnl < 0]
    gross_profit = float(winners.sum()) if len(winners) > 0 else 0.0
    gross_loss = abs(float(losers.sum())) if len(losers) > 0 else 0.0
    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def run_cpcv(
    trades: pd.DataFrame,
    n_splits: int = _DEFAULT_N_SPLITS,
    n_test_splits: int = _DEFAULT_N_TEST_SPLITS,
    embargo_hours: int = 24,
) -> CPCVResult:
    """Run Combinatorial Purged Cross-Validation on the trade log.

    Args:
        trades:         DataFrame with ``PnL``, ``EntryTime``, ``ExitTime`` columns.
        n_splits:       Number of folds (default 6).
        n_test_splits:  Number of test folds per combination (default 2).
        embargo_hours:  Embargo period in hours between test and train (default 24).

    Returns:
        :class:`CPCVResult` with per-path metrics.
    """
    df = trades.copy()

    # Parse timestamps
    df["EntryTime"] = pd.to_datetime(df["EntryTime"], utc=True)
    df["ExitTime"] = pd.to_datetime(df["ExitTime"], utc=True)

    # Sort by entry time for proper temporal ordering
    df = df.sort_values("EntryTime").reset_index(drop=True)

    # CombPurgedKFoldCV needs: X (DataFrame), pred_times (Series), eval_times (Series)
    pred_times = df["EntryTime"]
    eval_times = df["ExitTime"]

    cv = CombPurgedKFoldCV(
        n_splits=n_splits,
        n_test_splits=n_test_splits,
        embargo_td=pd.Timedelta(hours=embargo_hours),
    )

    from math import comb

    expected_paths = comb(n_splits, n_test_splits)

    logger.info(
        "cpcv_start",
        n_splits=n_splits,
        n_test_splits=n_test_splits,
        expected_paths=expected_paths,
        total_trades=len(df),
    )

    path_results: list[PathResult] = []
    path_id = 0

    for _train_idx, test_idx in cv.split(X=df, pred_times=pred_times, eval_times=eval_times):
        test_trades = df.iloc[test_idx]
        pnl = test_trades["PnL"]

        n_test = len(pnl)
        if n_test == 0:
            continue

        wins = int((pnl > 0).sum())
        wr = wins / n_test
        pf = _profit_factor(pnl)
        net = float(pnl.sum())

        # Date range for this test set
        earliest = test_trades["EntryTime"].min().strftime("%Y-%m-%d")
        latest = test_trades["EntryTime"].max().strftime("%Y-%m-%d")
        date_range = f"{earliest} .. {latest}"

        # Determine which fold indices make up this test set
        fold_size = len(df) // n_splits
        test_folds = tuple(sorted({idx // fold_size for idx in test_idx if idx < len(df)}))

        path_results.append(
            PathResult(
                path_id=path_id,
                test_folds=test_folds,
                n_trades=n_test,
                win_rate=round(wr, 4),
                profit_factor=round(pf, 4),
                net_pnl=round(net, 2),
                date_range=date_range,
            )
        )
        path_id += 1

    result = CPCVResult(
        n_splits=n_splits,
        n_test_splits=n_test_splits,
        n_paths=len(path_results),
        total_trades=len(df),
        paths=path_results,
    )

    logger.info(
        "cpcv_done",
        n_paths=result.n_paths,
        median_pf=round(result.median_pf, 4),
        min_pf=round(result.min_pf, 4),
        failing_paths=len(result.failing_paths),
    )

    return result


def print_cpcv_summary(result: CPCVResult) -> None:
    """Print a formatted CPCV results table to stdout."""
    sep = "-" * 90
    print(f"\n{'=' * 90}")
    print(
        f"  Combinatorial Purged Cross-Validation — "
        f"C({result.n_splits},{result.n_test_splits}) = {result.n_paths} paths, "
        f"{result.total_trades:,} trades"
    )
    print(sep)

    header = (
        f"  {'Path':>4}  {'Folds':>10}  {'Trades':>6}  "
        f"{'WR':>7}  {'PF':>7}  {'NetPnL':>11}  {'Date Range'}"
    )
    print(header)
    print(sep)

    for p in result.paths:
        folds_str = ",".join(str(f) for f in p.test_folds)
        wr_str = f"{p.win_rate * 100:.1f}%"
        pf_str = f"{p.profit_factor:.3f}" if p.profit_factor != float("inf") else "inf"
        flag = " << FAIL" if p.profit_factor < 1.0 else ""
        print(
            f"  {p.path_id:>4}  {folds_str:>10}  {p.n_trades:>6}  "
            f"{wr_str:>7}  {pf_str:>7}  ${p.net_pnl:>10,.2f}  "
            f"{p.date_range}{flag}"
        )

    print(sep)
    print(f"  Min PF:    {result.min_pf:.4f}")
    print(f"  Max PF:    {result.max_pf:.4f}")
    print(f"  Median PF: {result.median_pf:.4f}")
    print(f"  Mean PF:   {result.mean_pf:.4f}")
    print(sep)

    failing = result.failing_paths
    if failing:
        print(f"  WARNING: {len(failing)} path(s) with PF < 1.0:")
        for p in failing:
            folds_str = ",".join(str(f) for f in p.test_folds)
            print(
                f"    Path {p.path_id} (folds {folds_str}): PF={p.profit_factor:.3f}, {p.date_range}"
            )
    else:
        print("  All 15 paths profitable (PF > 1.0)")

    print(sep)
    if result.median_pf > 1.2:
        verdict = "ROBUST — median PF > 1.2 across all paths"
    elif result.median_pf > 1.0:
        verdict = "MARGINAL — median PF > 1.0 but below 1.2 threshold"
    else:
        verdict = "WEAK — median PF <= 1.0, edge may not be robust"

    print(f"  Verdict: {verdict}")
    print("=" * 90)


def save_cpcv_plot(result: CPCVResult, output_path: Path) -> None:
    """Save a bar chart of per-path profit factors.

    Args:
        result:      :class:`CPCVResult` from :func:`run_cpcv`.
        output_path: File path for the PNG output.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 6))

    pfs = [p.profit_factor for p in result.paths]
    labels = [",".join(str(f) for f in p.test_folds) for p in result.paths]
    colors = ["#4CAF50" if pf >= 1.0 else "#F44336" for pf in pfs]

    bars = ax.bar(range(len(pfs)), pfs, color=colors, edgecolor="#333", alpha=0.85)

    # Reference lines
    ax.axhline(
        1.0, color="#F44336", linewidth=1.5, linestyle="--", alpha=0.7, label="Break-even (PF=1.0)"
    )
    ax.axhline(
        1.2,
        color="#FF9800",
        linewidth=1.5,
        linestyle="--",
        alpha=0.7,
        label="Robust threshold (PF=1.2)",
    )
    ax.axhline(
        result.median_pf,
        color="#2196F3",
        linewidth=2,
        linestyle="-",
        alpha=0.8,
        label=f"Median PF={result.median_pf:.3f}",
    )

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_xlabel("Test Fold Combination", fontsize=11)
    ax.set_ylabel("Profit Factor", fontsize=11)
    ax.set_title(
        f"CPCV Profit Factor — C({result.n_splits},{result.n_test_splits}) = {result.n_paths} paths, "
        f"{result.total_trades:,} trades",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    # Annotate each bar with PF value
    for bar, pf in zip(bars, pfs, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{pf:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("cpcv_plot_saved", path=str(output_path))
