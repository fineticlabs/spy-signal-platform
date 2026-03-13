"""Monte Carlo sign-randomisation test for ORB backtest edge validation.

Tests the null hypothesis that the strategy has **no directional edge**
(i.e. each trade's sign is a coin flip).  For each of 10,000 permutations
we randomly flip the sign (+/-) of each trade's PnL independently with
50% probability, then recompute profit factor, net profit, max drawdown,
and Sharpe ratio.

If the actual metric sits in the extreme right tail of the null
distribution (p-value < 0.05), the edge is statistically significant.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import structlog

logger = structlog.get_logger(__name__)

_DEFAULT_N_PERMUTATIONS = 10_000
_SEED = 42


@dataclass(frozen=True)
class MonteCarloResult:
    """Holds actual vs. permuted metric distributions."""

    n_trades: int
    n_permutations: int

    actual_pf: float
    actual_net_profit: float
    actual_max_dd: float
    actual_sharpe: float

    permuted_pfs: np.ndarray
    permuted_net_profits: np.ndarray
    permuted_max_dds: np.ndarray
    permuted_sharpes: np.ndarray

    @property
    def p_value_pf(self) -> float:
        """Fraction of permuted PFs >= actual PF (one-tailed)."""
        return float(np.mean(self.permuted_pfs >= self.actual_pf))

    @property
    def p_value_net_profit(self) -> float:
        return float(np.mean(self.permuted_net_profits >= self.actual_net_profit))

    @property
    def p_value_sharpe(self) -> float:
        return float(np.mean(self.permuted_sharpes >= self.actual_sharpe))


def _profit_factor(pnl: np.ndarray) -> float:
    """Gross profit / gross loss.  Returns inf when gross loss is zero."""
    gross_profit = float(pnl[pnl > 0].sum())
    gross_loss = float(np.abs(pnl[pnl < 0]).sum())
    if gross_loss == 0:
        return float("inf")
    return gross_profit / gross_loss


def _max_drawdown_pct(pnl: np.ndarray, initial_capital: float = 50_000.0) -> float:
    """Peak-to-trough max drawdown as a fraction of initial capital."""
    equity = np.cumsum(pnl) + initial_capital
    running_max = np.maximum.accumulate(equity)
    dd = (equity - running_max) / running_max
    return float(dd.min())


def _sharpe(pnl: np.ndarray, periods_per_year: int = 252) -> float:
    """Annualised Sharpe from per-trade PnL (one trade ~ one day)."""
    if len(pnl) < 2:
        return 0.0
    std = float(pnl.std())
    if std == 0:
        return 0.0
    return float(pnl.mean() / std * math.sqrt(periods_per_year))


def run_monte_carlo(
    pnl: np.ndarray,
    n_permutations: int = _DEFAULT_N_PERMUTATIONS,
    seed: int = _SEED,
    initial_capital: float = 50_000.0,
) -> MonteCarloResult:
    """Run the Monte Carlo sign-randomisation test.

    For each permutation, randomly flip the sign of each trade's PnL
    (50/50 coin flip) to simulate a no-edge null hypothesis.  The
    absolute magnitudes are preserved — only the win/loss assignment
    is randomised.

    Args:
        pnl:             1-D array of per-trade PnL values.
        n_permutations:  Number of random sign-flips (default 10,000).
        seed:            RNG seed for reproducibility.
        initial_capital: Starting capital for drawdown normalisation.

    Returns:
        :class:`MonteCarloResult` with actual metrics and null distributions.
    """
    rng = np.random.default_rng(seed)

    # Actual metrics
    actual_pf = _profit_factor(pnl)
    actual_net = float(pnl.sum())
    actual_dd = _max_drawdown_pct(pnl, initial_capital)
    actual_sh = _sharpe(pnl)

    logger.info(
        "monte_carlo_start",
        n_trades=len(pnl),
        n_permutations=n_permutations,
        actual_pf=round(actual_pf, 4),
        actual_net_profit=round(actual_net, 2),
    )

    # Pre-allocate result arrays
    perm_pfs = np.empty(n_permutations)
    perm_nets = np.empty(n_permutations)
    perm_dds = np.empty(n_permutations)
    perm_sharpes = np.empty(n_permutations)

    abs_pnl = np.abs(pnl)
    for i in range(n_permutations):
        # Randomly flip each trade's sign with 50% probability
        signs = rng.choice(np.array([-1.0, 1.0]), size=len(pnl))
        randomised = abs_pnl * signs
        perm_pfs[i] = _profit_factor(randomised)
        perm_nets[i] = float(randomised.sum())
        perm_dds[i] = _max_drawdown_pct(randomised, initial_capital)
        perm_sharpes[i] = _sharpe(randomised)

    result = MonteCarloResult(
        n_trades=len(pnl),
        n_permutations=n_permutations,
        actual_pf=actual_pf,
        actual_net_profit=actual_net,
        actual_max_dd=actual_dd,
        actual_sharpe=actual_sh,
        permuted_pfs=perm_pfs,
        permuted_net_profits=perm_nets,
        permuted_max_dds=perm_dds,
        permuted_sharpes=perm_sharpes,
    )

    logger.info(
        "monte_carlo_done",
        p_value_pf=round(result.p_value_pf, 4),
        p_value_sharpe=round(result.p_value_sharpe, 4),
    )

    return result


def save_monte_carlo_plot(result: MonteCarloResult, output_path: Path) -> None:
    """Save a 2x2 histogram grid of permuted metric distributions.

    Each subplot shows the null distribution with a vertical red line at the
    actual observed value and annotated p-value.

    Args:
        result:      :class:`MonteCarloResult` from :func:`run_monte_carlo`.
        output_path: File path for the PNG output.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Monte Carlo Permutation Test — {result.n_trades:,} trades, "
        f"{result.n_permutations:,} permutations",
        fontsize=14,
        fontweight="bold",
    )

    panels = [
        (axes[0, 0], result.permuted_pfs, result.actual_pf, result.p_value_pf, "Profit Factor"),
        (
            axes[0, 1],
            result.permuted_net_profits,
            result.actual_net_profit,
            result.p_value_net_profit,
            "Net Profit ($)",
        ),
        (
            axes[1, 0],
            result.permuted_sharpes,
            result.actual_sharpe,
            result.p_value_sharpe,
            "Sharpe Ratio",
        ),
        (
            axes[1, 1],
            result.permuted_max_dds * 100,
            result.actual_max_dd * 100,
            None,
            "Max Drawdown (%)",
        ),
    ]

    for ax, data, actual, pval, title in panels:
        # Filter out infinities for histogram
        finite = data[np.isfinite(data)]
        ax.hist(finite, bins=80, color="#90CAF9", edgecolor="#42A5F5", alpha=0.8)
        ax.axvline(
            actual, color="#D32F2F", linewidth=2, linestyle="--", label=f"Actual: {actual:.3f}"
        )

        if pval is not None:
            sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.10 else ""
            ax.text(
                0.97,
                0.95,
                f"p = {pval:.4f} {sig}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=12,
                fontweight="bold",
                color="#D32F2F" if pval < 0.05 else "#FF9800",
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8},
            )

        ax.set_title(title, fontsize=12)
        ax.legend(loc="upper left")
        ax.grid(alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("monte_carlo_plot_saved", path=str(output_path))


def print_monte_carlo_summary(result: MonteCarloResult) -> None:
    """Print a formatted summary of the Monte Carlo test to stdout."""
    sep = "-" * 60
    print(f"\n{'=' * 60}")
    print("  Monte Carlo Permutation Test")
    print(f"  {result.n_trades:,} trades / {result.n_permutations:,} permutations")
    print(sep)

    rows = [
        ("Profit Factor", result.actual_pf, np.median(result.permuted_pfs), result.p_value_pf),
        (
            "Net Profit",
            result.actual_net_profit,
            float(np.median(result.permuted_net_profits)),
            result.p_value_net_profit,
        ),
        (
            "Sharpe Ratio",
            result.actual_sharpe,
            float(np.median(result.permuted_sharpes)),
            result.p_value_sharpe,
        ),
    ]

    print(f"  {'Metric':<20} {'Actual':>10} {'Median Perm':>12} {'p-value':>10} {'Sig?':>6}")
    print(sep)
    for name, actual, median_perm, pval in rows:
        sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.10 else ""
        if name == "Net Profit":
            print(f"  {name:<20} ${actual:>9,.2f} ${median_perm:>10,.2f} {pval:>10.4f} {sig:>6}")
        else:
            print(f"  {name:<20} {actual:>10.4f} {median_perm:>12.4f} {pval:>10.4f} {sig:>6}")

    print(sep)
    print(f"  Max Drawdown (actual): {result.actual_max_dd * 100:.2f}%")
    print(f"  Max Drawdown (median permuted): {np.median(result.permuted_max_dds) * 100:.2f}%")
    print(sep)

    pf_pval = result.p_value_pf
    if pf_pval < 0.01:
        verdict = "HIGHLY SIGNIFICANT — edge is real (p < 0.01)"
    elif pf_pval < 0.05:
        verdict = "SIGNIFICANT — edge is likely real (p < 0.05)"
    elif pf_pval < 0.10:
        verdict = "MARGINAL — edge may exist but not conclusive (p < 0.10)"
    else:
        verdict = "NOT SIGNIFICANT — edge may be random (p >= 0.10)"

    print(f"  Verdict: {verdict}")
    print("=" * 60)
