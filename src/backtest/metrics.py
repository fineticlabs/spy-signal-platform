"""Performance metrics for backtesting results.

All functions accept the ``_trades`` DataFrame returned by Backtesting.py stats.
Column names used: ``PnL``, ``ReturnPct``, ``EntryTime``, ``ExitTime``.

Functions
---------
- ``compute_metrics`` — full metric dict from trades
- ``print_summary``   — formatted console table
- ``save_equity_curve`` — render equity curve to PNG
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import structlog

if TYPE_CHECKING:
    from pathlib import Path

logger = structlog.get_logger(__name__)


# ── metric computation ────────────────────────────────────────────────────────


def compute_metrics(
    trades: pd.DataFrame,
    equity_curve: pd.Series | None = None,
    trading_days: int | None = None,
) -> dict[str, Any]:
    """Compute performance metrics from a Backtesting.py trades DataFrame.

    Args:
        trades:       ``stats._trades`` DataFrame from ``Backtest.run()``.
                      Must contain columns ``PnL``, ``ReturnPct``,
                      ``EntryTime``, ``ExitTime``.
        equity_curve: Optional equity-curve Series for max drawdown /
                      Sharpe.  If omitted, drawdown is approximated from
                      cumulative P&L and Sharpe is not computed.
        trading_days: Number of calendar days in the test period.  Used to
                      compute avg trades per day.

    Returns:
        Dict of metric names to values.
    """
    if trades is None or len(trades) == 0:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "loss_rate": 0.0,
            "profit_factor": 0.0,
            "expectancy": 0.0,
            "avg_winner": 0.0,
            "avg_loser": 0.0,
            "max_drawdown_pct": 0.0,
            "sharpe_ratio": 0.0,
            "avg_trades_per_day": 0.0,
            "realized_rr": 0.0,
        }

    pnl: pd.Series = trades["PnL"]
    winners = pnl[pnl > 0]
    losers = pnl[pnl < 0]

    total = len(pnl)
    n_wins = len(winners)
    n_losses = len(losers)

    win_rate = n_wins / total if total > 0 else 0.0
    loss_rate = n_losses / total if total > 0 else 0.0

    gross_profit = float(winners.sum()) if len(winners) > 0 else 0.0
    gross_loss = abs(float(losers.sum())) if len(losers) > 0 else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    avg_winner = float(winners.mean()) if len(winners) > 0 else 0.0
    avg_loser = float(abs(losers.mean())) if len(losers) > 0 else 0.0

    expectancy = avg_winner * win_rate - avg_loser * loss_rate

    # Realized average R:R (average winner / average loser)
    realized_rr = avg_winner / avg_loser if avg_loser > 0 else float("inf")

    # Max drawdown
    max_dd = _max_drawdown(equity_curve if equity_curve is not None else _cumulative_equity(pnl))

    # Sharpe ratio (annualized daily returns)
    sharpe = _sharpe_ratio(equity_curve if equity_curve is not None else _cumulative_equity(pnl))

    # Avg trades per day
    avg_per_day: float = 0.0
    if trading_days and trading_days > 0:
        avg_per_day = total / trading_days
    elif "EntryTime" in trades.columns:
        try:
            n_days = trades["EntryTime"].dt.normalize().nunique()
            avg_per_day = total / n_days if n_days > 0 else 0.0
        except Exception:
            avg_per_day = 0.0

    # Time-of-day breakdown (by ET hour of entry)
    tod_breakdown = _time_of_day_breakdown(trades)

    return {
        "total_trades": total,
        "n_wins": n_wins,
        "n_losses": n_losses,
        "win_rate": round(win_rate, 4),
        "loss_rate": round(loss_rate, 4),
        "profit_factor": round(profit_factor, 4),
        "expectancy": round(expectancy, 2),
        "avg_winner": round(avg_winner, 2),
        "avg_loser": round(avg_loser, 2),
        "gross_profit": round(gross_profit, 2),
        "gross_loss": round(gross_loss, 2),
        "max_drawdown_pct": round(max_dd, 4),
        "sharpe_ratio": round(sharpe, 4),
        "avg_trades_per_day": round(avg_per_day, 2),
        "realized_rr": round(realized_rr, 4),
        "tod_breakdown": tod_breakdown,
    }


def _cumulative_equity(pnl: pd.Series) -> pd.Series:
    """Build a synthetic equity curve from P&L series (starting at 0)."""
    return pnl.cumsum().reset_index(drop=True)


def _max_drawdown(equity: pd.Series) -> float:
    """Peak-to-trough max drawdown as a fraction of the running peak.

    Returns a negative number (e.g. -0.15 means 15% drawdown).
    """
    if equity.empty:
        return 0.0
    running_max = equity.cummax()
    # Avoid div-by-zero when peak is 0
    peak = running_max.replace(0, np.nan)
    dd = (equity - running_max) / peak.abs()
    val = float(dd.min())
    return val if not math.isnan(val) else 0.0


def _sharpe_ratio(equity: pd.Series, periods_per_year: int = 252) -> float:
    """Annualized Sharpe ratio from daily equity changes.

    Uses daily returns (assumes one data point per trading day).
    """
    if len(equity) < 2:
        return 0.0
    returns = equity.diff().dropna()
    std = float(returns.std())
    if std == 0:
        return 0.0
    mean = float(returns.mean())
    return float(mean / std * math.sqrt(periods_per_year))


def _time_of_day_breakdown(trades: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Aggregate P&L and win-rate by ET hour of entry."""
    if "EntryTime" not in trades.columns or "PnL" not in trades.columns:
        return {}

    try:
        et_hours = (trades["EntryTime"] - pd.Timedelta(hours=5)).dt.hour
        result: dict[str, dict[str, float]] = {}
        for hour in sorted(et_hours.unique()):
            mask = et_hours == hour
            subset = trades.loc[mask, "PnL"]
            wins = int((subset > 0).sum())
            total = len(subset)
            result[f"{hour:02d}:xx ET"] = {
                "trades": total,
                "win_rate": round(wins / total, 4) if total > 0 else 0.0,
                "total_pnl": round(float(subset.sum()), 2),
            }
        return result
    except Exception as exc:
        logger.warning("tod_breakdown_failed", error=str(exc))
        return {}


# ── display ───────────────────────────────────────────────────────────────────


def print_summary(metrics: dict[str, Any], label: str = "Backtest Results") -> None:
    """Print a formatted metric summary table to stdout."""
    sep = "-" * 50
    print(f"\n{'=' * 50}")
    print(f"  {label}")
    print(sep)

    _row("Total trades", metrics.get("total_trades", 0))
    _row("Win rate", f"{float(metrics.get('win_rate', 0)) * 100:.1f}%")
    _row("Loss rate", f"{float(metrics.get('loss_rate', 0)) * 100:.1f}%")
    _row("Profit factor", f"{metrics.get('profit_factor', 0):.3f}")
    _row("Expectancy ($/trade)", f"${float(metrics.get('expectancy', 0)):.2f}")
    _row("Avg winner", f"${float(metrics.get('avg_winner', 0)):.2f}")
    _row("Avg loser", f"${float(metrics.get('avg_loser', 0)):.2f}")
    _row("Gross profit", f"${float(metrics.get('gross_profit', 0)):.2f}")
    _row("Gross loss", f"${float(metrics.get('gross_loss', 0)):.2f}")
    _row("Max drawdown", f"{float(metrics.get('max_drawdown_pct', 0)) * 100:.2f}%")
    _row("Sharpe ratio", f"{float(metrics.get('sharpe_ratio', 0)):.3f}")
    _row("Realized R:R", f"{float(metrics.get('realized_rr', 0)):.2f}")
    _row("Avg trades/day", f"{float(metrics.get('avg_trades_per_day', 0)):.2f}")

    tod: dict[str, Any] = metrics.get("tod_breakdown", {})
    if tod:
        print(sep)
        print("  Time-of-day breakdown")
        print(sep)
        for period, vals in tod.items():
            print(
                f"  {period}: {vals['trades']} trades, "
                f"WR={vals['win_rate']*100:.0f}%, "
                f"PnL=${vals['total_pnl']:.2f}"
            )

    print("=" * 50)


def _row(label: str, value: object) -> None:
    print(f"  {label:<28} {value}")


# ── equity curve chart ────────────────────────────────────────────────────────


def save_equity_curve(
    equity_curve: pd.Series, output_path: Path, title: str = "Equity Curve"
) -> None:
    """Save a styled equity curve chart to *output_path* (PNG).

    Args:
        equity_curve: Cumulative equity series (index = datetime or int).
        output_path:  File path for the PNG output.
        title:        Chart title.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(equity_curve.values, color="#2196F3", linewidth=1.5, label="Equity")
    ax.axhline(0, color="#9E9E9E", linewidth=0.8, linestyle="--")
    ax.fill_between(
        range(len(equity_curve)),
        equity_curve.values,
        0,
        where=equity_curve.values >= 0,
        alpha=0.15,
        color="#4CAF50",
        label="Profit",
    )
    ax.fill_between(
        range(len(equity_curve)),
        equity_curve.values,
        0,
        where=equity_curve.values < 0,
        alpha=0.15,
        color="#F44336",
        label="Loss",
    )
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Trade #")
    ax.set_ylabel("Cumulative P&L ($)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("equity_curve_saved", path=str(output_path))
