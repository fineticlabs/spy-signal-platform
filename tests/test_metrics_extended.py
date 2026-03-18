"""Extended tests for src.backtest.metrics."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from src.backtest.metrics import (
    _cumulative_equity,
    _day_of_week_breakdown,
    _max_drawdown,
    _per_year_breakdown,
    _sharpe_ratio,
    _time_of_day_breakdown,
    compute_metrics,
    print_summary,
    save_equity_curve,
)

# ── helpers ──────────────────────────────────────────────────────────────────


def _make_trades(
    pnls: list[float],
    entry_times: list[datetime] | None = None,
) -> pd.DataFrame:
    """Build a minimal trades DataFrame from a list of PnL values."""
    n = len(pnls)
    if entry_times is None:
        # Default: spread across Mon 2024-01-15 starting 14:35 UTC (9:35 ET)
        entry_times = [datetime(2024, 1, 15, 14, 35 + i, tzinfo=UTC) for i in range(n)]
    exit_times = [et + pd.Timedelta(minutes=5) for et in entry_times]
    return pd.DataFrame(
        {
            "PnL": pnls,
            "ReturnPct": [p / 50_000 for p in pnls],
            "EntryTime": entry_times,
            "ExitTime": exit_times,
        }
    )


# ── compute_metrics ──────────────────────────────────────────────────────────


class TestComputeMetrics:
    """Tests for compute_metrics."""

    def test_none_trades_returns_zeros(self) -> None:
        result = compute_metrics(None)
        assert result["total_trades"] == 0
        assert result["win_rate"] == 0.0
        assert result["profit_factor"] == 0.0
        assert result["expectancy"] == 0.0
        assert result["realized_rr"] == 0.0

    def test_empty_dataframe_returns_zeros(self) -> None:
        result = compute_metrics(pd.DataFrame(columns=["PnL", "ReturnPct"]))
        assert result["total_trades"] == 0

    def test_three_wins_one_loss_win_rate(self) -> None:
        trades = _make_trades([100.0, 200.0, 150.0, -80.0])
        result = compute_metrics(trades)
        assert result["win_rate"] == 0.75
        assert result["loss_rate"] == 0.25
        assert result["total_trades"] == 4

    def test_all_losses_profit_factor_zero(self) -> None:
        trades = _make_trades([-50.0, -100.0, -30.0])
        result = compute_metrics(trades)
        assert result["profit_factor"] == 0.0

    def test_all_wins_profit_factor_inf(self) -> None:
        trades = _make_trades([100.0, 200.0])
        result = compute_metrics(trades)
        assert result["profit_factor"] == float("inf")

    def test_expectancy_formula(self) -> None:
        trades = _make_trades([100.0, 200.0, -60.0])
        result = compute_metrics(trades)
        # avg_winner = 150.0, avg_loser = 60.0
        # win_rate = 2/3, loss_rate = 1/3
        # expectancy = 150 * (2/3) - 60 * (1/3) = 100 - 20 = 80
        assert result["expectancy"] == 80.0

    def test_net_profit_equals_gross_profit_minus_gross_loss(self) -> None:
        trades = _make_trades([300.0, 200.0, -50.0, -100.0])
        result = compute_metrics(trades)
        assert result["net_profit"] == result["gross_profit"] - result["gross_loss"]
        assert result["gross_profit"] == 500.0
        assert result["gross_loss"] == 150.0
        assert result["net_profit"] == 350.0

    def test_realized_rr_correct(self) -> None:
        trades = _make_trades([200.0, 100.0, -50.0])
        result = compute_metrics(trades)
        # avg_winner = 150, avg_loser = 50 => rr = 3.0
        assert result["realized_rr"] == 3.0

    def test_realized_rr_inf_when_no_losers(self) -> None:
        trades = _make_trades([100.0, 200.0])
        result = compute_metrics(trades)
        assert result["realized_rr"] == float("inf")

    def test_avg_trades_per_day_from_trading_days(self) -> None:
        trades = _make_trades([100.0, -50.0, 200.0, 80.0])
        result = compute_metrics(trades, trading_days=2)
        assert result["avg_trades_per_day"] == 2.0

    def test_avg_trades_per_day_from_entry_time(self) -> None:
        # 4 trades across 2 unique dates
        entry_times = [
            datetime(2024, 1, 15, 14, 35, tzinfo=UTC),
            datetime(2024, 1, 15, 15, 0, tzinfo=UTC),
            datetime(2024, 1, 16, 14, 35, tzinfo=UTC),
            datetime(2024, 1, 16, 15, 0, tzinfo=UTC),
        ]
        trades = _make_trades([100.0, -50.0, 200.0, -30.0], entry_times=entry_times)
        result = compute_metrics(trades, trading_days=None)
        assert result["avg_trades_per_day"] == 2.0


# ── _time_of_day_breakdown ───────────────────────────────────────────────────


class TestTimeOfDayBreakdown:
    """Tests for _time_of_day_breakdown."""

    def test_groups_by_et_hour(self) -> None:
        # 14:35 UTC = 9:35 ET (EST), 15:10 UTC = 10:10 ET
        entry_times = [
            datetime(2024, 1, 15, 14, 35, tzinfo=UTC),
            datetime(2024, 1, 15, 14, 50, tzinfo=UTC),
            datetime(2024, 1, 15, 15, 10, tzinfo=UTC),
        ]
        trades = _make_trades([100.0, -50.0, 200.0], entry_times=entry_times)
        result = _time_of_day_breakdown(trades)
        assert "09:xx ET" in result
        assert "10:xx ET" in result
        assert result["09:xx ET"]["trades"] == 2
        assert result["10:xx ET"]["trades"] == 1

    def test_empty_when_no_entry_time_column(self) -> None:
        df = pd.DataFrame({"PnL": [100.0]})
        assert _time_of_day_breakdown(df) == {}

    def test_empty_when_no_pnl_column(self) -> None:
        df = pd.DataFrame({"EntryTime": [datetime(2024, 1, 15, 14, 35, tzinfo=UTC)]})
        assert _time_of_day_breakdown(df) == {}


# ── _day_of_week_breakdown ───────────────────────────────────────────────────


class TestDayOfWeekBreakdown:
    """Tests for _day_of_week_breakdown."""

    def test_keys_are_day_names(self) -> None:
        # Mon 2024-01-15, Tue 2024-01-16
        entry_times = [
            datetime(2024, 1, 15, 14, 35, tzinfo=UTC),
            datetime(2024, 1, 16, 14, 35, tzinfo=UTC),
        ]
        trades = _make_trades([100.0, -50.0], entry_times=entry_times)
        result = _day_of_week_breakdown(trades)
        assert "Monday" in result
        assert "Tuesday" in result
        for key in result:
            assert key in ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")


# ── _per_year_breakdown ──────────────────────────────────────────────────────


class TestPerYearBreakdown:
    """Tests for _per_year_breakdown."""

    def test_groups_by_et_year(self) -> None:
        entry_times = [
            datetime(2023, 12, 29, 14, 35, tzinfo=UTC),
            datetime(2024, 1, 2, 14, 35, tzinfo=UTC),
        ]
        trades = _make_trades([100.0, -50.0], entry_times=entry_times)
        result = _per_year_breakdown(trades)
        assert "2023" in result
        assert "2024" in result
        assert result["2023"]["trades"] == 1
        assert result["2024"]["trades"] == 1


# ── _max_drawdown ────────────────────────────────────────────────────────────


class TestMaxDrawdown:
    """Tests for _max_drawdown."""

    def test_monotonic_increase_zero_drawdown(self) -> None:
        eq = pd.Series([0, 100, 200, 300, 400])
        assert _max_drawdown(eq) == 0.0

    def test_initial_capital_normalization(self) -> None:
        # Equity: 0 -> 100 -> -400  (peak shift: 50100, trough shift: 49600)
        eq = pd.Series([0.0, 100.0, -400.0])
        dd = _max_drawdown(eq, initial_capital=50_000.0)
        # shifted: [50000, 50100, 49600]
        # running max: [50000, 50100, 50100]
        # dd: [0, 0, (49600-50100)/50100 = -500/50100]
        expected = -500.0 / 50100.0
        assert abs(dd - expected) < 1e-6

    def test_empty_equity(self) -> None:
        assert _max_drawdown(pd.Series([], dtype=float)) == 0.0


# ── _sharpe_ratio ────────────────────────────────────────────────────────────


class TestSharpeRatio:
    """Tests for _sharpe_ratio."""

    def test_flat_equity_returns_zero(self) -> None:
        eq = pd.Series([100.0, 100.0, 100.0, 100.0])
        assert _sharpe_ratio(eq) == 0.0

    def test_less_than_two_points_returns_zero(self) -> None:
        assert _sharpe_ratio(pd.Series([100.0])) == 0.0
        assert _sharpe_ratio(pd.Series([], dtype=float)) == 0.0

    def test_positive_sharpe_for_uptrend(self) -> None:
        # Non-constant diffs so std > 0; mean of diffs is positive
        eq = pd.Series([0, 10, 25, 32, 48, 55])
        assert _sharpe_ratio(eq) > 0


# ── print_summary ────────────────────────────────────────────────────────────


class TestPrintSummary:
    """Tests for print_summary."""

    def test_runs_without_error(self, capsys: pytest.CaptureFixture[str]) -> None:
        trades = _make_trades([100.0, 200.0, -50.0])
        metrics = compute_metrics(trades)
        print_summary(metrics, label="Test Run")
        captured = capsys.readouterr()
        assert "Test Run" in captured.out
        assert "Win rate" in captured.out

    def test_empty_metrics(self, capsys: pytest.CaptureFixture[str]) -> None:
        metrics = compute_metrics(None)
        print_summary(metrics)
        captured = capsys.readouterr()
        assert "Total trades" in captured.out


# ── save_equity_curve ────────────────────────────────────────────────────────


class TestSaveEquityCurve:
    """Tests for save_equity_curve."""

    def test_creates_file(self, tmp_path: Path) -> None:
        eq = pd.Series([0, 100, 50, 200, 150])
        out = tmp_path / "charts" / "equity.png"
        save_equity_curve(eq, out, title="Test Equity")
        assert out.exists()
        assert out.stat().st_size > 0


# ── _cumulative_equity ───────────────────────────────────────────────────────


class TestCumulativeEquity:
    """Tests for _cumulative_equity."""

    def test_cumsum(self) -> None:
        pnl = pd.Series([100, -50, 200])
        result = _cumulative_equity(pnl)
        assert list(result) == [100, 50, 250]
