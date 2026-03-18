"""Tests for src.backtest.monte_carlo module."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from src.backtest.monte_carlo import (
    MonteCarloResult,
    _max_drawdown_pct,
    _profit_factor,
    _sharpe,
    print_monte_carlo_summary,
    run_monte_carlo,
    save_monte_carlo_plot,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(
    pnl: np.ndarray,
    n_permutations: int = 100,
    seed: int = 42,
) -> MonteCarloResult:
    """Shortcut to build a MonteCarloResult with few permutations for speed."""
    return run_monte_carlo(pnl, n_permutations=n_permutations, seed=seed)


# ---------------------------------------------------------------------------
# _profit_factor
# ---------------------------------------------------------------------------


class TestProfitFactor:
    """Tests for _profit_factor."""

    def test_profit_factor_all_winners(self) -> None:
        pnl = np.array([100.0, 200.0, 50.0])
        assert _profit_factor(pnl) == float("inf")

    def test_profit_factor_all_losers(self) -> None:
        pnl = np.array([-100.0, -200.0, -50.0])
        assert _profit_factor(pnl) == 0.0

    def test_profit_factor_mixed(self) -> None:
        pnl = np.array([300.0, -100.0])
        assert _profit_factor(pnl) == pytest.approx(3.0)

    def test_profit_factor_single_winner(self) -> None:
        pnl = np.array([42.0])
        assert _profit_factor(pnl) == float("inf")

    def test_profit_factor_single_loser(self) -> None:
        pnl = np.array([-42.0])
        assert _profit_factor(pnl) == 0.0

    def test_profit_factor_equal_win_loss(self) -> None:
        pnl = np.array([100.0, -100.0])
        assert _profit_factor(pnl) == pytest.approx(1.0)

    def test_profit_factor_zero_trade_ignored(self) -> None:
        """A trade of exactly 0 is neither profit nor loss."""
        pnl = np.array([0.0, 100.0])
        assert _profit_factor(pnl) == float("inf")


# ---------------------------------------------------------------------------
# _max_drawdown_pct
# ---------------------------------------------------------------------------


class TestMaxDrawdownPct:
    """Tests for _max_drawdown_pct."""

    def test_max_drawdown_monotonic_increase(self) -> None:
        pnl = np.array([100.0, 200.0, 300.0])
        dd = _max_drawdown_pct(pnl)
        assert dd == pytest.approx(0.0)

    def test_max_drawdown_monotonic_decrease(self) -> None:
        pnl = np.array([-100.0, -100.0, -100.0])
        # equity: 49900, 49800, 49700
        # running_max: 49900, 49900, 49900
        # dd: 0, -100/49900, -200/49900
        expected = -200.0 / 49900.0
        assert _max_drawdown_pct(pnl) == pytest.approx(expected, rel=1e-6)

    def test_max_drawdown_v_shape(self) -> None:
        pnl = np.array([-500.0, -500.0, 1500.0])
        # equity: 49500, 49000, 50500
        # running_max: 49500, 49500, 50500
        # dd: -500/50000=-0.01, -1000/49500~-0.0202, 0
        # But initial_capital = 50000, so equity[0] = 49500 which is below start
        # running_max starts at 49500, so dd[0] = 0
        # dd[1] = (49000-49500)/49500
        expected = (49000.0 - 49500.0) / 49500.0
        assert _max_drawdown_pct(pnl) == pytest.approx(expected, rel=1e-6)

    def test_max_drawdown_custom_initial_capital(self) -> None:
        pnl = np.array([-1000.0])
        dd = _max_drawdown_pct(pnl, initial_capital=100_000.0)
        # equity: [99000], running_max: [99000], dd: [0]
        assert dd == pytest.approx(0.0)

    def test_max_drawdown_single_positive_trade(self) -> None:
        pnl = np.array([500.0])
        assert _max_drawdown_pct(pnl) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _sharpe
# ---------------------------------------------------------------------------


class TestSharpe:
    """Tests for _sharpe (annualised Sharpe ratio)."""

    def test_sharpe_fewer_than_two_trades(self) -> None:
        assert _sharpe(np.array([100.0])) == 0.0

    def test_sharpe_empty_array(self) -> None:
        assert _sharpe(np.array([])) == 0.0

    def test_sharpe_flat_returns(self) -> None:
        pnl = np.array([50.0, 50.0, 50.0, 50.0])
        assert _sharpe(pnl) == 0.0

    def test_sharpe_positive_mean_returns_positive(self) -> None:
        pnl = np.array([100.0, 200.0, 150.0, 300.0, 250.0])
        assert _sharpe(pnl) > 0.0

    def test_sharpe_negative_mean_returns_negative(self) -> None:
        pnl = np.array([-100.0, -200.0, -150.0, -50.0])
        assert _sharpe(pnl) < 0.0

    def test_sharpe_custom_periods(self) -> None:
        pnl = np.array([100.0, 200.0, 150.0])
        s252 = _sharpe(pnl, periods_per_year=252)
        s52 = _sharpe(pnl, periods_per_year=52)
        assert s252 > s52  # higher annualisation factor -> larger magnitude


# ---------------------------------------------------------------------------
# run_monte_carlo
# ---------------------------------------------------------------------------


class TestRunMonteCarlo:
    """Tests for run_monte_carlo."""

    @pytest.fixture()
    def sample_pnl(self) -> np.ndarray:
        rng = np.random.default_rng(99)
        return rng.normal(50, 200, size=50)

    def test_run_monte_carlo_result_shapes(self, sample_pnl: np.ndarray) -> None:
        result = _make_result(sample_pnl, n_permutations=200)
        assert result.permuted_pfs.shape == (200,)
        assert result.permuted_net_profits.shape == (200,)
        assert result.permuted_max_dds.shape == (200,)
        assert result.permuted_sharpes.shape == (200,)

    def test_run_monte_carlo_n_trades(self, sample_pnl: np.ndarray) -> None:
        result = _make_result(sample_pnl)
        assert result.n_trades == len(sample_pnl)

    def test_run_monte_carlo_n_permutations(self, sample_pnl: np.ndarray) -> None:
        result = _make_result(sample_pnl, n_permutations=77)
        assert result.n_permutations == 77

    def test_run_monte_carlo_seeded_reproducibility(self, sample_pnl: np.ndarray) -> None:
        r1 = _make_result(sample_pnl, seed=123)
        r2 = _make_result(sample_pnl, seed=123)
        np.testing.assert_array_equal(r1.permuted_pfs, r2.permuted_pfs)
        np.testing.assert_array_equal(r1.permuted_net_profits, r2.permuted_net_profits)

    def test_run_monte_carlo_different_seeds_differ(self, sample_pnl: np.ndarray) -> None:
        r1 = _make_result(sample_pnl, seed=1)
        r2 = _make_result(sample_pnl, seed=2)
        assert not np.array_equal(r1.permuted_pfs, r2.permuted_pfs)

    def test_run_monte_carlo_p_values_between_0_and_1(self, sample_pnl: np.ndarray) -> None:
        result = _make_result(sample_pnl, n_permutations=500)
        assert 0.0 <= result.p_value_pf <= 1.0
        assert 0.0 <= result.p_value_net_profit <= 1.0
        assert 0.0 <= result.p_value_sharpe <= 1.0

    def test_run_monte_carlo_actual_metrics_match_helpers(self, sample_pnl: np.ndarray) -> None:
        result = _make_result(sample_pnl)
        assert result.actual_pf == pytest.approx(_profit_factor(sample_pnl))
        assert result.actual_net_profit == pytest.approx(float(sample_pnl.sum()))
        assert result.actual_max_dd == pytest.approx(_max_drawdown_pct(sample_pnl))
        assert result.actual_sharpe == pytest.approx(_sharpe(sample_pnl))


# ---------------------------------------------------------------------------
# MonteCarloResult p-value properties
# ---------------------------------------------------------------------------


class TestMonteCarloResultProperties:
    """Tests for MonteCarloResult dataclass and p-value properties."""

    def test_p_value_pf_all_above(self) -> None:
        """When all permuted PFs >= actual, p-value should be 1.0."""
        result = MonteCarloResult(
            n_trades=10,
            n_permutations=5,
            actual_pf=1.0,
            actual_net_profit=100.0,
            actual_max_dd=-0.01,
            actual_sharpe=0.5,
            permuted_pfs=np.array([2.0, 3.0, 4.0, 5.0, 6.0]),
            permuted_net_profits=np.array([200.0, 300.0, 400.0, 500.0, 600.0]),
            permuted_max_dds=np.zeros(5),
            permuted_sharpes=np.array([1.0, 1.5, 2.0, 2.5, 3.0]),
        )
        assert result.p_value_pf == pytest.approx(1.0)
        assert result.p_value_net_profit == pytest.approx(1.0)
        assert result.p_value_sharpe == pytest.approx(1.0)

    def test_p_value_pf_none_above(self) -> None:
        """When no permuted PFs >= actual, p-value should be 0.0."""
        result = MonteCarloResult(
            n_trades=10,
            n_permutations=5,
            actual_pf=10.0,
            actual_net_profit=1000.0,
            actual_max_dd=-0.01,
            actual_sharpe=5.0,
            permuted_pfs=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            permuted_net_profits=np.array([100.0, 200.0, 300.0, 400.0, 500.0]),
            permuted_max_dds=np.zeros(5),
            permuted_sharpes=np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
        )
        assert result.p_value_pf == pytest.approx(0.0)
        assert result.p_value_net_profit == pytest.approx(0.0)
        assert result.p_value_sharpe == pytest.approx(0.0)

    def test_p_value_partial(self) -> None:
        """p-value = fraction of permuted values >= actual."""
        result = MonteCarloResult(
            n_trades=10,
            n_permutations=4,
            actual_pf=2.5,
            actual_net_profit=0.0,
            actual_max_dd=0.0,
            actual_sharpe=0.0,
            permuted_pfs=np.array([1.0, 2.0, 3.0, 4.0]),
            permuted_net_profits=np.zeros(4),
            permuted_max_dds=np.zeros(4),
            permuted_sharpes=np.zeros(4),
        )
        # 3.0 and 4.0 are >= 2.5 -> 2/4 = 0.5
        assert result.p_value_pf == pytest.approx(0.5)

    def test_frozen_dataclass(self) -> None:
        result = MonteCarloResult(
            n_trades=5,
            n_permutations=3,
            actual_pf=1.5,
            actual_net_profit=100.0,
            actual_max_dd=-0.02,
            actual_sharpe=0.8,
            permuted_pfs=np.array([1.0, 1.5, 2.0]),
            permuted_net_profits=np.array([50.0, 100.0, 150.0]),
            permuted_max_dds=np.array([-0.01, -0.02, -0.03]),
            permuted_sharpes=np.array([0.5, 0.8, 1.0]),
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.n_trades = 999  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge-case tests for helper functions and run_monte_carlo."""

    def test_profit_factor_empty_array(self) -> None:
        pnl = np.array([])
        # sum of empty = 0 for both profit and loss, so gross_loss == 0 -> inf
        assert _profit_factor(pnl) == float("inf")

    def test_sharpe_empty_array(self) -> None:
        assert _sharpe(np.array([])) == 0.0

    def test_run_monte_carlo_single_trade(self) -> None:
        pnl = np.array([100.0])
        result = _make_result(pnl, n_permutations=50)
        assert result.n_trades == 1
        assert result.actual_sharpe == 0.0  # < 2 trades

    def test_run_monte_carlo_all_same_pnl(self) -> None:
        pnl = np.array([50.0, 50.0, 50.0, 50.0])
        result = _make_result(pnl, n_permutations=50)
        assert result.actual_sharpe == 0.0  # std == 0

    def test_profit_factor_all_zeros(self) -> None:
        pnl = np.array([0.0, 0.0, 0.0])
        assert _profit_factor(pnl) == float("inf")

    def test_max_drawdown_single_negative_trade(self) -> None:
        pnl = np.array([-10_000.0])
        dd = _max_drawdown_pct(pnl, initial_capital=50_000.0)
        # equity = [40000], running_max = [40000], dd = [0.0]
        assert dd == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# save_monte_carlo_plot
# ---------------------------------------------------------------------------


class TestSaveMonteCarloPlot:
    """Tests for save_monte_carlo_plot."""

    def test_save_creates_file(self, tmp_path: Path) -> None:
        pnl = np.array([100.0, -50.0, 200.0, -30.0, 80.0, -10.0])
        result = _make_result(pnl, n_permutations=50)
        output = tmp_path / "mc_plot.png"
        save_monte_carlo_plot(result, output)
        assert output.exists()
        assert output.stat().st_size > 0

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        pnl = np.array([100.0, -50.0, 200.0])
        result = _make_result(pnl, n_permutations=20)
        output = tmp_path / "nested" / "dir" / "plot.png"
        save_monte_carlo_plot(result, output)
        assert output.exists()


# ---------------------------------------------------------------------------
# print_monte_carlo_summary
# ---------------------------------------------------------------------------


class TestPrintMonteCarloSummary:
    """Tests for print_monte_carlo_summary."""

    def test_print_runs_without_error(self, capsys: pytest.CaptureFixture[str]) -> None:
        pnl = np.array([100.0, -50.0, 200.0, -30.0, 80.0])
        result = _make_result(pnl, n_permutations=50)
        print_monte_carlo_summary(result)
        captured = capsys.readouterr()
        assert "Monte Carlo Permutation Test" in captured.out
        assert "Profit Factor" in captured.out
        assert "Verdict" in captured.out

    def test_print_contains_trade_count(self, capsys: pytest.CaptureFixture[str]) -> None:
        pnl = np.array([10.0, -5.0, 20.0])
        result = _make_result(pnl, n_permutations=30)
        print_monte_carlo_summary(result)
        captured = capsys.readouterr()
        assert "3 trades" in captured.out

    def test_print_verdict_not_significant(self, capsys: pytest.CaptureFixture[str]) -> None:
        """With random noise and few permutations, result is typically not significant."""
        rng = np.random.default_rng(0)
        pnl = rng.normal(0, 100, size=20)  # zero-mean -> no edge
        result = _make_result(pnl, n_permutations=200, seed=0)
        print_monte_carlo_summary(result)
        captured = capsys.readouterr()
        # Should contain one of the verdict strings
        assert "Verdict:" in captured.out
