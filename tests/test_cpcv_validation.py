"""Tests for src.backtest.cpcv_validation module."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.backtest.cpcv_validation import (
    CPCVResult,
    PathResult,
    _profit_factor,
    print_cpcv_summary,
    run_cpcv,
    save_cpcv_plot,
)

# ── fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture()
def synthetic_trades() -> pd.DataFrame:
    """200 synthetic trades spanning several weeks."""
    rng = np.random.default_rng(42)
    n = 200
    return pd.DataFrame(
        {
            "PnL": rng.normal(50, 200, n),
            "EntryTime": pd.date_range("2024-01-02", periods=n, freq="4h", tz="UTC"),
            "ExitTime": pd.date_range("2024-01-02", periods=n, freq="4h", tz="UTC")
            + pd.Timedelta(hours=2),
        }
    )


@pytest.fixture()
def sample_paths() -> list[PathResult]:
    return [
        PathResult(0, (0, 1), 20, 0.55, 1.5, 100.0, "2024-01-01 .. 2024-01-10"),
        PathResult(1, (0, 2), 25, 0.48, 0.8, -50.0, "2024-01-01 .. 2024-01-20"),
        PathResult(2, (1, 2), 30, 0.60, 2.0, 200.0, "2024-01-10 .. 2024-01-20"),
    ]


@pytest.fixture()
def sample_cpcv_result(sample_paths: list[PathResult]) -> CPCVResult:
    return CPCVResult(
        n_splits=6,
        n_test_splits=2,
        n_paths=3,
        total_trades=75,
        paths=sample_paths,
    )


# ── PathResult ───────────────────────────────────────────────────────────────


class TestPathResult:
    def test_frozen_dataclass(self) -> None:
        pr = PathResult(0, (0, 1), 10, 0.5, 1.2, 50.0, "2024-01-01 .. 2024-01-05")
        with pytest.raises(FrozenInstanceError):
            pr.path_id = 99  # type: ignore[misc]

    def test_fields_stored(self) -> None:
        pr = PathResult(3, (1, 3), 15, 0.6, 1.4, 75.0, "2024-02-01 .. 2024-02-10")
        assert pr.path_id == 3
        assert pr.test_folds == (1, 3)
        assert pr.n_trades == 15
        assert pr.win_rate == 0.6
        assert pr.profit_factor == 1.4
        assert pr.net_pnl == 75.0


# ── CPCVResult properties ────────────────────────────────────────────────────


class TestCPCVResult:
    def test_pfs_returns_array(self, sample_cpcv_result: CPCVResult) -> None:
        pfs = sample_cpcv_result.pfs
        assert isinstance(pfs, np.ndarray)
        assert len(pfs) == 3
        np.testing.assert_array_almost_equal(pfs, [1.5, 0.8, 2.0])

    def test_min_pf(self, sample_cpcv_result: CPCVResult) -> None:
        assert sample_cpcv_result.min_pf == pytest.approx(0.8)

    def test_max_pf(self, sample_cpcv_result: CPCVResult) -> None:
        assert sample_cpcv_result.max_pf == pytest.approx(2.0)

    def test_median_pf(self, sample_cpcv_result: CPCVResult) -> None:
        assert sample_cpcv_result.median_pf == pytest.approx(1.5)

    def test_mean_pf(self, sample_cpcv_result: CPCVResult) -> None:
        expected = (1.5 + 0.8 + 2.0) / 3
        assert sample_cpcv_result.mean_pf == pytest.approx(expected)

    def test_failing_paths(self, sample_cpcv_result: CPCVResult) -> None:
        failing = sample_cpcv_result.failing_paths
        assert len(failing) == 1
        assert failing[0].path_id == 1
        assert failing[0].profit_factor == 0.8

    def test_no_failing_paths(self) -> None:
        result = CPCVResult(
            n_splits=6,
            n_test_splits=2,
            n_paths=2,
            total_trades=50,
            paths=[
                PathResult(0, (0, 1), 25, 0.6, 1.5, 100.0, "2024-01-01 .. 2024-01-10"),
                PathResult(1, (0, 2), 25, 0.55, 1.2, 50.0, "2024-01-01 .. 2024-01-20"),
            ],
        )
        assert len(result.failing_paths) == 0


# ── _profit_factor ───────────────────────────────────────────────────────────


class TestProfitFactor:
    def test_mixed_winners_losers(self) -> None:
        pnl = pd.Series([100.0, 50.0, -30.0, -20.0])
        # gross_profit=150, gross_loss=50
        assert _profit_factor(pnl) == pytest.approx(3.0)

    def test_all_winners_returns_inf(self) -> None:
        pnl = pd.Series([100.0, 50.0, 25.0])
        assert _profit_factor(pnl) == float("inf")

    def test_no_profit_no_loss_returns_zero(self) -> None:
        pnl = pd.Series([0.0, 0.0, 0.0])
        assert _profit_factor(pnl) == 0.0

    def test_all_losers(self) -> None:
        pnl = pd.Series([-100.0, -50.0])
        assert _profit_factor(pnl) == 0.0

    def test_single_winner(self) -> None:
        pnl = pd.Series([100.0])
        assert _profit_factor(pnl) == float("inf")

    def test_single_loser(self) -> None:
        pnl = pd.Series([-100.0])
        assert _profit_factor(pnl) == 0.0


# ── run_cpcv ─────────────────────────────────────────────────────────────────


class TestRunCpcv:
    def test_returns_cpcv_result(self, synthetic_trades: pd.DataFrame) -> None:
        result = run_cpcv(synthetic_trades, n_splits=6, n_test_splits=2)
        assert isinstance(result, CPCVResult)

    def test_has_paths(self, synthetic_trades: pd.DataFrame) -> None:
        result = run_cpcv(synthetic_trades, n_splits=6, n_test_splits=2)
        assert result.n_paths > 0
        assert len(result.paths) == result.n_paths

    def test_all_paths_have_valid_profit_factor(self, synthetic_trades: pd.DataFrame) -> None:
        result = run_cpcv(synthetic_trades, n_splits=6, n_test_splits=2)
        for p in result.paths:
            assert p.profit_factor >= 0.0 or p.profit_factor == float("inf")

    def test_total_trades_matches_input(self, synthetic_trades: pd.DataFrame) -> None:
        result = run_cpcv(synthetic_trades, n_splits=6, n_test_splits=2)
        assert result.total_trades == len(synthetic_trades)

    def test_path_results_have_date_ranges(self, synthetic_trades: pd.DataFrame) -> None:
        result = run_cpcv(synthetic_trades, n_splits=6, n_test_splits=2)
        for p in result.paths:
            assert ".." in p.date_range


# ── print_cpcv_summary ──────────────────────────────────────────────────────


class TestPrintCpcvSummary:
    def test_runs_without_error(
        self, sample_cpcv_result: CPCVResult, capsys: pytest.CaptureFixture[str]
    ) -> None:
        print_cpcv_summary(sample_cpcv_result)
        captured = capsys.readouterr()
        assert "Combinatorial Purged Cross-Validation" in captured.out
        assert "Verdict:" in captured.out

    def test_shows_failing_paths(
        self, sample_cpcv_result: CPCVResult, capsys: pytest.CaptureFixture[str]
    ) -> None:
        print_cpcv_summary(sample_cpcv_result)
        captured = capsys.readouterr()
        assert "WARNING" in captured.out
        assert "PF < 1.0" in captured.out


# ── save_cpcv_plot ───────────────────────────────────────────────────────────


class TestSaveCpcvPlot:
    def test_creates_png(self, sample_cpcv_result: CPCVResult, tmp_path: Path) -> None:
        out = tmp_path / "cpcv_chart.png"
        save_cpcv_plot(sample_cpcv_result, out)
        assert out.exists()
        assert out.stat().st_size > 0
