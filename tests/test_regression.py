"""Regression tests for the spy-signal-platform.

Ensures deterministic behaviour, metric stability on known inputs, and
immutability of critical model constants and config defaults.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.models import (
    Direction,
    IndicatorSnapshot,
    LevelSnapshot,
    Regime,
    Signal,
    TimeFrame,
)

if TYPE_CHECKING:
    from src.models import Bar

from .conftest import make_bar, make_utc_ts

# ── helpers ──────────────────────────────────────────────────────────────────


def _make_bars_sequence(n: int) -> list[Bar]:
    """Generate a deterministic sequence of *n* bars at valid market hours."""
    bars: list[Bar] = []
    base_ts = datetime(2024, 1, 15, 14, 35, tzinfo=UTC)
    rng = np.random.default_rng(99)
    price = 480.0
    for i in range(n):
        noise = rng.normal(0, 0.3)
        price += noise
        o = Decimal(str(round(price, 2)))
        h = Decimal(str(round(price + abs(rng.normal(0, 0.5)), 2)))
        l_ = Decimal(str(round(price - abs(rng.normal(0, 0.5)), 2)))
        c = Decimal(str(round(price + rng.normal(0, 0.1), 2)))
        h = max(o, h, c)
        l_ = min(o, l_, c)
        bars.append(
            make_bar(
                timestamp=base_ts + timedelta(minutes=i),
                open_=o,
                high=h,
                low=l_,
                close=c,
                volume=int(rng.integers(50_000, 500_000)),
                vwap=Decimal(str(round(float(o + h + l_) / 3, 2))),
            )
        )
    return bars


# ── Signal Reproducibility ───────────────────────────────────────────────────


class TestSignalReproducibility:
    """Verify that strategies produce identical outputs on identical inputs."""

    @patch("src.strategies.orb.is_high_impact_day", return_value=False)
    @patch("src.strategies.orb.is_earnings_blackout", return_value=False)
    def test_same_50_bars_same_signal_three_times(
        self, _mock_earnings: MagicMock, _mock_econ: MagicMock
    ) -> None:
        from src.strategies.orb import ORBStrategy
        from src.strategies.regime import RegimeDetector

        bars = _make_bars_sequence(50)
        indicators = IndicatorSnapshot(atr=Decimal("1.50"), rsi=Decimal("55"))
        levels = LevelSnapshot(
            orb_high=Decimal("481.50"),
            orb_low=Decimal("479.00"),
            orb_complete=True,
        )

        results: list[list[Signal | None]] = []
        for _ in range(3):
            strategy = ORBStrategy(excluded_days=[])
            regime = RegimeDetector()
            regime.update(vix=Decimal("18"), adx=Decimal("28"), trending_up=True)
            run_signals = []
            for bar in bars:
                sig = strategy.evaluate(bar, indicators, levels, regime)
                run_signals.append(sig)
            results.append(run_signals)

        # All three runs must produce identical signal sequences
        for i in range(50):
            s0 = results[0][i]
            s1 = results[1][i]
            s2 = results[2][i]
            if s0 is None:
                assert s1 is None and s2 is None, f"Bar {i}: divergent None"
            else:
                assert s1 is not None and s2 is not None, f"Bar {i}: divergent None"
                assert s0.direction == s1.direction == s2.direction
                assert s0.entry_price == s1.entry_price == s2.entry_price
                assert s0.stop_price == s1.stop_price == s2.stop_price
                assert s0.target_price == s1.target_price == s2.target_price

    @patch("src.strategies.orb.is_high_impact_day", return_value=False)
    @patch("src.strategies.orb.is_earnings_blackout", return_value=False)
    def test_orb_strategy_is_deterministic(
        self, _mock_earnings: MagicMock, _mock_econ: MagicMock
    ) -> None:
        """ORBStrategy must not incorporate any randomness."""
        from src.strategies.orb import ORBStrategy
        from src.strategies.regime import RegimeDetector

        bars = _make_bars_sequence(20)
        indicators = IndicatorSnapshot(atr=Decimal("1.50"))
        levels = LevelSnapshot(
            orb_high=Decimal("481.50"),
            orb_low=Decimal("479.00"),
            orb_complete=True,
        )

        signals_a: list[Signal | None] = []
        signals_b: list[Signal | None] = []

        for run_signals in (signals_a, signals_b):
            strat = ORBStrategy(excluded_days=[])
            regime = RegimeDetector()
            regime.update(vix=Decimal("18"), adx=Decimal("28"), trending_up=True)
            for bar in bars:
                run_signals.append(strat.evaluate(bar, indicators, levels, regime))

        for i, (a, b) in enumerate(zip(signals_a, signals_b, strict=True)):
            if a is None:
                assert b is None, f"Bar {i}: non-deterministic"
            else:
                assert b is not None
                assert a.model_dump() == b.model_dump(), f"Bar {i}: non-deterministic"

    def test_risk_manager_is_deterministic(self) -> None:
        from src.config import RiskSettings
        from src.risk.cooldown import CooldownTracker
        from src.risk.manager import RiskManager

        settings = RiskSettings()
        signal = Signal(
            symbol="SPY",
            direction=Direction.LONG,
            strategy_name="ORB-5min",
            entry_price=Decimal("481.50"),
            stop_price=Decimal("479.00"),
            target_price=Decimal("486.50"),
            risk_reward_ratio=Decimal("2.0"),
            confidence_score=3,
            reason="test",
            timeframe=TimeFrame.ONE_MIN,
            regime=Regime.TRENDING_UP,
            timestamp=make_utc_ts(2024, 1, 15, 14, 36),
        )

        decisions = []
        for _ in range(3):
            cooldown = CooldownTracker()
            mgr = RiskManager(cooldown=cooldown, settings=settings)
            decisions.append(mgr.approve(signal))

        assert all(d.approved == decisions[0].approved for d in decisions)
        assert all(d.position_size == decisions[0].position_size for d in decisions)


# ── Metric Stability ─────────────────────────────────────────────────────────


class TestMetricStability:
    """Verify that metric functions produce expected values on known inputs."""

    def test_profit_factor_known_sequence(self) -> None:
        from src.backtest.monte_carlo import _profit_factor

        # wins: 100 + 200 = 300 gross profit
        # losses: -50 + -100 = 150 gross loss
        # PF = 300 / 150 = 2.0
        pnl = np.array([100.0, -50.0, 200.0, -100.0])
        assert _profit_factor(pnl) == pytest.approx(2.0)

    def test_win_rate_known_sequence(self) -> None:
        from src.backtest.metrics import compute_metrics

        trades = pd.DataFrame(
            {
                "PnL": [100.0, -50.0, 200.0, -100.0, 150.0],
                "ReturnPct": [0.02, -0.01, 0.04, -0.02, 0.03],
            }
        )
        metrics = compute_metrics(trades)
        # 3 wins / 5 total = 0.60
        assert metrics["win_rate"] == pytest.approx(0.6, abs=0.01)

    def test_max_drawdown_known_sequence(self) -> None:
        from src.backtest.monte_carlo import _max_drawdown_pct

        # Equity: 50000 + cumsum([100, -300, 50]) = [50100, 49800, 49850]
        # Peak at 50100, trough at 49800 → DD = (49800-50100)/50100
        pnl = np.array([100.0, -300.0, 50.0])
        dd = _max_drawdown_pct(pnl, initial_capital=50_000.0)
        expected = (49800 - 50100) / 50100
        assert dd == pytest.approx(expected, abs=1e-6)

    def test_sharpe_known_sequence(self) -> None:
        from src.backtest.monte_carlo import _sharpe

        pnl = np.array([10.0, 20.0, -5.0, 15.0, 10.0])
        result = _sharpe(pnl, periods_per_year=252)
        # Manual: mean=10.0, std=np.std(pnl, ddof=1)
        expected_mean = float(pnl.mean())
        expected_std = float(pnl.std())
        expected = expected_mean / expected_std * np.sqrt(252)
        assert result == pytest.approx(expected, rel=1e-4)


# ── Model Constants ──────────────────────────────────────────────────────────


class TestModelConstants:
    """Guard against accidental changes to critical constants."""

    def test_ticker_list_has_7_entries(self) -> None:
        from src.backtest.ml_features import TICKER_LIST

        assert len(TICKER_LIST) == 7

    def test_feature_cols_has_13_entries(self) -> None:
        from src.backtest.ml_features import FEATURE_COLS

        assert len(FEATURE_COLS) == 13

    def test_default_n_permutations_is_10000(self) -> None:
        from src.backtest.monte_carlo import _DEFAULT_N_PERMUTATIONS

        assert _DEFAULT_N_PERMUTATIONS == 10_000

    def test_seed_is_42(self) -> None:
        from src.backtest.monte_carlo import _SEED

        assert _SEED == 42


# ── Config Defaults ──────────────────────────────────────────────────────────


class TestConfigDefaults:
    """Verify that configuration defaults match documented values."""

    def test_risk_settings_defaults(self) -> None:
        from src.config import RiskSettings

        settings = RiskSettings()
        assert settings.account_size == Decimal("50000")
        assert settings.risk_per_trade_pct == Decimal("1.0")
        assert settings.max_daily_loss_pct == Decimal("3.0")
        assert settings.max_trades_per_day == 5

    def test_app_settings_default_symbols_has_26_tickers(self) -> None:
        from src.config import AppSettings

        settings = AppSettings()
        assert len(settings.symbols) == 26


# ── Backtest Helpers ─────────────────────────────────────────────────────────


class TestBacktestHelpers:
    """Verify that backtest helper functions match manual calculations."""

    def test_profit_factor_manual_calculation(self) -> None:
        from src.backtest.monte_carlo import _profit_factor

        # 3 wins: 50 + 120 + 80 = 250
        # 2 losses: -60 + -40 = 100
        # PF = 250 / 100 = 2.5
        pnl = np.array([50.0, -60.0, 120.0, -40.0, 80.0])
        assert _profit_factor(pnl) == pytest.approx(2.5)

    def test_profit_factor_no_losses_returns_inf(self) -> None:
        from src.backtest.monte_carlo import _profit_factor

        pnl = np.array([100.0, 200.0, 50.0])
        assert _profit_factor(pnl) == float("inf")

    def test_max_drawdown_pct_manual_calculation(self) -> None:
        from src.backtest.monte_carlo import _max_drawdown_pct

        # Capital = 50000
        # Equity: [50200, 50000, 49700, 49900]
        # Peak at 50200, trough at 49700 → DD = (49700-50200)/50200
        pnl = np.array([200.0, -200.0, -300.0, 200.0])
        dd = _max_drawdown_pct(pnl, initial_capital=50_000.0)
        expected = (49700 - 50200) / 50200
        assert dd == pytest.approx(expected, abs=1e-6)

    def test_max_drawdown_pct_monotonic_gains_is_zero(self) -> None:
        from src.backtest.monte_carlo import _max_drawdown_pct

        pnl = np.array([100.0, 100.0, 100.0])
        dd = _max_drawdown_pct(pnl, initial_capital=50_000.0)
        assert dd == pytest.approx(0.0)
