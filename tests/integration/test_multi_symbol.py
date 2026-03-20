"""Integration: multi-symbol isolation and shared-resource tests.

Verifies that per-symbol pipelines are isolated (indicators, levels, regime)
while shared resources (risk manager, cooldown) span all symbols.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock

import pytest

from src.config import RiskSettings
from src.levels import LevelManager
from src.main import _build_registry
from src.models import Bar, Direction, Regime, Signal, TimeFrame, TradeResult
from src.risk.cooldown import CooldownTracker
from src.risk.manager import RiskManager
from src.storage.database import BarDatabase
from src.storage.queries import ensure_schema
from src.strategies.regime import RegimeDetector

# ── Bar factory ──────────────────────────────────────────────────────────────

_BASE_TS = datetime(2024, 1, 16, 14, 30, tzinfo=UTC)  # Tuesday 9:30 ET


def _bar(
    minute_offset: int,
    open_: float,
    high: float,
    low: float,
    close: float,
    volume: int = 200_000,
    symbol: str = "SPY",
) -> Bar:
    return Bar(
        symbol=symbol,
        timeframe=TimeFrame.ONE_MIN,
        timestamp=_BASE_TS + timedelta(minutes=minute_offset),
        open=Decimal(str(open_)),
        high=Decimal(str(high)),
        low=Decimal(str(low)),
        close=Decimal(str(close)),
        volume=volume,
        vwap=Decimal(str((high + low + close) / 3)),
    )


def _orb_bars(symbol: str = "SPY") -> list[Bar]:
    return [
        _bar(0, 480.0, 481.5, 478.5, 480.0, symbol=symbol),
        _bar(1, 480.0, 482.0, 479.0, 481.0, symbol=symbol),
        _bar(2, 481.0, 481.8, 478.0, 479.5, symbol=symbol),
        _bar(3, 479.5, 481.5, 478.2, 480.5, symbol=symbol),
        _bar(4, 480.5, 481.7, 478.5, 481.0, symbol=symbol),
    ]


def _warmup_bars(start_offset: int, count: int, symbol: str = "SPY") -> list[Bar]:
    bars = []
    for i in range(count):
        c = 480.0 + i * 0.02
        bars.append(_bar(start_offset + i, c - 0.2, c + 0.3, c - 0.3, c, 250_000, symbol))
    return bars


def _breakout_bars(start_offset: int, symbol: str = "SPY") -> list[Bar]:
    return [
        _bar(start_offset, 481.5, 483.0, 481.0, 482.5, 500_000, symbol),
        _bar(start_offset + 1, 482.5, 484.0, 482.0, 483.5, 600_000, symbol),
    ]


_TEST_RISK = RiskSettings(
    account_size=Decimal("50000"),
    risk_per_trade_pct=Decimal("1.0"),
    max_daily_loss_pct=Decimal("3.0"),
    max_trades_per_day=5,
    max_concurrent_positions=3,
)


class FakeAlerter:
    def __init__(self) -> None:
        self.send_signal = AsyncMock(return_value=True)
        self.send_risk_warning = AsyncMock(return_value=True)
        self.send_daily_summary = AsyncMock(return_value=True)
        self.send_status = AsyncMock(return_value=True)


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def fake_db(tmp_path):
    db = BarDatabase(db_path=str(tmp_path / "test.db"))
    db.connect()
    ensure_schema(db.conn)
    return db


# ── Tests ────────────────────────────────────────────────────────────────────


class TestSymbolIsolation:
    """Verify per-symbol pipelines are fully isolated."""

    def test_spy_and_qqq_get_separate_orb_levels(self) -> None:
        """Prevents: ORB levels leaking between symbols."""
        spy_levels = LevelManager(db=None, symbol="SPY")
        qqq_levels = LevelManager(db=None, symbol="QQQ")

        # Feed SPY ORB bars (high=482, low=478)
        for b in _orb_bars("SPY"):
            spy_levels.update(b)
        spy_levels.update(_bar(5, 480.0, 481.0, 479.0, 480.5, symbol="SPY"))

        # Feed QQQ different bars (high=411, low=408)
        qqq_orb = [
            _bar(0, 410.0, 411.0, 408.5, 409.5, symbol="QQQ"),
            _bar(1, 409.5, 410.5, 408.0, 410.0, symbol="QQQ"),
            _bar(2, 410.0, 410.8, 408.2, 409.0, symbol="QQQ"),
            _bar(3, 409.0, 410.2, 408.5, 409.8, symbol="QQQ"),
            _bar(4, 409.8, 410.5, 408.3, 410.2, symbol="QQQ"),
        ]
        for b in qqq_orb:
            qqq_levels.update(b)
        qqq_levels.update(_bar(5, 410.0, 410.5, 409.0, 410.0, symbol="QQQ"))

        spy_snap = spy_levels.get_levels()
        qqq_snap = qqq_levels.get_levels()

        assert spy_snap.orb_high == Decimal("482.00")
        assert qqq_snap.orb_high == Decimal("411.00")
        assert spy_snap.orb_high != qqq_snap.orb_high

    def test_spy_signal_doesnt_affect_qqq_indicators(self) -> None:
        """Prevents: cross-contamination between symbol registries."""
        spy_reg = _build_registry()
        qqq_reg = _build_registry()

        spy_bar = _bar(0, 480.0, 481.0, 479.0, 480.0, symbol="SPY")
        spy_reg.update_all(spy_bar)

        # QQQ registry untouched — snapshot should be all None
        qqq_snap = qqq_reg.get_snapshot()
        assert qqq_snap.ema9 is None
        assert qqq_snap.atr is None

    def test_each_symbol_has_own_regime_detector(self) -> None:
        """Prevents: shared regime state causing wrong regime for a symbol."""
        spy_regime = RegimeDetector()
        qqq_regime = RegimeDetector()

        spy_regime.update(vix=Decimal("20"), adx=Decimal("30"), trending_up=True)
        qqq_regime.update(vix=Decimal("20"), adx=Decimal("10"), trending_up=False)

        assert spy_regime.current_regime == Regime.TRENDING_UP
        assert qqq_regime.current_regime == Regime.CHOPPY


class TestSharedRiskManager:
    """Verify risk manager is shared across all symbols."""

    def test_shared_risk_manager_across_symbols(self) -> None:
        """Prevents: per-symbol risk managers allowing 5 trades each (15 total)."""
        cooldown = CooldownTracker()
        rm = RiskManager(cooldown=cooldown, settings=_TEST_RISK)

        sig_spy = Signal(
            symbol="SPY",
            direction=Direction.LONG,
            strategy_name="ORB-5min",
            entry_price=Decimal("482.50"),
            stop_price=Decimal("480.00"),
            target_price=Decimal("487.50"),
            risk_reward_ratio=Decimal("2.0"),
            confidence_score=3,
            reason="test",
            timeframe=TimeFrame.ONE_MIN,
            regime=Regime.RANGING,
            timestamp=_BASE_TS + timedelta(minutes=35),
        )
        sig_qqq = Signal(
            symbol="QQQ",
            direction=Direction.LONG,
            strategy_name="ORB-5min",
            entry_price=Decimal("410.50"),
            stop_price=Decimal("408.00"),
            target_price=Decimal("415.50"),
            risk_reward_ratio=Decimal("2.0"),
            confidence_score=3,
            reason="test",
            timeframe=TimeFrame.ONE_MIN,
            regime=Regime.RANGING,
            timestamp=_BASE_TS + timedelta(minutes=36),
        )

        # Record 5 trades across both symbols
        for i in range(3):
            trade = TradeResult(
                signal=sig_spy,
                pnl=Decimal("100"),
                timestamp=_BASE_TS + timedelta(minutes=35 + i),
            )
            cooldown.record_trade(trade)
        for i in range(2):
            trade = TradeResult(
                signal=sig_qqq,
                pnl=Decimal("100"),
                timestamp=_BASE_TS + timedelta(minutes=40 + i),
            )
            cooldown.record_trade(trade)

        assert cooldown.daily_trade_count == 5
        decision = rm.approve(sig_spy)
        assert not decision.approved

    def test_daily_trade_count_shared_across_symbols(self) -> None:
        """Prevents: trade count tracked per-symbol instead of globally."""
        cooldown = CooldownTracker()
        sig = Signal(
            symbol="SPY",
            direction=Direction.LONG,
            strategy_name="ORB-5min",
            entry_price=Decimal("482.50"),
            stop_price=Decimal("480.00"),
            target_price=Decimal("487.50"),
            risk_reward_ratio=Decimal("2.0"),
            confidence_score=3,
            reason="test",
            timeframe=TimeFrame.ONE_MIN,
            regime=Regime.RANGING,
            timestamp=_BASE_TS + timedelta(minutes=35),
        )
        sig_qqq = Signal(
            symbol="QQQ",
            direction=Direction.LONG,
            strategy_name="ORB-5min",
            entry_price=Decimal("410.50"),
            stop_price=Decimal("408.00"),
            target_price=Decimal("415.50"),
            risk_reward_ratio=Decimal("2.0"),
            confidence_score=3,
            reason="test",
            timeframe=TimeFrame.ONE_MIN,
            regime=Regime.RANGING,
            timestamp=_BASE_TS + timedelta(minutes=36),
        )

        cooldown.record_trade(
            TradeResult(signal=sig, pnl=Decimal("50"), timestamp=_BASE_TS + timedelta(minutes=35))
        )
        cooldown.record_trade(
            TradeResult(
                signal=sig_qqq, pnl=Decimal("50"), timestamp=_BASE_TS + timedelta(minutes=36)
            )
        )
        assert cooldown.daily_trade_count == 2

    def test_cooldown_shared_across_symbols(self) -> None:
        """Prevents: tilt on SPY not blocking QQQ signals."""
        cooldown = CooldownTracker()
        rm = RiskManager(cooldown=cooldown, settings=_TEST_RISK)

        sig_spy = Signal(
            symbol="SPY",
            direction=Direction.LONG,
            strategy_name="ORB-5min",
            entry_price=Decimal("482.50"),
            stop_price=Decimal("480.00"),
            target_price=Decimal("487.50"),
            risk_reward_ratio=Decimal("2.0"),
            confidence_score=3,
            reason="test",
            timeframe=TimeFrame.ONE_MIN,
            regime=Regime.RANGING,
            timestamp=_BASE_TS + timedelta(minutes=35),
        )
        sig_qqq = Signal(
            symbol="QQQ",
            direction=Direction.LONG,
            strategy_name="ORB-5min",
            entry_price=Decimal("410.50"),
            stop_price=Decimal("408.00"),
            target_price=Decimal("415.50"),
            risk_reward_ratio=Decimal("2.0"),
            confidence_score=3,
            reason="test",
            timeframe=TimeFrame.ONE_MIN,
            regime=Regime.RANGING,
            timestamp=_BASE_TS + timedelta(minutes=36),
        )

        # 3 consecutive losses on SPY → tilt
        for i in range(3):
            cooldown.record_trade(
                TradeResult(
                    signal=sig_spy,
                    pnl=Decimal("-200"),
                    timestamp=_BASE_TS + timedelta(minutes=35 + i),
                )
            )

        assert cooldown.is_tilted()
        # QQQ signal should also be rejected due to shared tilt
        decision = rm.approve(sig_qqq)
        assert not decision.approved
        assert "Tilted" in decision.reason
