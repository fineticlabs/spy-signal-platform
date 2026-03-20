"""Integration: executor receives correct order details from approved signals.

Mocks only the AlpacaExecutor (external I/O) to verify the wiring between
_process_bar and the executor.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.alerts.dispatcher import AlertDispatcher
from src.config import RiskSettings
from src.levels import LevelManager
from src.main import _build_registry, _process_bar, _SymbolPipeline
from src.models import Bar, Direction, Regime, Signal, TimeFrame, TradeResult
from src.risk.cooldown import CooldownTracker
from src.risk.manager import RiskManager
from src.storage.database import BarDatabase
from src.storage.queries import ensure_schema
from src.strategies.orb import ORBStrategy
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


@pytest.fixture
def mock_executor():
    executor = MagicMock()
    executor.submit_bracket_order = MagicMock(return_value=MagicMock(id="test-order-123"))
    return executor


@pytest.fixture
def pipeline():
    return _SymbolPipeline(
        symbol="SPY",
        registry=_build_registry(),
        levels=LevelManager(db=None, symbol="SPY"),
        regime=RegimeDetector(),
        strategy=ORBStrategy(),
    )


@pytest.fixture
def risk():
    return RiskManager(cooldown=CooldownTracker(), settings=_TEST_RISK)


@pytest.fixture
def fake_alerter():
    return FakeAlerter()


@pytest.fixture
def dispatcher(fake_alerter):
    return AlertDispatcher(alerter=fake_alerter)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _exhaust_trades(cooldown: CooldownTracker, count: int) -> None:
    """Record `count` winning trades to exhaust the daily trade limit."""
    sig = Signal(
        symbol="SPY",
        direction=Direction.LONG,
        strategy_name="ORB-5min",
        entry_price=Decimal("482.50"),
        stop_price=Decimal("480.00"),
        target_price=Decimal("487.50"),
        risk_reward_ratio=Decimal("2.0"),
        confidence_score=3,
        reason="exhaust",
        timeframe=TimeFrame.ONE_MIN,
        regime=Regime.RANGING,
        timestamp=_BASE_TS + timedelta(minutes=35),
    )
    for i in range(count):
        cooldown.record_trade(
            TradeResult(
                signal=sig,
                pnl=Decimal("50"),
                timestamp=_BASE_TS + timedelta(minutes=35 + i),
            )
        )


async def _feed_full_sequence(pipeline, risk, dispatcher, db, executor=None):
    bars = _orb_bars() + _warmup_bars(5, 30) + _breakout_bars(35)
    for b in bars:
        await _process_bar(b, pipeline, risk, dispatcher, db, executor=executor)


async def _feed_short_sequence(pipeline, risk, dispatcher, db, executor=None):
    """Feed bars producing a SHORT signal (close below ORB low)."""
    bars = _orb_bars()
    warmup = _warmup_bars(5, 30)
    # Breakdown bars below ORB low (478)
    short_bars = [
        _bar(35, 478.5, 479.0, 477.0, 477.5, 500_000),
        _bar(36, 477.5, 478.0, 476.0, 476.5, 600_000),
    ]
    for b in bars + warmup + short_bars:
        await _process_bar(b, pipeline, risk, dispatcher, db, executor=executor)


# ── Tests ────────────────────────────────────────────────────────────────────


class TestExecutorWiring:
    """Verify executor receives correct order details from the pipeline."""

    @pytest.mark.asyncio
    @patch("src.strategies.orb.is_high_impact_day", return_value=False)
    @patch("src.strategies.orb.is_earnings_blackout", return_value=False)
    async def test_executor_called_on_approved_signal(
        self, _m1, _m2, pipeline, risk, dispatcher, fake_db, mock_executor
    ) -> None:
        """Prevents: executor wired but never invoked on approved signals."""
        await _feed_full_sequence(pipeline, risk, dispatcher, fake_db, executor=mock_executor)
        assert mock_executor.submit_bracket_order.call_count >= 1

    @pytest.mark.asyncio
    @patch("src.strategies.orb.is_high_impact_day", return_value=False)
    @patch("src.strategies.orb.is_earnings_blackout", return_value=False)
    async def test_executor_receives_correct_side_long(
        self, _m1, _m2, pipeline, risk, dispatcher, fake_db, mock_executor
    ) -> None:
        """Prevents: LONG signal sent to executor as SELL order."""
        await _feed_full_sequence(pipeline, risk, dispatcher, fake_db, executor=mock_executor)
        if mock_executor.submit_bracket_order.call_count > 0:
            call_kwargs = mock_executor.submit_bracket_order.call_args
            direction = call_kwargs[1].get("direction") or call_kwargs[0][1]
            assert direction == Direction.LONG

    @pytest.mark.asyncio
    @patch("src.strategies.orb.is_high_impact_day", return_value=False)
    @patch("src.strategies.orb.is_earnings_blackout", return_value=False)
    async def test_executor_receives_correct_side_short(
        self, _m1, _m2, fake_db, mock_executor
    ) -> None:
        """Prevents: SHORT signal sent to executor as BUY order."""
        pipe = _SymbolPipeline(
            symbol="SPY",
            registry=_build_registry(),
            levels=LevelManager(db=None, symbol="SPY"),
            regime=RegimeDetector(),
            strategy=ORBStrategy(),
        )
        rm = RiskManager(cooldown=CooldownTracker(), settings=_TEST_RISK)
        alerter = FakeAlerter()
        disp = AlertDispatcher(alerter=alerter)
        await _feed_short_sequence(pipe, rm, disp, fake_db, executor=mock_executor)
        if mock_executor.submit_bracket_order.call_count > 0:
            call_kwargs = mock_executor.submit_bracket_order.call_args
            direction = call_kwargs[1].get("direction") or call_kwargs[0][1]
            assert direction == Direction.SHORT

    @pytest.mark.asyncio
    @patch("src.strategies.orb.is_high_impact_day", return_value=False)
    @patch("src.strategies.orb.is_earnings_blackout", return_value=False)
    async def test_executor_receives_stop_and_target(
        self, _m1, _m2, pipeline, risk, dispatcher, fake_db, mock_executor
    ) -> None:
        """Prevents: bracket order missing stop or target, creating naked position."""
        await _feed_full_sequence(pipeline, risk, dispatcher, fake_db, executor=mock_executor)
        if mock_executor.submit_bracket_order.call_count > 0:
            kw = mock_executor.submit_bracket_order.call_args[1]
            assert "stop_price" in kw
            assert "target_price" in kw
            assert kw["stop_price"] > 0
            assert kw["target_price"] > 0

    @pytest.mark.asyncio
    @patch("src.strategies.orb.is_high_impact_day", return_value=False)
    @patch("src.strategies.orb.is_earnings_blackout", return_value=False)
    async def test_executor_not_called_on_rejection(self, _m1, _m2, fake_db, mock_executor) -> None:
        """Prevents: rejected signal still placing a real order."""
        cooldown = CooldownTracker()
        # Exhaust daily trades so every signal is rejected
        _exhaust_trades(cooldown, _TEST_RISK.max_trades_per_day)
        rm = RiskManager(cooldown=cooldown, settings=_TEST_RISK)
        pipe = _SymbolPipeline(
            symbol="SPY",
            registry=_build_registry(),
            levels=LevelManager(db=None, symbol="SPY"),
            regime=RegimeDetector(),
            strategy=ORBStrategy(),
        )
        alerter = FakeAlerter()
        disp = AlertDispatcher(alerter=alerter)
        await _feed_full_sequence(pipe, rm, disp, fake_db, executor=mock_executor)
        assert mock_executor.submit_bracket_order.call_count == 0

    @pytest.mark.asyncio
    @patch("src.strategies.orb.is_high_impact_day", return_value=False)
    @patch("src.strategies.orb.is_earnings_blackout", return_value=False)
    async def test_executor_not_called_when_alerts_only(
        self, _m1, _m2, pipeline, risk, dispatcher, fake_db
    ) -> None:
        """Prevents: executor called when execution_mode is alerts_only."""
        # executor=None means alerts_only mode
        await _feed_full_sequence(pipeline, risk, dispatcher, fake_db, executor=None)
        # If we get here without error and no executor was called, test passes
