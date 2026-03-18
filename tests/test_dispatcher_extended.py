"""Extended tests for src.alerts.dispatcher — covering risk_warning and daily_summary result paths."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import AsyncMock

import pytest

from src.alerts.dispatcher import AlertDispatcher
from src.models import (
    Direction,
    Regime,
    RiskDecision,
    Signal,
    TimeFrame,
    TradeResult,
)


def _make_signal() -> Signal:
    return Signal(
        symbol="SPY",
        direction=Direction.LONG,
        strategy_name="ORB",
        entry_price=Decimal("480.00"),
        stop_price=Decimal("478.00"),
        target_price=Decimal("484.00"),
        risk_reward_ratio=Decimal("2.0"),
        confidence_score=3,
        reason="Test signal",
        timeframe=TimeFrame.ONE_MIN,
        regime=Regime.TRENDING_UP,
        timestamp=datetime(2024, 1, 15, 14, 36, tzinfo=UTC),
    )


def _make_risk_decision() -> RiskDecision:
    return RiskDecision(approved=True, reason="OK", position_size=50)


def _make_alerter(
    *,
    signal_ok: bool = True,
    risk_ok: bool = True,
    summary_ok: bool = True,
) -> AsyncMock:
    alerter = AsyncMock()
    alerter.send_signal = AsyncMock(return_value=signal_ok)
    alerter.send_risk_warning = AsyncMock(return_value=risk_ok)
    alerter.send_daily_summary = AsyncMock(return_value=summary_ok)
    return alerter


class TestDispatchRiskWarning:
    """Cover dispatch_risk_warning success and failure paths (lines 100, 113)."""

    @pytest.mark.asyncio
    async def test_returns_true_on_success(self) -> None:
        alerter = _make_alerter(risk_ok=True)
        dispatcher = AlertDispatcher(alerter)
        result = await dispatcher.dispatch_risk_warning("Max loss approaching")
        assert result is True
        alerter.send_risk_warning.assert_awaited_once_with("Max loss approaching")

    @pytest.mark.asyncio
    async def test_returns_false_on_failure(self) -> None:
        alerter = _make_alerter(risk_ok=False)
        dispatcher = AlertDispatcher(alerter)
        result = await dispatcher.dispatch_risk_warning("Max loss approaching")
        assert result is False


class TestDispatchDailySummary:
    """Cover dispatch_daily_summary success and failure paths (lines 113, 125)."""

    @pytest.mark.asyncio
    async def test_returns_true_on_success(self) -> None:
        alerter = _make_alerter(summary_ok=True)
        dispatcher = AlertDispatcher(alerter)
        result = await dispatcher.dispatch_daily_summary([])
        assert result is True
        alerter.send_daily_summary.assert_awaited_once_with([])

    @pytest.mark.asyncio
    async def test_returns_false_on_failure(self) -> None:
        alerter = _make_alerter(summary_ok=False)
        dispatcher = AlertDispatcher(alerter)
        result = await dispatcher.dispatch_daily_summary([])
        assert result is False

    @pytest.mark.asyncio
    async def test_passes_trades_through(self) -> None:
        sig = _make_signal()
        trade = TradeResult(
            signal=sig,
            pnl=Decimal("150.00"),
            timestamp=datetime(2024, 1, 15, 20, 0, tzinfo=UTC),
        )
        alerter = _make_alerter(summary_ok=True)
        dispatcher = AlertDispatcher(alerter)
        result = await dispatcher.dispatch_daily_summary([trade])
        assert result is True
        alerter.send_daily_summary.assert_awaited_once_with([trade])


class TestDispatchSignalFailure:
    """Cover dispatch_signal returning False when channel fails."""

    @pytest.mark.asyncio
    async def test_returns_false_on_channel_failure(self) -> None:
        alerter = _make_alerter(signal_ok=False)
        dispatcher = AlertDispatcher(alerter)
        result = await dispatcher.dispatch_signal(_make_signal(), _make_risk_decision())
        assert result is False
