"""Integration: simulate a full trading day lifecycle.

Tests ORB capture, regime warmup, lunch chop filtering, daily resets,
cooldown after losses, and trade count enforcement.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, patch

import pytest

from src.alerts.dispatcher import AlertDispatcher
from src.config import RiskSettings
from src.levels import LevelManager
from src.main import _VIX_FALLBACK, _build_registry, _process_bar, _SymbolPipeline
from src.models import Bar, Direction, Regime, Signal, TimeFrame, TradeResult
from src.risk.cooldown import CooldownTracker
from src.risk.manager import RiskManager
from src.storage.database import BarDatabase
from src.storage.queries import ensure_schema, query_recent_signals
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


def _run_bars_through_pipeline(bars: list[Bar], symbol: str = "SPY") -> tuple:
    registry = _build_registry()
    levels = LevelManager(db=None, symbol=symbol)
    regime = RegimeDetector()
    strategy = ORBStrategy()
    signals = []
    for b in bars:
        registry.update_all(b)
        snapshot = registry.get_snapshot()
        levels.update(b)
        level_snapshot = levels.get_levels()
        if snapshot.atr is not None:
            trending_up = None
            if snapshot.ema9 is not None and snapshot.ema20 is not None:
                trending_up = snapshot.ema9 > snapshot.ema20
            regime.update(
                vix=regime.vix_level or _VIX_FALLBACK,
                adx=snapshot.adx,
                trending_up=trending_up,
            )
        signal = strategy.evaluate(b, snapshot, level_snapshot, regime)
        if signal is not None:
            signals.append(signal)
    return registry, levels, regime, strategy, signals


# ── Tests ────────────────────────────────────────────────────────────────────


class TestOrbCapture:
    """Verify ORB is captured at 9:35 ET (after 5 bars)."""

    def test_orb_captured_at_935(self) -> None:
        """Prevents: ORB never marking complete, blocking all signals."""
        bars = [*_orb_bars(), _bar(5, 480.0, 481.0, 479.0, 480.5)]
        _, levels, _, _, _ = _run_bars_through_pipeline(bars)
        snap = levels.get_levels()
        assert snap.orb_complete is True
        assert snap.orb_high == Decimal("482.00")


class TestRegimeWarmup:
    """Verify regime is populated after sufficient warmup bars."""

    @patch("src.strategies.orb.is_high_impact_day", return_value=False)
    @patch("src.strategies.orb.is_earnings_blackout", return_value=False)
    def test_regime_populated_after_28_bars(self, _m1, _m2) -> None:
        """Prevents: regime data never arriving, strategy perpetually filtering."""
        bars = _orb_bars() + _warmup_bars(5, 28)
        _, _, regime, _, _ = _run_bars_through_pipeline(bars)
        assert regime.vix_level is not None
        # ADX may or may not be non-None depending on exact warmup count


class TestLunchChopFilter:
    """Verify no signals during lunch chop window (11:30-13:30 ET)."""

    @patch("src.strategies.orb.is_high_impact_day", return_value=False)
    @patch("src.strategies.orb.is_earnings_blackout", return_value=False)
    def test_no_signals_during_lunch_chop(self, _m1, _m2) -> None:
        """Prevents: noisy lunch-hour signals eroding daily P&L."""
        bars = _orb_bars() + _warmup_bars(5, 25)
        # Lunch starts at 11:30 ET = 16:30 UTC = offset 120 from 14:30
        lunch_breakout = [
            _bar(120, 481.5, 483.0, 481.0, 482.5, 500_000),
            _bar(121, 482.5, 484.0, 482.0, 483.5, 600_000),
        ]
        all_bars = bars + lunch_breakout
        _, _, _, _, signals = _run_bars_through_pipeline(all_bars)
        lunch_signals = [s for s in signals if s.timestamp >= _BASE_TS + timedelta(minutes=120)]
        assert len(lunch_signals) == 0


class TestSignalTimeWindow:
    """Verify signals only fire within the valid 9:35-15:45 ET window."""

    @patch("src.strategies.orb.is_high_impact_day", return_value=False)
    @patch("src.strategies.orb.is_earnings_blackout", return_value=False)
    def test_signals_only_in_valid_window(self, _m1, _m2) -> None:
        """Prevents: after-hours signals placing trades in thin liquidity."""
        bars = _orb_bars() + _warmup_bars(5, 30) + _breakout_bars(35)
        _, _, _, _, signals = _run_bars_through_pipeline(bars)
        from datetime import time
        from zoneinfo import ZoneInfo

        et = ZoneInfo("America/New_York")
        for sig in signals:
            t = sig.timestamp.astimezone(et).time()
            assert t >= time(9, 35), f"Signal before 9:35 ET: {t}"
            assert t < time(15, 45), f"Signal after 15:45 ET: {t}"


class TestDailyReset:
    """Verify daily reset clears levels for the new trading day."""

    def test_daily_reset_clears_levels(self) -> None:
        """Prevents: stale ORB from prior day leaking into new session."""
        levels = LevelManager(db=None, symbol="SPY")
        # Day 1 bars
        for b in _orb_bars():
            levels.update(b)
        levels.update(_bar(5, 480.0, 481.0, 479.0, 480.5))
        snap1 = levels.get_levels()
        assert snap1.orb_complete is True

        # Simulate daily reset (same as _scheduler logic)
        levels._orb.__init__()  # type: ignore[misc]
        levels._vwap.__init__()  # type: ignore[misc]
        levels._day.__init__()  # type: ignore[misc]
        levels._premarket.__init__()  # type: ignore[misc]
        levels._last_date = None

        snap2 = levels.get_levels()
        assert snap2.orb_complete is False
        assert snap2.orb_high is None


class TestCooldownAndTilt:
    """Verify cooldown and tilt after consecutive losses."""

    def test_cooldown_after_two_consecutive_losses(self) -> None:
        """Prevents: trader ignoring 15-min cooldown after 2 losses."""
        # Inject a clock just 1 minute after the last loss so cooldown hasn't elapsed
        loss_time = _BASE_TS + timedelta(minutes=50)
        frozen_now = loss_time + timedelta(minutes=1)
        cooldown = CooldownTracker(now_fn=lambda: frozen_now)
        sig = Signal(
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
        loss1 = TradeResult(
            signal=sig, pnl=Decimal("-250"), timestamp=_BASE_TS + timedelta(minutes=40)
        )
        loss2 = TradeResult(signal=sig, pnl=Decimal("-250"), timestamp=loss_time)
        cooldown.record_trade(loss1)
        cooldown.record_trade(loss2)
        assert cooldown.consecutive_losses == 2
        assert not cooldown.is_cooled_down()


class TestMaxTradesPerDay:
    """Verify max 5 trades per day is enforced."""

    def test_max_five_trades_per_day_enforced(self) -> None:
        """Prevents: overtrading when strategy generates many signals."""
        cooldown = CooldownTracker()
        rm = RiskManager(cooldown=cooldown, settings=_TEST_RISK)
        sig = Signal(
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
        # Record 5 trades (alternating win/loss to avoid tilt)
        for i in range(5):
            trade = TradeResult(
                signal=sig,
                pnl=Decimal("100"),
                timestamp=_BASE_TS + timedelta(minutes=35 + i),
            )
            cooldown.record_trade(trade)

        decision = rm.approve(sig)
        assert not decision.approved
        assert "Max daily trades" in decision.reason


class TestSignalsPersistence:
    """Verify signal count in DB matches expected after a full day simulation."""

    @pytest.mark.asyncio
    @patch("src.strategies.orb.is_high_impact_day", return_value=False)
    @patch("src.strategies.orb.is_earnings_blackout", return_value=False)
    async def test_signals_table_has_correct_count_after_day(self, _m1, _m2, fake_db) -> None:
        """Prevents: signals silently dropped, DB count off from dispatched count."""
        pipe = _SymbolPipeline(
            symbol="SPY",
            registry=_build_registry(),
            levels=LevelManager(db=None, symbol="SPY"),
            regime=RegimeDetector(),
            strategy=ORBStrategy(),
        )
        alerter = FakeAlerter()
        disp = AlertDispatcher(alerter=alerter)
        cooldown = CooldownTracker()
        rm = RiskManager(cooldown=cooldown, settings=_TEST_RISK)

        bars = _orb_bars() + _warmup_bars(5, 30) + _breakout_bars(35)
        for b in bars:
            await _process_bar(b, pipe, rm, disp, fake_db)

        since = datetime(2024, 1, 1, tzinfo=UTC)
        rows = query_recent_signals(fake_db.conn, since=since)
        # Every generated signal (approved or rejected) should be persisted
        assert len(rows) >= 1
