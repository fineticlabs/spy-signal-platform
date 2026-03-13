"""Tests for risk management: RiskManager, CooldownTracker, position sizing."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo

from src.config import RiskSettings
from src.models import (
    Direction,
    Regime,
    Signal,
    TimeFrame,
    TradeResult,
)
from src.risk.cooldown import CooldownTracker
from src.risk.manager import RiskManager
from src.risk.position_sizing import calculate_position_size

_ET = ZoneInfo("America/New_York")


# ── shared helpers ────────────────────────────────────────────────────────────


def _et_ts(et_hour: int, et_minute: int, date_str: str = "2024-01-15") -> datetime:
    """Build a timezone-aware UTC datetime from an ET wall-clock time."""
    naive = datetime.strptime(f"{date_str} {et_hour:02d}:{et_minute:02d}", "%Y-%m-%d %H:%M")
    return naive.replace(tzinfo=_ET).astimezone(UTC)


def _make_settings(
    account_size: float = 50_000,
    risk_pct: float = 1.0,
    max_daily_loss_pct: float = 3.0,
    max_trades: int = 5,
) -> RiskSettings:
    return RiskSettings(
        account_size=Decimal(str(account_size)),
        risk_per_trade_pct=Decimal(str(risk_pct)),
        max_daily_loss_pct=Decimal(str(max_daily_loss_pct)),
        max_trades_per_day=max_trades,
    )


def _make_signal(
    direction: Direction = Direction.LONG,
    entry: float = 486.0,
    stop: float = 483.0,
    target: float = 492.0,
    rr: float = 2.0,
    confidence: int = 3,
    et_hour: int = 10,
    et_minute: int = 0,
) -> Signal:
    """Build a minimal valid Signal at the given ET time."""
    ts = _et_ts(et_hour, et_minute)
    return Signal(
        direction=direction,
        strategy_name="ORB-5min",
        entry_price=Decimal(str(entry)),
        stop_price=Decimal(str(stop)),
        target_price=Decimal(str(target)),
        risk_reward_ratio=Decimal(str(rr)),
        confidence_score=confidence,
        reason="test signal",
        timeframe=TimeFrame.ONE_MIN,
        regime=Regime.TRENDING_UP,
        vix=Decimal("18.0"),
        adx=Decimal("28.0"),
        timestamp=ts,
    )


def _make_loss(pnl: float = -300.0, et_hour: int = 10, et_minute: int = 0) -> TradeResult:
    return TradeResult(
        signal=_make_signal(),
        pnl=Decimal(str(pnl)),
        timestamp=_et_ts(et_hour, et_minute),
    )


def _make_win(pnl: float = 500.0) -> TradeResult:
    return TradeResult(
        signal=_make_signal(),
        pnl=Decimal(str(pnl)),
        timestamp=_et_ts(10, 30),
    )


# ── position sizing tests ─────────────────────────────────────────────────────


class TestCalculatePositionSize:
    def test_matches_manual_formula(self) -> None:
        """shares = floor(account * risk_pct/100 / risk_per_share)."""
        # account=$50k, risk=1%, dollar_risk=$500, risk_per_share=$3 → 166 shares
        size = calculate_position_size(
            account_size=Decimal("50000"),
            risk_pct=Decimal("1.0"),
            entry=Decimal("486.0"),
            stop=Decimal("483.0"),
        )
        assert size == 166

    def test_floors_to_whole_shares(self) -> None:
        """Partial shares are always floored, not rounded."""
        # dollar_risk=500, risk_per_share=3.01 → 166.11... → 166
        size = calculate_position_size(
            account_size=Decimal("50000"),
            risk_pct=Decimal("1.0"),
            entry=Decimal("486.01"),
            stop=Decimal("483.00"),
        )
        assert size == 166

    def test_returns_zero_when_stop_equals_entry(self) -> None:
        """Zero shares when stop distance is zero (cannot size position)."""
        size = calculate_position_size(
            account_size=Decimal("50000"),
            risk_pct=Decimal("1.0"),
            entry=Decimal("486.0"),
            stop=Decimal("486.0"),
        )
        assert size == 0

    def test_short_position_uses_abs_distance(self) -> None:
        """SHORT: stop > entry — distance is the same magnitude."""
        # entry=479, stop=482, risk_per_share=3 → same as LONG with same distance
        size = calculate_position_size(
            account_size=Decimal("50000"),
            risk_pct=Decimal("1.0"),
            entry=Decimal("479.0"),
            stop=Decimal("482.0"),
        )
        assert size == 166


# ── CooldownTracker tests ─────────────────────────────────────────────────────


class TestCooldownTracker:
    def test_initial_state(self) -> None:
        tracker = CooldownTracker()
        assert tracker.consecutive_losses == 0
        assert tracker.daily_pnl == Decimal("0")
        assert tracker.daily_trade_count == 0
        assert tracker.is_cooled_down()
        assert not tracker.is_tilted()

    def test_winning_trade_resets_consecutive_losses(self) -> None:
        tracker = CooldownTracker()
        tracker.record_trade(_make_loss())
        tracker.record_trade(_make_win())
        assert tracker.consecutive_losses == 0

    def test_two_losses_trigger_cooldown(self) -> None:
        loss_time = _et_ts(10, 0)
        five_min_later = loss_time + timedelta(minutes=5)
        tracker = CooldownTracker(now_fn=lambda: five_min_later)

        tracker.record_trade(_make_loss(et_hour=9, et_minute=45))
        tracker.record_trade(_make_loss(et_hour=10, et_minute=0))

        assert tracker.consecutive_losses == 2
        assert not tracker.is_cooled_down()  # only 5 min passed, need 15

    def test_cooldown_expires_after_15_minutes(self) -> None:
        loss_time = _et_ts(10, 0)
        sixteen_min_later = loss_time + timedelta(minutes=16)
        tracker = CooldownTracker(now_fn=lambda: sixteen_min_later)

        tracker.record_trade(_make_loss(et_hour=9, et_minute=45))
        tracker.record_trade(_make_loss(et_hour=10, et_minute=0))

        assert tracker.is_cooled_down()  # 16 min >= 15 min

    def test_three_losses_is_tilted(self) -> None:
        tracker = CooldownTracker()
        for i in range(3):
            tracker.record_trade(_make_loss(et_hour=10, et_minute=i))
        assert tracker.is_tilted()

    def test_daily_pnl_accumulates(self) -> None:
        tracker = CooldownTracker()
        tracker.record_trade(_make_loss(pnl=-300.0))
        tracker.record_trade(_make_loss(pnl=-250.0))
        tracker.record_trade(_make_win(pnl=400.0))
        assert tracker.daily_pnl == Decimal("-150")

    def test_reset_daily_clears_all_state(self) -> None:
        tracker = CooldownTracker()
        tracker.record_trade(_make_loss())
        tracker.record_trade(_make_loss())
        tracker.reset_daily()
        assert tracker.consecutive_losses == 0
        assert tracker.daily_pnl == Decimal("0")
        assert tracker.daily_trade_count == 0
        assert tracker.is_cooled_down()


# ── RiskManager tests ─────────────────────────────────────────────────────────


class TestRiskManager:
    def _manager(self, **kwargs: object) -> RiskManager:
        return RiskManager(
            cooldown=CooldownTracker(),
            settings=_make_settings(**kwargs),  # type: ignore[arg-type]
        )

    def test_approves_valid_signal(self) -> None:
        """Clean state + valid signal → approved with non-zero position size."""
        rm = self._manager()
        decision = rm.approve(_make_signal())
        assert decision.approved
        assert decision.position_size > 0
        assert "passed" in decision.reason

    def test_rejects_when_daily_loss_limit_hit(self) -> None:
        """Reject if cumulative daily loss >= max_daily_loss (3% of $50k = $1500)."""
        cooldown = CooldownTracker()
        # Record losses totalling $1500 (exactly at limit)
        cooldown.record_trade(
            TradeResult(
                signal=_make_signal(),
                pnl=Decimal("-1500"),
                timestamp=_et_ts(9, 45),
            )
        )
        rm = RiskManager(cooldown=cooldown, settings=_make_settings())
        decision = rm.approve(_make_signal())
        assert not decision.approved
        assert "loss limit" in decision.reason.lower() or "daily loss" in decision.reason.lower()

    def test_rejects_when_max_trades_reached(self) -> None:
        """Reject when daily trade count equals max_trades_per_day."""
        cooldown = CooldownTracker()
        for _ in range(5):
            cooldown.record_trade(_make_win())
        rm = RiskManager(cooldown=cooldown, settings=_make_settings(max_trades=5))
        decision = rm.approve(_make_signal())
        assert not decision.approved
        assert "trade" in decision.reason.lower()

    def test_rejects_when_tilted(self) -> None:
        """Reject when 3 consecutive losses (done for the day)."""
        cooldown = CooldownTracker()
        for i in range(3):
            cooldown.record_trade(_make_loss(et_hour=10, et_minute=i))
        rm = RiskManager(cooldown=cooldown, settings=_make_settings())
        decision = rm.approve(_make_signal())
        assert not decision.approved
        assert "tilt" in decision.reason.lower() or "consecutive" in decision.reason.lower()

    def test_rejects_during_cooldown(self) -> None:
        """Reject when 2 consecutive losses occurred < 15 min ago."""
        loss_time = _et_ts(10, 0)
        five_min_later = loss_time + timedelta(minutes=5)
        cooldown = CooldownTracker(now_fn=lambda: five_min_later)
        cooldown.record_trade(_make_loss(et_hour=9, et_minute=50))
        cooldown.record_trade(_make_loss(et_hour=10, et_minute=0))

        rm = RiskManager(cooldown=cooldown, settings=_make_settings())
        decision = rm.approve(_make_signal(et_hour=10, et_minute=5))
        assert not decision.approved
        assert "cooldown" in decision.reason.lower()

    def test_approves_after_cooldown_expires(self) -> None:
        """Approve again once 15 min have elapsed after 2 consecutive losses."""
        loss_time = _et_ts(10, 0)
        twenty_min_later = loss_time + timedelta(minutes=20)
        cooldown = CooldownTracker(now_fn=lambda: twenty_min_later)
        cooldown.record_trade(_make_loss(et_hour=9, et_minute=50))
        cooldown.record_trade(_make_loss(et_hour=10, et_minute=0))

        rm = RiskManager(cooldown=cooldown, settings=_make_settings())
        decision = rm.approve(_make_signal(et_hour=10, et_minute=20))
        assert decision.approved

    def test_rejects_poor_rr_ratio(self) -> None:
        """Reject when risk/reward < 1.5."""
        rm = self._manager()
        signal = _make_signal(entry=486.0, stop=483.0, target=487.5, rr=0.5)
        decision = rm.approve(signal)
        assert not decision.approved
        assert "r:r" in decision.reason.lower() or "risk" in decision.reason.lower()

    def test_rejects_outside_trading_window_before_935(self) -> None:
        """Reject signals timestamped before 9:35 ET."""
        rm = self._manager()
        decision = rm.approve(_make_signal(et_hour=9, et_minute=30))
        assert not decision.approved
        assert "window" in decision.reason.lower() or "time" in decision.reason.lower()

    def test_rejects_at_or_after_1545(self) -> None:
        """Reject signals at or after 15:45 ET."""
        rm = self._manager()
        decision = rm.approve(_make_signal(et_hour=15, et_minute=45))
        assert not decision.approved

    def test_rejects_during_lunch_low_confidence(self) -> None:
        """Reject lunch-hour signals with confidence < 4."""
        rm = self._manager()
        decision = rm.approve(_make_signal(et_hour=12, et_minute=0, confidence=3))
        assert not decision.approved

    def test_approves_during_lunch_high_confidence(self) -> None:
        """Allow lunch-hour signals with confidence >= 4."""
        rm = self._manager()
        decision = rm.approve(_make_signal(et_hour=12, et_minute=0, confidence=4))
        assert decision.approved

    def test_position_size_zero_when_stop_at_entry(self) -> None:
        """Reject when stop equals entry (position size would be 0)."""
        rm = self._manager()
        signal = _make_signal(entry=486.0, stop=486.0, rr=2.0)
        decision = rm.approve(signal)
        assert not decision.approved
        assert "position size" in decision.reason.lower()
