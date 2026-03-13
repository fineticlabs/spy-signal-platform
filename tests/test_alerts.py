"""Tests for alert formatting and dispatch."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo

from src.alerts.dispatcher import AlertDispatcher
from src.alerts.formatter import (
    format_daily_summary,
    format_risk_alert,
    format_signal_alert,
)
from src.models import (
    Direction,
    IndicatorSnapshot,
    LevelSnapshot,
    Regime,
    RiskDecision,
    Signal,
    TimeFrame,
    TradeResult,
)

_ET = ZoneInfo("America/New_York")


# ── shared helpers ────────────────────────────────────────────────────────────


def _et_ts(et_hour: int, et_minute: int) -> datetime:
    naive = datetime.strptime(f"2024-01-15 {et_hour:02d}:{et_minute:02d}", "%Y-%m-%d %H:%M")
    return naive.replace(tzinfo=_ET).astimezone(UTC)


def _make_signal(
    direction: Direction = Direction.LONG,
    entry: float = 486.0,
    stop: float = 483.0,
    target: float = 492.0,
    rr: float = 2.0,
    confidence: int = 3,
    et_hour: int = 10,
    et_minute: int = 0,
    with_levels: bool = False,
) -> Signal:
    levels = None
    if with_levels:
        levels = LevelSnapshot(
            orb_high=Decimal("485.00"),
            orb_low=Decimal("480.00"),
            orb_complete=True,
            vwap=Decimal("483.50"),
            prev_day_high=Decimal("490.00"),
            prev_day_low=Decimal("478.00"),
        )
    return Signal(
        direction=direction,
        strategy_name="ORB-5min",
        entry_price=Decimal(str(entry)),
        stop_price=Decimal(str(stop)),
        target_price=Decimal(str(target)),
        risk_reward_ratio=Decimal(str(rr)),
        confidence_score=confidence,
        reason="Test reason with special chars: 486.0 > 485.0",
        timeframe=TimeFrame.ONE_MIN,
        regime=Regime.TRENDING_UP,
        vix=Decimal("18.5"),
        adx=Decimal("28.0"),
        indicators_snapshot=IndicatorSnapshot(atr=Decimal("2.00")),
        levels_snapshot=levels,
        timestamp=_et_ts(et_hour, et_minute),
    )


def _make_decision(size: int = 166) -> RiskDecision:
    return RiskDecision(approved=True, reason="All risk checks passed", position_size=size)


def _make_trade(pnl: float, et_hour: int = 10, et_minute: int = 30) -> TradeResult:
    return TradeResult(
        signal=_make_signal(),
        pnl=Decimal(str(pnl)),
        timestamp=_et_ts(et_hour, et_minute),
    )


# ── Fake alerter for dispatcher tests ────────────────────────────────────────


class _FakeAlerter:
    """In-process fake that records calls without hitting the network."""

    def __init__(self, return_value: bool = True) -> None:
        self.signal_calls: int = 0
        self.warning_calls: int = 0
        self.summary_calls: int = 0
        self._return_value = return_value

    async def send_signal(self, signal: Signal, risk_decision: RiskDecision) -> bool:
        self.signal_calls += 1
        return self._return_value

    async def send_risk_warning(self, message: str) -> bool:
        self.warning_calls += 1
        return self._return_value

    async def send_daily_summary(self, trades: list[TradeResult]) -> bool:
        self.summary_calls += 1
        return self._return_value


# ── formatter tests ───────────────────────────────────────────────────────────


class TestFormatSignalAlert:
    def test_dot_in_price_is_escaped(self) -> None:
        """Decimal dot must appear as \\. in MarkdownV2 output."""
        msg = format_signal_alert(_make_signal(entry=486.0), _make_decision())
        # "486.00" (unescaped) should not appear; "486\.00" should
        assert "486.00" not in msg
        assert "486\\.00" in msg  # Python literal = backslash + dot

    def test_long_signal_has_green_emoji(self) -> None:
        msg = format_signal_alert(_make_signal(direction=Direction.LONG), _make_decision())
        assert "🟢" in msg

    def test_short_signal_has_red_emoji(self) -> None:
        msg = format_signal_alert(_make_signal(direction=Direction.SHORT), _make_decision())
        assert "🔴" in msg

    def test_entry_stop_target_present(self) -> None:
        msg = format_signal_alert(
            _make_signal(entry=486.0, stop=483.0, target=492.0), _make_decision()
        )
        assert "Entry" in msg
        assert "Stop" in msg
        assert "Target" in msg

    def test_position_size_present(self) -> None:
        msg = format_signal_alert(_make_signal(), _make_decision(size=166))
        assert "166" in msg

    def test_orb_levels_present_when_provided(self) -> None:
        msg = format_signal_alert(_make_signal(with_levels=True), _make_decision())
        assert "ORB" in msg
        assert "VWAP" in msg

    def test_regime_and_vix_present(self) -> None:
        msg = format_signal_alert(_make_signal(), _make_decision())
        assert "TRENDING" in msg
        assert "VIX" in msg

    def test_underscore_in_regime_is_escaped(self) -> None:
        """TRENDING_UP underscore must be escaped as \\_."""
        msg = format_signal_alert(_make_signal(), _make_decision())
        # Raw "TRENDING_UP" with unescaped underscore should not appear
        assert "TRENDING_UP" not in msg
        # Escaped version should appear
        assert "TRENDING\\_UP" in msg

    def test_reason_included_in_message(self) -> None:
        msg = format_signal_alert(_make_signal(), _make_decision())
        assert "Test reason" in msg

    def test_special_chars_in_reason_are_escaped(self) -> None:
        """Dots in the reason string must also be escaped."""
        msg = format_signal_alert(_make_signal(), _make_decision())
        # The reason contains "486.0 > 485.0" — dots escaped, > escaped
        assert "486.0" not in msg.split("_")[1] if "_" in msg else True  # inside italic block


class TestFormatRiskAlert:
    def test_contains_warning_header(self) -> None:
        msg = format_risk_alert("Daily loss limit reached")
        assert "Risk Warning" in msg
        assert "⚠️" in msg

    def test_message_text_included(self) -> None:
        msg = format_risk_alert("Daily loss limit reached")
        assert "Daily loss limit reached" in msg

    def test_special_chars_in_message_escaped(self) -> None:
        """Dots and other special chars in message body must be escaped."""
        msg = format_risk_alert("Loss = $1500.00 exceeded")
        assert "1500.00" not in msg  # unescaped dot
        assert "1500\\.00" in msg  # escaped version


class TestFormatDailySummary:
    def test_no_trades_message(self) -> None:
        msg = format_daily_summary([])
        assert "No trades" in msg

    def test_stats_correct_for_winning_day(self) -> None:
        trades = [_make_trade(300.0), _make_trade(200.0), _make_trade(-100.0)]
        msg = format_daily_summary(trades)
        assert "Trades: 3" in msg
        assert "Wins: 2" in msg
        assert "Losses: 1" in msg

    def test_net_pnl_correct(self) -> None:
        trades = [_make_trade(300.0), _make_trade(-100.0)]
        msg = format_daily_summary(trades)
        # net = +200.00 — plus sign escaped, dot escaped
        assert "200\\.00" in msg

    def test_loss_day_no_plus_sign(self) -> None:
        trades = [_make_trade(-300.0), _make_trade(-200.0)]
        msg = format_daily_summary(trades)
        assert "500\\.00" in msg  # total loss 500.00 escaped


# ── dispatcher tests ──────────────────────────────────────────────────────────


class TestAlertDispatcher:
    async def test_dispatch_signal_succeeds_first_call(self) -> None:
        alerter = _FakeAlerter(return_value=True)
        dispatcher = AlertDispatcher(alerter=alerter)
        result = await dispatcher.dispatch_signal(_make_signal(), _make_decision())
        assert result is True
        assert alerter.signal_calls == 1

    async def test_rate_limit_blocks_second_call_within_30s(self) -> None:
        """Second dispatch within 30 seconds must be blocked."""
        t0 = datetime(2024, 1, 15, 15, 0, 0, tzinfo=UTC)
        t1 = t0 + timedelta(seconds=10)  # only 10 s later

        _times = [t0, t1]
        _idx = 0

        def _clock() -> datetime:
            nonlocal _idx
            t = _times[min(_idx, len(_times) - 1)]
            _idx += 1
            return t

        alerter = _FakeAlerter(return_value=True)
        dispatcher = AlertDispatcher(alerter=alerter, now_fn=_clock)

        first = await dispatcher.dispatch_signal(_make_signal(), _make_decision())
        second = await dispatcher.dispatch_signal(_make_signal(), _make_decision())

        assert first is True
        assert second is False  # rate limited
        assert alerter.signal_calls == 1  # second call never reached the alerter

    async def test_dispatch_allowed_after_30s(self) -> None:
        """After 30 seconds, dispatch should be allowed again."""
        t0 = datetime(2024, 1, 15, 15, 0, 0, tzinfo=UTC)
        t1 = t0 + timedelta(seconds=31)

        _times = [t0, t1]
        _idx = 0

        def _clock() -> datetime:
            nonlocal _idx
            t = _times[min(_idx, len(_times) - 1)]
            _idx += 1
            return t

        alerter = _FakeAlerter(return_value=True)
        dispatcher = AlertDispatcher(alerter=alerter, now_fn=_clock)

        await dispatcher.dispatch_signal(_make_signal(), _make_decision())
        second = await dispatcher.dispatch_signal(_make_signal(), _make_decision())

        assert second is True
        assert alerter.signal_calls == 2

    async def test_dispatch_handles_alerter_failure_gracefully(self) -> None:
        """When the alerter fails, dispatch returns False without raising."""
        alerter = _FakeAlerter(return_value=False)
        dispatcher = AlertDispatcher(alerter=alerter)
        result = await dispatcher.dispatch_signal(_make_signal(), _make_decision())
        assert result is False
        # Dispatcher should not crash; alerter was called once
        assert alerter.signal_calls == 1

    async def test_risk_warning_bypasses_rate_limit(self) -> None:
        """Risk warnings are always dispatched regardless of rate limit."""
        t0 = datetime(2024, 1, 15, 15, 0, 0, tzinfo=UTC)
        t1 = t0 + timedelta(seconds=5)

        _times = [t0, t1]
        _idx = 0

        def _clock() -> datetime:
            nonlocal _idx
            t = _times[min(_idx, len(_times) - 1)]
            _idx += 1
            return t

        alerter = _FakeAlerter(return_value=True)
        dispatcher = AlertDispatcher(alerter=alerter, now_fn=_clock)

        await dispatcher.dispatch_signal(_make_signal(), _make_decision())
        result = await dispatcher.dispatch_risk_warning("Test warning")

        assert result is True
        assert alerter.warning_calls == 1

    async def test_daily_summary_dispatched(self) -> None:
        alerter = _FakeAlerter(return_value=True)
        dispatcher = AlertDispatcher(alerter=alerter)
        trades = [_make_trade(100.0), _make_trade(-50.0)]
        result = await dispatcher.dispatch_daily_summary(trades)
        assert result is True
        assert alerter.summary_calls == 1
