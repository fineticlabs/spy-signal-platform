"""Tests for src.alerts.telegram.TelegramAlerter."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from telegram.error import TelegramError

from src.alerts.telegram import _BACKOFF_BASE, _MAX_ATTEMPTS, TelegramAlerter
from src.models import Direction, Regime, RiskDecision, Signal, TimeFrame, TradeResult

# ── Fixtures ────────────────────────────────────────────────────────────────


def _make_signal() -> Signal:
    return Signal(
        symbol="SPY",
        direction=Direction.LONG,
        strategy_name="ORB_Breakout",
        entry_price=Decimal("481.00"),
        stop_price=Decimal("479.50"),
        target_price=Decimal("484.00"),
        risk_reward_ratio=Decimal("2.0"),
        confidence_score=4,
        reason="ORB high broken with volume surge",
        timeframe=TimeFrame.ONE_MIN,
        regime=Regime.TRENDING_UP,
        timestamp=datetime(2024, 1, 15, 14, 40, tzinfo=UTC),
    )


def _make_risk_decision(*, approved: bool = True) -> RiskDecision:
    return RiskDecision(
        approved=approved,
        reason="All risk checks passed",
        position_size=100,
    )


def _make_trade_result(pnl: Decimal | None = None) -> TradeResult:
    return TradeResult(
        signal=_make_signal(),
        pnl=pnl or Decimal("250.00"),
        timestamp=datetime(2024, 1, 15, 15, 30, tzinfo=UTC),
    )


@pytest.fixture()
def mock_bot() -> MagicMock:
    with patch("src.alerts.telegram.Bot") as bot_cls:
        bot_instance = MagicMock()
        bot_instance.send_message = AsyncMock()
        bot_cls.return_value = bot_instance
        yield bot_instance


@pytest.fixture()
def alerter(mock_bot: MagicMock) -> TelegramAlerter:
    return TelegramAlerter(
        bot_token="fake-token",  # noqa: S106
        chat_id="12345",
        account_size=Decimal("50000"),
        risk_pct=Decimal("1.0"),
        max_concurrent_positions=3,
    )


# ── Constructor ─────────────────────────────────────────────────────────────


class TestConstructor:
    def test_stores_chat_id(self, alerter: TelegramAlerter) -> None:
        assert alerter._chat_id == "12345"

    def test_stores_account_size(self, alerter: TelegramAlerter) -> None:
        assert alerter._account_size == Decimal("50000")

    def test_stores_risk_pct(self, alerter: TelegramAlerter) -> None:
        assert alerter._risk_pct == Decimal("1.0")

    def test_stores_max_concurrent_positions(self, alerter: TelegramAlerter) -> None:
        assert alerter._max_concurrent_positions == 3

    def test_creates_bot_with_token(self) -> None:
        with patch("src.alerts.telegram.Bot") as bot_cls:
            bot_cls.return_value = MagicMock()
            TelegramAlerter(bot_token="my-token", chat_id="999")  # noqa: S106
            bot_cls.assert_called_once_with(token="my-token")  # noqa: S106

    def test_optional_params_default_none(self) -> None:
        with patch("src.alerts.telegram.Bot") as bot_cls:
            bot_cls.return_value = MagicMock()
            a = TelegramAlerter(bot_token="tok", chat_id="1")  # noqa: S106
            assert a._account_size is None
            assert a._risk_pct is None
            assert a._max_concurrent_positions is None


# ── send_signal ─────────────────────────────────────────────────────────────


class TestSendSignal:
    @pytest.mark.asyncio()
    async def test_calls_send_message_with_markdown_v2(
        self, alerter: TelegramAlerter, mock_bot: MagicMock
    ) -> None:
        result = await alerter.send_signal(_make_signal(), _make_risk_decision())
        mock_bot.send_message.assert_called_once()
        call_kwargs = mock_bot.send_message.call_args.kwargs
        assert call_kwargs["parse_mode"] == "MarkdownV2"
        assert call_kwargs["chat_id"] == "12345"
        assert result is True

    @pytest.mark.asyncio()
    async def test_returns_true_on_success(
        self, alerter: TelegramAlerter, mock_bot: MagicMock
    ) -> None:
        result = await alerter.send_signal(_make_signal(), _make_risk_decision())
        assert result is True

    @pytest.mark.asyncio()
    async def test_retries_on_telegram_error_then_succeeds(
        self, alerter: TelegramAlerter, mock_bot: MagicMock
    ) -> None:
        mock_bot.send_message.side_effect = [
            TelegramError("rate limit"),
            TelegramError("network"),
            None,  # success on 3rd attempt
        ]
        with patch("src.alerts.telegram.asyncio.sleep", new_callable=AsyncMock):
            result = await alerter.send_signal(_make_signal(), _make_risk_decision())
        assert result is True
        assert mock_bot.send_message.call_count == 3

    @pytest.mark.asyncio()
    async def test_returns_false_after_max_failed_attempts(
        self, alerter: TelegramAlerter, mock_bot: MagicMock
    ) -> None:
        mock_bot.send_message.side_effect = TelegramError("permanent failure")
        with patch("src.alerts.telegram.asyncio.sleep", new_callable=AsyncMock):
            result = await alerter.send_signal(_make_signal(), _make_risk_decision())
        assert result is False
        assert mock_bot.send_message.call_count == _MAX_ATTEMPTS

    @pytest.mark.asyncio()
    async def test_formats_signal_with_account_params(
        self, alerter: TelegramAlerter, mock_bot: MagicMock
    ) -> None:
        with patch("src.alerts.telegram.format_signal_alert", return_value="test") as fmt:
            await alerter.send_signal(_make_signal(), _make_risk_decision())
            fmt.assert_called_once()
            kwargs = fmt.call_args.kwargs
            assert kwargs["account_size"] == Decimal("50000")
            assert kwargs["risk_pct"] == Decimal("1.0")
            assert kwargs["max_concurrent_positions"] == 3


# ── Exponential backoff ─────────────────────────────────────────────────────


class TestExponentialBackoff:
    @pytest.mark.asyncio()
    async def test_backoff_waits_correct_durations(
        self, alerter: TelegramAlerter, mock_bot: MagicMock
    ) -> None:
        """Backoff waits: 2^0=1s, 2^1=2s (no wait after 3rd failure)."""
        mock_bot.send_message.side_effect = TelegramError("fail")
        sleep_calls: list[float] = []

        async def capture_sleep(seconds: float) -> None:
            sleep_calls.append(seconds)

        with patch("src.alerts.telegram.asyncio.sleep", side_effect=capture_sleep):
            await alerter._send("test")

        # Attempt 0 fails -> sleep(2^0=1), attempt 1 fails -> sleep(2^1=2), attempt 2 fails -> no sleep
        assert sleep_calls == [_BACKOFF_BASE**0, _BACKOFF_BASE**1]

    @pytest.mark.asyncio()
    async def test_no_sleep_on_first_success(
        self, alerter: TelegramAlerter, mock_bot: MagicMock
    ) -> None:
        with patch("src.alerts.telegram.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await alerter._send("test")
        mock_sleep.assert_not_called()

    @pytest.mark.asyncio()
    async def test_single_retry_waits_one_second(
        self, alerter: TelegramAlerter, mock_bot: MagicMock
    ) -> None:
        mock_bot.send_message.side_effect = [TelegramError("once"), None]
        sleep_calls: list[float] = []

        async def capture_sleep(seconds: float) -> None:
            sleep_calls.append(seconds)

        with patch("src.alerts.telegram.asyncio.sleep", side_effect=capture_sleep):
            result = await alerter._send("test")

        assert result is True
        assert sleep_calls == [1.0]


# ── send_risk_warning ───────────────────────────────────────────────────────


class TestSendRiskWarning:
    @pytest.mark.asyncio()
    async def test_formats_and_sends(self, alerter: TelegramAlerter, mock_bot: MagicMock) -> None:
        with patch("src.alerts.telegram.format_risk_alert", return_value="⚠️ Risk") as fmt:
            result = await alerter.send_risk_warning("daily loss 2.5%")
            fmt.assert_called_once_with("daily loss 2.5%")
        assert result is True
        mock_bot.send_message.assert_called_once()

    @pytest.mark.asyncio()
    async def test_returns_false_on_failure(
        self, alerter: TelegramAlerter, mock_bot: MagicMock
    ) -> None:
        mock_bot.send_message.side_effect = TelegramError("fail")
        with patch("src.alerts.telegram.asyncio.sleep", new_callable=AsyncMock):
            result = await alerter.send_risk_warning("bad news")
        assert result is False


# ── send_daily_summary ──────────────────────────────────────────────────────


class TestSendDailySummary:
    @pytest.mark.asyncio()
    async def test_formats_and_sends(self, alerter: TelegramAlerter, mock_bot: MagicMock) -> None:
        trades = [_make_trade_result(), _make_trade_result(pnl=Decimal("-100.00"))]
        with patch("src.alerts.telegram.format_daily_summary", return_value="summary") as fmt:
            result = await alerter.send_daily_summary(trades)
            fmt.assert_called_once_with(trades)
        assert result is True

    @pytest.mark.asyncio()
    async def test_empty_trades_list(self, alerter: TelegramAlerter, mock_bot: MagicMock) -> None:
        with patch("src.alerts.telegram.format_daily_summary", return_value="no trades"):
            result = await alerter.send_daily_summary([])
        assert result is True


# ── send_status ───────────────────────────────────────────────────────────


class TestSendStatus:
    @pytest.mark.asyncio()
    async def test_formats_with_status_header(
        self, alerter: TelegramAlerter, mock_bot: MagicMock
    ) -> None:
        result = await alerter.send_status("Scanner ready")
        assert result is True
        call_text = mock_bot.send_message.call_args.kwargs["text"]
        assert "System Status" in call_text
        assert "Scanner ready" in call_text
        # Must NOT contain "Risk Warning"
        assert "Risk Warning" not in call_text


# ── Rate limiting NOT in TelegramAlerter ────────────────────────────────────


class TestNoRateLimiting:
    def test_no_rate_limit_attribute(self, alerter: TelegramAlerter) -> None:
        """Rate limiting is handled by AlertDispatcher, not TelegramAlerter."""
        assert not hasattr(alerter, "_rate_limit")
        assert not hasattr(alerter, "_last_sent")
        assert not hasattr(alerter, "_cooldown")

    def test_max_attempts_is_three(self) -> None:
        assert _MAX_ATTEMPTS == 3

    def test_backoff_base_is_two(self) -> None:
        assert _BACKOFF_BASE == 2.0
