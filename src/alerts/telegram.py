"""Async Telegram alerter with retry logic."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import structlog
from telegram import Bot
from telegram.error import TelegramError

from src.alerts.formatter import format_daily_summary, format_risk_alert, format_signal_alert

if TYPE_CHECKING:
    from src.models import RiskDecision, Signal, TradeResult

logger = structlog.get_logger(__name__)

_MAX_ATTEMPTS = 3
_BACKOFF_BASE = 2.0  # seconds; attempt N waits BACKOFF_BASE^(N-1) before retrying


class TelegramAlerter:
    """Sends formatted alerts to a Telegram chat via the Bot API.

    Args:
        bot_token: Telegram bot token from ``@BotFather``.
        chat_id:   Numeric or ``@username`` chat ID to send alerts to.
    """

    def __init__(self, bot_token: str, chat_id: str) -> None:
        self._bot = Bot(token=bot_token)
        self._chat_id = chat_id

    # ── public send methods ────────────────────────────────────────────────────

    async def send_signal(self, signal: Signal, risk_decision: RiskDecision) -> bool:
        """Send a formatted signal alert.

        Returns:
            ``True`` if delivered successfully, ``False`` if all retries failed.
        """
        text = format_signal_alert(signal, risk_decision)
        return await self._send(text)

    async def send_risk_warning(self, message: str) -> bool:
        """Send a risk-management warning message.

        Returns:
            ``True`` if delivered successfully, ``False`` if all retries failed.
        """
        text = format_risk_alert(message)
        return await self._send(text)

    async def send_daily_summary(self, trades: list[TradeResult]) -> bool:
        """Send the end-of-day trade summary.

        Returns:
            ``True`` if delivered successfully, ``False`` if all retries failed.
        """
        text = format_daily_summary(trades)
        return await self._send(text)

    # ── internal ───────────────────────────────────────────────────────────────

    async def _send(self, text: str) -> bool:
        """Send *text* with up to ``_MAX_ATTEMPTS`` retries and exponential backoff."""
        for attempt in range(_MAX_ATTEMPTS):
            try:
                await self._bot.send_message(
                    chat_id=self._chat_id,
                    text=text,
                    parse_mode="MarkdownV2",
                )
                logger.info("telegram_sent", attempt=attempt + 1, length=len(text))
                return True
            except TelegramError as exc:
                wait = _BACKOFF_BASE**attempt
                if attempt < _MAX_ATTEMPTS - 1:
                    logger.warning(
                        "telegram_retry",
                        attempt=attempt + 1,
                        wait_seconds=wait,
                        error=str(exc),
                    )
                    await asyncio.sleep(wait)
                else:
                    logger.error(
                        "telegram_send_failed",
                        attempts=_MAX_ATTEMPTS,
                        error=str(exc),
                    )
        return False
