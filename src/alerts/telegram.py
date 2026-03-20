"""Async Telegram alerter with retry logic."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import structlog
from telegram import Bot
from telegram.error import TelegramError

from src.alerts.formatter import format_daily_summary, format_risk_alert, format_signal_alert

if TYPE_CHECKING:
    from decimal import Decimal

    from src.models import RiskDecision, Signal, TradeResult

logger = structlog.get_logger(__name__)

_MAX_ATTEMPTS = 3
_BACKOFF_BASE = 2.0  # seconds; attempt N waits BACKOFF_BASE^(N-1) before retrying


class TelegramAlerter:
    """Sends formatted alerts to a Telegram chat via the Bot API.

    Args:
        bot_token:                Telegram bot token from ``@BotFather``.
        chat_id:                  Numeric or ``@username`` chat ID to send alerts to.
        account_size:             Account size in USD for position sizing display.
        risk_pct:                 Risk per trade as percentage for position sizing display.
        max_concurrent_positions: Max concurrent positions (appended as a reminder note).
    """

    def __init__(
        self,
        bot_token: str,
        chat_id: str,
        account_size: Decimal | None = None,
        risk_pct: Decimal | None = None,
        max_concurrent_positions: int | None = None,
    ) -> None:
        self._bot = Bot(token=bot_token)
        self._chat_id = chat_id
        self._account_size = account_size
        self._risk_pct = risk_pct
        self._max_concurrent_positions = max_concurrent_positions

    # ── public send methods ────────────────────────────────────────────────────

    async def send_signal(self, signal: Signal, risk_decision: RiskDecision) -> bool:
        """Send a formatted signal alert.

        Returns:
            ``True`` if delivered successfully, ``False`` if all retries failed.
        """
        text = format_signal_alert(
            signal,
            risk_decision,
            account_size=self._account_size,
            risk_pct=self._risk_pct,
            max_concurrent_positions=self._max_concurrent_positions,
        )
        return await self._send(text)

    async def send_risk_warning(self, message: str) -> bool:
        """Send a risk-management warning message.

        Returns:
            ``True`` if delivered successfully, ``False`` if all retries failed.
        """
        text = format_risk_alert(message)
        return await self._send(text)

    async def send_status(self, message: str) -> bool:
        """Send a system status notification as plain text (no Markdown parsing).

        Status messages use emoji for visual structure and don't need bold/italic.
        Sending as plain text avoids MarkdownV2 escaping issues with characters
        like ``|``, ``+``, ``(``, ``)`` that appear in stats.

        Returns:
            ``True`` if delivered successfully, ``False`` if all retries failed.
        """
        return await self._send(f"\u2139\ufe0f System Status\n\n{message}", parse_mode=None)

    async def send_daily_summary(self, trades: list[TradeResult]) -> bool:
        """Send the end-of-day trade summary.

        Returns:
            ``True`` if delivered successfully, ``False`` if all retries failed.
        """
        text = format_daily_summary(trades)
        return await self._send(text)

    # ── internal ───────────────────────────────────────────────────────────────

    async def _send(self, text: str, parse_mode: str | None = "MarkdownV2") -> bool:
        """Send *text* with up to ``_MAX_ATTEMPTS`` retries and exponential backoff.

        Args:
            text:       Message body.
            parse_mode: Telegram parse mode (``"MarkdownV2"``, ``"HTML"``, or
                        ``None`` for plain text).  Defaults to MarkdownV2.
        """
        kwargs: dict[str, object] = {"chat_id": self._chat_id, "text": text}
        if parse_mode is not None:
            kwargs["parse_mode"] = parse_mode
        for attempt in range(_MAX_ATTEMPTS):
            try:
                await self._bot.send_message(**kwargs)
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
