"""Alert dispatcher: routes approved signals to configured alert channels."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import structlog

if TYPE_CHECKING:
    from collections.abc import Callable

    from src.models import RiskDecision, Signal, TradeResult

logger = structlog.get_logger(__name__)

_RATE_LIMIT_SECONDS = 30


@runtime_checkable
class _AlertChannel(Protocol):
    """Protocol that any alert backend must satisfy."""

    async def send_signal(self, signal: Signal, risk_decision: RiskDecision) -> bool: ...

    async def send_risk_warning(self, message: str) -> bool: ...

    async def send_daily_summary(self, trades: list[TradeResult]) -> bool: ...


class AlertDispatcher:
    """Routes approved signals to one or more alert channels.

    For MVP, the single channel is Telegram.  Rate limiting prevents alert spam
    (max one signal alert per 30 seconds).  Risk warnings and daily summaries
    bypass the rate limit.

    Args:
        alerter: Any object implementing :class:`_AlertChannel` (e.g.
                 :class:`~src.alerts.telegram.TelegramAlerter`).
        now_fn:  Optional callable returning the current UTC datetime.  Inject
                 a fixed clock in tests to control rate-limit behaviour.
    """

    def __init__(
        self,
        alerter: _AlertChannel,
        now_fn: Callable[[], datetime] | None = None,
    ) -> None:
        self._alerter = alerter
        self._last_signal_time: datetime | None = None
        self._now_fn: Callable[[], datetime] = (
            now_fn if now_fn is not None else lambda: datetime.now(UTC)
        )

    # ── public dispatch methods ────────────────────────────────────────────────

    async def dispatch_signal(self, signal: Signal, risk_decision: RiskDecision) -> bool:
        """Dispatch a signal alert, subject to rate limiting.

        Returns:
            ``True`` if the alert was sent successfully, ``False`` if rate-limited
            or if the underlying channel failed.
        """
        if not self._rate_ok():
            logger.warning(
                "alert_rate_limited",
                strategy=signal.strategy_name,
                direction=str(signal.direction),
                seconds_remaining=self._seconds_until_allowed(),
            )
            return False

        success = await self._alerter.send_signal(signal, risk_decision)
        if success:
            self._last_signal_time = self._now_fn()
            logger.info(
                "alert_dispatched",
                strategy=signal.strategy_name,
                direction=str(signal.direction),
            )
        else:
            logger.error(
                "alert_dispatch_failed",
                strategy=signal.strategy_name,
                direction=str(signal.direction),
            )
        return success

    async def dispatch_risk_warning(self, message: str) -> bool:
        """Dispatch a risk-management warning (not rate-limited).

        Returns:
            ``True`` if delivered, ``False`` if the channel failed.
        """
        success = await self._alerter.send_risk_warning(message)
        if success:
            logger.info("risk_warning_dispatched", message=message[:80])
        else:
            logger.error("risk_warning_dispatch_failed", message=message[:80])
        return success

    async def dispatch_daily_summary(self, trades: list[TradeResult]) -> bool:
        """Dispatch the end-of-day trade summary (not rate-limited).

        Returns:
            ``True`` if delivered, ``False`` if the channel failed.
        """
        success = await self._alerter.send_daily_summary(trades)
        if success:
            logger.info("daily_summary_dispatched", trade_count=len(trades))
        else:
            logger.error("daily_summary_dispatch_failed", trade_count=len(trades))
        return success

    # ── helpers ────────────────────────────────────────────────────────────────

    def _rate_ok(self) -> bool:
        if self._last_signal_time is None:
            return True
        return (self._now_fn() - self._last_signal_time).total_seconds() >= _RATE_LIMIT_SECONDS

    def _seconds_until_allowed(self) -> float:
        if self._last_signal_time is None:
            return 0.0
        elapsed = (self._now_fn() - self._last_signal_time).total_seconds()
        return max(0.0, _RATE_LIMIT_SECONDS - elapsed)
