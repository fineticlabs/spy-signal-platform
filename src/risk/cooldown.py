"""Loss-streak cooldown and tilt detection for intraday risk management."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from collections.abc import Callable

    from src.models import TradeResult

logger = structlog.get_logger(__name__)

_COOLDOWN_MINUTES = 15
_TILT_THRESHOLD = 3  # consecutive losses that end the day
_COOLDOWN_THRESHOLD = 2  # consecutive losses that trigger the cooldown


class CooldownTracker:
    """Tracks consecutive losses, daily P&L, and the 15-min cooldown period.

    Feed completed trade results via :meth:`record_trade`.  Interrogate state
    via :meth:`is_cooled_down` and :meth:`is_tilted`.  Reset at session open
    with :meth:`reset_daily`.

    Args:
        now_fn: Optional callable that returns the current UTC datetime.
                Defaults to ``datetime.now(UTC)``.  Inject a fixed clock in
                tests to control cooldown timing.
    """

    def __init__(self, now_fn: Callable[[], datetime] | None = None) -> None:
        self._consecutive_losses: int = 0
        self._daily_pnl: Decimal = Decimal("0")
        self._daily_trade_count: int = 0
        self._last_loss_time: datetime | None = None
        self._now_fn: Callable[[], datetime] = (
            now_fn if now_fn is not None else lambda: datetime.now(UTC)
        )

    # ── public interface ───────────────────────────────────────────────────────

    def record_trade(self, result: TradeResult) -> None:
        """Update state from a completed trade.

        A winning trade (pnl >= 0) resets the consecutive-loss counter.
        A losing trade increments it and records the timestamp for cooldown.
        """
        self._daily_trade_count += 1
        self._daily_pnl += result.pnl

        if result.pnl < 0:
            self._consecutive_losses += 1
            self._last_loss_time = result.timestamp
        else:
            self._consecutive_losses = 0

        logger.info(
            "trade_recorded",
            pnl=str(result.pnl),
            consecutive_losses=self._consecutive_losses,
            daily_pnl=str(self._daily_pnl),
            daily_trade_count=self._daily_trade_count,
        )

    def is_cooled_down(self) -> bool:
        """``True`` when the cooldown period has elapsed and trading may resume.

        The cooldown activates after ``_COOLDOWN_THRESHOLD`` (2) consecutive
        losses and lasts ``_COOLDOWN_MINUTES`` (15) minutes from the last loss.

        Returns ``True`` immediately when fewer than 2 consecutive losses have
        occurred (no cooldown needed).
        """
        if self._consecutive_losses < _COOLDOWN_THRESHOLD:
            return True  # no cooldown triggered
        if self._last_loss_time is None:
            return True
        elapsed = self._now_fn() - self._last_loss_time
        return elapsed >= timedelta(minutes=_COOLDOWN_MINUTES)

    def is_tilted(self) -> bool:
        """``True`` when 3 or more consecutive losses have occurred (done for day)."""
        return self._consecutive_losses >= _TILT_THRESHOLD

    def reset_daily(self) -> None:
        """Reset all daily counters — call at 9:30 ET each session."""
        self._consecutive_losses = 0
        self._daily_pnl = Decimal("0")
        self._daily_trade_count = 0
        self._last_loss_time = None
        logger.info("cooldown_daily_reset")

    # ── read-only properties ───────────────────────────────────────────────────

    @property
    def consecutive_losses(self) -> int:
        """Number of consecutive losses since the last win."""
        return self._consecutive_losses

    @property
    def daily_pnl(self) -> Decimal:
        """Net realized P&L for the current session (negative = net loss)."""
        return self._daily_pnl

    @property
    def daily_trade_count(self) -> int:
        """Number of completed trades this session."""
        return self._daily_trade_count

    def __repr__(self) -> str:
        return (
            f"CooldownTracker(consecutive_losses={self._consecutive_losses}, "
            f"daily_pnl={self._daily_pnl}, trades={self._daily_trade_count})"
        )
