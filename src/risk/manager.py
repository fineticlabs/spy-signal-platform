"""Pre-trade risk gate: approves or rejects signals before execution."""

from __future__ import annotations

from datetime import time
from decimal import Decimal
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

import structlog

from src.models import RiskDecision
from src.risk.position_sizing import calculate_position_size

if TYPE_CHECKING:
    from src.config import RiskSettings
    from src.models import Signal
    from src.risk.cooldown import CooldownTracker

logger = structlog.get_logger(__name__)

_ET = ZoneInfo("America/New_York")
_SESSION_START = time(9, 35)
_CUTOFF = time(15, 45)
_LUNCH_START = time(11, 30)
_LUNCH_END = time(13, 30)
_MIN_RR = Decimal("1.5")
_LUNCH_MIN_CONFIDENCE = 4


class RiskManager:
    """Pre-trade risk gate.

    Call :meth:`approve` with each candidate :class:`~src.models.Signal`.
    Returns a :class:`~src.models.RiskDecision` (approved / rejected + reason).

    Checks are applied in fail-fast order:

    a. Daily loss limit not breached
    b. Current time within allowed window (9:35-15:45, lunch filtered unless confidence >= 4)
    c. Daily trade count below max
    d. Not tilted (3+ consecutive losses)
    e. Cooldown elapsed after 2 consecutive losses
    f. Risk/reward >= 1.5
    g. Position size feasible within 1% account risk

    Args:
        cooldown:  Shared :class:`CooldownTracker` instance.
        settings:  :class:`~src.config.RiskSettings` to use.  Defaults to the
                   application settings from the environment.
    """

    def __init__(
        self,
        cooldown: CooldownTracker,
        settings: RiskSettings | None = None,
    ) -> None:
        self._cooldown = cooldown
        if settings is not None:
            self._settings = settings
        else:
            from src.config import get_risk_settings

            self._settings = get_risk_settings()

    def approve(self, signal: Signal) -> RiskDecision:
        """Run all pre-trade checks and return a risk decision.

        Args:
            signal: Candidate signal to evaluate.

        Returns:
            :class:`~src.models.RiskDecision` with ``approved=True`` and the
            calculated ``position_size``, or ``approved=False`` with a reason.
        """
        # a. Daily loss limit
        if self._is_past_daily_loss_limit():
            reason = (
                f"Daily loss limit reached: PnL {self._cooldown.daily_pnl} "
                f">= -{self._max_daily_loss()}"
            )
            logger.warning("risk_rejected_daily_loss", reason=reason)
            return RiskDecision(approved=False, reason=reason)

        # b. Time window check
        if not self._is_time_allowed(signal):
            bar_time = signal.timestamp.astimezone(_ET).time()
            reason = f"Outside allowed trading window: {bar_time}"
            logger.warning("risk_rejected_time", reason=reason, bar_time=str(bar_time))
            return RiskDecision(approved=False, reason=reason)

        # c. Daily trade count
        if self._cooldown.daily_trade_count >= self._settings.max_trades_per_day:
            reason = (
                f"Max daily trades reached: {self._cooldown.daily_trade_count} "
                f"/ {self._settings.max_trades_per_day}"
            )
            logger.warning("risk_rejected_trade_count", reason=reason)
            return RiskDecision(approved=False, reason=reason)

        # d. Tilt check (3+ consecutive losses)
        if self._cooldown.is_tilted():
            reason = (
                f"Tilted: {self._cooldown.consecutive_losses} consecutive losses "
                f"(done for the day)"
            )
            logger.warning("risk_rejected_tilted", reason=reason)
            return RiskDecision(approved=False, reason=reason)

        # e. Cooldown check (15 min after 2 consecutive losses)
        if not self._cooldown.is_cooled_down():
            reason = (
                f"Cooldown active: {self._cooldown.consecutive_losses} consecutive losses, "
                f"15-min cooldown not elapsed"
            )
            logger.warning("risk_rejected_cooldown", reason=reason)
            return RiskDecision(approved=False, reason=reason)

        # f. Risk/reward ratio
        if signal.risk_reward_ratio < _MIN_RR:
            reason = f"R:R {signal.risk_reward_ratio:.2f} below minimum {_MIN_RR}"
            logger.warning("risk_rejected_rr", reason=reason, rr=str(signal.risk_reward_ratio))
            return RiskDecision(approved=False, reason=reason)

        # g. Position sizing
        size = calculate_position_size(
            account_size=self._settings.account_size,
            risk_pct=self._settings.risk_per_trade_pct,
            entry=signal.entry_price,
            stop=signal.stop_price,
        )
        if size == 0:
            reason = (
                f"Position size is 0: stop distance too small "
                f"(entry={signal.entry_price}, stop={signal.stop_price})"
            )
            logger.warning("risk_rejected_position_size", reason=reason)
            return RiskDecision(approved=False, reason=reason)

        logger.info(
            "risk_approved",
            strategy=signal.strategy_name,
            direction=str(signal.direction),
            entry=str(signal.entry_price),
            stop=str(signal.stop_price),
            size=size,
            rr=str(signal.risk_reward_ratio),
        )
        return RiskDecision(approved=True, reason="All risk checks passed", position_size=size)

    # ── internal helpers ───────────────────────────────────────────────────────

    def _max_daily_loss(self) -> Decimal:
        return self._settings.account_size * self._settings.max_daily_loss_pct / Decimal("100")

    def _is_past_daily_loss_limit(self) -> bool:
        return self._cooldown.daily_pnl <= -self._max_daily_loss()

    def _is_time_allowed(self, signal: Signal) -> bool:
        bar_time = signal.timestamp.astimezone(_ET).time()
        if bar_time < _SESSION_START or bar_time >= _CUTOFF:
            return False
        if _LUNCH_START <= bar_time < _LUNCH_END:
            return signal.confidence_score >= _LUNCH_MIN_CONFIDENCE
        return True
