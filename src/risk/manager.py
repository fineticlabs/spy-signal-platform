"""Pre-trade risk gate: approves or rejects signals before execution."""

from __future__ import annotations

from collections.abc import Callable
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

# Sector map for correlation-aware position limits
SECTOR_MAP: dict[str, str] = {
    "AMD": "semi",
    "INTC": "semi",
    "MU": "semi",
    "MRVL": "semi",
    "SMCI": "semi",
    "SOXL": "semi",
    "ARM": "semi",
    "MSFT": "mega_tech",
    "AMZN": "mega_tech",
    "META": "mega_tech",
    "NFLX": "mega_tech",
    "TSLA": "high_beta",
    "SHOP": "high_beta",
    "PLTR": "high_beta",
    "SNOW": "high_beta",
    "MSTR": "high_beta",
    "DASH": "high_beta",
    "SPY": "index",
    "QQQ": "index",
    "TQQQ": "index_lev",
    "BA": "industrial",
    "UBER": "consumer",
    "DKNG": "consumer",
    "ROKU": "consumer",
    "PYPL": "fintech",
    "HOOD": "fintech",
}


class RiskManager:
    """Pre-trade risk gate.

    Call :meth:`approve` with each candidate :class:`~src.models.Signal`.
    Returns a :class:`~src.models.RiskDecision` (approved / rejected + reason).

    Checks are applied in fail-fast order:

    a. Daily loss limit not breached
    b. Max concurrent positions not exceeded
    c. Current time within allowed window (9:35-15:45, lunch filtered unless confidence >= 4)
    d. Daily trade count below max
    e. Not tilted (3+ consecutive losses)
    f. Cooldown elapsed after 2 consecutive losses
    g. Risk/reward >= 1.5
    h. Position size feasible within 1% account risk

    Args:
        cooldown:  Shared :class:`CooldownTracker` instance.
        settings:  :class:`~src.config.RiskSettings` to use.  Defaults to the
                   application settings from the environment.
        get_position_count: Optional callable returning the current number of
                   open positions (from Alpaca executor). If not provided,
                   max_concurrent_positions check is skipped.
    """

    def __init__(
        self,
        cooldown: CooldownTracker,
        settings: RiskSettings | None = None,
        get_position_count: Callable[[], int] | None = None,
        get_open_symbols: Callable[[], list[str]] | None = None,
    ) -> None:
        self._cooldown = cooldown
        self._get_position_count = get_position_count
        self._get_open_symbols = get_open_symbols
        self._symbol_pnl: dict[str, Decimal] = {}  # per-symbol cumulative P&L today
        self._blocked_symbols: set[str] = set()  # symbols that hit max loss
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

        # b. Per-symbol max loss check
        sym = signal.symbol
        if sym in self._blocked_symbols:
            reason = f"Symbol {sym} blocked — daily loss limit reached"
            logger.warning("risk_symbol_max_loss", reason=reason, symbol=sym)
            return RiskDecision(approved=False, reason=reason)

        # c. Max concurrent positions
        if self._get_position_count is not None:
            try:
                pos_count = self._get_position_count()
                max_pos = self._settings.max_concurrent_positions
                if pos_count >= max_pos:
                    reason = f"Max concurrent positions reached: {pos_count}/{max_pos}"
                    logger.warning("risk_max_positions_reached", reason=reason, count=pos_count)
                    return RiskDecision(approved=False, reason=reason)
            except Exception as exc:
                logger.error("risk_position_count_failed", error=str(exc))

        # d. Sector concentration limit
        if self._get_open_symbols is not None:
            try:
                open_syms = self._get_open_symbols()
                signal_sector = SECTOR_MAP.get(sym, "other")
                same_sector_count = sum(
                    1 for s in open_syms if SECTOR_MAP.get(s, "other") == signal_sector
                )
                max_sector = self._settings.max_same_sector
                if same_sector_count >= max_sector:
                    reason = (
                        f"Sector limit: {same_sector_count}/{max_sector} "
                        f"{signal_sector} positions already open"
                    )
                    logger.warning(
                        "risk_sector_limit", reason=reason, symbol=sym, sector=signal_sector
                    )
                    return RiskDecision(approved=False, reason=reason)
            except Exception as exc:
                logger.error("risk_sector_check_failed", error=str(exc))

        # e. Time window check
        if not self._is_time_allowed(signal):
            bar_time = signal.timestamp.astimezone(_ET).time()
            reason = f"Outside allowed trading window: {bar_time}"
            logger.warning("risk_rejected_time", reason=reason, bar_time=str(bar_time))
            return RiskDecision(approved=False, reason=reason)

        # f. Daily trade count
        if self._cooldown.daily_trade_count >= self._settings.max_trades_per_day:
            reason = (
                f"Max daily trades reached: {self._cooldown.daily_trade_count} "
                f"/ {self._settings.max_trades_per_day}"
            )
            logger.warning("risk_rejected_trade_count", reason=reason)
            return RiskDecision(approved=False, reason=reason)

        # g. Tilt check (3+ consecutive losses)
        if self._cooldown.is_tilted():
            reason = (
                f"Tilted: {self._cooldown.consecutive_losses} consecutive losses "
                f"(done for the day)"
            )
            logger.warning("risk_rejected_tilted", reason=reason)
            return RiskDecision(approved=False, reason=reason)

        # h. Cooldown check (15 min after 2 consecutive losses)
        if not self._cooldown.is_cooled_down():
            reason = (
                f"Cooldown active: {self._cooldown.consecutive_losses} consecutive losses, "
                f"15-min cooldown not elapsed"
            )
            logger.warning("risk_rejected_cooldown", reason=reason)
            return RiskDecision(approved=False, reason=reason)

        # i. Risk/reward ratio
        if signal.risk_reward_ratio < _MIN_RR:
            reason = f"R:R {signal.risk_reward_ratio:.2f} below minimum {_MIN_RR}"
            logger.warning("risk_rejected_rr", reason=reason, rr=str(signal.risk_reward_ratio))
            return RiskDecision(approved=False, reason=reason)

        # j. Position sizing (scaled by position_scale_factor)
        size = calculate_position_size(
            account_size=self._settings.account_size,
            risk_pct=self._settings.risk_per_trade_pct,
            entry=signal.entry_price,
            stop=signal.stop_price,
            scale_factor=self._settings.position_scale_factor,
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

    def record_symbol_pnl(self, symbol: str, pnl: Decimal) -> None:
        """Record realized P&L for a symbol and block if max loss exceeded."""
        current = self._symbol_pnl.get(symbol, Decimal("0"))
        self._symbol_pnl[symbol] = current + pnl
        if self._symbol_pnl[symbol] <= -self._settings.max_symbol_loss:
            self._blocked_symbols.add(symbol)
            logger.warning(
                "risk_symbol_blocked",
                symbol=symbol,
                cumulative_pnl=str(self._symbol_pnl[symbol]),
                threshold=str(self._settings.max_symbol_loss),
            )

    @staticmethod
    def score_signal(signal: Signal) -> float:
        """Compute a composite quality score for signal ranking.

        Used when multiple signals fire in the same minute to determine
        which ones should fill the remaining position slots.

        Score = ADX * 0.3 + confidence * 0.2 + risk_reward * 0.2 + volume_proxy * 0.3

        Returns a float (higher = better quality signal).
        """
        adx = float(signal.adx) if signal.adx is not None else 20.0
        conf = float(signal.confidence_score)
        rr = float(signal.risk_reward_ratio)
        # Normalize ADX to 0-100 range, confidence to 0-5, rr to 0-3
        score = (
            (adx / 100 * 0.3) + (conf / 5 * 0.2) + (rr / 3 * 0.2) + 0.3
        )  # volume proxy = 1.0 (already filtered)
        return round(score, 4)

    def reset_symbol_tracking(self) -> None:
        """Reset per-symbol P&L tracking — call at daily reset."""
        self._symbol_pnl.clear()
        self._blocked_symbols.clear()

    def _is_time_allowed(self, signal: Signal) -> bool:
        bar_time = signal.timestamp.astimezone(_ET).time()
        if bar_time < _SESSION_START or bar_time >= _CUTOFF:
            return False
        if _LUNCH_START <= bar_time < _LUNCH_END:
            return signal.confidence_score >= _LUNCH_MIN_CONFIDENCE
        return True
