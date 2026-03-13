"""Market regime detector: classifies current conditions using ADX and VIX."""

from __future__ import annotations

from decimal import Decimal

import structlog

from src.models import Regime

logger = structlog.get_logger(__name__)

_ADX_CHOPPY = Decimal("15")
_ADX_TRENDING = Decimal("25")
_ADX_TRADEABLE = Decimal("20")
_VIX_HIGH = Decimal("25")


class RegimeDetector:
    """Classifies the current market regime from ADX (15-min) and VIX level.

    Update inputs via :meth:`update`.  Read the current classification via
    :attr:`current_regime` and tradability via :attr:`is_tradeable`.

    Regime logic::

        ADX < 15              -> CHOPPY
        ADX > 25, trend up    -> TRENDING_UP
        ADX > 25, trend down  -> TRENDING_DOWN
        otherwise             -> RANGING
    """

    def __init__(self) -> None:
        self._vix: Decimal | None = None
        self._adx: Decimal | None = None
        self._trend_up: bool | None = None

    def update(
        self,
        vix: Decimal | None = None,
        adx: Decimal | None = None,
        trending_up: bool | None = None,
    ) -> None:
        """Update regime inputs.

        Args:
            vix:         Current VIX index level.
            adx:         Current ADX(14) value on the 15-min chart.
            trending_up: ``True`` = price trending up, ``False`` = down.
                         Only used when ADX > 25 to distinguish direction.
        """
        if vix is not None:
            self._vix = vix
        if adx is not None:
            self._adx = adx
        if trending_up is not None:
            self._trend_up = trending_up
        logger.debug(
            "regime_updated",
            vix=str(self._vix),
            adx=str(self._adx),
            regime=str(self.current_regime),
        )

    @property
    def current_regime(self) -> Regime:
        """Current market regime classification."""
        adx = self._adx
        if adx is None:
            return Regime.RANGING
        if adx < _ADX_CHOPPY:
            return Regime.CHOPPY
        if adx > _ADX_TRENDING:
            if self._trend_up is True:
                return Regime.TRENDING_UP
            if self._trend_up is False:
                return Regime.TRENDING_DOWN
        return Regime.RANGING

    @property
    def vix_level(self) -> Decimal | None:
        """Current VIX level, or ``None`` if not yet set."""
        return self._vix

    @property
    def adx_value(self) -> Decimal | None:
        """Current ADX value, or ``None`` if not yet set."""
        return self._adx

    @property
    def is_tradeable(self) -> bool:
        """``True`` when VIX < 25 and ADX > 20.

        Returns ``False`` (conservative) if either VIX or ADX is not yet set.
        """
        vix = self._vix
        adx = self._adx
        if vix is None or adx is None:
            return False
        return vix < _VIX_HIGH and adx > _ADX_TRADEABLE

    def __repr__(self) -> str:
        return f"RegimeDetector(regime={self.current_regime}, " f"vix={self._vix}, adx={self._adx})"
