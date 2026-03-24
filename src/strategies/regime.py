"""Market regime detector: classifies current conditions using ADX and VIX.

Provides two ADX sources:
- ``adx_value``: Streaming 1-min ADX (for regime display: TRENDING/CHOPPY)
- ``daily_adx``: TA-Lib ADX(14) on daily bars (for signal gating: > 25)

Provides 15-min EMA(20):
- ``ema20_15m``: EMA of 15-min resampled close prices (for EMA trend filter)
"""

from __future__ import annotations

from collections import deque
from decimal import Decimal
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

import structlog

from src.models import Regime

if TYPE_CHECKING:
    from src.models import Bar

logger = structlog.get_logger(__name__)

_ADX_CHOPPY = Decimal("15")
_ADX_TRENDING = Decimal("25")
_ADX_TRADEABLE = Decimal("20")
_VIX_HIGH = Decimal("25")
_ET = ZoneInfo("America/New_York")


class RegimeDetector:
    """Classifies the current market regime from ADX and VIX level.

    Update inputs via :meth:`update`.  Read the current classification via
    :attr:`current_regime` and tradability via :attr:`is_tradeable`.

    Daily ADX and 15-min EMA are computed separately from historical bars
    (matching the backtest engine's methodology).

    Regime logic::

        ADX < 15              -> CHOPPY
        ADX > 25, trend up    -> TRENDING_UP
        ADX > 25, trend down  -> TRENDING_DOWN
        otherwise             -> RANGING
    """

    def __init__(self) -> None:
        self._vix: Decimal | None = None
        self._adx: Decimal | None = None  # streaming 1-min ADX (regime display)
        self._trend_up: bool | None = None
        self._daily_adx: Decimal | None = None  # TA-Lib daily ADX (signal gate)
        self._realized_vol: Decimal | None = None  # 20-day annualized HV (vol filter)
        self._ema20_15m: Decimal | None = None  # 15-min EMA(20) (trend filter)

        # 15-min bar aggregator state
        self._15m_bars: deque[float] = deque(maxlen=40)  # last 40 15-min closes (2 days)
        self._15m_bucket_closes: list[
            float
        ] = []  # accumulates 1-min closes for current 15-min bucket
        self._15m_ema_indicator: object | None = None  # lazy-init talipp EMA

    def update(
        self,
        vix: Decimal | None = None,
        adx: Decimal | None = None,
        trending_up: bool | None = None,
    ) -> None:
        """Update regime inputs.

        Args:
            vix:         Current VIX index level.
            adx:         Current ADX(14) value on the 1-min chart (streaming).
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

    def set_realized_vol(self, value: Decimal) -> None:
        """Set the 20-day realized volatility (annualized HV).

        Matches the backtest's realized vol filter methodology.
        """
        self._realized_vol = value
        logger.info("realized_vol_set", value=str(value))

    @property
    def realized_vol(self) -> Decimal | None:
        """20-day annualized realized vol, or ``None`` if not computed."""
        return self._realized_vol

    def set_daily_adx(self, value: Decimal) -> None:
        """Set the daily ADX(14) value computed from historical daily bars.

        This is the value used by the ORB strategy's ADX > 25 signal gate,
        matching the backtest's daily ADX methodology.
        """
        self._daily_adx = value
        logger.info("daily_adx_set", value=str(value))

    def update_15m_bar(self, bar: Bar) -> None:
        """Feed a 1-min bar into the 15-min aggregator.

        Every 15 bars (by ET clock bucket), emits a 15-min close and
        updates the 15-min EMA(20).
        """
        bar_time = bar.timestamp.astimezone(_ET).time()
        self._15m_bucket_closes.append(float(bar.close))

        # Check if this bar completes a 15-min bucket (minute % 15 == 14)
        # e.g. 9:44, 9:59, 10:14...
        minute = bar_time.hour * 60 + bar_time.minute
        if (minute + 1) % 15 == 0 and self._15m_bucket_closes:
            close_15m = self._15m_bucket_closes[-1]  # last close of the 15-min bucket
            self._15m_bars.append(close_15m)
            self._15m_bucket_closes.clear()
            self._update_15m_ema(close_15m)

    def _update_15m_ema(self, close: float) -> None:
        """Update the 15-min EMA(20) with a new 15-min close."""
        from talipp.indicators import EMA

        if self._15m_ema_indicator is None:
            self._15m_ema_indicator = EMA(period=20)

        self._15m_ema_indicator.add(close)  # type: ignore[union-attr]
        vals = self._15m_ema_indicator.output_values  # type: ignore[union-attr]
        if vals and vals[-1] is not None:
            self._ema20_15m = Decimal(str(vals[-1]))

    def seed_15m_ema(self, closes_15m: list[float]) -> None:
        """Seed the 15-min EMA with historical 15-min closes.

        Call at startup with the last ~40 15-min closes from the database.
        """
        from talipp.indicators import EMA

        self._15m_ema_indicator = EMA(period=20)
        for c in closes_15m:
            self._15m_ema_indicator.add(c)
            self._15m_bars.append(c)

        vals = self._15m_ema_indicator.output_values
        if vals and vals[-1] is not None:
            self._ema20_15m = Decimal(str(vals[-1]))
            logger.info(
                "ema20_15m_seeded",
                bars=len(closes_15m),
                value=str(self._ema20_15m),
            )

    @property
    def current_regime(self) -> Regime:
        """Current market regime classification (uses streaming 1-min ADX)."""
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
        """Daily ADX value for signal gating, falling back to streaming ADX.

        Prefers :meth:`set_daily_adx` (matching backtest methodology).
        Falls back to streaming 1-min ADX if daily is not yet computed.
        """
        if self._daily_adx is not None:
            return self._daily_adx
        return self._adx

    @property
    def streaming_adx(self) -> Decimal | None:
        """Raw streaming 1-min ADX (for regime display only)."""
        return self._adx

    @property
    def daily_adx(self) -> Decimal | None:
        """Daily ADX(14), or ``None`` if not yet computed."""
        return self._daily_adx

    @property
    def ema20_15m(self) -> Decimal | None:
        """15-min EMA(20) value, or ``None`` if not yet computed."""
        return self._ema20_15m

    @property
    def is_tradeable(self) -> bool:
        """``True`` when VIX < 25 and ADX > 20.

        Returns ``False`` (conservative) if either VIX or ADX is not yet set.
        """
        vix = self._vix
        adx = self.adx_value  # uses daily_adx if available
        if vix is None or adx is None:
            return False
        return vix < _VIX_HIGH and adx > _ADX_TRADEABLE

    def __repr__(self) -> str:
        return (
            f"RegimeDetector(regime={self.current_regime}, "
            f"vix={self._vix}, adx={self._adx}, daily_adx={self._daily_adx})"
        )
