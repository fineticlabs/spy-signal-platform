"""Key price level orchestration.

``LevelManager`` is the single entry point for all level trackers.  Feed it
completed bars; retrieve a :class:`~src.models.LevelSnapshot` on demand.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

import structlog

from src.levels.daily_levels import PremarketLevels, PreviousDayLevels
from src.levels.dynamic import DayTracker
from src.levels.opening_range import OpeningRangeTracker
from src.levels.vwap import SessionVWAP
from src.models import LevelSnapshot

if TYPE_CHECKING:
    from datetime import date

    from src.models import Bar
    from src.storage.database import BarDatabase

logger = structlog.get_logger(__name__)

_ET = ZoneInfo("America/New_York")


class LevelManager:
    """Orchestrates all key price level trackers.

    Args:
        db:     Optional :class:`~src.storage.database.BarDatabase` for
                previous-day levels.  If omitted, PDH/PDL/PDC are ``None``.
        symbol: Ticker symbol (default ``"SPY"``).
    """

    def __init__(
        self,
        db: BarDatabase | None = None,
        symbol: str = "SPY",
    ) -> None:
        self._db = db
        self._symbol = symbol
        self._orb = OpeningRangeTracker()
        self._vwap = SessionVWAP()
        self._day = DayTracker()
        self._premarket = PremarketLevels()
        self._pdl: PreviousDayLevels | None = (
            PreviousDayLevels(db=db, symbol=symbol) if db is not None else None
        )
        self._last_date: date | None = None

    def update(self, bar: Bar) -> None:
        """Feed a completed bar into all active trackers."""
        bar_date = bar.timestamp.astimezone(_ET).date()

        if bar_date != self._last_date:
            self._last_date = bar_date
            if self._pdl is not None:
                try:
                    self._pdl.load(bar_date)
                except Exception as exc:
                    logger.error("level_manager_pdl_load_failed", error=str(exc))

        self._premarket.update(bar)
        self._orb.update(bar)
        self._vwap.update(bar)
        self._day.update(bar)

    def get_levels(self) -> LevelSnapshot:
        """Return a :class:`~src.models.LevelSnapshot` of all current values."""
        pdl = self._pdl
        return LevelSnapshot(
            orb_high=self._orb.orb_high,
            orb_low=self._orb.orb_low,
            orb_midpoint=self._orb.orb_midpoint,
            orb_range=self._orb.orb_range,
            orb_complete=self._orb.is_complete,
            orb15_high=self._orb.orb15_high,
            orb15_low=self._orb.orb15_low,
            orb15_complete=self._orb.orb15_complete,
            vwap=self._vwap.vwap,
            vwap_upper_1=self._vwap.upper_1,
            vwap_lower_1=self._vwap.lower_1,
            vwap_upper_2=self._vwap.upper_2,
            vwap_lower_2=self._vwap.lower_2,
            high_of_day=self._day.high_of_day,
            low_of_day=self._day.low_of_day,
            last_price=self._day.last_price,
            prev_day_high=pdl.high if pdl is not None else None,
            prev_day_low=pdl.low if pdl is not None else None,
            prev_day_close=pdl.close if pdl is not None else None,
            premarket_high=self._premarket.high,
            premarket_low=self._premarket.low,
        )

    def __repr__(self) -> str:
        return f"LevelManager(symbol={self._symbol}, orb_complete={self._orb.is_complete})"
