"""Dynamic intraday price tracking (High of Day / Low of Day)."""

from __future__ import annotations

from datetime import date, time
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

import structlog

if TYPE_CHECKING:
    from decimal import Decimal

    from src.models import Bar

logger = structlog.get_logger(__name__)

_ET = ZoneInfo("America/New_York")
_SESSION_OPEN = time(9, 30)


class DayTracker:
    """Tracks high-of-day, low-of-day, and last price for the current session.

    Resets automatically when a bar from a new trading date is received.
    Only regular-session bars (>= 9:30 ET) are included.
    """

    def __init__(self) -> None:
        self._high: Decimal | None = None
        self._low: Decimal | None = None
        self._last: Decimal | None = None
        self._session_date: date | None = None

    def update(self, bar: Bar) -> None:
        """Update HOD / LOD / last price from a completed bar."""
        bar_date = bar.timestamp.astimezone(_ET).date()
        bar_time = bar.timestamp.astimezone(_ET).time()

        if self._session_date != bar_date:
            logger.info("daytracker_reset", date=str(bar_date))
            self._high = None
            self._low = None
            self._last = None
            self._session_date = bar_date

        if bar_time < _SESSION_OPEN:
            return  # ignore premarket

        if self._high is None or bar.high > self._high:
            self._high = bar.high
        if self._low is None or bar.low < self._low:
            self._low = bar.low
        self._last = bar.close

    @property
    def high_of_day(self) -> Decimal | None:
        """Highest high seen so far today."""
        return self._high

    @property
    def low_of_day(self) -> Decimal | None:
        """Lowest low seen so far today."""
        return self._low

    @property
    def last_price(self) -> Decimal | None:
        """Most recent close price."""
        return self._last

    def __repr__(self) -> str:
        return f"DayTracker(hod={self._high}, lod={self._low}, last={self._last})"
