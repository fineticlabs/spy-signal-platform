"""Opening Range Breakout level tracker.

The opening range is defined by the first N completed 1-min bars of the
regular session (9:30 ET).  Two reference ranges are tracked:

- 5-min ORB  (bars 9:30-9:34 ET, complete after bar at 9:34 closes at 9:35)
- 15-min ORB (bars 9:30-9:44 ET, complete after bar at 9:44 closes at 9:45)

Both trackers reset automatically each day at the regular session open.
"""

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
_ORB5_END = time(9, 35)  # first bar outside the 5-min window
_ORB15_END = time(9, 45)  # first bar outside the 15-min window
_PREMARKET_START = time(4, 0)


def _et_time(bar: Bar) -> time:
    """Return the bar open-time as a wall-clock time in US/Eastern."""
    return bar.timestamp.astimezone(_ET).time()


def _et_date(bar: Bar) -> date:
    """Return the bar date in US/Eastern."""
    return bar.timestamp.astimezone(_ET).date()


class _RangeAccumulator:
    """Accumulates OHLCV bars until a cutoff time and exposes the range stats.

    Args:
        cutoff: ET wall-clock time at which the range is considered complete.
                Bars with ``bar_time >= cutoff`` are ignored.
    """

    def __init__(self, cutoff: time) -> None:
        self._cutoff = cutoff
        self._highs: list[Decimal] = []
        self._lows: list[Decimal] = []
        self._complete = False

    def update(self, bar: Bar) -> None:
        bar_time = _et_time(bar)
        if bar_time < _SESSION_OPEN or bar_time >= self._cutoff:
            if bar_time >= self._cutoff and self._highs:
                self._complete = True
            return
        self._highs.append(bar.high)
        self._lows.append(bar.low)

    @property
    def is_complete(self) -> bool:
        return self._complete

    @property
    def high(self) -> Decimal | None:
        return max(self._highs) if self._highs else None

    @property
    def low(self) -> Decimal | None:
        return min(self._lows) if self._lows else None

    def reset(self) -> None:
        self._highs.clear()
        self._lows.clear()
        self._complete = False


class OpeningRangeTracker:
    """Tracks 5-min and 15-min Opening Range levels for the current session.

    Feed completed 1-min bars via :meth:`update`.  Resets automatically when a
    bar from a new trading date is received.

    Attributes:
        orb5:  5-min ORB accumulator.
        orb15: 15-min ORB accumulator.
    """

    def __init__(self) -> None:
        self.orb5 = _RangeAccumulator(cutoff=_ORB5_END)
        self.orb15 = _RangeAccumulator(cutoff=_ORB15_END)
        self._session_date: date | None = None

    def update(self, bar: Bar) -> None:
        """Feed a completed 1-min bar.  Resets trackers on a new trading date."""
        bar_date = _et_date(bar)

        if self._session_date != bar_date:
            logger.info("orb_daily_reset", date=str(bar_date))
            self.orb5.reset()
            self.orb15.reset()
            self._session_date = bar_date

        self.orb5.update(bar)
        self.orb15.update(bar)

    # ── 5-min ORB convenience properties ────────────────────────────────────

    @property
    def orb_high(self) -> Decimal | None:
        """5-min ORB high, or ``None`` before session open."""
        return self.orb5.high

    @property
    def orb_low(self) -> Decimal | None:
        """5-min ORB low, or ``None`` before session open."""
        return self.orb5.low

    @property
    def orb_midpoint(self) -> Decimal | None:
        """Midpoint of the 5-min ORB, or ``None`` if range not started."""
        h, lo = self.orb5.high, self.orb5.low
        if h is None or lo is None:
            return None
        return (h + lo) / 2

    @property
    def orb_range(self) -> Decimal | None:
        """Width of the 5-min ORB (high - low), or ``None`` if not started."""
        h, lo = self.orb5.high, self.orb5.low
        if h is None or lo is None:
            return None
        return h - lo

    @property
    def is_complete(self) -> bool:
        """``True`` once the 5-min ORB window has closed (>= 9:35 ET)."""
        return self.orb5.is_complete

    # ── 15-min ORB convenience properties ───────────────────────────────────

    @property
    def orb15_high(self) -> Decimal | None:
        return self.orb15.high

    @property
    def orb15_low(self) -> Decimal | None:
        return self.orb15.low

    @property
    def orb15_complete(self) -> bool:
        return self.orb15.is_complete

    def __repr__(self) -> str:
        return (
            f"OpeningRangeTracker("
            f"orb_high={self.orb_high}, orb_low={self.orb_low}, "
            f"is_complete={self.is_complete})"
        )
