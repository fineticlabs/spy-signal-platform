"""Previous-day and premarket price levels.

``PreviousDayLevels`` queries SQLite for yesterday's high, low, and close.
``PremarketLevels`` accumulates bars in the 4:00-9:30 ET window; if no
premarket data arrives it remains inactive and logs a warning once.
"""

from __future__ import annotations

from datetime import UTC, date, time
from datetime import datetime as _dt
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

import structlog

from src.levels.trading_calendar import last_trading_day

if TYPE_CHECKING:
    from decimal import Decimal

    from src.models import Bar
    from src.storage.database import BarDatabase

logger = structlog.get_logger(__name__)

_ET = ZoneInfo("America/New_York")
_PREMARKET_START = time(4, 0)
_SESSION_OPEN = time(9, 30)

# Regular-session window in ET; converted to UTC for bar queries.
_SESSION_START_ET = time(9, 30)
_SESSION_END_ET = time(16, 0)


class PreviousDayLevels:
    """Loads the prior day's high, low, and close from the bar database.

    Args:
        db:     A connected :class:`~src.storage.database.BarDatabase`.
        symbol: Ticker symbol (default ``"SPY"``).
    """

    def __init__(self, db: BarDatabase, symbol: str = "SPY") -> None:
        self._db = db
        self._symbol = symbol
        self._high: Decimal | None = None
        self._low: Decimal | None = None
        self._close: Decimal | None = None
        self._loaded_for: date | None = None

    def load(self, session_date: date) -> None:
        """Load yesterday's levels for *session_date*.

        Skips the load if levels for *session_date* are already cached.

        Args:
            session_date: Today's trading date in ET.
        """
        if self._loaded_for == session_date:
            return

        from src.models import TimeFrame

        prev_date = last_trading_day(session_date)

        # Build the UTC query window from the ET trading session, not from
        # midnight-to-midnight UTC.  This correctly handles the fact that ET
        # market hours map to a specific UTC range that varies by DST offset.
        session_start_et = _dt.combine(prev_date, _SESSION_START_ET).replace(tzinfo=_ET)
        session_end_et = _dt.combine(prev_date, _SESSION_END_ET).replace(tzinfo=_ET)
        start_utc = session_start_et.astimezone(UTC)
        end_utc = session_end_et.astimezone(UTC)

        try:
            bars = self._db.query_bars(
                symbol=self._symbol,
                timeframe=TimeFrame.ONE_MIN,
                start=start_utc,
                end=end_utc,
            )
        except Exception as exc:
            logger.error("pdl_query_failed", date=str(prev_date), error=str(exc))
            return

        if not bars:
            logger.warning("pdl_no_data", prev_date=str(prev_date), symbol=self._symbol)
            return

        self._high = max(b.high for b in bars)
        self._low = min(b.low for b in bars)
        self._close = bars[-1].close
        self._loaded_for = session_date

        logger.info(
            "pdl_loaded",
            prev_date=str(prev_date),
            session_date=str(session_date),
            high=str(self._high),
            low=str(self._low),
            close=str(self._close),
        )

    @property
    def high(self) -> Decimal | None:
        """Previous day high (PDH)."""
        return self._high

    @property
    def low(self) -> Decimal | None:
        """Previous day low (PDL)."""
        return self._low

    @property
    def close(self) -> Decimal | None:
        """Previous day close (PDC)."""
        return self._close

    def __repr__(self) -> str:
        return f"PreviousDayLevels(high={self._high}, low={self._low}, close={self._close})"


class PremarketLevels:
    """Tracks high and low during the premarket session (4:00-9:30 ET).

    If no bars fall in the premarket window a single warning is logged and the
    tracker stays dormant — downstream code must handle ``None`` values.
    """

    def __init__(self) -> None:
        self._highs: list[Decimal] = []
        self._lows: list[Decimal] = []
        self._session_date: date | None = None
        self._warned = False

    def _et_date(self, bar: Bar) -> date:
        return bar.timestamp.astimezone(_ET).date()

    def _et_time(self, bar: Bar) -> time:
        return bar.timestamp.astimezone(_ET).time()

    def update(self, bar: Bar) -> None:
        """Feed a bar.  Only premarket bars (4:00-9:30 ET) are accumulated."""
        bar_date = self._et_date(bar)

        if self._session_date != bar_date:
            self._highs.clear()
            self._lows.clear()
            self._session_date = bar_date
            self._warned = False

        bar_time = self._et_time(bar)
        if _PREMARKET_START <= bar_time < _SESSION_OPEN:
            self._highs.append(bar.high)
            self._lows.append(bar.low)

    def finalize(self) -> None:
        """Call once the regular session opens (9:30 ET) to log a warning if
        no premarket data was received for the day."""
        if not self._highs and not self._warned:
            logger.warning(
                "premarket_no_data",
                date=str(self._session_date),
                msg="No premarket bars received; premarket levels unavailable",
            )
            self._warned = True

    @property
    def high(self) -> Decimal | None:
        """Premarket high, or ``None`` if no premarket data received."""
        return max(self._highs) if self._highs else None

    @property
    def low(self) -> Decimal | None:
        """Premarket low, or ``None`` if no premarket data received."""
        return min(self._lows) if self._lows else None

    def __repr__(self) -> str:
        return f"PremarketLevels(high={self.high}, low={self.low})"
