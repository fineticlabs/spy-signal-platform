"""Session VWAP with volume-weighted standard deviation bands.

``SessionVWAP`` incrementally computes VWAP and +-1 sigma / +-2 sigma bands from the
regular session open (9:30 ET).  The tracker resets automatically on a new
trading date.

Band formula
------------
Let ``tp_i = (high_i + low_i + close_i) / 3`` (typical price).

::

    VWAP      = sum(tp * vol) / sum(vol)
    variance  = sum(vol * (tp - VWAP)^2) / sum(vol)   [volume-weighted]
    sigma     = sqrt(variance)
    upper_k   = VWAP + k * sigma
    lower_k   = VWAP - k * sigma
"""

from __future__ import annotations

import math
from datetime import date, time
from decimal import Decimal
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

import structlog

if TYPE_CHECKING:
    from src.models import Bar

logger = structlog.get_logger(__name__)

_ET = ZoneInfo("America/New_York")
_SESSION_OPEN = time(9, 30)


def _et_date(bar: Bar) -> date:
    return bar.timestamp.astimezone(_ET).date()


def _et_time(bar: Bar) -> time:
    return bar.timestamp.astimezone(_ET).time()


class SessionVWAP:
    """Incrementally computed session VWAP + deviation bands.

    Feed completed bars via :meth:`update`.  Access the current values through
    the ``vwap``, ``upper_1``, ``lower_1``, ``upper_2``, ``lower_2`` properties.

    All values are ``None`` until at least one bar has been processed.
    """

    def __init__(self) -> None:
        self._cum_tp_vol: float = 0.0  # sum(typical_price * volume)
        self._cum_vol: float = 0.0  # sum(volume)
        self._cum_tp2_vol: float = 0.0  # sum(typical_price^2 * volume) for variance
        self._session_date: date | None = None

    # ── internal state management ─────────────────────────────────────────────

    def _reset(self) -> None:
        self._cum_tp_vol = 0.0
        self._cum_vol = 0.0
        self._cum_tp2_vol = 0.0

    # ── public interface ──────────────────────────────────────────────────────

    def update(self, bar: Bar) -> None:
        """Incorporate a completed bar into the running VWAP."""
        bar_date = _et_date(bar)
        bar_time = _et_time(bar)

        if self._session_date != bar_date:
            logger.info("vwap_daily_reset", date=str(bar_date))
            self._reset()
            self._session_date = bar_date

        if bar_time < _SESSION_OPEN:
            return  # ignore premarket bars

        typical_price = (float(bar.high) + float(bar.low) + float(bar.close)) / 3.0
        vol = float(bar.volume)

        self._cum_vol += vol
        self._cum_tp_vol += typical_price * vol
        self._cum_tp2_vol += typical_price * typical_price * vol

    # ── computed values ───────────────────────────────────────────────────────

    @property
    def _vwap_float(self) -> float | None:
        if self._cum_vol == 0.0:
            return None
        return self._cum_tp_vol / self._cum_vol

    @property
    def _sigma_float(self) -> float:
        """Volume-weighted standard deviation of typical price from VWAP."""
        if self._cum_vol == 0.0:
            return 0.0
        vwap = self._cum_tp_vol / self._cum_vol
        # variance = E[tp^2] - E[tp]^2  (König-Huygens, volume-weighted)
        variance = self._cum_tp2_vol / self._cum_vol - vwap * vwap
        return math.sqrt(max(variance, 0.0))

    @property
    def vwap(self) -> Decimal | None:
        """Current session VWAP, or ``None`` before the first bar."""
        v = self._vwap_float
        return Decimal(str(round(v, 6))) if v is not None else None

    @property
    def sigma(self) -> Decimal:
        """Volume-weighted standard deviation of price from VWAP."""
        return Decimal(str(round(self._sigma_float, 6)))

    @property
    def upper_1(self) -> Decimal | None:
        """VWAP + 1 sigma band."""
        v = self._vwap_float
        if v is None:
            return None
        return Decimal(str(round(v + self._sigma_float, 6)))

    @property
    def lower_1(self) -> Decimal | None:
        """VWAP - 1 sigma band."""
        v = self._vwap_float
        if v is None:
            return None
        return Decimal(str(round(v - self._sigma_float, 6)))

    @property
    def upper_2(self) -> Decimal | None:
        """VWAP + 2 sigma band."""
        v = self._vwap_float
        if v is None:
            return None
        return Decimal(str(round(v + 2.0 * self._sigma_float, 6)))

    @property
    def lower_2(self) -> Decimal | None:
        """VWAP - 2 sigma band."""
        v = self._vwap_float
        if v is None:
            return None
        return Decimal(str(round(v - 2.0 * self._sigma_float, 6)))

    @property
    def is_active(self) -> bool:
        """``True`` once the session has started and at least one bar has been seen."""
        return self._cum_vol > 0.0

    def __repr__(self) -> str:
        return f"SessionVWAP(vwap={self.vwap}, sigma={self.sigma})"
