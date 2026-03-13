"""Economic calendar filter — block ORB trades on high-impact news days.

High-impact US economic events (FOMC, NFP, CPI, PPI) that occur before
11:00 AM ET make the opening range unreliable.  On these days, ORB trades
are blocked entirely.

For backtesting: uses hardcoded dates from 2020-2026.
For live trading: will be replaced with Finnhub API integration later.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

# ── Event time assumption ────────────────────────────────────────────────────
# All events listed here are released before 11:00 AM ET:
#   - FOMC decisions: 2:00 PM ET (but markets reprice starting at open)
#     NOTE: We block FOMC days because the *anticipation* of the decision
#     distorts the opening range, even though the release is at 2 PM.
#   - NFP: 8:30 AM ET
#   - CPI: 8:30 AM ET
#   - PPI: 8:30 AM ET

# ── FOMC rate decision dates ────────────────────────────────────────────────
# Source: Federal Reserve meeting calendar (announcement day only).
# 8 scheduled meetings per year.

_FOMC_DATES: set[date] = {
    # 2020
    date(2020, 1, 29),
    date(2020, 3, 3),
    date(2020, 3, 15),  # emergency cut
    date(2020, 4, 29),
    date(2020, 6, 10),
    date(2020, 7, 29),
    date(2020, 9, 16),
    date(2020, 11, 5),
    date(2020, 12, 16),
    # 2021
    date(2021, 1, 27),
    date(2021, 3, 17),
    date(2021, 4, 28),
    date(2021, 6, 16),
    date(2021, 7, 28),
    date(2021, 9, 22),
    date(2021, 11, 3),
    date(2021, 12, 15),
    # 2022
    date(2022, 1, 26),
    date(2022, 3, 16),
    date(2022, 5, 4),
    date(2022, 6, 15),
    date(2022, 7, 27),
    date(2022, 9, 21),
    date(2022, 11, 2),
    date(2022, 12, 14),
    # 2023
    date(2023, 2, 1),
    date(2023, 3, 22),
    date(2023, 5, 3),
    date(2023, 6, 14),
    date(2023, 7, 26),
    date(2023, 9, 20),
    date(2023, 11, 1),
    date(2023, 12, 13),
    # 2024
    date(2024, 1, 31),
    date(2024, 3, 20),
    date(2024, 5, 1),
    date(2024, 6, 12),
    date(2024, 7, 31),
    date(2024, 9, 18),
    date(2024, 11, 7),
    date(2024, 12, 18),
    # 2025
    date(2025, 1, 29),
    date(2025, 3, 19),
    date(2025, 5, 7),
    date(2025, 6, 18),
    date(2025, 7, 30),
    date(2025, 9, 17),
    date(2025, 10, 29),
    date(2025, 12, 17),
    # 2026
    date(2026, 1, 28),
    date(2026, 3, 18),
    date(2026, 4, 29),
    date(2026, 6, 17),
    date(2026, 7, 29),
    date(2026, 9, 16),
    date(2026, 11, 4),
    date(2026, 12, 16),
}

# ── CPI release dates ───────────────────────────────────────────────────────
# Bureau of Labor Statistics CPI release schedule (8:30 AM ET, mid-month).

_CPI_DATES: set[date] = {
    # 2020
    date(2020, 1, 14),
    date(2020, 2, 13),
    date(2020, 3, 11),
    date(2020, 4, 10),
    date(2020, 5, 12),
    date(2020, 6, 10),
    date(2020, 7, 14),
    date(2020, 8, 12),
    date(2020, 9, 11),
    date(2020, 10, 13),
    date(2020, 11, 12),
    date(2020, 12, 10),
    # 2021
    date(2021, 1, 13),
    date(2021, 2, 10),
    date(2021, 3, 10),
    date(2021, 4, 13),
    date(2021, 5, 12),
    date(2021, 6, 10),
    date(2021, 7, 13),
    date(2021, 8, 11),
    date(2021, 9, 14),
    date(2021, 10, 13),
    date(2021, 11, 10),
    date(2021, 12, 10),
    # 2022
    date(2022, 1, 12),
    date(2022, 2, 10),
    date(2022, 3, 10),
    date(2022, 4, 12),
    date(2022, 5, 11),
    date(2022, 6, 10),
    date(2022, 7, 13),
    date(2022, 8, 10),
    date(2022, 9, 13),
    date(2022, 10, 13),
    date(2022, 11, 10),
    date(2022, 12, 13),
    # 2023
    date(2023, 1, 12),
    date(2023, 2, 14),
    date(2023, 3, 14),
    date(2023, 4, 12),
    date(2023, 5, 10),
    date(2023, 6, 13),
    date(2023, 7, 12),
    date(2023, 8, 10),
    date(2023, 9, 13),
    date(2023, 10, 12),
    date(2023, 11, 14),
    date(2023, 12, 12),
    # 2024
    date(2024, 1, 11),
    date(2024, 2, 13),
    date(2024, 3, 12),
    date(2024, 4, 10),
    date(2024, 5, 15),
    date(2024, 6, 12),
    date(2024, 7, 11),
    date(2024, 8, 14),
    date(2024, 9, 11),
    date(2024, 10, 10),
    date(2024, 11, 13),
    date(2024, 12, 11),
    # 2025
    date(2025, 1, 15),
    date(2025, 2, 12),
    date(2025, 3, 12),
    date(2025, 4, 10),
    date(2025, 5, 13),
    date(2025, 6, 11),
    date(2025, 7, 15),
    date(2025, 8, 12),
    date(2025, 9, 10),
    date(2025, 10, 14),
    date(2025, 11, 12),
    date(2025, 12, 10),
    # 2026
    date(2026, 1, 14),
    date(2026, 2, 11),
    date(2026, 3, 11),
    date(2026, 4, 14),
    date(2026, 5, 12),
    date(2026, 6, 10),
    date(2026, 7, 14),
    date(2026, 8, 12),
    date(2026, 9, 16),
    date(2026, 10, 14),
    date(2026, 11, 12),
    date(2026, 12, 10),
}

# ── PPI release dates ───────────────────────────────────────────────────────
# Bureau of Labor Statistics PPI release schedule (8:30 AM ET).

_PPI_DATES: set[date] = {
    # 2020
    date(2020, 1, 15),
    date(2020, 2, 13),
    date(2020, 3, 12),
    date(2020, 4, 9),
    date(2020, 5, 12),
    date(2020, 6, 11),
    date(2020, 7, 14),
    date(2020, 8, 11),
    date(2020, 9, 10),
    date(2020, 10, 14),
    date(2020, 11, 12),
    date(2020, 12, 11),
    # 2021
    date(2021, 1, 14),
    date(2021, 2, 17),
    date(2021, 3, 12),
    date(2021, 4, 9),
    date(2021, 5, 13),
    date(2021, 6, 15),
    date(2021, 7, 14),
    date(2021, 8, 12),
    date(2021, 9, 10),
    date(2021, 10, 14),
    date(2021, 11, 9),
    date(2021, 12, 14),
    # 2022
    date(2022, 1, 13),
    date(2022, 2, 15),
    date(2022, 3, 15),
    date(2022, 4, 13),
    date(2022, 5, 12),
    date(2022, 6, 14),
    date(2022, 7, 14),
    date(2022, 8, 11),
    date(2022, 9, 14),
    date(2022, 10, 12),
    date(2022, 11, 15),
    date(2022, 12, 9),
    # 2023
    date(2023, 1, 18),
    date(2023, 2, 16),
    date(2023, 3, 15),
    date(2023, 4, 13),
    date(2023, 5, 11),
    date(2023, 6, 14),
    date(2023, 7, 13),
    date(2023, 8, 11),
    date(2023, 9, 14),
    date(2023, 10, 11),
    date(2023, 11, 15),
    date(2023, 12, 13),
    # 2024
    date(2024, 1, 12),
    date(2024, 2, 16),
    date(2024, 3, 14),
    date(2024, 4, 11),
    date(2024, 5, 14),
    date(2024, 6, 13),
    date(2024, 7, 12),
    date(2024, 8, 13),
    date(2024, 9, 12),
    date(2024, 10, 11),
    date(2024, 11, 14),
    date(2024, 12, 12),
    # 2025
    date(2025, 1, 14),
    date(2025, 2, 13),
    date(2025, 3, 13),
    date(2025, 4, 11),
    date(2025, 5, 15),
    date(2025, 6, 12),
    date(2025, 7, 15),
    date(2025, 8, 14),
    date(2025, 9, 11),
    date(2025, 10, 15),
    date(2025, 11, 13),
    date(2025, 12, 11),
    # 2026
    date(2026, 1, 15),
    date(2026, 2, 12),
    date(2026, 3, 12),
    date(2026, 4, 9),
    date(2026, 5, 13),
    date(2026, 6, 11),
    date(2026, 7, 14),
    date(2026, 8, 13),
    date(2026, 9, 15),
    date(2026, 10, 15),
    date(2026, 11, 13),
    date(2026, 12, 11),
}


def _nfp_dates(start_year: int = 2020, end_year: int = 2026) -> set[date]:
    """Generate NFP release dates (first Friday of each month).

    NFP (Non-Farm Payrolls) is released on the first Friday of each month
    at 8:30 AM ET by the Bureau of Labor Statistics.

    Args:
        start_year: First year to generate (inclusive).
        end_year:   Last year to generate (inclusive).

    Returns:
        Set of NFP release dates.
    """
    dates: set[date] = set()
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            first_day = date(year, month, 1)
            # Monday=0, Friday=4
            days_until_friday = (4 - first_day.weekday()) % 7
            first_friday = first_day + timedelta(days=days_until_friday)
            dates.add(first_friday)
    return dates


_NFP_DATES: set[date] = _nfp_dates()

# Combined set of all high-impact event dates
ALL_EVENT_DATES: set[date] = _FOMC_DATES | _NFP_DATES | _CPI_DATES | _PPI_DATES


def is_high_impact_day(d: date) -> bool:
    """Check whether a given date has a high-impact economic event scheduled.

    Args:
        d: Calendar date to check.

    Returns:
        ``True`` if FOMC, NFP, CPI, or PPI is scheduled on this date.
    """
    return d in ALL_EVENT_DATES


def get_event_types(d: date) -> list[str]:
    """Return the list of event types scheduled on a given date.

    Args:
        d: Calendar date to check.

    Returns:
        List of event type strings (e.g. ``["FOMC", "CPI"]``).
        Empty list if no events.
    """
    events: list[str] = []
    if d in _FOMC_DATES:
        events.append("FOMC")
    if d in _NFP_DATES:
        events.append("NFP")
    if d in _CPI_DATES:
        events.append("CPI")
    if d in _PPI_DATES:
        events.append("PPI")
    return events


def compute_econ_blocked_array(
    index: pd.DatetimeIndex,
) -> np.ndarray[Any, np.dtype[np.bool_]]:
    """Pre-compute a boolean array marking bars on high-impact event days.

    Used by the backtest engine to block ORB trades on FOMC/NFP/CPI/PPI days.
    Each bar on an event day is marked ``True`` (blocked).

    Args:
        index: UTC DatetimeIndex aligned with 1-min bar data.

    Returns:
        Boolean numpy array of length ``len(index)``.  ``True`` means the
        bar falls on a high-impact event day and ORB trades should be blocked.
    """
    from zoneinfo import ZoneInfo

    et_tz = ZoneInfo("America/New_York")
    n = len(index)
    blocked = np.zeros(n, dtype=bool)

    idx = index.tz_localize("UTC") if index.tzinfo is None else index
    et_index = idx.tz_convert(et_tz)

    for i, ts in enumerate(et_index):
        if ts.date() in ALL_EVENT_DATES:
            blocked[i] = True

    return blocked
