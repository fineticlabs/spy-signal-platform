"""US equity market trading calendar utilities.

Resolves the most recent trading day by skipping weekends and US market
holidays.  Uses a static set of observed NYSE holiday dates for the
current and surrounding years — no external dependencies required.
"""

from __future__ import annotations

from datetime import date, timedelta


def _us_market_holidays(year: int) -> set[date]:
    """Return NYSE observed holidays for *year*.

    Covers: New Year's Day, MLK Day, Presidents' Day, Good Friday,
    Memorial Day, Juneteenth, Independence Day, Labor Day,
    Thanksgiving, Christmas.
    """
    holidays: set[date] = set()

    # New Year's Day (Jan 1, observed Fri/Mon if weekend)
    holidays.add(_observed(date(year, 1, 1)))

    # MLK Day — 3rd Monday in January
    holidays.add(_nth_weekday(year, 1, 0, 3))

    # Presidents' Day — 3rd Monday in February
    holidays.add(_nth_weekday(year, 2, 0, 3))

    # Good Friday — 2 days before Easter Sunday
    holidays.add(_easter(year) - timedelta(days=2))

    # Memorial Day — last Monday in May
    holidays.add(_last_weekday(year, 5, 0))

    # Juneteenth (Jun 19, observed)
    holidays.add(_observed(date(year, 6, 19)))

    # Independence Day (Jul 4, observed)
    holidays.add(_observed(date(year, 7, 4)))

    # Labor Day — 1st Monday in September
    holidays.add(_nth_weekday(year, 9, 0, 1))

    # Thanksgiving — 4th Thursday in November
    holidays.add(_nth_weekday(year, 11, 3, 4))

    # Christmas (Dec 25, observed)
    holidays.add(_observed(date(year, 12, 25)))

    return holidays


def _observed(d: date) -> date:
    """Shift Saturday → Friday, Sunday → Monday for observed holidays."""
    if d.weekday() == 5:  # Saturday
        return d - timedelta(days=1)
    if d.weekday() == 6:  # Sunday
        return d + timedelta(days=1)
    return d


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """Return the *n*-th occurrence of *weekday* (0=Mon) in *month*."""
    first = date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + timedelta(days=offset + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    """Return the last occurrence of *weekday* (0=Mon) in *month*."""
    if month == 12:
        last_day = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        last_day = date(year, month + 1, 1) - timedelta(days=1)
    offset = (last_day.weekday() - weekday) % 7
    return last_day - timedelta(days=offset)


def _easter(year: int) -> date:
    """Compute Easter Sunday using the Anonymous Gregorian algorithm."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l_ = (32 + 2 * e + 2 * i - h - k) % 7  # — algorithm variable
    m = (a + 11 * h + 22 * l_) // 451
    month, day = divmod(h + l_ - 7 * m + 114, 31)
    return date(year, month, day + 1)


def is_trading_day(d: date) -> bool:
    """Return ``True`` if *d* is a US equity trading day (not weekend/holiday)."""
    if d.weekday() >= 5:
        return False
    return d not in _us_market_holidays(d.year)


def last_trading_day(session_date: date) -> date:
    """Return the most recent trading day strictly before *session_date*.

    Walks backwards from ``session_date - 1`` skipping weekends and
    US market holidays.  Will look back at most 10 days (handles
    extended holiday weekends).
    """
    candidate = session_date - timedelta(days=1)
    for _ in range(10):
        if is_trading_day(candidate):
            return candidate
        candidate -= timedelta(days=1)
    return candidate
