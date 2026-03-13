"""Tests for key price level trackers."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

from src.levels.dynamic import DayTracker
from src.levels.opening_range import OpeningRangeTracker
from src.levels.vwap import SessionVWAP
from src.models import Bar, TimeFrame

_ET = ZoneInfo("America/New_York")

# ── helpers ──────────────────────────────────────────────────────────────────


def _bar(
    et_hour: int,
    et_minute: int,
    close: float,
    high: float | None = None,
    low: float | None = None,
    volume: int = 1_000_000,
    date_str: str = "2024-01-15",
) -> Bar:
    """Build a Bar whose timestamp corresponds to the given ET wall-clock time."""
    h = high if high is not None else close + 0.5
    lo = low if low is not None else close - 0.5
    naive_et = datetime.strptime(f"{date_str} {et_hour:02d}:{et_minute:02d}", "%Y-%m-%d %H:%M")
    ts_et = naive_et.replace(tzinfo=_ET)
    ts_utc = ts_et.astimezone(UTC)
    return Bar(
        symbol="SPY",
        timeframe=TimeFrame.ONE_MIN,
        timestamp=ts_utc,
        open=Decimal(str(close)),
        high=Decimal(str(h)),
        low=Decimal(str(lo)),
        close=Decimal(str(close)),
        volume=volume,
        vwap=Decimal(str(close)),
    )


def _session_bars(
    n: int,
    start_hour: int = 9,
    start_minute: int = 30,
    base_close: float = 480.0,
    date_str: str = "2024-01-15",
) -> list[Bar]:
    """Build *n* sequential 1-min bars starting at the given ET time."""
    bars = []
    for i in range(n):
        total_minutes = start_minute + i
        h = start_hour + total_minutes // 60
        m = total_minutes % 60
        bars.append(
            _bar(
                et_hour=h,
                et_minute=m,
                close=base_close + i * 0.10,
                date_str=date_str,
            )
        )
    return bars


# ── OpeningRangeTracker tests ─────────────────────────────────────────────────


class TestOpeningRangeTracker:
    def test_not_complete_before_9_35(self) -> None:
        """ORB must be incomplete before 9:35 ET."""
        tracker = OpeningRangeTracker()
        # Feed bars at 9:30, 9:31, 9:32, 9:33, 9:34 (5 bars inside window)
        for i in range(5):
            tracker.update(_bar(9, 30 + i, close=480.0 + i))
        assert not tracker.is_complete

    def test_complete_after_first_bar_past_935(self) -> None:
        """ORB becomes complete when a bar >= 9:35 ET arrives."""
        tracker = OpeningRangeTracker()
        for i in range(5):
            tracker.update(_bar(9, 30 + i, close=480.0 + i))
        # Bar at 9:35 closes the window
        tracker.update(_bar(9, 35, close=485.0))
        assert tracker.is_complete

    def test_orb_high_is_max_of_window_highs(self) -> None:
        """ORB high = maximum of all bar highs in the 9:30-9:34 window."""
        tracker = OpeningRangeTracker()
        highs = [481.0, 483.5, 482.0, 480.5, 484.0]
        for i, h in enumerate(highs):
            tracker.update(_bar(9, 30 + i, close=480.0, high=h, low=479.0))
        # Trigger completion
        tracker.update(_bar(9, 35, close=480.0))
        assert tracker.orb_high == Decimal(str(max(highs)))

    def test_orb_low_is_min_of_window_lows(self) -> None:
        tracker = OpeningRangeTracker()
        lows = [479.5, 478.0, 479.0, 477.5, 480.0]
        for i, lo in enumerate(lows):
            tracker.update(_bar(9, 30 + i, close=480.0, high=481.0, low=lo))
        tracker.update(_bar(9, 35, close=480.0))
        assert tracker.orb_low == Decimal(str(min(lows)))

    def test_orb_midpoint(self) -> None:
        tracker = OpeningRangeTracker()
        tracker.update(_bar(9, 30, close=480.0, high=484.0, low=476.0))
        tracker.update(_bar(9, 35, close=480.0))
        expected = (Decimal("484.0") + Decimal("476.0")) / 2
        assert tracker.orb_midpoint == expected

    def test_orb_range(self) -> None:
        tracker = OpeningRangeTracker()
        tracker.update(_bar(9, 30, close=480.0, high=484.0, low=476.0))
        tracker.update(_bar(9, 35, close=480.0))
        assert tracker.orb_range == Decimal("484.0") - Decimal("476.0")

    def test_premarket_bars_ignored(self) -> None:
        """Bars before 9:30 ET must not affect ORB levels."""
        tracker = OpeningRangeTracker()
        tracker.update(_bar(8, 0, close=470.0, high=999.0, low=1.0))
        tracker.update(_bar(9, 30, close=480.0, high=481.0, low=479.0))
        tracker.update(_bar(9, 35, close=480.0))
        assert tracker.orb_high == Decimal("481.0")  # explicit high from 9:30 bar only
        assert tracker.orb_low == Decimal("479.0")  # explicit low from 9:30 bar

    def test_daily_reset(self) -> None:
        """Tracker resets when a bar from a new date arrives."""
        tracker = OpeningRangeTracker()
        # Day 1
        for i in range(5):
            tracker.update(_bar(9, 30 + i, close=480.0 + i, date_str="2024-01-15"))
        tracker.update(_bar(9, 35, close=485.0, date_str="2024-01-15"))
        assert tracker.is_complete

        # Day 2 — first bar resets the tracker
        tracker.update(_bar(9, 30, close=490.0, date_str="2024-01-16"))
        assert not tracker.is_complete
        # ORB high should now only reflect the new day's bar
        assert tracker.orb_high == Decimal("490.5")

    def test_orb15_not_complete_at_935(self) -> None:
        tracker = OpeningRangeTracker()
        for i in range(5):
            tracker.update(_bar(9, 30 + i, close=480.0))
        tracker.update(_bar(9, 35, close=480.0))
        # 5-min ORB complete, 15-min still open
        assert tracker.is_complete
        assert not tracker.orb15_complete

    def test_orb15_complete_at_945(self) -> None:
        tracker = OpeningRangeTracker()
        for i in range(15):
            tracker.update(_bar(9, 30 + i, close=480.0 + i * 0.1))
        tracker.update(_bar(9, 45, close=481.5))
        assert tracker.orb15_complete


# ── SessionVWAP tests ─────────────────────────────────────────────────────────


class TestSessionVWAP:
    def test_not_active_before_any_bar(self) -> None:
        vwap = SessionVWAP()
        assert not vwap.is_active
        assert vwap.vwap is None

    def test_single_bar_vwap_equals_typical_price(self) -> None:
        vwap = SessionVWAP()
        bar = _bar(9, 30, close=480.0, high=481.0, low=479.0, volume=1_000_000)
        vwap.update(bar)
        tp = (481.0 + 479.0 + 480.0) / 3
        assert vwap.vwap is not None
        assert abs(float(vwap.vwap) - tp) < 1e-6

    def test_vwap_matches_manual_formula(self) -> None:
        """VWAP = sum(tp * vol) / sum(vol) over all session bars."""
        vwap = SessionVWAP()
        data = [
            (480.0, 481.0, 479.0, 1_000_000),
            (481.0, 482.0, 480.0, 1_200_000),
            (482.0, 483.0, 481.0, 800_000),
        ]
        for close, high, lo, vol in data:
            vwap.update(_bar(9, 30, close=close, high=high, low=lo, volume=vol))

        tp_list = [(h + lo + c) / 3 for c, h, lo, _ in data]
        vols = [v for _, _, _, v in data]
        expected = sum(t * v for t, v in zip(tp_list, vols, strict=False)) / sum(vols)

        assert vwap.vwap is not None
        assert abs(float(vwap.vwap) - expected) < 1e-5

    def test_premarket_bars_excluded(self) -> None:
        """Bars before 9:30 ET must not contribute to VWAP."""
        vwap = SessionVWAP()
        vwap.update(_bar(8, 0, close=470.0, volume=5_000_000))  # premarket
        assert not vwap.is_active

    def test_sigma_zero_when_all_prices_equal(self) -> None:
        """With constant typical price, variance and sigma should be zero."""
        vwap = SessionVWAP()
        for i in range(5):
            vwap.update(_bar(9, 30 + i, close=480.0, high=480.5, low=479.5, volume=1_000_000))
        assert float(vwap.sigma) < 1e-9

    def test_bands_above_and_below_vwap(self) -> None:
        """upper_1 > vwap > lower_1 and upper_2 > upper_1."""
        vwap = SessionVWAP()
        closes = [480.0, 481.5, 479.5, 482.0, 478.0, 481.0]
        for i, c in enumerate(closes):
            vwap.update(_bar(9, 30 + i, close=c, high=c + 1.0, low=c - 1.0, volume=1_000_000))

        assert vwap.vwap is not None
        assert vwap.upper_1 is not None
        assert vwap.lower_1 is not None
        assert vwap.upper_2 is not None
        assert vwap.lower_2 is not None
        assert vwap.upper_1 >= vwap.vwap
        assert vwap.lower_1 <= vwap.vwap
        assert vwap.upper_2 >= vwap.upper_1
        assert vwap.lower_2 <= vwap.lower_1

    def test_daily_reset(self) -> None:
        """VWAP resets on a new trading date."""
        vwap = SessionVWAP()
        vwap.update(_bar(9, 30, close=480.0, date_str="2024-01-15"))
        assert vwap.is_active

        vwap.update(_bar(9, 30, close=490.0, date_str="2024-01-16"))
        # After reset, only the new day's bar counts
        tp = (490.5 + 489.5 + 490.0) / 3
        assert vwap.vwap is not None
        assert abs(float(vwap.vwap) - tp) < 1e-5


# ── DayTracker tests ──────────────────────────────────────────────────────────


class TestDayTracker:
    def test_hod_and_lod_update_correctly(self) -> None:
        tracker = DayTracker()
        highs = [481.0, 483.5, 482.0, 484.0, 480.0]
        lows = [479.5, 478.0, 479.0, 477.5, 480.0]
        for i, (h, lo) in enumerate(zip(highs, lows, strict=False)):
            tracker.update(_bar(9, 30 + i, close=480.0, high=h, low=lo))

        assert tracker.high_of_day == Decimal(str(max(highs) + 0.0))
        assert tracker.low_of_day == Decimal(str(min(lows) + 0.0))

    def test_last_price_is_most_recent_close(self) -> None:
        tracker = DayTracker()
        bars = _session_bars(5)
        for b in bars:
            tracker.update(b)
        assert tracker.last_price == bars[-1].close

    def test_premarket_bars_do_not_affect_hod_lod(self) -> None:
        tracker = DayTracker()
        tracker.update(_bar(8, 0, close=450.0, high=999.0, low=1.0))  # premarket
        tracker.update(_bar(9, 30, close=480.0, high=481.0, low=479.0))
        assert tracker.high_of_day == Decimal("481.0")
        assert tracker.low_of_day == Decimal("479.0")

    def test_daily_reset_clears_hod_lod(self) -> None:
        tracker = DayTracker()
        tracker.update(_bar(9, 30, close=480.0, high=490.0, low=470.0, date_str="2024-01-15"))
        tracker.update(_bar(9, 30, close=495.0, date_str="2024-01-16"))
        # After reset only new day bar counts
        assert tracker.high_of_day == Decimal("495.5")
        assert tracker.low_of_day == Decimal("494.5")

    def test_none_before_any_bar(self) -> None:
        tracker = DayTracker()
        assert tracker.high_of_day is None
        assert tracker.low_of_day is None
        assert tracker.last_price is None
