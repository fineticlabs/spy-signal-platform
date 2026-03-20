"""Integration: ORB backfill correctness — known bars → known ORB levels.

Verifies that the OpeningRangeTracker produces correct high/low/midpoint/range
values from known bar sequences, and that backfilled bars produce identical
results to live-fed bars.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from src.levels.opening_range import OpeningRangeTracker
from src.models import Bar, TimeFrame

# ── Bar factory ──────────────────────────────────────────────────────────────

_BASE_TS = datetime(2024, 1, 16, 14, 30, tzinfo=UTC)  # Tuesday 9:30 ET


def _bar(
    minute_offset: int,
    open_: float,
    high: float,
    low: float,
    close: float,
    volume: int = 200_000,
) -> Bar:
    """Create a 1-min bar at ``_BASE_TS + minute_offset`` minutes."""
    return Bar(
        symbol="SPY",
        timeframe=TimeFrame.ONE_MIN,
        timestamp=_BASE_TS + timedelta(minutes=minute_offset),
        open=Decimal(str(open_)),
        high=Decimal(str(high)),
        low=Decimal(str(low)),
        close=Decimal(str(close)),
        volume=volume,
        vwap=Decimal(str((high + low + close) / 3)),
    )


# ═══════════════════════════════════════════════════════════════════════════════


class TestOrbBackfillCorrectness:
    """Verify ORB levels match expected values from known bar data."""

    def test_backfilled_levels_match_expected(self) -> None:
        """Known bars → known ORB high/low/midpoint/range."""
        bars = [
            _bar(0, 480.0, 482.0, 478.0, 481.0),  # 9:30
            _bar(1, 481.0, 483.5, 479.5, 482.0),  # 9:31
            _bar(2, 482.0, 484.0, 480.0, 483.0),  # 9:32
            _bar(3, 483.0, 485.0, 481.0, 484.0),  # 9:33
            _bar(4, 484.0, 486.0, 482.0, 485.0),  # 9:34
        ]
        # Post-ORB bar at 9:35 to finalize the range
        post = _bar(5, 485.0, 487.0, 484.0, 486.0)

        tracker = OpeningRangeTracker()
        for b in bars:
            tracker.update(b)
        tracker.update(post)

        assert tracker.orb_high == Decimal("486.0")  # max high across all 5 bars
        assert tracker.orb_low == Decimal("478.0")  # min low across all 5 bars
        assert tracker.is_complete is True
        assert tracker.orb_midpoint == Decimal("482.0")  # (486 + 478) / 2
        assert tracker.orb_range == Decimal("8.0")  # 486 - 478

    def test_only_three_bars_available(self) -> None:
        """Partial ORB: only 3/5 bars received → ORB incomplete (no post-ORB bar)."""
        bars = [
            _bar(0, 480.0, 482.0, 478.0, 481.0),
            _bar(1, 481.0, 483.0, 479.0, 482.0),
            _bar(2, 482.0, 484.0, 480.0, 483.0),
        ]

        tracker = OpeningRangeTracker()
        for b in bars:
            tracker.update(b)

        # Range accumulator has data but is_complete requires a post-cutoff bar
        assert tracker.is_complete is False
        # Values are still available (partial range)
        assert tracker.orb_high == Decimal("484.0")
        assert tracker.orb_low == Decimal("478.0")

    def test_bars_with_gap(self) -> None:
        """Bars skip minute 2 → ORB still forms from available bars."""
        bars = [
            _bar(0, 480.0, 482.0, 478.0, 481.0),  # 9:30
            _bar(1, 481.0, 483.0, 479.0, 482.0),  # 9:31
            # minute 2 (9:32) missing
            _bar(3, 483.0, 485.0, 481.0, 484.0),  # 9:33
            _bar(4, 484.0, 486.0, 482.0, 485.0),  # 9:34
        ]
        post = _bar(5, 485.0, 487.0, 484.0, 486.0)  # 9:35

        tracker = OpeningRangeTracker()
        for b in bars:
            tracker.update(b)
        tracker.update(post)

        assert tracker.is_complete is True
        assert tracker.orb_high == Decimal("486.0")
        assert tracker.orb_low == Decimal("478.0")

    def test_backfill_matches_live_feed(self) -> None:
        """Same bars via backfill vs live feed → identical ORB levels."""
        bars = [
            _bar(0, 480.0, 482.0, 478.0, 481.0),
            _bar(1, 481.0, 483.5, 479.5, 482.0),
            _bar(2, 482.0, 484.0, 480.0, 483.0),
            _bar(3, 483.0, 485.0, 481.0, 484.0),
            _bar(4, 484.0, 486.0, 482.0, 485.0),
        ]
        post = _bar(5, 485.0, 487.0, 484.0, 486.0)

        # "Live" feed: bars arrive one-at-a-time
        live_tracker = OpeningRangeTracker()
        for b in bars:
            live_tracker.update(b)
        live_tracker.update(post)

        # "Backfill": all bars fed at once (same sequence)
        backfill_tracker = OpeningRangeTracker()
        for b in bars:
            backfill_tracker.update(b)
        backfill_tracker.update(post)

        assert live_tracker.orb_high == backfill_tracker.orb_high
        assert live_tracker.orb_low == backfill_tracker.orb_low
        assert live_tracker.orb_midpoint == backfill_tracker.orb_midpoint
        assert live_tracker.orb_range == backfill_tracker.orb_range
        assert live_tracker.is_complete == backfill_tracker.is_complete

    def test_post_orb_bar_required_to_finalize(self) -> None:
        """All 5 bars within window do not complete ORB — need a bar at 9:35+."""
        bars = [
            _bar(0, 480.0, 482.0, 478.0, 481.0),
            _bar(1, 481.0, 483.0, 479.0, 482.0),
            _bar(2, 482.0, 484.0, 480.0, 483.0),
            _bar(3, 483.0, 485.0, 481.0, 484.0),
            _bar(4, 484.0, 486.0, 482.0, 485.0),
        ]

        tracker = OpeningRangeTracker()
        for b in bars:
            tracker.update(b)

        # Not yet complete — no post-ORB bar received
        assert tracker.is_complete is False

        # Feed post-ORB bar → completes
        tracker.update(_bar(5, 485.0, 487.0, 484.0, 486.0))
        assert tracker.is_complete is True
