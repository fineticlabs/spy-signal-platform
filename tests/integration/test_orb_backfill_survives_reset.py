"""Test that backfilled ORB data survives the scheduler's daily_reset.

Reproduces the 2026-03-20 bug where:
  1. _backfill_orb_bars sets orb_complete=True for all tickers
  2. Scheduler daily_reset fires and wipes the ORB state
  3. orb_formation_check sees 0/26 and sends false "No ORB" Telegram alert
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from src.levels import LevelManager
from src.levels.opening_range import OpeningRangeTracker
from src.main import _scheduler
from src.models import Bar, TimeFrame

_ET = ZoneInfo("America/New_York")


def _bar(minute: int, high: float = 482.0, low: float = 478.0) -> Bar:
    """Bar at 9:{30+minute} ET on 2024-01-16 (Tuesday)."""
    return Bar(
        symbol="SPY",
        timeframe=TimeFrame.ONE_MIN,
        timestamp=datetime(2024, 1, 16, 14, 30 + minute, tzinfo=UTC),
        open=Decimal("480.00"),
        high=Decimal(str(high)),
        low=Decimal(str(low)),
        close=Decimal("480.50"),
        volume=100_000,
        vwap=Decimal("480.25"),
    )


class TestOrbSurvivesDailyReset:
    """Prevents bug: daily_reset wipes backfilled ORB data."""

    def test_backfilled_orb_not_wiped_by_scheduler_reset(self) -> None:
        """The core bug: ORB complete for today should not be reset."""
        tracker = OpeningRangeTracker()

        # Simulate backfill: feed 5 ORB bars + 1 post-ORB bar
        for m in range(5):
            tracker.update(_bar(m))
        tracker.update(_bar(5))  # post-ORB bar triggers completion

        assert tracker.is_complete is True
        # 14:30 UTC = 9:30 ET on Jan 16 → session_date is Jan 16
        today = tracker._session_date
        assert today.isoformat() == "2024-01-16"

        # Simulate what the scheduler's daily_reset does (FIXED version):
        # Skip reset if already complete for today
        if not tracker.is_complete or tracker._session_date != today:
            tracker.__init__()

        # ORB must survive
        assert tracker.is_complete is True
        assert tracker.orb_high == Decimal("482.0")
        assert tracker.orb_low == Decimal("478.0")

    def test_orb_reset_when_date_changes(self) -> None:
        """ORB should reset when a new trading day arrives."""
        tracker = OpeningRangeTracker()

        # Day 1: complete ORB
        for m in range(5):
            tracker.update(_bar(m))
        tracker.update(_bar(5))
        assert tracker.is_complete is True
        day1_date = tracker._session_date

        # Simulate scheduler reset for a DIFFERENT date (day 2)
        day2 = day1_date + timedelta(days=1)
        if not tracker.is_complete or tracker._session_date != day2:
            tracker.__init__()

        # ORB should be wiped for the new day
        assert tracker.is_complete is False

    def test_orb_reset_when_not_complete(self) -> None:
        """Incomplete ORB should be reset by daily_reset."""
        tracker = OpeningRangeTracker()

        # Feed only 3 bars — ORB not complete
        for m in range(3):
            tracker.update(_bar(m))
        assert tracker.is_complete is False

        today = tracker._session_date

        # Scheduler reset: not complete → should reset
        if not tracker.is_complete or tracker._session_date != today:
            tracker.__init__()

        assert tracker.is_complete is False
        assert tracker.orb_high is None

    def test_live_bars_dont_re_reset_backfilled_orb(self) -> None:
        """When backfill set _session_date, live bars with same date don't reset."""
        tracker = OpeningRangeTracker()

        # Backfill bars
        for m in range(5):
            tracker.update(_bar(m))
        tracker.update(_bar(5))
        assert tracker.is_complete is True
        saved_high = tracker.orb_high

        # Live bar arrives for the same date (minute 10 = 9:40 ET)
        live_bar = _bar(10, high=490.0, low=470.0)
        tracker.update(live_bar)

        # ORB should NOT change — same date, already complete
        assert tracker.is_complete is True
        assert tracker.orb_high == saved_high  # not affected by live bar

    @pytest.mark.asyncio
    async def test_scheduler_preserves_backfilled_orb(self) -> None:
        """Full scheduler integration: backfilled ORB survives daily_reset cycle."""
        # Set up pipeline with backfilled ORB
        levels = LevelManager(db=None, symbol="SPY")
        for m in range(5):
            levels.update(_bar(m))
        levels.update(_bar(5))
        assert levels._orb.is_complete is True

        pipeline = MagicMock()
        pipeline.levels = levels
        pipelines = {"SPY": pipeline}
        cooldown = MagicMock()
        dispatcher = AsyncMock()
        db = MagicMock()
        db.conn = MagicMock()

        # Simulate scheduler at 9:31 ET (triggers daily_reset)
        mock_now = datetime(2024, 1, 16, 9, 31, tzinfo=_ET)

        sleep_count = 0

        async def _fake_sleep(_s: float) -> None:
            nonlocal sleep_count
            sleep_count += 1
            if sleep_count >= 2:
                raise asyncio.CancelledError

        with (
            patch("src.main.asyncio.sleep", side_effect=_fake_sleep),
            patch("src.main.datetime") as mock_dt,
        ):
            mock_dt.now.return_value = mock_now
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            with suppress(asyncio.CancelledError):
                await _scheduler(pipelines, cooldown, dispatcher, db)

        # ORB must still be complete after daily_reset
        assert levels._orb.is_complete is True
        assert levels._orb.orb_high == Decimal("482.0")

    @pytest.mark.asyncio
    async def test_no_false_orb_alert_after_backfill(self) -> None:
        """9:50 ET check should NOT alert when ORBs were backfilled."""
        levels = LevelManager(db=None, symbol="SPY")
        for m in range(5):
            levels.update(_bar(m))
        levels.update(_bar(5))

        pipeline = MagicMock()
        pipeline.levels = levels
        pipelines = {"SPY": pipeline}
        cooldown = MagicMock()
        dispatcher = AsyncMock()
        db = MagicMock()
        db.conn = MagicMock()

        # Simulate 9:51 ET — after daily_reset AND orb_check time
        mock_now = datetime(2024, 1, 16, 9, 51, tzinfo=_ET)
        sleep_count = 0

        async def _fake_sleep(_s: float) -> None:
            nonlocal sleep_count
            sleep_count += 1
            if sleep_count >= 2:
                raise asyncio.CancelledError

        with (
            patch("src.main.asyncio.sleep", side_effect=_fake_sleep),
            patch("src.main.datetime") as mock_dt,
        ):
            mock_dt.now.return_value = mock_now
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            with suppress(asyncio.CancelledError):
                await _scheduler(pipelines, cooldown, dispatcher, db)

        # No "No ORB ranges formed" risk warning
        for call in dispatcher.dispatch_risk_warning.call_args_list:
            assert "No ORB Ranges" not in str(call), f"False positive ORB alert fired: {call}"
