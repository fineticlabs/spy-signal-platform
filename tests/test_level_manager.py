"""Tests for LevelManager orchestration (src/levels/__init__.py)."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import MagicMock

from src.levels import LevelManager
from src.models import Bar, LevelSnapshot, TimeFrame


def _make_bar(
    hour: int = 14,
    minute: int = 35,
    close: Decimal = Decimal("480.00"),
    high: Decimal = Decimal("481.00"),
    low: Decimal = Decimal("479.00"),
    volume: int = 100_000,
    day: int = 15,
) -> Bar:
    return Bar(
        symbol="SPY",
        timeframe=TimeFrame.ONE_MIN,
        timestamp=datetime(2024, 1, day, hour, minute, tzinfo=UTC),
        open=Decimal("480.00"),
        high=high,
        low=low,
        close=close,
        volume=volume,
        vwap=Decimal("480.00"),
    )


class TestLevelManagerInit:
    def test_no_db(self) -> None:
        lm = LevelManager()
        assert lm._pdl is None

    def test_with_db(self) -> None:
        mock_db = MagicMock()
        mock_db.query_bars.return_value = []
        lm = LevelManager(db=mock_db, symbol="TSLA")
        assert lm._pdl is not None

    def test_repr(self) -> None:
        lm = LevelManager(symbol="AAPL")
        assert "AAPL" in repr(lm)


class TestLevelManagerUpdate:
    def test_update_propagates_to_all_trackers(self) -> None:
        lm = LevelManager()
        bar = _make_bar()
        lm.update(bar)
        levels = lm.get_levels()
        assert isinstance(levels, LevelSnapshot)
        # Day tracker should have updated
        assert levels.high_of_day is not None
        assert levels.low_of_day is not None

    def test_update_loads_pdl_on_new_date(self) -> None:
        mock_db = MagicMock()
        mock_db.query_bars.return_value = []
        lm = LevelManager(db=mock_db, symbol="SPY")
        bar = _make_bar()
        lm.update(bar)
        # PreviousDayLevels.load should have been called
        assert lm._pdl is not None
        assert lm._last_date is not None

    def test_update_pdl_load_error_caught(self) -> None:
        mock_db = MagicMock()
        mock_db.query_bars.side_effect = RuntimeError("DB down")
        lm = LevelManager(db=mock_db, symbol="SPY")
        bar = _make_bar()
        # Should not crash
        lm.update(bar)

    def test_get_levels_returns_snapshot(self) -> None:
        lm = LevelManager()
        levels = lm.get_levels()
        assert isinstance(levels, LevelSnapshot)
        # Before any bars, everything should be None
        assert levels.orb_high is None
        assert levels.vwap is None
        assert levels.high_of_day is None

    def test_premarket_bar_tracked(self) -> None:
        lm = LevelManager()
        # 8:00 AM ET = 13:00 UTC in January (EST)
        bar = _make_bar(hour=13, minute=0, high=Decimal("482.00"), low=Decimal("478.00"))
        lm.update(bar)
        levels = lm.get_levels()
        assert levels.premarket_high == Decimal("482.00")
        assert levels.premarket_low == Decimal("478.00")

    def test_orb_bars_tracked(self) -> None:
        lm = LevelManager()
        # Feed 5 bars at 9:30-9:34 ET = 14:30-14:34 UTC in January
        for minute in range(30, 35):
            bar = _make_bar(hour=14, minute=minute)
            lm.update(bar)
        # Then a bar at 9:35 ET
        bar = _make_bar(hour=14, minute=35)
        lm.update(bar)
        levels = lm.get_levels()
        assert levels.orb_complete is True

    def test_pdl_none_without_db(self) -> None:
        lm = LevelManager()
        levels = lm.get_levels()
        assert levels.prev_day_high is None
        assert levels.prev_day_low is None
        assert levels.prev_day_close is None
