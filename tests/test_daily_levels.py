"""Tests for src.levels.daily_levels — PreviousDayLevels & PremarketLevels."""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from src.levels.daily_levels import PremarketLevels, PreviousDayLevels
from tests.conftest import make_bar

if TYPE_CHECKING:
    from src.models import Bar


# ── helpers ──────────────────────────────────────────────────────────────────


def _make_premarket_bar(
    hour_et: int,
    minute_et: int,
    high: Decimal,
    low: Decimal,
    close: Decimal | None = None,
    dt: date | None = None,
) -> Bar:
    """Create a Bar at a specific ET time, converted to UTC for January (EST = UTC-5)."""
    d = dt or date(2024, 1, 15)
    # EST offset: ET + 5h = UTC
    utc_ts = datetime(d.year, d.month, d.day, hour_et + 5, minute_et, tzinfo=UTC)
    return make_bar(
        timestamp=utc_ts,
        high=high,
        low=low,
        close=close or high,
        open_=low,
        vwap=low,
    )


def _mock_db_with_bars(bars: list[Bar]) -> MagicMock:
    """Create a mock BarDatabase whose query_bars returns *bars*."""
    db = MagicMock()
    db.query_bars.return_value = bars
    return db


# ── PreviousDayLevels ────────────────────────────────────────────────────────


class TestPreviousDayLevels:
    """Tests for PreviousDayLevels."""

    def test_load_finds_pdh_pdl_pdc(self) -> None:
        bars = [
            make_bar(high=Decimal("482.00"), low=Decimal("478.00"), close=Decimal("479.50")),
            make_bar(high=Decimal("484.00"), low=Decimal("477.00"), close=Decimal("483.00")),
            make_bar(high=Decimal("481.00"), low=Decimal("479.00"), close=Decimal("480.00")),
        ]
        db = _mock_db_with_bars(bars)
        pdl = PreviousDayLevels(db, symbol="SPY")
        pdl.load(date(2024, 1, 16))

        assert pdl.high == Decimal("484.00")
        assert pdl.low == Decimal("477.00")
        assert pdl.close == Decimal("480.00")  # last bar's close

    def test_caches_same_date_skips_db(self) -> None:
        bars = [make_bar(high=Decimal("482.00"), low=Decimal("478.00"), close=Decimal("480.00"))]
        db = _mock_db_with_bars(bars)
        pdl = PreviousDayLevels(db, symbol="SPY")

        pdl.load(date(2024, 1, 16))
        pdl.load(date(2024, 1, 16))  # second call — should be cached

        assert db.query_bars.call_count == 1

    def test_no_data_properties_none(self) -> None:
        db = _mock_db_with_bars([])
        pdl = PreviousDayLevels(db, symbol="SPY")
        pdl.load(date(2024, 1, 16))

        assert pdl.high is None
        assert pdl.low is None
        assert pdl.close is None

    def test_db_query_error_properties_none(self) -> None:
        db = MagicMock()
        db.query_bars.side_effect = RuntimeError("connection lost")
        pdl = PreviousDayLevels(db, symbol="SPY")
        pdl.load(date(2024, 1, 16))

        assert pdl.high is None
        assert pdl.low is None
        assert pdl.close is None

    def test_high_is_max_of_all_bars(self) -> None:
        bars = [
            make_bar(high=Decimal("490.00"), low=Decimal("480.00"), close=Decimal("485.00")),
            make_bar(high=Decimal("495.00"), low=Decimal("482.00"), close=Decimal("493.00")),
            make_bar(high=Decimal("488.00"), low=Decimal("481.00"), close=Decimal("486.00")),
        ]
        db = _mock_db_with_bars(bars)
        pdl = PreviousDayLevels(db, symbol="SPY")
        pdl.load(date(2024, 1, 16))
        assert pdl.high == Decimal("495.00")

    def test_close_is_last_bar_close(self) -> None:
        bars = [
            make_bar(high=Decimal("490.00"), low=Decimal("480.00"), close=Decimal("485.00")),
            make_bar(high=Decimal("492.00"), low=Decimal("481.00"), close=Decimal("491.00")),
        ]
        db = _mock_db_with_bars(bars)
        pdl = PreviousDayLevels(db, symbol="SPY")
        pdl.load(date(2024, 1, 16))
        assert pdl.close == Decimal("491.00")

    def test_load_new_date_re_queries(self) -> None:
        bars = [make_bar(high=Decimal("482.00"), low=Decimal("478.00"), close=Decimal("480.00"))]
        db = _mock_db_with_bars(bars)
        pdl = PreviousDayLevels(db, symbol="SPY")

        pdl.load(date(2024, 1, 16))
        pdl.load(date(2024, 1, 17))

        assert db.query_bars.call_count == 2


# ── PremarketLevels ──────────────────────────────────────────────────────────


class TestPremarketLevels:
    """Tests for PremarketLevels."""

    def test_accumulates_premarket_bars(self) -> None:
        pm = PremarketLevels()
        pm.update(_make_premarket_bar(5, 0, high=Decimal("482.00"), low=Decimal("480.00")))
        pm.update(_make_premarket_bar(6, 30, high=Decimal("484.00"), low=Decimal("479.00")))
        assert pm.high == Decimal("484.00")
        assert pm.low == Decimal("479.00")

    def test_ignores_bars_outside_premarket_window(self) -> None:
        pm = PremarketLevels()
        # 3:59 ET — before premarket
        pm.update(_make_premarket_bar(3, 59, high=Decimal("500.00"), low=Decimal("470.00")))
        # 9:30 ET — session open, not premarket
        bar_930 = make_bar(
            timestamp=datetime(2024, 1, 15, 14, 30, tzinfo=UTC),
            high=Decimal("500.00"),
            low=Decimal("470.00"),
        )
        pm.update(bar_930)
        assert pm.high is None
        assert pm.low is None

    def test_resets_on_new_date(self) -> None:
        pm = PremarketLevels()
        pm.update(_make_premarket_bar(5, 0, high=Decimal("490.00"), low=Decimal("485.00")))
        assert pm.high == Decimal("490.00")

        # Next day bar
        pm.update(
            _make_premarket_bar(
                5, 0, high=Decimal("492.00"), low=Decimal("488.00"), dt=date(2024, 1, 16)
            )
        )
        # Old data should be gone; only new day data
        assert pm.high == Decimal("492.00")
        assert pm.low == Decimal("488.00")

    def test_high_is_max_low_is_min(self) -> None:
        pm = PremarketLevels()
        pm.update(_make_premarket_bar(4, 0, high=Decimal("481.00"), low=Decimal("479.00")))
        pm.update(_make_premarket_bar(5, 0, high=Decimal("483.00"), low=Decimal("480.00")))
        pm.update(_make_premarket_bar(8, 0, high=Decimal("482.00"), low=Decimal("477.00")))
        assert pm.high == Decimal("483.00")
        assert pm.low == Decimal("477.00")

    def test_none_when_no_premarket_data(self) -> None:
        pm = PremarketLevels()
        assert pm.high is None
        assert pm.low is None

    def test_finalize_logs_warning_when_no_data(self, caplog: pytest.LogCaptureFixture) -> None:
        pm = PremarketLevels()
        pm._session_date = date(2024, 1, 15)
        with patch("src.levels.daily_levels.logger") as mock_logger:
            pm.finalize()
            mock_logger.warning.assert_called_once()

    def test_finalize_only_warns_once(self) -> None:
        pm = PremarketLevels()
        pm._session_date = date(2024, 1, 15)
        with patch("src.levels.daily_levels.logger") as mock_logger:
            pm.finalize()
            pm.finalize()
            assert mock_logger.warning.call_count == 1

    def test_finalize_no_warning_when_data_exists(self) -> None:
        pm = PremarketLevels()
        pm.update(_make_premarket_bar(5, 0, high=Decimal("482.00"), low=Decimal("480.00")))
        with patch("src.levels.daily_levels.logger") as mock_logger:
            pm.finalize()
            mock_logger.warning.assert_not_called()
