"""Tests for data_loader.py — load_bars and WalkForwardWindow."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from src.backtest.data_loader import (
    WalkForwardWindow,
    load_bars,
    resample,
)
from src.models import Bar, TimeFrame


def _make_bar_obj(ts: datetime) -> Bar:
    return Bar(
        symbol="SPY",
        timeframe=TimeFrame.ONE_MIN,
        timestamp=ts,
        open=Decimal("480.00"),
        high=Decimal("481.00"),
        low=Decimal("479.00"),
        close=Decimal("480.50"),
        volume=100_000,
        vwap=Decimal("480.25"),
    )


class TestLoadBars:
    def test_empty_query_returns_empty_df(self) -> None:
        mock_db = MagicMock()
        mock_db.query_bars.return_value = []
        df = load_bars(mock_db, symbol="SPY")
        assert df.empty
        assert "close" in df.columns

    def test_loads_bars_to_dataframe(self) -> None:
        start = datetime(2024, 1, 15, 14, 30, tzinfo=UTC)
        bars = [_make_bar_obj(start + timedelta(minutes=i)) for i in range(5)]
        mock_db = MagicMock()
        mock_db.query_bars.return_value = bars
        df = load_bars(mock_db, symbol="SPY")
        assert len(df) == 5
        assert df.index.tzinfo is not None  # UTC-aware

    def test_columns_are_float(self) -> None:
        start = datetime(2024, 1, 15, 14, 30, tzinfo=UTC)
        bars = [_make_bar_obj(start)]
        mock_db = MagicMock()
        mock_db.query_bars.return_value = bars
        df = load_bars(mock_db)
        assert df["close"].dtype == float
        assert df["volume"].dtype == int

    def test_sorted_by_timestamp(self) -> None:
        start = datetime(2024, 1, 15, 14, 30, tzinfo=UTC)
        bars = [_make_bar_obj(start + timedelta(minutes=i)) for i in range(3)]
        mock_db = MagicMock()
        mock_db.query_bars.return_value = list(reversed(bars))
        df = load_bars(mock_db)
        assert df.index.is_monotonic_increasing


class TestResample:
    def test_one_min_passthrough(self) -> None:
        from tests.conftest import make_1min_df

        df = make_1min_df(n_days=1)
        result = resample(df, TimeFrame.ONE_MIN)
        assert len(result) == len(df)

    def test_unsupported_timeframe_raises(self) -> None:
        from tests.conftest import make_1min_df

        df = make_1min_df(n_days=1)
        with pytest.raises(ValueError, match="Unsupported"):
            resample(df, TimeFrame.DAILY)


class TestWalkForwardWindow:
    def test_frozen(self) -> None:
        from datetime import date

        w = WalkForwardWindow(
            in_sample_start=date(2024, 1, 1),
            in_sample_end=date(2024, 3, 1),
            out_of_sample_start=date(2024, 3, 2),
            out_of_sample_end=date(2024, 3, 22),
        )
        with pytest.raises(AttributeError):
            w.in_sample_start = date(2025, 1, 1)  # type: ignore[misc]

    def test_str_representation(self) -> None:
        from datetime import date

        w = WalkForwardWindow(
            in_sample_start=date(2024, 1, 1),
            in_sample_end=date(2024, 3, 1),
            out_of_sample_start=date(2024, 3, 2),
            out_of_sample_end=date(2024, 3, 22),
        )
        s = str(w)
        assert "IS=" in s
        assert "OOS=" in s
