"""Tests for websocket watchdog: seconds_since_last_bar and _backfill_orb_bars."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from src.ingestion.websocket import AlpacaBarStream
from src.models import Bar, TimeFrame

_ET = ZoneInfo("America/New_York")


def _bar(symbol: str = "SPY") -> Bar:
    return Bar(
        symbol=symbol,
        timeframe=TimeFrame.ONE_MIN,
        timestamp=datetime(2024, 1, 16, 14, 35, tzinfo=UTC),
        open=Decimal("480.00"),
        high=Decimal("481.00"),
        low=Decimal("479.00"),
        close=Decimal("480.50"),
        volume=100_000,
        vwap=Decimal("480.25"),
    )


class TestSecondsSinceLastBar:
    """Prevents bug: websocket silently stops receiving bars with no detection."""

    def test_none_before_any_bar(self) -> None:
        queue: asyncio.Queue[Bar] = asyncio.Queue()
        stream = AlpacaBarStream(symbols=["SPY"], queue=queue)
        assert stream.seconds_since_last_bar is None

    @pytest.mark.asyncio
    async def test_updates_after_bar_received(self) -> None:
        queue: asyncio.Queue[Bar] = asyncio.Queue()
        stream = AlpacaBarStream(symbols=["SPY"], queue=queue)

        alpaca_bar = MagicMock()
        alpaca_bar.symbol = "SPY"
        alpaca_bar.timestamp = datetime(2024, 1, 16, 14, 35, tzinfo=UTC)
        alpaca_bar.open = 480.0
        alpaca_bar.high = 481.0
        alpaca_bar.low = 479.0
        alpaca_bar.close = 480.5
        alpaca_bar.volume = 100_000
        alpaca_bar.vwap = 480.25

        await stream._on_bar(alpaca_bar)

        assert stream.seconds_since_last_bar is not None
        assert stream.seconds_since_last_bar >= 0
        assert stream.seconds_since_last_bar < 5  # just happened

    @pytest.mark.asyncio
    async def test_grows_with_time(self) -> None:
        queue: asyncio.Queue[Bar] = asyncio.Queue()
        stream = AlpacaBarStream(symbols=["SPY"], queue=queue)
        # Manually set last_bar_time to 60s ago
        stream._last_bar_time = datetime.now(UTC) - timedelta(seconds=60)
        assert stream.seconds_since_last_bar is not None
        assert stream.seconds_since_last_bar >= 59


class TestBackfillOrbBars:
    """Prevents bug: late start misses 9:30-9:34 ORB window, zero signals all day."""

    @pytest.mark.asyncio
    async def test_skips_before_935_et(self) -> None:
        """No backfill needed if scanner starts before ORB window closes."""
        from src.main import _backfill_orb_bars

        pipelines = {"SPY": MagicMock()}
        # Mock time to be 9:30 ET (before ORB closes)
        mock_now = datetime(2024, 1, 16, 9, 30, tzinfo=_ET)
        with patch("src.main.datetime") as mock_dt:
            mock_dt.now.return_value = mock_now
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            await _backfill_orb_bars(pipelines)

        # levels.update should NOT have been called (no backfill)
        pipelines["SPY"].levels.update.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_after_market_close(self) -> None:
        """No backfill after 4:00 PM ET."""
        from src.main import _backfill_orb_bars

        pipelines = {"SPY": MagicMock()}
        mock_now = datetime(2024, 1, 16, 17, 0, tzinfo=_ET)
        with patch("src.main.datetime") as mock_dt:
            mock_dt.now.return_value = mock_now
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            await _backfill_orb_bars(pipelines)

        pipelines["SPY"].levels.update.assert_not_called()

    @pytest.mark.asyncio
    async def test_attempts_backfill_after_935(self) -> None:
        """After 9:35 ET, should attempt REST fetch (we mock the API)."""
        from src.main import _backfill_orb_bars

        mock_pipeline = MagicMock()
        mock_pipeline.levels._orb.is_complete = True
        pipelines = {"SPY": mock_pipeline}

        mock_now = datetime(2024, 1, 16, 9, 40, tzinfo=_ET)

        mock_bar = MagicMock()
        mock_bar.open = 480.0
        mock_bar.high = 481.0
        mock_bar.low = 479.0
        mock_bar.close = 480.5
        mock_bar.volume = 100_000
        mock_bar.vwap = 480.25
        mock_bar.timestamp = datetime(2024, 1, 16, 14, 31, tzinfo=UTC)

        mock_response = MagicMock()
        mock_response.data = {"SPY": [mock_bar]}

        mock_client = MagicMock()
        mock_client.get_stock_bars.return_value = mock_response

        with (
            patch("src.main.datetime") as mock_dt,
            patch("src.main.get_alpaca_settings") as mock_settings,
            patch("alpaca.data.historical.StockHistoricalDataClient", return_value=mock_client),
        ):
            mock_dt.now.return_value = mock_now
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            mock_settings.return_value = MagicMock(
                api_key="test",
                secret_key="test",  # noqa: S106
                feed="iex",
            )
            await _backfill_orb_bars(pipelines)

        # levels.update should have been called with the backfilled bar
        mock_pipeline.levels.update.assert_called()
