"""Tests for src.ingestion.websocket.AlpacaBarStream."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.ingestion.websocket import _FEED_MAP, AlpacaBarStream
from src.models import Bar, TimeFrame

# ── Helpers ─────────────────────────────────────────────────────────────────


def _make_alpaca_bar(
    *,
    symbol: str = "SPY",
    timestamp: datetime | None = None,
    open_: float = 480.0,
    high: float = 481.5,
    low: float = 479.8,
    close: float = 481.0,
    volume: int = 100_000,
    vwap: float | None = 480.55,
) -> SimpleNamespace:
    """Create a mock Alpaca bar object with the expected attributes."""
    return SimpleNamespace(
        symbol=symbol,
        timestamp=timestamp or datetime(2024, 1, 15, 14, 35, tzinfo=UTC),
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
        vwap=vwap,
    )


@pytest.fixture()
def queue() -> asyncio.Queue[Bar]:
    return asyncio.Queue()


@pytest.fixture()
def stream(queue: asyncio.Queue[Bar]) -> AlpacaBarStream:
    return AlpacaBarStream(symbols=["SPY", "QQQ"], queue=queue, timeframe=TimeFrame.ONE_MIN)


# ── Constructor ─────────────────────────────────────────────────────────────


class TestConstructor:
    def test_stores_symbols(self, stream: AlpacaBarStream) -> None:
        assert stream._symbols == ["SPY", "QQQ"]

    def test_stores_timeframe(self, stream: AlpacaBarStream) -> None:
        assert stream._timeframe == TimeFrame.ONE_MIN

    def test_stores_queue(self, stream: AlpacaBarStream, queue: asyncio.Queue[Bar]) -> None:
        assert stream._queue is queue

    def test_client_initially_none(self, stream: AlpacaBarStream) -> None:
        assert stream._client is None

    def test_default_timeframe(self, queue: asyncio.Queue[Bar]) -> None:
        s = AlpacaBarStream(symbols=["AAPL"], queue=queue)
        assert s._timeframe == TimeFrame.ONE_MIN


# ── _on_bar ─────────────────────────────────────────────────────────────────


class TestOnBar:
    @pytest.mark.asyncio()
    async def test_converts_alpaca_bar_and_queues(
        self, stream: AlpacaBarStream, queue: asyncio.Queue[Bar]
    ) -> None:
        alpaca_bar = _make_alpaca_bar()
        await stream._on_bar(alpaca_bar)

        assert queue.qsize() == 1
        bar = queue.get_nowait()
        assert isinstance(bar, Bar)
        assert bar.symbol == "SPY"
        assert bar.close == Decimal("481.0")
        assert bar.vwap == Decimal("480.55")
        assert bar.timeframe == TimeFrame.ONE_MIN
        assert bar.timestamp.tzinfo is not None

    @pytest.mark.asyncio()
    async def test_dict_bar_logs_warning_and_skips(
        self, stream: AlpacaBarStream, queue: asyncio.Queue[Bar]
    ) -> None:
        dict_bar = {"symbol": "SPY", "close": 481.0}
        await stream._on_bar(dict_bar)
        assert queue.qsize() == 0

    @pytest.mark.asyncio()
    async def test_tz_naive_timestamp_gets_utc(
        self, stream: AlpacaBarStream, queue: asyncio.Queue[Bar]
    ) -> None:
        naive_ts = datetime(2024, 1, 15, 14, 35)
        alpaca_bar = _make_alpaca_bar(timestamp=naive_ts)
        await stream._on_bar(alpaca_bar)

        bar = queue.get_nowait()
        assert bar.timestamp.tzinfo is UTC

    @pytest.mark.asyncio()
    async def test_tz_aware_timestamp_preserved(
        self, stream: AlpacaBarStream, queue: asyncio.Queue[Bar]
    ) -> None:
        aware_ts = datetime(2024, 1, 15, 14, 35, tzinfo=UTC)
        alpaca_bar = _make_alpaca_bar(timestamp=aware_ts)
        await stream._on_bar(alpaca_bar)

        bar = queue.get_nowait()
        assert bar.timestamp == aware_ts

    @pytest.mark.asyncio()
    async def test_conversion_error_logs_and_does_not_crash(
        self, stream: AlpacaBarStream, queue: asyncio.Queue[Bar]
    ) -> None:
        """A bar with invalid data (e.g. negative price) should not crash."""
        bad_bar = _make_alpaca_bar(close=-1.0, low=-2.0, open_=-0.5, high=-0.1, vwap=-1.0)
        # Should not raise
        await stream._on_bar(bad_bar)
        assert queue.qsize() == 0

    @pytest.mark.asyncio()
    async def test_vwap_none_falls_back_to_close(
        self, stream: AlpacaBarStream, queue: asyncio.Queue[Bar]
    ) -> None:
        alpaca_bar = _make_alpaca_bar(vwap=None, close=482.0)
        await stream._on_bar(alpaca_bar)

        bar = queue.get_nowait()
        assert bar.vwap == Decimal("482.0")

    @pytest.mark.asyncio()
    async def test_volume_converted_to_int(
        self, stream: AlpacaBarStream, queue: asyncio.Queue[Bar]
    ) -> None:
        alpaca_bar = _make_alpaca_bar(volume=55555)
        await stream._on_bar(alpaca_bar)

        bar = queue.get_nowait()
        assert bar.volume == 55555
        assert isinstance(bar.volume, int)

    @pytest.mark.asyncio()
    async def test_prices_converted_to_decimal(
        self, stream: AlpacaBarStream, queue: asyncio.Queue[Bar]
    ) -> None:
        alpaca_bar = _make_alpaca_bar(open_=100.25, high=101.5, low=99.75, close=100.5)
        await stream._on_bar(alpaca_bar)

        bar = queue.get_nowait()
        assert isinstance(bar.open, Decimal)
        assert isinstance(bar.high, Decimal)
        assert isinstance(bar.low, Decimal)
        assert isinstance(bar.close, Decimal)


# ── stop ────────────────────────────────────────────────────────────────────


class TestStop:
    @pytest.mark.asyncio()
    async def test_calls_client_close(self, stream: AlpacaBarStream) -> None:
        mock_client = MagicMock()
        mock_client.close = AsyncMock()
        stream._client = mock_client

        await stream.stop()
        mock_client.close.assert_awaited_once()

    @pytest.mark.asyncio()
    async def test_handles_close_error_gracefully(self, stream: AlpacaBarStream) -> None:
        mock_client = MagicMock()
        mock_client.close = AsyncMock(side_effect=RuntimeError("ws already closed"))
        stream._client = mock_client

        # Should not raise
        await stream.stop()

    @pytest.mark.asyncio()
    async def test_no_client_does_not_crash(self, stream: AlpacaBarStream) -> None:
        assert stream._client is None
        # Should not raise
        await stream.stop()


# ── _FEED_MAP ───────────────────────────────────────────────────────────────


class TestFeedMap:
    def test_has_exactly_two_entries(self) -> None:
        assert len(_FEED_MAP) == 2

    def test_iex_maps_to_data_feed_iex(self) -> None:
        from alpaca.data.enums import DataFeed

        assert _FEED_MAP["iex"] is DataFeed.IEX

    def test_sip_maps_to_data_feed_sip(self) -> None:
        from alpaca.data.enums import DataFeed

        assert _FEED_MAP["sip"] is DataFeed.SIP

    def test_keys_are_lowercase(self) -> None:
        for key in _FEED_MAP:
            assert key == key.lower()
