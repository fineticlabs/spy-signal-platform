"""Extended ingestion tests — websocket start/stop, historical fetch edge cases."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.ingestion.websocket import _FEED_MAP, AlpacaBarStream

if TYPE_CHECKING:
    from src.models import Bar


class TestAlpacaBarStreamStart:
    @pytest.mark.asyncio
    async def test_start_connects_and_subscribes(self) -> None:
        queue: asyncio.Queue[Bar] = asyncio.Queue()
        stream = AlpacaBarStream(symbols=["SPY", "QQQ"], queue=queue)

        mock_client = MagicMock()
        mock_client._run_forever = AsyncMock(side_effect=asyncio.CancelledError)
        mock_client.subscribe_bars = MagicMock()

        mock_cfg = MagicMock(api_key="test", secret_key="test", feed="iex")  # noqa: S106
        with (
            patch("src.ingestion.websocket.get_alpaca_settings", return_value=mock_cfg),
            patch("src.ingestion.websocket.StockDataStream", return_value=mock_client),
        ):
            with pytest.raises(asyncio.CancelledError):
                await stream.start()

            mock_client.subscribe_bars.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_generic_exception_propagated(self) -> None:
        queue: asyncio.Queue[Bar] = asyncio.Queue()
        stream = AlpacaBarStream(symbols=["SPY"], queue=queue)

        mock_client = MagicMock()
        mock_client._run_forever = AsyncMock(side_effect=RuntimeError("connection lost"))
        mock_client.subscribe_bars = MagicMock()

        mock_cfg = MagicMock(api_key="test", secret_key="test", feed="iex")  # noqa: S106
        with (
            patch("src.ingestion.websocket.get_alpaca_settings", return_value=mock_cfg),
            patch("src.ingestion.websocket.StockDataStream", return_value=mock_client),
            pytest.raises(RuntimeError, match="connection lost"),
        ):
            await stream.start()


class TestFeedMapCompleteness:
    def test_all_feeds_have_valid_enum_values(self) -> None:
        from alpaca.data.enums import DataFeed

        for key, val in _FEED_MAP.items():
            assert isinstance(val, DataFeed)
            assert isinstance(key, str)
