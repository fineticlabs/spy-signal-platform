from __future__ import annotations

import asyncio
from datetime import UTC
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import structlog
from alpaca.data.live import StockDataStream

from src.config import get_alpaca_settings
from src.models import Bar, TimeFrame

if TYPE_CHECKING:
    from alpaca.data.models.bars import Bar as AlpacaBar

logger = structlog.get_logger(__name__)


class AlpacaBarStream:
    """Async WebSocket listener that streams 1-min bars for the given symbols.

    Received bars are converted to :class:`~src.models.Bar` and placed onto an
    :class:`asyncio.Queue` for downstream consumers.

    Usage::

        queue: asyncio.Queue[Bar] = asyncio.Queue()
        stream = AlpacaBarStream(symbols=["SPY"], queue=queue)
        await stream.start()          # runs until cancelled
    """

    def __init__(
        self,
        symbols: list[str],
        queue: asyncio.Queue[Bar],
        timeframe: TimeFrame = TimeFrame.ONE_MIN,
    ) -> None:
        self._symbols = symbols
        self._queue = queue
        self._timeframe = timeframe
        self._client: StockDataStream | None = None

    async def _on_bar(self, bar: AlpacaBar | dict[Any, Any]) -> None:
        """Callback invoked by the Alpaca SDK for every incoming bar."""
        try:
            if isinstance(bar, dict):
                logger.warning("bar_received_as_dict", data=bar)
                return

            ts = bar.timestamp
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=UTC)

            converted = Bar(
                symbol=bar.symbol,
                timeframe=self._timeframe,
                timestamp=ts,
                open=Decimal(str(bar.open)),
                high=Decimal(str(bar.high)),
                low=Decimal(str(bar.low)),
                close=Decimal(str(bar.close)),
                volume=int(bar.volume),
                vwap=Decimal(str(bar.vwap)) if bar.vwap is not None else Decimal(str(bar.close)),
            )
            await self._queue.put(converted)
            logger.debug(
                "bar_received",
                symbol=bar.symbol,
                timestamp=ts.isoformat(),
                close=str(converted.close),
            )
        except Exception as exc:
            logger.error("bar_conversion_failed", error=str(exc), raw=repr(bar))

    async def start(self) -> None:
        """Connect to Alpaca WebSocket and stream bars indefinitely.

        This coroutine blocks until the task is cancelled or the connection
        is permanently lost.  It should be run as an :mod:`asyncio` task.
        """
        settings = get_alpaca_settings()
        self._client = StockDataStream(
            api_key=settings.api_key,
            secret_key=settings.secret_key,
            feed=settings.feed,  # type: ignore[arg-type]
        )
        self._client.subscribe_bars(self._on_bar, *self._symbols)

        logger.info("websocket_connecting", symbols=self._symbols, feed=settings.feed)
        try:
            await self._client._run_forever()
        except asyncio.CancelledError:
            logger.info("websocket_cancelled", symbols=self._symbols)
            raise
        except Exception as exc:
            logger.error("websocket_error", error=str(exc))
            raise

    async def stop(self) -> None:
        """Gracefully close the WebSocket connection."""
        if self._client is not None:
            try:
                await self._client.close()
            except Exception as exc:
                logger.warning("websocket_close_error", error=str(exc))
        logger.info("websocket_stopped", symbols=self._symbols)
