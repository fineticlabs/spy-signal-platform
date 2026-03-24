"""Alpaca WebSocket bar stream with connection state tracking and retry logic.

Wraps the alpaca-py ``StockDataStream`` to add:

- Structured logging for auth, subscribe, and failure events
- ``connected`` and ``subscribed`` asyncio Events for startup confirmation
- Retry with exponential backoff (3 attempts, 10s base)
- ``on_failure`` callback for Telegram escalation after exhausting retries
- ``seconds_since_last_bar`` for watchdog monitoring
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import structlog
from alpaca.data.enums import DataFeed
from alpaca.data.live import StockDataStream

from src.config import get_alpaca_settings
from src.models import Bar, TimeFrame

_FEED_MAP: dict[str, DataFeed] = {
    "iex": DataFeed.IEX,
    "sip": DataFeed.SIP,
}

_MAX_RETRIES = 3
_RETRY_BACKOFF_BASE = 10  # seconds

if TYPE_CHECKING:
    from alpaca.data.models.bars import Bar as AlpacaBar

logger = structlog.get_logger(__name__)


class AlpacaBarStream:
    """Async WebSocket listener that streams 1-min bars for the given symbols.

    Received bars are converted to :class:`~src.models.Bar` and placed onto an
    :class:`asyncio.Queue` for downstream consumers.

    Connection lifecycle events are signalled via :attr:`connected` and
    :attr:`subscribed` asyncio Events, which the caller can ``await`` to
    confirm the websocket is live before proceeding.

    Args:
        symbols:    Ticker symbols to subscribe to.
        queue:      Destination queue for converted bars.
        timeframe:  Bar timeframe (default ``ONE_MIN``).
        on_failure: Optional async callback invoked after all retries are
                    exhausted.  Receives the last exception message as a string.
                    Use this to send a Telegram alert.
    """

    def __init__(
        self,
        symbols: list[str],
        queue: asyncio.Queue[Bar],
        timeframe: TimeFrame = TimeFrame.ONE_MIN,
        on_failure: Callable[[str], Awaitable[None]] | None = None,
    ) -> None:
        self._symbols = symbols
        self._queue = queue
        self._timeframe = timeframe
        self._on_failure = on_failure
        self._client: StockDataStream | None = None
        self._last_bar_time: datetime | None = None
        self._overflow_count: int = 0

        # Connection state events — await these from the caller
        self.connected: asyncio.Event = asyncio.Event()
        self.subscribed: asyncio.Event = asyncio.Event()

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

            # Queue overflow protection: if full, drop oldest bar to make room
            if self._queue.full():
                try:
                    self._queue.get_nowait()
                    self._overflow_count += 1
                    logger.warning(
                        "bar_queue_overflow",
                        symbol=bar.symbol,
                        overflow_count=self._overflow_count,
                    )
                except asyncio.QueueEmpty:
                    pass

            await self._queue.put(converted)
            self._last_bar_time = datetime.now(UTC)
            logger.debug(
                "bar_received",
                symbol=bar.symbol,
                timestamp=ts.isoformat(),
                close=str(converted.close),
            )
        except Exception as exc:
            logger.error("bar_conversion_failed", error=str(exc), raw=repr(bar))

    @property
    def seconds_since_last_bar(self) -> float | None:
        """Seconds since the last bar was received, or None if no bars yet."""
        if self._last_bar_time is None:
            return None
        return (datetime.now(UTC) - self._last_bar_time).total_seconds()

    async def start(self) -> None:
        """Connect to Alpaca WebSocket with retry logic.

        Retries up to ``_MAX_RETRIES`` times with exponential backoff.
        After each successful connect+auth+subscribe, signals
        :attr:`connected` and :attr:`subscribed` events.
        On reconnect, re-subscribes to all symbols and sends Telegram alerts.

        If all retries are exhausted, calls ``on_failure`` (if set) and raises
        the last exception.
        """
        settings = get_alpaca_settings()
        data_feed = _FEED_MAP.get(settings.feed, DataFeed.IEX)
        last_error: str = "unknown"
        is_reconnect = False

        for attempt in range(1, _MAX_RETRIES + 1):
            self._client = StockDataStream(
                api_key=settings.api_key,
                secret_key=settings.secret_key,
                feed=data_feed,
            )
            # Re-subscribe on every connection (including reconnects)
            self._client.subscribe_bars(self._on_bar, *self._symbols)

            # Wrap _start_ws to detect auth+subscribe success.
            # Default args bind the current value (avoids B023 closure-in-loop).
            original_start_ws = self._client._start_ws
            _is_reconnect = is_reconnect  # capture for closure

            async def _instrumented_start_ws(
                _orig: Any = original_start_ws,
                _reconn: bool = _is_reconnect,
            ) -> None:
                await _orig()
                logger.info("websocket_authenticated", symbols=self._symbols)
                self.connected.set()
                if _reconn:
                    logger.info(
                        "websocket_resubscribed",
                        symbols=self._symbols,
                        count=len(self._symbols),
                    )
                    if self._on_failure is not None:
                        try:
                            await self._on_failure(
                                f"Websocket reconnected, {len(self._symbols)} symbols resubscribed"
                            )
                        except Exception as _exc:
                            logger.warning("reconnect_notify_failed", error=str(_exc))

            self._client._start_ws = _instrumented_start_ws  # type: ignore[assignment]

            original_subscribe = self._client._send_subscribe_msg

            async def _instrumented_subscribe(
                _orig: Any = original_subscribe,
            ) -> None:
                await _orig()
                logger.info(
                    "websocket_subscribed",
                    symbols=self._symbols,
                    count=len(self._symbols),
                )
                self.subscribed.set()

            self._client._send_subscribe_msg = _instrumented_subscribe  # type: ignore[assignment]

            logger.info(
                "websocket_connecting",
                symbols=self._symbols,
                feed=settings.feed,
                attempt=attempt,
                is_reconnect=is_reconnect,
            )

            try:
                await self._client._run_forever()
            except asyncio.CancelledError:
                logger.info("websocket_cancelled", symbols=self._symbols)
                raise
            except Exception as exc:
                last_error = str(exc)
                logger.error(
                    "websocket_disconnected",
                    error=last_error,
                    attempt=attempt,
                    max_retries=_MAX_RETRIES,
                )
                # Alert on disconnect
                if self._on_failure is not None:
                    try:
                        disconnect_time = datetime.now(UTC).strftime("%H:%M:%S UTC")
                        await self._on_failure(
                            f"Websocket disconnected at {disconnect_time}: {last_error}"
                        )
                    except Exception as _exc:
                        logger.warning("disconnect_notify_failed", error=str(_exc))
                if attempt < _MAX_RETRIES:
                    is_reconnect = True
                    wait = _RETRY_BACKOFF_BASE * attempt
                    logger.info("websocket_retry_wait", seconds=wait, next_attempt=attempt + 1)
                    await asyncio.sleep(wait)
                    continue
                # All retries exhausted
                logger.error(
                    "websocket_retries_exhausted",
                    error=last_error,
                    total_attempts=_MAX_RETRIES,
                )
                if self._on_failure is not None:
                    try:
                        await self._on_failure(
                            f"Websocket failed after {_MAX_RETRIES} attempts: {last_error}"
                        )
                    except Exception as _exc:
                        logger.warning("failure_notify_failed", error=str(_exc))
                raise

    async def stop(self) -> None:
        """Gracefully close the WebSocket connection."""
        if self._client is not None:
            try:
                await self._client.close()
            except Exception as exc:
                logger.warning("websocket_close_error", error=str(exc))
        logger.info("websocket_stopped", symbols=self._symbols)
