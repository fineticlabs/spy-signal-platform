"""Tests for startup reliability: websocket connection tracking, retry logic,
readiness notifications, and failure escalation.

These tests prevent the class of bugs where the scanner starts but silently
fails to connect, leaving the user unaware until the trading day is lost.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.ingestion.websocket import _MAX_RETRIES, _RETRY_BACKOFF_BASE, AlpacaBarStream

if TYPE_CHECKING:
    from src.models import Bar

# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_alpaca_bar() -> MagicMock:
    bar = MagicMock()
    bar.symbol = "SPY"
    bar.timestamp = datetime(2024, 1, 16, 14, 35, tzinfo=UTC)
    bar.open = 480.0
    bar.high = 481.0
    bar.low = 479.0
    bar.close = 480.5
    bar.volume = 100_000
    bar.vwap = 480.25
    return bar


# ══════════════════════════════════════════════════════════════════════════════
# Connection state events
# ══════════════════════════════════════════════════════════════════════════════


class TestConnectionStateEvents:
    """Prevents: scanner starts but websocket silently dead (2026-03-20 bug)."""

    def test_connected_event_starts_unset(self) -> None:
        queue: asyncio.Queue[Bar] = asyncio.Queue()
        stream = AlpacaBarStream(symbols=["SPY"], queue=queue)
        assert not stream.connected.is_set()

    def test_subscribed_event_starts_unset(self) -> None:
        queue: asyncio.Queue[Bar] = asyncio.Queue()
        stream = AlpacaBarStream(symbols=["SPY"], queue=queue)
        assert not stream.subscribed.is_set()

    @pytest.mark.asyncio
    async def test_connected_set_after_successful_auth(self) -> None:
        """After _start_ws succeeds, connected event must be set."""
        queue: asyncio.Queue[Bar] = asyncio.Queue()
        stream = AlpacaBarStream(symbols=["SPY"], queue=queue)

        mock_client = MagicMock()
        mock_client._start_ws = AsyncMock()
        mock_client._send_subscribe_msg = AsyncMock()
        mock_client._run_forever = AsyncMock(side_effect=asyncio.CancelledError)
        mock_client.subscribe_bars = MagicMock()
        mock_client.close = AsyncMock()

        mock_cfg = MagicMock(api_key="k", secret_key="s", feed="iex")  # noqa: S106
        with (
            patch("src.ingestion.websocket.get_alpaca_settings", return_value=mock_cfg),
            patch("src.ingestion.websocket.StockDataStream", return_value=mock_client),
        ):
            # _run_forever calls _start_ws internally via the instrumented wrapper.
            # We need to simulate _run_forever calling _start_ws then _send_subscribe_msg.
            async def _fake_run_forever() -> None:
                await mock_client._start_ws()
                await mock_client._send_subscribe_msg()
                raise asyncio.CancelledError

            mock_client._run_forever = _fake_run_forever

            with pytest.raises(asyncio.CancelledError):
                await stream.start()

        assert stream.connected.is_set()
        assert stream.subscribed.is_set()

    @pytest.mark.asyncio
    async def test_on_failure_called_after_retries_exhausted(self) -> None:
        """After MAX_RETRIES failures, on_failure callback fires."""
        queue: asyncio.Queue[Bar] = asyncio.Queue()
        failure_messages: list[str] = []

        async def capture_failure(msg: str) -> None:
            failure_messages.append(msg)

        stream = AlpacaBarStream(symbols=["SPY"], queue=queue, on_failure=capture_failure)

        mock_client = MagicMock()
        mock_client._run_forever = AsyncMock(side_effect=ConnectionError("Errno 49"))
        mock_client.subscribe_bars = MagicMock()
        mock_client.close = AsyncMock()

        mock_cfg = MagicMock(api_key="k", secret_key="s", feed="iex")  # noqa: S106
        with (
            patch("src.ingestion.websocket.get_alpaca_settings", return_value=mock_cfg),
            patch("src.ingestion.websocket.StockDataStream", return_value=mock_client),
            patch("src.ingestion.websocket.asyncio.sleep", new_callable=AsyncMock),
            pytest.raises(ConnectionError),
        ):
            await stream.start()

        # 3 disconnect alerts + 1 final exhaustion alert = 4 total
        assert len(failure_messages) == 4
        # Last message should be the exhaustion alert
        assert "3 attempts" in failure_messages[-1]
        # Earlier messages should be disconnect alerts
        assert "disconnected" in failure_messages[0].lower()

    @pytest.mark.asyncio
    async def test_retry_count_matches_max_retries(self) -> None:
        """start() retries exactly MAX_RETRIES times."""
        queue: asyncio.Queue[Bar] = asyncio.Queue()
        stream = AlpacaBarStream(symbols=["SPY"], queue=queue)

        call_count = 0
        mock_client = MagicMock()

        async def _count_and_fail() -> None:
            nonlocal call_count
            call_count += 1
            raise RuntimeError("fail")

        mock_client._run_forever = _count_and_fail
        mock_client.subscribe_bars = MagicMock()
        mock_client.close = AsyncMock()

        mock_cfg = MagicMock(api_key="k", secret_key="s", feed="iex")  # noqa: S106
        with (
            patch("src.ingestion.websocket.get_alpaca_settings", return_value=mock_cfg),
            patch("src.ingestion.websocket.StockDataStream", return_value=mock_client),
            patch("src.ingestion.websocket.asyncio.sleep", new_callable=AsyncMock),
            pytest.raises(RuntimeError),
        ):
            await stream.start()

        assert call_count == _MAX_RETRIES


class TestRetryBackoff:
    """Prevents: rapid reconnect loops that exhaust Alpaca connection limits."""

    @pytest.mark.asyncio
    async def test_backoff_increases_with_attempt(self) -> None:
        queue: asyncio.Queue[Bar] = asyncio.Queue()
        stream = AlpacaBarStream(symbols=["SPY"], queue=queue)

        sleep_calls: list[float] = []

        async def _capture_sleep(seconds: float) -> None:
            sleep_calls.append(seconds)

        mock_client = MagicMock()
        mock_client._run_forever = AsyncMock(side_effect=RuntimeError("fail"))
        mock_client.subscribe_bars = MagicMock()

        mock_cfg = MagicMock(api_key="k", secret_key="s", feed="iex")  # noqa: S106
        with (
            patch("src.ingestion.websocket.get_alpaca_settings", return_value=mock_cfg),
            patch("src.ingestion.websocket.StockDataStream", return_value=mock_client),
            patch("src.ingestion.websocket.asyncio.sleep", side_effect=_capture_sleep),
            pytest.raises(RuntimeError),
        ):
            await stream.start()

        # Retries 1 and 2 sleep; retry 3 fails and raises (no sleep after last)
        assert len(sleep_calls) == _MAX_RETRIES - 1
        assert sleep_calls[0] == _RETRY_BACKOFF_BASE * 1  # 10s
        assert sleep_calls[1] == _RETRY_BACKOFF_BASE * 2  # 20s


# ══════════════════════════════════════════════════════════════════════════════
# Readiness notifications
# ══════════════════════════════════════════════════════════════════════════════


class TestReadinessNotifications:
    """Prevents: user has no idea if scanner is alive at 6:18 AM."""

    @pytest.mark.asyncio
    async def test_notify_readiness_sends_telegram_on_connect(self) -> None:
        from src.main import _notify_readiness

        stream = MagicMock()
        stream.connected = asyncio.Event()
        stream.subscribed = asyncio.Event()
        stream.seconds_since_last_bar = None

        dispatcher = MagicMock()
        dispatcher.dispatch_status = AsyncMock(return_value=True)

        # Simulate connection happening immediately
        stream.connected.set()
        stream.subscribed.set()

        # Simulate first bar arriving after 1 check
        call_count = 0

        async def _fake_sleep(s: float) -> None:
            nonlocal call_count
            call_count += 1
            if call_count >= 1:
                stream.seconds_since_last_bar = 0.5

        with patch("src.main.asyncio.sleep", side_effect=_fake_sleep):
            await _notify_readiness(stream, dispatcher, 26)

        # Should have sent "Scanner ready" and "System GO" via status (not warning)
        assert dispatcher.dispatch_status.call_count == 2
        first_msg = dispatcher.dispatch_status.call_args_list[0][0][0]
        second_msg = dispatcher.dispatch_status.call_args_list[1][0][0]
        assert "Scanner Ready" in first_msg or "Scanner ready" in first_msg
        assert "26 symbols" in first_msg
        assert "System GO" in second_msg

    @pytest.mark.asyncio
    async def test_notify_readiness_alerts_on_timeout(self) -> None:
        from src.main import _notify_readiness

        stream = MagicMock()
        stream.connected = asyncio.Event()  # never set → timeout
        stream.subscribed = asyncio.Event()

        dispatcher = MagicMock()
        dispatcher.dispatch_risk_warning = AsyncMock(return_value=True)

        with patch("src.main._WS_CONNECT_TIMEOUT", 0.1):
            await _notify_readiness(stream, dispatcher, 26)

        # Should have sent failure alert
        dispatcher.dispatch_risk_warning.assert_called_once()
        msg = dispatcher.dispatch_risk_warning.call_args[0][0]
        assert "STARTUP FAILED" in msg


class TestOnFailureCallback:
    """Prevents: websocket dies silently with no Telegram notification."""

    def test_on_failure_param_stored(self) -> None:
        queue: asyncio.Queue[Bar] = asyncio.Queue()

        async def noop(msg: str) -> None:
            pass

        stream = AlpacaBarStream(symbols=["SPY"], queue=queue, on_failure=noop)
        assert stream._on_failure is noop

    def test_on_failure_default_is_none(self) -> None:
        queue: asyncio.Queue[Bar] = asyncio.Queue()
        stream = AlpacaBarStream(symbols=["SPY"], queue=queue)
        assert stream._on_failure is None


class TestConstants:
    """Verify retry constants are sensible."""

    def test_max_retries_is_three(self) -> None:
        assert _MAX_RETRIES == 3

    def test_backoff_base_is_ten_seconds(self) -> None:
        assert _RETRY_BACKOFF_BASE == 10
