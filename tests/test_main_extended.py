"""Extended tests for src/main.py — covers _bar_loop, _scheduler,
_process_bar regime update, _start_api, _install_signal_handlers, and run().
"""

from __future__ import annotations

import asyncio
import signal as _signal
from contextlib import suppress
from datetime import datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from src.main import (
    _bar_loop,
    _install_signal_handlers,
    _process_bar,
    _regime_first_populated,
    _regime_warmup_bar_counts,
    _scheduler,
    _send_eod_status,
    _start_api,
    _SymbolPipeline,
)
from src.models import IndicatorSnapshot, LevelSnapshot
from tests.conftest import make_bar

_ET = ZoneInfo("America/New_York")


# ── helpers ──────────────────────────────────────────────────────────────────


def _make_pipeline(
    symbol: str = "SPY",
    *,
    atr: Decimal | None = None,
    ema9: Decimal | None = None,
    ema20: Decimal | None = None,
    signal_return: object = None,
) -> _SymbolPipeline:
    """Build a mock _SymbolPipeline with configurable indicator values."""
    snapshot = IndicatorSnapshot(atr=atr, ema9=ema9, ema20=ema20)
    level_snapshot = LevelSnapshot()

    pipeline = MagicMock()
    pipeline.symbol = symbol
    pipeline.registry.update_all = MagicMock()
    pipeline.registry.get_snapshot.return_value = snapshot
    pipeline.levels.update = MagicMock()
    pipeline.levels.get_levels.return_value = level_snapshot
    pipeline.strategy.evaluate.return_value = signal_return
    pipeline.regime = MagicMock()
    return pipeline


def _make_risk() -> MagicMock:
    risk = MagicMock()
    risk.approve.return_value = MagicMock(approved=False, reason="test")
    return risk


def _make_dispatcher() -> AsyncMock:
    return AsyncMock()


def _make_db() -> MagicMock:
    return MagicMock()


# ═════════════════════════════════════════════════════════════════════════════
# _process_bar — regime update (lines 157-162)
# ═════════════════════════════════════════════════════════════════════════════


class TestProcessBarRegimeUpdate:
    """Cover lines 157-162: regime.update called when ATR is available."""

    @pytest.mark.asyncio
    async def test_regime_update_with_ema9_and_ema20(self) -> None:
        """When atr, ema9, and ema20 are all present, trending_up = ema9 > ema20."""
        pipeline = _make_pipeline(
            atr=Decimal("2.5"),
            ema9=Decimal("485.00"),
            ema20=Decimal("480.00"),
        )
        bar = make_bar(symbol="SPY")
        await _process_bar(bar, pipeline, _make_risk(), _make_dispatcher(), _make_db())

        pipeline.regime.update.assert_called_once()
        assert pipeline.regime.update.call_args.kwargs["trending_up"] is True

    @pytest.mark.asyncio
    async def test_regime_update_trending_down(self) -> None:
        """ema9 < ema20 → trending_up=False."""
        pipeline = _make_pipeline(
            atr=Decimal("2.5"),
            ema9=Decimal("475.00"),
            ema20=Decimal("480.00"),
        )
        bar = make_bar(symbol="SPY")
        await _process_bar(bar, pipeline, _make_risk(), _make_dispatcher(), _make_db())

        pipeline.regime.update.assert_called_once()
        assert pipeline.regime.update.call_args.kwargs["trending_up"] is False

    @pytest.mark.asyncio
    async def test_regime_update_ema9_none(self) -> None:
        """When ema9 is None, trending_up stays None."""
        pipeline = _make_pipeline(
            atr=Decimal("2.5"),
            ema9=None,
            ema20=Decimal("480.00"),
        )
        bar = make_bar(symbol="SPY")
        await _process_bar(bar, pipeline, _make_risk(), _make_dispatcher(), _make_db())

        pipeline.regime.update.assert_called_once()
        assert pipeline.regime.update.call_args.kwargs["trending_up"] is None

    @pytest.mark.asyncio
    async def test_regime_update_ema20_none(self) -> None:
        """When ema20 is None, trending_up stays None."""
        pipeline = _make_pipeline(
            atr=Decimal("2.5"),
            ema9=Decimal("485.00"),
            ema20=None,
        )
        bar = make_bar(symbol="SPY")
        await _process_bar(bar, pipeline, _make_risk(), _make_dispatcher(), _make_db())

        pipeline.regime.update.assert_called_once()
        assert pipeline.regime.update.call_args.kwargs["trending_up"] is None

    @pytest.mark.asyncio
    async def test_regime_update_both_emas_none(self) -> None:
        """When both emas are None, trending_up stays None."""
        pipeline = _make_pipeline(
            atr=Decimal("2.5"),
            ema9=None,
            ema20=None,
        )
        bar = make_bar(symbol="SPY")
        await _process_bar(bar, pipeline, _make_risk(), _make_dispatcher(), _make_db())

        pipeline.regime.update.assert_called_once()
        assert pipeline.regime.update.call_args.kwargs["trending_up"] is None

    @pytest.mark.asyncio
    async def test_no_regime_update_when_atr_none(self) -> None:
        """When ATR is None, regime.update is never called."""
        pipeline = _make_pipeline(
            atr=None,
            ema9=Decimal("485.00"),
            ema20=Decimal("480.00"),
        )
        bar = make_bar(symbol="SPY")
        await _process_bar(bar, pipeline, _make_risk(), _make_dispatcher(), _make_db())

        pipeline.regime.update.assert_not_called()


# ═════════════════════════════════════════════════════════════════════════════
# _bar_loop (lines 230-248)
# ═════════════════════════════════════════════════════════════════════════════


class TestBarLoop:
    """Cover _bar_loop: known symbol, unknown symbol, exception handling."""

    @pytest.mark.asyncio
    async def test_processes_known_symbol(self) -> None:
        """Bar with known symbol routes through the pipeline."""
        queue: asyncio.Queue = asyncio.Queue()
        bar = make_bar(symbol="SPY")
        await queue.put(bar)

        pipeline = _make_pipeline(symbol="SPY")
        pipelines = {"SPY": pipeline}

        task = asyncio.create_task(
            _bar_loop(queue, pipelines, _make_risk(), _make_dispatcher(), _make_db())
        )
        await queue.join()
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task

        pipeline.registry.update_all.assert_called_once_with(bar)

    @pytest.mark.asyncio
    async def test_unknown_symbol_logs_warning_and_continues(self) -> None:
        """Bar with unknown symbol calls task_done without crash."""
        queue: asyncio.Queue = asyncio.Queue()
        bar = make_bar(symbol="UNKNOWN")
        await queue.put(bar)

        pipelines: dict[str, _SymbolPipeline] = {"SPY": _make_pipeline(symbol="SPY")}

        task = asyncio.create_task(
            _bar_loop(queue, pipelines, _make_risk(), _make_dispatcher(), _make_db())
        )
        await queue.join()
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task

        # Pipeline for SPY should NOT have been invoked
        pipelines["SPY"].registry.update_all.assert_not_called()

    @pytest.mark.asyncio
    async def test_exception_in_process_bar_caught(self) -> None:
        """Exception inside _process_bar is caught; loop continues."""
        queue: asyncio.Queue = asyncio.Queue()
        bar = make_bar(symbol="SPY")
        await queue.put(bar)

        pipeline = _make_pipeline(symbol="SPY")
        # Make registry.update_all raise to simulate crash
        pipeline.registry.update_all.side_effect = RuntimeError("boom")
        pipelines = {"SPY": pipeline}

        task = asyncio.create_task(
            _bar_loop(queue, pipelines, _make_risk(), _make_dispatcher(), _make_db())
        )
        await queue.join()  # should not hang — error caught, task_done() called
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task

    @pytest.mark.asyncio
    async def test_multiple_bars_processed_sequentially(self) -> None:
        """Multiple bars on the queue are all processed."""
        queue: asyncio.Queue = asyncio.Queue()
        for _i in range(3):
            await queue.put(make_bar(symbol="SPY"))

        pipeline = _make_pipeline(symbol="SPY")
        pipelines = {"SPY": pipeline}

        task = asyncio.create_task(
            _bar_loop(queue, pipelines, _make_risk(), _make_dispatcher(), _make_db())
        )
        await queue.join()
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task

        assert pipeline.registry.update_all.call_count == 3

    @pytest.mark.asyncio
    async def test_bar_loop_with_executor(self) -> None:
        """Executor parameter is forwarded to _process_bar."""
        queue: asyncio.Queue = asyncio.Queue()
        bar = make_bar(symbol="SPY")
        await queue.put(bar)

        pipeline = _make_pipeline(symbol="SPY")
        pipelines = {"SPY": pipeline}
        executor = MagicMock()

        with patch("src.main._process_bar", new_callable=AsyncMock) as mock_pb:
            task = asyncio.create_task(
                _bar_loop(
                    queue,
                    pipelines,
                    _make_risk(),
                    _make_dispatcher(),
                    _make_db(),
                    executor=executor,
                )
            )
            await queue.join()
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

            mock_pb.assert_called_once()
            assert mock_pb.call_args.kwargs["executor"] is executor


# ═════════════════════════════════════════════════════════════════════════════
# _scheduler (lines 262-318)
# ═════════════════════════════════════════════════════════════════════════════


class TestScheduler:
    """Cover _scheduler: daily reset, EOD flatten, EOD summary."""

    @pytest.mark.asyncio
    async def test_daily_reset_triggered(self) -> None:
        """At 9:31 ET the scheduler fires daily reset."""
        mock_now = datetime(2024, 1, 15, 9, 31, tzinfo=_ET)

        pipeline = _make_pipeline(symbol="SPY")
        pipeline.levels = MagicMock()
        pipelines = {"SPY": pipeline}
        cooldown = MagicMock()
        dispatcher = AsyncMock()
        db = _make_db()

        sleep_calls = 0

        async def _fake_sleep(_sec: float) -> None:
            nonlocal sleep_calls
            sleep_calls += 1
            if sleep_calls >= 2:
                raise asyncio.CancelledError

        with (
            patch("src.main.asyncio.sleep", side_effect=_fake_sleep),
            patch("src.main.datetime") as mock_dt,
        ):
            mock_dt.now.return_value = mock_now
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            with suppress(asyncio.CancelledError):
                await _scheduler(pipelines, cooldown, dispatcher, db)

        cooldown.reset_daily.assert_called_once()

    @pytest.mark.asyncio
    async def test_daily_reset_not_triggered_before_930(self) -> None:
        """Before 9:30 ET the scheduler does NOT fire daily reset."""
        mock_now = datetime(2024, 1, 15, 9, 0, tzinfo=_ET)

        pipelines = {"SPY": _make_pipeline(symbol="SPY")}
        cooldown = MagicMock()

        async def _cancel_after_one(_sec: float) -> None:
            raise asyncio.CancelledError

        with (
            patch("src.main.asyncio.sleep", side_effect=_cancel_after_one),
            patch("src.main.datetime") as mock_dt,
        ):
            mock_dt.now.return_value = mock_now
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            with suppress(asyncio.CancelledError):
                await _scheduler(pipelines, cooldown, AsyncMock(), _make_db())

        cooldown.reset_daily.assert_not_called()

    @pytest.mark.asyncio
    async def test_eod_flatten_with_executor(self) -> None:
        """At 15:56 ET with an executor, EOD flatten fires."""
        mock_now = datetime(2024, 1, 15, 15, 56, tzinfo=_ET)

        pipelines = {"SPY": _make_pipeline(symbol="SPY")}
        pipelines["SPY"].levels = MagicMock()
        cooldown = MagicMock()
        executor = MagicMock()
        executor.cancel_open_orders.return_value = 2
        executor.flatten_all_positions.return_value = 1

        sleep_calls = 0

        async def _fake_sleep(_sec: float) -> None:
            nonlocal sleep_calls
            sleep_calls += 1
            if sleep_calls >= 2:
                raise asyncio.CancelledError

        with (
            patch("src.main.asyncio.sleep", side_effect=_fake_sleep),
            patch("src.main.datetime") as mock_dt,
        ):
            mock_dt.now.return_value = mock_now
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            with suppress(asyncio.CancelledError):
                await _scheduler(pipelines, cooldown, AsyncMock(), _make_db(), executor=executor)

        executor.cancel_open_orders.assert_called_once()
        executor.flatten_all_positions.assert_called_once()

    @pytest.mark.asyncio
    async def test_eod_flatten_skipped_without_executor(self) -> None:
        """At 15:56 ET without executor, EOD flatten does NOT fire."""
        mock_now = datetime(2024, 1, 15, 15, 56, tzinfo=_ET)

        pipelines = {"SPY": _make_pipeline(symbol="SPY")}
        pipelines["SPY"].levels = MagicMock()
        cooldown = MagicMock()

        sleep_calls = 0

        async def _fake_sleep(_sec: float) -> None:
            nonlocal sleep_calls
            sleep_calls += 1
            if sleep_calls >= 2:
                raise asyncio.CancelledError

        with (
            patch("src.main.asyncio.sleep", side_effect=_fake_sleep),
            patch("src.main.datetime") as mock_dt,
        ):
            mock_dt.now.return_value = mock_now
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            with suppress(asyncio.CancelledError):
                await _scheduler(pipelines, cooldown, AsyncMock(), _make_db(), executor=None)

        # No executor — nothing to assert on, just verify no crash

    @pytest.mark.asyncio
    async def test_eod_flatten_exception_caught(self) -> None:
        """Exception in EOD flatten is caught and logged, not propagated."""
        mock_now = datetime(2024, 1, 15, 15, 56, tzinfo=_ET)

        pipelines = {"SPY": _make_pipeline(symbol="SPY")}
        pipelines["SPY"].levels = MagicMock()
        cooldown = MagicMock()
        executor = MagicMock()
        executor.cancel_open_orders.side_effect = RuntimeError("API error")

        sleep_calls = 0

        async def _fake_sleep(_sec: float) -> None:
            nonlocal sleep_calls
            sleep_calls += 1
            if sleep_calls >= 2:
                raise asyncio.CancelledError

        with (
            patch("src.main.asyncio.sleep", side_effect=_fake_sleep),
            patch("src.main.datetime") as mock_dt,
        ):
            mock_dt.now.return_value = mock_now
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            with suppress(asyncio.CancelledError):
                await _scheduler(pipelines, cooldown, AsyncMock(), _make_db(), executor=executor)

        # Exception caught — no crash

    @pytest.mark.asyncio
    async def test_eod_summary_triggered(self) -> None:
        """At 16:06 ET the EOD summary fires."""
        mock_now = datetime(2024, 1, 15, 16, 6, tzinfo=_ET)

        pipelines = {"SPY": _make_pipeline(symbol="SPY")}
        pipelines["SPY"].levels = MagicMock()
        cooldown = MagicMock()
        dispatcher = AsyncMock()
        db = _make_db()

        sleep_calls = 0

        async def _fake_sleep(_sec: float) -> None:
            nonlocal sleep_calls
            sleep_calls += 1
            if sleep_calls >= 2:
                raise asyncio.CancelledError

        with (
            patch("src.main.asyncio.sleep", side_effect=_fake_sleep),
            patch("src.main.datetime") as mock_dt,
            patch("src.main.query_recent_trades", return_value=[], create=True),
            patch("src.storage.queries.query_recent_trades", return_value=[]),
        ):
            mock_dt.now.return_value = mock_now
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            with suppress(asyncio.CancelledError):
                await _scheduler(pipelines, cooldown, dispatcher, db)

        dispatcher.dispatch_daily_summary.assert_called_once()

    @pytest.mark.asyncio
    async def test_eod_summary_exception_caught(self) -> None:
        """Exception in EOD summary is caught and logged."""
        mock_now = datetime(2024, 1, 15, 16, 6, tzinfo=_ET)

        pipelines = {"SPY": _make_pipeline(symbol="SPY")}
        pipelines["SPY"].levels = MagicMock()
        cooldown = MagicMock()
        dispatcher = AsyncMock()
        db = _make_db()
        # Make the db.conn trigger an error via query_recent_trades
        db.conn = MagicMock()

        sleep_calls = 0

        async def _fake_sleep(_sec: float) -> None:
            nonlocal sleep_calls
            sleep_calls += 1
            if sleep_calls >= 2:
                raise asyncio.CancelledError

        with (
            patch("src.main.asyncio.sleep", side_effect=_fake_sleep),
            patch("src.main.datetime") as mock_dt,
            patch(
                "src.storage.queries.query_recent_trades",
                side_effect=RuntimeError("db error"),
            ),
        ):
            mock_dt.now.return_value = mock_now
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            with suppress(asyncio.CancelledError):
                await _scheduler(pipelines, cooldown, dispatcher, db)

        # No crash — exception caught

    @pytest.mark.asyncio
    async def test_daily_reset_only_once_per_day(self) -> None:
        """Daily reset fires once per day, not on subsequent scheduler ticks."""
        mock_now = datetime(2024, 1, 15, 9, 31, tzinfo=_ET)

        pipeline = _make_pipeline(symbol="SPY")
        pipeline.levels = MagicMock()
        pipelines = {"SPY": pipeline}
        cooldown = MagicMock()

        sleep_calls = 0

        async def _fake_sleep(_sec: float) -> None:
            nonlocal sleep_calls
            sleep_calls += 1
            if sleep_calls >= 3:
                raise asyncio.CancelledError

        with (
            patch("src.main.asyncio.sleep", side_effect=_fake_sleep),
            patch("src.main.datetime") as mock_dt,
        ):
            mock_dt.now.return_value = mock_now
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            with suppress(asyncio.CancelledError):
                await _scheduler(pipelines, cooldown, AsyncMock(), _make_db())

        # Two scheduler iterations after 9:30 but reset should fire only once
        cooldown.reset_daily.assert_called_once()


# ═════════════════════════════════════════════════════════════════════════════
# _install_signal_handlers (lines 370-376)
# ═════════════════════════════════════════════════════════════════════════════


class TestInstallSignalHandlers:
    """Cover _install_signal_handlers."""

    def test_registers_sigint_and_sigterm(self) -> None:
        """Should register handlers for both SIGINT and SIGTERM."""
        mock_loop = MagicMock()
        _install_signal_handlers(mock_loop)
        assert mock_loop.add_signal_handler.call_count == 2

    def test_registered_signals(self) -> None:
        """The two registered signals are SIGINT and SIGTERM."""
        mock_loop = MagicMock()
        _install_signal_handlers(mock_loop)
        registered_signals = {call.args[0] for call in mock_loop.add_signal_handler.call_args_list}
        assert registered_signals == {_signal.SIGINT, _signal.SIGTERM}

    def test_handler_sets_shutdown_event(self) -> None:
        """The registered handler sets the shutdown event."""

        mock_loop = MagicMock()
        _install_signal_handlers(mock_loop)

        from src.main import _shutdown_event as event

        assert event is not None
        assert not event.is_set()

        # Extract the handler function from the first call
        handler_fn = mock_loop.add_signal_handler.call_args_list[0].args[1]
        sig_value = mock_loop.add_signal_handler.call_args_list[0].args[2]
        handler_fn(sig_value)

        assert event.is_set()


class TestShutdownWatcher:
    """Tests for _shutdown_watcher — cancels sibling tasks on shutdown event."""

    @pytest.mark.asyncio
    async def test_cancels_tasks_when_event_set(self) -> None:
        import src.main as main_mod
        from src.main import _shutdown_watcher

        event = asyncio.Event()
        main_mod._shutdown_event = event

        mock_task = MagicMock()

        # Set event immediately so watcher doesn't block
        event.set()
        await _shutdown_watcher([mock_task])

        mock_task.cancel.assert_called_once()


# ═════════════════════════════════════════════════════════════════════════════
# _start_api (lines 356-361)
# ═════════════════════════════════════════════════════════════════════════════


class TestStartApi:
    """Cover _start_api."""

    @pytest.mark.asyncio
    async def test_start_api_creates_server_and_serves(self) -> None:
        """_start_api creates a uvicorn.Server and calls serve()."""
        mock_server_instance = MagicMock()
        mock_server_instance.serve = AsyncMock()

        with (
            patch("src.main.uvicorn.Config") as mock_config_cls,
            patch("src.main.uvicorn.Server", return_value=mock_server_instance) as mock_server_cls,
            patch("src.api.routes.app", create=True),
        ):
            await _start_api(host="0.0.0.0", port=9000)  # noqa: S104 - test value

        mock_config_cls.assert_called_once()
        config_kwargs = mock_config_cls.call_args
        assert config_kwargs.kwargs["host"] == "0.0.0.0"  # noqa: S104 - test assertion
        assert config_kwargs.kwargs["port"] == 9000
        mock_server_cls.assert_called_once()
        mock_server_instance.serve.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_start_api_default_params(self) -> None:
        """_start_api uses default host/port."""
        mock_server_instance = MagicMock()
        mock_server_instance.serve = AsyncMock()

        with (
            patch("src.main.uvicorn.Config") as mock_config_cls,
            patch("src.main.uvicorn.Server", return_value=mock_server_instance),
            patch("src.api.routes.app", create=True),
        ):
            await _start_api()

        config_kwargs = mock_config_cls.call_args
        assert config_kwargs.kwargs["host"] == "127.0.0.1"
        assert config_kwargs.kwargs["port"] == 8000


# ═════════════════════════════════════════════════════════════════════════════
# run() (lines 384-473) and main() (line 478, 482)
# ═════════════════════════════════════════════════════════════════════════════


class TestRun:
    """Cover run() entry point — heavily mocked."""

    @pytest.mark.asyncio
    async def test_run_wires_pipeline_and_shuts_down(self) -> None:
        """run() connects DB, builds pipelines, starts TaskGroup, shuts down cleanly."""
        from src.main import run

        mock_db = MagicMock()
        mock_db.conn = MagicMock()

        mock_app_settings = MagicMock()
        mock_app_settings.symbols = ["SPY"]
        mock_app_settings.db_path = ":memory:"
        mock_app_settings.trading_mode = "signal_only"
        mock_app_settings.execution_mode = "signal_only"

        mock_risk_settings = MagicMock()
        mock_risk_settings.account_size = Decimal("50000")
        mock_risk_settings.risk_per_trade_pct = Decimal("0.01")
        mock_risk_settings.max_concurrent_positions = 3

        mock_telegram_settings = MagicMock()
        mock_telegram_settings.bot_token = "fake"  # noqa: S105 - test stub
        mock_telegram_settings.chat_id = "fake"

        async def _cancel_task_group(**_kw: object) -> None:
            """Simulate immediate shutdown."""
            raise asyncio.CancelledError

        with (
            patch("src.main.get_app_settings", return_value=mock_app_settings),
            patch("src.main.get_risk_settings", return_value=mock_risk_settings),
            patch("src.main.get_telegram_settings", return_value=mock_telegram_settings),
            patch("src.main.BarDatabase", return_value=mock_db),
            patch("src.main.ensure_schema"),
            patch("src.main._build_pipelines") as mock_build,
            patch("src.main.CooldownTracker"),
            patch("src.main.RiskManager"),
            patch("src.main.TelegramAlerter"),
            patch("src.main.AlertDispatcher"),
            patch("src.main.AlpacaBarStream") as mock_stream_cls,
            patch("src.main._install_signal_handlers"),
            patch("src.api.routes.set_dependencies"),
            patch("src.main._start_api", new_callable=AsyncMock, side_effect=_cancel_task_group),
            patch("src.main._bar_loop", new_callable=AsyncMock, side_effect=_cancel_task_group),
            patch("src.main._scheduler", new_callable=AsyncMock, side_effect=_cancel_task_group),
        ):
            mock_pipeline = _make_pipeline(symbol="SPY")
            mock_build.return_value = {"SPY": mock_pipeline}
            mock_stream_cls.return_value.start = AsyncMock(side_effect=_cancel_task_group)
            mock_stream_cls.return_value.stop = AsyncMock()

            await run()

        mock_db.connect.assert_called_once()
        mock_db.close.assert_called_once()
        mock_stream_cls.return_value.stop.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_run_with_executor_enabled(self) -> None:
        """run() with paper_trade mode creates an AlpacaExecutor."""
        from src.main import run

        mock_db = MagicMock()
        mock_db.conn = MagicMock()

        mock_app_settings = MagicMock()
        mock_app_settings.symbols = ["SPY"]
        mock_app_settings.db_path = ":memory:"
        mock_app_settings.trading_mode = "signal_only"
        mock_app_settings.execution_mode = "paper_trade"

        mock_risk_settings = MagicMock()
        mock_risk_settings.account_size = Decimal("50000")
        mock_risk_settings.risk_per_trade_pct = Decimal("0.01")
        mock_risk_settings.max_concurrent_positions = 3

        mock_telegram_settings = MagicMock()
        mock_telegram_settings.bot_token = "fake"  # noqa: S105 - test stub
        mock_telegram_settings.chat_id = "fake"

        mock_alpaca_settings = MagicMock()
        mock_alpaca_settings.api_key = "key"
        mock_alpaca_settings.secret_key = "secret"  # noqa: S105 - test stub

        async def _cancel_task_group(**_kw: object) -> None:
            raise asyncio.CancelledError

        with (
            patch("src.main.get_app_settings", return_value=mock_app_settings),
            patch("src.main.get_risk_settings", return_value=mock_risk_settings),
            patch("src.main.get_telegram_settings", return_value=mock_telegram_settings),
            patch("src.main.get_alpaca_settings", return_value=mock_alpaca_settings),
            patch("src.main.BarDatabase", return_value=mock_db),
            patch("src.main.ensure_schema"),
            patch("src.main._build_pipelines") as mock_build,
            patch("src.main.CooldownTracker"),
            patch("src.main.RiskManager"),
            patch("src.main.TelegramAlerter"),
            patch("src.main.AlertDispatcher"),
            patch("src.main.AlpacaExecutor") as mock_executor_cls,
            patch("src.main.AlpacaBarStream") as mock_stream_cls,
            patch("src.main._install_signal_handlers"),
            patch("src.api.routes.set_dependencies"),
            patch("src.main._start_api", new_callable=AsyncMock, side_effect=_cancel_task_group),
            patch("src.main._bar_loop", new_callable=AsyncMock, side_effect=_cancel_task_group),
            patch("src.main._scheduler", new_callable=AsyncMock, side_effect=_cancel_task_group),
        ):
            mock_pipeline = _make_pipeline(symbol="SPY")
            mock_build.return_value = {"SPY": mock_pipeline}
            mock_stream_cls.return_value.start = AsyncMock(side_effect=_cancel_task_group)
            mock_stream_cls.return_value.stop = AsyncMock()

            await run()

        expected_secret = "secret"  # noqa: S105
        mock_executor_cls.assert_called_once_with(
            api_key="key",
            secret_key=expected_secret,
            paper=True,
        )


class TestMain:
    """Cover main() entry point (line 478, 482)."""

    def test_main_calls_asyncio_run(self) -> None:
        """main() delegates to asyncio.run(run())."""
        # Verify the source code wires main → asyncio.run(run()).
        # We avoid calling main() directly because `run` is an async def
        # and any mock for it produces an AsyncMock whose unawaited
        # coroutine leaks a RuntimeWarning during GC.
        import ast
        import inspect

        import src.main as main_mod

        source = inspect.getsource(main_mod.main)
        tree = ast.parse(source)
        # The function body should contain an expression calling asyncio.run(run())
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "run"
        ]
        assert len(calls) >= 1, "main() must call asyncio.run(...)"


# ═════════════════════════════════════════════════════════════════════════════
# _send_eod_status
# ═════════════════════════════════════════════════════════════════════════════


class TestSendEodStatus:
    """Tests for _send_eod_status()."""

    @pytest.mark.asyncio
    async def test_zero_signals_includes_warning(self) -> None:
        """When no signals today, the message contains a zero-signal warning."""
        dispatcher = AsyncMock()
        db = MagicMock()
        cooldown = MagicMock()
        cooldown.daily_trade_count = 0
        cooldown.consecutive_losses = 0

        pipeline = _make_pipeline(symbol="SPY")
        pipeline.levels._orb.is_complete = False
        pipelines = {"SPY": pipeline}

        with (
            patch("src.main.query_recent_signals", return_value=[]),
            patch("src.main.query_recent_trades", return_value=[]),
        ):
            await _send_eod_status(dispatcher, db, pipelines, cooldown)

        dispatcher.dispatch_status.assert_awaited_once()
        msg = dispatcher.dispatch_status.call_args[0][0]
        assert "No signals today" in msg or "No trades today" in msg

    @pytest.mark.asyncio
    async def test_with_signals_and_trades(self) -> None:
        """When signals and trades exist, message includes counts and P&L."""
        dispatcher = AsyncMock()
        db = MagicMock()
        cooldown = MagicMock()
        cooldown.daily_trade_count = 3
        cooldown.consecutive_losses = 1

        pipeline = _make_pipeline(symbol="SPY")
        pipeline.levels._orb.is_complete = True
        pipelines = {"SPY": pipeline}

        signals = [
            {"approved": 1, "reject_reason": ""},
            {"approved": 0, "reject_reason": "cooldown"},
            {"approved": 0, "reject_reason": "cooldown"},
        ]
        trades = [{"pnl": "150.00"}, {"pnl": "-50.00"}]

        with (
            patch("src.main.query_recent_signals", return_value=signals),
            patch("src.main.query_recent_trades", return_value=trades),
        ):
            await _send_eod_status(dispatcher, db, pipelines, cooldown)

        dispatcher.dispatch_status.assert_awaited_once()
        msg = dispatcher.dispatch_status.call_args[0][0]
        assert "Generated: 3" in msg
        assert "Approved: 1" in msg
        assert "Rejected: 2" in msg
        assert "$+100.00" in msg
        assert "cooldown: 2" in msg

    @pytest.mark.asyncio
    async def test_orb_count_in_message(self) -> None:
        """ORB completion count is reported correctly."""
        dispatcher = AsyncMock()
        db = MagicMock()
        cooldown = MagicMock()
        cooldown.daily_trade_count = 0
        cooldown.consecutive_losses = 0

        p1 = _make_pipeline(symbol="SPY")
        p1.levels._orb.is_complete = True
        p2 = _make_pipeline(symbol="QQQ")
        p2.levels._orb.is_complete = False
        pipelines = {"SPY": p1, "QQQ": p2}

        with (
            patch("src.main.query_recent_signals", return_value=[]),
            patch("src.main.query_recent_trades", return_value=[]),
        ):
            await _send_eod_status(dispatcher, db, pipelines, cooldown)

        msg = dispatcher.dispatch_status.call_args[0][0]
        assert "Formed: 1/2" in msg

    @pytest.mark.asyncio
    async def test_all_rejected_shows_top_filter(self) -> None:
        """When signals exist but all rejected, message names the top filter."""
        dispatcher = AsyncMock()
        db = MagicMock()
        cooldown = MagicMock()
        cooldown.daily_trade_count = 0
        cooldown.consecutive_losses = 0

        pipeline = _make_pipeline(symbol="SPY")
        pipeline.levels._orb.is_complete = True
        pipelines = {"SPY": pipeline}

        signals = [
            {"approved": 0, "reject_reason": "regime_gate"},
            {"approved": 0, "reject_reason": "regime_gate"},
            {"approved": 0, "reject_reason": "lunch_chop"},
        ]

        with (
            patch("src.main.query_recent_signals", return_value=signals),
            patch("src.main.query_recent_trades", return_value=[]),
        ):
            await _send_eod_status(dispatcher, db, pipelines, cooldown)

        msg = dispatcher.dispatch_status.call_args[0][0]
        assert "No trades today" in msg
        assert "regime_gate" in msg


# ═════════════════════════════════════════════════════════════════════════════
# _scheduler — ORB check at 9:50 ET
# ═════════════════════════════════════════════════════════════════════════════


class TestSchedulerOrbCheck:
    """Tests for the ORB formation watchdog at 9:50 ET."""

    @pytest.mark.asyncio
    async def test_orb_check_no_orbs_sends_risk_warning(self) -> None:
        """At 9:51 ET with zero ORBs, a risk warning fires."""
        mock_now = datetime(2024, 1, 15, 9, 51, tzinfo=_ET)

        pipeline = _make_pipeline(symbol="SPY")
        pipeline.levels = MagicMock()
        pipeline.levels._orb.is_complete = False
        pipelines = {"SPY": pipeline}
        cooldown = MagicMock()
        dispatcher = AsyncMock()

        sleep_calls = 0

        async def _fake_sleep(_sec: float) -> None:
            nonlocal sleep_calls
            sleep_calls += 1
            if sleep_calls >= 2:
                raise asyncio.CancelledError

        with (
            patch("src.main.asyncio.sleep", side_effect=_fake_sleep),
            patch("src.main.datetime") as mock_dt,
        ):
            mock_dt.now.return_value = mock_now
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            with suppress(asyncio.CancelledError):
                await _scheduler(pipelines, cooldown, dispatcher, _make_db())

        dispatcher.dispatch_risk_warning.assert_called_once()
        msg = dispatcher.dispatch_risk_warning.call_args[0][0]
        assert "No ORB ranges formed" in msg

    @pytest.mark.asyncio
    async def test_orb_check_partial_sends_status(self) -> None:
        """At 9:51 ET with < half ORBs, a status warning fires."""
        mock_now = datetime(2024, 1, 15, 9, 51, tzinfo=_ET)

        # 1 out of 4 complete → 1 < 4 // 2 (=2) → triggers partial warning
        p1 = _make_pipeline(symbol="SPY")
        p1.levels = MagicMock()
        p1.levels._orb.is_complete = True
        p2 = _make_pipeline(symbol="QQQ")
        p2.levels = MagicMock()
        p2.levels._orb.is_complete = False
        p3 = _make_pipeline(symbol="AAPL")
        p3.levels = MagicMock()
        p3.levels._orb.is_complete = False
        p4 = _make_pipeline(symbol="MSFT")
        p4.levels = MagicMock()
        p4.levels._orb.is_complete = False
        pipelines = {"SPY": p1, "QQQ": p2, "AAPL": p3, "MSFT": p4}
        cooldown = MagicMock()
        dispatcher = AsyncMock()

        sleep_calls = 0

        async def _fake_sleep(_sec: float) -> None:
            nonlocal sleep_calls
            sleep_calls += 1
            if sleep_calls >= 2:
                raise asyncio.CancelledError

        with (
            patch("src.main.asyncio.sleep", side_effect=_fake_sleep),
            patch("src.main.datetime") as mock_dt,
        ):
            mock_dt.now.return_value = mock_now
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            with suppress(asyncio.CancelledError):
                await _scheduler(pipelines, cooldown, dispatcher, _make_db())

        dispatcher.dispatch_status.assert_called()
        # Find the ORB-related call
        orb_calls = [
            c
            for c in dispatcher.dispatch_status.call_args_list
            if "Partial ORB" in str(c) or "formed ORB" in str(c)
        ]
        assert len(orb_calls) == 1

    @pytest.mark.asyncio
    async def test_orb_check_all_complete_no_alert(self) -> None:
        """At 9:51 ET with all ORBs complete, no alert fires."""
        mock_now = datetime(2024, 1, 15, 9, 51, tzinfo=_ET)

        pipeline = _make_pipeline(symbol="SPY")
        pipeline.levels = MagicMock()
        pipeline.levels._orb.is_complete = True
        pipelines = {"SPY": pipeline}
        cooldown = MagicMock()
        dispatcher = AsyncMock()

        sleep_calls = 0

        async def _fake_sleep(_sec: float) -> None:
            nonlocal sleep_calls
            sleep_calls += 1
            if sleep_calls >= 2:
                raise asyncio.CancelledError

        with (
            patch("src.main.asyncio.sleep", side_effect=_fake_sleep),
            patch("src.main.datetime") as mock_dt,
        ):
            mock_dt.now.return_value = mock_now
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            with suppress(asyncio.CancelledError):
                await _scheduler(pipelines, cooldown, dispatcher, _make_db())

        dispatcher.dispatch_risk_warning.assert_not_called()


# ═════════════════════════════════════════════════════════════════════════════
# _scheduler — EOD status called after daily summary
# ═════════════════════════════════════════════════════════════════════════════


class TestSchedulerEodStatus:
    """Tests that _send_eod_status is called during EOD summary."""

    @pytest.mark.asyncio
    async def test_eod_status_called_at_summary_time(self) -> None:
        """At 16:06 ET, both dispatch_daily_summary and _send_eod_status fire."""
        mock_now = datetime(2024, 1, 15, 16, 6, tzinfo=_ET)

        pipeline = _make_pipeline(symbol="SPY")
        pipeline.levels = MagicMock()
        pipelines = {"SPY": pipeline}
        cooldown = MagicMock()
        cooldown.daily_trade_count = 0
        cooldown.consecutive_losses = 0
        dispatcher = AsyncMock()
        db = _make_db()

        sleep_calls = 0

        async def _fake_sleep(_sec: float) -> None:
            nonlocal sleep_calls
            sleep_calls += 1
            if sleep_calls >= 2:
                raise asyncio.CancelledError

        with (
            patch("src.main.asyncio.sleep", side_effect=_fake_sleep),
            patch("src.main.datetime") as mock_dt,
            patch("src.main.query_recent_trades", return_value=[]),
            patch("src.main.query_recent_signals", return_value=[]),
        ):
            mock_dt.now.return_value = mock_now
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            with suppress(asyncio.CancelledError):
                await _scheduler(pipelines, cooldown, dispatcher, db)

        dispatcher.dispatch_daily_summary.assert_called_once()
        # _send_eod_status calls dispatch_status
        dispatcher.dispatch_status.assert_called()


# ═════════════════════════════════════════════════════════════════════════════
# Regime warmup tracking
# ═════════════════════════════════════════════════════════════════════════════


class TestRegimeWarmupTracking:
    """Tests for _regime_warmup_bar_counts and _regime_first_populated."""

    @pytest.mark.asyncio
    async def test_bar_count_increments(self) -> None:
        """Each call to _process_bar increments the per-symbol bar count."""
        # Clear module-level state
        _regime_warmup_bar_counts.clear()
        _regime_first_populated.clear()

        pipeline = _make_pipeline(atr=Decimal("2.5"), ema9=Decimal("485"), ema20=Decimal("480"))
        bar = make_bar(symbol="TEST_WU")
        await _process_bar(bar, pipeline, _make_risk(), _make_dispatcher(), _make_db())

        assert _regime_warmup_bar_counts.get("TEST_WU") == 1

        await _process_bar(bar, pipeline, _make_risk(), _make_dispatcher(), _make_db())
        assert _regime_warmup_bar_counts.get("TEST_WU") == 2

    @pytest.mark.asyncio
    async def test_regime_first_populated_logged(self) -> None:
        """When regime gets vix + adx, symbol is added to _regime_first_populated."""
        _regime_warmup_bar_counts.clear()
        _regime_first_populated.clear()

        pipeline = _make_pipeline(atr=Decimal("2.5"), ema9=Decimal("485"), ema20=Decimal("480"))
        pipeline.regime.vix_level = Decimal("20")
        pipeline.regime.adx_value = Decimal("25")
        pipeline.regime.current_regime = "TRENDING_UP"

        bar = make_bar(symbol="TEST_FP")
        await _process_bar(bar, pipeline, _make_risk(), _make_dispatcher(), _make_db())

        assert "TEST_FP" in _regime_first_populated
