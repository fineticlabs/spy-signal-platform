"""Main asyncio orchestrator for the Signal Platform (multi-ticker).

Startup sequence
----------------
1. Load config from .env — reads SYMBOLS for multi-ticker list
2. Connect SQLite — ensure schema
3. Per-symbol: build IndicatorRegistry, LevelManager, RegimeDetector, ORBStrategy
4. Shared: RiskManager, AlertDispatcher (Telegram)
5. Inject live state into FastAPI
6. Start FastAPI server (uvicorn, port 8000) as a background asyncio task
7. Start Alpaca WebSocket stream subscribing to all symbols

Main loop
---------
For every bar arriving on the queue:
  - Route to the correct per-symbol pipeline (IndicatorRegistry, LevelManager, etc.)
  - Update indicators
  - Update levels
  - Evaluate strategy → Signal | None
  - Tag signal with ticker symbol
  - If signal: run through shared RiskManager
  - If approved: dispatch alert (Telegram message shows ticker prominently)
  - Log signal + decision to SQLite

Scheduled tasks
---------------
- Daily reset at 9:30 ET — clear all per-symbol level/indicator state, reset risk counters
- End-of-day summary at 16:05 ET — dispatch daily P&L summary via Telegram

Graceful shutdown
-----------------
SIGINT / SIGTERM cancel the bar-processing loop and the WebSocket task, then
flush pending state and close the DB connection.
"""

from __future__ import annotations

import asyncio
import signal as _signal
from dataclasses import dataclass
from datetime import UTC, datetime, time
from decimal import Decimal
from zoneinfo import ZoneInfo

import structlog
import uvicorn

from src.alerts.dispatcher import AlertDispatcher
from src.alerts.telegram import TelegramAlerter
from src.config import (
    get_app_settings,
    get_risk_settings,
    get_telegram_settings,
)
from src.indicators.registry import IndicatorRegistry
from src.indicators.streaming import StreamingATR, StreamingEMA, StreamingMACD, StreamingRSI
from src.ingestion.websocket import AlpacaBarStream
from src.levels import LevelManager
from src.models import Bar, TimeFrame, TradeResult
from src.risk.cooldown import CooldownTracker
from src.risk.manager import RiskManager
from src.storage.database import BarDatabase
from src.storage.queries import ensure_schema, insert_signal
from src.strategies.orb import ORBStrategy
from src.strategies.regime import RegimeDetector

logger = structlog.get_logger(__name__)

_ET = ZoneInfo("America/New_York")

# Scheduled ET wall-clock times
_DAILY_RESET_ET = time(9, 30)
_EOD_SUMMARY_ET = time(16, 5)

# How often the scheduler wakes (seconds)
_SCHEDULER_INTERVAL = 30


# ── per-symbol pipeline container ─────────────────────────────────────────────


@dataclass
class _SymbolPipeline:
    """All per-symbol stateful components for one ticker."""

    symbol: str
    registry: IndicatorRegistry
    levels: LevelManager
    regime: RegimeDetector
    strategy: ORBStrategy


# ── pipeline wiring ────────────────────────────────────────────────────────────


def _build_registry() -> IndicatorRegistry:
    """Construct and return a fully wired IndicatorRegistry."""
    registry = IndicatorRegistry()
    registry.register("ema9", StreamingEMA(9))
    registry.register("ema20", StreamingEMA(20))
    registry.register("ema50", StreamingEMA(50))
    registry.register("rsi", StreamingRSI(14))
    registry.register("macd", StreamingMACD())
    registry.register("atr", StreamingATR(14))
    return registry


def _build_pipelines(symbols: list[str], db: BarDatabase) -> dict[str, _SymbolPipeline]:
    """Build one :class:`_SymbolPipeline` per ticker symbol.

    Args:
        symbols: List of uppercase ticker symbols.
        db:      Connected :class:`BarDatabase` (needed by LevelManager).

    Returns:
        Dict mapping symbol → pipeline.
    """
    pipelines: dict[str, _SymbolPipeline] = {}
    for symbol in symbols:
        pipelines[symbol] = _SymbolPipeline(
            symbol=symbol,
            registry=_build_registry(),
            levels=LevelManager(db=db, symbol=symbol),
            regime=RegimeDetector(),
            strategy=ORBStrategy(),
        )
        logger.info("pipeline_built", symbol=symbol)
    return pipelines


# ── bar processing ─────────────────────────────────────────────────────────────


async def _process_bar(
    bar: Bar,
    pipeline: _SymbolPipeline,
    risk: RiskManager,
    dispatcher: AlertDispatcher,
    db: BarDatabase,
) -> None:
    """Run a single bar through the full pipeline for its symbol."""
    # 1. Indicators
    pipeline.registry.update_all(bar)
    indicator_snapshot = pipeline.registry.get_snapshot()

    # 2. Levels
    pipeline.levels.update(bar)
    level_snapshot = pipeline.levels.get_levels()

    # Update regime from freshest indicator values
    atr = indicator_snapshot.atr
    if atr is not None:
        e9 = indicator_snapshot.ema9
        e20 = indicator_snapshot.ema20
        trending_up: bool | None = None
        if e9 is not None and e20 is not None:
            trending_up = e9 > e20
        pipeline.regime.update(trending_up=trending_up)

    # 3. Strategy evaluation
    signal = pipeline.strategy.evaluate(bar, indicator_snapshot, level_snapshot, pipeline.regime)

    if signal is None:
        return

    # Tag signal with the actual ticker symbol
    signal = signal.model_copy(update={"symbol": bar.symbol})

    logger.info(
        "signal_generated",
        symbol=bar.symbol,
        strategy=signal.strategy_name,
        direction=str(signal.direction),
        entry=str(signal.entry_price),
    )

    # 4. Risk gate (shared across all tickers — account-level limits)
    decision = risk.approve(signal)

    # 5. Persist signal + decision regardless of approval
    try:
        insert_signal(db.conn, signal, decision)
    except Exception as exc:
        logger.error("signal_persist_failed", symbol=bar.symbol, error=str(exc))

    # 6. Alert if approved
    if decision.approved:
        await dispatcher.dispatch_signal(signal, decision)
    else:
        logger.warning(
            "signal_rejected",
            symbol=bar.symbol,
            reason=decision.reason,
            strategy=signal.strategy_name,
        )


# ── bar ingestion loop ─────────────────────────────────────────────────────────


async def _bar_loop(
    queue: asyncio.Queue[Bar],
    pipelines: dict[str, _SymbolPipeline],
    risk: RiskManager,
    dispatcher: AlertDispatcher,
    db: BarDatabase,
) -> None:
    """Consume bars from the queue, route to the correct per-symbol pipeline."""
    logger.info("bar_loop_started", symbols=list(pipelines.keys()))
    while True:
        bar = await queue.get()
        pipeline = pipelines.get(bar.symbol)
        if pipeline is None:
            logger.warning("bar_unknown_symbol", symbol=bar.symbol)
            queue.task_done()
            continue
        try:
            await _process_bar(bar, pipeline, risk, dispatcher, db)
        except Exception as exc:
            logger.error(
                "bar_processing_error",
                symbol=bar.symbol,
                error=str(exc),
                bar_ts=bar.timestamp.isoformat(),
            )
        finally:
            queue.task_done()


# ── scheduler ─────────────────────────────────────────────────────────────────


async def _scheduler(
    pipelines: dict[str, _SymbolPipeline],
    risk_cooldown: CooldownTracker,
    dispatcher: AlertDispatcher,
    db: BarDatabase,
) -> None:
    """Fire periodic events: daily reset at 9:30 ET, EOD summary at 16:05 ET."""
    last_reset_date: datetime | None = None
    last_eod_date: datetime | None = None

    while True:
        await asyncio.sleep(_SCHEDULER_INTERVAL)
        now_et = datetime.now(_ET)
        today = now_et.date()

        # ── daily reset ──────────────────────────────────────────────────────
        if now_et.time() >= _DAILY_RESET_ET and (
            last_reset_date is None or last_reset_date.date() < today
        ):
            logger.info("daily_reset", date=str(today), symbols=list(pipelines.keys()))
            risk_cooldown.reset_daily()
            for pipeline in pipelines.values():
                # Reset all per-symbol level state
                pipeline.levels._orb.__init__()  # type: ignore[misc]
                pipeline.levels._vwap.__init__()  # type: ignore[misc]
                pipeline.levels._day.__init__()  # type: ignore[misc]
                pipeline.levels._premarket.__init__()  # type: ignore[misc]
                pipeline.levels._last_date = None
            last_reset_date = now_et

        # ── end-of-day summary ───────────────────────────────────────────────
        if now_et.time() >= _EOD_SUMMARY_ET and (
            last_eod_date is None or last_eod_date.date() < today
        ):
            logger.info("eod_summary", date=str(today))
            try:
                since = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
                from src.storage.queries import query_recent_trades

                raw_trades = query_recent_trades(db.conn, since=since, limit=500)
                await dispatcher.dispatch_daily_summary(cast_trades(raw_trades))
            except Exception as exc:
                logger.error("eod_summary_failed", error=str(exc))
            last_eod_date = now_et


def cast_trades(raw: list[dict[str, object]]) -> list[TradeResult]:
    """Convert raw DB dicts back to minimal TradeResult objects for the summary."""
    from src.models import Direction, Regime, Signal, TimeFrame, TradeResult

    results: list[TradeResult] = []
    for row in raw:
        try:
            ts = datetime.fromisoformat(str(row["timestamp"]))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=UTC)
            sig = Signal(
                symbol=str(row.get("symbol", "?")),
                direction=Direction(str(row["direction"])),
                strategy_name=str(row["strategy_name"]),
                entry_price=Decimal(str(row["entry_price"])),
                stop_price=Decimal(str(row["stop_price"])),
                target_price=Decimal(str(row["target_price"])),
                risk_reward_ratio=Decimal("2.0"),
                confidence_score=3,
                reason="Reconstructed from DB",
                timeframe=TimeFrame.ONE_MIN,
                regime=Regime.RANGING,
                timestamp=ts,
            )
            results.append(TradeResult(signal=sig, pnl=Decimal(str(row["pnl"])), timestamp=ts))
        except Exception as exc:
            logger.warning("cast_trade_failed", error=str(exc))
    return results


# ── FastAPI server ─────────────────────────────────────────────────────────────


async def _start_api(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Run the FastAPI app inside the asyncio event loop via uvicorn."""
    from src.api.routes import app as fastapi_app

    config = uvicorn.Config(fastapi_app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    logger.info("api_server_starting", host=host, port=port)
    await server.serve()


# ── graceful shutdown ──────────────────────────────────────────────────────────


def _install_signal_handlers(loop: asyncio.AbstractEventLoop) -> None:
    """Register SIGINT / SIGTERM to cancel all running tasks."""

    def _handle(sig: int) -> None:
        logger.info("shutdown_signal_received", signal=sig)
        for task in asyncio.all_tasks(loop):
            task.cancel()

    for sig in (_signal.SIGINT, _signal.SIGTERM):
        loop.add_signal_handler(sig, _handle, sig)


# ── entry point ────────────────────────────────────────────────────────────────


async def run() -> None:
    """Build the full pipeline and run until cancelled."""
    app_settings = get_app_settings()
    risk_settings = get_risk_settings()
    symbols = app_settings.symbols

    # ── database ─────────────────────────────────────────────────────────────
    db = BarDatabase(db_path=app_settings.db_path)
    db.connect()
    ensure_schema(db.conn)

    # ── per-symbol pipelines ──────────────────────────────────────────────────
    pipelines = _build_pipelines(symbols, db)

    # ── shared components ─────────────────────────────────────────────────────
    cooldown = CooldownTracker()
    risk = RiskManager(cooldown=cooldown, settings=risk_settings)

    telegram_settings = get_telegram_settings()
    alerter = TelegramAlerter(
        bot_token=telegram_settings.bot_token,
        chat_id=telegram_settings.chat_id,
    )
    dispatcher = AlertDispatcher(alerter=alerter)

    # ── inject live state into FastAPI (first symbol's pipeline for dashboard) ─
    from src.api.routes import set_dependencies

    first_pipeline = next(iter(pipelines.values()))
    set_dependencies(
        conn=db.conn,
        registry=first_pipeline.registry,
        levels=first_pipeline.levels,
        regime=first_pipeline.regime,
        cooldown=cooldown,
    )

    # ── WebSocket ingestion queue — all symbols on one stream ─────────────────
    queue: asyncio.Queue[Bar] = asyncio.Queue(maxsize=1000)
    stream = AlpacaBarStream(symbols=symbols, queue=queue, timeframe=TimeFrame.ONE_MIN)

    loop = asyncio.get_running_loop()
    _install_signal_handlers(loop)

    logger.info(
        "platform_starting",
        symbols=symbols,
        mode=app_settings.trading_mode,
        db=app_settings.db_path,
    )

    try:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(_start_api(), name="api_server")
            tg.create_task(stream.start(), name="websocket_stream")
            tg.create_task(
                _bar_loop(queue, pipelines, risk, dispatcher, db),
                name="bar_loop",
            )
            tg.create_task(
                _scheduler(pipelines, cooldown, dispatcher, db),
                name="scheduler",
            )
    except* asyncio.CancelledError:
        logger.info("platform_shutting_down")
    except* Exception as eg:
        for exc in eg.exceptions:
            logger.error("platform_error", error=str(exc))
    finally:
        await stream.stop()
        db.close()
        logger.info("platform_stopped")


def main() -> None:
    """CLI entry point."""
    asyncio.run(run())


if __name__ == "__main__":
    main()
