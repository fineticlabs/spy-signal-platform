"""Main asyncio orchestrator for the SPY Signal Platform.

Startup sequence
----------------
1. Load config from .env
2. Connect SQLite — ensure schema
3. Initialise indicator registry (streaming talipp indicators)
4. Initialise level trackers (ORB, VWAP, PDH/PDL, HOD/LOD)
5. Initialise ORB strategy + regime detector
6. Initialise risk manager (with cooldown tracker)
7. Initialise alert dispatcher (Telegram)
8. Inject live state into FastAPI
9. Start FastAPI server (uvicorn, port 8000) as a background asyncio task
10. Start Alpaca WebSocket stream as a background asyncio task

Main loop
---------
For every bar arriving on the queue:
  - Update indicators
  - Update levels
  - Evaluate strategy → Signal | None
  - If signal: run through RiskManager
  - If approved: dispatch alert
  - Log signal + decision to SQLite

Scheduled tasks
---------------
- Daily reset at 9:30 ET — clear all level/indicator state, reset risk counters
- End-of-day summary at 16:05 ET — dispatch daily P&L summary via Telegram

Graceful shutdown
-----------------
SIGINT / SIGTERM cancel the bar-processing loop and the WebSocket task, then
flush pending state and close the DB connection.
"""

from __future__ import annotations

import asyncio
import signal as _signal
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
_SYMBOL = "SPY"

# Scheduled ET wall-clock times
_DAILY_RESET_ET = time(9, 30)
_EOD_SUMMARY_ET = time(16, 5)

# How often the scheduler wakes (seconds)
_SCHEDULER_INTERVAL = 30


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


# ── bar processing ─────────────────────────────────────────────────────────────


async def _process_bar(
    bar: Bar,
    registry: IndicatorRegistry,
    levels: LevelManager,
    regime: RegimeDetector,
    strategy: ORBStrategy,
    risk: RiskManager,
    dispatcher: AlertDispatcher,
    db: BarDatabase,
) -> None:
    """Run a single bar through the full pipeline."""
    # 1. Indicators
    registry.update_all(bar)
    indicator_snapshot = registry.get_snapshot()

    # 2. Levels
    levels.update(bar)
    level_snapshot = levels.get_levels()

    # Update regime from freshest indicator values (ADX on 15-min is approximated
    # here using the 1-min ATR-proxy until a dedicated 15-min registry is wired).
    # For MVP: caller updates VIX externally; ADX from snapshot if available.
    atr = indicator_snapshot.atr
    if atr is not None:
        # Heuristic: use EMA slope as trending_up proxy (ema9 vs ema20)
        e9 = indicator_snapshot.ema9
        e20 = indicator_snapshot.ema20
        trending_up: bool | None = None
        if e9 is not None and e20 is not None:
            trending_up = e9 > e20

        # ADX not yet computed inline — keep whatever was last set externally.
        # trending_up is updated so regime at least tracks EMA crossover.
        regime.update(trending_up=trending_up)

    # 3. Strategy evaluation
    signal = strategy.evaluate(bar, indicator_snapshot, level_snapshot, regime)

    if signal is None:
        return

    logger.info(
        "signal_generated",
        strategy=signal.strategy_name,
        direction=str(signal.direction),
        entry=str(signal.entry_price),
    )

    # 4. Risk gate
    decision = risk.approve(signal)

    # 5. Persist signal + decision regardless of approval
    try:
        insert_signal(db.conn, signal, decision)
    except Exception as exc:
        logger.error("signal_persist_failed", error=str(exc))

    # 6. Alert if approved
    if decision.approved:
        await dispatcher.dispatch_signal(signal, decision)
    else:
        logger.warning(
            "signal_rejected",
            reason=decision.reason,
            strategy=signal.strategy_name,
        )


# ── bar ingestion loop ─────────────────────────────────────────────────────────


async def _bar_loop(
    queue: asyncio.Queue[Bar],
    registry: IndicatorRegistry,
    levels: LevelManager,
    regime: RegimeDetector,
    strategy: ORBStrategy,
    risk: RiskManager,
    dispatcher: AlertDispatcher,
    db: BarDatabase,
) -> None:
    """Consume bars from the queue and push them through the pipeline."""
    logger.info("bar_loop_started")
    while True:
        bar = await queue.get()
        try:
            await _process_bar(bar, registry, levels, regime, strategy, risk, dispatcher, db)
        except Exception as exc:
            logger.error("bar_processing_error", error=str(exc), bar_ts=bar.timestamp.isoformat())
        finally:
            queue.task_done()


# ── scheduler ─────────────────────────────────────────────────────────────────


async def _scheduler(
    registry: IndicatorRegistry,
    levels: LevelManager,
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
            logger.info("daily_reset", date=str(today))
            risk_cooldown.reset_daily()
            # Rebuild levels (resets ORB, VWAP, HOD/LOD; keeps DB for PDL)
            levels._orb.__init__()  # type: ignore[misc]
            levels._vwap.__init__()  # type: ignore[misc]
            levels._day.__init__()  # type: ignore[misc]
            levels._premarket.__init__()  # type: ignore[misc]
            levels._last_date = None
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

    # ── database ─────────────────────────────────────────────────────────────
    db = BarDatabase(db_path=app_settings.db_path)
    db.connect()
    ensure_schema(db.conn)

    # ── pipeline components ───────────────────────────────────────────────────
    registry = _build_registry()
    levels = LevelManager(db=db, symbol=_SYMBOL)
    regime = RegimeDetector()
    strategy = ORBStrategy()
    cooldown = CooldownTracker()
    risk = RiskManager(cooldown=cooldown, settings=risk_settings)

    # ── alerts ────────────────────────────────────────────────────────────────
    telegram_settings = get_telegram_settings()
    alerter = TelegramAlerter(
        bot_token=telegram_settings.bot_token,
        chat_id=telegram_settings.chat_id,
    )
    dispatcher = AlertDispatcher(alerter=alerter)

    # ── inject live state into FastAPI ────────────────────────────────────────
    from src.api.routes import set_dependencies

    set_dependencies(
        conn=db.conn,
        registry=registry,
        levels=levels,
        regime=regime,
        cooldown=cooldown,
    )

    # ── WebSocket ingestion queue ─────────────────────────────────────────────
    queue: asyncio.Queue[Bar] = asyncio.Queue(maxsize=1000)
    stream = AlpacaBarStream(symbols=[_SYMBOL], queue=queue, timeframe=TimeFrame.ONE_MIN)

    loop = asyncio.get_running_loop()
    _install_signal_handlers(loop)

    logger.info(
        "platform_starting",
        symbol=_SYMBOL,
        mode=app_settings.trading_mode,
        db=app_settings.db_path,
    )

    try:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(_start_api(), name="api_server")
            tg.create_task(stream.start(), name="websocket_stream")
            tg.create_task(
                _bar_loop(queue, registry, levels, regime, strategy, risk, dispatcher, db),
                name="bar_loop",
            )
            tg.create_task(
                _scheduler(registry, levels, cooldown, dispatcher, db),
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
