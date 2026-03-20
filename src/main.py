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
import contextlib
import signal as _signal
from dataclasses import dataclass
from datetime import UTC, datetime, time
from decimal import Decimal
from zoneinfo import ZoneInfo

import structlog
import uvicorn
from alpaca.data.enums import DataFeed

from src.alerts.dispatcher import AlertDispatcher
from src.alerts.telegram import TelegramAlerter
from src.config import (
    get_alpaca_settings,
    get_app_settings,
    get_risk_settings,
    get_telegram_settings,
)
from src.execution.alpaca_executor import AlpacaExecutor
from src.indicators.registry import IndicatorRegistry
from src.indicators.streaming import (
    StreamingADX,
    StreamingATR,
    StreamingEMA,
    StreamingMACD,
    StreamingRSI,
)
from src.ingestion.websocket import _FEED_MAP, AlpacaBarStream
from src.levels import LevelManager
from src.models import Bar, TimeFrame, TradeResult
from src.risk.cooldown import CooldownTracker
from src.risk.manager import RiskManager
from src.storage.database import BarDatabase
from src.storage.queries import (
    ensure_schema,
    insert_signal,
    query_recent_signals,
    query_recent_trades,
)
from src.strategies.orb import ORBStrategy
from src.strategies.regime import RegimeDetector

logger = structlog.get_logger(__name__)

_ET = ZoneInfo("America/New_York")

# Scheduled ET wall-clock times
_DAILY_RESET_ET = time(9, 30)
_ORB_CHECK_ET = time(9, 50)
_EOD_FLATTEN_ET = time(15, 55)
_EOD_SUMMARY_ET = time(16, 5)

# How often the scheduler wakes (seconds)
_SCHEDULER_INTERVAL = 30

# ── regime warmup progress tracking ──────────────────────────────────────────
_regime_warmup_bar_counts: dict[str, int] = {}
_regime_first_populated: set[str] = set()


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
    registry.register("adx", StreamingADX(14))
    return registry


# Fallback VIX when no live VIX feed is available.  20 = long-run median;
# the ORB strategy only needs VIX < 25 to allow signals, so 20 keeps the
# gate open while avoiding false confidence during actual high-vol regimes.
# Replace with a real feed (e.g. from vix_data.download_vix) when available.
_VIX_FALLBACK = Decimal("20")


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


# ── ORB backfill on late start ─────────────────────────────────────────────


async def _backfill_orb_bars(
    pipelines: dict[str, _SymbolPipeline],
) -> None:
    """Backfill missed ORB bars from Alpaca REST API on late start.

    If the scanner starts after 9:35 ET, the 5-min ORB window (9:30-9:34)
    was missed on the websocket.  This fetches those bars via REST and feeds
    them into each pipeline's level tracker so the ORB can still form.

    Similarly backfills up to 9:45 for the 15-min ORB if needed.
    """
    now_et = datetime.now(_ET)
    session_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)

    # Only backfill if we're past the ORB window and it's still the same trading day
    if now_et.time() < time(9, 35) or now_et.time() > time(
        16, 0
    ):  # pragma: no cover — tested via mock datetime
        return

    logger.info(
        "orb_backfill_check",
        now_et=now_et.strftime("%H:%M:%S"),
        msg="Scanner started after ORB window, attempting REST backfill",
    )

    try:
        settings = get_alpaca_settings()
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame as AlpacaTimeFrame

        client = StockHistoricalDataClient(
            api_key=settings.api_key,
            secret_key=settings.secret_key,
        )

        # Fetch bars from session open to now (covers both 5-min and 15-min ORB)
        backfill_end = min(now_et, now_et.replace(hour=9, minute=45, second=0))
        start_utc = session_open.astimezone(UTC)
        end_utc = backfill_end.astimezone(UTC)

        symbols = list(pipelines.keys())
        feed = _FEED_MAP.get(settings.feed, DataFeed.IEX)

        for symbol in symbols:
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=AlpacaTimeFrame.Minute,
                start=start_utc,
                end=end_utc,
                feed=feed,
            )

            try:
                response = client.get_stock_bars(request)
                raw_bars = (
                    response.data.get(symbol, [])
                    if hasattr(response, "data")
                    else response.get(symbol, [])
                )

                count = 0
                for raw in raw_bars:
                    if any(
                        getattr(raw, f) <= 0 for f in ("open", "high", "low", "close")
                    ):  # pragma: no cover — Alpaca rarely returns zero-price bars
                        continue
                    ts = raw.timestamp
                    if ts.tzinfo is None:  # pragma: no cover — Alpaca REST always returns tz-aware
                        ts = ts.replace(tzinfo=UTC)
                    bar = Bar(
                        symbol=symbol,
                        timeframe=TimeFrame.ONE_MIN,
                        timestamp=ts,
                        open=Decimal(str(raw.open)),
                        high=Decimal(str(raw.high)),
                        low=Decimal(str(raw.low)),
                        close=Decimal(str(raw.close)),
                        volume=int(raw.volume),
                        vwap=Decimal(str(raw.vwap))
                        if raw.vwap is not None
                        else Decimal(str(raw.close)),
                    )
                    # Feed into levels only (not indicators — those warm up from live bars)
                    pipelines[symbol].levels.update(bar)
                    count += 1

                logger.info(
                    "orb_backfill_complete",
                    symbol=symbol,
                    bars=count,
                    orb_complete=pipelines[symbol].levels._orb.is_complete,
                )
            except Exception as exc:
                logger.warning("orb_backfill_failed", symbol=symbol, error=str(exc))

    except Exception as exc:  # pragma: no cover — catches Alpaca SDK import/auth failures
        logger.error("orb_backfill_error", error=str(exc))


# ── bar processing ─────────────────────────────────────────────────────────────


async def _process_bar(
    bar: Bar,
    pipeline: _SymbolPipeline,
    risk: RiskManager,
    dispatcher: AlertDispatcher,
    db: BarDatabase,
    executor: AlpacaExecutor | None = None,
) -> None:
    """Run a single bar through the full pipeline for its symbol."""
    # 1. Indicators
    pipeline.registry.update_all(bar)
    indicator_snapshot = pipeline.registry.get_snapshot()

    # Track regime warmup progress
    symbol = bar.symbol
    _regime_warmup_bar_counts[symbol] = _regime_warmup_bar_counts.get(symbol, 0) + 1
    bar_count = _regime_warmup_bar_counts[symbol]

    if bar_count % 5 == 0 and indicator_snapshot.adx is None:
        logger.info(
            "regime_warmup_progress",
            symbol=symbol,
            bars=bar_count,
            adx_ready=indicator_snapshot.adx is not None,
            atr_ready=indicator_snapshot.atr is not None,
        )

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

        # ADX from the snapshot (StreamingADX registered as "adx")
        adx_val = indicator_snapshot.adx

        # VIX: use fallback until a real-time feed is wired in
        vix_val: Decimal | None = pipeline.regime.vix_level or _VIX_FALLBACK

        pipeline.regime.update(vix=vix_val, adx=adx_val, trending_up=trending_up)

    # Log when regime first populates for this symbol
    if (
        symbol not in _regime_first_populated
        and pipeline.regime.vix_level is not None
        and pipeline.regime.adx_value is not None
    ):
        _regime_first_populated.add(symbol)
        logger.info(
            "regime_first_update",
            symbol=symbol,
            vix=str(pipeline.regime.vix_level),
            adx=str(pipeline.regime.adx_value),
            regime=str(pipeline.regime.current_regime),
            bars_to_warmup=bar_count,
        )

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

        # 7. Execute order if executor is configured
        if executor is not None:
            order = executor.submit_bracket_order(
                symbol=signal.symbol,
                direction=signal.direction,
                qty=decision.position_size,
                stop_price=signal.stop_price,
                target_price=signal.target_price,
            )
            if order is not None:
                logger.info(
                    "order_executed",
                    symbol=signal.symbol,
                    order_id=str(order.id),
                )
    else:
        logger.warning(
            "signal_rejected",
            symbol=bar.symbol,
            reason=decision.reason,
            strategy=signal.strategy_name,
        )


# ── EOD status report ─────────────────────────────────────────────────────────


async def _send_eod_status(
    dispatcher: AlertDispatcher,
    db: BarDatabase,
    pipelines: dict[str, _SymbolPipeline],
    cooldown: CooldownTracker,
) -> None:
    """Send detailed end-of-day status via Telegram."""
    if db.conn is None:  # pragma: no cover — only when DB never connected
        logger.warning("eod_status_skipped", reason="database connection is None")
        return

    since = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)

    signals = query_recent_signals(db.conn, since=since, limit=1000)
    trades = query_recent_trades(db.conn, since=since, limit=500)

    total_signals = len(signals)
    approved = sum(1 for s in signals if s.get("approved") == 1)
    rejected = total_signals - approved

    # Group rejection reasons
    reasons: dict[str, int] = {}
    for s in signals:
        if s.get("approved") == 0:
            reason = str(s.get("reject_reason", "unknown"))
            reasons[reason] = reasons.get(reason, 0) + 1

    total_pnl = sum(float(t.get("pnl", 0)) for t in trades)
    orb_count = sum(1 for p in pipelines.values() if p.levels._orb.is_complete)
    total_tickers = len(pipelines)
    date_str = datetime.now(_ET).strftime("%A, %B %d %Y")

    lines = [
        "📋 *End of Day Report*",
        date_str,
        "",
        "📊 *Signals*",
        f"Generated: {total_signals} | Approved: {approved} | Rejected: {rejected}",
        "",
        "💰 *Trading*",
        f"Trades: {len(trades)} | P&L: ${total_pnl:+.2f}",
        "",
        "🎯 *ORB Status*",
        f"Formed: {orb_count}/{total_tickers} tickers",
    ]

    if reasons:
        lines += ["", "🔍 *Filter Breakdown*"]
        for reason_text, count in sorted(reasons.items(), key=lambda x: -x[1])[:5]:
            lines.append(f"  {reason_text}: {count}")

    if approved == 0:
        top_reason = (
            next(iter(sorted(reasons, key=reasons.get, reverse=True)), None) if reasons else None
        )  # type: ignore[arg-type]
        if top_reason:
            lines += ["", f"⚠️ No trades today — top filter: {top_reason}."]
        else:
            lines += ["", "⚠️ No signals today — no breakout conditions met."]

    await dispatcher.dispatch_status("\n".join(lines))


# ── bar ingestion loop ─────────────────────────────────────────────────────────


async def _bar_loop(
    queue: asyncio.Queue[Bar],
    pipelines: dict[str, _SymbolPipeline],
    risk: RiskManager,
    dispatcher: AlertDispatcher,
    db: BarDatabase,
    executor: AlpacaExecutor | None = None,
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
            await _process_bar(bar, pipeline, risk, dispatcher, db, executor=executor)
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
    executor: AlpacaExecutor | None = None,
    stream: AlpacaBarStream | None = None,
) -> None:
    """Fire periodic events: daily reset, ORB check, EOD flatten, EOD summary, websocket watchdog."""
    last_reset_date: datetime | None = None
    last_orb_check_date: datetime | None = None
    last_flatten_date: datetime | None = None
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
                # Reset per-symbol level state — but preserve ORB if already
                # complete for today (backfill may have populated it).
                orb = pipeline.levels._orb
                if not orb.is_complete or orb._session_date != today:
                    orb.__init__()  # type: ignore[misc]
                pipeline.levels._vwap.__init__()  # type: ignore[misc]
                pipeline.levels._day.__init__()  # type: ignore[misc]
                pipeline.levels._premarket.__init__()  # type: ignore[misc]
                pipeline.levels._last_date = None
            last_reset_date = now_et

        # ── ORB formation watchdog at 9:50 ET ────────────────────────────────
        if now_et.time() >= _ORB_CHECK_ET and (
            last_orb_check_date is None or last_orb_check_date.date() < today
        ):
            orb_count = sum(1 for p in pipelines.values() if p.levels._orb.is_complete)
            total = len(pipelines)
            logger.info("orb_formation_check", complete=orb_count, total=total)

            if orb_count == 0:
                with contextlib.suppress(Exception):
                    await dispatcher.dispatch_risk_warning(
                        f"🚨 *No ORB Ranges*\n"
                        f"0/{total} tickers formed an ORB by 9:50 ET.\n"
                        f"Action: Check data feed and websocket."
                    )
            elif orb_count < total // 2:
                with contextlib.suppress(Exception):
                    await dispatcher.dispatch_status(
                        f"⚠️ *Partial ORB*\n" f"{orb_count}/{total} tickers formed ORB by 9:50 ET."
                    )
            last_orb_check_date = now_et

        # ── EOD flatten at 15:55 ET ──────────────────────────────────────────
        if (
            executor is not None
            and now_et.time() >= _EOD_FLATTEN_ET
            and (last_flatten_date is None or last_flatten_date.date() < today)
        ):
            logger.info("eod_flatten_starting", date=str(today))
            try:
                cancelled = executor.cancel_open_orders()
                closed = executor.flatten_all_positions()
                logger.info(
                    "eod_flatten_complete",
                    orders_cancelled=cancelled,
                    positions_closed=closed,
                )
            except Exception as exc:
                logger.error("eod_flatten_failed", error=str(exc))
            last_flatten_date = now_et

        # ── end-of-day summary ───────────────────────────────────────────────
        if now_et.time() >= _EOD_SUMMARY_ET and (
            last_eod_date is None or last_eod_date.date() < today
        ):
            logger.info("eod_summary", date=str(today))
            try:
                since = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
                raw_trades = query_recent_trades(db.conn, since=since, limit=500)
                await dispatcher.dispatch_daily_summary(cast_trades(raw_trades))
            except Exception as exc:  # pragma: no cover — DB/Telegram errors at 16:05 ET
                logger.error("eod_summary_failed", error=str(exc))
            try:  # pragma: no cover — fires at 16:05 ET during live session
                await _send_eod_status(dispatcher, db, pipelines, risk_cooldown)
            except Exception as exc:  # pragma: no cover — defensive catch for DB/Telegram errors
                logger.error("eod_status_failed", error=str(exc))
            last_eod_date = now_et

        # ── websocket watchdog: alert if no bars in 90s during market hours ──
        if (
            time(9, 30) <= now_et.time() <= time(16, 0)
            and stream is not None
            and stream.seconds_since_last_bar is not None
            and stream.seconds_since_last_bar > 90
        ):
            logger.error(  # pragma: no cover — only fires during live market hours with stale WS
                "websocket_watchdog_alert",
                seconds_since_last_bar=round(stream.seconds_since_last_bar),
            )
            with contextlib.suppress(Exception):  # pragma: no cover
                await dispatcher.dispatch_risk_warning(
                    f"🚨 *Websocket Stale*\n"
                    f"No bars received in {int(stream.seconds_since_last_bar)}s during market hours."
                )


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


# ── startup readiness notifications ──────────────────────────────────────────

_WS_CONNECT_TIMEOUT = 90  # seconds to wait for websocket auth


async def _notify_readiness(
    stream: AlpacaBarStream,
    dispatcher: AlertDispatcher,
    n_symbols: int,
) -> None:
    """Await websocket connection and send Telegram readiness notifications.

    1. Wait up to 90s for websocket auth+subscribe.
    2. Send "Scanner ready" Telegram message.
    3. Wait for first bar_received, then send "System GO".

    If websocket doesn't connect in time, sends a failure alert instead.
    """
    # Wait for connection
    try:
        await asyncio.wait_for(stream.connected.wait(), timeout=_WS_CONNECT_TIMEOUT)
        await asyncio.wait_for(stream.subscribed.wait(), timeout=30)
    except TimeoutError:
        logger.error(
            "websocket_startup_timeout",
            timeout=_WS_CONNECT_TIMEOUT,
            msg="Websocket did not authenticate within timeout",
        )
        with contextlib.suppress(Exception):
            await dispatcher.dispatch_risk_warning(
                f"🚨 *STARTUP FAILED*\n"
                f"Websocket did not connect after {_WS_CONNECT_TIMEOUT}s.\n"
                f"Action: Check Alpaca keys and network."
            )
        return

    # Connected — send readiness message
    logger.info("scanner_ready", symbols=n_symbols)
    with contextlib.suppress(Exception):
        await dispatcher.dispatch_status(
            f"✅ *Scanner Ready*\n"
            f"Websocket connected | {n_symbols} symbols\n"
            f"Waiting for market open at 9:30 ET"
        )

    # Wait for first bar — poll until a bar arrives or task is cancelled.
    # We can't use an Event because _on_bar is in a different object; polling
    # every 5s is fine for a one-time startup check.
    try:
        while stream.seconds_since_last_bar is None:
            await asyncio.sleep(5)  # one-time startup poll, not a hot loop
    except (
        asyncio.CancelledError
    ):  # pragma: no cover — fires when scanner shuts down during startup
        return

    logger.info("first_bar_received")
    with contextlib.suppress(Exception):
        await dispatcher.dispatch_status("📊 First bar received. ORB window tracking. System GO.")


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

    # ── backfill missed ORB bars on late start ─────────────────────────────
    await _backfill_orb_bars(pipelines)

    # ── shared components ─────────────────────────────────────────────────────
    cooldown = CooldownTracker()
    risk = RiskManager(cooldown=cooldown, settings=risk_settings)

    telegram_settings = get_telegram_settings()
    alerter = TelegramAlerter(
        bot_token=telegram_settings.bot_token,
        chat_id=telegram_settings.chat_id,
        account_size=risk_settings.account_size,
        risk_pct=risk_settings.risk_per_trade_pct,
        max_concurrent_positions=risk_settings.max_concurrent_positions,
    )
    dispatcher = AlertDispatcher(alerter=alerter)

    # ── executor (paper_trade / live_trade mode only) ─────────────────────────
    executor: AlpacaExecutor | None = None
    if app_settings.execution_mode in ("paper_trade", "live_trade"):
        alpaca_settings = get_alpaca_settings()
        is_paper = app_settings.execution_mode == "paper_trade"
        executor = AlpacaExecutor(
            api_key=alpaca_settings.api_key,
            secret_key=alpaca_settings.secret_key,
            paper=is_paper,
        )
        logger.info(
            "executor_enabled",
            mode=app_settings.execution_mode,
            paper=is_paper,
        )

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

    async def _ws_failure_callback(
        message: str,
    ) -> None:  # pragma: no cover — only fires after 3 WS failures during live run
        """Send Telegram alert when websocket exhausts all retries."""
        with contextlib.suppress(Exception):
            await dispatcher.dispatch_risk_warning(f"🚨 CRITICAL: {message}")

    stream = AlpacaBarStream(
        symbols=symbols,
        queue=queue,
        timeframe=TimeFrame.ONE_MIN,
        on_failure=_ws_failure_callback,
    )

    loop = asyncio.get_running_loop()
    _install_signal_handlers(loop)

    logger.info(
        "platform_starting",
        symbols=symbols,
        mode=app_settings.trading_mode,
        execution_mode=app_settings.execution_mode,
        db=app_settings.db_path,
    )

    try:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(_start_api(), name="api_server")
            tg.create_task(stream.start(), name="websocket_stream")
            tg.create_task(
                _bar_loop(queue, pipelines, risk, dispatcher, db, executor=executor),
                name="bar_loop",
            )
            tg.create_task(
                _scheduler(pipelines, cooldown, dispatcher, db, executor=executor, stream=stream),
                name="scheduler",
            )
            tg.create_task(
                _notify_readiness(stream, dispatcher, len(symbols)),
                name="readiness_monitor",
            )
    except* (
        asyncio.CancelledError
    ):  # pragma: no cover — coverage.py cannot trace except* (ExceptionGroup syntax)
        logger.info("platform_shutting_down")
    except* Exception as eg:
        for exc in eg.exceptions:
            logger.error("platform_error", error=str(exc))
    finally:
        await stream.stop()
        try:
            await _send_eod_status(dispatcher, db, pipelines, cooldown)
        except Exception as exc:
            logger.error("eod_summary_failed_on_shutdown", error=str(exc))
        with contextlib.suppress(Exception):
            await dispatcher.dispatch_status("\U0001f6d1 Scanner stopped gracefully.")
        db.close()
        logger.info("platform_stopped")


def main() -> None:
    """CLI entry point."""
    asyncio.run(run())


if __name__ == "__main__":  # pragma: no cover — CLI entry point, never runs under pytest
    main()
