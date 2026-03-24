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
from dataclasses import dataclass, field
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
    mark_signal_executed,
    query_executed_signals,
    query_recent_signals,
    query_recent_trades,
    update_signal_fill,
    update_signal_outcome,
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

# Excluded weekdays (0=Mon).  Set once at startup via run(), checked in
# _process_bar as a redundant safety net on top of the ORB strategy's own
# excluded-day filter.
_excluded_weekdays: set[int] = set()


# ── per-symbol pipeline container ─────────────────────────────────────────────

_BAR_CACHE_MAX = 400  # slightly more than a full day of 390 bars


@dataclass
class _SymbolPipeline:
    """All per-symbol stateful components for one ticker."""

    symbol: str
    registry: IndicatorRegistry
    levels: LevelManager
    regime: RegimeDetector
    strategy: ORBStrategy
    bar_cache: list[Bar] = field(default_factory=list)


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


def _build_pipelines(
    symbols: list[str],
    db: BarDatabase,
    excluded_days: list[int] | None = None,
) -> dict[str, _SymbolPipeline]:
    """Build one :class:`_SymbolPipeline` per ticker symbol.

    Args:
        symbols:       List of uppercase ticker symbols.
        db:            Connected :class:`BarDatabase` (needed by LevelManager).
        excluded_days: Weekdays to skip (0=Mon). Passed to ORBStrategy.

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
            strategy=ORBStrategy(excluded_days=excluded_days),
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
            except Exception as exc:  # pragma: no cover — Alpaca REST call failures
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
    # 0. Excluded-day gate (redundant safety net — catches Mondays even if
    #    the ORB strategy's own excluded_days check is bypassed or misconfigured).
    #    Uses the module-level _excluded_weekdays set, populated once at startup.
    if _excluded_weekdays:
        bar_date = bar.timestamp.astimezone(_ET).date()
        if bar_date.weekday() in _excluded_weekdays:
            return

    # 1. Indicators
    pipeline.registry.update_all(bar)
    pipeline.bar_cache.append(bar)
    if len(pipeline.bar_cache) > _BAR_CACHE_MAX:
        pipeline.bar_cache = pipeline.bar_cache[-_BAR_CACHE_MAX:]
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
    signal_id = 0
    try:
        signal_id = insert_signal(db.conn, signal, decision)
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
                # 8. Mark signal as executed in DB
                if signal_id > 0:
                    try:
                        mark_signal_executed(db.conn, signal_id, str(order.id))
                    except Exception as exc:
                        logger.error(
                            "signal_mark_executed_failed",
                            signal_id=signal_id,
                            error=str(exc),
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
    *,
    skip_rest_fetch: bool = False,
    executor: AlpacaExecutor | None = None,
) -> None:
    """Send detailed end-of-day status via Telegram."""
    if db.conn is None:  # pragma: no cover — only when DB never connected
        logger.warning("eod_status_skipped", reason="database connection is None")
        return

    since = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)

    # Reconcile Alpaca fills into signal/trades tables before building the report
    if executor is not None:
        try:
            _reconcile_alpaca_fills(executor, db, since)
        except Exception as exc:
            logger.error("eod_reconcile_failed", error=str(exc))

    signals = query_recent_signals(db.conn, since=since, limit=1000)
    trades = query_recent_trades(db.conn, since=since, limit=500)

    total_signals = len(signals)
    approved = sum(1 for s in signals if s.get("approved") == 1)
    rejected = total_signals - approved

    # Execution counts from signals table (executed=1 flag)
    executed_signals = query_executed_signals(db.conn, since=since)
    n_executed = len(executed_signals)
    n_skipped = approved - n_executed

    # Group rejection reasons
    reasons: dict[str, int] = {}
    for s in signals:
        if s.get("approved") == 0:
            reason = str(s.get("reject_reason", "unknown"))
            reasons[reason] = reasons.get(reason, 0) + 1

    # P&L: prefer Alpaca account data for accuracy, fall back to trades table
    actual_pnl: float = 0.0
    if executor is not None:
        try:
            actual_pnl = float(executor.get_daily_pnl())
        except Exception as exc:
            logger.warning("eod_pnl_fetch_failed", error=str(exc))
            actual_pnl = sum(float(t.get("pnl", 0)) for t in trades)
    else:
        actual_pnl = sum(float(t.get("pnl", 0)) for t in trades)

    orb_count = sum(1 for p in pipelines.values() if p.levels._orb.is_complete)
    total_tickers = len(pipelines)
    date_str = datetime.now(_ET).strftime("%A, %B %d %Y")

    lines = [
        "📋 End of Day Report",
        date_str,
        "",
        "📊 Signals",
        f"Generated: {total_signals} | Approved: {approved} | Rejected: {rejected}",
        "",
        "💰 Trading",
        f"Trades: {n_executed} | P&L: ${actual_pnl:+.2f}",
        "",
        "🎯 ORB Status",
        f"Formed: {orb_count}/{total_tickers} tickers",
    ]

    if reasons:
        lines += ["", "🔍 Filter Breakdown"]
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

    # Signal quality analysis — wrapped in try/except so a slow REST fetch
    # or DB error doesn't prevent the rest of the EOD report from sending.
    try:
        outcomes = _evaluate_signal_outcomes(db, pipelines, skip_rest_fetch=skip_rest_fetch)
    except Exception as exc:  # pragma: no cover — DB or REST failure during analysis
        logger.warning("signal_analysis_failed", error=str(exc))
        outcomes = None

    if outcomes is None:  # pragma: no cover — only when analysis fails
        lines += ["", "📈 Signal Quality", "Analysis failed (see logs)"]
        await dispatcher.dispatch_status("\n".join(lines))
        return

    n_winners = len(outcomes["winners"])
    n_losers = len(outcomes["losers"])
    n_open = len(outcomes["open"])
    n_no_data = len(outcomes["no_data"])
    n_evaluated = n_winners + n_losers + n_open + n_no_data

    if n_evaluated > 0:
        all_evaluated = outcomes["winners"] + outcomes["losers"] + outcomes["open"]
        theoretical_pnl = sum(float(s["theoretical_pnl"]) for s in all_evaluated)
        decided = n_winners + n_losers
        win_rate = n_winners / decided * 100 if decided > 0 else 0

        # Direction breakdown
        n_long = sum(1 for s in all_evaluated if s.get("direction") == "LONG")
        n_short = sum(1 for s in all_evaluated if s.get("direction") == "SHORT")

        lines += [
            "",
            "📈 Signal Quality",
            f"Win rate: {win_rate:.0f}% ({n_winners}/{decided})",
            f"Theoretical P&L: ${theoretical_pnl:+.2f}",
            f"Direction: LONG {n_long} | SHORT {n_short}",
        ]
        if n_open > 0:
            lines.append(f"Still open at close: {n_open}")
        if n_no_data > 0:
            lines.append(f"No data: {n_no_data} (cache empty after restart)")
        if win_rate == 0 and decided >= 5:
            lines.append("⚠️ All signals lost — review strategy conditions for this session")

    # Execution comparison (from signals.executed column, not trades table)
    if approved > 0:
        lines += [
            "",
            "💰 Execution",
            f"Executed: {n_executed}/{approved} signals",
        ]
        if n_skipped > 0:
            lines.append(f"Skipped: {n_skipped}")
        lines.append(f"Actual P&L: ${actual_pnl:+.2f}")

    await dispatcher.dispatch_status("\n".join(lines))


# ── Alpaca fill reconciliation ─────────────────────────────────────────────────


def _reconcile_alpaca_fills(
    executor: AlpacaExecutor,
    db: BarDatabase,
    since: datetime,
) -> int:
    """Reconcile Alpaca filled orders back to signal records.

    For each filled order from today, find the matching executed signal
    (by order_id) and update it with fill price, realized P&L, and outcome.
    Also inserts a trade row so the trades table is populated.

    Handles three exit scenarios:
    1. Bracket leg fired (stop/target hit during session)
    2. EOD flatten closed the position (legs cancelled, separate close order)
    3. Position still open (no exit — mark as 'open')

    Returns:
        Number of signals reconciled.
    """
    if db.conn is None:
        return 0

    filled_orders = executor.get_closed_orders_today()
    if not filled_orders:
        logger.debug("reconcile_no_filled_orders")
        return 0

    # Build order_id → filled order lookup
    order_map: dict[str, object] = {}
    for order in filled_orders:
        order_map[str(order.id)] = order

    # Build symbol → list of filled orders (for matching flatten close orders)
    symbol_orders: dict[str, list[object]] = {}
    for order in filled_orders:
        sym = getattr(order, "symbol", "")
        symbol_orders.setdefault(sym, []).append(order)

    # Get executed signals that have an order_id
    executed = query_executed_signals(db.conn, since=since)
    # Collect bracket parent order_ids so we can exclude them when finding
    # flatten orders later.
    bracket_order_ids = {str(sig.get("order_id", "")) for sig in executed if sig.get("order_id")}
    reconciled = 0

    for sig in executed:
        order_id = sig.get("order_id")
        if not order_id or sig.get("realized_pnl") is not None:
            continue  # already reconciled or no order_id

        order = order_map.get(str(order_id))
        if order is None:
            continue  # order not yet filled or not from today

        # Calculate P&L from fill price vs entry
        entry = float(sig.get("entry_price", 0))
        direction = str(sig.get("direction", ""))
        position_size = int(sig.get("position_size", 0))
        stop = float(sig.get("stop_price", 0))
        target = float(sig.get("target_price", 0))
        symbol = str(sig.get("symbol", "SPY"))

        fill_price = float(getattr(order, "filled_avg_price", 0) or 0)
        if fill_price <= 0:
            fill_price = entry

        # Determine exit price from bracket legs or flatten order.
        # For bracket orders the parent fills at entry; P&L comes from
        # whichever child leg fired (stop or target).
        legs = getattr(order, "legs", None) or []
        exit_price: float | None = None
        for leg in legs:
            leg_status = str(getattr(leg, "status", ""))
            if leg_status in ("filled", "OrderStatus.FILLED"):
                leg_fill = float(getattr(leg, "filled_avg_price", 0) or 0)
                if leg_fill > 0:
                    exit_price = leg_fill

        # If no bracket leg filled (e.g. EOD flatten cancelled them), find
        # the flatten close order for this symbol.
        if exit_price is None:
            for candidate in symbol_orders.get(symbol, []):
                cand_id = str(getattr(candidate, "id", ""))
                if cand_id in bracket_order_ids:
                    continue  # skip the bracket parent itself
                # Flatten orders are simple market sells/buys, not brackets
                cand_class = str(getattr(candidate, "order_class", ""))
                if cand_class in ("", "simple", "OrderClass.SIMPLE"):
                    cand_fill = float(getattr(candidate, "filled_avg_price", 0) or 0)
                    if cand_fill > 0:
                        exit_price = cand_fill
                        break

        # If we still have no exit, the position may still be open — use
        # entry as a placeholder (pnl=0, outcome=open).
        if exit_price is None:
            exit_price = fill_price

        if direction == "LONG":
            pnl = (exit_price - entry) * position_size
        else:
            pnl = (entry - exit_price) * position_size

        # Determine outcome
        if exit_price == fill_price and not legs:
            outcome = "open"
        elif abs(exit_price - target) < abs(exit_price - stop):
            outcome = "winner"
        elif abs(exit_price - stop) < abs(exit_price - target):
            outcome = "loser"
        else:
            outcome = "open"

        signal_id = int(sig.get("id", 0))
        if signal_id > 0:
            try:
                update_signal_fill(
                    db.conn,
                    signal_id,
                    str(exit_price),
                    str(round(pnl, 2)),
                    outcome,
                )
            except Exception as exc:
                logger.error(
                    "reconcile_signal_update_failed",
                    signal_id=signal_id,
                    error=str(exc),
                )
                continue

            # Also insert into trades table for backward compat
            try:
                db.conn.execute(
                    """INSERT INTO trades
                       (timestamp, strategy_name, direction, entry_price,
                        stop_price, target_price, pnl, symbol)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        str(sig.get("timestamp", "")),
                        str(sig.get("strategy_name", "")),
                        direction,
                        str(entry),
                        str(stop),
                        str(target),
                        str(round(pnl, 2)),
                        symbol,
                    ),
                )
                db.conn.commit()
            except Exception as exc:
                logger.error(
                    "reconcile_trade_insert_failed",
                    signal_id=signal_id,
                    error=str(exc),
                )

            reconciled += 1

    logger.info(
        "reconcile_complete",
        filled_orders=len(filled_orders),
        executed_signals=len(executed),
        reconciled=reconciled,
    )
    return reconciled


# ── signal outcome evaluation ──────────────────────────────────────────────────


def _evaluate_signal_outcomes(
    db: BarDatabase,
    pipelines: dict[str, _SymbolPipeline],
    *,
    skip_rest_fetch: bool = False,
) -> dict[str, list[dict[str, object]]]:
    """Evaluate today's approved signals against cached bar data.

    For each approved signal, scan post-signal bars to determine if target
    or stop was hit first.

    Returns:
        Dict with keys ``'winners'``, ``'losers'``, ``'open'`` — each a list
        of signal dicts enriched with ``outcome``, ``theoretical_pnl``, and
        ``exit_price``.
    """
    if db.conn is None:  # pragma: no cover — only when DB never connected
        return {"winners": [], "losers": [], "open": [], "no_data": []}

    since = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    signals = query_recent_signals(db.conn, since=since, limit=1000)
    approved_signals = [s for s in signals if s.get("approved") == 1]

    results: dict[str, list[dict[str, object]]] = {
        "winners": [],
        "losers": [],
        "open": [],
        "no_data": [],
    }

    for sig in approved_signals:
        symbol = str(sig.get("symbol", "SPY"))
        direction = str(sig.get("direction", ""))
        entry = float(sig.get("entry_price", 0))
        stop = float(sig.get("stop_price", 0))
        target = float(sig.get("target_price", 0))
        position_size = int(sig.get("position_size", 0))
        signal_ts = str(sig.get("timestamp", ""))
        signal_id = int(sig.get("id", 0))

        if entry <= 0 or stop <= 0 or target <= 0:
            continue

        # Get cached bars for this symbol after signal time
        pipeline = pipelines.get(symbol)
        if pipeline is None:
            # Try first pipeline as default (signals table might not have symbol)
            pipeline = next(iter(pipelines.values()), None)
        if pipeline is None:
            continue

        bars_after = [b for b in pipeline.bar_cache if b.timestamp.isoformat() > signal_ts]

        # If cache is empty (e.g. after restart), try REST fetch —
        # but skip during shutdown to avoid blocking the event loop.
        if not bars_after and not skip_rest_fetch:
            bars_after = _fetch_bars_for_signal(symbol, signal_ts)

        # If still no bars, mark as no_data instead of misleading "open"
        if not bars_after:
            no_data_result: dict[str, object] = {
                **sig,
                "outcome": "no_data",
                "theoretical_pnl": 0.0,
                "exit_price": entry,
            }
            results["no_data"].append(no_data_result)
            if signal_id > 0:
                with contextlib.suppress(Exception):
                    update_signal_outcome(db.conn, signal_id, "no_data")
            continue

        outcome = "open"
        exit_price = entry  # default if still open

        for bar in bars_after:
            bar_high = float(bar.high)
            bar_low = float(bar.low)

            if direction == "LONG":
                if bar_high >= target:
                    outcome = "winner"
                    exit_price = target
                    break
                if bar_low <= stop:
                    outcome = "loser"
                    exit_price = stop
                    break
            elif direction == "SHORT":
                if bar_low <= target:
                    outcome = "winner"
                    exit_price = target
                    break
                if bar_high >= stop:
                    outcome = "loser"
                    exit_price = stop
                    break

        # Calculate P&L
        if direction == "LONG":
            pnl = (exit_price - entry) * position_size
        else:
            pnl = (entry - exit_price) * position_size

        sig_result: dict[str, object] = {
            **sig,
            "outcome": outcome,
            "theoretical_pnl": round(pnl, 2),
            "exit_price": exit_price,
        }
        # Map singular outcome to plural dict key
        bucket = {"winner": "winners", "loser": "losers", "open": "open"}[outcome]
        results[bucket].append(sig_result)

        # Update DB
        if signal_id > 0:
            with contextlib.suppress(Exception):
                update_signal_outcome(db.conn, signal_id, outcome)

    return results


def _fetch_bars_for_signal(
    symbol: str, signal_ts: str
) -> list[Bar]:  # pragma: no cover — Alpaca REST
    """Fetch post-signal bars from Alpaca REST API as a cache-miss fallback.

    Used when the bar cache is empty (e.g. after a restart) so signal
    outcomes can still be evaluated.  Returns an empty list on failure.
    """
    try:
        settings = get_alpaca_settings()
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame as AlpacaTimeFrame

        client = StockHistoricalDataClient(
            api_key=settings.api_key,
            secret_key=settings.secret_key,
        )

        start = datetime.fromisoformat(signal_ts)
        if start.tzinfo is None:
            start = start.replace(tzinfo=UTC)

        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=AlpacaTimeFrame.Minute,
            start=start,
            end=datetime.now(UTC),
            feed=_FEED_MAP.get(settings.feed, DataFeed.IEX),
        )

        response = client.get_stock_bars(request)
        raw_bars = (
            response.data.get(symbol, []) if hasattr(response, "data") else response.get(symbol, [])
        )

        bars: list[Bar] = []
        for raw in raw_bars:
            if any(getattr(raw, f) <= 0 for f in ("open", "high", "low", "close")):
                continue
            ts = raw.timestamp
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=UTC)
            bars.append(
                Bar(
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
            )

        logger.info("signal_bars_fetched", symbol=symbol, bars=len(bars))
        return bars

    except Exception as exc:  # pragma: no cover — Alpaca REST failure
        logger.warning("signal_bars_fetch_failed", symbol=symbol, error=str(exc))
        return []


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
            # Notify on excluded weekdays (e.g. Monday)
            from src.config import get_app_settings as _get_app_settings

            _excluded = set(_get_app_settings().excluded_days)
            if today.weekday() in _excluded:
                _day_names = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 4: "Friday"}
                day_name = _day_names.get(today.weekday(), str(today.weekday()))
                logger.info("excluded_day_no_trading", date=str(today), day=day_name)
                with contextlib.suppress(Exception):
                    await dispatcher.dispatch_status(
                        f"📅 {day_name} — no trading per strategy rules.\n"
                        f"ORB signals suppressed for all {len(pipelines)} tickers."
                    )

            last_reset_date = now_et

        # ── ORB formation watchdog at 9:50 ET (market hours only) ─────────────
        if _ORB_CHECK_ET <= now_et.time() <= time(16, 0) and (
            last_orb_check_date is None or last_orb_check_date.date() < today
        ):
            orb_count = sum(1 for p in pipelines.values() if p.levels._orb.is_complete)
            total = len(pipelines)
            logger.info("orb_formation_check", complete=orb_count, total=total)

            if orb_count == 0:
                with contextlib.suppress(Exception):
                    await dispatcher.dispatch_risk_warning(
                        f"No ORB ranges formed: 0/{total} tickers by 9:50 ET. "
                        f"Check data feed and websocket."
                    )
            elif orb_count < total // 2:
                with contextlib.suppress(Exception):
                    await dispatcher.dispatch_status(
                        f"⚠️ Partial ORB\n{orb_count}/{total} tickers formed ORB by 9:50 ET."
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
                executor.clear_submitted_symbols()
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
            # Send detailed EOD status first — this runs Alpaca reconciliation
            # which populates the trades table with actual fill data.
            try:  # pragma: no cover — fires at 16:05 ET during live session
                await _send_eod_status(
                    dispatcher,
                    db,
                    pipelines,
                    risk_cooldown,
                    executor=executor,
                )
                global _eod_sent_today
                _eod_sent_today = True
            except Exception as exc:  # pragma: no cover — defensive catch for DB/Telegram errors
                logger.error("eod_status_failed", error=str(exc))
            # Daily summary dispatched AFTER reconciliation so trades table has data
            try:
                since = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
                raw_trades = query_recent_trades(db.conn, since=since, limit=500)
                await dispatcher.dispatch_daily_summary(cast_trades(raw_trades))
            except Exception as exc:  # pragma: no cover — DB/Telegram errors at 16:05 ET
                logger.error("eod_summary_failed", error=str(exc))
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
                    f"Websocket stale: no bars received in {int(stream.seconds_since_last_bar)}s during market hours."
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
                f"STARTUP FAILED: Websocket did not connect after {_WS_CONNECT_TIMEOUT}s. "
                f"Check Alpaca keys and network."
            )
        return

    # Connected — send readiness message
    logger.info("scanner_ready", symbols=n_symbols)
    with contextlib.suppress(Exception):
        await dispatcher.dispatch_status(
            f"✅ Scanner Ready\n"
            f"Websocket connected | {n_symbols} symbols\n"
            f"Waiting for market open at 9:30 ET"
        )

    # Wait for first bar — poll until a bar arrives or task is cancelled.
    # We can't use an Event because _on_bar is in a different object; polling
    # every 5s is fine for a one-time startup check.
    try:
        while stream.seconds_since_last_bar is None:
            await asyncio.sleep(5)
    except (
        asyncio.CancelledError
    ):  # pragma: no cover — fires when scanner shuts down during startup
        return

    logger.info("first_bar_received")
    with contextlib.suppress(Exception):
        await dispatcher.dispatch_status("📊 First bar received. ORB window tracking. System GO.")


# ── graceful shutdown ──────────────────────────────────────────────────────────


_shutdown_event: asyncio.Event | None = None
_eod_sent_today: bool = False


def _install_signal_handlers(loop: asyncio.AbstractEventLoop) -> None:
    """Register SIGINT / SIGTERM to set the shutdown event.

    Instead of cancelling all tasks (which cancels the ``run()`` task itself
    and prevents its ``finally`` block from executing), we set an event that
    a watcher task inside the TaskGroup picks up.  The watcher then cancels
    only the sibling tasks, allowing the TaskGroup to exit normally so the
    ``finally`` block runs.
    """
    global _shutdown_event
    _shutdown_event = asyncio.Event()

    def _handle(sig: int) -> None:
        logger.info("shutdown_signal_received", signal=sig)
        if _shutdown_event is not None:
            _shutdown_event.set()

    for sig in (_signal.SIGINT, _signal.SIGTERM):
        loop.add_signal_handler(sig, _handle, sig)


async def _shutdown_watcher(tg_tasks: list[asyncio.Task[None]]) -> None:
    """Wait for the shutdown event, then cancel all sibling tasks."""
    if _shutdown_event is None:
        return  # pragma: no cover — only if signal handlers not installed
    await _shutdown_event.wait()
    logger.info("shutdown_watcher_triggered")
    for task in tg_tasks:
        task.cancel()


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

    # ── excluded-day safety net (module-level cache for _process_bar) ────────
    global _excluded_weekdays
    _excluded_weekdays = set(app_settings.excluded_days)

    # ── per-symbol pipelines ──────────────────────────────────────────────────
    pipelines = _build_pipelines(symbols, db, excluded_days=app_settings.excluded_days)

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
        # Pre-load carry-over positions so the dedup set knows about them
        # BEFORE the first signal cycle fires.
        executor.load_existing_positions()

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
            tasks: list[asyncio.Task[None]] = [
                tg.create_task(_start_api(), name="api_server"),
                tg.create_task(stream.start(), name="websocket_stream"),
                tg.create_task(
                    _bar_loop(queue, pipelines, risk, dispatcher, db, executor=executor),
                    name="bar_loop",
                ),
                tg.create_task(
                    _scheduler(
                        pipelines, cooldown, dispatcher, db, executor=executor, stream=stream
                    ),
                    name="scheduler",
                ),
                tg.create_task(
                    _notify_readiness(stream, dispatcher, len(symbols)),
                    name="readiness_monitor",
                ),
            ]
            tg.create_task(_shutdown_watcher(tasks), name="shutdown_watcher")
    except* (
        asyncio.CancelledError
    ):  # pragma: no cover — coverage.py cannot trace except* (ExceptionGroup syntax)
        logger.info("platform_shutting_down")
    except* Exception as eg:
        for exc in eg.exceptions:
            logger.error("platform_error", error=str(exc))
    finally:
        logger.info("finally_block_entered")
        await stream.stop()
        if not _eod_sent_today:  # pragma: no cover — shutdown-only path
            try:
                await asyncio.wait_for(
                    _send_eod_status(
                        dispatcher,
                        db,
                        pipelines,
                        cooldown,
                        skip_rest_fetch=True,
                        executor=executor,
                    ),
                    timeout=10.0,
                )
            except TimeoutError:
                logger.error("eod_summary_timed_out_on_shutdown")
            except Exception as exc:
                logger.error("eod_summary_failed_on_shutdown", error=str(exc))
        else:  # pragma: no cover — only when 16:05 ET EOD already fired
            logger.info("eod_skipped_on_shutdown", reason="already sent at 16:05 ET")
        with contextlib.suppress(Exception):
            await dispatcher.dispatch_status("\U0001f6d1 Scanner stopped gracefully.")
        db.close()
        logger.info("platform_stopped")


def main() -> None:
    """CLI entry point."""
    asyncio.run(run())


if __name__ == "__main__":  # pragma: no cover — CLI entry point, never runs under pytest
    main()
