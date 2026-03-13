"""FastAPI REST endpoints for the dashboard and manual trade logging.

Endpoints
---------
GET  /health          System status (always 200 if running).
GET  /state           Current indicators, levels, regime, and risk state.
GET  /signals         Recent signals (last 24 h, newest first).
GET  /trades          Recent trades (last 30 days, newest first).
POST /trades          Manually log a completed trade result.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import structlog
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.models import Direction, Regime, Signal, TimeFrame, TradeResult

if TYPE_CHECKING:
    import sqlite3

    from src.indicators.registry import IndicatorRegistry
    from src.levels import LevelManager
    from src.models import IndicatorSnapshot, LevelSnapshot
    from src.risk.cooldown import CooldownTracker
    from src.strategies.regime import RegimeDetector

logger = structlog.get_logger(__name__)

app = FastAPI(title="SPY Signal Platform", version="0.1.0")

# ── mutable app-level state (injected by main.py at startup) ──────────────────
# These are replaced by main.py after it builds the pipeline components.

_conn: sqlite3.Connection | None = None
_registry: IndicatorRegistry | None = None
_levels: LevelManager | None = None
_regime: RegimeDetector | None = None
_cooldown: CooldownTracker | None = None


def set_dependencies(
    conn: sqlite3.Connection,
    registry: IndicatorRegistry,
    levels: LevelManager,
    regime: RegimeDetector,
    cooldown: CooldownTracker,
) -> None:
    """Called once at startup to inject live pipeline state into the API."""
    global _conn, _registry, _levels, _regime, _cooldown
    _conn = conn
    _registry = registry
    _levels = levels
    _regime = regime
    _cooldown = cooldown
    logger.info("api_dependencies_set")


# ── request / response models ─────────────────────────────────────────────────


class TradeLogRequest(BaseModel):
    """Payload for POST /trades — manually record a completed trade."""

    strategy_name: str = Field(..., description="Name of the strategy that generated the signal")
    direction: Direction = Field(..., description="LONG or SHORT")
    entry_price: Decimal = Field(..., description="Actual entry price")
    stop_price: Decimal = Field(..., description="Initial stop price")
    target_price: Decimal = Field(..., description="Initial target price")
    pnl: Decimal = Field(..., description="Realized P&L in USD (negative = loss)")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When the trade was closed (UTC)",
    )


# ── helpers ───────────────────────────────────────────────────────────────────


def _require_conn() -> sqlite3.Connection:
    if _conn is None:
        raise HTTPException(status_code=503, detail="Database not connected")
    return _conn


# ── endpoints ─────────────────────────────────────────────────────────────────


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe — always returns 200 while the process is running."""
    return {"status": "ok", "timestamp": datetime.now(UTC).isoformat()}


@app.get("/state")
def state() -> dict[str, Any]:
    """Return a snapshot of all live pipeline state."""
    ind: IndicatorSnapshot | None = _registry.get_snapshot() if _registry is not None else None
    lvl: LevelSnapshot | None = _levels.get_levels() if _levels is not None else None

    regime_name: str | None = None
    vix: str | None = None
    adx: str | None = None
    tradeable: bool = False
    if _regime is not None:
        regime_name = str(_regime.current_regime)
        vix = str(_regime.vix_level) if _regime.vix_level is not None else None
        adx = str(_regime.adx_value) if _regime.adx_value is not None else None
        tradeable = _regime.is_tradeable

    risk: dict[str, Any] = {}
    if _cooldown is not None:
        risk = {
            "daily_pnl": str(_cooldown.daily_pnl),
            "daily_trades": _cooldown.daily_trade_count,
            "consecutive_losses": _cooldown.consecutive_losses,
            "cooled_down": _cooldown.is_cooled_down(),
            "tilted": _cooldown.is_tilted(),
        }

    return {
        "indicators": ind.model_dump() if ind is not None else None,
        "levels": lvl.model_dump() if lvl is not None else None,
        "regime": {
            "name": regime_name,
            "vix": vix,
            "adx": adx,
            "tradeable": tradeable,
        },
        "risk": risk,
    }


@app.get("/signals")
def signals(hours: int = 24, limit: int = 100) -> list[dict[str, Any]]:
    """Return recent signals, newest first.

    Query params:
        hours: look-back window in hours (default 24).
        limit: max rows returned (default 100, max 500).
    """
    conn = _require_conn()
    limit = min(limit, 500)
    since = datetime.now(UTC) - timedelta(hours=hours)
    from src.storage.queries import query_recent_signals

    return query_recent_signals(conn, since=since, limit=limit)


@app.get("/trades")
def trades(days: int = 30, limit: int = 100) -> list[dict[str, Any]]:
    """Return recent trade results, newest first.

    Query params:
        days:  look-back window in days (default 30).
        limit: max rows returned (default 100, max 500).
    """
    conn = _require_conn()
    limit = min(limit, 500)
    since = datetime.now(UTC) - timedelta(days=days)
    from src.storage.queries import query_recent_trades

    return query_recent_trades(conn, since=since, limit=limit)


@app.post("/trades", status_code=201)
def log_trade(body: TradeLogRequest) -> dict[str, Any]:
    """Manually record a completed trade result.

    This updates the cooldown tracker and persists the trade to SQLite.
    """
    conn = _require_conn()

    # Build a minimal Signal so we can store a proper TradeResult.
    ts = body.timestamp if body.timestamp.tzinfo is not None else body.timestamp.replace(tzinfo=UTC)
    signal = Signal(
        direction=body.direction,
        strategy_name=body.strategy_name,
        entry_price=body.entry_price,
        stop_price=body.stop_price,
        target_price=body.target_price,
        risk_reward_ratio=(
            abs(body.target_price - body.entry_price) / abs(body.stop_price - body.entry_price)
            if body.stop_price != body.entry_price
            else Decimal("0")
        ),
        confidence_score=3,
        reason="Manually logged via API",
        timeframe=TimeFrame.ONE_MIN,
        regime=Regime.RANGING,
        timestamp=ts,
    )
    result = TradeResult(signal=signal, pnl=body.pnl, timestamp=ts)

    # Update risk cooldown
    if _cooldown is not None:
        _cooldown.record_trade(result)

    # Persist
    from src.storage.queries import insert_trade

    row_id = insert_trade(conn, result)
    logger.info("trade_logged_via_api", row_id=row_id, pnl=str(body.pnl))
    return {"id": row_id, "pnl": str(result.pnl)}
