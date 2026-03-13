"""Named query functions for the signals and trades database tables.

Schema (created here alongside the existing bars table):

    signals  — one row per Signal produced by a strategy
    trades   — one row per manually-logged TradeResult
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from src.models import RiskDecision, Signal, TradeResult

logger = structlog.get_logger(__name__)


# ── DDL ───────────────────────────────────────────────────────────────────────

_CREATE_SIGNALS_TABLE = """
CREATE TABLE IF NOT EXISTS signals (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp        TEXT    NOT NULL,
    strategy_name    TEXT    NOT NULL,
    direction        TEXT    NOT NULL,
    entry_price      TEXT    NOT NULL,
    stop_price       TEXT    NOT NULL,
    target_price     TEXT    NOT NULL,
    risk_reward      TEXT    NOT NULL,
    confidence       INTEGER NOT NULL,
    reason           TEXT    NOT NULL,
    timeframe        TEXT    NOT NULL,
    regime           TEXT    NOT NULL,
    vix              TEXT,
    adx              TEXT,
    approved         INTEGER NOT NULL,  -- 0/1
    position_size    INTEGER NOT NULL,
    reject_reason    TEXT    NOT NULL
);
"""

_CREATE_TRADES_TABLE = """
CREATE TABLE IF NOT EXISTS trades (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp        TEXT    NOT NULL,
    strategy_name    TEXT    NOT NULL,
    direction        TEXT    NOT NULL,
    entry_price      TEXT    NOT NULL,
    stop_price       TEXT    NOT NULL,
    target_price     TEXT    NOT NULL,
    pnl              TEXT    NOT NULL
);
"""

_CREATE_SIGNALS_INDEX = """
CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signals(timestamp);
"""

_CREATE_TRADES_INDEX = """
CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp);
"""


def ensure_schema(conn: sqlite3.Connection) -> None:
    """Create signals and trades tables if they do not exist."""
    with conn:
        conn.execute(_CREATE_SIGNALS_TABLE)
        conn.execute(_CREATE_TRADES_TABLE)
        conn.execute(_CREATE_SIGNALS_INDEX)
        conn.execute(_CREATE_TRADES_INDEX)
    logger.debug("signal_trade_schema_ensured")


# ── write helpers ─────────────────────────────────────────────────────────────


def insert_signal(
    conn: sqlite3.Connection,
    signal: Signal,
    decision: RiskDecision,
) -> int:
    """Persist a signal + its risk decision.  Returns the new row id."""
    with conn:
        cursor = conn.execute(
            """
            INSERT INTO signals
                (timestamp, strategy_name, direction, entry_price, stop_price,
                 target_price, risk_reward, confidence, reason, timeframe, regime,
                 vix, adx, approved, position_size, reject_reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                signal.timestamp.astimezone(UTC).isoformat(),
                signal.strategy_name,
                str(signal.direction),
                str(signal.entry_price),
                str(signal.stop_price),
                str(signal.target_price),
                str(signal.risk_reward_ratio),
                signal.confidence_score,
                signal.reason,
                signal.timeframe.value,
                str(signal.regime),
                str(signal.vix) if signal.vix is not None else None,
                str(signal.adx) if signal.adx is not None else None,
                1 if decision.approved else 0,
                decision.position_size,
                decision.reason,
            ),
        )
    row_id: int = cursor.lastrowid or 0
    logger.debug("signal_inserted", row_id=row_id, strategy=signal.strategy_name)
    return row_id


def insert_trade(conn: sqlite3.Connection, result: TradeResult) -> int:
    """Persist a completed trade result.  Returns the new row id."""
    sig = result.signal
    with conn:
        cursor = conn.execute(
            """
            INSERT INTO trades
                (timestamp, strategy_name, direction, entry_price,
                 stop_price, target_price, pnl)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result.timestamp.astimezone(UTC).isoformat(),
                sig.strategy_name,
                str(sig.direction),
                str(sig.entry_price),
                str(sig.stop_price),
                str(sig.target_price),
                str(result.pnl),
            ),
        )
    row_id: int = cursor.lastrowid or 0
    logger.debug("trade_inserted", row_id=row_id, pnl=str(result.pnl))
    return row_id


# ── read helpers ──────────────────────────────────────────────────────────────


def query_recent_signals(
    conn: sqlite3.Connection,
    since: datetime,
    limit: int = 100,
) -> list[dict[str, object]]:
    """Return raw signal rows (as dicts) newer than *since*, newest first."""
    rows = conn.execute(
        """
        SELECT * FROM signals
        WHERE timestamp >= ?
        ORDER BY timestamp DESC
        LIMIT ?
        """,
        (since.astimezone(UTC).isoformat(), limit),
    ).fetchall()
    return [dict(row) for row in rows]


def query_recent_trades(
    conn: sqlite3.Connection,
    since: datetime,
    limit: int = 100,
) -> list[dict[str, object]]:
    """Return raw trade rows (as dicts) newer than *since*, newest first."""
    rows = conn.execute(
        """
        SELECT * FROM trades
        WHERE timestamp >= ?
        ORDER BY timestamp DESC
        LIMIT ?
        """,
        (since.astimezone(UTC).isoformat(), limit),
    ).fetchall()
    return [dict(row) for row in rows]
