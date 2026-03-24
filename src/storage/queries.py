"""Named query functions for the signals and trades database tables.

Schema (created here alongside the existing bars table):

    signals  — one row per Signal produced by a strategy
    trades   — one row per manually-logged TradeResult
"""

from __future__ import annotations

import contextlib
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

    # Migrate: add outcome column if it doesn't exist
    with contextlib.suppress(sqlite3.OperationalError):
        conn.execute("ALTER TABLE signals ADD COLUMN outcome TEXT DEFAULT NULL")

    # Migrate: add symbol column if it doesn't exist
    with contextlib.suppress(sqlite3.OperationalError):
        conn.execute("ALTER TABLE signals ADD COLUMN symbol TEXT DEFAULT 'SPY'")

    # Migrate: add execution tracking columns
    with contextlib.suppress(sqlite3.OperationalError):
        conn.execute("ALTER TABLE signals ADD COLUMN order_id TEXT DEFAULT NULL")
    with contextlib.suppress(sqlite3.OperationalError):
        conn.execute("ALTER TABLE signals ADD COLUMN executed INTEGER DEFAULT 0")
    with contextlib.suppress(sqlite3.OperationalError):
        conn.execute("ALTER TABLE signals ADD COLUMN fill_price TEXT DEFAULT NULL")
    with contextlib.suppress(sqlite3.OperationalError):
        conn.execute("ALTER TABLE signals ADD COLUMN realized_pnl TEXT DEFAULT NULL")

    # Migrate: add symbol column to trades if it doesn't exist
    with contextlib.suppress(sqlite3.OperationalError):
        conn.execute("ALTER TABLE trades ADD COLUMN symbol TEXT DEFAULT 'SPY'")

    # Migrate: create cooldown_state table for crash-persistent cooldown tracking
    conn.execute("""
        CREATE TABLE IF NOT EXISTS cooldown_state (
            id                 INTEGER PRIMARY KEY CHECK (id = 1),
            consecutive_losses INTEGER NOT NULL DEFAULT 0,
            daily_pnl          TEXT    NOT NULL DEFAULT '0',
            daily_trade_count  INTEGER NOT NULL DEFAULT 0,
            last_loss_time     TEXT,
            session_date       TEXT    NOT NULL
        )
    """)

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
                 vix, adx, approved, position_size, reject_reason, symbol)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                signal.symbol,
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


def mark_signal_executed(
    conn: sqlite3.Connection,
    signal_id: int,
    order_id: str,
) -> None:
    """Mark a signal as executed with its Alpaca order ID.

    Args:
        conn: Active SQLite connection.
        signal_id: Primary key of the signal row.
        order_id: Alpaca order ID from the submitted bracket order.
    """
    with conn:
        conn.execute(
            "UPDATE signals SET executed = 1, order_id = ? WHERE id = ?",
            (order_id, signal_id),
        )
    logger.debug("signal_marked_executed", signal_id=signal_id, order_id=order_id)


def update_signal_fill(
    conn: sqlite3.Connection,
    signal_id: int,
    fill_price: str,
    realized_pnl: str,
    outcome: str,
) -> None:
    """Update a signal with actual fill data from Alpaca reconciliation.

    Args:
        conn: Active SQLite connection.
        signal_id: Primary key of the signal row.
        fill_price: Actual fill/exit price.
        realized_pnl: Realized P&L from the trade.
        outcome: One of ``'winner'``, ``'loser'``, ``'open'``, ``'stopped'``.
    """
    with conn:
        conn.execute(
            """UPDATE signals
               SET fill_price = ?, realized_pnl = ?, outcome = ?
               WHERE id = ?""",
            (fill_price, realized_pnl, outcome, signal_id),
        )
    logger.debug(
        "signal_fill_updated",
        signal_id=signal_id,
        fill_price=fill_price,
        realized_pnl=realized_pnl,
        outcome=outcome,
    )


def query_executed_signals(
    conn: sqlite3.Connection,
    since: datetime,
) -> list[dict[str, object]]:
    """Return executed signal rows (approved=1, executed=1) newer than *since*."""
    rows = conn.execute(
        """
        SELECT * FROM signals
        WHERE timestamp >= ? AND approved = 1 AND executed = 1
        ORDER BY timestamp DESC
        """,
        (since.astimezone(UTC).isoformat(),),
    ).fetchall()
    return [dict(row) for row in rows]


def save_cooldown_state(
    conn: sqlite3.Connection,
    consecutive_losses: int,
    daily_pnl: str,
    daily_trade_count: int,
    last_loss_time: str | None,
    session_date: str,
) -> None:
    """Persist cooldown state to DB (upsert single row)."""
    with conn:
        conn.execute(
            """INSERT INTO cooldown_state (id, consecutive_losses, daily_pnl, daily_trade_count, last_loss_time, session_date)
               VALUES (1, ?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                 consecutive_losses = excluded.consecutive_losses,
                 daily_pnl = excluded.daily_pnl,
                 daily_trade_count = excluded.daily_trade_count,
                 last_loss_time = excluded.last_loss_time,
                 session_date = excluded.session_date""",
            (consecutive_losses, daily_pnl, daily_trade_count, last_loss_time, session_date),
        )


def load_cooldown_state(
    conn: sqlite3.Connection,
    session_date: str,
) -> dict[str, object] | None:
    """Load cooldown state from DB if it matches the current session date.

    Returns None if no state exists or the stored state is from a different day.
    """
    row = conn.execute(
        "SELECT * FROM cooldown_state WHERE id = 1 AND session_date = ?",
        (session_date,),
    ).fetchone()
    if row is None:
        return None
    return dict(row)


def update_signal_outcome(conn: sqlite3.Connection, signal_id: int, outcome: str) -> None:
    """Update the outcome for a signal.

    Args:
        conn: Active SQLite connection.
        signal_id: Primary key of the signal row.
        outcome: One of ``'winner'``, ``'loser'``, ``'open'``, ``'skipped'``.
    """
    with conn:
        conn.execute("UPDATE signals SET outcome = ? WHERE id = ?", (outcome, signal_id))
