from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import structlog

from src.config import get_app_settings
from src.models import Bar, TimeFrame

logger = structlog.get_logger(__name__)

_CREATE_BARS_TABLE = """
CREATE TABLE IF NOT EXISTS bars (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol      TEXT    NOT NULL,
    timeframe   TEXT    NOT NULL,
    timestamp   TEXT    NOT NULL,   -- ISO-8601 UTC
    open        TEXT    NOT NULL,   -- stored as TEXT to preserve Decimal precision
    high        TEXT    NOT NULL,
    low         TEXT    NOT NULL,
    close       TEXT    NOT NULL,
    volume      INTEGER NOT NULL,
    vwap        TEXT    NOT NULL,
    UNIQUE(symbol, timeframe, timestamp)
);
"""

_CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_bars_symbol_tf_ts
    ON bars(symbol, timeframe, timestamp);
"""


def _connect(db_path: str) -> sqlite3.Connection:
    """Return a WAL-mode SQLite connection."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.row_factory = sqlite3.Row
    return conn


class BarDatabase:
    """SQLite data-access object for :class:`~src.models.Bar` records."""

    def __init__(self, db_path: str | None = None) -> None:
        self._db_path = db_path or get_app_settings().db_path
        self._conn: sqlite3.Connection | None = None

    # ── connection lifecycle ────────────────────────────────────────────────

    def connect(self) -> None:
        """Open the database connection and ensure the schema exists."""
        self._conn = _connect(self._db_path)
        self._create_schema()
        logger.info("db_connected", path=self._db_path)

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
            logger.info("db_closed", path=self._db_path)

    def __enter__(self) -> BarDatabase:
        self.connect()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self.connect()
        return self._conn  # type: ignore[return-value]

    # ── schema ──────────────────────────────────────────────────────────────

    def _create_schema(self) -> None:
        with self.conn:
            self.conn.execute(_CREATE_BARS_TABLE)
            self.conn.execute(_CREATE_INDEX)

    # ── write operations ────────────────────────────────────────────────────

    def insert_bars(self, bars: list[Bar]) -> int:
        """Insert *bars* using INSERT OR IGNORE (skip duplicates).

        Returns:
            Number of rows actually inserted.
        """
        if not bars:
            return 0

        rows = [
            (
                b.symbol,
                b.timeframe.value,
                b.timestamp.astimezone(UTC).isoformat(),
                str(b.open),
                str(b.high),
                str(b.low),
                str(b.close),
                b.volume,
                str(b.vwap),
            )
            for b in bars
        ]

        with self.conn:
            cursor = self.conn.executemany(
                """
                INSERT OR IGNORE INTO bars
                    (symbol, timeframe, timestamp, open, high, low, close, volume, vwap)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
        inserted = cursor.rowcount
        logger.debug("bars_inserted", count=inserted, skipped=len(bars) - inserted)
        return inserted

    # ── read operations ─────────────────────────────────────────────────────

    def query_bars(
        self,
        symbol: str,
        timeframe: TimeFrame,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int | None = None,
    ) -> list[Bar]:
        """Return bars matching the given filters, ordered by timestamp ascending."""
        sql = "SELECT * FROM bars WHERE symbol = ? AND timeframe = ?"
        params: list[object] = [symbol, timeframe.value]

        if start is not None:
            sql += " AND timestamp >= ?"
            params.append(start.astimezone(UTC).isoformat())
        if end is not None:
            sql += " AND timestamp <= ?"
            params.append(end.astimezone(UTC).isoformat())

        sql += " ORDER BY timestamp ASC"

        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)

        rows = self.conn.execute(sql, params).fetchall()
        bars: list[Bar] = []
        for row in rows:
            ts = datetime.fromisoformat(row["timestamp"])
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=UTC)
            bars.append(
                Bar(
                    symbol=row["symbol"],
                    timeframe=TimeFrame(row["timeframe"]),
                    timestamp=ts,
                    open=Decimal(row["open"]),
                    high=Decimal(row["high"]),
                    low=Decimal(row["low"]),
                    close=Decimal(row["close"]),
                    volume=row["volume"],
                    vwap=Decimal(row["vwap"]),
                )
            )
        return bars

    def count_bars(self, symbol: str, timeframe: TimeFrame) -> int:
        """Return the total number of stored bars for *symbol*/*timeframe*."""
        row = self.conn.execute(
            "SELECT COUNT(*) as cnt FROM bars WHERE symbol = ? AND timeframe = ?",
            (symbol, timeframe.value),
        ).fetchone()
        return int(row["cnt"])
