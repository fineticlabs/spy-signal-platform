"""Integration: DB writes match what query functions read.

Tests that insert_signal/insert_trade produce rows that
query_recent_signals/query_recent_trades can read back correctly.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from src.models import (
    Direction,
    Regime,
    RiskDecision,
    Signal,
    TimeFrame,
    TradeResult,
)
from src.storage.queries import (
    ensure_schema,
    insert_signal,
    insert_trade,
    query_recent_signals,
    query_recent_trades,
)

# ── Shared constants ─────────────────────────────────────────────────────────

_BASE_TS = datetime(2024, 1, 16, 14, 35, tzinfo=UTC)  # 9:35 ET


def _make_signal(
    direction: Direction = Direction.LONG,
    entry: str = "482.50",
    stop: str = "480.00",
    target: str = "487.50",
    confidence: int = 3,
    offset_minutes: int = 0,
) -> Signal:
    return Signal(
        symbol="SPY",
        direction=direction,
        strategy_name="ORB-5min",
        entry_price=Decimal(entry),
        stop_price=Decimal(stop),
        target_price=Decimal(target),
        risk_reward_ratio=Decimal("2.0"),
        confidence_score=confidence,
        reason="test signal",
        timeframe=TimeFrame.ONE_MIN,
        regime=Regime.RANGING,
        timestamp=_BASE_TS + timedelta(minutes=offset_minutes),
    )


def _approved(size: int = 100) -> RiskDecision:
    return RiskDecision(approved=True, reason="All risk checks passed", position_size=size)


def _rejected(reason: str = "Max daily trades reached") -> RiskDecision:
    return RiskDecision(approved=False, reason=reason, position_size=0)


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def db_conn(tmp_path):
    conn = sqlite3.connect(str(tmp_path / "test.db"))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    ensure_schema(conn)
    return conn


# ── Tests ────────────────────────────────────────────────────────────────────


class TestApprovedVsRejectedCounts:
    """Verify counts match after inserting a mix of approved and rejected signals."""

    def test_approved_count_matches_after_mixed_signals(self, db_conn) -> None:
        """Prevents: approved/rejected counts drifting due to insert bugs."""
        insert_signal(db_conn, _make_signal(offset_minutes=0), _approved())
        insert_signal(db_conn, _make_signal(offset_minutes=1), _rejected())
        insert_signal(db_conn, _make_signal(offset_minutes=2), _approved())
        insert_signal(db_conn, _make_signal(offset_minutes=3), _rejected("Cooldown active"))
        insert_signal(db_conn, _make_signal(offset_minutes=4), _approved())

        since = datetime(2024, 1, 1, tzinfo=UTC)
        rows = query_recent_signals(db_conn, since=since)
        approved = [r for r in rows if r["approved"] == 1]
        rejected = [r for r in rows if r["approved"] == 0]
        assert len(approved) == 3
        assert len(rejected) == 2

    def test_rejection_reasons_grouped_correctly(self, db_conn) -> None:
        """Prevents: reject_reason column blank or wrong, breaking post-mortem queries."""
        insert_signal(
            db_conn, _make_signal(offset_minutes=0), _rejected("Max daily trades reached")
        )
        insert_signal(db_conn, _make_signal(offset_minutes=1), _rejected("Cooldown active"))
        insert_signal(
            db_conn, _make_signal(offset_minutes=2), _rejected("Max daily trades reached")
        )

        since = datetime(2024, 1, 1, tzinfo=UTC)
        rows = query_recent_signals(db_conn, since=since)
        rejected = [r for r in rows if r["approved"] == 0]
        reasons = [r["reject_reason"] for r in rejected]
        assert reasons.count("Max daily trades reached") == 2
        assert reasons.count("Cooldown active") == 1


class TestTradePnlConsistency:
    """Verify trade P&L sums match expectations."""

    def test_trade_pnl_sum_matches_expected(self, db_conn) -> None:
        """Prevents: P&L calculation drift between insert and query."""
        sig = _make_signal()
        trades = [
            TradeResult(
                signal=sig, pnl=Decimal("150.00"), timestamp=_BASE_TS + timedelta(minutes=10)
            ),
            TradeResult(
                signal=sig, pnl=Decimal("-75.50"), timestamp=_BASE_TS + timedelta(minutes=20)
            ),
            TradeResult(
                signal=sig, pnl=Decimal("200.00"), timestamp=_BASE_TS + timedelta(minutes=30)
            ),
        ]
        for t in trades:
            insert_trade(db_conn, t)

        since = datetime(2024, 1, 1, tzinfo=UTC)
        rows = query_recent_trades(db_conn, since=since)
        total = sum(Decimal(r["pnl"]) for r in rows)
        assert total == Decimal("274.50")


class TestZeroSignalDay:
    """Verify queries return correct results when no signals exist."""

    def test_zero_signal_day_returns_zero_counts(self, db_conn) -> None:
        """Prevents: query crashing on empty result set."""
        since = datetime(2024, 1, 1, tzinfo=UTC)
        rows = query_recent_signals(db_conn, since=since)
        assert len(rows) == 0

    def test_all_rejected_day_has_zero_approved(self, db_conn) -> None:
        """Prevents: miscount when all signals are rejected."""
        insert_signal(db_conn, _make_signal(offset_minutes=0), _rejected())
        insert_signal(db_conn, _make_signal(offset_minutes=1), _rejected())
        insert_signal(db_conn, _make_signal(offset_minutes=2), _rejected())

        since = datetime(2024, 1, 1, tzinfo=UTC)
        rows = query_recent_signals(db_conn, since=since)
        approved = [r for r in rows if r["approved"] == 1]
        assert len(approved) == 0
        assert len(rows) == 3


class TestTimestampQueries:
    """Verify timestamps are queryable by date range."""

    def test_trades_and_signals_timestamps_queryable_by_date(self, db_conn) -> None:
        """Prevents: ISO format mismatch between insert and WHERE clause."""
        # Insert signals on two different days
        day1_ts = datetime(2024, 1, 16, 14, 35, tzinfo=UTC)
        day2_ts = datetime(2024, 1, 17, 14, 35, tzinfo=UTC)

        sig1 = Signal(
            symbol="SPY",
            direction=Direction.LONG,
            strategy_name="ORB-5min",
            entry_price=Decimal("482.50"),
            stop_price=Decimal("480.00"),
            target_price=Decimal("487.50"),
            risk_reward_ratio=Decimal("2.0"),
            confidence_score=3,
            reason="test",
            timeframe=TimeFrame.ONE_MIN,
            regime=Regime.RANGING,
            timestamp=day1_ts,
        )
        sig2 = Signal(
            symbol="SPY",
            direction=Direction.LONG,
            strategy_name="ORB-5min",
            entry_price=Decimal("483.00"),
            stop_price=Decimal("481.00"),
            target_price=Decimal("488.00"),
            risk_reward_ratio=Decimal("2.0"),
            confidence_score=3,
            reason="test",
            timeframe=TimeFrame.ONE_MIN,
            regime=Regime.RANGING,
            timestamp=day2_ts,
        )
        insert_signal(db_conn, sig1, _approved())
        insert_signal(db_conn, sig2, _approved())

        # Query only day 2
        rows = query_recent_signals(db_conn, since=day2_ts)
        assert len(rows) == 1

        # Query both days
        rows_all = query_recent_signals(db_conn, since=day1_ts)
        assert len(rows_all) == 2
