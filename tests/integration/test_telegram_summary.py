"""Integration tests for the EOD Telegram summary produced by auto_stop.sh.

Exercises the same SQLite queries and message-building logic that auto_stop.sh
uses, against a real (temp) database with signals and trades inserted via the
production schema.

Two scenarios:
  1. Zero live trades, some rejected signals → "No live trades today" + reasons
  2. Two live paper trades (1 win, 1 loss) → both shown, emoji 📈
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from decimal import Decimal

import pytest

from src.models import Direction, Regime, RiskDecision, Signal, TimeFrame, TradeResult
from src.storage.queries import ensure_schema, insert_signal, insert_trade

# ── Helpers ──────────────────────────────────────────────────────────────────

_TODAY = "2026-03-19"


def _make_signal(
    direction: Direction = Direction.LONG,
    entry: str = "481.50",
    stop: str = "479.00",
    target: str = "486.50",
    confidence: int = 3,
    reason: str = "ORB long breakout",
    ts: str = f"{_TODAY}T14:36:00+00:00",
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
        reason=reason,
        timeframe=TimeFrame.ONE_MIN,
        regime=Regime.TRENDING_UP,
        vix=Decimal("20"),
        adx=Decimal("28"),
        timestamp=datetime.fromisoformat(ts),
    )


def _approved(position_size: int = 166) -> RiskDecision:
    return RiskDecision(approved=True, reason="Approved", position_size=position_size)


def _rejected(reason: str = "orb_filter_no_regime_data") -> RiskDecision:
    return RiskDecision(approved=False, reason=reason, position_size=0)


def _make_trade(signal: Signal, pnl: str, ts: str | None = None) -> TradeResult:
    return TradeResult(
        signal=signal,
        pnl=Decimal(pnl),
        timestamp=datetime.fromisoformat(ts or signal.timestamp.isoformat()),
    )


# ── Query helpers (same SQL as auto_stop.sh) ─────────────────────────────────


def _query_signal_count(conn: sqlite3.Connection, today: str) -> int:
    row = conn.execute(
        "SELECT COUNT(*) FROM signals WHERE timestamp >= ?",
        (f"{today}T00:00:00",),
    ).fetchone()
    return int(row[0])


def _query_approved_count(conn: sqlite3.Connection, today: str) -> int:
    row = conn.execute(
        "SELECT COUNT(*) FROM signals WHERE timestamp >= ? AND approved = 1",
        (f"{today}T00:00:00",),
    ).fetchone()
    return int(row[0])


def _query_rejected_reasons(conn: sqlite3.Connection, today: str) -> list[tuple[str, int]]:
    rows = conn.execute(
        """SELECT reject_reason, COUNT(*) as cnt FROM signals
           WHERE timestamp >= ? AND approved = 0
           GROUP BY reject_reason ORDER BY cnt DESC LIMIT 5""",
        (f"{today}T00:00:00",),
    ).fetchall()
    return [(str(r[0]), int(r[1])) for r in rows]


def _query_approved_details(conn: sqlite3.Connection, today: str) -> list[dict]:
    rows = conn.execute(
        """SELECT direction, entry_price, stop_price, target_price, confidence
           FROM signals
           WHERE timestamp >= ? AND approved = 1
           ORDER BY timestamp""",
        (f"{today}T00:00:00",),
    ).fetchall()
    return [
        {
            "direction": r[0],
            "entry_price": r[1],
            "stop_price": r[2],
            "target_price": r[3],
            "confidence": r[4],
        }
        for r in rows
    ]


def _query_realized_pnl(conn: sqlite3.Connection, today: str) -> tuple[float, int]:
    row = conn.execute(
        "SELECT COALESCE(SUM(CAST(pnl AS REAL)), 0), COUNT(*) FROM trades WHERE timestamp >= ?",
        (f"{today}T00:00:00",),
    ).fetchone()
    return float(row[0]), int(row[1])


# ── Message builder (mirrors auto_stop.sh logic in Python) ───────────────────


def _build_live_section(conn: sqlite3.Connection, today: str) -> str:
    """Build the LIVE PAPER TRADES section — same logic as auto_stop.sh."""
    signal_count = _query_signal_count(conn, today)
    approved_count = _query_approved_count(conn, today)

    if approved_count > 0:
        details = _query_approved_details(conn, today)
        lines = ""
        for d in details:
            emoji = "🟢" if d["direction"] == "LONG" else "🔴"
            lines += (
                f"  {emoji} {d['direction']} ${d['entry_price']} "
                f"(stop ${d['stop_price']}, target ${d['target_price']}) "
                f"★{d['confidence']}\n"
            )

        total_pnl, trade_count = _query_realized_pnl(conn, today)
        realized = ""
        if trade_count > 0:
            sign = "" if total_pnl < 0 else "+"
            realized = f"  Realized: {sign}${total_pnl:.2f} ({trade_count} closed)"

        return (
            f"✅ *LIVE PAPER TRADES*\n"
            f"{approved_count} signal(s) approved by risk manager\n"
            f"{lines}{realized}"
        )

    # No approved trades
    reject_summary = ""
    if signal_count > 0:
        reasons = _query_rejected_reasons(conn, today)
        if reasons:
            top_reason, top_count = reasons[0]
            reject_summary = (
                f"\n{signal_count} signal(s) generated but all rejected:\n"
                f"  {top_reason} \u00d7 {top_count}"  # multiplication sign matches auto_stop.sh
            )
    elif signal_count == 0:
        reject_summary = "\nNo signals generated (filters active)."

    return f"✅ *LIVE PAPER TRADES*\nNo live trades today.{reject_summary}"


def _build_full_message(
    conn: sqlite3.Connection,
    today: str,
    replay_section: str,
) -> str:
    """Build the full Telegram message — same structure as auto_stop.sh."""
    live_section = _build_live_section(conn, today)
    approved_count = _query_approved_count(conn, today)
    day_emoji = "📈" if approved_count > 0 else "📋"

    return (
        f"{day_emoji} *ORB Daily Report — Wednesday {today}*\n\n"
        f"{live_section}\n\n"
        f"─────────────────────────\n\n"
        f"{replay_section}"
    )


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture()
def db(tmp_path: pytest.TempPathFactory) -> sqlite3.Connection:
    """Fresh SQLite DB with schema, row_factory set for dict access."""
    db_path = str(tmp_path / "test.db")  # type: ignore[operator]
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    ensure_schema(conn)
    return conn


# ══════════════════════════════════════════════════════════════════════════════
# Scenario 1: Zero live trades, rejected signals
# ══════════════════════════════════════════════════════════════════════════════


class TestZeroLiveTradesRejectedSignals:
    """Day where the live scanner generated signals but all were rejected."""

    def _seed_db(self, conn: sqlite3.Connection) -> None:
        """Insert 4 rejected signals with the same reason."""
        for i in range(4):
            sig = _make_signal(ts=f"{_TODAY}T14:{36 + i}:00+00:00")
            insert_signal(conn, sig, _rejected("orb_filter_no_regime_data"))

    def test_signal_count_is_four(self, db: sqlite3.Connection) -> None:
        self._seed_db(db)
        assert _query_signal_count(db, _TODAY) == 4

    def test_approved_count_is_zero(self, db: sqlite3.Connection) -> None:
        self._seed_db(db)
        assert _query_approved_count(db, _TODAY) == 0

    def test_rejection_reason_captured(self, db: sqlite3.Connection) -> None:
        self._seed_db(db)
        reasons = _query_rejected_reasons(db, _TODAY)
        assert len(reasons) == 1
        assert reasons[0] == ("orb_filter_no_regime_data", 4)

    def test_live_section_says_no_trades(self, db: sqlite3.Connection) -> None:
        self._seed_db(db)
        section = _build_live_section(db, _TODAY)
        assert "No live trades today" in section

    def test_live_section_shows_rejection_reason(self, db: sqlite3.Connection) -> None:
        self._seed_db(db)
        section = _build_live_section(db, _TODAY)
        assert "orb_filter_no_regime_data" in section
        assert "4 signal(s) generated but all rejected" in section

    def test_full_message_emoji_is_clipboard(self, db: sqlite3.Connection) -> None:
        self._seed_db(db)
        replay = "📊 *REPLAY BACKTEST* _(hypothetical)_\n⚠️ These are backtest results"
        msg = _build_full_message(db, _TODAY, replay)
        assert msg.startswith("📋")

    def test_full_message_contains_hypothetical_label(self, db: sqlite3.Connection) -> None:
        self._seed_db(db)
        replay = "📊 *REPLAY BACKTEST* _(hypothetical)_\n⚠️ These are backtest results"
        msg = _build_full_message(db, _TODAY, replay)
        assert "hypothetical" in msg
        assert "REPLAY BACKTEST" in msg

    def test_full_message_has_both_sections(self, db: sqlite3.Connection) -> None:
        self._seed_db(db)
        replay = "📊 *REPLAY BACKTEST* _(hypothetical)_"
        msg = _build_full_message(db, _TODAY, replay)
        assert "LIVE PAPER TRADES" in msg
        assert "REPLAY BACKTEST" in msg
        assert "─────────────────────────" in msg


# ══════════════════════════════════════════════════════════════════════════════
# Scenario 1b: Zero signals generated at all
# ══════════════════════════════════════════════════════════════════════════════


class TestZeroSignalsGenerated:
    """Day where the live scanner generated zero signals (all filtered)."""

    def test_live_section_says_no_signals(self, db: sqlite3.Connection) -> None:
        section = _build_live_section(db, _TODAY)
        assert "No live trades today" in section
        assert "No signals generated (filters active)" in section


# ══════════════════════════════════════════════════════════════════════════════
# Scenario 2: Two live paper trades (1 win, 1 loss)
# ══════════════════════════════════════════════════════════════════════════════


class TestTwoLiveTrades:
    """Day with 2 approved signals and realized P&L."""

    def _seed_db(self, conn: sqlite3.Connection) -> None:
        """Insert 2 approved signals + 2 closed trades."""
        sig1 = _make_signal(
            direction=Direction.LONG,
            entry="481.50",
            stop="479.00",
            target="486.50",
            confidence=4,
            ts=f"{_TODAY}T14:36:00+00:00",
        )
        insert_signal(conn, sig1, _approved(166))

        sig2 = _make_signal(
            direction=Direction.SHORT,
            entry="490.00",
            stop="492.50",
            target="485.00",
            confidence=3,
            ts=f"{_TODAY}T15:10:00+00:00",
        )
        insert_signal(conn, sig2, _approved(100))

        # Trade 1: winner (+$350)
        insert_trade(conn, _make_trade(sig1, "350.00", f"{_TODAY}T14:55:00+00:00"))
        # Trade 2: loser (-$200)
        insert_trade(conn, _make_trade(sig2, "-200.00", f"{_TODAY}T15:30:00+00:00"))

    def test_approved_count_is_two(self, db: sqlite3.Connection) -> None:
        self._seed_db(db)
        assert _query_approved_count(db, _TODAY) == 2

    def test_live_section_shows_both_trades(self, db: sqlite3.Connection) -> None:
        self._seed_db(db)
        section = _build_live_section(db, _TODAY)
        assert "2 signal(s) approved" in section
        assert "LONG" in section
        assert "SHORT" in section
        assert "🟢" in section
        assert "🔴" in section

    def test_live_section_shows_entry_stop_target(self, db: sqlite3.Connection) -> None:
        self._seed_db(db)
        section = _build_live_section(db, _TODAY)
        assert "$481.50" in section  # entry
        assert "$479.00" in section  # stop
        assert "$486.50" in section  # target

    def test_live_section_shows_confidence_stars(self, db: sqlite3.Connection) -> None:
        self._seed_db(db)
        section = _build_live_section(db, _TODAY)
        assert "★4" in section
        assert "★3" in section

    def test_live_section_shows_realized_pnl(self, db: sqlite3.Connection) -> None:
        self._seed_db(db)
        section = _build_live_section(db, _TODAY)
        assert "Realized:" in section
        assert "2 closed" in section

    def test_realized_pnl_net_is_positive(self, db: sqlite3.Connection) -> None:
        self._seed_db(db)
        total_pnl, count = _query_realized_pnl(db, _TODAY)
        assert count == 2
        assert total_pnl == pytest.approx(150.0)  # 350 - 200

    def test_live_section_does_not_say_no_trades(self, db: sqlite3.Connection) -> None:
        self._seed_db(db)
        section = _build_live_section(db, _TODAY)
        assert "No live trades today" not in section

    def test_full_message_emoji_is_chart(self, db: sqlite3.Connection) -> None:
        self._seed_db(db)
        replay = "📊 *REPLAY BACKTEST* _(hypothetical)_"
        msg = _build_full_message(db, _TODAY, replay)
        assert msg.startswith("📈")

    def test_full_message_has_divider(self, db: sqlite3.Connection) -> None:
        self._seed_db(db)
        replay = "📊 *REPLAY BACKTEST* _(hypothetical)_"
        msg = _build_full_message(db, _TODAY, replay)
        assert "─────────────────────────" in msg


# ══════════════════════════════════════════════════════════════════════════════
# Scenario 3: Mixed — some approved, some rejected
# ══════════════════════════════════════════════════════════════════════════════


class TestMixedApprovedAndRejected:
    """Day with 1 approved signal and 3 rejected."""

    def _seed_db(self, conn: sqlite3.Connection) -> None:
        sig1 = _make_signal(ts=f"{_TODAY}T14:36:00+00:00")
        insert_signal(conn, sig1, _approved(166))

        for i in range(3):
            sig = _make_signal(ts=f"{_TODAY}T14:{40 + i}:00+00:00")
            insert_signal(conn, sig, _rejected("Daily loss limit reached"))

    def test_approved_one_rejected_three(self, db: sqlite3.Connection) -> None:
        self._seed_db(db)
        assert _query_approved_count(db, _TODAY) == 1
        assert _query_signal_count(db, _TODAY) == 4

    def test_live_section_shows_approved_trade(self, db: sqlite3.Connection) -> None:
        self._seed_db(db)
        section = _build_live_section(db, _TODAY)
        assert "1 signal(s) approved" in section
        assert "No live trades today" not in section
