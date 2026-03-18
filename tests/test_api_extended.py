"""Extended API tests: query params, validation, ISO timestamps, concurrent requests.

Exercises FastAPI endpoint edge cases beyond the basic happy-path integration
tests, including parameter filtering, Pydantic 422 validation, and concurrent
request safety.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from fastapi.testclient import TestClient

from src.api import routes
from src.api.routes import app
from src.indicators.registry import IndicatorRegistry
from src.indicators.streaming import StreamingATR, StreamingEMA, StreamingRSI
from src.models import (
    Direction,
    Regime,
    RiskDecision,
    Signal,
    TimeFrame,
    TradeResult,
)
from src.risk.cooldown import CooldownTracker
from src.storage.queries import ensure_schema, insert_signal, insert_trade
from tests.conftest import make_bar, make_utc_ts

# ── helpers ──────────────────────────────────────────────────────────────────


def _reset() -> None:
    routes._conn = None
    routes._registry = None
    routes._levels = None
    routes._regime = None
    routes._cooldown = None


def _db(tmp_path: object) -> sqlite3.Connection:
    conn = sqlite3.connect(str(tmp_path) + "/api_ext.db", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    ensure_schema(conn)
    return conn


def _signal(ts: datetime | None = None) -> Signal:
    return Signal(
        symbol="SPY",
        direction=Direction.LONG,
        strategy_name="ORB-5min",
        entry_price=Decimal("486.0"),
        stop_price=Decimal("483.0"),
        target_price=Decimal("492.0"),
        risk_reward_ratio=Decimal("2.0"),
        confidence_score=3,
        reason="test",
        timeframe=TimeFrame.ONE_MIN,
        regime=Regime.TRENDING_UP,
        timestamp=ts or datetime.now(UTC),
    )


def _trade_result(ts: datetime | None = None) -> TradeResult:
    sig = _signal(ts)
    return TradeResult(signal=sig, pnl=Decimal("200.0"), timestamp=sig.timestamp)


def _build_registry(n_bars: int = 50) -> IndicatorRegistry:
    reg = IndicatorRegistry()
    reg.register("ema9", StreamingEMA(9))
    reg.register("rsi", StreamingRSI(14))
    reg.register("atr", StreamingATR(14))
    from datetime import timedelta as _td

    base_ts = make_utc_ts(2024, 1, 15, 14, 30)
    for i in range(n_bars):
        close = Decimal(str(480 + (i % 5) * 0.3))
        high = close + Decimal("0.5")
        low = close - Decimal("0.5")
        bar = make_bar(
            timestamp=base_ts + _td(minutes=i),
            open_=close - Decimal("0.1"),
            high=high,
            low=low,
            close=close,
            volume=200_000,
        )
        reg.update_all(bar)
    return reg


class _MockRegime:
    current_regime = Regime.TRENDING_UP
    vix_level = Decimal("18.0")
    adx_value = Decimal("28.0")
    is_tradeable = True


class _MockLevels:
    """Minimal stand-in for LevelManager.get_levels()."""

    def get_levels(self):
        from src.models import LevelSnapshot

        return LevelSnapshot(orb_high=Decimal("483.5"), orb_low=Decimal("478.5"))


# ── tests ────────────────────────────────────────────────────────────────────


class TestHealthExtended:
    """Extended /health checks."""

    def test_timestamp_is_iso_format(self) -> None:
        _reset()
        client = TestClient(app)
        body = client.get("/health").json()
        ts = body["timestamp"]
        # Must parse without error
        parsed = datetime.fromisoformat(ts)
        assert parsed.tzinfo is not None or "+" in ts or ts.endswith("Z")


class TestSignalsExtended:
    """Extended /signals query-param tests."""

    def test_hours_filter(self, tmp_path: object) -> None:
        """/signals?hours=1 excludes signals older than 1 hour."""
        _reset()
        conn = _db(tmp_path)
        routes._conn = conn
        decision = RiskDecision(approved=True, reason="ok", position_size=100)

        now = datetime.now(UTC)
        old_ts = now - timedelta(hours=2)
        recent_ts = now - timedelta(minutes=30)

        insert_signal(conn, _signal(ts=old_ts), decision)
        insert_signal(conn, _signal(ts=recent_ts), decision)

        client = TestClient(app)
        rows = client.get("/signals?hours=1").json()
        assert len(rows) == 1
        conn.close()

    def test_limit_caps_results(self, tmp_path: object) -> None:
        """/signals?limit=5 returns at most 5 rows."""
        _reset()
        conn = _db(tmp_path)
        routes._conn = conn
        decision = RiskDecision(approved=True, reason="ok", position_size=100)

        now = datetime.now(UTC)
        for i in range(10):
            insert_signal(conn, _signal(ts=now - timedelta(minutes=i)), decision)

        client = TestClient(app)
        rows = client.get("/signals?limit=5").json()
        assert len(rows) == 5
        conn.close()


class TestTradesExtended:
    """Extended /trades query-param and validation tests."""

    def test_days_filter(self, tmp_path: object) -> None:
        """/trades?days=1 excludes trades older than 1 day."""
        _reset()
        conn = _db(tmp_path)
        routes._conn = conn

        now = datetime.now(UTC)
        old_trade = _trade_result(ts=now - timedelta(days=3))
        recent_trade = _trade_result(ts=now - timedelta(hours=6))

        insert_trade(conn, old_trade)
        insert_trade(conn, recent_trade)

        client = TestClient(app)
        rows = client.get("/trades?days=1").json()
        assert len(rows) == 1
        conn.close()

    def test_post_missing_field_422(self, tmp_path: object) -> None:
        """POST /trades with a missing required field returns 422."""
        _reset()
        conn = _db(tmp_path)
        routes._conn = conn
        routes._cooldown = CooldownTracker()

        client = TestClient(app)
        resp = client.post(
            "/trades",
            json={
                "strategy_name": "ORB-5min",
                # direction is missing
                "entry_price": "486.0",
                "stop_price": "483.0",
                "target_price": "492.0",
                "pnl": "300.0",
            },
        )
        assert resp.status_code == 422
        conn.close()

    def test_post_invalid_direction_422(self, tmp_path: object) -> None:
        """POST /trades with an invalid direction value returns 422."""
        _reset()
        conn = _db(tmp_path)
        routes._conn = conn
        routes._cooldown = CooldownTracker()

        client = TestClient(app)
        resp = client.post(
            "/trades",
            json={
                "strategy_name": "ORB-5min",
                "direction": "SIDEWAYS",  # invalid
                "entry_price": "486.0",
                "stop_price": "483.0",
                "target_price": "492.0",
                "pnl": "300.0",
            },
        )
        assert resp.status_code == 422
        conn.close()


class TestStateExtended:
    """Extended /state endpoint tests."""

    def test_full_deps_all_fields_populated(self) -> None:
        """/state returns all sections when full dependencies are injected."""
        _reset()
        reg = _build_registry(n_bars=50)
        cooldown = CooldownTracker()
        regime = _MockRegime()
        levels = _MockLevels()

        routes._registry = reg
        routes._cooldown = cooldown
        routes._regime = regime  # type: ignore[assignment]
        routes._levels = levels  # type: ignore[assignment]

        client = TestClient(app)
        body = client.get("/state").json()

        # indicators section
        assert body["indicators"] is not None
        assert body["indicators"]["ema9"] is not None
        assert body["indicators"]["rsi"] is not None
        assert body["indicators"]["atr"] is not None

        # regime section
        assert body["regime"]["name"] == "TRENDING_UP"
        assert body["regime"]["vix"] is not None
        assert body["regime"]["adx"] is not None
        assert body["regime"]["tradeable"] is True

        # risk section
        assert body["risk"]["daily_trades"] == 0
        assert body["risk"]["cooled_down"] is True
        assert body["risk"]["tilted"] is False

        # levels section
        assert body["levels"] is not None


class TestConcurrentRequests:
    """Verify sequential (concurrent-style) requests do not crash."""

    def test_rapid_sequential_requests(self, tmp_path: object) -> None:
        """Fire 20 rapid GET requests to different endpoints; none should crash."""
        _reset()
        conn = _db(tmp_path)
        routes._conn = conn
        routes._cooldown = CooldownTracker()

        client = TestClient(app)
        for _ in range(20):
            assert client.get("/health").status_code == 200
            assert client.get("/state").status_code == 200
            assert client.get("/signals").status_code == 200
            assert client.get("/trades").status_code == 200
        conn.close()
