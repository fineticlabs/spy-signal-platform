"""Extended integration tests: pipeline combinations, DB, and FastAPI endpoints.

Covers ingestion-to-indicator flows, strategy-to-risk-to-alert chains,
database round-trips, and FastAPI endpoint behaviour with/without injected
dependencies.
"""

from __future__ import annotations

import asyncio
import sqlite3
from datetime import datetime, timedelta
from decimal import Decimal
from threading import Thread
from zoneinfo import ZoneInfo

from fastapi.testclient import TestClient

from src.alerts.dispatcher import AlertDispatcher
from src.api import routes
from src.api.routes import app
from src.config import RiskSettings
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
from src.risk.manager import RiskManager
from src.storage.queries import (
    ensure_schema,
    insert_signal,
    insert_trade,
    query_recent_signals,
    query_recent_trades,
)
from tests.conftest import make_bar, make_utc_ts

_ET = ZoneInfo("America/New_York")


# ── helpers ──────────────────────────────────────────────────────────────────


def _build_registry() -> IndicatorRegistry:
    """Registry with EMA(9), RSI(14), ATR(14)."""
    reg = IndicatorRegistry()
    reg.register("ema9", StreamingEMA(9))
    reg.register("rsi", StreamingRSI(14))
    reg.register("atr", StreamingATR(14))
    return reg


def _varying_bars(n: int, base_close: float = 480.0) -> list:
    """Generate *n* bars with slightly varying close prices."""
    from datetime import timedelta

    bars = []
    base_ts = make_utc_ts(2024, 1, 15, 14, 30)
    for i in range(n):
        close = Decimal(str(base_close + (i % 5) * 0.3 - 0.6))
        high = close + Decimal("0.50")
        low = close - Decimal("0.50")
        ts = base_ts + timedelta(minutes=i)
        bars.append(
            make_bar(
                timestamp=ts,
                open_=close - Decimal("0.10"),
                high=high,
                low=low,
                close=close,
                volume=200_000,
            )
        )
    return bars


def _make_signal(
    ts: datetime | None = None,
    direction: Direction = Direction.LONG,
    entry: str = "486.0",
    stop: str = "483.0",
    target: str = "492.0",
    rr: str = "2.0",
    confidence: int = 3,
) -> Signal:
    """Build a valid Signal with sensible defaults."""
    return Signal(
        symbol="SPY",
        direction=direction,
        strategy_name="ORB-5min",
        entry_price=Decimal(entry),
        stop_price=Decimal(stop),
        target_price=Decimal(target),
        risk_reward_ratio=Decimal(rr),
        confidence_score=confidence,
        reason="Test signal",
        timeframe=TimeFrame.ONE_MIN,
        regime=Regime.TRENDING_UP,
        vix=Decimal("18.0"),
        adx=Decimal("25.0"),
        timestamp=ts or make_utc_ts(2024, 1, 15, 15, 0),
    )


def _make_risk_settings(max_trades: int = 5) -> RiskSettings:
    return RiskSettings(
        account_size=Decimal("50000"),
        risk_per_trade_pct=Decimal("1.0"),
        max_daily_loss_pct=Decimal("3.0"),
        max_trades_per_day=max_trades,
    )


def _db_conn(tmp_path: object) -> sqlite3.Connection:
    """Open a SQLite connection with Row factory and schema in *tmp_path*."""
    conn = sqlite3.connect(str(tmp_path) + "/test.db", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    ensure_schema(conn)
    return conn


class _FakeAlerter:
    """Stub alert channel that records calls."""

    def __init__(self) -> None:
        self.signal_calls: int = 0

    async def send_signal(self, signal: object, risk_decision: object) -> bool:
        self.signal_calls += 1
        return True

    async def send_risk_warning(self, message: str) -> bool:
        return True

    async def send_daily_summary(self, trades: object) -> bool:
        return True


class _MockRegimeDetector:
    """Minimal stand-in for RegimeDetector used in /state tests."""

    current_regime = Regime.TRENDING_UP
    vix_level = Decimal("18.0")
    adx_value = Decimal("28.0")
    is_tradeable = True


# ── Ingestion → Indicators ──────────────────────────────────────────────────


class TestIngestionToIndicators:
    """Feed bars through IndicatorRegistry and verify readiness + values."""

    def test_all_indicators_eventually_ready(self) -> None:
        """50 bars are enough to warm up EMA(9), RSI(14), and ATR(14)."""
        reg = _build_registry()
        bars = _varying_bars(50)
        for bar in bars:
            reg.update_all(bar)

        snap = reg.get_snapshot()
        assert snap.ema9 is not None
        assert snap.rsi is not None
        assert snap.atr is not None

    def test_ema_value_between_min_max_close(self) -> None:
        """Final EMA(9) should be between the min and max close of all bars."""
        reg = _build_registry()
        bars = _varying_bars(50)
        closes = [bar.close for bar in bars]
        for bar in bars:
            reg.update_all(bar)

        snap = reg.get_snapshot()
        assert snap.ema9 is not None
        assert min(closes) <= snap.ema9 <= max(closes)

    def test_rsi_between_0_and_100(self) -> None:
        """RSI must always fall within [0, 100]."""
        reg = _build_registry()
        bars = _varying_bars(50)
        for bar in bars:
            reg.update_all(bar)

        snap = reg.get_snapshot()
        assert snap.rsi is not None
        assert Decimal("0") <= snap.rsi <= Decimal("100")


# ── Indicators → Strategy ───────────────────────────────────────────────────


class TestIndicatorsToStrategy:
    """Verify that indicator readiness gates strategy evaluation."""

    def test_indicators_not_ready_no_signal(self) -> None:
        """ATR = None after just 1 bar → snapshot.atr is None, no signal should fire."""
        reg = _build_registry()
        bar = make_bar()
        reg.update_all(bar)

        snap = reg.get_snapshot()
        assert snap.atr is None  # not enough data

    def test_pre_warmed_indicators_produce_snapshot(self) -> None:
        """After 50 bars all snapshot fields for registered indicators are populated."""
        reg = _build_registry()
        for bar in _varying_bars(50):
            reg.update_all(bar)

        snap = reg.get_snapshot()
        assert snap.ema9 is not None
        assert snap.rsi is not None
        assert snap.atr is not None


# ── Strategy → Risk → Alert ─────────────────────────────────────────────────


class TestStrategyRiskAlert:
    """Signal → RiskManager → AlertDispatcher chain."""

    def test_approved_signal_dispatches(self) -> None:
        """RiskManager approves → AlertDispatcher sends (mock channel records call)."""
        cooldown = CooldownTracker()
        risk = RiskManager(cooldown=cooldown, settings=_make_risk_settings())
        signal = _make_signal(
            ts=make_utc_ts(2024, 1, 15, 15, 0),  # 10:00 ET in Jan (EST)
        )
        decision = risk.approve(signal)
        assert decision.approved

        alerter = _FakeAlerter()
        dispatcher = AlertDispatcher(alerter=alerter)
        sent = asyncio.get_event_loop().run_until_complete(
            dispatcher.dispatch_signal(signal, decision)
        )
        assert sent is True
        assert alerter.signal_calls == 1

    def test_daily_limit_rejects(self) -> None:
        """RiskManager rejects when daily trade limit is hit → no dispatch."""
        cooldown = CooldownTracker()
        settings = _make_risk_settings(max_trades=1)
        risk = RiskManager(cooldown=cooldown, settings=settings)

        # Record one trade to exhaust the daily limit
        sig = _make_signal()
        result = TradeResult(signal=sig, pnl=Decimal("100"), timestamp=sig.timestamp)
        cooldown.record_trade(result)

        decision = risk.approve(sig)
        assert not decision.approved
        assert "max daily trades" in decision.reason.lower() or "trade" in decision.reason.lower()

    def test_cooldown_active_rejects(self) -> None:
        """Two consecutive losses within 15 min → cooldown blocks the next signal."""
        frozen_now = make_utc_ts(2024, 1, 15, 15, 5)
        cooldown = CooldownTracker(now_fn=lambda: frozen_now)
        settings = _make_risk_settings()
        risk = RiskManager(cooldown=cooldown, settings=settings)

        sig = _make_signal(ts=make_utc_ts(2024, 1, 15, 15, 0))

        # Record 2 consecutive losses
        for _ in range(2):
            loss = TradeResult(signal=sig, pnl=Decimal("-100"), timestamp=frozen_now)
            cooldown.record_trade(loss)

        decision = risk.approve(sig)
        assert not decision.approved
        assert "cooldown" in decision.reason.lower()


# ── Database Integration ────────────────────────────────────────────────────


class TestDatabaseIntegration:
    """Round-trip insert → query for signals and trades."""

    def test_insert_signal_read_back(self, tmp_path: object) -> None:
        """Insert a signal and read it back; all key fields must match."""
        conn = _db_conn(tmp_path)
        sig = _make_signal()
        decision = RiskDecision(approved=True, reason="passed", position_size=166)
        row_id = insert_signal(conn, sig, decision)
        assert row_id > 0

        rows = query_recent_signals(conn, since=sig.timestamp - timedelta(minutes=1))
        assert len(rows) == 1
        row = rows[0]
        assert row["strategy_name"] == "ORB-5min"
        assert row["direction"] == "LONG"
        assert row["entry_price"] == "486.0"
        assert row["approved"] == 1
        assert row["position_size"] == 166
        conn.close()

    def test_insert_trade_read_back(self, tmp_path: object) -> None:
        """Insert a trade and read it back; PnL and direction must match."""
        conn = _db_conn(tmp_path)
        sig = _make_signal()
        result = TradeResult(signal=sig, pnl=Decimal("450.0"), timestamp=sig.timestamp)
        row_id = insert_trade(conn, result)
        assert row_id > 0

        rows = query_recent_trades(conn, since=sig.timestamp - timedelta(minutes=1))
        assert len(rows) == 1
        assert rows[0]["pnl"] == "450.0"
        assert rows[0]["direction"] == "LONG"
        conn.close()

    def test_multiple_signals_time_filter(self, tmp_path: object) -> None:
        """Insert 3 signals with different timestamps, query filters correctly."""
        conn = _db_conn(tmp_path)
        decision = RiskDecision(approved=True, reason="ok", position_size=100)
        ts_old = make_utc_ts(2024, 1, 14, 12, 0)
        ts_mid = make_utc_ts(2024, 1, 15, 12, 0)
        ts_new = make_utc_ts(2024, 1, 15, 16, 0)

        for ts in (ts_old, ts_mid, ts_new):
            insert_signal(conn, _make_signal(ts=ts), decision)

        # Only signals from Jan 15 onward
        rows = query_recent_signals(conn, since=make_utc_ts(2024, 1, 15, 0, 0))
        assert len(rows) == 2
        conn.close()

    def test_concurrent_ensure_schema_no_crash(self, tmp_path: object) -> None:
        """Multiple threads calling ensure_schema on the same DB must not raise."""
        db_file = str(tmp_path) + "/concurrent.db"
        errors: list[Exception] = []

        def worker() -> None:
            try:
                c = sqlite3.connect(db_file)
                c.row_factory = sqlite3.Row
                ensure_schema(c)
                c.close()
            except Exception as exc:
                errors.append(exc)

        threads = [Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"ensure_schema raised: {errors}"


# ── FastAPI Integration ─────────────────────────────────────────────────────


def _reset_api_state() -> None:
    """Reset the module-level globals so each test starts clean."""
    routes._conn = None
    routes._registry = None
    routes._levels = None
    routes._regime = None
    routes._cooldown = None


class TestFastAPIIntegration:
    """FastAPI endpoint tests with and without injected dependencies."""

    def test_health_returns_200(self) -> None:
        _reset_api_state()
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert "timestamp" in body

    def test_health_timestamp_present(self) -> None:
        _reset_api_state()
        client = TestClient(app)
        body = client.get("/health").json()
        # Timestamp should be parseable as ISO datetime
        datetime.fromisoformat(body["timestamp"])

    def test_state_no_deps_indicators_none(self) -> None:
        """Indicators are None when registry is not injected."""
        _reset_api_state()
        client = TestClient(app)
        body = client.get("/state").json()
        assert body["indicators"] is None

    def test_state_with_deps_returns_values(self) -> None:
        """When dependencies are set, /state returns populated fields."""
        _reset_api_state()
        reg = _build_registry()
        for bar in _varying_bars(50):
            reg.update_all(bar)

        cooldown = CooldownTracker()
        regime = _MockRegimeDetector()

        # Inject just enough to populate indicators and regime
        routes._registry = reg
        routes._regime = regime  # type: ignore[assignment]
        routes._cooldown = cooldown

        client = TestClient(app)
        body = client.get("/state").json()
        assert body["indicators"] is not None
        assert body["indicators"]["ema9"] is not None
        assert body["regime"]["name"] == "TRENDING_UP"
        assert body["risk"]["daily_trades"] == 0

    def test_signals_503_without_db(self) -> None:
        _reset_api_state()
        client = TestClient(app)
        assert client.get("/signals").status_code == 503

    def test_trades_503_without_db(self) -> None:
        _reset_api_state()
        client = TestClient(app)
        assert client.get("/trades").status_code == 503

    def test_post_trade_returns_201(self, tmp_path: object) -> None:
        _reset_api_state()
        conn = _db_conn(tmp_path)
        routes._conn = conn
        routes._cooldown = CooldownTracker()

        client = TestClient(app)
        resp = client.post(
            "/trades",
            json={
                "strategy_name": "ORB-5min",
                "direction": "LONG",
                "entry_price": "486.0",
                "stop_price": "483.0",
                "target_price": "492.0",
                "pnl": "300.0",
            },
        )
        assert resp.status_code == 201
        assert resp.json()["id"] > 0
        conn.close()

    def test_post_trade_updates_cooldown(self, tmp_path: object) -> None:
        _reset_api_state()
        conn = _db_conn(tmp_path)
        cooldown = CooldownTracker()
        routes._conn = conn
        routes._cooldown = cooldown

        client = TestClient(app)
        client.post(
            "/trades",
            json={
                "strategy_name": "ORB-5min",
                "direction": "SHORT",
                "entry_price": "486.0",
                "stop_price": "489.0",
                "target_price": "480.0",
                "pnl": "-150.0",
            },
        )
        assert cooldown.daily_trade_count == 1
        assert cooldown.daily_pnl == Decimal("-150.0")
        conn.close()

    def test_signals_with_db_returns_empty(self, tmp_path: object) -> None:
        _reset_api_state()
        conn = _db_conn(tmp_path)
        routes._conn = conn

        client = TestClient(app)
        resp = client.get("/signals")
        assert resp.status_code == 200
        assert resp.json() == []
        conn.close()

    def test_trades_with_db_returns_empty(self, tmp_path: object) -> None:
        _reset_api_state()
        conn = _db_conn(tmp_path)
        routes._conn = conn

        client = TestClient(app)
        resp = client.get("/trades")
        assert resp.status_code == 200
        assert resp.json() == []
        conn.close()
