"""Integration tests: bar sequence flows through the full pipeline.

These tests verify that a realistic ORB breakout scenario produces a signal
with correct properties, passes the risk gate, and would be dispatched — all
without any network I/O (no Alpaca, no Telegram).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo

from src.alerts.dispatcher import AlertDispatcher
from src.indicators.registry import IndicatorRegistry
from src.indicators.streaming import StreamingATR, StreamingEMA, StreamingMACD, StreamingRSI
from src.levels import LevelManager
from src.models import Bar, Direction, RiskDecision, Signal, TimeFrame
from src.risk.cooldown import CooldownTracker
from src.risk.manager import RiskManager
from src.strategies.orb import ORBStrategy
from src.strategies.regime import RegimeDetector

_ET = ZoneInfo("America/New_York")


# ── helpers ───────────────────────────────────────────────────────────────────


def _et_bar(
    et_hour: int,
    et_minute: int,
    close: float = 480.0,
    high: float | None = None,
    low: float | None = None,
    volume: int = 500_000,
    date_str: str = "2024-01-15",
) -> Bar:
    """Build a 1-min bar at the given ET wall-clock time."""
    if high is None:
        high = close + 0.5
    if low is None:
        low = close - 0.5
    naive = datetime.strptime(f"{date_str} {et_hour:02d}:{et_minute:02d}", "%Y-%m-%d %H:%M")
    ts = naive.replace(tzinfo=_ET).astimezone(UTC)
    return Bar(
        symbol="SPY",
        timeframe=TimeFrame.ONE_MIN,
        timestamp=ts,
        open=Decimal(str(close - 0.1)),
        high=Decimal(str(high)),
        low=Decimal(str(low)),
        close=Decimal(str(close)),
        volume=volume,
        vwap=Decimal(str(close)),
    )


def _build_registry() -> IndicatorRegistry:
    r = IndicatorRegistry()
    r.register("ema9", StreamingEMA(9))
    r.register("ema20", StreamingEMA(20))
    r.register("ema50", StreamingEMA(50))
    r.register("rsi", StreamingRSI(14))
    r.register("macd", StreamingMACD())
    r.register("atr", StreamingATR(14))
    return r


class _FakeAlerter:
    """Stub alerter that records calls without sending anything."""

    def __init__(self) -> None:
        self.signal_calls: int = 0
        self.warning_calls: int = 0
        self.summary_calls: int = 0

    async def send_signal(self, signal: object, risk_decision: object) -> bool:
        self.signal_calls += 1
        return True

    async def send_risk_warning(self, message: str) -> bool:
        self.warning_calls += 1
        return True

    async def send_daily_summary(self, trades: object) -> bool:
        self.summary_calls += 1
        return True


def _build_pipeline(
    account_size: float = 50_000,
) -> tuple[
    IndicatorRegistry,
    LevelManager,
    RegimeDetector,
    ORBStrategy,
    RiskManager,
    _FakeAlerter,
    AlertDispatcher,
]:
    from src.config import RiskSettings

    registry = _build_registry()
    levels = LevelManager(db=None, symbol="SPY")
    regime = RegimeDetector()
    strategy = ORBStrategy()
    cooldown = CooldownTracker()
    settings = RiskSettings(
        account_size=Decimal(str(account_size)),
        risk_per_trade_pct=Decimal("1.0"),
        max_daily_loss_pct=Decimal("3.0"),
        max_trades_per_day=5,
    )
    risk = RiskManager(cooldown=cooldown, settings=settings)
    alerter = _FakeAlerter()
    dispatcher = AlertDispatcher(alerter=alerter)
    return registry, levels, regime, strategy, risk, alerter, dispatcher


def _prime_atr(registry: IndicatorRegistry, levels: LevelManager, n: int = 20) -> None:
    """Feed n bars at 9:30 ET (premarket) to warm up ATR without triggering ORB."""
    for _ in range(n):
        bar = _et_bar(9, 29, close=480.0, volume=100_000)
        registry.update_all(bar)
        levels.update(bar)


def _feed_orb_bars(
    registry: IndicatorRegistry, levels: LevelManager, orb_high: float = 483.0
) -> None:
    """Feed 5 1-min bars 9:30-9:34 ET to establish the ORB window.

    The 5th bar sets orb_high = the max seen across those bars.
    """
    prices = [479.0, 480.0, 481.0, orb_high, 482.0]
    for minute, price in zip(range(30, 35), prices, strict=False):
        bar = _et_bar(9, minute, close=price, high=price + 0.5, low=price - 0.5, volume=500_000)
        registry.update_all(bar)
        levels.update(bar)


# ── tests ─────────────────────────────────────────────────────────────────────


class TestORBPipelineIntegration:
    """Full pipeline: warm-up → ORB window → breakout bar → signal → risk → alert."""

    def _run_pipeline(
        self,
        breakout_close: float = 485.0,
        breakout_volume: int = 1_500_000,
        et_hour: int = 10,
        et_minute: int = 0,
    ) -> tuple[Signal | None, RiskDecision | None, _FakeAlerter]:
        """Return (signal, decision, alerter) for a single breakout bar."""
        registry, levels, regime, strategy, risk, alerter, _dispatcher = _build_pipeline()

        # Warm up ATR with 20 flat bars
        _prime_atr(registry, levels, n=20)

        # Feed ORB window (9:30-9:34 ET), ORB high ~483.5
        _feed_orb_bars(registry, levels, orb_high=483.0)

        # Set regime so ORB filters pass
        regime.update(vix=Decimal("18"), adx=Decimal("25"), trending_up=True)

        # Feed enough additional bars so volume deque has 20 entries at 500k avg
        for minute in range(35, 55):
            bar = _et_bar(9, minute, close=482.0, volume=500_000)
            registry.update_all(bar)
            levels.update(bar)
            strategy.evaluate(bar, registry.get_snapshot(), levels.get_levels(), regime)

        # Feed the breakout bar
        breakout = _et_bar(et_hour, et_minute, close=breakout_close, volume=breakout_volume)
        registry.update_all(breakout)
        levels.update(breakout)
        snapshot = registry.get_snapshot()
        level_snap = levels.get_levels()

        signal = strategy.evaluate(breakout, snapshot, level_snap, regime)
        decision = risk.approve(signal) if signal is not None else None
        return signal, decision, alerter

    def test_orb_breakout_produces_long_signal(self) -> None:
        """Price above ORB high + high volume → LONG signal."""
        signal, _decision, _ = self._run_pipeline(breakout_close=486.0, breakout_volume=1_500_000)
        assert signal is not None
        assert signal.direction == Direction.LONG
        assert signal.strategy_name == "ORB-5min"
        assert signal.entry_price == Decimal("486.0")

    def test_signal_stop_is_below_entry_for_long(self) -> None:
        """Stop price must be below entry for a LONG signal."""
        signal, _, _ = self._run_pipeline(breakout_close=486.0)
        assert signal is not None
        assert signal.stop_price < signal.entry_price

    def test_signal_target_is_above_entry_for_long(self) -> None:
        """Target price must be above entry for a LONG signal."""
        signal, _, _ = self._run_pipeline(breakout_close=486.0)
        assert signal is not None
        assert signal.target_price > signal.entry_price

    def test_signal_rr_is_two(self) -> None:
        """ORB strategy always uses 2R."""
        signal, _, _ = self._run_pipeline(breakout_close=486.0)
        assert signal is not None
        assert signal.risk_reward_ratio == Decimal("2.0")

    def test_risk_manager_approves_valid_breakout(self) -> None:
        """Risk gate must approve a valid ORB breakout at 10:00 ET."""
        signal, decision, _ = self._run_pipeline(
            breakout_close=486.0, breakout_volume=1_500_000, et_hour=10, et_minute=0
        )
        assert signal is not None
        assert decision is not None
        assert decision.approved
        assert decision.position_size > 0

    def test_risk_manager_rejects_low_rr_signal(self) -> None:
        """Decision is rejected if the strategy somehow produced an RR < 1.5."""
        from src.models import Signal

        registry, levels, regime, _strategy, risk, _alerter, _ = _build_pipeline()
        _prime_atr(registry, levels)
        _feed_orb_bars(registry, levels, orb_high=483.0)
        regime.update(vix=Decimal("18"), adx=Decimal("25"), trending_up=True)

        # Build a signal with RR = 0.5 (manually crafted)
        ts = _et_bar(10, 0).timestamp
        signal = Signal(
            direction=Direction.LONG,
            strategy_name="ORB-5min",
            entry_price=Decimal("486.0"),
            stop_price=Decimal("483.0"),
            target_price=Decimal("487.5"),  # only 1.5 points up vs 3 down
            risk_reward_ratio=Decimal("0.5"),
            confidence_score=3,
            reason="test",
            timeframe=TimeFrame.ONE_MIN,
            regime=regime.current_regime,
            timestamp=ts,
        )
        decision = risk.approve(signal)
        assert not decision.approved
        assert "r:r" in decision.reason.lower() or "risk" in decision.reason.lower()

    def test_no_signal_inside_orb_range(self) -> None:
        """Price inside ORB range (no breakout) → no signal."""
        signal, _, _ = self._run_pipeline(breakout_close=481.0)
        assert signal is None

    def test_no_signal_without_sufficient_volume(self) -> None:
        """Low volume (below 1.5x avg) → no signal even with price breakout."""
        signal, _, _ = self._run_pipeline(breakout_close=486.0, breakout_volume=100)
        assert signal is None

    def test_short_signal_on_breakdown(self) -> None:
        """Price below ORB low → SHORT signal."""
        registry, levels, regime, strategy, _risk, _alerter, _ = _build_pipeline()
        _prime_atr(registry, levels, n=20)
        # ORB low is min of the window -- set window prices so low is ~476.5
        prices = [479.0, 477.0, 476.0, 480.0, 481.0]
        for minute, price in zip(range(30, 35), prices, strict=False):
            bar = _et_bar(9, minute, close=price, high=price + 0.5, low=price - 0.5, volume=500_000)
            registry.update_all(bar)
            levels.update(bar)

        regime.update(vix=Decimal("18"), adx=Decimal("25"), trending_up=False)

        # Prime volume deque with 20 avg-500k bars
        for minute in range(35, 55):
            bar = _et_bar(9, minute, close=478.0, volume=500_000)
            registry.update_all(bar)
            levels.update(bar)
            strategy.evaluate(bar, registry.get_snapshot(), levels.get_levels(), regime)

        breakdown = _et_bar(10, 0, close=474.0, volume=1_500_000)
        registry.update_all(breakdown)
        levels.update(breakdown)
        snapshot = registry.get_snapshot()
        level_snap = levels.get_levels()
        signal = strategy.evaluate(breakdown, snapshot, level_snap, regime)

        assert signal is not None
        assert signal.direction == Direction.SHORT
        assert signal.stop_price > signal.entry_price
        assert signal.target_price < signal.entry_price


class TestAPIRoutes:
    """Unit-level tests for the FastAPI routes (no network)."""

    def test_health_endpoint(self) -> None:
        from fastapi.testclient import TestClient

        from src.api.routes import app

        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert "timestamp" in body

    def test_state_endpoint_without_dependencies(self) -> None:
        """State endpoint returns nulls gracefully when pipeline not injected."""
        from fastapi.testclient import TestClient

        from src.api import routes
        from src.api.routes import app

        # Reset global state for isolation
        routes._conn = None
        routes._registry = None
        routes._levels = None
        routes._regime = None
        routes._cooldown = None

        client = TestClient(app)
        response = client.get("/state")
        assert response.status_code == 200
        body = response.json()
        assert body["indicators"] is None
        assert body["levels"] is None

    def test_signals_endpoint_requires_db(self) -> None:
        """GET /signals returns 503 if DB not connected."""
        from fastapi.testclient import TestClient

        from src.api import routes
        from src.api.routes import app

        routes._conn = None

        client = TestClient(app)
        response = client.get("/signals")
        assert response.status_code == 503

    def test_trades_endpoint_requires_db(self) -> None:
        """GET /trades returns 503 if DB not connected."""
        from fastapi.testclient import TestClient

        from src.api import routes
        from src.api.routes import app

        routes._conn = None

        client = TestClient(app)
        response = client.get("/trades")
        assert response.status_code == 503

    def test_post_trade_with_live_db(self, tmp_path: object) -> None:
        """POST /trades persists a trade and returns the row id."""
        import sqlite3

        from fastapi.testclient import TestClient

        from src.api import routes
        from src.api.routes import app
        from src.storage.queries import ensure_schema

        db_file = str(tmp_path) + "/test.db"
        # check_same_thread=False required because FastAPI runs sync endpoints
        # in a worker thread via anyio, which is a different thread from the test.
        conn = sqlite3.connect(db_file, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        ensure_schema(conn)

        cooldown = CooldownTracker()
        routes._conn = conn
        routes._cooldown = cooldown

        client = TestClient(app)
        response = client.post(
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
        assert response.status_code == 201
        body = response.json()
        assert body["id"] > 0
        assert body["pnl"] == "300.0"

        # Cooldown should have recorded the trade
        assert cooldown.daily_trade_count == 1
        conn.close()


class TestStorageQueries:
    """Tests for the queries module."""

    def test_insert_and_query_signal(self, tmp_path: object) -> None:
        import sqlite3

        from src.models import Regime, Signal, TimeFrame
        from src.storage.queries import ensure_schema, insert_signal, query_recent_signals

        db_file = str(tmp_path) + "/test.db"
        conn = sqlite3.connect(db_file)
        conn.row_factory = sqlite3.Row
        ensure_schema(conn)

        from src.models import Direction, RiskDecision

        ts = datetime(2024, 1, 15, 15, 0, tzinfo=UTC)
        signal = Signal(
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
            vix=Decimal("18.0"),
            adx=Decimal("25.0"),
            timestamp=ts,
        )
        decision = RiskDecision(approved=True, reason="passed", position_size=166)

        row_id = insert_signal(conn, signal, decision)
        assert row_id > 0

        rows = query_recent_signals(conn, since=ts - timedelta(minutes=1))
        assert len(rows) == 1
        assert rows[0]["strategy_name"] == "ORB-5min"
        assert rows[0]["approved"] == 1
        conn.close()

    def test_insert_and_query_trade(self, tmp_path: object) -> None:
        import sqlite3

        from src.models import Direction, Regime, Signal, TimeFrame, TradeResult
        from src.storage.queries import ensure_schema, insert_trade, query_recent_trades

        db_file = str(tmp_path) + "/test.db"
        conn = sqlite3.connect(db_file)
        conn.row_factory = sqlite3.Row
        ensure_schema(conn)

        ts = datetime(2024, 1, 15, 15, 0, tzinfo=UTC)
        sig = Signal(
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
            timestamp=ts,
        )
        result = TradeResult(signal=sig, pnl=Decimal("600.0"), timestamp=ts)
        row_id = insert_trade(conn, result)
        assert row_id > 0

        rows = query_recent_trades(conn, since=ts - timedelta(minutes=1))
        assert len(rows) == 1
        assert rows[0]["pnl"] == "600.0"
        conn.close()
