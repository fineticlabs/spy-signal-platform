"""Tests for the 5 bug fixes deployed 2026-03-24.

Covers:
  P0 — Monday filter in _process_bar
  P0 — In-memory submitted_symbols dedup in AlpacaExecutor
  P1 — Pre-load overnight positions at startup
  P1 — PDL date resolution (ET session window in UTC)
  P2 — EOD reconciliation (fill_price + trades table)
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from src.levels.daily_levels import PreviousDayLevels
from src.levels.trading_calendar import is_trading_day, last_trading_day
from src.models import Bar, Direction, Regime, Signal, TimeFrame
from src.storage.database import BarDatabase
from src.storage.queries import (
    ensure_schema,
    insert_signal,
    mark_signal_executed,
    query_executed_signals,
)

_ET = ZoneInfo("America/New_York")


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_bar(
    symbol: str = "SPY",
    et_hour: int = 10,
    et_minute: int = 0,
    close: float = 480.0,
    volume: int = 200_000,
    date_str: str = "2024-01-16",
) -> Bar:
    """Build a bar at the given ET hour/minute."""
    naive = datetime.strptime(f"{date_str} {et_hour:02d}:{et_minute:02d}", "%Y-%m-%d %H:%M")
    ts_utc = naive.replace(tzinfo=_ET).astimezone(UTC)
    return Bar(
        symbol=symbol,
        timeframe=TimeFrame.ONE_MIN,
        timestamp=ts_utc,
        open=Decimal(str(close)),
        high=Decimal(str(close + 0.5)),
        low=Decimal(str(close - 0.5)),
        close=Decimal(str(close)),
        volume=volume,
        vwap=Decimal(str(close)),
    )


def _monday_bar(symbol: str = "SPY") -> Bar:
    """Build a bar on a known Monday (2024-01-15 is a Monday)."""
    return _make_bar(symbol=symbol, date_str="2024-01-15", et_hour=10)


def _tuesday_bar(symbol: str = "SPY") -> Bar:
    """Build a bar on a known Tuesday (2024-01-16 is a Tuesday)."""
    return _make_bar(symbol=symbol, date_str="2024-01-16", et_hour=10)


def _make_signal(symbol: str = "SPY", direction: Direction = Direction.LONG) -> Signal:
    return Signal(
        symbol=symbol,
        direction=direction,
        strategy_name="ORB-5min",
        entry_price=Decimal("481.50"),
        stop_price=Decimal("479.00"),
        target_price=Decimal("486.50"),
        risk_reward_ratio=Decimal("2.0"),
        confidence_score=3,
        reason="test signal",
        timeframe=TimeFrame.ONE_MIN,
        regime=Regime.TRENDING_UP,
        timestamp=datetime(2024, 1, 16, 14, 36, tzinfo=UTC),
    )


@pytest.fixture
def tmp_db(tmp_path):
    db = BarDatabase(db_path=str(tmp_path / "test.db"))
    db.connect()
    ensure_schema(db.conn)
    yield db
    db.close()


# =============================================================================
# P0: Monday filter in _process_bar
# =============================================================================


class TestMondayFilter:
    """_process_bar should skip bars on excluded days (Monday by default)."""

    @pytest.mark.asyncio
    async def test_monday_bar_skipped(self) -> None:
        """Bars on Monday must be silently dropped — no signal generated."""
        import src.main as main_mod
        from src.alerts.dispatcher import AlertDispatcher
        from src.config import RiskSettings
        from src.levels import LevelManager
        from src.main import _build_registry, _process_bar, _SymbolPipeline
        from src.risk.cooldown import CooldownTracker
        from src.risk.manager import RiskManager
        from src.strategies.orb import ORBStrategy
        from src.strategies.regime import RegimeDetector

        # Set the module-level excluded weekdays (simulates what run() does)
        original = main_mod._excluded_weekdays
        main_mod._excluded_weekdays = {0}  # Monday

        try:
            pipeline = _SymbolPipeline(
                symbol="SPY",
                registry=_build_registry(),
                levels=LevelManager(db=None, symbol="SPY"),
                regime=RegimeDetector(),
                strategy=ORBStrategy(
                    excluded_days=[]
                ),  # intentionally empty — test the _process_bar gate
            )
            risk = RiskManager(
                cooldown=CooldownTracker(),
                settings=RiskSettings(
                    account_size=Decimal("50000"),
                    risk_per_trade_pct=Decimal("1.0"),
                    max_daily_loss_pct=Decimal("3.0"),
                    max_trades_per_day=5,
                    max_concurrent_positions=3,
                ),
            )
            alerter = MagicMock()
            alerter.send_signal = AsyncMock(return_value=True)
            alerter.send_risk_warning = AsyncMock(return_value=True)
            alerter.send_status = AsyncMock(return_value=True)
            dispatcher = AlertDispatcher(alerter=alerter)

            import tempfile
            from pathlib import Path

            with tempfile.TemporaryDirectory() as tmpdir:
                db = BarDatabase(db_path=str(Path(tmpdir) / "test.db"))
                db.connect()
                ensure_schema(db.conn)

                bar = _monday_bar()
                await _process_bar(bar, pipeline, risk, dispatcher, db, executor=None)

                # Verify no signal was generated (indicators not updated on Monday)
                snapshot = pipeline.registry.get_snapshot()
                assert snapshot.atr is None  # indicators not updated = bar was skipped

                db.close()
        finally:
            main_mod._excluded_weekdays = original

    @pytest.mark.asyncio
    async def test_tuesday_bar_not_skipped(self) -> None:
        """Bars on non-excluded days must proceed normally."""
        import src.main as main_mod
        from src.alerts.dispatcher import AlertDispatcher
        from src.config import RiskSettings
        from src.levels import LevelManager
        from src.main import _build_registry, _process_bar, _SymbolPipeline
        from src.risk.cooldown import CooldownTracker
        from src.risk.manager import RiskManager
        from src.strategies.orb import ORBStrategy
        from src.strategies.regime import RegimeDetector

        original = main_mod._excluded_weekdays
        main_mod._excluded_weekdays = {0}  # Monday only

        try:
            pipeline = _SymbolPipeline(
                symbol="SPY",
                registry=_build_registry(),
                levels=LevelManager(db=None, symbol="SPY"),
                regime=RegimeDetector(),
                strategy=ORBStrategy(excluded_days=[]),
            )
            risk = RiskManager(
                cooldown=CooldownTracker(),
                settings=RiskSettings(
                    account_size=Decimal("50000"),
                    risk_per_trade_pct=Decimal("1.0"),
                    max_daily_loss_pct=Decimal("3.0"),
                    max_trades_per_day=5,
                    max_concurrent_positions=3,
                ),
            )
            alerter = MagicMock()
            alerter.send_signal = AsyncMock(return_value=True)
            alerter.send_risk_warning = AsyncMock(return_value=True)
            alerter.send_status = AsyncMock(return_value=True)
            dispatcher = AlertDispatcher(alerter=alerter)

            import tempfile
            from pathlib import Path

            with tempfile.TemporaryDirectory() as tmpdir:
                db = BarDatabase(db_path=str(Path(tmpdir) / "test.db"))
                db.connect()
                ensure_schema(db.conn)

                bar = _tuesday_bar()
                await _process_bar(bar, pipeline, risk, dispatcher, db, executor=None)

                # Tuesday bar should have been processed — bar_cache populated
                assert len(pipeline.bar_cache) == 1

                db.close()
        finally:
            main_mod._excluded_weekdays = original


# =============================================================================
# P0: In-memory submitted_symbols dedup
# =============================================================================


class TestSubmittedSymbolsDedup:
    """AlpacaExecutor must track submitted symbols locally to prevent duplicates."""

    def _make_executor(self) -> MagicMock:
        """Create a mock executor with the real dedup logic."""
        from src.execution.alpaca_executor import AlpacaExecutor

        with patch.object(AlpacaExecutor, "__init__", lambda self, **kw: None):
            executor = AlpacaExecutor(api_key="", secret_key="", paper=True)
            executor._submitted_symbols = set()
            executor._paper = True
            executor._client = MagicMock()
        return executor

    def test_submitted_symbol_tracked(self) -> None:
        """After a successful bracket order, the symbol is in the local set."""

        executor = self._make_executor()
        # Simulate: no existing position, market open, paper account OK, BP OK
        executor._has_existing_position_or_order = MagicMock(return_value=False)
        executor._check_market_hours = MagicMock(return_value=True)
        executor._assert_paper_account = MagicMock(return_value=True)
        executor._cap_to_buying_power = MagicMock(return_value=100)

        mock_order = MagicMock()
        mock_order.id = "order-123"
        executor._client.submit_order = MagicMock(return_value=mock_order)

        result = executor.submit_bracket_order(
            symbol="AAPL",
            direction=Direction.LONG,
            qty=100,
            stop_price=Decimal("179.00"),
            target_price=Decimal("185.00"),
        )

        assert result is not None
        assert "AAPL" in executor._submitted_symbols

    def test_second_order_same_symbol_blocked(self) -> None:
        """A second bracket order for the same symbol must be blocked locally."""

        executor = self._make_executor()
        executor._submitted_symbols.add("AAPL")

        result = executor.submit_bracket_order(
            symbol="AAPL",
            direction=Direction.LONG,
            qty=100,
            stop_price=Decimal("179.00"),
            target_price=Decimal("185.00"),
        )

        assert result is None
        # Alpaca API should NOT have been called
        executor._client.submit_order.assert_not_called()

    def test_different_symbol_not_blocked(self) -> None:
        """A different symbol should not be blocked by existing entries."""

        executor = self._make_executor()
        executor._submitted_symbols.add("AAPL")
        executor._has_existing_position_or_order = MagicMock(return_value=False)
        executor._check_market_hours = MagicMock(return_value=True)
        executor._assert_paper_account = MagicMock(return_value=True)
        executor._cap_to_buying_power = MagicMock(return_value=100)

        mock_order = MagicMock()
        mock_order.id = "order-456"
        executor._client.submit_order = MagicMock(return_value=mock_order)

        result = executor.submit_bracket_order(
            symbol="MSFT",
            direction=Direction.SHORT,
            qty=50,
            stop_price=Decimal("400.00"),
            target_price=Decimal("390.00"),
        )

        assert result is not None
        assert "MSFT" in executor._submitted_symbols

    def test_clear_submitted_symbols(self) -> None:
        """clear_submitted_symbols should empty the set (EOD reset)."""

        executor = self._make_executor()
        executor._submitted_symbols = {"AAPL", "MSFT", "TSLA"}
        executor.clear_submitted_symbols()
        assert executor._submitted_symbols == set()


# =============================================================================
# P1: Pre-load overnight positions
# =============================================================================


class TestLoadExistingPositions:
    """load_existing_positions must populate the local dedup set at startup."""

    def test_positions_loaded_into_dedup_set(self) -> None:
        from src.execution.alpaca_executor import AlpacaExecutor

        with patch.object(AlpacaExecutor, "__init__", lambda self, **kw: None):
            executor = AlpacaExecutor(api_key="", secret_key="", paper=True)
            executor._submitted_symbols = set()
            executor._client = MagicMock()

        # Mock Alpaca returning 3 existing positions
        pos1 = MagicMock(symbol="HOOD")
        pos2 = MagicMock(symbol="MRVL")
        pos3 = MagicMock(symbol="TSLA")
        executor._client.get_all_positions.return_value = [pos1, pos2, pos3]

        symbols = executor.load_existing_positions()

        assert set(symbols) == {"HOOD", "MRVL", "TSLA"}
        assert executor._submitted_symbols == {"HOOD", "MRVL", "TSLA"}

    def test_api_failure_returns_empty(self) -> None:
        from alpaca.common.exceptions import APIError

        from src.execution.alpaca_executor import AlpacaExecutor

        with patch.object(AlpacaExecutor, "__init__", lambda self, **kw: None):
            executor = AlpacaExecutor(api_key="", secret_key="", paper=True)
            executor._submitted_symbols = set()
            executor._client = MagicMock()

        executor._client.get_all_positions.side_effect = APIError("connection refused")

        symbols = executor.load_existing_positions()

        assert symbols == []
        assert executor._submitted_symbols == set()  # fail-safe: don't block everything

    def test_existing_position_blocks_new_order(self) -> None:
        """After loading positions, the executor must block orders for those symbols."""
        from src.execution.alpaca_executor import AlpacaExecutor

        with patch.object(AlpacaExecutor, "__init__", lambda self, **kw: None):
            executor = AlpacaExecutor(api_key="", secret_key="", paper=True)
            executor._submitted_symbols = set()
            executor._paper = True
            executor._client = MagicMock()

        # Load HOOD as existing
        executor._client.get_all_positions.return_value = [MagicMock(symbol="HOOD")]
        executor.load_existing_positions()

        # Attempt order on HOOD — should be blocked by local check
        result = executor.submit_bracket_order(
            symbol="HOOD",
            direction=Direction.LONG,
            qty=100,
            stop_price=Decimal("20.00"),
            target_price=Decimal("25.00"),
        )

        assert result is None


# =============================================================================
# P1: PDL date resolution
# =============================================================================


class TestPDLDateResolution:
    """PreviousDayLevels must use ET session windows, not UTC midnight-midnight."""

    def test_last_trading_day_skips_weekend(self) -> None:
        """Monday session → previous trading day = Friday."""
        # Monday 2024-01-15
        monday = date(2024, 1, 15)
        assert monday.weekday() == 0  # confirm Monday
        prev = last_trading_day(monday)
        assert prev == date(2024, 1, 12)  # Friday
        assert prev.weekday() == 4

    def test_last_trading_day_tuesday(self) -> None:
        """Tuesday session → previous trading day = Monday (if not a holiday)."""
        # 2024-01-23 is a Tuesday with no holiday on the preceding Monday
        tuesday = date(2024, 1, 23)
        assert tuesday.weekday() == 1
        prev = last_trading_day(tuesday)
        assert prev == date(2024, 1, 22)  # Monday
        assert prev.weekday() == 0

    def test_last_trading_day_skips_holiday(self) -> None:
        """Day after MLK Day (2024-01-16 Tue) → prev = Friday Jan 12.
        MLK Day 2024 = Monday Jan 15."""
        assert not is_trading_day(date(2024, 1, 15))  # MLK Day
        prev = last_trading_day(date(2024, 1, 16))
        assert prev == date(2024, 1, 12)  # Skips MLK Mon → Friday

    def test_pdl_load_uses_et_session_window(self, tmp_db) -> None:
        """PDL query window must be ET 9:30-16:00, not UTC midnight-midnight."""
        # Store a bar for Friday Jan 12, 2024 at 10:00 ET (15:00 UTC in EST)
        bar_ts_et = datetime(2024, 1, 12, 10, 0, tzinfo=_ET)
        bar_ts_utc = bar_ts_et.astimezone(UTC)

        bar = Bar(
            symbol="AAPL",
            timeframe=TimeFrame.ONE_MIN,
            timestamp=bar_ts_utc,
            open=Decimal("180.00"),
            high=Decimal("181.00"),
            low=Decimal("179.00"),
            close=Decimal("180.50"),
            volume=100_000,
            vwap=Decimal("180.20"),
        )
        tmp_db.insert_bars([bar])

        # Load PDL for Tuesday Jan 16 (Monday is MLK Day → skips to Friday)
        pdl = PreviousDayLevels(db=tmp_db, symbol="AAPL")
        pdl.load(date(2024, 1, 16))

        assert pdl.high == Decimal("181.00")
        assert pdl.low == Decimal("179.00")
        assert pdl.close == Decimal("180.50")

    def test_pdl_no_data_for_missing_date(self, tmp_db) -> None:
        """When no bars exist for the previous trading day, levels stay None."""
        pdl = PreviousDayLevels(db=tmp_db, symbol="AAPL")
        pdl.load(date(2024, 1, 16))

        assert pdl.high is None
        assert pdl.low is None
        assert pdl.close is None


# =============================================================================
# P2: EOD reconciliation
# =============================================================================


class TestEODReconciliation:
    """_reconcile_alpaca_fills must populate fill_price AND insert trade rows."""

    @pytest.fixture
    def reconcile_db(self, tmp_path):
        """DB with an executed signal row ready for reconciliation."""
        db = BarDatabase(db_path=str(tmp_path / "reconcile.db"))
        db.connect()
        ensure_schema(db.conn)
        return db

    def _insert_executed_signal(
        self,
        db: BarDatabase,
        symbol: str = "AAPL",
        direction: str = "LONG",
        entry: str = "180.00",
        stop: str = "178.00",
        target: str = "184.00",
        order_id: str = "order-abc-123",
        position_size: int = 100,
    ) -> int:
        """Insert a signal that has been marked as executed."""
        from src.models import RiskDecision

        sig = _make_signal(
            symbol=symbol, direction=Direction.LONG if direction == "LONG" else Direction.SHORT
        )
        sig = sig.model_copy(
            update={
                "entry_price": Decimal(entry),
                "stop_price": Decimal(stop),
                "target_price": Decimal(target),
            }
        )

        decision = RiskDecision(
            approved=True,
            position_size=position_size,
            reason="test",
        )
        signal_id = insert_signal(db.conn, sig, decision)
        mark_signal_executed(db.conn, signal_id, order_id)
        return signal_id

    def _mock_executor_with_fills(
        self,
        filled_orders: list[MagicMock],
    ) -> MagicMock:
        executor = MagicMock()
        executor.get_closed_orders_today.return_value = filled_orders
        return executor

    def test_bracket_leg_filled_populates_fill_price(self, reconcile_db) -> None:
        """When a bracket leg fires, fill_price and pnl must be updated."""
        from src.main import _reconcile_alpaca_fills

        signal_id = self._insert_executed_signal(reconcile_db, order_id="parent-001")

        # Mock parent order with a filled target leg
        parent = MagicMock()
        parent.id = "parent-001"
        parent.symbol = "AAPL"
        parent.filled_avg_price = "180.00"  # entry fill
        parent.order_class = "bracket"

        target_leg = MagicMock()
        target_leg.status = "filled"
        target_leg.filled_avg_price = "184.00"  # target hit

        stop_leg = MagicMock()
        stop_leg.status = "canceled"
        stop_leg.filled_avg_price = None

        parent.legs = [target_leg, stop_leg]

        executor = self._mock_executor_with_fills([parent])
        since = datetime(2024, 1, 16, 0, 0, tzinfo=UTC)

        reconciled = _reconcile_alpaca_fills(executor, reconcile_db, since)

        assert reconciled == 1

        # Verify signal was updated
        sigs = query_executed_signals(reconcile_db.conn, since=since)
        sig = next(s for s in sigs if s["id"] == signal_id)
        assert sig["fill_price"] == "184.0"
        assert sig["outcome"] == "winner"
        assert float(sig["realized_pnl"]) == pytest.approx(400.0, abs=1)  # (184-180)*100

    def test_flatten_order_used_when_legs_cancelled(self, reconcile_db) -> None:
        """When bracket legs are cancelled (EOD flatten), the close order exit is used."""
        from src.main import _reconcile_alpaca_fills

        signal_id = self._insert_executed_signal(
            reconcile_db, order_id="parent-002", entry="180.00"
        )

        # Parent order — both legs cancelled (EOD flatten)
        parent = MagicMock()
        parent.id = "parent-002"
        parent.symbol = "AAPL"
        parent.filled_avg_price = "180.00"
        parent.order_class = "bracket"

        cancelled_leg1 = MagicMock()
        cancelled_leg1.status = "canceled"
        cancelled_leg1.filled_avg_price = None

        cancelled_leg2 = MagicMock()
        cancelled_leg2.status = "canceled"
        cancelled_leg2.filled_avg_price = None

        parent.legs = [cancelled_leg1, cancelled_leg2]

        # Flatten close order — separate simple order for the same symbol
        flatten_order = MagicMock()
        flatten_order.id = "flatten-aapl-001"
        flatten_order.symbol = "AAPL"
        flatten_order.filled_avg_price = "179.50"
        flatten_order.order_class = "simple"
        flatten_order.status = "filled"

        executor = self._mock_executor_with_fills([parent, flatten_order])
        since = datetime(2024, 1, 16, 0, 0, tzinfo=UTC)

        reconciled = _reconcile_alpaca_fills(executor, reconcile_db, since)

        assert reconciled == 1

        sigs = query_executed_signals(reconcile_db.conn, since=since)
        sig = next(s for s in sigs if s["id"] == signal_id)
        assert sig["fill_price"] == "179.5"
        assert sig["outcome"] == "loser"

    def test_trade_row_inserted(self, reconcile_db) -> None:
        """A row must be inserted into the trades table after reconciliation."""
        from src.main import _reconcile_alpaca_fills

        self._insert_executed_signal(reconcile_db, order_id="parent-003")

        parent = MagicMock()
        parent.id = "parent-003"
        parent.symbol = "AAPL"
        parent.filled_avg_price = "180.00"
        parent.order_class = "bracket"

        leg = MagicMock()
        leg.status = "filled"
        leg.filled_avg_price = "184.00"
        parent.legs = [leg]

        executor = self._mock_executor_with_fills([parent])
        since = datetime(2024, 1, 16, 0, 0, tzinfo=UTC)

        _reconcile_alpaca_fills(executor, reconcile_db, since)

        trades = reconcile_db.conn.execute("SELECT * FROM trades").fetchall()
        assert len(trades) >= 1
        trade = dict(trades[0])
        assert trade["symbol"] == "AAPL"
        assert float(trade["pnl"]) == pytest.approx(400.0, abs=1)

    def test_already_reconciled_signal_skipped(self, reconcile_db) -> None:
        """Signals with realized_pnl already set must not be double-reconciled."""
        from src.main import _reconcile_alpaca_fills

        signal_id = self._insert_executed_signal(reconcile_db, order_id="parent-004")

        # Manually set realized_pnl to simulate prior reconciliation
        reconcile_db.conn.execute(
            "UPDATE signals SET realized_pnl = '400.0' WHERE id = ?", (signal_id,)
        )
        reconcile_db.conn.commit()

        parent = MagicMock()
        parent.id = "parent-004"
        parent.symbol = "AAPL"
        parent.filled_avg_price = "180.00"
        parent.order_class = "bracket"
        parent.legs = []

        executor = self._mock_executor_with_fills([parent])
        since = datetime(2024, 1, 16, 0, 0, tzinfo=UTC)

        reconciled = _reconcile_alpaca_fills(executor, reconcile_db, since)

        assert reconciled == 0  # already reconciled, should skip

    def test_no_filled_orders_returns_zero(self, reconcile_db) -> None:
        """When Alpaca returns no filled orders, reconciliation returns 0."""
        from src.main import _reconcile_alpaca_fills

        executor = self._mock_executor_with_fills([])
        since = datetime(2024, 1, 16, 0, 0, tzinfo=UTC)

        reconciled = _reconcile_alpaca_fills(executor, reconcile_db, since)

        assert reconciled == 0
