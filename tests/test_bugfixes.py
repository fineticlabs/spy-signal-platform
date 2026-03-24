"""Tests for bug fixes: PDL trading calendar, position guard, volume filter."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.levels.trading_calendar import is_trading_day, last_trading_day

# ── Bug 1: PDL resolves to last trading day ─────────────────────────────────


class TestLastTradingDay:
    """Verify last_trading_day skips weekends and US market holidays."""

    def test_monday_resolves_to_friday(self) -> None:
        """When today is Monday, prev trading day is Friday."""
        monday = date(2026, 3, 23)  # Monday
        assert monday.weekday() == 0
        result = last_trading_day(monday)
        assert result == date(2026, 3, 20)  # Friday
        assert result.weekday() == 4

    def test_tuesday_resolves_to_monday(self) -> None:
        """When today is Tuesday, prev trading day is Monday."""
        tuesday = date(2026, 3, 24)
        assert tuesday.weekday() == 1
        result = last_trading_day(tuesday)
        assert result == date(2026, 3, 23)  # Monday
        assert result.weekday() == 0

    def test_sunday_resolves_to_friday(self) -> None:
        """When today is Sunday, prev trading day is Friday."""
        sunday = date(2026, 3, 22)
        assert sunday.weekday() == 6
        result = last_trading_day(sunday)
        assert result == date(2026, 3, 20)  # Friday

    def test_friday_holiday_resolves_to_thursday(self) -> None:
        """When Friday is Good Friday (holiday), Monday resolves to Thursday."""
        # Good Friday 2026 is April 3
        monday_after_good_friday = date(2026, 4, 6)
        result = last_trading_day(monday_after_good_friday)
        assert result == date(2026, 4, 2)  # Thursday before Good Friday
        assert result.weekday() == 3

    def test_day_after_holiday_skips_holiday(self) -> None:
        """Day after July 4th (observed) skips the holiday."""
        # July 4, 2026 is Saturday → observed Friday July 3
        monday_jul6 = date(2026, 7, 6)
        result = last_trading_day(monday_jul6)
        assert result == date(2026, 7, 2)  # Thursday (Friday=observed holiday)

    def test_new_years_day_2026(self) -> None:
        """Jan 2 2026 (Fri) → prev trading day is Dec 31 2025."""
        jan2 = date(2026, 1, 2)
        result = last_trading_day(jan2)
        assert result == date(2025, 12, 31)

    def test_is_trading_day_weekday(self) -> None:
        """Regular Tuesday is a trading day."""
        assert is_trading_day(date(2026, 3, 24)) is True

    def test_is_trading_day_weekend(self) -> None:
        """Saturday is not a trading day."""
        assert is_trading_day(date(2026, 3, 21)) is False

    def test_is_trading_day_holiday(self) -> None:
        """Christmas (observed) is not a trading day."""
        assert is_trading_day(date(2025, 12, 25)) is False


class TestPDLUsesLastTradingDay:
    """Verify PreviousDayLevels calls last_trading_day instead of today - 1."""

    def test_load_on_monday_queries_friday(self) -> None:
        """Loading on Monday should query Friday's bars, not Sunday's."""
        from src.levels.daily_levels import PreviousDayLevels

        db = MagicMock()
        db.query_bars.return_value = []

        pdl = PreviousDayLevels(db, symbol="SPY")
        monday = date(2026, 3, 23)
        pdl.load(monday)

        call_args = db.query_bars.call_args
        start_utc = call_args.kwargs.get("start") or call_args[1].get("start")
        # The start date should be Friday March 20, not Sunday March 22
        assert start_utc.date() == date(2026, 3, 20)


# ── Bug 2: Position guard prevents duplicate orders ─────────────────────────


def _mock_account(
    buying_power: str = "100000",
    account_number: str = "PA12345678",
) -> SimpleNamespace:
    return SimpleNamespace(
        buying_power=buying_power,
        equity="100000",
        cash="100000",
        account_number=account_number,
        daytrade_count=0,
        pattern_day_trader=False,
    )


def _mock_order(symbol: str = "SPY", order_id: str = "test-123") -> SimpleNamespace:
    return SimpleNamespace(
        id=order_id,
        symbol=symbol,
        side="buy",
        qty="100",
        order_class="bracket",
        status="accepted",
    )


def _mock_position(symbol: str = "SPY") -> SimpleNamespace:
    return SimpleNamespace(
        symbol=symbol,
        qty="100",
        side="long",
        avg_entry_price="480.00",
        current_price="482.00",
        unrealized_pl="200.00",
    )


def _make_executor():
    """Create an AlpacaExecutor with a mocked TradingClient."""
    from src.execution.alpaca_executor import AlpacaExecutor

    with patch("src.execution.alpaca_executor.TradingClient"):
        executor = AlpacaExecutor(
            api_key="test-key",
            secret_key="test-secret",  # noqa: S106
            paper=True,
        )
    executor._client = MagicMock()
    return executor


class TestPositionGuard:
    """submit_bracket_order skips when position/order already exists."""

    def test_skips_when_position_exists(self) -> None:
        """Signal is skipped when an open position exists for the symbol."""
        from src.models import Direction

        executor = _make_executor()
        executor._client.get_all_positions.return_value = [_mock_position("HOOD")]
        executor._client.get_orders.return_value = []

        with patch.object(executor, "_check_market_hours", return_value=True):
            order = executor.submit_bracket_order(
                symbol="HOOD",
                direction=Direction.LONG,
                qty=50,
                stop_price=Decimal("20.00"),
                target_price=Decimal("30.00"),
            )

        assert order is None
        executor._client.submit_order.assert_not_called()

    def test_skips_when_pending_order_exists(self) -> None:
        """Signal is skipped when a pending order exists for the symbol."""
        from src.models import Direction

        executor = _make_executor()
        executor._client.get_all_positions.return_value = []
        executor._client.get_orders.return_value = [_mock_order(symbol="TSLA")]

        with patch.object(executor, "_check_market_hours", return_value=True):
            order = executor.submit_bracket_order(
                symbol="TSLA",
                direction=Direction.LONG,
                qty=25,
                stop_price=Decimal("170.00"),
                target_price=Decimal("180.00"),
            )

        assert order is None
        executor._client.submit_order.assert_not_called()

    def test_allows_when_no_existing_position_or_order(self) -> None:
        """Signal proceeds when no existing position or order for the symbol."""
        from src.models import Direction

        executor = _make_executor()
        executor._client.get_all_positions.return_value = [_mock_position("AAPL")]
        executor._client.get_orders.return_value = [_mock_order(symbol="AAPL")]
        executor._client.get_account.return_value = _mock_account()
        executor._client.submit_order.return_value = _mock_order(symbol="ROKU")

        with patch.object(executor, "_check_market_hours", return_value=True):
            order = executor.submit_bracket_order(
                symbol="ROKU",
                direction=Direction.LONG,
                qty=50,
                stop_price=Decimal("60.00"),
                target_price=Decimal("80.00"),
            )

        assert order is not None
        executor._client.submit_order.assert_called_once()

    def test_fails_closed_on_position_api_error(self) -> None:
        """If position check API fails, order is blocked (fail-closed)."""
        from alpaca.common.exceptions import APIError

        from src.models import Direction

        executor = _make_executor()
        executor._client.get_all_positions.side_effect = APIError("timeout")

        with patch.object(executor, "_check_market_hours", return_value=True):
            order = executor.submit_bracket_order(
                symbol="SPY",
                direction=Direction.LONG,
                qty=100,
                stop_price=Decimal("478.00"),
                target_price=Decimal("486.00"),
            )

        assert order is None


# ── Bug 3: Volume filter rejects low-volume signals ─────────────────────────


class TestVolumeFilterRejectsLowVolume:
    """ORB strategy rejects signals when volume < 0.5 * avg."""

    def _setup_strategy(self, avg_volume: int = 1_000_000):
        """Create an ORBStrategy primed with a volume baseline."""
        from src.strategies.orb import ORBStrategy

        strategy = ORBStrategy(excluded_days=[], signal_cutoff_et="15:45")
        # Prime with baseline volume data (ORB incomplete, no signals fire)
        from src.models import IndicatorSnapshot, LevelSnapshot
        from src.strategies.regime import RegimeDetector

        levels = LevelSnapshot(orb_complete=False)
        indicators = IndicatorSnapshot(atr=Decimal("2.0"))
        regime = RegimeDetector()
        regime.update(vix=Decimal("18"), adx=Decimal("28"), trending_up=True)

        from datetime import UTC, datetime
        from zoneinfo import ZoneInfo

        et = ZoneInfo("America/New_York")

        for i in range(20):
            from src.models import Bar, TimeFrame

            naive_et = datetime(2024, 1, 15, 9, 36 + i, tzinfo=et)
            ts_utc = naive_et.astimezone(UTC)
            bar = Bar(
                symbol="DASH",
                timeframe=TimeFrame.ONE_MIN,
                timestamp=ts_utc,
                open=Decimal("482.00"),
                high=Decimal("482.50"),
                low=Decimal("481.50"),
                close=Decimal("482.00"),
                volume=avg_volume,
                vwap=Decimal("482.00"),
            )
            strategy.evaluate(bar, indicators, levels, regime)

        return strategy

    def test_rejects_volume_below_half_average(self) -> None:
        """Signal is rejected when volume < 0.5 * avg."""
        from datetime import UTC, datetime
        from zoneinfo import ZoneInfo

        from src.models import Bar, IndicatorSnapshot, LevelSnapshot, TimeFrame
        from src.strategies.regime import RegimeDetector

        strategy = self._setup_strategy(avg_volume=1_000_000)
        et = ZoneInfo("America/New_York")

        levels = LevelSnapshot(
            orb_complete=True,
            orb_high=Decimal("485.00"),
            orb_low=Decimal("480.00"),
        )
        indicators = IndicatorSnapshot(atr=Decimal("2.0"))
        regime = RegimeDetector()
        regime.update(vix=Decimal("18"), adx=Decimal("28"), trending_up=True)

        # Volume 400K < 0.5 * 1M = 500K → should be rejected
        ts = datetime(2024, 1, 15, 10, 0, tzinfo=et).astimezone(UTC)
        bar = Bar(
            symbol="DASH",
            timeframe=TimeFrame.ONE_MIN,
            timestamp=ts,
            open=Decimal("486.00"),
            high=Decimal("486.50"),
            low=Decimal("485.50"),
            close=Decimal("486.00"),
            volume=400_000,
            vwap=Decimal("486.00"),
        )
        signal = strategy.evaluate(bar, indicators, levels, regime)
        assert signal is None

    def test_allows_volume_above_breakout_threshold(self) -> None:
        """Signal fires when volume >= 1.5 * avg."""
        from datetime import UTC, datetime
        from zoneinfo import ZoneInfo

        from src.models import Bar, IndicatorSnapshot, LevelSnapshot, TimeFrame
        from src.strategies.regime import RegimeDetector

        strategy = self._setup_strategy(avg_volume=1_000_000)
        et = ZoneInfo("America/New_York")

        levels = LevelSnapshot(
            orb_complete=True,
            orb_high=Decimal("485.00"),
            orb_low=Decimal("480.00"),
        )
        indicators = IndicatorSnapshot(atr=Decimal("2.0"))
        regime = RegimeDetector()
        regime.update(vix=Decimal("18"), adx=Decimal("28"), trending_up=True)

        # Volume 2M >= 1.5 * 1M → breakout confirmed (2-bar confirmation needed)
        ts1 = datetime(2024, 1, 15, 9, 56, tzinfo=et).astimezone(UTC)
        bar1 = Bar(
            symbol="SPY",
            timeframe=TimeFrame.ONE_MIN,
            timestamp=ts1,
            open=Decimal("486.00"),
            high=Decimal("486.50"),
            low=Decimal("485.50"),
            close=Decimal("486.00"),
            volume=2_000_000,
            vwap=Decimal("486.00"),
        )
        strategy.evaluate(bar1, indicators, levels, regime)  # first bar: pending

        ts2 = datetime(2024, 1, 15, 9, 57, tzinfo=et).astimezone(UTC)
        bar2 = Bar(
            symbol="SPY",
            timeframe=TimeFrame.ONE_MIN,
            timestamp=ts2,
            open=Decimal("486.00"),
            high=Decimal("486.50"),
            low=Decimal("485.50"),
            close=Decimal("486.00"),
            volume=2_000_000,
            vwap=Decimal("486.00"),
        )
        signal = strategy.evaluate(bar2, indicators, levels, regime)  # second bar: confirmed
        assert signal is not None
        assert signal.direction.name == "LONG"


# ── Execution tracking: signal marked executed after order ───────────────────


class TestSignalExecutionTracking:
    """Signal DB row is updated with order_id after successful bracket order."""

    def test_mark_signal_executed(self, tmp_path) -> None:
        """mark_signal_executed sets executed=1 and stores order_id."""
        import sqlite3

        from src.storage.queries import (
            ensure_schema,
            insert_signal,
            mark_signal_executed,
            query_executed_signals,
        )

        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        ensure_schema(conn)

        # Create a minimal signal + decision
        from datetime import UTC, datetime

        from src.models import (
            Direction,
            Regime,
            RiskDecision,
            Signal,
            TimeFrame,
        )

        sig = Signal(
            symbol="TSLA",
            direction=Direction.LONG,
            strategy_name="ORB-5min",
            entry_price=Decimal("180.00"),
            stop_price=Decimal("175.00"),
            target_price=Decimal("190.00"),
            risk_reward_ratio=Decimal("2.0"),
            confidence_score=3,
            reason="test",
            timeframe=TimeFrame.ONE_MIN,
            regime=Regime.RANGING,
            timestamp=datetime.now(UTC),
        )
        decision = RiskDecision(approved=True, position_size=50, reason="approved")
        signal_id = insert_signal(conn, sig, decision)

        # Mark as executed
        mark_signal_executed(conn, signal_id, "alpaca-order-abc123")

        # Query executed signals
        since = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        executed = query_executed_signals(conn, since)
        assert len(executed) == 1
        assert executed[0]["order_id"] == "alpaca-order-abc123"
        assert executed[0]["executed"] == 1

        conn.close()

    def test_update_signal_fill(self, tmp_path) -> None:
        """update_signal_fill stores fill_price, realized_pnl, outcome."""
        import sqlite3

        from src.storage.queries import (
            ensure_schema,
            insert_signal,
            mark_signal_executed,
            update_signal_fill,
        )

        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        ensure_schema(conn)

        from datetime import UTC, datetime

        from src.models import (
            Direction,
            Regime,
            RiskDecision,
            Signal,
            TimeFrame,
        )

        sig = Signal(
            symbol="HOOD",
            direction=Direction.LONG,
            strategy_name="ORB-5min",
            entry_price=Decimal("25.00"),
            stop_price=Decimal("23.00"),
            target_price=Decimal("29.00"),
            risk_reward_ratio=Decimal("2.0"),
            confidence_score=3,
            reason="test",
            timeframe=TimeFrame.ONE_MIN,
            regime=Regime.RANGING,
            timestamp=datetime.now(UTC),
        )
        decision = RiskDecision(approved=True, position_size=200, reason="approved")
        signal_id = insert_signal(conn, sig, decision)
        mark_signal_executed(conn, signal_id, "order-xyz")

        update_signal_fill(conn, signal_id, "29.00", "800.00", "winner")

        row = dict(conn.execute("SELECT * FROM signals WHERE id = ?", (signal_id,)).fetchone())
        assert row["fill_price"] == "29.00"
        assert row["realized_pnl"] == "800.00"
        assert row["outcome"] == "winner"

        conn.close()


# ── EOD summary uses real execution data ─────────────────────────────────────


class TestEodSummaryExecutionData:
    """EOD summary correctly counts executed trades and shows real P&L."""

    @pytest.mark.asyncio
    async def test_eod_counts_executed_from_signals_table(self) -> None:
        """Executed count comes from signals.executed=1, not trades table."""
        from unittest.mock import AsyncMock

        from src.main import _send_eod_status

        dispatcher = AsyncMock()
        db = MagicMock()
        cooldown = MagicMock()

        pipeline = MagicMock()
        pipeline.levels._orb.is_complete = True
        pipelines = {"SPY": pipeline}

        # 3 approved signals, 2 were executed
        signals = [
            {"approved": 1, "reject_reason": ""},
            {"approved": 1, "reject_reason": ""},
            {"approved": 1, "reject_reason": ""},
        ]
        executed_signals = [{"id": 1}, {"id": 2}]
        trades: list[dict] = []  # trades table empty (pre-reconciliation)

        outcomes = {"winners": [], "losers": [], "open": [], "no_data": []}

        with (
            patch("src.main.query_recent_signals", return_value=signals),
            patch("src.main.query_recent_trades", return_value=trades),
            patch("src.main.query_executed_signals", return_value=executed_signals),
            patch("src.main._evaluate_signal_outcomes", return_value=outcomes),
        ):
            await _send_eod_status(dispatcher, db, pipelines, cooldown)

        msg = dispatcher.dispatch_status.call_args[0][0]
        assert "Executed: 2/3 signals" in msg
        assert "Skipped: 1" in msg

    @pytest.mark.asyncio
    async def test_eod_pnl_from_alpaca_executor(self) -> None:
        """When executor is present, P&L comes from Alpaca account."""
        from unittest.mock import AsyncMock

        from src.main import _send_eod_status

        dispatcher = AsyncMock()
        db = MagicMock()
        cooldown = MagicMock()

        pipeline = MagicMock()
        pipeline.levels._orb.is_complete = True
        pipelines = {"SPY": pipeline}

        signals = [{"approved": 1, "reject_reason": ""}]
        executed_signals = [{"id": 1}]

        outcomes = {"winners": [], "losers": [], "open": [], "no_data": []}

        mock_executor = MagicMock()
        mock_executor.get_daily_pnl.return_value = Decimal("-2213.88")
        mock_executor.get_closed_orders_today.return_value = []

        with (
            patch("src.main.query_recent_signals", return_value=signals),
            patch("src.main.query_recent_trades", return_value=[]),
            patch("src.main.query_executed_signals", return_value=executed_signals),
            patch("src.main._evaluate_signal_outcomes", return_value=outcomes),
        ):
            await _send_eod_status(
                dispatcher,
                db,
                pipelines,
                cooldown,
                executor=mock_executor,
            )

        msg = dispatcher.dispatch_status.call_args[0][0]
        assert "Actual P&L: $-2213.88" in msg

    @pytest.mark.asyncio
    async def test_eod_reconciliation_populates_trades(self) -> None:
        """Reconciliation inserts trade rows for executed signals."""
        from src.main import _reconcile_alpaca_fills

        db = MagicMock()
        db.conn = MagicMock()
        db.conn.commit = MagicMock()

        mock_executor = MagicMock()
        # One filled order
        filled_order = SimpleNamespace(
            id="order-abc",
            symbol="TSLA",
            status="filled",
            filled_avg_price="180.50",
            legs=[],
        )
        mock_executor.get_closed_orders_today.return_value = [filled_order]

        from datetime import UTC, datetime

        since = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        executed_signal = {
            "id": 42,
            "order_id": "order-abc",
            "symbol": "TSLA",
            "direction": "LONG",
            "entry_price": "180.00",
            "stop_price": "175.00",
            "target_price": "190.00",
            "position_size": 25,
            "realized_pnl": None,
            "timestamp": datetime.now(UTC).isoformat(),
            "strategy_name": "ORB-5min",
        }

        with patch("src.main.query_executed_signals", return_value=[executed_signal]):
            reconciled = _reconcile_alpaca_fills(mock_executor, db, since)

        assert reconciled == 1
        # Verify trade was inserted into trades table
        db.conn.execute.assert_called()
        db.conn.commit.assert_called()


# ── Monday filter: live scanner skips excluded days ──────────────────────────


class TestMondayFilterLive:
    """ORB strategy skips all evaluation on excluded weekdays."""

    def _make_bar_on_day(self, weekday_date: str, close: float = 486.0, volume: int = 2_000_000):
        """Create a bar on the given date at 10:00 ET."""
        from datetime import UTC, datetime
        from zoneinfo import ZoneInfo

        from src.models import Bar, TimeFrame

        et = ZoneInfo("America/New_York")
        ts = datetime.fromisoformat(f"{weekday_date}T10:00:00").replace(tzinfo=et).astimezone(UTC)
        return Bar(
            symbol="SPY",
            timeframe=TimeFrame.ONE_MIN,
            timestamp=ts,
            open=Decimal(str(close)),
            high=Decimal(str(close + 0.5)),
            low=Decimal(str(close - 0.5)),
            close=Decimal(str(close)),
            volume=volume,
            vwap=Decimal(str(close)),
        )

    def _make_deps(self):
        from src.models import IndicatorSnapshot, LevelSnapshot
        from src.strategies.regime import RegimeDetector

        levels = LevelSnapshot(
            orb_complete=True,
            orb_high=Decimal("485.00"),
            orb_low=Decimal("480.00"),
        )
        indicators = IndicatorSnapshot(atr=Decimal("2.0"))
        regime = RegimeDetector()
        regime.update(vix=Decimal("18"), adx=Decimal("28"), trending_up=True)
        return indicators, levels, regime

    def test_monday_returns_none(self) -> None:
        """Signal is None on Monday when excluded_days=[0]."""
        from zoneinfo import ZoneInfo

        from src.strategies.orb import ORBStrategy

        strategy = ORBStrategy(excluded_days=[0])
        indicators, levels, regime = self._make_deps()

        # 2026-03-23 is a Monday
        bar = self._make_bar_on_day("2026-03-23")
        assert bar.timestamp.astimezone(ZoneInfo("America/New_York")).date().weekday() == 0

        signal = strategy.evaluate(bar, indicators, levels, regime)
        assert signal is None

    def test_tuesday_trades_normally(self) -> None:
        """Signal fires on Tuesday when excluded_days=[0]."""
        from src.models import LevelSnapshot
        from src.strategies.orb import ORBStrategy

        strategy = ORBStrategy(excluded_days=[0], signal_cutoff_et="15:45")

        # Prime with volume data on a Tuesday
        indicators, levels_incomplete, regime = self._make_deps()
        levels_incomplete = LevelSnapshot(orb_complete=False)
        for _i in range(20):
            bar = self._make_bar_on_day("2026-03-24", close=482.0, volume=1_000_000)
            strategy.evaluate(bar, indicators, levels_incomplete, regime)

        indicators, levels, regime = self._make_deps()
        # 2026-03-24 is a Tuesday — needs 2 bars for confirmation
        bar1 = self._make_bar_on_day("2026-03-24", close=486.0, volume=2_000_000)
        strategy.evaluate(bar1, indicators, levels, regime)  # pending
        bar2 = self._make_bar_on_day("2026-03-24", close=486.0, volume=2_000_000)
        signal = strategy.evaluate(bar2, indicators, levels, regime)  # confirmed
        assert signal is not None
        assert signal.direction.name == "LONG"

    def test_excluded_days_empty_allows_monday(self) -> None:
        """When excluded_days=[], Monday trades are allowed."""
        from src.models import LevelSnapshot
        from src.strategies.orb import ORBStrategy

        strategy = ORBStrategy(excluded_days=[], signal_cutoff_et="15:45")

        # Prime
        indicators, levels_incomplete, regime = self._make_deps()
        levels_incomplete = LevelSnapshot(orb_complete=False)
        for _i in range(20):
            bar = self._make_bar_on_day("2026-03-23", close=482.0, volume=1_000_000)
            strategy.evaluate(bar, indicators, levels_incomplete, regime)

        indicators, levels, regime = self._make_deps()
        bar1 = self._make_bar_on_day("2026-03-23", close=486.0, volume=2_000_000)
        strategy.evaluate(bar1, indicators, levels, regime)  # pending
        bar2 = self._make_bar_on_day("2026-03-23", close=486.0, volume=2_000_000)
        signal = strategy.evaluate(bar2, indicators, levels, regime)  # confirmed
        assert signal is not None

    def test_config_excluded_days_default_is_monday(self) -> None:
        """AppSettings.excluded_days defaults to [0] (Monday)."""
        from src.config import AppSettings

        settings = AppSettings(_env_file=None)  # type: ignore[call-arg]
        assert settings.excluded_days == [0]

    def test_config_excluded_days_from_string(self) -> None:
        """excluded_days can be parsed from comma-separated string."""
        from src.config import AppSettings

        settings = AppSettings(_env_file=None, excluded_days="0,4")  # type: ignore[call-arg]
        assert settings.excluded_days == [0, 4]


# ── Position scale factor ────────────────────────────────────────────────────


class TestPositionScaleFactor:
    """position_scale_factor reduces position sizes in both live and backtest."""

    def test_live_default_scale_produces_25pct(self) -> None:
        """Default 0.25 scale factor gives 25% of base size."""
        from src.risk.position_sizing import calculate_position_size

        base = calculate_position_size(
            account_size=Decimal("50000"),
            risk_pct=Decimal("1.0"),
            entry=Decimal("100.00"),
            stop=Decimal("99.00"),
            scale_factor=1.0,
        )
        scaled = calculate_position_size(
            account_size=Decimal("50000"),
            risk_pct=Decimal("1.0"),
            entry=Decimal("100.00"),
            stop=Decimal("99.00"),
            scale_factor=0.25,
        )
        assert base == 500  # $500 risk / $1 risk-per-share
        assert scaled == 125  # 500 * 0.25
        assert scaled == base // 4

    def test_live_scale_factor_1_gives_original(self) -> None:
        """Scale factor of 1.0 gives original behavior."""
        from src.risk.position_sizing import calculate_position_size

        original = calculate_position_size(
            account_size=Decimal("50000"),
            risk_pct=Decimal("1.0"),
            entry=Decimal("480.00"),
            stop=Decimal("477.00"),
        )
        scaled = calculate_position_size(
            account_size=Decimal("50000"),
            risk_pct=Decimal("1.0"),
            entry=Decimal("480.00"),
            stop=Decimal("477.00"),
            scale_factor=1.0,
        )
        assert original == scaled

    def test_live_scale_applied_before_bp_cap(self) -> None:
        """Scaled size should fit within buying power without downsizing."""
        from src.risk.position_sizing import calculate_position_size

        # At $480, base size=166 shares → $79,680 position value
        # With 0.25 scale → 41 shares → $19,680 position value (fits in $50K BP)
        base = calculate_position_size(
            account_size=Decimal("50000"),
            risk_pct=Decimal("1.0"),
            entry=Decimal("480.00"),
            stop=Decimal("477.00"),
            scale_factor=1.0,
        )
        scaled = calculate_position_size(
            account_size=Decimal("50000"),
            risk_pct=Decimal("1.0"),
            entry=Decimal("480.00"),
            stop=Decimal("477.00"),
            scale_factor=0.25,
        )
        assert base == 166
        assert scaled == 41
        # Scaled position value fits in 50% of $100K buying power
        assert scaled * 480 < 50000

    def test_risk_manager_uses_scale_factor(self) -> None:
        """RiskManager passes scale_factor from RiskSettings to position sizing."""
        from datetime import UTC, datetime

        from src.config import RiskSettings
        from src.models import Direction, Regime, Signal, TimeFrame
        from src.risk.cooldown import CooldownTracker
        from src.risk.manager import RiskManager

        settings = RiskSettings(
            account_size=Decimal("50000"),
            risk_per_trade_pct=Decimal("1.0"),
            position_scale_factor=0.25,
        )
        cooldown = CooldownTracker()
        manager = RiskManager(cooldown=cooldown, settings=settings)

        signal = Signal(
            symbol="SPY",
            direction=Direction.LONG,
            strategy_name="ORB-5min",
            entry_price=Decimal("100.00"),
            stop_price=Decimal("99.00"),
            target_price=Decimal("103.00"),
            risk_reward_ratio=Decimal("3.0"),
            confidence_score=3,
            reason="test",
            timeframe=TimeFrame.ONE_MIN,
            regime=Regime.TRENDING_UP,
            vix=Decimal("18"),
            adx=Decimal("28"),
            timestamp=datetime(2024, 1, 15, 14, 36, tzinfo=UTC),
        )

        decision = manager.approve(signal)
        assert decision.approved
        # Base would be 500 (50000 * 0.01 / 1.00), scaled to 125
        assert decision.position_size == 125

    def test_config_default_is_025(self) -> None:
        """RiskSettings default position_scale_factor is 0.25."""
        from src.config import RiskSettings

        settings = RiskSettings()
        assert settings.position_scale_factor == 0.25

    def test_config_scale_from_env(self, monkeypatch) -> None:
        """position_scale_factor can be set via env var."""
        monkeypatch.setenv("POSITION_SCALE_FACTOR", "0.5")
        from src.config import RiskSettings

        settings = RiskSettings()
        assert settings.position_scale_factor == 0.5

    def test_backtest_engine_has_scale_param(self) -> None:
        """Backtest ORBStrategy has position_scale_factor attribute."""
        from src.backtest.engine import ORBStrategy as BacktestORB

        assert hasattr(BacktestORB, "position_scale_factor")
        assert BacktestORB.position_scale_factor == 0.25

    def test_both_paths_same_config(self) -> None:
        """Both live and backtest use the same default scale factor value."""
        from src.backtest.engine import ORBStrategy as BacktestORB
        from src.config import RiskSettings

        settings = RiskSettings()
        assert settings.position_scale_factor == BacktestORB.position_scale_factor
