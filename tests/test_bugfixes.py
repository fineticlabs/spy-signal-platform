"""Tests for bug fixes: PDL trading calendar, position guard, volume filter."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

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

        strategy = ORBStrategy()
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

        # Volume 2M >= 1.5 * 1M → breakout confirmed
        ts = datetime(2024, 1, 15, 10, 0, tzinfo=et).astimezone(UTC)
        bar = Bar(
            symbol="SPY",
            timeframe=TimeFrame.ONE_MIN,
            timestamp=ts,
            open=Decimal("486.00"),
            high=Decimal("486.50"),
            low=Decimal("485.50"),
            close=Decimal("486.00"),
            volume=2_000_000,
            vwap=Decimal("486.00"),
        )
        signal = strategy.evaluate(bar, indicators, levels, regime)
        assert signal is not None
        assert signal.direction.name == "LONG"
