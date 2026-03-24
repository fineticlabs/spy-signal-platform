"""Tests targeting remaining small coverage gaps across multiple source files."""

from __future__ import annotations

import sqlite3
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.models import (
    Bar,
    Direction,
    Regime,
    RiskDecision,
    Signal,
    TimeFrame,
    TradeResult,
)

_TEST_SECRET = "s"  # noqa: S105 — fake credential for test mocks

# ── helpers ──────────────────────────────────────────────────────────────────


def _make_signal(**overrides: object) -> Signal:
    defaults: dict = dict(
        symbol="SPY",
        direction=Direction.LONG,
        strategy_name="orb_breakout",
        entry_price=Decimal("480.00"),
        stop_price=Decimal("478.00"),
        target_price=Decimal("484.00"),
        risk_reward_ratio=Decimal("2.0"),
        confidence_score=3,
        reason="Test signal",
        timeframe=TimeFrame.ONE_MIN,
        regime=Regime.TRENDING_UP,
        timestamp=datetime(2024, 1, 15, 14, 35, tzinfo=UTC),
    )
    defaults.update(overrides)
    return Signal(**defaults)


def _make_risk_decision(**overrides: object) -> RiskDecision:
    defaults: dict = dict(approved=True, reason="approved", position_size=100)
    defaults.update(overrides)
    return RiskDecision(**defaults)


def _make_trade_result(**overrides: object) -> TradeResult:
    defaults: dict = dict(
        signal=_make_signal(),
        pnl=Decimal("150.00"),
        timestamp=datetime(2024, 1, 15, 20, 0, tzinfo=UTC),
    )
    defaults.update(overrides)
    return TradeResult(**defaults)


def _make_bar(
    ts: datetime | None = None,
    high: Decimal = Decimal("481.50"),
    low: Decimal = Decimal("479.80"),
    close: Decimal = Decimal("481.00"),
    volume: int = 100_000,
) -> Bar:
    return Bar(
        symbol="SPY",
        timeframe=TimeFrame.ONE_MIN,
        timestamp=ts or datetime(2024, 1, 15, 14, 35, tzinfo=UTC),
        open=Decimal("480.00"),
        high=high,
        low=low,
        close=close,
        volume=volume,
        vwap=Decimal("480.55"),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# AlertDispatcher — daily summary failure path (line 125 / 109-113)
# ═══════════════════════════════════════════════════════════════════════════════


class TestAlertDispatcher:
    @pytest.mark.asyncio
    async def test_dispatch_daily_summary_failure(self) -> None:
        from src.alerts.dispatcher import AlertDispatcher

        alerter = AsyncMock()
        alerter.send_daily_summary = AsyncMock(return_value=False)
        dispatcher = AlertDispatcher(alerter=alerter)

        result = await dispatcher.dispatch_daily_summary([_make_trade_result()])

        assert result is False
        alerter.send_daily_summary.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_dispatch_daily_summary_success(self) -> None:
        from src.alerts.dispatcher import AlertDispatcher

        alerter = AsyncMock()
        alerter.send_daily_summary = AsyncMock(return_value=True)
        dispatcher = AlertDispatcher(alerter=alerter)

        result = await dispatcher.dispatch_daily_summary([_make_trade_result()])

        assert result is True


# ═══════════════════════════════════════════════════════════════════════════════
# API routes — set_dependencies (lines 56-61)
# ═══════════════════════════════════════════════════════════════════════════════


class TestApiRoutes:
    def test_set_dependencies_sets_globals(self) -> None:
        from src.api import routes
        from src.api.routes import set_dependencies

        conn = MagicMock(spec=sqlite3.Connection)
        registry = MagicMock()
        levels = MagicMock()
        regime = MagicMock()
        cooldown = MagicMock()

        set_dependencies(conn, registry, levels, regime, cooldown)

        assert routes._conn is conn
        assert routes._registry is registry
        assert routes._levels is levels
        assert routes._regime is regime
        assert routes._cooldown is cooldown


# ═══════════════════════════════════════════════════════════════════════════════
# Config — parse_symbols + cached getters (lines 115, 151-166)
# ═══════════════════════════════════════════════════════════════════════════════


class TestConfig:
    def test_parse_symbols_string_input(self) -> None:
        from src.config import AppSettings

        settings = AppSettings(symbols="AAPL, MSFT, GOOG")  # type: ignore[call-arg]
        assert settings.symbols == ["AAPL", "MSFT", "GOOG"]

    def test_parse_symbols_strips_whitespace(self) -> None:
        from src.config import AppSettings

        settings = AppSettings(symbols="  tsla , amd ")  # type: ignore[call-arg]
        assert settings.symbols == ["TSLA", "AMD"]

    def test_get_alpaca_settings_cached(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.config import get_alpaca_settings

        get_alpaca_settings.cache_clear()
        monkeypatch.setenv("ALPACA_API_KEY", "test-key")
        monkeypatch.setenv("ALPACA_SECRET_KEY", "test-secret")
        try:
            s1 = get_alpaca_settings()
            s2 = get_alpaca_settings()
            assert s1 is s2
            assert s1.api_key == "test-key"
        finally:
            get_alpaca_settings.cache_clear()

    def test_get_telegram_settings_cached(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.config import get_telegram_settings

        get_telegram_settings.cache_clear()
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok123")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "456")
        try:
            s1 = get_telegram_settings()
            s2 = get_telegram_settings()
            assert s1 is s2
        finally:
            get_telegram_settings.cache_clear()

    def test_get_risk_settings_cached(self) -> None:
        from src.config import get_risk_settings

        get_risk_settings.cache_clear()
        try:
            s1 = get_risk_settings()
            s2 = get_risk_settings()
            assert s1 is s2
        finally:
            get_risk_settings.cache_clear()

    def test_get_app_settings_cached(self) -> None:
        from src.config import get_app_settings

        get_app_settings.cache_clear()
        try:
            s1 = get_app_settings()
            s2 = get_app_settings()
            assert s1 is s2
        finally:
            get_app_settings.cache_clear()


# ═══════════════════════════════════════════════════════════════════════════════
# Indicators — calculate_bollinger (lines 90-95)
# ═══════════════════════════════════════════════════════════════════════════════


class TestCalculateBollinger:
    def test_returns_three_series(self) -> None:
        from src.indicators.batch import calculate_bollinger

        df = pd.DataFrame({"close": np.random.default_rng(42).uniform(100, 110, 30)})
        upper, middle, lower = calculate_bollinger(df)

        assert isinstance(upper, pd.Series)
        assert isinstance(middle, pd.Series)
        assert isinstance(lower, pd.Series)
        assert len(upper) == 30
        # Upper > middle > lower where not NaN
        valid = ~upper.isna()
        assert (upper[valid] >= middle[valid]).all()
        assert (middle[valid] >= lower[valid]).all()

    def test_bollinger_custom_period(self) -> None:
        from src.indicators.batch import calculate_bollinger

        df = pd.DataFrame({"close": np.random.default_rng(7).uniform(50, 60, 15)})
        upper, _middle, _lower = calculate_bollinger(df, period=5, std=1)
        assert len(upper) == 15


# ═══════════════════════════════════════════════════════════════════════════════
# Ingestion — historical fetch error handling (lines 90-92, 97, 129-131)
# ═══════════════════════════════════════════════════════════════════════════════


class TestFetchHistoricalBars:
    @patch("src.ingestion.historical.get_alpaca_settings")
    @patch("src.ingestion.historical.StockHistoricalDataClient")
    def test_alpaca_request_exception_propagates(
        self, mock_client_cls: MagicMock, mock_settings: MagicMock
    ) -> None:
        from src.ingestion.historical import fetch_historical_bars

        mock_settings.return_value = MagicMock(
            api_key="k",
            secret_key=_TEST_SECRET,
            feed="iex",
        )
        client = mock_client_cls.return_value
        client.get_stock_bars.side_effect = RuntimeError("network error")

        mock_db = MagicMock()
        with pytest.raises(RuntimeError, match="network error"):
            fetch_historical_bars("SPY", days=1, db=mock_db)

    @patch("src.ingestion.historical.get_alpaca_settings")
    @patch("src.ingestion.historical.StockHistoricalDataClient")
    def test_zero_price_bars_skipped(
        self, mock_client_cls: MagicMock, mock_settings: MagicMock
    ) -> None:
        from src.ingestion.historical import fetch_historical_bars

        mock_settings.return_value = MagicMock(
            api_key="k",
            secret_key=_TEST_SECRET,
            feed="iex",
        )
        client = mock_client_cls.return_value

        bad_bar = MagicMock()
        bad_bar.open = 0
        bad_bar.high = 100.0
        bad_bar.low = 99.0
        bad_bar.close = 99.5
        bad_bar.volume = 1000
        bad_bar.vwap = 99.7
        bad_bar.timestamp = datetime(2024, 1, 15, 14, 30, tzinfo=UTC)

        barset = MagicMock()
        barset.data = {"SPY": [bad_bar]}
        client.get_stock_bars.return_value = barset

        mock_db = MagicMock()
        result = fetch_historical_bars("SPY", days=1, db=mock_db)
        # Bad bar skipped, nothing inserted
        assert result == []
        mock_db.insert_bars.assert_not_called()

    @patch("src.ingestion.historical.get_alpaca_settings")
    @patch("src.ingestion.historical.StockHistoricalDataClient")
    def test_dict_response_format(
        self, mock_client_cls: MagicMock, mock_settings: MagicMock
    ) -> None:
        """When Alpaca returns a plain dict instead of BarSet."""
        from src.ingestion.historical import fetch_historical_bars

        mock_settings.return_value = MagicMock(
            api_key="k",
            secret_key=_TEST_SECRET,
            feed="iex",
        )
        client = mock_client_cls.return_value

        good_bar = MagicMock()
        good_bar.open = 100.0
        good_bar.high = 101.0
        good_bar.low = 99.0
        good_bar.close = 100.5
        good_bar.volume = 5000
        good_bar.vwap = 100.2
        good_bar.timestamp = datetime(2024, 1, 15, 14, 30, tzinfo=UTC)

        # Return plain dict (not BarSet)
        client.get_stock_bars.return_value = {"SPY": [good_bar]}

        mock_db = MagicMock()
        result = fetch_historical_bars("SPY", days=1, db=mock_db)
        assert len(result) == 1
        mock_db.insert_bars.assert_called_once()

    @patch("src.ingestion.historical.get_alpaca_settings")
    @patch("src.ingestion.historical.StockHistoricalDataClient")
    def test_db_insert_failure_raises(
        self, mock_client_cls: MagicMock, mock_settings: MagicMock
    ) -> None:
        from src.ingestion.historical import fetch_historical_bars

        mock_settings.return_value = MagicMock(
            api_key="k",
            secret_key=_TEST_SECRET,
            feed="iex",
        )
        client = mock_client_cls.return_value

        good_bar = MagicMock()
        good_bar.open = 100.0
        good_bar.high = 101.0
        good_bar.low = 99.0
        good_bar.close = 100.5
        good_bar.volume = 5000
        good_bar.vwap = 100.2
        good_bar.timestamp = datetime(2024, 1, 15, 14, 30, tzinfo=UTC)

        from alpaca.data.models.bars import BarSet

        barset = MagicMock(spec=BarSet)
        barset.data = {"SPY": [good_bar]}
        client.get_stock_bars.return_value = barset

        mock_db = MagicMock()
        mock_db.insert_bars.side_effect = sqlite3.Error("disk full")

        with pytest.raises(sqlite3.Error, match="disk full"):
            fetch_historical_bars("SPY", days=1, db=mock_db)


# ═══════════════════════════════════════════════════════════════════════════════
# LevelManager — PDL load exception (lines 65-66)
# ═══════════════════════════════════════════════════════════════════════════════


class TestLevelManager:
    def test_pdl_load_exception_handled(self) -> None:
        from src.levels import LevelManager

        mock_db = MagicMock()
        mgr = LevelManager(db=mock_db, symbol="SPY")
        # Force the internal PDL to raise on load
        mgr._pdl = MagicMock()
        mgr._pdl.load.side_effect = RuntimeError("db error")

        # First bar sets the date
        bar1 = _make_bar(ts=datetime(2024, 1, 15, 14, 35, tzinfo=UTC))
        mgr.update(bar1)

        # Second bar on a new date triggers PDL load → exception caught
        bar2 = _make_bar(ts=datetime(2024, 1, 16, 14, 35, tzinfo=UTC))
        mgr.update(bar2)  # should not raise

        mgr._pdl.load.assert_called()


# ═══════════════════════════════════════════════════════════════════════════════
# SessionVWAP — edge cases (lines 103, 158)
# ═══════════════════════════════════════════════════════════════════════════════


class TestSessionVWAP:
    def test_vwap_none_when_not_active(self) -> None:
        from src.levels.vwap import SessionVWAP

        vwap = SessionVWAP()
        assert vwap.vwap is None
        assert vwap.upper_1 is None
        assert vwap.lower_1 is None
        assert vwap.upper_2 is None
        assert vwap.lower_2 is None
        assert vwap.is_active is False

    def test_sigma_zero_when_no_volume(self) -> None:
        from src.levels.vwap import SessionVWAP

        vwap = SessionVWAP()
        assert vwap.sigma == Decimal("0.0")

    def test_sigma_bands_after_update(self) -> None:
        from src.levels.vwap import SessionVWAP

        vwap = SessionVWAP()
        # Bar during regular session (14:35 UTC = 9:35 ET in January/EST)
        bar = _make_bar(ts=datetime(2024, 1, 15, 14, 35, tzinfo=UTC))
        vwap.update(bar)

        assert vwap.vwap is not None
        assert vwap.sigma >= Decimal("0")
        assert vwap.upper_1 is not None
        assert vwap.lower_1 is not None
        assert vwap.upper_2 is not None
        assert vwap.lower_2 is not None
        assert vwap.is_active is True

    def test_repr(self) -> None:
        from src.levels.vwap import SessionVWAP

        vwap = SessionVWAP()
        r = repr(vwap)
        assert "SessionVWAP" in r
        assert "vwap=" in r
        assert "sigma=" in r


# ═══════════════════════════════════════════════════════════════════════════════
# BarDatabase — connect error, query with filters, count_bars (lines 79, 98, 146-153, 160)
# ═══════════════════════════════════════════════════════════════════════════════


class TestBarDatabase:
    def test_conn_property_auto_connects(self, tmp_path: Path) -> None:
        from src.storage.database import BarDatabase

        db = BarDatabase(db_path=str(tmp_path / "test.db"))
        # Accessing .conn without explicit connect() triggers auto-connect
        conn = db.conn
        assert conn is not None

    def test_insert_empty_list_returns_zero(self, tmp_path: Path) -> None:
        from src.storage.database import BarDatabase

        db = BarDatabase(db_path=str(tmp_path / "test.db"))
        db.connect()
        assert db.insert_bars([]) == 0
        db.close()

    def test_query_bars_with_time_filter(self, tmp_path: Path) -> None:
        from src.storage.database import BarDatabase

        db = BarDatabase(db_path=str(tmp_path / "test.db"))
        db.connect()

        bars = [
            _make_bar(ts=datetime(2024, 1, 15, 14, 30, tzinfo=UTC)),
            _make_bar(ts=datetime(2024, 1, 15, 14, 31, tzinfo=UTC)),
            _make_bar(ts=datetime(2024, 1, 15, 14, 32, tzinfo=UTC)),
        ]
        db.insert_bars(bars)

        # Query with start and end filters
        result = db.query_bars(
            symbol="SPY",
            timeframe=TimeFrame.ONE_MIN,
            start=datetime(2024, 1, 15, 14, 30, tzinfo=UTC),
            end=datetime(2024, 1, 15, 14, 31, tzinfo=UTC),
        )
        assert len(result) == 2

    def test_query_bars_with_limit(self, tmp_path: Path) -> None:
        from src.storage.database import BarDatabase

        db = BarDatabase(db_path=str(tmp_path / "test.db"))
        db.connect()

        bars = [_make_bar(ts=datetime(2024, 1, 15, 14, i, tzinfo=UTC)) for i in range(30, 35)]
        db.insert_bars(bars)

        result = db.query_bars(symbol="SPY", timeframe=TimeFrame.ONE_MIN, limit=2)
        assert len(result) == 2

    def test_count_bars(self, tmp_path: Path) -> None:
        from src.storage.database import BarDatabase

        db = BarDatabase(db_path=str(tmp_path / "test.db"))
        db.connect()

        bars = [_make_bar(ts=datetime(2024, 1, 15, 14, i, tzinfo=UTC)) for i in range(30, 33)]
        db.insert_bars(bars)

        count = db.count_bars("SPY", TimeFrame.ONE_MIN)
        assert count == 3

    def test_count_bars_empty(self, tmp_path: Path) -> None:
        from src.storage.database import BarDatabase

        db = BarDatabase(db_path=str(tmp_path / "test.db"))
        db.connect()
        assert db.count_bars("SPY", TimeFrame.ONE_MIN) == 0

    def test_context_manager(self, tmp_path: Path) -> None:
        from src.storage.database import BarDatabase

        with BarDatabase(db_path=str(tmp_path / "test.db")) as db:
            assert db.conn is not None


# ═══════════════════════════════════════════════════════════════════════════════
# Candlestick filters — inside bar compression (lines 118-120)
# ═══════════════════════════════════════════════════════════════════════════════


class TestCandlestickFilters:
    def test_inside_bar_compression_insufficient_data(self) -> None:
        from src.strategies.candlestick_filters import has_inside_bar_compression

        # lookback=3 needs at least 4 elements
        assert has_inside_bar_compression([100.0, 101.0], [99.0, 100.0]) is False

    def test_inside_bar_compression_exactly_at_threshold(self) -> None:
        from src.strategies.candlestick_filters import has_inside_bar_compression

        # 4 bars, lookback=3 → needs indices 1..3 checked
        highs = [105.0, 104.0, 103.0, 102.0]
        lows = [95.0, 96.0, 97.0, 98.0]
        # Each bar is an inside bar (high < prev high, low > prev low)
        assert has_inside_bar_compression(highs, lows, lookback=3) is True

    def test_inside_bar_compression_no_inside_bar(self) -> None:
        from src.strategies.candlestick_filters import has_inside_bar_compression

        highs = [100.0, 102.0, 104.0, 106.0]
        lows = [99.0, 98.0, 97.0, 96.0]
        assert has_inside_bar_compression(highs, lows, lookback=3) is False


# ═══════════════════════════════════════════════════════════════════════════════
# Backtest data_loader — WalkForwardWindow __str__, load_bars, resample
# ═══════════════════════════════════════════════════════════════════════════════


class TestBacktestDataLoader:
    def test_walk_forward_window_str(self) -> None:
        from src.backtest.data_loader import WalkForwardWindow

        w = WalkForwardWindow(
            in_sample_start=date(2024, 1, 1),
            in_sample_end=date(2024, 3, 1),
            out_of_sample_start=date(2024, 3, 2),
            out_of_sample_end=date(2024, 3, 22),
        )
        s = str(w)
        assert "IS=2024-01-01..2024-03-01" in s
        assert "OOS=2024-03-02..2024-03-22" in s

    def test_load_bars_with_data(self, tmp_path: Path) -> None:
        from src.backtest.data_loader import load_bars
        from src.storage.database import BarDatabase

        db = BarDatabase(db_path=str(tmp_path / "test.db"))
        db.connect()

        bars = [_make_bar(ts=datetime(2024, 1, 15, 14, i, tzinfo=UTC)) for i in range(30, 35)]
        db.insert_bars(bars)

        df = load_bars(db, symbol="SPY", timeframe=TimeFrame.ONE_MIN)
        assert len(df) == 5
        assert list(df.columns) == ["open", "high", "low", "close", "volume", "vwap"]
        assert df.index.tz is not None  # UTC-aware index

    def test_load_bars_empty(self, tmp_path: Path) -> None:
        from src.backtest.data_loader import load_bars
        from src.storage.database import BarDatabase

        db = BarDatabase(db_path=str(tmp_path / "test.db"))
        db.connect()

        df = load_bars(db, symbol="SPY", timeframe=TimeFrame.ONE_MIN)
        assert len(df) == 0

    def test_resample_one_min_returns_copy(self) -> None:
        from src.backtest.data_loader import resample

        idx = pd.date_range("2024-01-15 14:30", periods=5, freq="1min", tz="UTC")
        df = pd.DataFrame(
            {
                "open": [100.0] * 5,
                "high": [101.0] * 5,
                "low": [99.0] * 5,
                "close": [100.5] * 5,
                "volume": [1000] * 5,
                "vwap": [100.2] * 5,
            },
            index=idx,
        )
        result = resample(df, TimeFrame.ONE_MIN)
        assert len(result) == 5

    def test_resample_five_min(self) -> None:
        from src.backtest.data_loader import resample

        idx = pd.date_range("2024-01-15 14:30", periods=10, freq="1min", tz="UTC")
        df = pd.DataFrame(
            {
                "open": np.arange(100.0, 110.0),
                "high": np.arange(101.0, 111.0),
                "low": np.arange(99.0, 109.0),
                "close": np.arange(100.5, 110.5),
                "volume": [1000] * 10,
                "vwap": [100.2] * 10,
            },
            index=idx,
        )
        result = resample(df, TimeFrame.FIVE_MIN)
        assert len(result) == 2

    def test_resample_unsupported_raises(self) -> None:
        from src.backtest.data_loader import resample

        idx = pd.date_range("2024-01-15", periods=5, freq="1min", tz="UTC")
        df = pd.DataFrame(
            {
                "open": [100.0] * 5,
                "high": [101.0] * 5,
                "low": [99.0] * 5,
                "close": [100.5] * 5,
                "volume": [1000] * 5,
                "vwap": [100.2] * 5,
            },
            index=idx,
        )
        with pytest.raises(ValueError, match="Unsupported resample target"):
            resample(df, TimeFrame.DAILY)


# ═══════════════════════════════════════════════════════════════════════════════
# AlpacaExecutor — flatten failure, cancel failure, position retrieval
# (lines 157-159, 200-202, 206, 220, 241-243, 262-264)
# ═══════════════════════════════════════════════════════════════════════════════


class TestAlpacaExecutor:
    def _make_executor(self) -> MagicMock:
        """Create an AlpacaExecutor with a mocked TradingClient."""
        with patch("src.execution.alpaca_executor.TradingClient"):
            from src.execution.alpaca_executor import AlpacaExecutor

            executor = AlpacaExecutor(
                api_key="test",
                secret_key=_TEST_SECRET,
                paper=True,
            )
        return executor

    def test_flatten_get_positions_api_error(self) -> None:
        from alpaca.common.exceptions import APIError

        executor = self._make_executor()
        executor._client.get_all_positions.side_effect = APIError({"message": "fail"})

        result = executor.flatten_all_positions()
        assert result == {}

    def test_flatten_no_positions(self) -> None:
        executor = self._make_executor()
        executor._client.get_all_positions.return_value = []

        result = executor.flatten_all_positions()
        assert result == {}

    def test_flatten_close_failure(self) -> None:
        from alpaca.common.exceptions import APIError

        executor = self._make_executor()
        pos = MagicMock()
        pos.symbol = "SPY"
        pos.qty = "100"
        pos.side = "long"
        executor._client.get_all_positions.return_value = [pos]
        executor._client.close_position.side_effect = APIError({"message": "fail"})

        result = executor.flatten_all_positions()
        # SPY should show as failed after all retry attempts
        assert result.get("SPY") != "closed"

    def test_cancel_open_orders_api_error(self) -> None:
        from alpaca.common.exceptions import APIError

        executor = self._make_executor()
        executor._client.get_orders.side_effect = APIError({"message": "fail"})

        result = executor.cancel_open_orders()
        assert result == 0

    def test_cancel_open_orders_success(self) -> None:
        executor = self._make_executor()
        executor._client.get_orders.return_value = [MagicMock(), MagicMock()]
        executor._client.cancel_orders.return_value = None

        result = executor.cancel_open_orders()
        assert result == 2

    def test_get_account_delegates(self) -> None:
        executor = self._make_executor()
        executor._client.get_account.return_value = MagicMock()

        account = executor.get_account()
        executor._client.get_account.assert_called_once()
        assert account is not None

    def test_get_positions_delegates(self) -> None:
        executor = self._make_executor()
        executor._client.get_all_positions.return_value = []

        positions = executor.get_positions()
        assert positions == []

    def test_now_et_time(self) -> None:
        executor = self._make_executor()
        t = executor._now_et_time()
        assert t is not None

    def test_cap_buying_power_api_error_returns_zero(self) -> None:
        from alpaca.common.exceptions import APIError

        executor = self._make_executor()
        executor._client.get_account.side_effect = APIError({"message": "fail"})

        result = executor._cap_to_buying_power("SPY", 10, Decimal("100.00"))
        assert result == 0

    def test_assert_paper_account_api_error(self) -> None:
        from alpaca.common.exceptions import APIError

        executor = self._make_executor()
        executor._client.get_account.side_effect = APIError({"message": "fail"})

        result = executor._assert_paper_account()
        assert result is False

    def test_assert_paper_account_not_paper(self) -> None:
        executor = self._make_executor()
        account = MagicMock()
        account.account_number = "LA12345"
        executor._client.get_account.return_value = account

        result = executor._assert_paper_account()
        assert result is False

    def test_assert_paper_account_is_paper(self) -> None:
        executor = self._make_executor()
        account = MagicMock()
        account.account_number = "PA12345"
        executor._client.get_account.return_value = account

        result = executor._assert_paper_account()
        assert result is True
