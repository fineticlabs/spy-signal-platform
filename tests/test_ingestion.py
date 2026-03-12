"""Tests for Bar model, config loading, and database CRUD."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

from src.models import Bar, TimeFrame

if TYPE_CHECKING:
    from pathlib import Path

# ── helpers ─────────────────────────────────────────────────────────────────


def _make_bar(**overrides: object) -> Bar:
    defaults: dict[str, object] = {
        "symbol": "SPY",
        "timeframe": TimeFrame.ONE_MIN,
        "timestamp": datetime(2024, 1, 15, 14, 30, tzinfo=UTC),
        "open": Decimal("480.00"),
        "high": Decimal("481.50"),
        "low": Decimal("479.80"),
        "close": Decimal("481.00"),
        "volume": 1_200_000,
        "vwap": Decimal("480.55"),
    }
    defaults.update(overrides)
    return Bar(**defaults)  # type: ignore[arg-type]


# ── Bar model tests ──────────────────────────────────────────────────────────


class TestBarModel:
    def test_valid_bar(self) -> None:
        bar = _make_bar()
        assert bar.symbol == "SPY"
        assert bar.close == Decimal("481.00")
        assert bar.volume == 1_200_000

    def test_all_timeframes_accepted(self) -> None:
        for tf in TimeFrame:
            bar = _make_bar(timeframe=tf)
            assert bar.timeframe == tf

    def test_requires_timezone_aware_timestamp(self) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            _make_bar(timestamp=datetime(2024, 1, 15, 14, 30))  # naive

    def test_rejects_negative_price(self) -> None:
        with pytest.raises(ValueError):
            _make_bar(close=Decimal("-1.00"))

    def test_rejects_zero_price(self) -> None:
        with pytest.raises(ValueError):
            _make_bar(open=Decimal("0"))

    def test_rejects_negative_volume(self) -> None:
        with pytest.raises(ValueError):
            _make_bar(volume=-1)

    def test_bar_is_immutable(self) -> None:
        bar = _make_bar()
        with pytest.raises(ValidationError):
            bar.close = Decimal("999")

    def test_decimal_precision_preserved(self) -> None:
        bar = _make_bar(close=Decimal("481.123456789"))
        assert bar.close == Decimal("481.123456789")


# ── Config tests ─────────────────────────────────────────────────────────────


class TestConfig:
    def test_alpaca_settings_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ALPACA_API_KEY", "test_key")
        monkeypatch.setenv("ALPACA_SECRET_KEY", "test_secret")
        monkeypatch.setenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
        monkeypatch.setenv("ALPACA_FEED", "iex")

        # Import here so the monkeypatch takes effect before Settings is constructed.
        from src.config import AlpacaSettings

        settings = AlpacaSettings()  # type: ignore[call-arg]
        assert settings.api_key == "test_key"
        assert settings.secret_key == "test_secret"  # noqa: S105
        assert settings.feed == "iex"

    def test_risk_settings_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Clear any existing env vars that might interfere
        for key in (
            "ACCOUNT_SIZE",
            "RISK_PER_TRADE_PCT",
            "MAX_DAILY_LOSS_PCT",
            "MAX_TRADES_PER_DAY",
        ):
            monkeypatch.delenv(key, raising=False)

        from src.config import RiskSettings

        settings = RiskSettings()
        assert settings.account_size == Decimal("50000")
        assert settings.risk_per_trade_pct == Decimal("1.0")
        assert settings.max_daily_loss_pct == Decimal("3.0")
        assert settings.max_trades_per_day == 5

    def test_app_settings_trading_mode_validation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TRADING_MODE", "invalid_mode")

        from src.config import AppSettings

        with pytest.raises(ValueError):
            AppSettings()

    def test_app_settings_log_level_uppercased(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LOG_LEVEL", "debug")

        from src.config import AppSettings

        settings = AppSettings()
        assert settings.log_level == "DEBUG"


# ── Database tests ───────────────────────────────────────────────────────────


class TestBarDatabase:
    def test_create_and_query_bars(self, tmp_path: Path) -> None:
        from src.storage.database import BarDatabase

        db_file = str(tmp_path / "test.db")
        bar = _make_bar()

        with BarDatabase(db_path=db_file) as db:
            inserted = db.insert_bars([bar])
            assert inserted == 1

            results = db.query_bars(symbol="SPY", timeframe=TimeFrame.ONE_MIN)

        assert len(results) == 1
        result = results[0]
        assert result.symbol == "SPY"
        assert result.close == Decimal("481.00")
        assert result.timestamp == datetime(2024, 1, 15, 14, 30, tzinfo=UTC)

    def test_insert_duplicate_is_ignored(self, tmp_path: Path) -> None:
        from src.storage.database import BarDatabase

        db_file = str(tmp_path / "test.db")
        bar = _make_bar()

        with BarDatabase(db_path=db_file) as db:
            db.insert_bars([bar])
            inserted_second = db.insert_bars([bar])
            assert inserted_second == 0
            assert db.count_bars("SPY", TimeFrame.ONE_MIN) == 1

    def test_count_bars(self, tmp_path: Path) -> None:
        from src.storage.database import BarDatabase

        db_file = str(tmp_path / "test.db")
        bars = [
            _make_bar(
                timestamp=datetime(2024, 1, 15, 14, 30 + i, tzinfo=UTC),
            )
            for i in range(5)
        ]

        with BarDatabase(db_path=db_file) as db:
            db.insert_bars(bars)
            assert db.count_bars("SPY", TimeFrame.ONE_MIN) == 5

    def test_query_with_time_filter(self, tmp_path: Path) -> None:
        from src.storage.database import BarDatabase

        db_file = str(tmp_path / "test.db")
        bars = [
            _make_bar(
                timestamp=datetime(2024, 1, 15, 14, 30 + i, tzinfo=UTC),
            )
            for i in range(10)
        ]

        start_filter = datetime(2024, 1, 15, 14, 33, tzinfo=UTC)

        with BarDatabase(db_path=db_file) as db:
            db.insert_bars(bars)
            results = db.query_bars("SPY", TimeFrame.ONE_MIN, start=start_filter)

        assert len(results) == 7  # bars at :33 through :39

    def test_wal_mode_enabled(self, tmp_path: Path) -> None:
        from src.storage.database import BarDatabase

        db_file = str(tmp_path / "test.db")
        with BarDatabase(db_path=db_file) as db:
            row = db.conn.execute("PRAGMA journal_mode;").fetchone()
            assert row[0] == "wal"
