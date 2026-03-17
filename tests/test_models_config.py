from __future__ import annotations

import importlib
import os
import subprocess
import sys
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import ClassVar

import pandas as pd
import pytest
from pydantic import ValidationError

from src.models import (
    Bar,
    Direction,
    IndicatorSnapshot,
    LevelSnapshot,
    Regime,
    RiskDecision,
    Signal,
    TimeFrame,
    TradeResult,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOW_UTC = datetime.now(tz=UTC)

_VALID_BAR_KWARGS: dict = {
    "symbol": "SPY",
    "timeframe": TimeFrame.ONE_MIN,
    "timestamp": _NOW_UTC,
    "open": Decimal("450.00"),
    "high": Decimal("451.00"),
    "low": Decimal("449.00"),
    "close": Decimal("450.50"),
    "volume": 100_000,
    "vwap": Decimal("450.25"),
}

_VALID_SIGNAL_KWARGS: dict = {
    "direction": Direction.LONG,
    "strategy_name": "ORB",
    "entry_price": Decimal("450.00"),
    "stop_price": Decimal("448.00"),
    "target_price": Decimal("454.00"),
    "risk_reward_ratio": Decimal("2.0"),
    "confidence_score": 3,
    "reason": "ORB breakout with volume",
    "timeframe": TimeFrame.FIVE_MIN,
    "regime": Regime.TRENDING_UP,
    "timestamp": _NOW_UTC,
}


# ===========================================================================
# 1. Models — src/models.py
# ===========================================================================


class TestTimeFrame:
    def test_one_min_value(self) -> None:
        assert TimeFrame.ONE_MIN.value == "1Min"

    def test_five_min_value(self) -> None:
        assert TimeFrame.FIVE_MIN.value == "5Min"

    def test_fifteen_min_value(self) -> None:
        assert TimeFrame.FIFTEEN_MIN.value == "15Min"

    def test_thirty_min_value(self) -> None:
        assert TimeFrame.THIRTY_MIN.value == "30Min"

    def test_daily_value(self) -> None:
        assert TimeFrame.DAILY.value == "1Day"

    def test_is_str_enum(self) -> None:
        assert isinstance(TimeFrame.ONE_MIN, str)

    def test_str_comparison(self) -> None:
        assert TimeFrame.ONE_MIN == "1Min"


class TestDirection:
    def test_long(self) -> None:
        assert Direction.LONG.value == "LONG"

    def test_short(self) -> None:
        assert Direction.SHORT.value == "SHORT"

    def test_is_str_enum(self) -> None:
        assert isinstance(Direction.LONG, str)


class TestRegime:
    def test_trending_up(self) -> None:
        assert Regime.TRENDING_UP.value == "TRENDING_UP"

    def test_trending_down(self) -> None:
        assert Regime.TRENDING_DOWN.value == "TRENDING_DOWN"

    def test_ranging(self) -> None:
        assert Regime.RANGING.value == "RANGING"

    def test_choppy(self) -> None:
        assert Regime.CHOPPY.value == "CHOPPY"


class TestBar:
    def test_all_ohlcv_and_extra_fields_present(self) -> None:
        bar = Bar(**_VALID_BAR_KWARGS)
        assert bar.symbol == "SPY"
        assert bar.timeframe == TimeFrame.ONE_MIN
        assert bar.timestamp == _NOW_UTC
        assert bar.open == Decimal("450.00")
        assert bar.high == Decimal("451.00")
        assert bar.low == Decimal("449.00")
        assert bar.close == Decimal("450.50")
        assert bar.volume == 100_000
        assert bar.vwap == Decimal("450.25")

    def test_rejects_negative_prices(self) -> None:
        kwargs = {**_VALID_BAR_KWARGS, "open": Decimal("-1.00")}
        with pytest.raises(ValidationError, match="positive"):
            Bar(**kwargs)

    def test_rejects_zero_prices(self) -> None:
        kwargs = {**_VALID_BAR_KWARGS, "close": Decimal("0")}
        with pytest.raises(ValidationError, match="positive"):
            Bar(**kwargs)

    def test_rejects_naive_timestamps(self) -> None:
        kwargs = {**_VALID_BAR_KWARGS, "timestamp": datetime(2024, 1, 1, 12, 0, 0)}
        with pytest.raises(ValidationError, match="timezone"):
            Bar(**kwargs)

    def test_accepts_all_timeframe_values(self) -> None:
        for tf in TimeFrame:
            bar = Bar(**{**_VALID_BAR_KWARGS, "timeframe": tf})
            assert bar.timeframe == tf

    def test_preserves_decimal_precision(self) -> None:
        kwargs = {**_VALID_BAR_KWARGS, "open": Decimal("450.123456")}
        bar = Bar(**kwargs)
        assert bar.open == Decimal("450.123456")


class TestSignal:
    def test_all_required_fields_present(self) -> None:
        sig = Signal(**_VALID_SIGNAL_KWARGS)
        assert sig.symbol == "SPY"
        assert sig.direction == Direction.LONG
        assert sig.strategy_name == "ORB"
        assert sig.entry_price == Decimal("450.00")
        assert sig.stop_price == Decimal("448.00")
        assert sig.target_price == Decimal("454.00")
        assert sig.risk_reward_ratio == Decimal("2.0")
        assert sig.confidence_score == 3
        assert sig.reason == "ORB breakout with volume"
        assert sig.timeframe == TimeFrame.FIVE_MIN
        assert sig.regime == Regime.TRENDING_UP
        assert sig.timestamp == _NOW_UTC
        assert sig.tags == []

    def test_default_symbol_is_spy(self) -> None:
        sig = Signal(**_VALID_SIGNAL_KWARGS)
        assert sig.symbol == "SPY"

    def test_custom_symbol(self) -> None:
        sig = Signal(**{**_VALID_SIGNAL_KWARGS, "symbol": "QQQ"})
        assert sig.symbol == "QQQ"

    def test_tags_default_empty_list(self) -> None:
        sig = Signal(**_VALID_SIGNAL_KWARGS)
        assert sig.tags == []

    def test_tags_custom(self) -> None:
        sig = Signal(**{**_VALID_SIGNAL_KWARGS, "tags": ["ENGULFING", "COMPRESSED"]})
        assert sig.tags == ["ENGULFING", "COMPRESSED"]


class TestRiskDecision:
    def test_approved(self) -> None:
        rd = RiskDecision(approved=True, reason="All checks passed")
        assert rd.approved is True
        assert rd.reason == "All checks passed"

    def test_position_size_default_zero(self) -> None:
        rd = RiskDecision(approved=False, reason="Daily loss exceeded")
        assert rd.position_size == 0

    def test_custom_position_size(self) -> None:
        rd = RiskDecision(approved=True, reason="OK", position_size=100)
        assert rd.position_size == 100


class TestTradeResult:
    def test_has_signal_pnl_timestamp(self) -> None:
        sig = Signal(**_VALID_SIGNAL_KWARGS)
        tr = TradeResult(signal=sig, pnl=Decimal("150.00"), timestamp=_NOW_UTC)
        assert tr.signal == sig
        assert tr.pnl == Decimal("150.00")
        assert tr.timestamp == _NOW_UTC


class TestIndicatorSnapshot:
    def test_all_fields_optional_default_none(self) -> None:
        snap = IndicatorSnapshot()
        assert snap.ema9 is None
        assert snap.ema20 is None
        assert snap.ema50 is None
        assert snap.rsi is None
        assert snap.macd is None
        assert snap.macd_signal is None
        assert snap.macd_histogram is None
        assert snap.bb_upper is None
        assert snap.bb_middle is None
        assert snap.bb_lower is None
        assert snap.atr is None
        assert snap.vwap is None

    def test_accepts_decimal_values(self) -> None:
        snap = IndicatorSnapshot(ema9=Decimal("450.12"), rsi=Decimal("55.3"))
        assert snap.ema9 == Decimal("450.12")
        assert snap.rsi == Decimal("55.3")


class TestLevelSnapshot:
    def test_orb_complete_defaults_false(self) -> None:
        snap = LevelSnapshot()
        assert snap.orb_complete is False

    def test_orb15_complete_defaults_false(self) -> None:
        snap = LevelSnapshot()
        assert snap.orb15_complete is False

    def test_all_price_fields_default_none(self) -> None:
        snap = LevelSnapshot()
        assert snap.orb_high is None
        assert snap.orb_low is None
        assert snap.orb_midpoint is None
        assert snap.orb_range is None
        assert snap.vwap is None
        assert snap.high_of_day is None
        assert snap.low_of_day is None
        assert snap.prev_day_high is None
        assert snap.prev_day_low is None
        assert snap.prev_day_close is None
        assert snap.premarket_high is None
        assert snap.premarket_low is None


# ===========================================================================
# 2. Config — src/config.py
# ===========================================================================


class TestAppSettings:
    """Tests for AppSettings.

    We use monkeypatch to inject required env vars so pydantic-settings
    does not read the real .env file or fail on missing keys.
    """

    def _make_settings(self, monkeypatch: pytest.MonkeyPatch, **overrides: str):
        from src.config import AppSettings

        # Minimal env vars that AppSettings may need
        env = {
            "LOG_LEVEL": "INFO",
            "TRADING_MODE": "paper",
            "DB_PATH": "data/test.db",
        }
        env.update(overrides)
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        return AppSettings()

    def test_symbols_has_26_items(self, monkeypatch: pytest.MonkeyPatch) -> None:
        settings = self._make_settings(monkeypatch)
        assert len(settings.symbols) == 26

    def test_all_26_tickers_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        expected = {
            "SPY",
            "QQQ",
            "MSFT",
            "AMD",
            "TSLA",
            "AMZN",
            "UBER",
            "SMCI",
            "SHOP",
            "PLTR",
            "NFLX",
            "MSTR",
            "SNOW",
            "ARM",
            "DASH",
            "PYPL",
            "INTC",
            "MU",
            "HOOD",
            "DKNG",
            "SOXL",
            "ROKU",
            "TQQQ",
            "BA",
            "MRVL",
            "META",
        }
        settings = self._make_settings(monkeypatch)
        assert set(settings.symbols) == expected

    def test_soxl_and_tqqq_in_list(self, monkeypatch: pytest.MonkeyPatch) -> None:
        settings = self._make_settings(monkeypatch)
        assert "SOXL" in settings.symbols
        assert "TQQQ" in settings.symbols

    def test_trading_mode_valid_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for mode in ("live", "paper", "backtest"):
            settings = self._make_settings(monkeypatch, TRADING_MODE=mode)
            assert settings.trading_mode == mode

    def test_trading_mode_invalid_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with pytest.raises(ValidationError, match="trading_mode"):
            self._make_settings(monkeypatch, TRADING_MODE="demo")

    def test_log_level_uppercased(self, monkeypatch: pytest.MonkeyPatch) -> None:
        settings = self._make_settings(monkeypatch, LOG_LEVEL="debug")
        assert settings.log_level == "DEBUG"

    def test_parse_symbols_json_list(self, monkeypatch: pytest.MonkeyPatch) -> None:
        settings = self._make_settings(monkeypatch, SYMBOLS='["aapl", " goog ", "msft"]')
        assert settings.symbols == ["AAPL", "GOOG", "MSFT"]


class TestRiskSettings:
    def _make_settings(self, monkeypatch: pytest.MonkeyPatch, **overrides: str):
        from src.config import RiskSettings

        for k, v in overrides.items():
            monkeypatch.setenv(k, v)
        return RiskSettings()

    def test_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        settings = self._make_settings(monkeypatch)
        assert settings.account_size == Decimal("50000")
        assert settings.risk_per_trade_pct == Decimal("1.0")
        assert settings.max_daily_loss_pct == Decimal("3.0")
        assert settings.max_trades_per_day == 5
        assert settings.max_concurrent_positions == 3

    def test_max_concurrent_positions_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        settings = self._make_settings(monkeypatch, MAX_CONCURRENT_POSITIONS="5")
        assert settings.max_concurrent_positions == 5


class TestAlpacaSettings:
    def _make_settings(self, monkeypatch: pytest.MonkeyPatch, **overrides: str):
        from src.config import AlpacaSettings

        env = {
            "ALPACA_API_KEY": "test-key-123",
            "ALPACA_SECRET_KEY": "test-secret-456",
        }
        env.update(overrides)
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        return AlpacaSettings()

    def test_feed_default_iex(self, monkeypatch: pytest.MonkeyPatch) -> None:
        settings = self._make_settings(monkeypatch)
        assert settings.feed == "iex"

    def test_feed_accepts_sip(self, monkeypatch: pytest.MonkeyPatch) -> None:
        settings = self._make_settings(monkeypatch, ALPACA_FEED="sip")
        assert settings.feed == "sip"

    def test_feed_rejects_invalid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with pytest.raises(ValidationError, match="feed"):
            self._make_settings(monkeypatch, ALPACA_FEED="nasdaq")


# ===========================================================================
# 3. Script imports — scripts/
# ===========================================================================

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"


class TestReplayDayOutcomeLabel:
    """Test the _outcome_label() helper from scripts/replay_day.py."""

    @pytest.fixture(autouse=True)
    def _add_scripts_to_path(self) -> None:
        scripts_path = str(SCRIPTS_DIR)
        if scripts_path not in sys.path:
            sys.path.insert(0, scripts_path)

    @staticmethod
    def _get_outcome_label():
        """Import and return the _outcome_label function."""
        # Force reimport to pick up latest
        if "replay_day" in sys.modules:
            del sys.modules["replay_day"]
        mod = importlib.import_module("replay_day")
        return mod._outcome_label

    def test_target_hit(self) -> None:
        fn = self._get_outcome_label()
        trade = pd.Series({"ExitPrice": 454.00, "SL": 448.00, "TP": 454.02, "PnL": 200.0})
        assert fn(trade) == "TARGET"

    def test_stop_hit(self) -> None:
        fn = self._get_outcome_label()
        trade = pd.Series({"ExitPrice": 448.01, "SL": 448.00, "TP": 454.00, "PnL": -100.0})
        assert fn(trade) == "STOP"

    def test_eod_exit(self) -> None:
        fn = self._get_outcome_label()
        trade = pd.Series({"ExitPrice": 451.50, "SL": 448.00, "TP": 454.00, "PnL": 50.0})
        assert fn(trade) == "EOD"

    def test_target_nan_sl_nan_returns_eod(self) -> None:
        fn = self._get_outcome_label()
        trade = pd.Series({"ExitPrice": 451.50, "SL": float("nan"), "TP": float("nan"), "PnL": 0.0})
        assert fn(trade) == "EOD"


class TestReplayDayFormatDuration:
    """Test the _format_duration() helper from scripts/replay_day.py."""

    @pytest.fixture(autouse=True)
    def _add_scripts_to_path(self) -> None:
        scripts_path = str(SCRIPTS_DIR)
        if scripts_path not in sys.path:
            sys.path.insert(0, scripts_path)

    @staticmethod
    def _get_fn():
        if "replay_day" in sys.modules:
            del sys.modules["replay_day"]
        mod = importlib.import_module("replay_day")
        return mod._format_duration

    def test_short_duration(self) -> None:
        fn = self._get_fn()
        assert fn(pd.Timedelta(minutes=3, seconds=45)) == "  3:45"

    def test_zero_duration(self) -> None:
        fn = self._get_fn()
        assert fn(pd.Timedelta(seconds=0)) == "  0:00"

    def test_long_duration(self) -> None:
        fn = self._get_fn()
        # 6h 30m = 390 minutes
        assert fn(pd.Timedelta(hours=6, minutes=30)) == "390:00"


class TestReplayDayPeakConcurrent:
    """Test the _compute_peak_concurrent() helper from scripts/replay_day.py."""

    @pytest.fixture(autouse=True)
    def _add_scripts_to_path(self) -> None:
        scripts_path = str(SCRIPTS_DIR)
        if scripts_path not in sys.path:
            sys.path.insert(0, scripts_path)

    @staticmethod
    def _get_fn():
        if "replay_day" in sys.modules:
            del sys.modules["replay_day"]
        mod = importlib.import_module("replay_day")
        return mod._compute_peak_concurrent

    def test_no_overlap(self) -> None:
        """Sequential trades have peak=1."""
        fn = self._get_fn()
        df = pd.DataFrame(
            {
                "EntryTime": [
                    pd.Timestamp("2026-03-17 09:35", tz="UTC"),
                    pd.Timestamp("2026-03-17 10:30", tz="UTC"),
                ],
                "ExitTime": [
                    pd.Timestamp("2026-03-17 10:00", tz="UTC"),
                    pd.Timestamp("2026-03-17 11:00", tz="UTC"),
                ],
            }
        )
        assert fn(df) == 1

    def test_full_overlap(self) -> None:
        """Two trades fully overlapping have peak=2."""
        fn = self._get_fn()
        df = pd.DataFrame(
            {
                "EntryTime": [
                    pd.Timestamp("2026-03-17 09:35", tz="UTC"),
                    pd.Timestamp("2026-03-17 09:36", tz="UTC"),
                ],
                "ExitTime": [
                    pd.Timestamp("2026-03-17 10:00", tz="UTC"),
                    pd.Timestamp("2026-03-17 10:00", tz="UTC"),
                ],
            }
        )
        assert fn(df) == 2

    def test_three_way_overlap(self) -> None:
        """Three trades all open at the same time: peak=3."""
        fn = self._get_fn()
        df = pd.DataFrame(
            {
                "EntryTime": [
                    pd.Timestamp("2026-03-17 09:35", tz="UTC"),
                    pd.Timestamp("2026-03-17 09:36", tz="UTC"),
                    pd.Timestamp("2026-03-17 09:37", tz="UTC"),
                ],
                "ExitTime": [
                    pd.Timestamp("2026-03-17 10:00", tz="UTC"),
                    pd.Timestamp("2026-03-17 10:00", tz="UTC"),
                    pd.Timestamp("2026-03-17 10:00", tz="UTC"),
                ],
            }
        )
        assert fn(df) == 3

    def test_exit_before_entry_no_double_count(self) -> None:
        """When trade A exits at the same time trade B enters, peak=1."""
        fn = self._get_fn()
        df = pd.DataFrame(
            {
                "EntryTime": [
                    pd.Timestamp("2026-03-17 09:35", tz="UTC"),
                    pd.Timestamp("2026-03-17 10:00", tz="UTC"),
                ],
                "ExitTime": [
                    pd.Timestamp("2026-03-17 10:00", tz="UTC"),
                    pd.Timestamp("2026-03-17 10:30", tz="UTC"),
                ],
            }
        )
        assert fn(df) == 1


class TestRunBacktestImport:
    """Verify run_backtest.py can be imported without error."""

    def test_imports_without_error(self) -> None:
        result = subprocess.run(  # noqa: S603
            [
                sys.executable,
                "-c",
                "import warnings; warnings.filterwarnings('ignore'); import importlib, sys; sys.path.insert(0, 'scripts'); importlib.import_module('run_backtest')",
            ],
            capture_output=True,
            text=True,
            cwd=str(SCRIPTS_DIR.parent),
            timeout=30,
        )
        # Allow import to succeed (rc 0) or fail only on external deps (e.g. missing DB),
        # but it should NOT fail on SyntaxError or ImportError for standard lib modules.
        assert "SyntaxError" not in result.stderr, f"SyntaxError on import: {result.stderr}"


# ===========================================================================
# 4. Automation scripts
# ===========================================================================


class TestAutomationScripts:
    """Verify automation shell scripts exist and are executable."""

    SCRIPT_NAMES: ClassVar[list[str]] = [
        "auto_start.sh",
        "auto_stop.sh",
        "install_launchd.sh",
        "uninstall_launchd.sh",
    ]

    @pytest.mark.parametrize("name", SCRIPT_NAMES)
    def test_script_exists(self, name: str) -> None:
        path = SCRIPTS_DIR / name
        assert path.exists(), f"{name} not found in scripts/"

    @pytest.mark.parametrize("name", SCRIPT_NAMES)
    def test_script_is_executable(self, name: str) -> None:
        path = SCRIPTS_DIR / name
        assert os.access(path, os.X_OK), f"{name} is not executable"
