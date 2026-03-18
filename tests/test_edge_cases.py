"""Edge case and failure mode tests for the spy-signal-platform.

Covers validation boundaries, indicator convergence, risk edge cases,
level tracker quirks, storage idempotency, and cooldown state machines.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

from src.config import AlpacaSettings, AppSettings, RiskSettings
from src.indicators.streaming import StreamingATR, StreamingEMA, StreamingRSI
from src.levels.opening_range import OpeningRangeTracker
from src.levels.vwap import SessionVWAP
from src.models import Bar, Direction, Regime, Signal, TimeFrame, TradeResult
from src.risk.cooldown import CooldownTracker
from src.risk.position_sizing import calculate_position_size
from tests.conftest import make_bar, make_utc_ts

if TYPE_CHECKING:
    from src.storage.database import BarDatabase


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_signal(**overrides) -> Signal:
    """Build a valid Signal, merging caller overrides."""
    defaults = dict(
        symbol="SPY",
        direction=Direction.LONG,
        strategy_name="ORB",
        entry_price=Decimal("481.50"),
        stop_price=Decimal("479.00"),
        target_price=Decimal("486.50"),
        risk_reward_ratio=Decimal("2.0"),
        confidence_score=3,
        reason="test signal",
        timeframe=TimeFrame.ONE_MIN,
        regime=Regime.TRENDING_UP,
        timestamp=make_utc_ts(2024, 1, 15, 14, 36),
    )
    defaults.update(overrides)
    return Signal(**defaults)


def _make_trade_result(pnl: Decimal, ts: datetime | None = None) -> TradeResult:
    """Build a TradeResult with the given P&L."""
    return TradeResult(
        signal=_make_signal(),
        pnl=pnl,
        timestamp=ts or make_utc_ts(2024, 1, 15, 15, 0),
    )


# ===========================================================================
# Data Edge Cases — Bar
# ===========================================================================


class TestBarEdgeCases:
    """Validation edge cases for src.models.Bar."""

    def test_zero_volume_allowed(self) -> None:
        """ge=0 means volume=0 is valid."""
        bar = make_bar(volume=0)
        assert bar.volume == 0

    def test_high_less_than_low_accepted_due_to_field_order(self) -> None:
        """high < low does NOT raise because `low` is declared after `high`.

        The ``high_gte_low`` validator runs before ``low`` is available in
        ``info.data``, so the cross-field check is a no-op.  This test
        documents the current (permissive) behaviour.
        """
        bar = Bar(
            symbol="SPY",
            timestamp=make_utc_ts(2024, 1, 15, 14, 35),
            open=Decimal("480.00"),
            high=Decimal("470.00"),
            low=Decimal("480.00"),
            close=Decimal("475.00"),
            volume=100,
            vwap=Decimal("475.00"),
        )
        # Validator silently passes — high < low is stored as-is
        assert bar.high < bar.low

    def test_close_outside_high_low_accepted(self) -> None:
        """Bar model does not enforce close in [low, high] — verify it passes."""
        bar = make_bar(
            open_=Decimal("480.00"),
            high=Decimal("482.00"),
            low=Decimal("479.00"),
            close=Decimal("483.00"),  # above high
            vwap=Decimal("480.50"),
        )
        assert bar.close == Decimal("483.00")

    def test_zero_price_raises(self) -> None:
        """Price fields must be strictly positive."""
        with pytest.raises(ValidationError, match="positive"):
            Bar(
                symbol="SPY",
                timestamp=make_utc_ts(2024, 1, 15, 14, 35),
                open=Decimal("0"),
                high=Decimal("481.00"),
                low=Decimal("479.00"),
                close=Decimal("480.00"),
                volume=100,
                vwap=Decimal("480.00"),
            )

    def test_negative_price_raises(self) -> None:
        with pytest.raises(ValidationError, match="positive"):
            Bar(
                symbol="SPY",
                timestamp=make_utc_ts(2024, 1, 15, 14, 35),
                open=Decimal("480.00"),
                high=Decimal("481.00"),
                low=Decimal("479.00"),
                close=Decimal("-1.00"),
                volume=100,
                vwap=Decimal("480.00"),
            )

    def test_tz_naive_timestamp_raises(self) -> None:
        """Timezone-naive datetime must be rejected."""
        naive_ts = datetime(2024, 1, 15, 14, 35)  # intentionally naive
        with pytest.raises(ValidationError, match="timezone-aware"):
            make_bar(timestamp=naive_ts)

    def test_bar_is_frozen(self) -> None:
        """Bar instances should be immutable."""
        bar = make_bar()
        with pytest.raises(ValidationError):
            bar.volume = 999  # type: ignore[misc]


# ===========================================================================
# Data Edge Cases — Signal
# ===========================================================================


class TestSignalEdgeCases:
    """Validation edge cases for src.models.Signal."""

    def test_confidence_below_min_raises(self) -> None:
        with pytest.raises(ValidationError, match="greater than or equal to 1"):
            _make_signal(confidence_score=0)

    def test_confidence_above_max_raises(self) -> None:
        with pytest.raises(ValidationError, match="less than or equal to 5"):
            _make_signal(confidence_score=6)

    def test_signal_is_frozen(self) -> None:
        sig = _make_signal()
        with pytest.raises(ValidationError):
            sig.reason = "mutated"  # type: ignore[misc]


# ===========================================================================
# Config Edge Cases
# ===========================================================================


class TestRiskSettingsEdgeCases:
    """RiskSettings validation via monkeypatch."""

    def test_risk_pct_zero_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RISK_PER_TRADE_PCT", "0")
        with pytest.raises(ValidationError, match="percentage"):
            RiskSettings()

    def test_risk_pct_over_100_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RISK_PER_TRADE_PCT", "101")
        with pytest.raises(ValidationError, match="percentage"):
            RiskSettings()

    def test_max_trades_per_day_zero_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MAX_TRADES_PER_DAY", "0")
        with pytest.raises(ValidationError, match="must be >= 1"):
            RiskSettings()


class TestAppSettingsEdgeCases:
    """AppSettings validation via monkeypatch."""

    def test_invalid_trading_mode_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TRADING_MODE", "yolo")
        with pytest.raises(ValidationError, match="trading_mode"):
            AppSettings()

    def test_invalid_execution_mode_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EXECUTION_MODE", "nuke")
        with pytest.raises(ValidationError, match="execution_mode"):
            AppSettings()

    def test_invalid_log_level_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LOG_LEVEL", "VERBOSE")
        with pytest.raises(ValidationError, match="log_level"):
            AppSettings()


class TestAlpacaSettingsEdgeCases:
    """AlpacaSettings validation via monkeypatch."""

    def test_invalid_feed_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ALPACA_API_KEY", "test")
        monkeypatch.setenv("ALPACA_SECRET_KEY", "test")
        monkeypatch.setenv("ALPACA_FEED", "nasdaq")
        with pytest.raises(ValidationError, match="feed must be"):
            AlpacaSettings()


# ===========================================================================
# Indicator Edge Cases
# ===========================================================================


class TestStreamingEMAEdgeCases:
    """StreamingEMA boundary behaviour."""

    def test_not_ready_before_period(self) -> None:
        ema = StreamingEMA(period=5)
        for i in range(4):
            bar = make_bar(
                close=Decimal("480.00"),
                timestamp=make_utc_ts(2024, 1, 15, 14, 30 + i),
            )
            ema.update(bar)
        assert not ema.ready

    def test_all_same_price_converges(self) -> None:
        """EMA fed constant-price bars should converge to that price."""
        ema = StreamingEMA(period=5)
        price = Decimal("500.00")
        for i in range(20):
            bar = make_bar(
                open_=price,
                high=price + Decimal("0.01"),
                low=price - Decimal("0.01"),
                close=price,
                vwap=price,
                timestamp=make_utc_ts(2024, 1, 15, 14, 30 + i),
            )
            ema.update(bar)
        assert ema.ready
        assert abs(ema.value - price) < Decimal("0.01")


class TestStreamingRSIEdgeCases:
    """StreamingRSI boundary behaviour."""

    def test_rsi_in_0_100_after_ready(self) -> None:
        rsi = StreamingRSI(period=5)
        for i in range(20):
            bar = make_bar(
                close=Decimal(str(480 + i * 0.1)),
                timestamp=make_utc_ts(2024, 1, 15, 14, 30 + i),
            )
            rsi.update(bar)
        assert rsi.ready
        assert Decimal("0") <= rsi.value <= Decimal("100")

    def test_all_increasing_approaches_100(self) -> None:
        """Monotonically increasing prices should push RSI near 100."""
        rsi = StreamingRSI(period=5)
        for i in range(30):
            bar = make_bar(
                open_=Decimal(str(400 + i)),
                high=Decimal(str(401 + i)),
                low=Decimal(str(399 + i)),
                close=Decimal(str(400 + i)),
                vwap=Decimal(str(400 + i)),
                timestamp=make_utc_ts(2024, 1, 15, 14, 30 + i),
            )
            rsi.update(bar)
        assert rsi.ready
        assert rsi.value > Decimal("90")


class TestStreamingATREdgeCases:
    """StreamingATR boundary behaviour."""

    def test_atr_positive_after_ready(self) -> None:
        atr = StreamingATR(period=5)
        for i in range(10):
            bar = make_bar(
                open_=Decimal(str(480 + i)),
                high=Decimal(str(482 + i)),
                low=Decimal(str(478 + i)),
                close=Decimal(str(481 + i)),
                vwap=Decimal(str(480 + i)),
                timestamp=make_utc_ts(2024, 1, 15, 14, 30 + i),
            )
            atr.update(bar)
        assert atr.ready
        assert atr.value > Decimal("0")


# ===========================================================================
# Risk — Position Sizing Edge Cases
# ===========================================================================


class TestCalculatePositionSize:
    """Edge cases for fixed-fractional position sizing."""

    def test_stop_at_entry_returns_zero(self) -> None:
        size = calculate_position_size(
            account_size=Decimal("50000"),
            risk_pct=Decimal("1.0"),
            entry=Decimal("480.00"),
            stop=Decimal("480.00"),
        )
        assert size == 0

    def test_very_tight_stop_returns_int(self) -> None:
        """Tight stop should produce a large integer (floor)."""
        size = calculate_position_size(
            account_size=Decimal("50000"),
            risk_pct=Decimal("1.0"),
            entry=Decimal("480.00"),
            stop=Decimal("479.99"),
        )
        assert isinstance(size, int)
        assert size > 0
        # $500 risk / $0.01 per share = 50000 shares
        assert size == 50000

    def test_negative_risk_pct(self) -> None:
        """Negative risk_pct yields negative dollar_risk -> size truncates toward 0."""
        size = calculate_position_size(
            account_size=Decimal("50000"),
            risk_pct=Decimal("-1.0"),
            entry=Decimal("480.00"),
            stop=Decimal("478.00"),
        )
        assert size <= 0

    def test_zero_account_size(self) -> None:
        size = calculate_position_size(
            account_size=Decimal("0"),
            risk_pct=Decimal("1.0"),
            entry=Decimal("480.00"),
            stop=Decimal("478.00"),
        )
        assert size == 0


# ===========================================================================
# Levels — SessionVWAP Edge Cases
# ===========================================================================


class TestSessionVWAPEdgeCases:
    """Edge cases for the session VWAP tracker."""

    def test_zero_volume_bar(self) -> None:
        """Zero-volume bar should not move VWAP (no contribution)."""
        vwap = SessionVWAP()
        # First bar with volume to establish a VWAP
        bar1 = make_bar(
            high=Decimal("481.00"),
            low=Decimal("479.00"),
            close=Decimal("480.00"),
            volume=1000,
            vwap=Decimal("480.00"),
            timestamp=make_utc_ts(2024, 1, 15, 14, 35),
        )
        vwap.update(bar1)
        vwap_after_one = vwap.vwap

        # Zero-volume bar should not change VWAP
        bar2 = make_bar(
            high=Decimal("490.00"),
            low=Decimal("470.00"),
            close=Decimal("495.00"),
            volume=0,
            vwap=Decimal("480.00"),
            timestamp=make_utc_ts(2024, 1, 15, 14, 36),
        )
        vwap.update(bar2)
        assert vwap.vwap == vwap_after_one

    def test_single_bar_equals_typical_price(self) -> None:
        """VWAP from a single bar should equal (H+L+C)/3."""
        vwap = SessionVWAP()
        h, l_, c = Decimal("482.00"), Decimal("478.00"), Decimal("480.00")
        bar = make_bar(
            high=h,
            low=l_,
            close=c,
            volume=5000,
            vwap=Decimal("480.00"),
            timestamp=make_utc_ts(2024, 1, 15, 14, 35),
        )
        vwap.update(bar)
        expected = (float(h) + float(l_) + float(c)) / 3.0
        assert abs(float(vwap.vwap) - expected) < 1e-4

    def test_vwap_resets_on_new_day(self) -> None:
        """VWAP should reset when a bar from a new trading date arrives."""
        vwap = SessionVWAP()
        bar_day1 = make_bar(
            high=Decimal("481.00"),
            low=Decimal("479.00"),
            close=Decimal("480.00"),
            volume=5000,
            vwap=Decimal("480.00"),
            timestamp=make_utc_ts(2024, 1, 15, 14, 35),
        )
        vwap.update(bar_day1)
        assert vwap.vwap is not None

        # Next day bar — VWAP resets; this premarket bar is ignored
        bar_day2_pre = make_bar(
            high=Decimal("490.00"),
            low=Decimal("488.00"),
            close=Decimal("489.00"),
            volume=1000,
            vwap=Decimal("489.00"),
            timestamp=make_utc_ts(2024, 1, 16, 13, 0),  # 8:00 ET — premarket
        )
        vwap.update(bar_day2_pre)
        # After reset + premarket bar ignored, VWAP should be None
        assert vwap.vwap is None


# ===========================================================================
# Levels — Opening Range Edge Cases
# ===========================================================================


class TestOpeningRangeEdgeCases:
    """Edge cases for the ORB tracker."""

    def test_single_bar_at_930_incomplete(self) -> None:
        """One bar at 9:30 ET is inside the range but ORB is not yet complete."""
        ort = OpeningRangeTracker()
        bar = make_bar(
            high=Decimal("481.00"),
            low=Decimal("479.00"),
            close=Decimal("480.00"),
            # 9:30 ET in January (EST) = 14:30 UTC
            timestamp=make_utc_ts(2024, 1, 15, 14, 30),
        )
        ort.update(bar)
        assert ort.orb_high is not None
        assert not ort.is_complete

    def test_orb_complete_after_935_bar(self) -> None:
        """ORB completes when a bar at or after 9:35 ET arrives."""
        ort = OpeningRangeTracker()
        # 5 bars: 9:30-9:34 ET (14:30-14:34 UTC)
        for minute in range(5):
            bar = make_bar(
                high=Decimal(str(481 + minute)),
                low=Decimal(str(479 - minute)),
                close=Decimal("480.00"),
                vwap=Decimal("480.00"),
                timestamp=make_utc_ts(2024, 1, 15, 14, 30 + minute),
            )
            ort.update(bar)
        assert not ort.is_complete  # need a bar at/after 9:35

        # Bar at 9:35 ET (14:35 UTC) triggers completion
        bar_935 = make_bar(
            high=Decimal("482.00"),
            low=Decimal("478.00"),
            close=Decimal("480.00"),
            vwap=Decimal("480.00"),
            timestamp=make_utc_ts(2024, 1, 15, 14, 35),
        )
        ort.update(bar_935)
        assert ort.is_complete

    def test_premarket_bars_ignored(self) -> None:
        """Bars before 9:30 ET should not affect the ORB range."""
        ort = OpeningRangeTracker()
        # 8:00 ET = 13:00 UTC in January
        pre_bar = make_bar(
            high=Decimal("500.00"),
            low=Decimal("400.00"),
            close=Decimal("450.00"),
            vwap=Decimal("450.00"),
            timestamp=make_utc_ts(2024, 1, 15, 13, 0),
        )
        ort.update(pre_bar)
        assert ort.orb_high is None
        assert ort.orb_low is None


# ===========================================================================
# Storage — BarDatabase Edge Cases
# ===========================================================================


class TestBarDatabaseEdgeCases:
    """Edge cases for SQLite bar storage."""

    def test_insert_duplicate_ignored(self, tmp_db: BarDatabase) -> None:
        """Inserting the same bar twice should silently ignore the duplicate."""
        bar = make_bar()
        inserted_first = tmp_db.insert_bars([bar])
        inserted_second = tmp_db.insert_bars([bar])
        assert inserted_first == 1
        assert inserted_second == 0
        assert tmp_db.count_bars("SPY", TimeFrame.ONE_MIN) == 1

    def test_query_empty_database(self, tmp_db: BarDatabase) -> None:
        """Querying an empty database returns an empty list, not an error."""
        bars = tmp_db.query_bars("SPY", TimeFrame.ONE_MIN)
        assert bars == []

    def test_wal_mode_enabled(self, tmp_db: BarDatabase) -> None:
        """WAL journal mode should be active after connect."""
        row = tmp_db.conn.execute("PRAGMA journal_mode;").fetchone()
        assert row["journal_mode"] == "wal"

    def test_count_no_bars_returns_zero(self, tmp_db: BarDatabase) -> None:
        assert tmp_db.count_bars("SPY", TimeFrame.ONE_MIN) == 0


# ===========================================================================
# Risk — Cooldown Edge Cases
# ===========================================================================


class TestCooldownEdgeCases:
    """Cooldown state-machine edge cases."""

    def test_one_loss_still_cooled_down(self) -> None:
        """1 loss is below threshold — is_cooled_down() returns True."""
        tracker = CooldownTracker()
        tracker.record_trade(_make_trade_result(Decimal("-100")))
        assert tracker.consecutive_losses == 1
        assert tracker.is_cooled_down()

    def test_two_losses_not_cooled_down(self) -> None:
        """Exactly 2 losses triggers cooldown — is_cooled_down() returns False."""
        now = make_utc_ts(2024, 1, 15, 15, 0)
        tracker = CooldownTracker(now_fn=lambda: now)
        tracker.record_trade(_make_trade_result(Decimal("-100"), ts=now))
        tracker.record_trade(_make_trade_result(Decimal("-50"), ts=now + timedelta(minutes=1)))
        assert tracker.consecutive_losses == 2
        assert not tracker.is_cooled_down()

    def test_three_losses_tilted(self) -> None:
        """3 consecutive losses → tilted (done for day)."""
        tracker = CooldownTracker()
        for _ in range(3):
            tracker.record_trade(_make_trade_result(Decimal("-100")))
        assert tracker.is_tilted()

    def test_win_after_two_losses_resets(self) -> None:
        """A win resets consecutive loss count to 0."""
        tracker = CooldownTracker()
        tracker.record_trade(_make_trade_result(Decimal("-100")))
        tracker.record_trade(_make_trade_result(Decimal("-100")))
        assert tracker.consecutive_losses == 2
        tracker.record_trade(_make_trade_result(Decimal("200")))
        assert tracker.consecutive_losses == 0
        assert tracker.is_cooled_down()
        assert not tracker.is_tilted()

    def test_daily_reset_clears_everything(self) -> None:
        """reset_daily() zeroes all counters."""
        tracker = CooldownTracker()
        for _ in range(3):
            tracker.record_trade(_make_trade_result(Decimal("-100")))
        assert tracker.is_tilted()

        tracker.reset_daily()
        assert tracker.consecutive_losses == 0
        assert tracker.daily_pnl == Decimal("0")
        assert tracker.daily_trade_count == 0
        assert tracker.is_cooled_down()
        assert not tracker.is_tilted()
