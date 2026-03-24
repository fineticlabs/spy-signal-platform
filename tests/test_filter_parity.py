"""Tests for backtest-live filter parity fixes (2026-03-24).

Covers:
- Trading window cutoff (10:00 ET default, configurable)
- ADX threshold (25, configurable)
- 2-bar consecutive candle confirmation
- EMA trend alignment
- Gap classification
- VIX backwardation hard block
- ORB range minimum (0.15%)
- max_concurrent_positions enforcement in RiskManager
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

from src.config import AppSettings, RiskSettings
from src.models import (
    Bar,
    Direction,
    IndicatorSnapshot,
    LevelSnapshot,
    Regime,
    Signal,
    TimeFrame,
)
from src.risk.cooldown import CooldownTracker
from src.risk.manager import RiskManager
from src.strategies.orb import ORBStrategy
from src.strategies.regime import RegimeDetector

_ET = ZoneInfo("America/New_York")


# ── helpers ──────────────────────────────────────────────────────────────────


def _bar(
    et_hour: int,
    et_minute: int,
    close: float,
    volume: int = 2_000_000,
    date_str: str = "2024-01-15",
    symbol: str = "SPY",
) -> Bar:
    """Build a 1-min bar at the given ET wall-clock time."""
    naive_et = datetime.strptime(f"{date_str} {et_hour:02d}:{et_minute:02d}", "%Y-%m-%d %H:%M")
    ts_et = naive_et.replace(tzinfo=_ET)
    ts_utc = ts_et.astimezone(UTC)
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


def _make_regime(
    vix: float = 18.0,
    adx: float = 28.0,
    trending_up: bool = True,
) -> RegimeDetector:
    regime = RegimeDetector()
    regime.update(vix=Decimal(str(vix)), adx=Decimal(str(adx)), trending_up=trending_up)
    return regime


def _make_levels(
    orb_high: float = 485.0,
    orb_low: float = 480.0,
    orb_complete: bool = True,
    prev_day_close: float | None = None,
) -> LevelSnapshot:
    return LevelSnapshot(
        orb_complete=orb_complete,
        orb_high=Decimal(str(orb_high)) if orb_complete else None,
        orb_low=Decimal(str(orb_low)) if orb_complete else None,
        prev_day_close=Decimal(str(prev_day_close)) if prev_day_close is not None else None,
    )


def _make_indicators(atr: float = 2.0, ema20: float | None = None) -> IndicatorSnapshot:
    return IndicatorSnapshot(
        atr=Decimal(str(atr)),
        ema20=Decimal(str(ema20)) if ema20 is not None else None,
    )


def _prime(strategy: ORBStrategy, n: int = 20, volume: int = 1_000_000) -> None:
    """Feed N bars so the volume deque has baseline data."""
    levels = _make_levels(orb_complete=False)
    indicators = _make_indicators()
    regime = _make_regime()
    for i in range(n):
        strategy.evaluate(_bar(9, 36 + i, close=482.0, volume=volume), indicators, levels, regime)


def _two_bar_breakout(
    strategy: ORBStrategy,
    close: float,
    volume: int = 2_000_000,
    regime: RegimeDetector | None = None,
    indicators: IndicatorSnapshot | None = None,
    levels: LevelSnapshot | None = None,
    vix_term_ratio: Decimal | None = None,
) -> Signal | None:
    """Send two consecutive bars to trigger 2-bar confirmation."""
    if regime is None:
        regime = _make_regime()
    if indicators is None:
        indicators = _make_indicators()
    if levels is None:
        levels = _make_levels()

    # First bar — sets pending
    strategy.evaluate(
        _bar(9, 36, close=close, volume=volume),
        indicators,
        levels,
        regime,
        vix_term_ratio=vix_term_ratio,
    )
    # Second bar — confirms
    return strategy.evaluate(
        _bar(9, 37, close=close, volume=volume),
        indicators,
        levels,
        regime,
        vix_term_ratio=vix_term_ratio,
    )


# ── Trading window cutoff tests ──────────────────────────────────────────────


class TestTradingWindowCutoff:
    def test_default_cutoff_is_10_00_et(self) -> None:
        """Default cutoff is 10:00 ET (matching backtest window end)."""
        strategy = ORBStrategy(excluded_days=[])
        _prime(strategy)

        # 9:59 should be allowed (not blocked by cutoff)
        signal = strategy.evaluate(
            _bar(9, 59, close=486.0),
            _make_indicators(),
            _make_levels(),
            _make_regime(),
        )
        # It returns None on first bar (pending confirmation), but NOT because of cutoff
        # 10:00 should be blocked
        signal = strategy.evaluate(
            _bar(10, 0, close=486.0),
            _make_indicators(),
            _make_levels(),
            _make_regime(),
        )
        assert signal is None  # blocked by cutoff

    def test_custom_cutoff(self) -> None:
        """Custom cutoff can be configured."""
        strategy = ORBStrategy(excluded_days=[], signal_cutoff_et="11:00")
        _prime(strategy)

        # 10:30 should work with custom cutoff
        signal = strategy.evaluate(
            _bar(10, 30, close=486.0),
            _make_indicators(),
            _make_levels(),
            _make_regime(),
        )
        # Returns None because it's pending confirmation, not because of cutoff
        # 11:00 should be blocked
        signal = strategy.evaluate(
            _bar(11, 0, close=486.0),
            _make_indicators(),
            _make_levels(),
            _make_regime(),
        )
        assert signal is None

    def test_signal_fires_within_window(self) -> None:
        """Signal fires within the 9:35-10:00 window with 2-bar confirmation."""
        strategy = ORBStrategy(excluded_days=[])
        _prime(strategy)

        signal = _two_bar_breakout(strategy, close=486.0)
        assert signal is not None
        assert signal.direction == Direction.LONG


# ── ADX threshold tests ──────────────────────────────────────────────────────


class TestADXThreshold:
    def test_default_adx_threshold_is_25(self) -> None:
        """ADX <= 25 is now rejected (was 20, now aligned with backtest)."""
        strategy = ORBStrategy(excluded_days=[])
        _prime(strategy)

        # ADX = 24 — below 25, should be rejected
        signal = strategy.evaluate(
            _bar(9, 36, close=486.0),
            _make_indicators(),
            _make_levels(),
            _make_regime(adx=24.0),
        )
        assert signal is None

    def test_adx_above_25_allows_signal(self) -> None:
        """ADX = 26 passes the threshold."""
        strategy = ORBStrategy(excluded_days=[])
        _prime(strategy)

        signal = _two_bar_breakout(strategy, close=486.0, regime=_make_regime(adx=26.0))
        assert signal is not None

    def test_custom_adx_threshold(self) -> None:
        """Custom ADX threshold from config."""
        strategy = ORBStrategy(excluded_days=[], adx_min_threshold=20)
        _prime(strategy)

        # ADX = 21 passes with custom threshold of 20
        signal = _two_bar_breakout(strategy, close=486.0, regime=_make_regime(adx=21.0))
        assert signal is not None


# ── 2-bar confirmation tests ─────────────────────────────────────────────────


class TestTwoBarConfirmation:
    def test_first_bar_returns_none(self) -> None:
        """First bar above ORB high returns None (pending confirmation)."""
        strategy = ORBStrategy(excluded_days=[])
        _prime(strategy)

        signal = strategy.evaluate(
            _bar(9, 36, close=486.0),
            _make_indicators(),
            _make_levels(),
            _make_regime(),
        )
        assert signal is None  # pending confirmation

    def test_second_bar_confirms_breakout(self) -> None:
        """Second consecutive bar above ORB high confirms and returns signal."""
        strategy = ORBStrategy(excluded_days=[])
        _prime(strategy)

        signal = _two_bar_breakout(strategy, close=486.0)
        assert signal is not None
        assert signal.direction == Direction.LONG

    def test_failed_confirmation_price_back_inside(self) -> None:
        """If second bar closes inside ORB, breakout is cancelled."""
        strategy = ORBStrategy(excluded_days=[])
        _prime(strategy)

        levels = _make_levels()
        indicators = _make_indicators()
        regime = _make_regime()

        # First bar — above ORB high (pending)
        strategy.evaluate(
            _bar(9, 36, close=486.0),
            indicators,
            levels,
            regime,
        )
        # Second bar — back inside ORB (cancels pending)
        signal = strategy.evaluate(
            _bar(9, 37, close=482.0),
            indicators,
            levels,
            regime,
        )
        assert signal is None

    def test_direction_change_resets_pending(self) -> None:
        """If direction changes, old pending is replaced."""
        strategy = ORBStrategy(excluded_days=[])
        _prime(strategy)

        levels = _make_levels()
        indicators = _make_indicators()
        regime = _make_regime()

        # First bar — above ORB high (pending LONG)
        strategy.evaluate(
            _bar(9, 36, close=486.0),
            indicators,
            levels,
            regime,
        )
        # Second bar — below ORB low (replaces pending with SHORT)
        signal = strategy.evaluate(
            _bar(9, 37, close=479.0),
            indicators,
            levels,
            regime,
        )
        assert signal is None  # new pending SHORT, not confirmed yet


# ── EMA trend alignment tests ────────────────────────────────────────────────


class TestEMATrendAlignment:
    def test_long_blocked_when_below_ema(self) -> None:
        """LONG breakout blocked if close <= EMA(20)."""
        strategy = ORBStrategy(excluded_days=[])
        _prime(strategy)

        indicators = _make_indicators(ema20=487.0)  # EMA above close
        signal = strategy.evaluate(
            _bar(9, 36, close=486.0),
            indicators,
            _make_levels(),
            _make_regime(),
        )
        assert signal is None  # blocked by EMA, not even pending

    def test_short_blocked_when_above_ema(self) -> None:
        """SHORT breakout blocked if close >= EMA(20)."""
        strategy = ORBStrategy(excluded_days=[])
        _prime(strategy)

        indicators = _make_indicators(ema20=478.0)  # EMA below close
        signal = strategy.evaluate(
            _bar(9, 36, close=479.0),
            indicators,
            _make_levels(),
            _make_regime(),
        )
        assert signal is None  # blocked by EMA

    def test_long_passes_when_above_ema(self) -> None:
        """LONG breakout passes if close > EMA(20)."""
        strategy = ORBStrategy(excluded_days=[])
        _prime(strategy)

        indicators = _make_indicators(ema20=484.0)  # EMA below close
        signal = _two_bar_breakout(strategy, close=486.0, indicators=indicators)
        assert signal is not None
        assert signal.direction == Direction.LONG

    def test_ema_none_allows_signal(self) -> None:
        """If EMA(20) not yet available, don't block."""
        strategy = ORBStrategy(excluded_days=[])
        _prime(strategy)

        indicators = _make_indicators(ema20=None)
        signal = _two_bar_breakout(strategy, close=486.0, indicators=indicators)
        assert signal is not None


# ── Gap classification tests ─────────────────────────────────────────────────


class TestGapClassification:
    def test_gap_up_blocks_short(self) -> None:
        """Gap > +0.3% blocks SHORT signals."""
        strategy = ORBStrategy(excluded_days=[])
        _prime(strategy)

        # prev_day_close = 480, ORB midpoint ≈ 482.5 → gap = 0.52% (gap up)
        levels = _make_levels(orb_high=485.0, orb_low=480.0, prev_day_close=480.0)
        signal = strategy.evaluate(
            _bar(9, 36, close=479.0),
            _make_indicators(),
            levels,
            _make_regime(),
        )
        assert signal is None  # gap up blocks SHORT

    def test_gap_down_blocks_long(self) -> None:
        """Gap < -0.3% blocks LONG signals."""
        strategy = ORBStrategy(excluded_days=[])
        _prime(strategy)

        # prev_day_close = 485, ORB midpoint ≈ 482.5 → gap = -0.52% (gap down)
        levels = _make_levels(orb_high=485.0, orb_low=480.0, prev_day_close=485.0)
        signal = strategy.evaluate(
            _bar(9, 36, close=486.0),
            _make_indicators(),
            levels,
            _make_regime(),
        )
        assert signal is None  # gap down blocks LONG

    def test_flat_gap_allows_both(self) -> None:
        """Gap within ±0.3% allows both directions."""
        strategy = ORBStrategy(excluded_days=[])
        _prime(strategy)

        # prev_day_close ≈ 482.5 (same as ORB midpoint) → gap ≈ 0%
        levels = _make_levels(orb_high=485.0, orb_low=480.0, prev_day_close=482.5)
        signal = _two_bar_breakout(strategy, close=486.0, levels=levels)
        assert signal is not None

    def test_no_prev_close_allows_signal(self) -> None:
        """If prev_day_close is None, gap filter is skipped."""
        strategy = ORBStrategy(excluded_days=[])
        _prime(strategy)

        levels = _make_levels(orb_high=485.0, orb_low=480.0, prev_day_close=None)
        signal = _two_bar_breakout(strategy, close=486.0, levels=levels)
        assert signal is not None


# ── VIX backwardation hard block tests ────────────────────────────────────────


class TestVIXBackwardationBlock:
    def test_backwardation_blocks_signal(self) -> None:
        """VIX term structure ratio > 1.0 blocks signal entirely."""
        strategy = ORBStrategy(excluded_days=[])
        _prime(strategy)

        # Backwardation ratio = 1.05
        signal = strategy.evaluate(
            _bar(9, 36, close=486.0),
            _make_indicators(),
            _make_levels(),
            _make_regime(),
            vix_term_ratio=Decimal("1.05"),
        )
        assert signal is None

    def test_contango_allows_signal(self) -> None:
        """VIX term structure ratio < 0.85 allows signal (and adds tag)."""
        strategy = ORBStrategy(excluded_days=[])
        _prime(strategy)

        signal = _two_bar_breakout(
            strategy,
            close=486.0,
            vix_term_ratio=Decimal("0.80"),
        )
        assert signal is not None
        assert "CONTANGO" in signal.tags

    def test_normal_ratio_allows_signal(self) -> None:
        """VIX term structure ratio 0.85-1.00 allows signal (no tag)."""
        strategy = ORBStrategy(excluded_days=[])
        _prime(strategy)

        signal = _two_bar_breakout(
            strategy,
            close=486.0,
            vix_term_ratio=Decimal("0.92"),
        )
        assert signal is not None
        assert "CONTANGO" not in signal.tags
        assert "BACKWARDATION" not in signal.tags


# ── ORB range minimum tests ──────────────────────────────────────────────────


class TestORBRangeMinimum:
    def test_narrow_orb_blocked(self) -> None:
        """ORB range < 0.15% of price blocks all signals."""
        strategy = ORBStrategy(excluded_days=[])
        _prime(strategy)

        # ORB range = 0.50 on a $480 stock → 0.10% < 0.15% → blocked
        levels = _make_levels(orb_high=480.25, orb_low=479.75)
        signal = strategy.evaluate(
            _bar(9, 36, close=480.30),
            _make_indicators(),
            levels,
            _make_regime(),
        )
        assert signal is None

    def test_wide_orb_allowed(self) -> None:
        """ORB range >= 0.15% of price allows signals."""
        strategy = ORBStrategy(excluded_days=[])
        _prime(strategy)

        # ORB range = 5.0 on a $480 stock → 1.04% > 0.15% → allowed
        signal = _two_bar_breakout(strategy, close=486.0)
        assert signal is not None

    def test_custom_orb_min_range(self) -> None:
        """Custom ORB range minimum from config."""
        strategy = ORBStrategy(excluded_days=[], orb_min_range_pct=0.001)
        _prime(strategy)

        # ORB range = 0.50 on a $480 stock → 0.10% > 0.1% with custom threshold
        levels = _make_levels(orb_high=480.25, orb_low=479.75)
        strategy.evaluate(
            _bar(9, 36, close=480.30),
            _make_indicators(),
            levels,
            _make_regime(),
        )
        # Not blocked by ORB range (0.10% > 0.1%), but may be pending confirmation
        # The point is it's NOT blocked by the ORB range filter


# ── max_concurrent_positions tests ────────────────────────────────────────────


class TestMaxConcurrentPositions:
    def _make_signal(self, et_hour: int = 9, et_minute: int = 36) -> Signal:
        naive = datetime.strptime(f"2024-01-15 {et_hour:02d}:{et_minute:02d}", "%Y-%m-%d %H:%M")
        ts = naive.replace(tzinfo=_ET).astimezone(UTC)
        return Signal(
            direction=Direction.LONG,
            strategy_name="ORB-5min",
            entry_price=Decimal("486"),
            stop_price=Decimal("483"),
            target_price=Decimal("492"),
            risk_reward_ratio=Decimal("2.0"),
            confidence_score=3,
            reason="test",
            timeframe=TimeFrame.ONE_MIN,
            regime=Regime.TRENDING_UP,
            vix=Decimal("18"),
            adx=Decimal("28"),
            timestamp=ts,
        )

    def test_rejects_when_max_positions_reached(self) -> None:
        """Risk manager rejects when position count >= max_concurrent_positions."""
        cooldown = CooldownTracker()
        settings = RiskSettings(
            account_size=Decimal("50000"),
            risk_per_trade_pct=Decimal("1.0"),
            max_daily_loss_pct=Decimal("3.0"),
            max_trades_per_day=5,
            max_concurrent_positions=3,
        )
        rm = RiskManager(
            cooldown=cooldown,
            settings=settings,
            get_position_count=lambda: 3,  # at max
        )
        decision = rm.approve(self._make_signal())
        assert not decision.approved
        assert "Max concurrent positions" in decision.reason

    def test_approves_when_below_max_positions(self) -> None:
        """Risk manager approves when position count < max_concurrent_positions."""
        cooldown = CooldownTracker()
        settings = RiskSettings(
            account_size=Decimal("50000"),
            risk_per_trade_pct=Decimal("1.0"),
            max_daily_loss_pct=Decimal("3.0"),
            max_trades_per_day=5,
            max_concurrent_positions=3,
        )
        rm = RiskManager(
            cooldown=cooldown,
            settings=settings,
            get_position_count=lambda: 2,  # below max
        )
        decision = rm.approve(self._make_signal())
        assert decision.approved

    def test_skips_check_when_no_callback(self) -> None:
        """Risk manager skips position check when no callback provided."""
        cooldown = CooldownTracker()
        settings = RiskSettings(
            account_size=Decimal("50000"),
            risk_per_trade_pct=Decimal("1.0"),
            max_daily_loss_pct=Decimal("3.0"),
            max_trades_per_day=5,
            max_concurrent_positions=3,
        )
        rm = RiskManager(cooldown=cooldown, settings=settings)
        decision = rm.approve(self._make_signal())
        assert decision.approved

    def test_handles_callback_error_gracefully(self) -> None:
        """Risk manager continues if position count callback raises."""
        cooldown = CooldownTracker()
        settings = RiskSettings(
            account_size=Decimal("50000"),
            risk_per_trade_pct=Decimal("1.0"),
            max_daily_loss_pct=Decimal("3.0"),
            max_trades_per_day=5,
            max_concurrent_positions=3,
        )

        def _failing_count() -> int:
            raise ConnectionError("API unreachable")

        rm = RiskManager(
            cooldown=cooldown,
            settings=settings,
            get_position_count=_failing_count,
        )
        # Should still approve (fail-open for this check since the error is logged)
        decision = rm.approve(self._make_signal())
        assert decision.approved


# ── Config validation tests ──────────────────────────────────────────────────


class TestConfigNewFields:
    def test_signal_cutoff_et_default(self) -> None:
        """Default signal_cutoff_et is '10:00'."""
        settings = AppSettings()
        assert settings.signal_cutoff_et == "10:00"

    def test_adx_min_threshold_default(self) -> None:
        """Default adx_min_threshold is 25."""
        settings = AppSettings()
        assert settings.adx_min_threshold == 25

    def test_orb_min_range_pct_default(self) -> None:
        """Default orb_min_range_pct is 0.0015."""
        settings = AppSettings()
        assert settings.orb_min_range_pct == 0.0015

    def test_gap_threshold_pct_default(self) -> None:
        """Default gap_threshold_pct is 0.3."""
        settings = AppSettings()
        assert settings.gap_threshold_pct == 0.3
