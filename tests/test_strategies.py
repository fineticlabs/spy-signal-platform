"""Tests for trading strategy implementations."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

from src.models import Bar, Direction, IndicatorSnapshot, LevelSnapshot, Regime, TimeFrame
from src.strategies.orb import ORBStrategy
from src.strategies.regime import RegimeDetector

_ET = ZoneInfo("America/New_York")

# ── helpers ───────────────────────────────────────────────────────────────────


def _bar(
    et_hour: int,
    et_minute: int,
    close: float,
    volume: int = 1_000_000,
    date_str: str = "2024-01-15",
) -> Bar:
    """Build a 1-min bar at the given ET wall-clock time."""
    naive_et = datetime.strptime(f"{date_str} {et_hour:02d}:{et_minute:02d}", "%Y-%m-%d %H:%M")
    ts_et = naive_et.replace(tzinfo=_ET)
    ts_utc = ts_et.astimezone(UTC)
    return Bar(
        symbol="SPY",
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
    regime.update(
        vix=Decimal(str(vix)),
        adx=Decimal(str(adx)),
        trending_up=trending_up,
    )
    return regime


def _make_levels(
    orb_complete: bool = True,
    orb_high: float = 485.0,
    orb_low: float = 480.0,
) -> LevelSnapshot:
    return LevelSnapshot(
        orb_complete=orb_complete,
        orb_high=Decimal(str(orb_high)) if orb_complete else None,
        orb_low=Decimal(str(orb_low)) if orb_complete else None,
    )


def _make_indicators(atr: float = 2.0) -> IndicatorSnapshot:
    return IndicatorSnapshot(atr=Decimal(str(atr)))


def _prime(
    strategy: ORBStrategy,
    n: int = 20,
    volume: int = 1_000_000,
) -> None:
    """Feed N bars (incomplete ORB) so the volume deque has baseline data."""
    levels = _make_levels(orb_complete=False)
    indicators = _make_indicators()
    regime = _make_regime()
    for i in range(n):
        strategy.evaluate(_bar(9, 36 + i, close=482.0, volume=volume), indicators, levels, regime)


# ── RegimeDetector tests ──────────────────────────────────────────────────────


class TestRegimeDetector:
    def test_no_data_is_ranging(self) -> None:
        regime = RegimeDetector()
        assert regime.current_regime == Regime.RANGING

    def test_choppy_when_adx_below_15(self) -> None:
        regime = RegimeDetector()
        regime.update(adx=Decimal("12"))
        assert regime.current_regime == Regime.CHOPPY

    def test_trending_up_when_adx_above_25_and_direction_up(self) -> None:
        regime = RegimeDetector()
        regime.update(adx=Decimal("30"), trending_up=True)
        assert regime.current_regime == Regime.TRENDING_UP

    def test_trending_down_when_adx_above_25_and_direction_down(self) -> None:
        regime = RegimeDetector()
        regime.update(adx=Decimal("30"), trending_up=False)
        assert regime.current_regime == Regime.TRENDING_DOWN

    def test_ranging_when_adx_between_15_and_25(self) -> None:
        regime = RegimeDetector()
        regime.update(adx=Decimal("20"))
        assert regime.current_regime == Regime.RANGING

    def test_not_tradeable_without_data(self) -> None:
        regime = RegimeDetector()
        assert not regime.is_tradeable

    def test_not_tradeable_when_vix_high(self) -> None:
        regime = _make_regime(vix=26.0, adx=28.0)
        assert not regime.is_tradeable

    def test_not_tradeable_when_adx_low(self) -> None:
        regime = _make_regime(vix=18.0, adx=15.0)
        assert not regime.is_tradeable

    def test_tradeable_when_conditions_met(self) -> None:
        regime = _make_regime(vix=18.0, adx=28.0)
        assert regime.is_tradeable


# ── ORBStrategy tests ─────────────────────────────────────────────────────────


class TestORBStrategy:
    def test_long_signal_fires_above_orb_high(self) -> None:
        """LONG signal when close > ORB high with sufficient volume."""
        strategy = ORBStrategy()
        _prime(strategy, n=20, volume=1_000_000)

        levels = _make_levels(orb_high=485.0, orb_low=480.0)
        indicators = _make_indicators(atr=2.0)
        regime = _make_regime()

        # Close above ORB high, volume 2x average
        signal = strategy.evaluate(
            _bar(10, 0, close=486.0, volume=2_000_000),
            indicators,
            levels,
            regime,
        )

        assert signal is not None
        assert signal.direction == Direction.LONG
        assert signal.entry_price == Decimal("486.0")

    def test_short_signal_fires_below_orb_low(self) -> None:
        """SHORT signal when close < ORB low with sufficient volume."""
        strategy = ORBStrategy()
        _prime(strategy, n=20, volume=1_000_000)

        levels = _make_levels(orb_high=485.0, orb_low=480.0)
        indicators = _make_indicators(atr=2.0)
        regime = _make_regime(trending_up=False)

        signal = strategy.evaluate(
            _bar(10, 0, close=479.0, volume=2_000_000),
            indicators,
            levels,
            regime,
        )

        assert signal is not None
        assert signal.direction == Direction.SHORT
        assert signal.entry_price == Decimal("479.0")

    def test_no_signal_inside_orb_range(self) -> None:
        """No signal when close is between ORB high and low."""
        strategy = ORBStrategy()
        _prime(strategy, n=20, volume=1_000_000)

        signal = strategy.evaluate(
            _bar(10, 0, close=482.5, volume=2_000_000),
            _make_indicators(),
            _make_levels(orb_high=485.0, orb_low=480.0),
            _make_regime(),
        )
        assert signal is None

    def test_no_signal_before_orb_complete(self) -> None:
        """No signal while the ORB window is still open."""
        strategy = ORBStrategy()
        _prime(strategy, n=20, volume=1_000_000)

        signal = strategy.evaluate(
            _bar(9, 33, close=486.0, volume=2_000_000),
            _make_indicators(),
            _make_levels(orb_complete=False),
            _make_regime(),
        )
        assert signal is None

    def test_no_signal_during_lunch_chop(self) -> None:
        """No signal between 11:30 and 13:30 ET."""
        strategy = ORBStrategy()
        _prime(strategy, n=20, volume=1_000_000)

        signal = strategy.evaluate(
            _bar(12, 0, close=486.0, volume=2_000_000),
            _make_indicators(),
            _make_levels(orb_high=485.0, orb_low=480.0),
            _make_regime(),
        )
        assert signal is None

    def test_no_signal_after_cutoff(self) -> None:
        """No signal at or after 15:45 ET."""
        strategy = ORBStrategy()
        _prime(strategy, n=20, volume=1_000_000)

        signal = strategy.evaluate(
            _bar(15, 45, close=486.0, volume=2_000_000),
            _make_indicators(),
            _make_levels(orb_high=485.0, orb_low=480.0),
            _make_regime(),
        )
        assert signal is None

    def test_no_signal_when_vix_too_high(self) -> None:
        """No signal when VIX >= 25."""
        strategy = ORBStrategy()
        _prime(strategy, n=20, volume=1_000_000)

        signal = strategy.evaluate(
            _bar(10, 0, close=486.0, volume=2_000_000),
            _make_indicators(),
            _make_levels(orb_high=485.0, orb_low=480.0),
            _make_regime(vix=26.0),
        )
        assert signal is None

    def test_no_signal_when_adx_too_low(self) -> None:
        """No signal when ADX <= 20."""
        strategy = ORBStrategy()
        _prime(strategy, n=20, volume=1_000_000)

        signal = strategy.evaluate(
            _bar(10, 0, close=486.0, volume=2_000_000),
            _make_indicators(),
            _make_levels(orb_high=485.0, orb_low=480.0),
            _make_regime(adx=18.0),
        )
        assert signal is None

    def test_no_signal_when_volume_insufficient(self) -> None:
        """No signal when volume < 1.5x 20-bar average."""
        strategy = ORBStrategy()
        _prime(strategy, n=20, volume=1_000_000)

        # Volume only 1.2x average — below the 1.5x threshold
        signal = strategy.evaluate(
            _bar(10, 0, close=486.0, volume=1_200_000),
            _make_indicators(),
            _make_levels(orb_high=485.0, orb_low=480.0),
            _make_regime(),
        )
        assert signal is None

    def test_long_stop_and_target_calculated_from_atr(self) -> None:
        """LONG: stop = entry - 1.5*ATR, target = entry + 2*risk."""
        strategy = ORBStrategy()
        _prime(strategy, n=20, volume=1_000_000)

        atr = Decimal("2.0")
        entry = Decimal("486.0")
        signal = strategy.evaluate(
            _bar(10, 0, close=float(entry), volume=2_000_000),
            _make_indicators(atr=float(atr)),
            _make_levels(orb_high=485.0, orb_low=480.0),
            _make_regime(),
        )

        assert signal is not None
        expected_stop = entry - Decimal("1.5") * atr  # 483.0
        expected_risk = entry - expected_stop  # 3.0
        expected_target = entry + Decimal("2.0") * expected_risk  # 492.0
        assert signal.stop_price == expected_stop
        assert signal.target_price == expected_target
        assert signal.risk_reward_ratio == Decimal("2.0")

    def test_short_stop_and_target_calculated_from_atr(self) -> None:
        """SHORT: stop = entry + 1.5*ATR, target = entry - 2*risk."""
        strategy = ORBStrategy()
        _prime(strategy, n=20, volume=1_000_000)

        atr = Decimal("2.0")
        entry = Decimal("479.0")
        signal = strategy.evaluate(
            _bar(10, 0, close=float(entry), volume=2_000_000),
            _make_indicators(atr=float(atr)),
            _make_levels(orb_high=485.0, orb_low=480.0),
            _make_regime(trending_up=False),
        )

        assert signal is not None
        expected_stop = entry + Decimal("1.5") * atr  # 482.0
        expected_risk = expected_stop - entry  # 3.0
        expected_target = entry - Decimal("2.0") * expected_risk  # 473.0
        assert signal.stop_price == expected_stop
        assert signal.target_price == expected_target

    def test_signal_carries_regime_context(self) -> None:
        """Signal includes VIX, ADX, and regime on the output."""
        strategy = ORBStrategy()
        _prime(strategy, n=20, volume=1_000_000)

        regime = _make_regime(vix=18.5, adx=28.0, trending_up=True)
        signal = strategy.evaluate(
            _bar(10, 0, close=486.0, volume=2_000_000),
            _make_indicators(),
            _make_levels(orb_high=485.0, orb_low=480.0),
            regime,
        )

        assert signal is not None
        assert signal.vix == Decimal("18.5")
        assert signal.adx == Decimal("28.0")
        assert signal.regime == Regime.TRENDING_UP

    def test_no_signal_without_atr(self) -> None:
        """No signal when ATR is None (indicator not yet warmed up)."""
        strategy = ORBStrategy()
        _prime(strategy, n=20, volume=1_000_000)

        signal = strategy.evaluate(
            _bar(10, 0, close=486.0, volume=2_000_000),
            IndicatorSnapshot(),  # all None
            _make_levels(orb_high=485.0, orb_low=480.0),
            _make_regime(),
        )
        assert signal is None
