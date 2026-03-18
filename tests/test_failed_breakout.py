"""Tests for the FailedBreakoutStrategy state machine."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from src.models import (
    Bar,
    Direction,
    IndicatorSnapshot,
    LevelSnapshot,
)
from src.strategies.failed_breakout import FailedBreakoutStrategy
from src.strategies.regime import RegimeDetector
from tests.conftest import make_bar

# ── Helpers ──────────────────────────────────────────────────────────────────

# 14:35 UTC on a January day = 9:35 ET (EST, UTC-5).  ORB is complete.
_BASE_TS = datetime(2024, 1, 15, 14, 35, tzinfo=UTC)

# ORB boundaries used across most tests
_ORB_HIGH = Decimal("482.00")
_ORB_LOW = Decimal("478.00")
_ATR = Decimal("2.00")


def _ts(minutes_offset: int = 0, *, day_offset: int = 0) -> datetime:
    """Return a UTC timestamp offset from the base."""
    return _BASE_TS + timedelta(minutes=minutes_offset, days=day_offset)


def _bar(
    close: Decimal,
    minutes_offset: int = 0,
    volume: int = 100_000,
    *,
    day_offset: int = 0,
) -> Bar:
    """Build a Bar at a specific time with the given close price."""
    c = close
    return make_bar(
        timestamp=_ts(minutes_offset, day_offset=day_offset),
        open_=c,
        high=c + Decimal("0.10"),
        low=c - Decimal("0.10"),
        close=c,
        volume=volume,
    )


def _indicators(atr: Decimal | None = _ATR) -> IndicatorSnapshot:
    return IndicatorSnapshot(atr=atr)


def _levels(
    orb_high: Decimal | None = _ORB_HIGH,
    orb_low: Decimal | None = _ORB_LOW,
    orb_complete: bool = True,
) -> LevelSnapshot:
    return LevelSnapshot(orb_high=orb_high, orb_low=orb_low, orb_complete=orb_complete)


def _regime(
    vix: Decimal = Decimal("20"),
    adx: Decimal = Decimal("25"),
) -> RegimeDetector:
    r = RegimeDetector()
    r.update(vix=vix, adx=adx)
    return r


def _prime_volume(strategy: FailedBreakoutStrategy, n: int = 20) -> None:
    """Feed *n* bars into the strategy so the volume deque is populated.

    Uses a time well inside filters (14:36-14:55 UTC = 9:36-9:55 ET).
    """
    regime = _regime()
    levels = _levels()
    indicators = _indicators()
    for i in range(n):
        bar = _bar(Decimal("480.00"), minutes_offset=i + 1, volume=100_000)
        strategy.evaluate(bar, indicators, levels, regime)


# ── Tests: interface / metadata ──────────────────────────────────────────────


class TestInterface:
    """Strategy interface compliance."""

    def test_name_returns_failed_breakout(self) -> None:
        assert FailedBreakoutStrategy().name == "FailedBreakout"

    def test_required_indicators_returns_atr(self) -> None:
        assert FailedBreakoutStrategy().required_indicators() == ["atr"]


# ── Tests: _detect_setup ─────────────────────────────────────────────────────


class TestDetectSetup:
    """Unit tests for the setup-detection helper."""

    def test_failed_long_setup_close_above_orb_high(self) -> None:
        """Close 0.7 ATR above ORB high → failed long (-1)."""
        s = FailedBreakoutStrategy()
        # 482 + 0.7*2 = 483.40  (above = 1.40, min_ext=1.0, max_ext=3.0)
        result = s._detect_setup(Decimal("483.40"), _ORB_HIGH, _ORB_LOW, _ATR)
        assert result == -1

    def test_failed_short_setup_close_below_orb_low(self) -> None:
        """Close 0.7 ATR below ORB low → failed short (+1)."""
        s = FailedBreakoutStrategy()
        # 478 - 0.7*2 = 476.60  (below = 1.40)
        result = s._detect_setup(Decimal("476.60"), _ORB_HIGH, _ORB_LOW, _ATR)
        assert result == +1

    def test_no_setup_extension_too_small(self) -> None:
        """Close < 0.5 ATR past ORB → too small, no setup."""
        s = FailedBreakoutStrategy()
        # 482 + 0.3*2 = 482.60  (above = 0.60, min_ext = 1.0 → not triggered)
        result = s._detect_setup(Decimal("482.60"), _ORB_HIGH, _ORB_LOW, _ATR)
        assert result == 0

    def test_no_setup_extension_too_large(self) -> None:
        """Close > 1.5 ATR past ORB → real breakout, no setup."""
        s = FailedBreakoutStrategy()
        # 482 + 1.6*2 = 485.20  (above = 3.20, max_ext = 3.0 → not triggered)
        result = s._detect_setup(Decimal("485.20"), _ORB_HIGH, _ORB_LOW, _ATR)
        assert result == 0

    def test_no_setup_inside_orb(self) -> None:
        """Close inside ORB → no setup."""
        s = FailedBreakoutStrategy()
        result = s._detect_setup(Decimal("480.00"), _ORB_HIGH, _ORB_LOW, _ATR)
        assert result == 0

    def test_boundary_exactly_min_ext_not_triggered(self) -> None:
        """Extension == 0.5 ATR is NOT triggered (strict >)."""
        s = FailedBreakoutStrategy()
        # 482 + 0.5*2 = 483.00  (above = 1.00, min_ext = 1.00 → above is NOT > min_ext)
        result = s._detect_setup(Decimal("483.00"), _ORB_HIGH, _ORB_LOW, _ATR)
        assert result == 0

    def test_boundary_exactly_max_ext_is_triggered(self) -> None:
        """Extension == 1.5 ATR IS triggered (<=)."""
        s = FailedBreakoutStrategy()
        # 482 + 1.5*2 = 485.00  (above = 3.00, max_ext = 3.00 → above <= max_ext)
        result = s._detect_setup(Decimal("485.00"), _ORB_HIGH, _ORB_LOW, _ATR)
        assert result == -1


# ── Tests: filters ───────────────────────────────────────────────────────────


class TestFilters:
    """Verify that hard filters gate the strategy correctly."""

    def test_orb_not_complete_returns_none(self) -> None:
        s = FailedBreakoutStrategy()
        _prime_volume(s)
        bar = _bar(Decimal("483.40"), minutes_offset=25)
        result = s.evaluate(bar, _indicators(), _levels(orb_complete=False), _regime())
        assert result is None

    def test_vix_too_high_returns_none(self) -> None:
        s = FailedBreakoutStrategy()
        _prime_volume(s)
        bar = _bar(Decimal("483.40"), minutes_offset=25)
        result = s.evaluate(bar, _indicators(), _levels(), _regime(vix=Decimal("25")))
        assert result is None

    def test_adx_too_low_returns_none(self) -> None:
        s = FailedBreakoutStrategy()
        _prime_volume(s)
        bar = _bar(Decimal("483.40"), minutes_offset=25)
        result = s.evaluate(bar, _indicators(), _levels(), _regime(adx=Decimal("20")))
        assert result is None

    def test_after_cutoff_returns_none(self) -> None:
        """Bar at 15:45 ET (20:45 UTC in January) → filtered."""
        s = FailedBreakoutStrategy()
        _prime_volume(s)
        # 20:45 UTC = 15:45 ET in January (EST)
        late_ts = datetime(2024, 1, 15, 20, 45, tzinfo=UTC)
        bar = make_bar(timestamp=late_ts, close=Decimal("483.40"))
        result = s.evaluate(bar, _indicators(), _levels(), _regime())
        assert result is None

    def test_atr_none_returns_none(self) -> None:
        s = FailedBreakoutStrategy()
        _prime_volume(s)
        bar = _bar(Decimal("483.40"), minutes_offset=25)
        result = s.evaluate(bar, _indicators(atr=None), _levels(), _regime())
        assert result is None

    def test_orb_high_none_returns_none(self) -> None:
        s = FailedBreakoutStrategy()
        _prime_volume(s)
        bar = _bar(Decimal("483.40"), minutes_offset=25)
        result = s.evaluate(bar, _indicators(), _levels(orb_high=None), _regime())
        assert result is None

    def test_vix_none_returns_none(self) -> None:
        s = FailedBreakoutStrategy()
        _prime_volume(s)
        bar = _bar(Decimal("483.40"), minutes_offset=25)
        regime = RegimeDetector()
        regime.update(adx=Decimal("25"))
        result = s.evaluate(bar, _indicators(), _levels(), regime)
        assert result is None


# ── Tests: full reversal sequence ────────────────────────────────────────────


class TestReversalShort:
    """Failed long → SHORT reversal signal."""

    def test_full_short_signal(self) -> None:
        """Setup above ORB high, then reversal bar back inside ORB with volume → SHORT."""
        s = FailedBreakoutStrategy()
        _prime_volume(s)
        regime = _regime()
        levels = _levels()
        indicators = _indicators()

        # Setup bar: close 0.7 ATR above ORB high → 483.40
        setup = _bar(Decimal("483.40"), minutes_offset=25)
        result = s.evaluate(setup, indicators, levels, regime)
        assert result is None  # no signal on setup bar

        # Reversal bar: close back inside ORB (below orb_high) with high volume
        # close = 480 → risk = 483.40 - 480 = 3.40, reward = 480 - 478 = 2
        # R:R = 2/3.40 ≈ 0.59 — too low.  Use close nearer to ORB high for good R:R.
        # close = 481.50 → risk = 483.40 - 481.50 = 1.90, reward = 481.50 - 478 = 3.50
        # R:R = 3.50/1.90 ≈ 1.84 ✓
        reversal = _bar(Decimal("481.50"), minutes_offset=26, volume=200_000)
        result = s.evaluate(reversal, indicators, levels, regime)
        assert result is not None
        assert result.direction == Direction.SHORT
        assert result.entry_price == Decimal("481.50")

    def test_short_signal_stop_at_extreme(self) -> None:
        """Stop price equals the extreme close of the fake breakout."""
        s = FailedBreakoutStrategy()
        _prime_volume(s)
        regime = _regime()
        levels = _levels()
        indicators = _indicators()

        setup = _bar(Decimal("483.40"), minutes_offset=25)
        s.evaluate(setup, indicators, levels, regime)

        reversal = _bar(Decimal("481.50"), minutes_offset=26, volume=200_000)
        result = s.evaluate(reversal, indicators, levels, regime)
        assert result is not None
        # Extreme is updated: max(setup_close=483.40, reversal_close=481.50) = 483.40
        assert result.stop_price == Decimal("483.40")

    def test_short_signal_target_at_orb_low(self) -> None:
        """Target equals opposite ORB boundary (ORB low)."""
        s = FailedBreakoutStrategy()
        _prime_volume(s)
        regime = _regime()
        levels = _levels()
        indicators = _indicators()

        setup = _bar(Decimal("483.40"), minutes_offset=25)
        s.evaluate(setup, indicators, levels, regime)

        reversal = _bar(Decimal("481.50"), minutes_offset=26, volume=200_000)
        result = s.evaluate(reversal, indicators, levels, regime)
        assert result is not None
        assert result.target_price == _ORB_LOW

    def test_short_close_must_be_below_orb_high(self) -> None:
        """Reversal close still above ORB high → no signal."""
        s = FailedBreakoutStrategy()
        _prime_volume(s)
        regime = _regime()
        levels = _levels()
        indicators = _indicators()

        setup = _bar(Decimal("483.40"), minutes_offset=25)
        s.evaluate(setup, indicators, levels, regime)

        # Close at exactly orb_high → not below → no signal
        still_above = _bar(Decimal("482.00"), minutes_offset=26, volume=200_000)
        result = s.evaluate(still_above, indicators, levels, regime)
        assert result is None


class TestReversalLong:
    """Failed short → LONG reversal signal."""

    def test_full_long_signal(self) -> None:
        """Setup below ORB low, then reversal bar back inside ORB with volume → LONG."""
        s = FailedBreakoutStrategy()
        _prime_volume(s)
        regime = _regime()
        levels = _levels()
        indicators = _indicators()

        # Setup bar: close 0.7 ATR below ORB low → 478 - 1.40 = 476.60
        setup = _bar(Decimal("476.60"), minutes_offset=25)
        s.evaluate(setup, indicators, levels, regime)

        # Reversal bar: close back inside ORB (above orb_low) with high volume
        # close = 478.50 → risk = 478.50 - 476.60 = 1.90, reward = 482 - 478.50 = 3.50
        # R:R = 3.50/1.90 ≈ 1.84 ✓
        reversal = _bar(Decimal("478.50"), minutes_offset=26, volume=200_000)
        result = s.evaluate(reversal, indicators, levels, regime)
        assert result is not None
        assert result.direction == Direction.LONG
        assert result.entry_price == Decimal("478.50")

    def test_long_signal_stop_at_extreme(self) -> None:
        """Stop price equals extreme (lowest close) of the fake breakout."""
        s = FailedBreakoutStrategy()
        _prime_volume(s)
        regime = _regime()
        levels = _levels()
        indicators = _indicators()

        setup = _bar(Decimal("476.60"), minutes_offset=25)
        s.evaluate(setup, indicators, levels, regime)

        reversal = _bar(Decimal("478.50"), minutes_offset=26, volume=200_000)
        result = s.evaluate(reversal, indicators, levels, regime)
        assert result is not None
        # Extreme is min(setup=476.60, reversal=478.50) = 476.60
        assert result.stop_price == Decimal("476.60")

    def test_long_signal_target_at_orb_high(self) -> None:
        """Target equals opposite ORB boundary (ORB high)."""
        s = FailedBreakoutStrategy()
        _prime_volume(s)
        regime = _regime()
        levels = _levels()
        indicators = _indicators()

        setup = _bar(Decimal("476.60"), minutes_offset=25)
        s.evaluate(setup, indicators, levels, regime)

        reversal = _bar(Decimal("478.50"), minutes_offset=26, volume=200_000)
        result = s.evaluate(reversal, indicators, levels, regime)
        assert result is not None
        assert result.target_price == _ORB_HIGH

    def test_long_close_must_be_above_orb_low(self) -> None:
        """Reversal close still at or below ORB low → no signal."""
        s = FailedBreakoutStrategy()
        _prime_volume(s)
        regime = _regime()
        levels = _levels()
        indicators = _indicators()

        setup = _bar(Decimal("476.60"), minutes_offset=25)
        s.evaluate(setup, indicators, levels, regime)

        # Close at exactly orb_low → not above → no signal
        still_below = _bar(Decimal("478.00"), minutes_offset=26, volume=200_000)
        result = s.evaluate(still_below, indicators, levels, regime)
        assert result is None


# ── Tests: expiry, volume, R:R ───────────────────────────────────────────────


class TestStateMachine:
    """State machine edge cases: expiry, volume, R:R filters."""

    def test_reversal_expires_after_5_bars(self) -> None:
        """After 5 bars with no reversal → state cleared, no signal."""
        s = FailedBreakoutStrategy()
        _prime_volume(s)
        regime = _regime()
        levels = _levels()
        indicators = _indicators()

        # Setup bar
        setup = _bar(Decimal("483.40"), minutes_offset=25)
        s.evaluate(setup, indicators, levels, regime)

        # 5 bars inside the watch window, but price stays above orb_high (no reversal)
        for i in range(5):
            bar = _bar(Decimal("482.50"), minutes_offset=26 + i, volume=200_000)
            result = s.evaluate(bar, indicators, levels, regime)

        # The 5th bar should have expired the watch
        assert result is None
        assert s._fb_active is False

    def test_reversal_insufficient_volume(self) -> None:
        """Reversal bar with volume < 1.5x avg → no signal."""
        s = FailedBreakoutStrategy()
        _prime_volume(s)  # avg volume = 100,000
        regime = _regime()
        levels = _levels()
        indicators = _indicators()

        setup = _bar(Decimal("483.40"), minutes_offset=25)
        s.evaluate(setup, indicators, levels, regime)

        # Volume = 100,000.  Need >= 150,000.
        low_vol = _bar(Decimal("481.50"), minutes_offset=26, volume=100_000)
        result = s.evaluate(low_vol, indicators, levels, regime)
        assert result is None

    def test_reversal_rr_too_low_short(self) -> None:
        """Reversal for SHORT with R:R < 1.5 → no signal."""
        s = FailedBreakoutStrategy()
        _prime_volume(s)
        regime = _regime()
        levels = _levels()
        indicators = _indicators()

        # Setup: close 0.7 ATR above ORB high → 483.40
        setup = _bar(Decimal("483.40"), minutes_offset=25)
        s.evaluate(setup, indicators, levels, regime)

        # Reversal close = 479.00
        # risk = 483.40 - 479.00 = 4.40
        # reward = 479.00 - 478.00 = 1.00
        # R:R = 1.0/4.4 = 0.23 → below 1.5
        bad_rr = _bar(Decimal("479.00"), minutes_offset=26, volume=200_000)
        result = s.evaluate(bad_rr, indicators, levels, regime)
        assert result is None

    def test_reversal_rr_too_low_long(self) -> None:
        """Reversal for LONG with R:R < 1.5 → no signal."""
        s = FailedBreakoutStrategy()
        _prime_volume(s)
        regime = _regime()
        levels = _levels()
        indicators = _indicators()

        # Setup: close 0.7 ATR below ORB low → 476.60
        setup = _bar(Decimal("476.60"), minutes_offset=25)
        s.evaluate(setup, indicators, levels, regime)

        # Reversal close = 481.00
        # risk = 481.00 - 476.60 = 4.40
        # reward = 482.00 - 481.00 = 1.00
        # R:R = 1.0/4.4 = 0.23 → below 1.5
        bad_rr = _bar(Decimal("481.00"), minutes_offset=26, volume=200_000)
        result = s.evaluate(bad_rr, indicators, levels, regime)
        assert result is None


# ── Tests: volume tracking ───────────────────────────────────────────────────


class TestVolumeTracking:
    """Rolling average volume helper."""

    def test_avg_volume_empty_deque_returns_none(self) -> None:
        s = FailedBreakoutStrategy()
        assert s._avg_volume() is None

    def test_avg_volume_after_bars(self) -> None:
        s = FailedBreakoutStrategy()
        s._volumes.append(100)
        s._volumes.append(200)
        avg = s._avg_volume()
        assert avg is not None
        assert avg == Decimal("150.0")


# ── Tests: daily reset ───────────────────────────────────────────────────────


class TestDailyReset:
    """New trading date clears the failed-breakout state."""

    def test_new_date_clears_fb_state(self) -> None:
        s = FailedBreakoutStrategy()
        _prime_volume(s)
        regime = _regime()
        levels = _levels()
        indicators = _indicators()

        # Setup bar on day 0
        setup = _bar(Decimal("483.40"), minutes_offset=25, day_offset=0)
        s.evaluate(setup, indicators, levels, regime)
        assert s._fb_active is True

        # First bar on the next day → reset
        next_day = _bar(Decimal("480.00"), minutes_offset=25, day_offset=1)
        s.evaluate(next_day, indicators, levels, regime)
        assert s._fb_active is False


# ── Tests: extreme tracking ──────────────────────────────────────────────────


class TestExtremeTracking:
    """Verify that the FB extreme is updated correctly during the watch window."""

    def test_extreme_updated_for_failed_long(self) -> None:
        """For a failed long (trade_dir=-1), extreme = max close seen."""
        s = FailedBreakoutStrategy()
        _prime_volume(s)
        regime = _regime()
        levels = _levels()
        indicators = _indicators()

        setup = _bar(Decimal("483.40"), minutes_offset=25)
        s.evaluate(setup, indicators, levels, regime)

        # Next bar closes even higher — extreme should update
        higher = _bar(Decimal("483.80"), minutes_offset=26, volume=50_000)
        s.evaluate(higher, indicators, levels, regime)
        assert s._fb_extreme == Decimal("483.80")

    def test_extreme_updated_for_failed_short(self) -> None:
        """For a failed short (trade_dir=+1), extreme = min close seen."""
        s = FailedBreakoutStrategy()
        _prime_volume(s)
        regime = _regime()
        levels = _levels()
        indicators = _indicators()

        setup = _bar(Decimal("476.60"), minutes_offset=25)
        s.evaluate(setup, indicators, levels, regime)

        # Next bar closes even lower — extreme should update
        lower = _bar(Decimal("476.20"), minutes_offset=26, volume=50_000)
        s.evaluate(lower, indicators, levels, regime)
        assert s._fb_extreme == Decimal("476.20")

    def test_stop_reflects_updated_extreme(self) -> None:
        """Stop on the signal uses the worst extreme, not the setup close."""
        s = FailedBreakoutStrategy()
        _prime_volume(s)
        regime = _regime()
        levels = _levels()
        indicators = _indicators()

        # Setup
        setup = _bar(Decimal("483.40"), minutes_offset=25)
        s.evaluate(setup, indicators, levels, regime)

        # Bar that pushes extreme higher (low volume, no reversal)
        push = _bar(Decimal("484.00"), minutes_offset=26, volume=50_000)
        s.evaluate(push, indicators, levels, regime)

        # Reversal with volume on bar 3
        # risk = 484.00 - 481.50 = 2.50, reward = 481.50 - 478 = 3.50, R:R = 1.4
        # Need close closer to ORB high for R:R >= 1.5
        # close = 481.80 → risk = 484.00 - 481.80 = 2.20, reward = 481.80 - 478 = 3.80
        # R:R = 3.80/2.20 ≈ 1.73 ✓
        reversal = _bar(Decimal("481.80"), minutes_offset=27, volume=200_000)
        result = s.evaluate(reversal, indicators, levels, regime)
        assert result is not None
        assert result.stop_price == Decimal("484.00")


# ── Tests: signal metadata ───────────────────────────────────────────────────


class TestSignalMetadata:
    """Verify signal fields are populated correctly."""

    def test_signal_strategy_name(self) -> None:
        s = FailedBreakoutStrategy()
        _prime_volume(s)
        regime = _regime()
        levels = _levels()
        indicators = _indicators()

        setup = _bar(Decimal("483.40"), minutes_offset=25)
        s.evaluate(setup, indicators, levels, regime)

        reversal = _bar(Decimal("481.50"), minutes_offset=26, volume=200_000)
        result = s.evaluate(reversal, indicators, levels, regime)
        assert result is not None
        assert result.strategy_name == "FailedBreakout"

    def test_signal_confidence_score_is_3(self) -> None:
        s = FailedBreakoutStrategy()
        _prime_volume(s)
        regime = _regime()
        levels = _levels()
        indicators = _indicators()

        setup = _bar(Decimal("483.40"), minutes_offset=25)
        s.evaluate(setup, indicators, levels, regime)

        reversal = _bar(Decimal("481.50"), minutes_offset=26, volume=200_000)
        result = s.evaluate(reversal, indicators, levels, regime)
        assert result is not None
        assert result.confidence_score == 3

    def test_signal_clears_fb_active_state(self) -> None:
        """After signal fires, fb_active must be False."""
        s = FailedBreakoutStrategy()
        _prime_volume(s)
        regime = _regime()
        levels = _levels()
        indicators = _indicators()

        setup = _bar(Decimal("483.40"), minutes_offset=25)
        s.evaluate(setup, indicators, levels, regime)

        reversal = _bar(Decimal("481.50"), minutes_offset=26, volume=200_000)
        s.evaluate(reversal, indicators, levels, regime)
        assert s._fb_active is False
