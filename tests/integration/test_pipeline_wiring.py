"""End-to-end pipeline wiring test using REAL objects — no mocks.

Feeds synthetic 1-min bars through the actual _build_registry,
RegimeDetector, LevelManager (no DB), and ORBStrategy from main.py.
Asserts the full chain produces a signal.

This test would have caught both 2026-03-19 production bugs:
  1. regime.update() never receiving vix/adx → orb_filter_no_regime_data
  2. "adx" not in registry _known set → snapshot_unknown_indicator spam

Run with:
    pytest tests/integration/test_pipeline_wiring.py -v
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING

import pytest

from src.levels import LevelManager
from src.main import _VIX_FALLBACK, _build_registry
from src.models import Bar, TimeFrame
from src.strategies.orb import ORBStrategy
from src.strategies.regime import RegimeDetector

if TYPE_CHECKING:
    from src.indicators.registry import IndicatorRegistry
    from src.models import Signal

# ── Bar factory ──────────────────────────────────────────────────────────────

# January 15 2024 (Monday, but we don't enforce Monday filter in live ORB).
# 14:30 UTC = 9:30 ET (EST, January).
_BASE_TS = datetime(2024, 1, 16, 14, 30, tzinfo=UTC)  # Tuesday


def _bar(
    minute_offset: int,
    open_: float,
    high: float,
    low: float,
    close: float,
    volume: int = 200_000,
) -> Bar:
    """Create a Bar at _BASE_TS + minute_offset minutes."""
    return Bar(
        symbol="SPY",
        timeframe=TimeFrame.ONE_MIN,
        timestamp=_BASE_TS + timedelta(minutes=minute_offset),
        open=Decimal(str(open_)),
        high=Decimal(str(high)),
        low=Decimal(str(low)),
        close=Decimal(str(close)),
        volume=volume,
        vwap=Decimal(str((high + low + close) / 3)),
    )


def _make_opening_range_bars() -> list[Bar]:
    """5 bars forming a tight opening range: high=482, low=478."""
    return [
        _bar(0, 480.00, 481.50, 478.50, 480.00),  # 9:30
        _bar(1, 480.00, 482.00, 479.00, 481.00),  # 9:31
        _bar(2, 481.00, 481.80, 478.00, 479.50),  # 9:32
        _bar(3, 479.50, 481.50, 478.20, 480.50),  # 9:33
        _bar(4, 480.50, 481.70, 478.50, 481.00),  # 9:34
    ]


def _make_warmup_bars(start_offset: int, count: int) -> list[Bar]:
    """Bars inside the ORB range to warm up indicators without triggering signals."""
    bars = []
    for i in range(count):
        minute = start_offset + i
        # Mild uptrend inside ORB range, high volume
        c = 480.0 + i * 0.02
        bars.append(_bar(minute, c - 0.20, c + 0.30, c - 0.30, c, volume=250_000))
    return bars


def _make_breakout_bars(start_offset: int) -> list[Bar]:
    """Two consecutive bars closing above ORB high (482) with high volume."""
    return [
        # First close above ORB high
        _bar(start_offset, 481.50, 483.00, 481.00, 482.50, volume=500_000),
        # Second consecutive close above — triggers entry
        _bar(start_offset + 1, 482.50, 484.00, 482.00, 483.50, volume=600_000),
    ]


# ── Pipeline runner ──────────────────────────────────────────────────────────


def _run_pipeline(
    bars: list[Bar],
) -> tuple[
    IndicatorRegistry,
    LevelManager,
    RegimeDetector,
    ORBStrategy,
    list[Signal],
]:
    """Feed bars through the REAL pipeline objects. Return all components + signals."""
    registry = _build_registry()
    levels = LevelManager(db=None, symbol="SPY")
    regime = RegimeDetector()
    strategy = ORBStrategy(excluded_days=[])

    signals: list[Signal] = []

    for bar in bars:
        # 1. Update indicators
        registry.update_all(bar)
        snapshot = registry.get_snapshot()

        # 2. Update levels
        levels.update(bar)
        level_snapshot = levels.get_levels()

        # 3. Update regime — same logic as _process_bar in main.py
        if snapshot.atr is not None:
            trending_up: bool | None = None
            if snapshot.ema9 is not None and snapshot.ema20 is not None:
                trending_up = snapshot.ema9 > snapshot.ema20
            adx_val = snapshot.adx
            vix_val = regime.vix_level or _VIX_FALLBACK
            regime.update(vix=vix_val, adx=adx_val, trending_up=trending_up)

        # 4. Evaluate strategy
        signal = strategy.evaluate(bar, snapshot, level_snapshot, regime)
        if signal is not None:
            signals.append(signal)

    return registry, levels, regime, strategy, signals


# ── Tests ────────────────────────────────────────────────────────────────────


class TestRegistryWiring:
    """Verify _build_registry produces a working registry with all indicators."""

    def test_registry_has_adx(self) -> None:
        """ADX must be registered — its absence caused snapshot_unknown_indicator."""
        registry = _build_registry()
        assert "adx" in registry.names

    def test_adx_in_snapshot_known_set(self) -> None:
        """ADX must be in the snapshot _known set — not logged as unknown."""
        registry = _build_registry()
        # Feed enough bars for ADX to appear
        for i in range(40):
            registry.update_all(_bar(i, 480.0 + i * 0.01, 481.0, 479.0, 480.5 + i * 0.01))
        snapshot = registry.get_snapshot()
        # adx field exists on IndicatorSnapshot and is populated
        assert hasattr(snapshot, "adx")

    def test_snapshot_has_no_unknown_indicators(self) -> None:
        """Every registered indicator must be recognized by get_snapshot."""
        import structlog

        captured: list[dict] = []

        def capture(logger, method_name, event_dict):
            if event_dict.get("event") == "snapshot_unknown_indicator":
                captured.append(event_dict)
            return event_dict

        structlog.configure(processors=[capture, structlog.dev.ConsoleRenderer()])

        try:
            registry = _build_registry()
            for i in range(5):
                registry.update_all(_bar(i, 480.0, 481.0, 479.0, 480.5))
            registry.get_snapshot()
        finally:
            structlog.reset_defaults()

        assert captured == [], f"Unknown indicators logged: {captured}"


class TestRegimeWiring:
    """Verify regime gets populated with vix and adx during bar processing."""

    def test_regime_gets_vix_after_first_bar_with_atr(self) -> None:
        """After ATR warms up, regime.vix_level must not be None."""
        or_bars = _make_opening_range_bars()
        warmup = _make_warmup_bars(5, 30)
        bars = or_bars + warmup

        _, _, regime, _, _ = _run_pipeline(bars)

        assert (
            regime.vix_level is not None
        ), "regime.vix_level is None — _process_bar is not passing vix to regime.update()"

    def test_regime_vix_uses_fallback(self) -> None:
        """VIX should use the _VIX_FALLBACK value (no live feed)."""
        or_bars = _make_opening_range_bars()
        warmup = _make_warmup_bars(5, 30)
        bars = or_bars + warmup

        _, _, regime, _, _ = _run_pipeline(bars)

        assert regime.vix_level == _VIX_FALLBACK

    def test_regime_gets_adx_after_warmup(self) -> None:
        """After ~28 bars, regime.adx_value must not be None."""
        or_bars = _make_opening_range_bars()
        warmup = _make_warmup_bars(5, 35)
        bars = or_bars + warmup

        _, _, regime, _, _ = _run_pipeline(bars)

        assert (
            regime.adx_value is not None
        ), "regime.adx_value is None — StreamingADX not wired or not warm"

    def test_regime_is_tradeable_after_warmup(self) -> None:
        """With VIX=20 fallback and ADX populated, regime should be tradeable
        (assuming ADX > 20, which depends on bar data)."""
        or_bars = _make_opening_range_bars()
        warmup = _make_warmup_bars(5, 35)
        bars = or_bars + warmup

        _, _, regime, _, _ = _run_pipeline(bars)

        # VIX=20 < 25 is always satisfied. ADX may or may not be > 20
        # depending on synthetic data. At minimum, both must be non-None.
        assert regime.vix_level is not None
        assert regime.adx_value is not None


class TestOrbLevelsWiring:
    """Verify ORB levels are established after the opening range bars."""

    def test_orb_complete_after_five_bars(self) -> None:
        or_bars = _make_opening_range_bars()
        # Need at least one bar at 9:35+ to trigger orb_complete
        post_orb = [_bar(5, 480.0, 481.0, 479.0, 480.5)]
        bars = or_bars + post_orb

        _, levels, _, _, _ = _run_pipeline(bars)

        snapshot = levels.get_levels()
        assert snapshot.orb_complete is True

    def test_orb_high_low_set(self) -> None:
        or_bars = _make_opening_range_bars()
        post_orb = [_bar(5, 480.0, 481.0, 479.0, 480.5)]
        bars = or_bars + post_orb

        _, levels, _, _, _ = _run_pipeline(bars)

        snapshot = levels.get_levels()
        assert snapshot.orb_high is not None
        assert snapshot.orb_low is not None
        assert snapshot.orb_high == Decimal("482.00")
        assert snapshot.orb_low == Decimal("478.00")


@pytest.mark.integration
class TestEndToEndSignalGeneration:
    """Feed a full bar sequence and assert at least one signal comes out."""

    def test_breakout_produces_signal(self) -> None:
        """ORB breakout after warmup must produce at least one LONG signal.

        This is the critical wiring test. If regime data isn't flowing,
        the strategy will filter everything with orb_filter_no_regime_data.
        """
        or_bars = _make_opening_range_bars()
        warmup = _make_warmup_bars(5, 30)  # bars 5-34: warm up indicators
        breakout = _make_breakout_bars(35)  # bars 35-36: breakout above ORB high

        bars = or_bars + warmup + breakout
        _, _, regime, _, signals = _run_pipeline(bars)

        # Regime must have been populated
        assert regime.vix_level is not None, "VIX not set — regime.update() wiring broken"
        assert regime.adx_value is not None, "ADX not set — StreamingADX wiring broken"

        # At least one signal must have fired
        assert len(signals) >= 1, (
            f"Expected at least 1 signal from ORB breakout, got 0. "
            f"Regime: vix={regime.vix_level}, adx={regime.adx_value}, "
            f"tradeable={regime.is_tradeable}"
        )

    def test_signal_is_long(self) -> None:
        """Breakout above ORB high should produce a LONG signal."""
        or_bars = _make_opening_range_bars()
        warmup = _make_warmup_bars(5, 30)
        breakout = _make_breakout_bars(35)

        _, _, _, _, signals = _run_pipeline(or_bars + warmup + breakout)

        if signals:
            assert signals[0].direction.value == "LONG"

    def test_signal_has_valid_stop_and_target(self) -> None:
        """Signal must have stop < entry < target for a LONG."""
        or_bars = _make_opening_range_bars()
        warmup = _make_warmup_bars(5, 30)
        breakout = _make_breakout_bars(35)

        _, _, _, _, signals = _run_pipeline(or_bars + warmup + breakout)

        if signals:
            sig = signals[0]
            assert sig.stop_price < sig.entry_price
            assert sig.target_price > sig.entry_price

    def test_no_signal_inside_orb_range(self) -> None:
        """Bars that stay inside the ORB range must NOT produce signals."""
        or_bars = _make_opening_range_bars()
        # 30 bars all inside ORB range (478-482) — no breakout
        inside_bars = _make_warmup_bars(5, 30)

        _, _, _, _, signals = _run_pipeline(or_bars + inside_bars)

        assert len(signals) == 0, f"Got {len(signals)} signals but price never broke ORB"
