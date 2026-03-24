"""Integration: bar -> indicator -> snapshot -> levels -> regime wiring.

Tests the REAL chain with no mocking — ensures indicators, levels, and regime
are wired correctly through the pipeline.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING

from src.levels import LevelManager
from src.main import _VIX_FALLBACK, _build_registry
from src.models import Bar, TimeFrame
from src.strategies.orb import ORBStrategy
from src.strategies.regime import RegimeDetector

if TYPE_CHECKING:
    from src.indicators.registry import IndicatorRegistry

# ── Bar factory ──────────────────────────────────────────────────────────────

_BASE_TS = datetime(2024, 1, 16, 14, 30, tzinfo=UTC)  # Tuesday 9:30 ET


def _bar(
    minute_offset: int,
    open_: float,
    high: float,
    low: float,
    close: float,
    volume: int = 200_000,
    symbol: str = "SPY",
) -> Bar:
    return Bar(
        symbol=symbol,
        timeframe=TimeFrame.ONE_MIN,
        timestamp=_BASE_TS + timedelta(minutes=minute_offset),
        open=Decimal(str(open_)),
        high=Decimal(str(high)),
        low=Decimal(str(low)),
        close=Decimal(str(close)),
        volume=volume,
        vwap=Decimal(str((high + low + close) / 3)),
    )


def _orb_bars(symbol: str = "SPY") -> list[Bar]:
    return [
        _bar(0, 480.0, 481.5, 478.5, 480.0, symbol=symbol),
        _bar(1, 480.0, 482.0, 479.0, 481.0, symbol=symbol),
        _bar(2, 481.0, 481.8, 478.0, 479.5, symbol=symbol),
        _bar(3, 479.5, 481.5, 478.2, 480.5, symbol=symbol),
        _bar(4, 480.5, 481.7, 478.5, 481.0, symbol=symbol),
    ]


def _warmup_bars(start_offset: int, count: int, symbol: str = "SPY") -> list[Bar]:
    bars = []
    for i in range(count):
        c = 480.0 + i * 0.02
        bars.append(_bar(start_offset + i, c - 0.2, c + 0.3, c - 0.3, c, 250_000, symbol))
    return bars


def _breakout_bars(start_offset: int, symbol: str = "SPY") -> list[Bar]:
    return [
        _bar(start_offset, 481.5, 483.0, 481.0, 482.5, 500_000, symbol),
        _bar(start_offset + 1, 482.5, 484.0, 482.0, 483.5, 600_000, symbol),
    ]


def _run_bars_through_pipeline(
    bars: list[Bar], symbol: str = "SPY"
) -> tuple[IndicatorRegistry, LevelManager, RegimeDetector, ORBStrategy, list]:
    registry = _build_registry()
    levels = LevelManager(db=None, symbol=symbol)
    regime = RegimeDetector()
    strategy = ORBStrategy(excluded_days=[], signal_cutoff_et="15:45", adx_min_threshold=15)
    signals = []
    for b in bars:
        registry.update_all(b)
        snapshot = registry.get_snapshot()
        levels.update(b)
        level_snapshot = levels.get_levels()
        if snapshot.atr is not None:
            trending_up = None
            if snapshot.ema9 is not None and snapshot.ema20 is not None:
                trending_up = snapshot.ema9 > snapshot.ema20
            regime.update(
                vix=regime.vix_level or _VIX_FALLBACK,
                adx=snapshot.adx,
                trending_up=trending_up,
            )
        signal = strategy.evaluate(b, snapshot, level_snapshot, regime)
        if signal is not None:
            signals.append(signal)
    return registry, levels, regime, strategy, signals


# ── Tests ────────────────────────────────────────────────────────────────────


class TestBarToIndicatorWiring:
    """Verify each bar reaches all registered indicators."""

    def test_bar_reaches_all_seven_indicators(self) -> None:
        """Prevents: indicator silently not receiving bar updates."""
        registry = _build_registry()
        b = _bar(0, 480.0, 481.0, 479.0, 480.0)
        registry.update_all(b)
        # All 7 indicators received the bar without error
        assert len(registry) == 7
        for name in ["ema9", "ema20", "ema50", "rsi", "macd", "atr", "adx"]:
            assert name in registry.names

    def test_snapshot_fully_populated_after_warmup(self) -> None:
        """Prevents: indicator returning None forever due to warmup miscalculation."""
        registry = _build_registry()
        bars = _orb_bars() + _warmup_bars(5, 50)
        for b in bars:
            registry.update_all(b)
        snapshot = registry.get_snapshot()
        assert snapshot.ema9 is not None
        assert snapshot.ema20 is not None
        assert snapshot.ema50 is not None
        assert snapshot.rsi is not None
        assert snapshot.atr is not None
        assert snapshot.adx is not None
        assert snapshot.macd is not None

    def test_adx_in_snapshot_not_none_after_warmup(self) -> None:
        """Prevents: the 2026-03-19 bug where ADX was not in snapshot _known set."""
        registry = _build_registry()
        bars = _orb_bars() + _warmup_bars(5, 45)
        for b in bars:
            registry.update_all(b)
        snapshot = registry.get_snapshot()
        assert (
            snapshot.adx is not None
        ), "ADX is None after 50 bars — StreamingADX not wired or not in _known set"


class TestOrbLevelsWiring:
    """Verify ORB levels are established after the 5-bar opening range."""

    def test_orb_levels_set_after_five_bars(self) -> None:
        """Prevents: ORB never completing due to timestamp/timezone mismatch."""
        levels = LevelManager(db=None, symbol="SPY")
        for b in _orb_bars():
            levels.update(b)
        # Feed one post-ORB bar to finalize
        levels.update(_bar(5, 480.0, 481.0, 479.0, 480.5))
        snap = levels.get_levels()
        assert snap.orb_complete is True
        assert snap.orb_high is not None
        assert snap.orb_low is not None
        assert snap.orb_high == Decimal("482.00")
        assert snap.orb_low == Decimal("478.00")


class TestRegimeWiring:
    """Verify regime detector receives VIX and ADX values from the pipeline."""

    def test_regime_update_called_with_vix_and_adx(self) -> None:
        """Prevents: the 2026-03-19 bug where regime.update() never received vix/adx."""
        bars = _orb_bars() + _warmup_bars(5, 35)
        _, _, regime, _, _ = _run_bars_through_pipeline(bars)
        assert regime.vix_level is not None, "VIX never passed to regime.update()"
        assert regime.adx_value is not None, "ADX never passed to regime.update()"

    def test_regime_vix_never_none_after_atr_warmup(self) -> None:
        """Prevents: VIX fallback not applied, leaving regime.vix_level as None."""
        bars = _orb_bars() + _warmup_bars(5, 30)
        _, _, regime, _, _ = _run_bars_through_pipeline(bars)
        assert regime.vix_level == _VIX_FALLBACK


class TestEdgeCaseBars:
    """Verify pipeline doesn't crash on edge-case bar data."""

    def test_high_volume_bar_doesnt_crash(self) -> None:
        """Prevents: integer overflow or unexpected behavior with extreme volume."""
        bars = [*_orb_bars(), _bar(5, 480.0, 481.0, 479.0, 480.5, volume=999_999_999)]
        _run_bars_through_pipeline(bars)  # no exception = pass

    def test_zero_volume_bar_doesnt_crash(self) -> None:
        """Prevents: division by zero in volume-average calculations."""
        bars = [*_orb_bars(), _bar(5, 480.0, 481.0, 479.0, 480.5, volume=0)]
        _run_bars_through_pipeline(bars)  # no exception = pass

    def test_price_gap_bar_doesnt_crash(self) -> None:
        """Prevents: ATR/indicator blow-up on a 10% gap-up open."""
        bars = _orb_bars() + _warmup_bars(5, 20)
        gap_bar = _bar(25, 528.0, 530.0, 527.0, 529.0, 300_000)
        bars.append(gap_bar)
        _run_bars_through_pipeline(bars)  # no exception = pass
