"""Integration: full morning startup event chain.

Tests the real pipeline wiring from bars → ORB formation → regime warmup →
signal evaluation, mocking only external I/O (Alpaca websocket, Telegram).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING

import pytest

from src.levels import LevelManager
from src.main import _VIX_FALLBACK, _build_registry, _SymbolPipeline
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
    """Create a 1-min bar at ``_BASE_TS + minute_offset`` minutes."""
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
    """5 bars for the ORB window (9:30-9:34 ET)."""
    return [
        _bar(0, 480.0, 481.5, 478.5, 480.0, symbol=symbol),
        _bar(1, 480.0, 482.0, 479.0, 481.0, symbol=symbol),
        _bar(2, 481.0, 481.8, 478.0, 479.5, symbol=symbol),
        _bar(3, 479.5, 481.5, 478.2, 480.5, symbol=symbol),
        _bar(4, 480.5, 481.7, 478.5, 481.0, symbol=symbol),
    ]


def _warmup_bars(start_offset: int, count: int, symbol: str = "SPY") -> list[Bar]:
    """Generate *count* bars starting at ``start_offset`` for warmup."""
    bars: list[Bar] = []
    for i in range(count):
        c = 480.0 + i * 0.02
        bars.append(_bar(start_offset + i, c - 0.2, c + 0.3, c - 0.3, c, 250_000, symbol))
    return bars


def _breakout_bars(start_offset: int, symbol: str = "SPY") -> list[Bar]:
    """Two bars above ORB high to trigger a breakout signal."""
    return [
        _bar(start_offset, 481.5, 483.0, 481.0, 482.5, 500_000, symbol),
        _bar(start_offset + 1, 482.5, 484.0, 482.0, 483.5, 600_000, symbol),
    ]


def _run_bars_through_pipeline(
    bars: list[Bar], symbol: str = "SPY"
) -> tuple[IndicatorRegistry, LevelManager, RegimeDetector, ORBStrategy, list[object]]:
    """Feed bars through a complete pipeline, return components and signals."""
    registry = _build_registry()
    levels = LevelManager(db=None, symbol=symbol)
    regime = RegimeDetector()
    strategy = ORBStrategy()
    signals: list[object] = []
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


# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.slow
class TestMorningStartupSequence:
    """Verify the full morning startup event chain end-to-end."""

    def test_normal_start_event_chain(self) -> None:
        """Verify: bars → ORB forms → regime populates → strategy evaluates."""
        orb = _orb_bars()
        warmup = _warmup_bars(5, 45)
        breakout = _breakout_bars(50)
        all_bars = orb + warmup + breakout

        registry, levels, regime, _strategy, _signals = _run_bars_through_pipeline(all_bars)

        # ORB must be complete after feeding 5 ORB bars + first post-ORB bar
        assert levels._orb.is_complete, "ORB should be complete after 5 bars + post-ORB bar"
        assert levels._orb.orb_high == Decimal("482.0")
        assert levels._orb.orb_low == Decimal("478.0")

        # Regime must have values after warmup
        assert regime.vix_level is not None, "VIX should be set via fallback"
        assert regime.adx_value is not None, "ADX should populate after warmup"
        assert regime.current_regime is not None, "Regime should be classified"

        # Snapshot should be fully populated
        snapshot = registry.get_snapshot()
        assert snapshot.ema9 is not None
        assert snapshot.atr is not None
        assert snapshot.adx is not None

    def test_late_start_backfill_triggers(self) -> None:
        """Verify: late start → backfill → ORB forms from REST bars."""
        pipeline = _SymbolPipeline(
            symbol="SPY",
            registry=_build_registry(),
            levels=LevelManager(db=None, symbol="SPY"),
            regime=RegimeDetector(),
            strategy=ORBStrategy(),
        )
        # Feed ORB bars directly into levels (simulating REST backfill)
        for b in _orb_bars():
            pipeline.levels.update(b)
        # Feed post-ORB bar to complete
        pipeline.levels.update(_bar(5, 480.0, 481.0, 479.0, 480.5))

        assert pipeline.levels._orb.is_complete, "ORB should form from backfilled bars"
        assert pipeline.levels._orb.orb_high == Decimal("482.0")
        assert pipeline.levels._orb.orb_low == Decimal("478.0")

    def test_warmup_bars_populate_all_indicators(self) -> None:
        """After enough bars, all 7 indicators must be non-None."""
        bars = _orb_bars() + _warmup_bars(5, 50)
        registry, _, _, _, _ = _run_bars_through_pipeline(bars)
        snapshot = registry.get_snapshot()
        assert snapshot.ema9 is not None
        assert snapshot.ema20 is not None
        assert snapshot.ema50 is not None
        assert snapshot.rsi is not None
        assert snapshot.atr is not None
        assert snapshot.adx is not None
        assert snapshot.macd is not None
