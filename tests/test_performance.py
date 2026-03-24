"""Performance tests for the spy-signal-platform.

All tests are marked ``@pytest.mark.slow`` so they can be excluded from fast CI
runs via ``pytest -m "not slow"``.

Timing uses ``time.perf_counter`` for high-resolution wall-clock measurement.
"""

from __future__ import annotations

import time
import tracemalloc
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.indicators.batch import calculate_ema, calculate_rsi
from src.indicators.registry import IndicatorRegistry
from src.indicators.streaming import StreamingATR, StreamingEMA, StreamingRSI
from src.levels.opening_range import OpeningRangeTracker
from src.levels.vwap import SessionVWAP
from src.models import TimeFrame

if TYPE_CHECKING:
    from src.models import Bar
    from src.storage.database import BarDatabase

from .conftest import make_1min_df, make_bar

# ── helpers ──────────────────────────────────────────────────────────────────


def _make_bars(n: int, base_minute: int = 35) -> list[Bar]:
    """Generate *n* bars at valid market hours (14:35+ UTC = 9:35+ ET in Jan)."""
    bars: list[Bar] = []
    base_ts = datetime(2024, 1, 15, 14, base_minute, tzinfo=UTC)
    rng = np.random.default_rng(42)
    price = 480.0
    for i in range(n):
        noise = rng.normal(0, 0.3)
        price += noise
        o = Decimal(str(round(price, 2)))
        h = Decimal(str(round(price + abs(rng.normal(0, 0.5)), 2)))
        l_ = Decimal(str(round(price - abs(rng.normal(0, 0.5)), 2)))
        c = Decimal(str(round(price + rng.normal(0, 0.1), 2)))
        # Ensure OHLC consistency
        h = max(o, h, c)
        l_ = min(o, l_, c)
        bars.append(
            make_bar(
                timestamp=base_ts + timedelta(minutes=i),
                open_=o,
                high=h,
                low=l_,
                close=c,
                volume=int(rng.integers(50_000, 500_000)),
                vwap=Decimal(str(round(float(o + h + l_) / 3, 2))),
            )
        )
    return bars


# ── Indicator Performance ────────────────────────────────────────────────────


class TestIndicatorPerformance:
    """Streaming and batch indicator throughput benchmarks."""

    @pytest.mark.slow
    def test_streaming_ema_1000_bars_under_1s(self) -> None:
        bars = _make_bars(1000)
        ema = StreamingEMA(period=9)
        start = time.perf_counter()
        for bar in bars:
            ema.update(bar)
        elapsed = time.perf_counter() - start
        assert elapsed < 1.0, f"StreamingEMA took {elapsed:.3f}s (> 1s)"
        assert ema.ready

    @pytest.mark.slow
    def test_streaming_rsi_1000_bars_under_1s(self) -> None:
        bars = _make_bars(1000)
        rsi = StreamingRSI(period=14)
        start = time.perf_counter()
        for bar in bars:
            rsi.update(bar)
        elapsed = time.perf_counter() - start
        assert elapsed < 1.0, f"StreamingRSI took {elapsed:.3f}s (> 1s)"
        assert rsi.ready

    @pytest.mark.slow
    def test_streaming_atr_1000_bars_under_1s(self) -> None:
        bars = _make_bars(1000)
        atr = StreamingATR(period=14)
        start = time.perf_counter()
        for bar in bars:
            atr.update(bar)
        elapsed = time.perf_counter() - start
        assert elapsed < 1.0, f"StreamingATR took {elapsed:.3f}s (> 1s)"
        assert atr.ready

    @pytest.mark.slow
    def test_indicator_registry_4_indicators_1000_bars_under_2s(self) -> None:
        bars = _make_bars(1000)
        registry = IndicatorRegistry()
        registry.register("ema9", StreamingEMA(9))
        registry.register("ema20", StreamingEMA(20))
        registry.register("rsi", StreamingRSI(14))
        registry.register("atr", StreamingATR(14))

        start = time.perf_counter()
        for bar in bars:
            registry.update_all(bar)
        elapsed = time.perf_counter() - start
        assert elapsed < 2.0, f"IndicatorRegistry took {elapsed:.3f}s (> 2s)"

    @pytest.mark.slow
    def test_batch_ema_10000_bars_under_2s(self) -> None:
        df = make_1min_df(n_days=26, bars_per_day=390)
        # Trim to exactly 10000 rows
        df = df.iloc[:10_000]
        start = time.perf_counter()
        result = calculate_ema(df, period=20)
        elapsed = time.perf_counter() - start
        assert elapsed < 2.0, f"batch calculate_ema took {elapsed:.3f}s (> 2s)"
        assert len(result) == 10_000

    @pytest.mark.slow
    def test_batch_rsi_10000_bars_under_2s(self) -> None:
        df = make_1min_df(n_days=26, bars_per_day=390)
        df = df.iloc[:10_000]
        start = time.perf_counter()
        result = calculate_rsi(df, period=14)
        elapsed = time.perf_counter() - start
        assert elapsed < 2.0, f"batch calculate_rsi took {elapsed:.3f}s (> 2s)"
        assert len(result) == 10_000


# ── Level Performance ────────────────────────────────────────────────────────


class TestLevelPerformance:
    """VWAP and Opening Range tracker throughput benchmarks."""

    @pytest.mark.slow
    def test_session_vwap_390_bars_under_half_second(self) -> None:
        bars = _make_bars(390, base_minute=30)
        vwap = SessionVWAP()
        start = time.perf_counter()
        for bar in bars:
            vwap.update(bar)
        elapsed = time.perf_counter() - start
        assert elapsed < 0.5, f"SessionVWAP took {elapsed:.3f}s (> 0.5s)"
        assert vwap.is_active

    @pytest.mark.slow
    def test_opening_range_tracker_5_bars_under_100ms(self) -> None:
        # First 5 bars at 9:30-9:34 ET = 14:30-14:34 UTC (January)
        bars = []
        base_ts = datetime(2024, 1, 15, 14, 30, tzinfo=UTC)
        for i in range(5):
            bars.append(
                make_bar(
                    timestamp=base_ts + timedelta(minutes=i),
                    open_=Decimal("480.00"),
                    high=Decimal(str(481 + i * 0.1)),
                    low=Decimal(str(479 - i * 0.1)),
                    close=Decimal("480.50"),
                )
            )
        tracker = OpeningRangeTracker()
        start = time.perf_counter()
        for bar in bars:
            tracker.update(bar)
        elapsed = time.perf_counter() - start
        assert elapsed < 0.1, f"OpeningRangeTracker took {elapsed:.3f}s (> 0.1s)"


# ── Strategy Performance ─────────────────────────────────────────────────────


class TestStrategyPerformance:
    """ORBStrategy throughput with mocked external dependencies."""

    @pytest.mark.slow
    @patch("src.strategies.orb.is_high_impact_day", return_value=False)
    @patch("src.strategies.orb.is_earnings_blackout", return_value=False)
    def test_orb_strategy_390_bars_under_2s(
        self, _mock_earnings: MagicMock, _mock_econ: MagicMock
    ) -> None:
        from src.models import IndicatorSnapshot, LevelSnapshot
        from src.strategies.orb import ORBStrategy
        from src.strategies.regime import RegimeDetector

        strategy = ORBStrategy(excluded_days=[])
        regime = RegimeDetector()
        regime.update(vix=Decimal("18"), adx=Decimal("28"), trending_up=True)

        indicators = IndicatorSnapshot(
            atr=Decimal("1.50"),
            ema9=Decimal("481.00"),
            rsi=Decimal("55"),
        )
        levels = LevelSnapshot(
            orb_high=Decimal("481.50"),
            orb_low=Decimal("479.00"),
            orb_complete=True,
            vwap=Decimal("480.50"),
        )

        bars = _make_bars(390, base_minute=35)

        start = time.perf_counter()
        for bar in bars:
            strategy.evaluate(bar, indicators, levels, regime)
        elapsed = time.perf_counter() - start
        assert elapsed < 2.0, f"ORBStrategy.evaluate took {elapsed:.3f}s (> 2s)"


# ── Database Performance ─────────────────────────────────────────────────────


class TestDatabasePerformance:
    """SQLite insert and query throughput benchmarks."""

    @pytest.mark.slow
    def test_insert_1000_bars_under_2s(self, tmp_db: BarDatabase) -> None:
        bars = _make_bars(1000)
        start = time.perf_counter()
        inserted = tmp_db.insert_bars(bars)
        elapsed = time.perf_counter() - start
        assert elapsed < 2.0, f"insert_bars took {elapsed:.3f}s (> 2s)"
        assert inserted == 1000

    @pytest.mark.slow
    def test_query_1000_bars_under_1s(self, tmp_db: BarDatabase) -> None:
        bars = _make_bars(1000)
        tmp_db.insert_bars(bars)

        start = time.perf_counter()
        result = tmp_db.query_bars(
            symbol="SPY",
            timeframe=TimeFrame.ONE_MIN,
        )
        elapsed = time.perf_counter() - start
        assert elapsed < 1.0, f"query_bars took {elapsed:.3f}s (> 1s)"
        assert len(result) == 1000


# ── Monte Carlo Performance ──────────────────────────────────────────────────


class TestMonteCarloPerformance:
    """Monte Carlo permutation test throughput benchmark."""

    @pytest.mark.slow
    def test_monte_carlo_100_trades_1000_perms_under_5s(self) -> None:
        from src.backtest.monte_carlo import run_monte_carlo

        rng = np.random.default_rng(42)
        pnl = rng.normal(50, 200, size=100)

        start = time.perf_counter()
        result = run_monte_carlo(pnl, n_permutations=1000, seed=42)
        elapsed = time.perf_counter() - start
        assert elapsed < 5.0, f"run_monte_carlo took {elapsed:.3f}s (> 5s)"
        assert result.n_trades == 100
        assert result.n_permutations == 1000


# ── Memory ───────────────────────────────────────────────────────────────────


class TestMemoryUsage:
    """Verify that streaming processing does not leak memory."""

    @pytest.mark.slow
    def test_indicator_registry_5000_bars_bounded_memory(self) -> None:
        registry = IndicatorRegistry()
        registry.register("ema9", StreamingEMA(9))
        registry.register("ema20", StreamingEMA(20))
        registry.register("rsi", StreamingRSI(14))
        registry.register("atr", StreamingATR(14))

        # Warm up with 100 bars so initial allocations settle
        warmup_bars = _make_bars(100)
        for bar in warmup_bars:
            registry.update_all(bar)

        tracemalloc.start()
        snapshot_before = tracemalloc.take_snapshot()
        mem_before = sum(stat.size for stat in snapshot_before.statistics("filename"))

        bars = _make_bars(5000)
        for bar in bars:
            registry.update_all(bar)

        snapshot_after = tracemalloc.take_snapshot()
        mem_after = sum(stat.size for stat in snapshot_after.statistics("filename"))
        tracemalloc.stop()

        growth_mb = (mem_after - mem_before) / (1024 * 1024)
        # Memory growth should be bounded — not proportional to 5000 bars.
        # Allow up to 50 MB as a generous upper bound; streaming indicators
        # with talipp do store history internally but should not be unbounded.
        assert growth_mb < 50, f"Memory grew by {growth_mb:.2f} MB after processing 5000 bars"
