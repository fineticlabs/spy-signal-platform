"""Tests for batch and streaming indicator computation."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import numpy as np
import pandas as pd

from src.indicators.batch import (
    calculate_atr,
    calculate_ema,
    calculate_rsi,
    calculate_vwap,
)
from src.indicators.streaming import StreamingATR, StreamingEMA, StreamingMACD, StreamingRSI
from src.models import Bar, TimeFrame

# ── shared fixtures / helpers ────────────────────────────────────────────────

_BASE_TS = datetime(2024, 1, 15, 14, 30, tzinfo=UTC)


def _make_bar(
    close: float,
    high: float | None = None,
    low: float | None = None,
    open_: float | None = None,
    volume: int = 1_000_000,
    idx: int = 0,
) -> Bar:
    """Build a Bar with sensible high/low defaults."""
    h = high if high is not None else close + 1.0
    low_ = low if low is not None else close - 1.0
    o = open_ if open_ is not None else close
    return Bar(
        symbol="SPY",
        timeframe=TimeFrame.ONE_MIN,
        timestamp=_BASE_TS + timedelta(minutes=idx),
        open=Decimal(str(o)),
        high=Decimal(str(h)),
        low=Decimal(str(low_)),
        close=Decimal(str(c := close)),
        volume=volume,
        vwap=Decimal(str(c)),
    )


def _bars_to_df(bars: list[Bar]) -> pd.DataFrame:
    """Convert a list of Bars to a DataFrame for batch functions."""
    return pd.DataFrame(
        {
            "open": [float(b.open) for b in bars],
            "high": [float(b.high) for b in bars],
            "low": [float(b.low) for b in bars],
            "close": [float(b.close) for b in bars],
            "volume": [float(b.volume) for b in bars],
        }
    )


def _linear_bars(n: int = 30, start: float = 100.0) -> list[Bar]:
    """Bars with close prices start, start+1, …, start+n-1."""
    return [_make_bar(close=start + i, idx=i) for i in range(n)]


# ── batch.py tests ───────────────────────────────────────────────────────────


class TestBatchEMA:
    def test_ema9_last_value_linear_sequence(self) -> None:
        """For a strictly linear sequence EMA(n) converges to the actual close."""
        bars = _linear_bars(20)
        df = _bars_to_df(bars)
        ema = calculate_ema(df, period=9)

        # For a perfectly linear sequence talib EMA(9) equals close-4 at position 20
        # Verified: last value == 115.0
        assert not np.isnan(ema.iloc[-1])
        assert abs(ema.iloc[-1] - 115.0) < 1e-6

    def test_ema9_leading_nans(self) -> None:
        """EMA(9) must have exactly 8 leading NaN values."""
        bars = _linear_bars(20)
        df = _bars_to_df(bars)
        ema = calculate_ema(df, period=9)

        nan_count = ema.isna().sum()
        assert nan_count == 8  # period - 1

    def test_ema_series_name(self) -> None:
        df = _bars_to_df(_linear_bars(10))
        assert calculate_ema(df, period=9).name == "ema_9"
        assert calculate_ema(df, period=20).name == "ema_20"

    def test_ema_index_aligned(self) -> None:
        bars = _linear_bars(15)
        df = _bars_to_df(bars)
        ema = calculate_ema(df, period=9)
        assert list(ema.index) == list(df.index)


class TestBatchRSI:
    def test_rsi_range_0_to_100(self) -> None:
        """RSI must always be in [0, 100]."""
        # Mix of ups and downs so RSI is meaningful
        closes = [100, 102, 101, 103, 100, 99, 101, 104, 102, 103, 105, 104, 106, 108, 107, 109]
        bars = [_make_bar(close=c, idx=i) for i, c in enumerate(closes)]
        df = _bars_to_df(bars)
        rsi = calculate_rsi(df, period=14)

        valid = rsi.dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_rsi_strictly_increasing_is_100(self) -> None:
        """Strictly rising prices → RSI == 100 (no losses)."""
        bars = _linear_bars(20)
        df = _bars_to_df(bars)
        rsi = calculate_rsi(df, period=14)

        assert abs(rsi.iloc[-1] - 100.0) < 1e-6

    def test_rsi_leading_nans(self) -> None:
        bars = _linear_bars(20)
        df = _bars_to_df(bars)
        rsi = calculate_rsi(df, period=14)
        assert rsi.isna().sum() == 14  # period bars consumed before first output


class TestBatchATR:
    def test_atr_always_positive(self) -> None:
        """ATR must be > 0 for any bar with non-zero range."""
        closes = [100, 102, 101, 103, 100, 99, 101, 104, 102, 103, 105, 104, 106, 108, 107, 109]
        bars = [_make_bar(close=c, idx=i) for i, c in enumerate(closes)]
        df = _bars_to_df(bars)
        atr = calculate_atr(df, period=14)

        valid = atr.dropna()
        assert (valid > 0).all()

    def test_atr_constant_range(self) -> None:
        """Bars with high=close+1, low=close-1 → ATR converges to 2.0."""
        bars = _linear_bars(30)  # default high=close+1, low=close-1
        df = _bars_to_df(bars)
        atr = calculate_atr(df, period=14)

        # After 30 bars, ATR should have converged to ≈ 2.0
        assert abs(atr.iloc[-1] - 2.0) < 1e-6


class TestBatchVWAP:
    def test_vwap_matches_manual_formula(self) -> None:
        """VWAP = cumsum(typical_price * volume) / cumsum(volume)."""
        data = [
            (100.0, 101.0, 99.0, 1_000_000),
            (101.0, 102.0, 100.0, 1_200_000),
            (102.0, 103.0, 101.0, 800_000),
            (103.0, 104.0, 102.0, 1_100_000),
            (104.0, 105.0, 103.0, 900_000),
        ]
        bars = [
            _make_bar(close=c, high=h, low=lo, volume=v, idx=i)
            for i, (c, h, lo, v) in enumerate(data)
        ]
        df = _bars_to_df(bars)
        vwap_series = calculate_vwap(df)

        # Manual cumulative VWAP after all 5 bars
        tp = [(h + lo + c) / 3 for c, h, lo, _ in data]
        vols = [v for _, _, _, v in data]
        expected_final = sum(t * v for t, v in zip(tp, vols, strict=False)) / sum(vols)

        assert abs(vwap_series.iloc[-1] - expected_final) < 1e-6

    def test_vwap_first_bar_equals_typical_price(self) -> None:
        """VWAP of a single bar equals its typical price."""
        bar = _make_bar(close=102.0, high=103.0, low=101.0, volume=1_000_000)
        df = _bars_to_df([bar])
        vwap = calculate_vwap(df)

        expected = (103.0 + 101.0 + 102.0) / 3
        assert abs(vwap.iloc[0] - expected) < 1e-9


# ── streaming indicator tests ─────────────────────────────────────────────────


class TestStreamingEMA:
    def test_not_ready_until_period_bars(self) -> None:
        ema = StreamingEMA(period=9)
        for i in range(8):
            ema.update(_make_bar(close=100.0 + i, idx=i))
            assert not ema.ready

        ema.update(_make_bar(close=108.0, idx=8))
        assert ema.ready

    def test_value_is_decimal(self) -> None:
        ema = StreamingEMA(period=9)
        bars = _linear_bars(15)
        for b in bars:
            ema.update(b)
        assert isinstance(ema.value, Decimal)

    def test_streaming_vs_batch_agreement(self) -> None:
        """StreamingEMA and calculate_ema must agree within 1e-8."""
        bars = _linear_bars(30)
        df = _bars_to_df(bars)
        batch_ema = calculate_ema(df, period=9)

        stream_ema = StreamingEMA(period=9)
        for b in bars:
            stream_ema.update(b)

        assert stream_ema.value is not None
        assert abs(float(stream_ema.value) - batch_ema.iloc[-1]) < 1e-8


class TestStreamingRSI:
    def test_rsi_value_in_range(self) -> None:
        closes = [100, 102, 101, 103, 100, 99, 101, 104, 102, 103, 105, 104, 106, 108, 107, 109]
        rsi = StreamingRSI(period=14)
        for i, c in enumerate(closes):
            rsi.update(_make_bar(close=c, idx=i))

        if rsi.ready:
            assert Decimal("0") <= rsi.value <= Decimal("100")  # type: ignore[operator]

    def test_rsi_not_ready_before_period_plus_one(self) -> None:
        rsi = StreamingRSI(period=14)
        for i in range(14):
            rsi.update(_make_bar(close=100.0 + i, idx=i))
            assert not rsi.ready

    def test_streaming_rsi_vs_batch(self) -> None:
        """StreamingRSI and calculate_rsi agree within 1e-6."""
        closes = [100 + (i % 5) - 2 for i in range(30)]
        bars = [_make_bar(close=float(c), idx=i) for i, c in enumerate(closes)]
        df = _bars_to_df(bars)
        batch_rsi = calculate_rsi(df, period=14).iloc[-1]

        stream_rsi = StreamingRSI(period=14)
        for b in bars:
            stream_rsi.update(b)

        assert stream_rsi.ready
        assert abs(float(stream_rsi.value) - batch_rsi) < 1e-6  # type: ignore[arg-type]


class TestStreamingATR:
    def test_atr_always_positive_when_ready(self) -> None:
        atr = StreamingATR(period=14)
        bars = _linear_bars(20)
        for b in bars:
            atr.update(b)
        assert atr.ready
        assert atr.value > Decimal("0")  # type: ignore[operator]

    def test_streaming_atr_vs_batch(self) -> None:
        """StreamingATR and calculate_atr agree within 1e-6."""
        bars = _linear_bars(30)
        df = _bars_to_df(bars)
        batch_atr = calculate_atr(df, period=14).iloc[-1]

        stream_atr = StreamingATR(period=14)
        for b in bars:
            stream_atr.update(b)

        assert stream_atr.ready
        assert abs(float(stream_atr.value) - batch_atr) < 1e-6  # type: ignore[arg-type]


class TestStreamingMACD:
    def test_not_ready_until_enough_bars(self) -> None:
        macd = StreamingMACD(fast=12, slow=26, signal=9)
        bars = _linear_bars(33)  # 26 + 9 - 1 = 34 bars needed
        for i, b in enumerate(bars):
            macd.update(b)
            # Should not be ready with fewer than 34 bars
            if i < 33:
                assert not macd.ready

    def test_value_has_macd_signal_histogram(self) -> None:
        macd = StreamingMACD(fast=12, slow=26, signal=9)
        bars = _linear_bars(40)
        for b in bars:
            macd.update(b)

        assert macd.ready
        v = macd.value
        assert v is not None
        assert isinstance(v.macd, Decimal)
        assert isinstance(v.signal, Decimal)
        assert isinstance(v.histogram, Decimal)

    def test_histogram_equals_macd_minus_signal(self) -> None:
        macd = StreamingMACD(fast=12, slow=26, signal=9)
        bars = _linear_bars(40)
        for b in bars:
            macd.update(b)

        v = macd.value
        assert v is not None
        assert abs(float(v.histogram) - float(v.macd - v.signal)) < 1e-8
