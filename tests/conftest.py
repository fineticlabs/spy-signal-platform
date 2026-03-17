"""Shared fixtures for the spy-signal-platform test suite."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from src.models import Bar, Direction, Regime, Signal, TimeFrame

# ── Date/time helpers ────────────────────────────────────────────────────────


def make_utc_ts(year: int, month: int, day: int, hour: int, minute: int) -> datetime:
    """Shorthand for a UTC-aware datetime."""
    return datetime(year, month, day, hour, minute, tzinfo=UTC)


# ET market hours in UTC (EST = +5, EDT = +4)
# Use January dates (EST) for consistency
_BASE_DATE = datetime(2024, 1, 15, tzinfo=UTC)


# ── Bar factory ──────────────────────────────────────────────────────────────


def make_bar(
    symbol: str = "SPY",
    timestamp: datetime | None = None,
    open_: Decimal | None = None,
    high: Decimal | None = None,
    low: Decimal | None = None,
    close: Decimal | None = None,
    volume: int = 100_000,
    vwap: Decimal | None = None,
    timeframe: TimeFrame = TimeFrame.ONE_MIN,
) -> Bar:
    """Create a Bar with sensible defaults."""
    ts = timestamp or make_utc_ts(2024, 1, 15, 14, 35)
    o = open_ or Decimal("480.00")
    h = high or Decimal("481.50")
    l_ = low or Decimal("479.80")
    c = close or Decimal("481.00")
    v = vwap or Decimal("480.55")
    return Bar(
        symbol=symbol,
        timeframe=timeframe,
        timestamp=ts,
        open=o,
        high=h,
        low=l_,
        close=c,
        volume=volume,
        vwap=v,
    )


# ── OHLCV DataFrame generators ──────────────────────────────────────────────


def make_1min_df(
    n_days: int = 1,
    bars_per_day: int = 390,
    base_price: float = 480.0,
    start_date: str = "2024-01-15",
    symbol: str = "SPY",
) -> pd.DataFrame:
    """Generate synthetic 1-min OHLCV DataFrame with DatetimeIndex.

    Prices follow a slight uptrend with random noise.
    Index is UTC timestamps for market hours (14:30-21:00 UTC = 9:30-16:00 ET in winter).
    """
    rng = np.random.default_rng(42)
    all_rows = []
    start = pd.Timestamp(start_date, tz="America/New_York").tz_convert("UTC")

    for day_offset in range(n_days):
        day_start = start + pd.Timedelta(days=day_offset)
        # Skip weekends
        while day_start.dayofweek >= 5:
            day_start += pd.Timedelta(days=1)

        market_open = day_start.replace(hour=14, minute=30, second=0, microsecond=0)

        for bar_idx in range(bars_per_day):
            ts = market_open + pd.Timedelta(minutes=bar_idx)
            drift = 0.001 * bar_idx / bars_per_day
            noise = rng.normal(0, 0.3)
            mid = base_price * (1 + drift) + noise
            spread = rng.uniform(0.1, 0.5)
            o = mid + rng.normal(0, 0.1)
            c = mid + rng.normal(0, 0.1)
            h = max(o, c) + spread
            l_ = min(o, c) - spread
            vol = int(rng.integers(50_000, 500_000))

            all_rows.append(
                {
                    "open": h if o > h else o,
                    "high": h,
                    "low": l_,
                    "close": c,
                    "volume": vol,
                    "timestamp": ts,
                }
            )

    df = pd.DataFrame(all_rows)
    df = df.set_index("timestamp")
    df.index = pd.DatetimeIndex(df.index, tz="UTC")
    # Ensure OHLC consistency
    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)
    return df


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_bar() -> Bar:
    """A single valid Bar for SPY."""
    return make_bar()


@pytest.fixture
def sample_df_1day() -> pd.DataFrame:
    """One day of synthetic 1-min bars."""
    return make_1min_df(n_days=1)


@pytest.fixture
def sample_df_5days() -> pd.DataFrame:
    """Five days of synthetic 1-min bars."""
    return make_1min_df(n_days=5)


@pytest.fixture
def sample_signal() -> Signal:
    """A valid long ORB signal."""
    return Signal(
        symbol="SPY",
        direction=Direction.LONG,
        strategy_name="ORB",
        entry_price=Decimal("481.50"),
        stop_price=Decimal("479.00"),
        target_price=Decimal("486.50"),
        risk_reward_ratio=Decimal("2.0"),
        confidence_score=3,
        reason="ORB long breakout with volume confirmation",
        timeframe=TimeFrame.ONE_MIN,
        regime=Regime.TRENDING_UP,
        vix=Decimal("18.5"),
        adx=Decimal("28.0"),
        indicators_snapshot=None,
        levels_snapshot=None,
        timestamp=make_utc_ts(2024, 1, 15, 14, 36),
        tags=["VOLUME_CONFIRM"],
    )


@pytest.fixture
def tmp_db(tmp_path):
    """Yield a connected BarDatabase in a temp directory, close on teardown."""
    from src.storage.database import BarDatabase

    db = BarDatabase(db_path=str(tmp_path / "test.db"))
    db.connect()
    yield db
    db.close()
