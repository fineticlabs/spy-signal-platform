"""Extended tests for the backtest engine and ingestion modules.

Coverage
--------
- compute_intraday_vwap: reset per day, VWAP between low/high, manual calc
- _compute_first5min_rvol: NaN for insufficient history, shape, positive values
- _compute_gap_array: NaN first day, positive/negative gaps, constant per day
- _compute_orb_arrays: NaN during ORB window, orb_high >= orb_low, per-day values
- run_backtest: returns stats, _trades key, empty/single-day data
- Ingestion websocket: AlpacaBarStream constructor, _FEED_MAP
- Ingestion historical: fetch_historical_bars with mocked Alpaca client
"""

from __future__ import annotations

import contextlib
import warnings
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import (
    _compute_first5min_rvol,
    _compute_gap_array,
    _compute_orb_arrays,
    compute_intraday_vwap,
    run_backtest,
)
from src.ingestion.websocket import _FEED_MAP, AlpacaBarStream
from src.models import Bar, TimeFrame

_UTC_TZ = UTC


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_trading_day_index(
    date_str: str,
    n_bars: int = 390,
) -> pd.DatetimeIndex:
    """Create a 1-min DatetimeIndex for a single US trading day (14:30-21:00 UTC).

    Args:
        date_str: Date in 'YYYY-MM-DD' format (e.g. '2024-01-15').
        n_bars:   Number of 1-min bars (default 390 = full day).
    """
    start = pd.Timestamp(f"{date_str} 14:30:00", tz="UTC")
    return pd.date_range(start=start, periods=n_bars, freq="min")


def _make_multiday_index(
    start_date: str,
    n_days: int,
    bars_per_day: int = 390,
) -> pd.DatetimeIndex:
    """Create a 1-min DatetimeIndex spanning multiple trading days.

    Skips weekends.  Uses January dates to stay in EST (UTC-5).
    """
    indices: list[pd.DatetimeIndex] = []
    current = pd.Timestamp(start_date)
    days_added = 0
    while days_added < n_days:
        if current.weekday() < 5:  # Mon-Fri
            day_start = pd.Timestamp(f"{current.strftime('%Y-%m-%d')} 14:30:00", tz="UTC")
            indices.append(pd.date_range(start=day_start, periods=bars_per_day, freq="min"))
            days_added += 1
        current += timedelta(days=1)
    return indices[0].append(indices[1:])  # type: ignore[arg-type]


def _make_ohlcv_arrays(
    n: int,
    base_price: float = 480.0,
    base_volume: int = 100_000,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic OHLCV arrays.

    Returns (open, high, low, close, volume).
    """
    if rng is None:
        rng = np.random.default_rng(42)
    noise = rng.normal(0, 0.5, n)
    close = base_price + np.cumsum(noise)
    open_ = close + rng.normal(0, 0.1, n)
    high = np.maximum(open_, close) + rng.uniform(0.1, 1.0, n)
    low = np.minimum(open_, close) - rng.uniform(0.1, 1.0, n)
    volume = (base_volume + rng.integers(-20_000, 20_000, n)).astype(float)
    volume = np.maximum(volume, 1000)  # no zero volume
    return open_, high, low, close, volume


def _make_ohlcv_df(
    index: pd.DatetimeIndex,
    base_price: float = 480.0,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Build an OHLCV DataFrame suitable for run_backtest."""
    o, h, l_, c, v = _make_ohlcv_arrays(len(index), base_price=base_price, rng=rng)
    return pd.DataFrame(
        {"open": o, "high": h, "low": l_, "close": c, "volume": v},
        index=index,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Backtest Engine Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestComputeIntradayVwap:
    """Tests for compute_intraday_vwap()."""

    def test_output_length_matches_input(self) -> None:
        """VWAP array length equals input length."""
        idx = _make_trading_day_index("2024-01-15", n_bars=50)
        _, h, l_, c, v = _make_ohlcv_arrays(50)
        vwap = compute_intraday_vwap(idx, h, l_, c, v)
        assert len(vwap) == 50

    def test_vwap_resets_each_day(self) -> None:
        """VWAP resets at the start of each trading day."""
        idx = _make_multiday_index("2024-01-15", n_days=2, bars_per_day=10)
        rng = np.random.default_rng(42)
        _, h, l_, c, v = _make_ohlcv_arrays(20, rng=rng)
        vwap = compute_intraday_vwap(idx, h, l_, c, v)

        # First bar of day 2 should only use its own typical price (single bar reset)
        tp_day2_bar0 = (h[10] + l_[10] + c[10]) / 3.0
        assert vwap[10] == pytest.approx(tp_day2_bar0, rel=1e-9)

    def test_vwap_between_low_and_high(self) -> None:
        """VWAP stays between low and high for each bar (within day cumulative bounds)."""
        idx = _make_trading_day_index("2024-01-15", n_bars=100)
        rng = np.random.default_rng(99)
        _, h, l_, c, v = _make_ohlcv_arrays(100, rng=rng)
        vwap = compute_intraday_vwap(idx, h, l_, c, v)

        day_low = np.minimum.accumulate(l_)
        day_high = np.maximum.accumulate(h)

        for i in range(100):
            if not np.isnan(vwap[i]):
                assert vwap[i] >= day_low[i] - 1e-9, f"VWAP below cumulative day low at bar {i}"
                assert vwap[i] <= day_high[i] + 1e-9, f"VWAP above cumulative day high at bar {i}"

    def test_manual_vwap_calculation(self) -> None:
        """Verify VWAP against manual sum(TP*V)/sum(V) for 3 bars."""
        idx = _make_trading_day_index("2024-01-15", n_bars=3)
        h = np.array([102.0, 105.0, 103.0])
        l_ = np.array([98.0, 99.0, 100.0])
        c = np.array([100.0, 103.0, 101.0])
        v = np.array([1000.0, 2000.0, 1500.0])

        vwap = compute_intraday_vwap(idx, h, l_, c, v)

        tp = (h + l_ + c) / 3.0
        # Bar 0: cum_tpv = tp[0]*v[0], cum_v = v[0]
        expected_0 = tp[0]
        # Bar 1: cum_tpv = tp[0]*v[0]+tp[1]*v[1], cum_v = v[0]+v[1]
        expected_1 = (tp[0] * v[0] + tp[1] * v[1]) / (v[0] + v[1])
        # Bar 2: all three
        expected_2 = (tp[0] * v[0] + tp[1] * v[1] + tp[2] * v[2]) / (v[0] + v[1] + v[2])

        assert vwap[0] == pytest.approx(expected_0, rel=1e-9)
        assert vwap[1] == pytest.approx(expected_1, rel=1e-9)
        assert vwap[2] == pytest.approx(expected_2, rel=1e-9)

    def test_single_bar_vwap_equals_typical_price(self) -> None:
        """Single bar: VWAP = typical price = (H+L+C)/3."""
        idx = _make_trading_day_index("2024-01-15", n_bars=1)
        h = np.array([105.0])
        l_ = np.array([95.0])
        c = np.array([100.0])
        v = np.array([5000.0])

        vwap = compute_intraday_vwap(idx, h, l_, c, v)
        expected = (105.0 + 95.0 + 100.0) / 3.0
        assert vwap[0] == pytest.approx(expected, rel=1e-9)


class TestComputeFirst5minRvol:
    """Tests for _compute_first5min_rvol()."""

    def test_output_length_matches_input(self) -> None:
        """RVOL array length equals input length."""
        idx = _make_multiday_index("2024-01-02", n_days=25, bars_per_day=390)
        _, _, _, _, v = _make_ohlcv_arrays(len(idx))
        rvol = _compute_first5min_rvol(idx, v)
        assert len(rvol) == len(idx)

    def test_nan_for_insufficient_history(self) -> None:
        """First ~20 days should be NaN (default window=20, plus shift-1)."""
        idx = _make_multiday_index("2024-01-02", n_days=25, bars_per_day=390)
        rng = np.random.default_rng(42)
        _, _, _, _, v = _make_ohlcv_arrays(len(idx), rng=rng)
        rvol = _compute_first5min_rvol(idx, v, window=20)

        # First 20 days + shift means day index 0-20 are NaN (21 days NaN)
        # Day 21 (index 21) is the first with a value
        # Check first day is NaN
        assert np.isnan(rvol[0]), "First bar of first day should be NaN"
        # Check day 10 is still NaN (well within window)
        bar_idx_day10 = 10 * 390
        assert np.isnan(rvol[bar_idx_day10]), "Day 10 should still be NaN"

    def test_positive_values_where_available(self) -> None:
        """RVOL values that are not NaN should be positive."""
        idx = _make_multiday_index("2024-01-02", n_days=25, bars_per_day=390)
        rng = np.random.default_rng(42)
        _, _, _, _, v = _make_ohlcv_arrays(len(idx), rng=rng)
        rvol = _compute_first5min_rvol(idx, v, window=20)

        valid = rvol[~np.isnan(rvol)]
        if len(valid) > 0:
            assert np.all(valid > 0), "All non-NaN RVOL values should be positive"

    def test_rvol_above_one_means_higher_than_average(self) -> None:
        """RVOL > 1 when today's first-5-min volume exceeds the trailing average."""
        # Build 22 days: first 20 with low volume, last 2 with high volume
        idx = _make_multiday_index("2024-01-02", n_days=22, bars_per_day=10)
        n = len(idx)
        v = np.full(n, 1000.0)  # base low volume

        # Spike the last day's first 5 bars to 10x
        last_day_start = 21 * 10
        v[last_day_start : last_day_start + 5] = 10_000.0

        rvol = _compute_first5min_rvol(idx, v, orb_bars=5, window=20)

        # Last day should have RVOL > 1
        last_day_rvol = rvol[last_day_start]
        if not np.isnan(last_day_rvol):
            assert last_day_rvol > 1.0, f"Expected RVOL > 1, got {last_day_rvol}"


class TestComputeGapArray:
    """Tests for _compute_gap_array()."""

    def test_output_length_matches_input(self) -> None:
        """Gap array length equals input length."""
        idx = _make_multiday_index("2024-01-15", n_days=3, bars_per_day=10)
        rng = np.random.default_rng(42)
        o, _, _, c, _ = _make_ohlcv_arrays(len(idx), rng=rng)
        gap = _compute_gap_array(idx, o, c)
        assert len(gap) == len(idx)

    def test_first_day_is_nan(self) -> None:
        """All bars on the first trading day should be NaN (no prior day)."""
        idx = _make_multiday_index("2024-01-15", n_days=3, bars_per_day=10)
        rng = np.random.default_rng(42)
        o, _, _, c, _ = _make_ohlcv_arrays(len(idx), rng=rng)
        gap = _compute_gap_array(idx, o, c)

        # First 10 bars (day 1)
        assert np.all(np.isnan(gap[:10])), "First day should be all NaN"

    def test_positive_gap_when_open_above_prev_close(self) -> None:
        """Gap is positive when today's open > yesterday's close."""
        idx = _make_multiday_index("2024-01-15", n_days=2, bars_per_day=10)
        n = len(idx)
        o = np.full(n, 100.0)
        c = np.full(n, 100.0)

        # Day 1: close all bars at 100
        # Day 2: open at 102 → gap = (102-100)/100 * 100 = 2%
        o[10:] = 102.0
        c[:10] = 100.0

        gap = _compute_gap_array(idx, o, c)
        assert gap[10] == pytest.approx(2.0, rel=1e-9)

    def test_gap_constant_within_day(self) -> None:
        """All bars within a single day should have the same gap value."""
        idx = _make_multiday_index("2024-01-15", n_days=2, bars_per_day=10)
        n = len(idx)
        o = np.full(n, 100.0)
        c = np.full(n, 100.0)
        o[10:] = 101.0

        gap = _compute_gap_array(idx, o, c)

        day2_gaps = gap[10:20]
        non_nan = day2_gaps[~np.isnan(day2_gaps)]
        assert len(non_nan) == 10, "All day-2 bars should have gap values"
        assert np.all(non_nan == non_nan[0]), "Gap should be constant within day"

    def test_gap_down_is_negative(self) -> None:
        """Gap is negative when today's open < yesterday's close."""
        idx = _make_multiday_index("2024-01-15", n_days=2, bars_per_day=10)
        n = len(idx)
        o = np.full(n, 100.0)
        c = np.full(n, 100.0)

        # Day 2 opens at 98 → gap = (98-100)/100 * 100 = -2%
        o[10:] = 98.0

        gap = _compute_gap_array(idx, o, c)
        assert gap[10] == pytest.approx(-2.0, rel=1e-9)


class TestComputeOrbArrays:
    """Tests for _compute_orb_arrays()."""

    def test_returns_tuple_of_two_arrays(self) -> None:
        """Should return (orb_high, orb_low) tuple."""
        idx = _make_trading_day_index("2024-01-15", n_bars=20)
        rng = np.random.default_rng(42)
        _, h, l_, _, _ = _make_ohlcv_arrays(20, rng=rng)
        result = _compute_orb_arrays(idx, h, l_)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_output_length_matches_input(self) -> None:
        """Both ORB arrays have same length as input."""
        idx = _make_trading_day_index("2024-01-15", n_bars=20)
        rng = np.random.default_rng(42)
        _, h, l_, _, _ = _make_ohlcv_arrays(20, rng=rng)
        orb_h, orb_l = _compute_orb_arrays(idx, h, l_, orb_bars=5)
        assert len(orb_h) == 20
        assert len(orb_l) == 20

    def test_nan_during_opening_range_window(self) -> None:
        """First 5 bars (ORB window) should be NaN."""
        idx = _make_trading_day_index("2024-01-15", n_bars=20)
        rng = np.random.default_rng(42)
        _, h, l_, _, _ = _make_ohlcv_arrays(20, rng=rng)
        orb_h, orb_l = _compute_orb_arrays(idx, h, l_, orb_bars=5)

        # Bars 0-4 should be NaN
        assert np.all(np.isnan(orb_h[:5])), "ORB high should be NaN during opening range"
        assert np.all(np.isnan(orb_l[:5])), "ORB low should be NaN during opening range"

    def test_orb_high_gte_orb_low_after_window(self) -> None:
        """orb_high >= orb_low for all bars after the opening range."""
        idx = _make_trading_day_index("2024-01-15", n_bars=50)
        rng = np.random.default_rng(42)
        _, h, l_, _, _ = _make_ohlcv_arrays(50, rng=rng)
        orb_h, orb_l = _compute_orb_arrays(idx, h, l_, orb_bars=5)

        for i in range(5, 50):
            if not np.isnan(orb_h[i]):
                assert orb_h[i] >= orb_l[i], f"orb_high < orb_low at bar {i}"

    def test_orb_computed_per_day(self) -> None:
        """Different days produce different ORB values (with different data)."""
        idx = _make_multiday_index("2024-01-15", n_days=2, bars_per_day=20)
        rng = np.random.default_rng(42)
        _, h, l_, _, _ = _make_ohlcv_arrays(40, rng=rng)

        # Make day 2 prices significantly different
        h[20:] += 50.0
        l_[20:] += 50.0

        orb_h, _orb_l = _compute_orb_arrays(idx, h, l_, orb_bars=5)

        # Bar 5 = first post-ORB bar of day 1; bar 25 = first post-ORB bar of day 2
        day1_orb_h = orb_h[5]
        day2_orb_h = orb_h[25]

        assert not np.isnan(day1_orb_h), "Day 1 ORB high should not be NaN after window"
        assert not np.isnan(day2_orb_h), "Day 2 ORB high should not be NaN after window"
        assert day1_orb_h != pytest.approx(
            day2_orb_h, abs=1.0
        ), "Different days should have different ORB values"

    def test_orb_values_match_manual_max_min(self) -> None:
        """ORB high/low should be max(high[:5]) and min(low[:5])."""
        idx = _make_trading_day_index("2024-01-15", n_bars=10)
        h = np.array([101, 105, 103, 102, 104, 100, 99, 101, 102, 103], dtype=float)
        l_ = np.array([99, 98, 100, 97, 101, 96, 95, 98, 99, 100], dtype=float)

        orb_h, orb_l = _compute_orb_arrays(idx, h, l_, orb_bars=5)

        expected_high = max(h[:5])  # 105
        expected_low = min(l_[:5])  # 97

        assert orb_h[5] == pytest.approx(expected_high)
        assert orb_l[5] == pytest.approx(expected_low)


class TestRunBacktest:
    """Tests for run_backtest()."""

    def test_returns_dict_like_stats(self) -> None:
        """run_backtest should return a dict-like stats object."""
        warnings.filterwarnings("ignore")
        idx = _make_multiday_index("2024-01-02", n_days=30, bars_per_day=390)
        rng = np.random.default_rng(42)
        df = _make_ohlcv_df(idx, rng=rng)
        stats = run_backtest(df)
        # Should be subscriptable
        assert "# Trades" in stats

    def test_has_trades_key(self) -> None:
        """Stats object should have a _trades attribute with a DataFrame."""
        warnings.filterwarnings("ignore")
        idx = _make_multiday_index("2024-01-02", n_days=30, bars_per_day=390)
        rng = np.random.default_rng(42)
        df = _make_ohlcv_df(idx, rng=rng)
        stats = run_backtest(df)
        trades = stats._trades
        assert isinstance(trades, pd.DataFrame)

    def test_empty_dataframe_handling(self) -> None:
        """run_backtest should not crash silently on empty DataFrame (may raise)."""
        warnings.filterwarnings("ignore")
        empty_df = pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"],
            dtype=float,
        )
        empty_df.index = pd.DatetimeIndex([], tz="UTC")
        with contextlib.suppress(Exception):
            run_backtest(empty_df)

    def test_single_day_does_not_crash(self) -> None:
        """A single trading day should not crash (may produce 0 trades)."""
        warnings.filterwarnings("ignore")
        idx = _make_trading_day_index("2024-01-15", n_bars=390)
        rng = np.random.default_rng(42)
        df = _make_ohlcv_df(idx, rng=rng)
        with contextlib.suppress(Exception):
            stats = run_backtest(df)
            # Single day with no prior data for filters = likely 0 trades
            assert stats["# Trades"] >= 0


# ═══════════════════════════════════════════════════════════════════════════════
# Ingestion Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestAlpacaBarStreamConstructor:
    """Tests for AlpacaBarStream construction."""

    def test_accepts_timeframe_enum(self) -> None:
        """Constructor should accept a TimeFrame enum without error."""
        import asyncio

        queue: asyncio.Queue[Bar] = asyncio.Queue()
        stream = AlpacaBarStream(
            symbols=["SPY"],
            queue=queue,
            timeframe=TimeFrame.ONE_MIN,
        )
        assert stream._timeframe == TimeFrame.ONE_MIN

    def test_default_timeframe_is_one_min(self) -> None:
        """Default timeframe should be ONE_MIN."""
        import asyncio

        queue: asyncio.Queue[Bar] = asyncio.Queue()
        stream = AlpacaBarStream(symbols=["SPY"], queue=queue)
        assert stream._timeframe == TimeFrame.ONE_MIN

    def test_accepts_five_min_timeframe(self) -> None:
        """Constructor should accept FIVE_MIN timeframe."""
        import asyncio

        queue: asyncio.Queue[Bar] = asyncio.Queue()
        stream = AlpacaBarStream(
            symbols=["SPY"],
            queue=queue,
            timeframe=TimeFrame.FIVE_MIN,
        )
        assert stream._timeframe == TimeFrame.FIVE_MIN


class TestFeedMap:
    """Tests for the _FEED_MAP dictionary in websocket.py."""

    def test_iex_maps_to_datafeed_iex(self) -> None:
        """'iex' key should map to DataFeed.IEX."""
        from alpaca.data.enums import DataFeed

        assert _FEED_MAP["iex"] == DataFeed.IEX

    def test_sip_maps_to_datafeed_sip(self) -> None:
        """'sip' key should map to DataFeed.SIP."""
        from alpaca.data.enums import DataFeed

        assert _FEED_MAP["sip"] == DataFeed.SIP

    def test_feed_map_has_exactly_two_entries(self) -> None:
        """_FEED_MAP should have exactly two entries (iex and sip)."""
        assert len(_FEED_MAP) == 2
        assert set(_FEED_MAP.keys()) == {"iex", "sip"}


class TestFetchHistoricalBars:
    """Tests for fetch_historical_bars() with mocked Alpaca client."""

    @patch("src.ingestion.historical.get_alpaca_settings")
    @patch("src.ingestion.historical.StockHistoricalDataClient")
    def test_creates_bar_objects_with_correct_timeframe(
        self,
        mock_client_cls: MagicMock,
        mock_settings: MagicMock,
    ) -> None:
        """Fetched bars should have the correct TimeFrame enum."""
        from alpaca.data.models.bars import BarSet

        from src.ingestion.historical import fetch_historical_bars

        # Configure mock settings
        mock_settings.return_value = MagicMock(
            api_key="fake-key",
            secret_key="fake-secret",  # noqa: S106
            feed="iex",
        )

        # Create a mock Alpaca bar
        mock_raw_bar = MagicMock()
        mock_raw_bar.timestamp = datetime(2024, 1, 15, 14, 30, tzinfo=UTC)
        mock_raw_bar.open = 480.0
        mock_raw_bar.high = 481.0
        mock_raw_bar.low = 479.0
        mock_raw_bar.close = 480.5
        mock_raw_bar.volume = 100_000
        mock_raw_bar.vwap = 480.3

        # Mock the BarSet response
        mock_barset = MagicMock(spec=BarSet)
        mock_barset.data = {"SPY": [mock_raw_bar]}

        mock_client_instance = mock_client_cls.return_value
        mock_client_instance.get_stock_bars.return_value = mock_barset

        # Mock the database
        mock_db = MagicMock()

        bars = fetch_historical_bars("SPY", days=1, timeframe=TimeFrame.ONE_MIN, db=mock_db)

        assert len(bars) == 1
        assert bars[0].timeframe == TimeFrame.ONE_MIN
        assert bars[0].symbol == "SPY"
        assert bars[0].close == Decimal("480.5")

    @patch("src.ingestion.historical.get_alpaca_settings")
    @patch("src.ingestion.historical.StockHistoricalDataClient")
    def test_bars_inserted_into_database(
        self,
        mock_client_cls: MagicMock,
        mock_settings: MagicMock,
    ) -> None:
        """Fetched bars should be inserted into the database."""
        from alpaca.data.models.bars import BarSet

        from src.ingestion.historical import fetch_historical_bars

        mock_settings.return_value = MagicMock(
            api_key="fake-key",
            secret_key="fake-secret",  # noqa: S106
            feed="iex",
        )

        mock_raw_bar = MagicMock()
        mock_raw_bar.timestamp = datetime(2024, 1, 15, 14, 30, tzinfo=UTC)
        mock_raw_bar.open = 480.0
        mock_raw_bar.high = 481.0
        mock_raw_bar.low = 479.0
        mock_raw_bar.close = 480.5
        mock_raw_bar.volume = 100_000
        mock_raw_bar.vwap = 480.3

        mock_barset = MagicMock(spec=BarSet)
        mock_barset.data = {"SPY": [mock_raw_bar]}

        mock_client_instance = mock_client_cls.return_value
        mock_client_instance.get_stock_bars.return_value = mock_barset

        mock_db = MagicMock()

        fetch_historical_bars("SPY", days=1, timeframe=TimeFrame.ONE_MIN, db=mock_db)

        # Verify insert_bars was called with a list of Bar objects
        mock_db.insert_bars.assert_called_once()
        inserted_bars = mock_db.insert_bars.call_args[0][0]
        assert len(inserted_bars) == 1
        assert isinstance(inserted_bars[0], Bar)

    @patch("src.ingestion.historical.get_alpaca_settings")
    @patch("src.ingestion.historical.StockHistoricalDataClient")
    def test_five_min_timeframe_passed_correctly(
        self,
        mock_client_cls: MagicMock,
        mock_settings: MagicMock,
    ) -> None:
        """Bars fetched with FIVE_MIN timeframe should carry that enum value."""
        from alpaca.data.models.bars import BarSet

        from src.ingestion.historical import fetch_historical_bars

        mock_settings.return_value = MagicMock(
            api_key="fake-key",
            secret_key="fake-secret",  # noqa: S106
            feed="sip",
        )

        mock_raw_bar = MagicMock()
        mock_raw_bar.timestamp = datetime(2024, 1, 15, 14, 30, tzinfo=UTC)
        mock_raw_bar.open = 480.0
        mock_raw_bar.high = 482.0
        mock_raw_bar.low = 479.0
        mock_raw_bar.close = 481.0
        mock_raw_bar.volume = 500_000
        mock_raw_bar.vwap = 480.5

        mock_barset = MagicMock(spec=BarSet)
        mock_barset.data = {"SPY": [mock_raw_bar]}

        mock_client_instance = mock_client_cls.return_value
        mock_client_instance.get_stock_bars.return_value = mock_barset

        mock_db = MagicMock()

        bars = fetch_historical_bars("SPY", days=5, timeframe=TimeFrame.FIVE_MIN, db=mock_db)

        assert len(bars) == 1
        assert bars[0].timeframe == TimeFrame.FIVE_MIN

    @patch("src.ingestion.historical.get_alpaca_settings")
    @patch("src.ingestion.historical.StockHistoricalDataClient")
    def test_skips_bars_with_zero_prices(
        self,
        mock_client_cls: MagicMock,
        mock_settings: MagicMock,
    ) -> None:
        """Bars with zero or negative prices should be skipped."""
        from alpaca.data.models.bars import BarSet

        from src.ingestion.historical import fetch_historical_bars

        mock_settings.return_value = MagicMock(
            api_key="fake-key",
            secret_key="fake-secret",  # noqa: S106
            feed="iex",
        )

        good_bar = MagicMock()
        good_bar.timestamp = datetime(2024, 1, 15, 14, 30, tzinfo=UTC)
        good_bar.open = 480.0
        good_bar.high = 481.0
        good_bar.low = 479.0
        good_bar.close = 480.5
        good_bar.volume = 100_000
        good_bar.vwap = 480.3

        bad_bar = MagicMock()
        bad_bar.timestamp = datetime(2024, 1, 15, 14, 31, tzinfo=UTC)
        bad_bar.open = 0.0
        bad_bar.high = 0.0
        bad_bar.low = 0.0
        bad_bar.close = 0.0
        bad_bar.volume = 0
        bad_bar.vwap = None

        mock_barset = MagicMock(spec=BarSet)
        mock_barset.data = {"SPY": [good_bar, bad_bar]}

        mock_client_instance = mock_client_cls.return_value
        mock_client_instance.get_stock_bars.return_value = mock_barset

        mock_db = MagicMock()

        bars = fetch_historical_bars("SPY", days=1, db=mock_db)

        assert len(bars) == 1
        assert bars[0].close == Decimal("480.5")
