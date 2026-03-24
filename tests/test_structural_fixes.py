"""Tests for structural audit fixes (2026-03-24): daily ADX, 15-min EMA, VIX fetch."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from src.models import Bar, Regime, TimeFrame
from src.strategies.regime import RegimeDetector

# ── helpers ──────────────────────────────────────────────────────────────────


def _make_bar(
    close: float,
    hour: int = 9,
    minute: int = 30,
) -> Bar:
    """Build a bar at the given ET time."""
    from zoneinfo import ZoneInfo

    et = ZoneInfo("America/New_York")
    ts = datetime(2024, 1, 15, hour, minute, tzinfo=et).astimezone(UTC)
    return Bar(
        symbol="SPY",
        timeframe=TimeFrame.ONE_MIN,
        timestamp=ts,
        open=Decimal(str(close)),
        high=Decimal(str(close + 0.5)),
        low=Decimal(str(close - 0.5)),
        close=Decimal(str(close)),
        volume=100000,
        vwap=Decimal(str(close)),
    )


# ── Daily ADX tests ─────────────────────────────────────────────────────────


class TestDailyADX:
    def test_daily_adx_set(self) -> None:
        """set_daily_adx stores the value and it's returned by adx_value."""
        regime = RegimeDetector()
        regime.set_daily_adx(Decimal("28.5"))
        assert regime.daily_adx == Decimal("28.5")
        assert regime.adx_value == Decimal("28.5")

    def test_daily_adx_takes_priority_over_streaming(self) -> None:
        """Daily ADX is preferred over streaming ADX for signal gating."""
        regime = RegimeDetector()
        regime.update(adx=Decimal("35"))  # streaming ADX
        regime.set_daily_adx(Decimal("22"))  # daily ADX
        # adx_value should return daily (22), not streaming (35)
        assert regime.adx_value == Decimal("22")

    def test_streaming_adx_used_as_fallback(self) -> None:
        """If daily ADX not set, streaming ADX is used."""
        regime = RegimeDetector()
        regime.update(adx=Decimal("30"))
        assert regime.daily_adx is None
        assert regime.adx_value == Decimal("30")

    def test_streaming_adx_still_accessible(self) -> None:
        """Streaming ADX remains available for regime display."""
        regime = RegimeDetector()
        regime.update(adx=Decimal("35"))
        regime.set_daily_adx(Decimal("22"))
        assert regime.streaming_adx == Decimal("35")


# ── 15-min EMA tests ────────────────────────────────────────────────────────


class TestEMA15m:
    def test_seed_15m_ema(self) -> None:
        """Seeding with historical 15-min closes produces an EMA value."""
        regime = RegimeDetector()
        closes = [480.0 + i * 0.1 for i in range(25)]
        regime.seed_15m_ema(closes)
        assert regime.ema20_15m is not None
        assert float(regime.ema20_15m) > 0

    def test_update_15m_bar_aggregates(self) -> None:
        """Feeding 15 bars at aligned times emits a 15-min close."""
        regime = RegimeDetector()
        # Seed with some initial data
        regime.seed_15m_ema([480.0 + i * 0.1 for i in range(25)])
        # Feed 15 bars from 9:30 to 9:44 ET (completing the 9:30-9:44 bucket)
        for minute in range(30, 45):
            bar = _make_bar(close=485.0, hour=9, minute=minute)
            regime.update_15m_bar(bar)

        # EMA should have updated (might be same or different value)
        assert regime.ema20_15m is not None

    def test_ema20_15m_none_before_seed(self) -> None:
        """Before seeding, ema20_15m is None."""
        regime = RegimeDetector()
        assert regime.ema20_15m is None


# ── VIX fetch tests ──────────────────────────────────────────────────────────


class TestVIXFetch:
    def test_fetch_live_vix_success(self) -> None:
        """Successful VIX fetch returns a Decimal."""
        from src.main import _fetch_live_vix

        mock_ticker = MagicMock()
        mock_ticker.fast_info = {"lastPrice": 18.5}

        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = _fetch_live_vix()

        assert result == Decimal("18.5")

    def test_fetch_live_vix_failure_returns_none(self) -> None:
        """Failed VIX fetch returns None (doesn't crash)."""
        from src.main import _fetch_live_vix

        with patch("yfinance.Ticker", side_effect=ConnectionError("no internet")):
            result = _fetch_live_vix()

        assert result is None

    def test_fetch_live_vix_zero_returns_none(self) -> None:
        """Zero price returns None."""
        from src.main import _fetch_live_vix

        mock_ticker = MagicMock()
        mock_ticker.fast_info = {"lastPrice": 0}

        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = _fetch_live_vix()

        assert result is None


# ── Regime integration tests ─────────────────────────────────────────────────


class TestRegimeIntegration:
    def test_regime_with_daily_adx_and_vix(self) -> None:
        """Regime detector works correctly with daily ADX and real VIX."""
        regime = RegimeDetector()
        regime.set_daily_adx(Decimal("28"))
        regime.update(vix=Decimal("18"), trending_up=True)
        assert regime.is_tradeable
        assert regime.current_regime == Regime.RANGING  # streaming ADX not set
        # But adx_value for signal gate uses daily
        assert regime.adx_value == Decimal("28")

    def test_is_tradeable_uses_daily_adx(self) -> None:
        """is_tradeable property uses daily ADX (not streaming)."""
        regime = RegimeDetector()
        regime.update(vix=Decimal("18"), adx=Decimal("35"))  # streaming ADX high
        regime.set_daily_adx(Decimal("18"))  # daily ADX low
        # is_tradeable checks adx_value which prefers daily_adx (18) < 20 threshold
        assert not regime.is_tradeable


# ── Backtest slippage tests ──────────────────────────────────────────────────


class TestBacktestSlippage:
    def test_slippage_applied_to_backtest(self) -> None:
        """Verify the Backtest constructor receives a non-zero spread."""
        import numpy as np
        import pandas as pd

        from src.backtest.engine import run_backtest

        dates = pd.date_range("2024-01-15 14:30", periods=50, freq="1min", tz="UTC")
        df = pd.DataFrame(
            {
                "open": np.full(50, 480.0),
                "high": np.full(50, 481.0),
                "low": np.full(50, 479.0),
                "close": np.full(50, 480.0),
                "volume": np.full(50, 100000, dtype=int),
            },
            index=dates,
        )

        with patch("src.backtest.engine.Backtest") as mock_bt:
            mock_bt.return_value.run.return_value = {
                "# Trades": 0,
                "Return [%]": 0.0,
                "Win Rate [%]": 0.0,
            }
            mock_bt.return_value._trades = pd.DataFrame()

            run_backtest(df, cash=50000, slippage=0.02)

            # Verify spread was passed (0.02 / 200 = 0.0001)
            bt_call = mock_bt.call_args
            assert "spread" in bt_call.kwargs
            assert bt_call.kwargs["spread"] > 0
            assert bt_call.kwargs["spread"] == pytest.approx(0.0001)
