"""Tests for src.backtest.vix_data — VIX daily data loader and per-bar mapper."""

from __future__ import annotations

import sqlite3
from datetime import date
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.backtest.vix_data import (
    _find_prev_vix,
    _get_conn,
    compute_vix_for_bars,
    download_vix,
    load_vix_series,
)

# ── _get_conn ────────────────────────────────────────────────────────────────


class TestGetConn:
    def test_creates_vix_daily_table(self, tmp_path: object) -> None:
        db_path = str(tmp_path / "test.db")  # type: ignore[operator]
        conn = _get_conn(db_path)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='vix_daily'"
        )
        assert cursor.fetchone() is not None
        conn.close()

    def test_enables_wal_mode(self, tmp_path: object) -> None:
        db_path = str(tmp_path / "test.db")  # type: ignore[operator]
        conn = _get_conn(db_path)
        mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]
        assert mode == "wal"
        conn.close()

    def test_creates_parent_directories(self, tmp_path: object) -> None:
        db_path = str(tmp_path / "nested" / "dir" / "test.db")  # type: ignore[operator]
        conn = _get_conn(db_path)
        assert conn is not None
        conn.close()


# ── download_vix ─────────────────────────────────────────────────────────────


class TestDownloadVix:
    @patch("src.backtest.vix_data.yf")
    def test_inserts_rows_from_yfinance(self, mock_yf: MagicMock, tmp_path: object) -> None:
        db_path = str(tmp_path / "test.db")  # type: ignore[operator]
        # Build a fake yfinance history DataFrame
        dates = pd.to_datetime(["2024-01-10", "2024-01-11", "2024-01-12"])
        hist = pd.DataFrame(
            {"Close": [18.5, 19.2, 17.8], "Open": [18.0, 18.5, 19.0]},
            index=dates,
        )
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = hist
        mock_yf.Ticker.return_value = mock_ticker

        count = download_vix(start="2024-01-10", end="2024-01-13", db_path=db_path)

        assert count == 3
        # Verify rows in SQLite
        conn = sqlite3.connect(db_path)
        rows = conn.execute("SELECT * FROM vix_daily ORDER BY date").fetchall()
        conn.close()
        assert len(rows) == 3
        assert rows[0] == ("2024-01-10", 18.5)
        assert rows[1] == ("2024-01-11", 19.2)

    @patch("src.backtest.vix_data.yf")
    def test_empty_yfinance_result_returns_zero(self, mock_yf: MagicMock, tmp_path: object) -> None:
        db_path = str(tmp_path / "test.db")  # type: ignore[operator]
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame()
        mock_yf.Ticker.return_value = mock_ticker

        count = download_vix(start="2024-01-10", end="2024-01-11", db_path=db_path)
        assert count == 0

    @patch("src.backtest.vix_data.yf")
    def test_upsert_overwrites_existing_rows(self, mock_yf: MagicMock, tmp_path: object) -> None:
        db_path = str(tmp_path / "test.db")  # type: ignore[operator]
        dates = pd.to_datetime(["2024-01-10"])
        hist = pd.DataFrame({"Close": [18.5], "Open": [18.0]}, index=dates)
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = hist
        mock_yf.Ticker.return_value = mock_ticker

        download_vix(start="2024-01-10", end="2024-01-11", db_path=db_path)

        # Second download with updated value
        hist2 = pd.DataFrame({"Close": [20.0], "Open": [18.0]}, index=dates)
        mock_ticker.history.return_value = hist2
        download_vix(start="2024-01-10", end="2024-01-11", db_path=db_path)

        conn = sqlite3.connect(db_path)
        row = conn.execute("SELECT close FROM vix_daily WHERE date='2024-01-10'").fetchone()
        conn.close()
        assert row[0] == 20.0


# ── load_vix_series ──────────────────────────────────────────────────────────


class TestLoadVixSeries:
    def test_empty_table_returns_empty_series(self, tmp_path: object) -> None:
        db_path = str(tmp_path / "test.db")  # type: ignore[operator]
        _get_conn(db_path).close()  # create table
        series = load_vix_series(db_path)
        assert isinstance(series, pd.Series)
        assert len(series) == 0

    def test_returns_correct_dates_and_values(self, tmp_path: object) -> None:
        db_path = str(tmp_path / "test.db")  # type: ignore[operator]
        conn = _get_conn(db_path)
        conn.execute("INSERT INTO vix_daily VALUES ('2024-01-10', 18.5)")
        conn.execute("INSERT INTO vix_daily VALUES ('2024-01-11', 19.2)")
        conn.commit()
        conn.close()

        series = load_vix_series(db_path)
        assert len(series) == 2
        assert series[date(2024, 1, 10)] == 18.5
        assert series[date(2024, 1, 11)] == 19.2

    def test_index_is_date_objects(self, tmp_path: object) -> None:
        db_path = str(tmp_path / "test.db")  # type: ignore[operator]
        conn = _get_conn(db_path)
        conn.execute("INSERT INTO vix_daily VALUES ('2024-01-10', 18.5)")
        conn.commit()
        conn.close()

        series = load_vix_series(db_path)
        assert isinstance(series.index[0], date)


# ── compute_vix_for_bars ─────────────────────────────────────────────────────


class TestComputeVixForBars:
    def _make_vix_series(self, data: dict[str, float]) -> pd.Series:
        """Helper: build a VIX series from {date_str: close} dict."""
        dates = [date.fromisoformat(d) for d in data]
        return pd.Series(list(data.values()), index=dates)

    def test_uses_previous_day_vix_not_current(self) -> None:
        """Bars on 2024-01-12 should get VIX from 2024-01-11, not 01-12."""
        vix = self._make_vix_series(
            {
                "2024-01-11": 18.0,
                "2024-01-12": 22.0,
            }
        )
        # Bars on 2024-01-12 (Friday) in UTC market hours
        idx = pd.date_range("2024-01-12 14:30", periods=5, freq="min", tz="UTC")
        result = compute_vix_for_bars(idx, vix)
        # Should use D-1 (Jan 11) = 18.0, NOT Jan 12 = 22.0
        assert result[0] == pytest.approx(18.0)
        assert result[-1] == pytest.approx(18.0)

    def test_nan_for_first_trading_day_no_prior_vix(self) -> None:
        """If VIX data starts on same day as bars, there is no D-1 value."""
        vix = self._make_vix_series({"2024-01-15": 20.0})
        idx = pd.date_range("2024-01-15 14:30", periods=3, freq="min", tz="UTC")
        result = compute_vix_for_bars(idx, vix)
        assert all(np.isnan(result))

    def test_empty_vix_series_returns_all_nan(self) -> None:
        vix = pd.Series(dtype=float)
        idx = pd.date_range("2024-01-15 14:30", periods=5, freq="min", tz="UTC")
        result = compute_vix_for_bars(idx, vix)
        assert len(result) == 5
        assert all(np.isnan(result))

    def test_handles_tz_naive_index(self) -> None:
        """tz-naive index should be localized to UTC before converting to ET."""
        vix = self._make_vix_series(
            {
                "2024-01-11": 18.0,
                "2024-01-12": 22.0,
            }
        )
        idx = pd.date_range("2024-01-12 14:30", periods=3, freq="min")  # naive
        result = compute_vix_for_bars(idx, vix)
        assert result[0] == pytest.approx(18.0)

    def test_multi_day_bars_use_correct_prev_vix(self) -> None:
        """Each day's bars should map to that day's D-1 VIX."""
        vix = self._make_vix_series(
            {
                "2024-01-10": 15.0,
                "2024-01-11": 18.0,
                "2024-01-12": 22.0,
            }
        )
        idx_day1 = pd.date_range("2024-01-11 14:30", periods=2, freq="min", tz="UTC")
        idx_day2 = pd.date_range("2024-01-12 14:30", periods=2, freq="min", tz="UTC")
        idx = idx_day1.append(idx_day2)
        result = compute_vix_for_bars(idx, vix)
        # Jan 11 bars -> VIX from Jan 10 = 15.0
        assert result[0] == pytest.approx(15.0)
        assert result[1] == pytest.approx(15.0)
        # Jan 12 bars -> VIX from Jan 11 = 18.0
        assert result[2] == pytest.approx(18.0)
        assert result[3] == pytest.approx(18.0)


# ── _find_prev_vix ───────────────────────────────────────────────────────────


class TestFindPrevVix:
    def test_finds_closest_prior_date(self) -> None:
        sorted_dates = [date(2024, 1, 8), date(2024, 1, 9), date(2024, 1, 10)]
        vix_dict = {d: float(i + 15) for i, d in enumerate(sorted_dates)}
        result = _find_prev_vix(date(2024, 1, 10), sorted_dates, vix_dict)
        # Strictly before Jan 10 -> Jan 9 -> value 16.0
        assert result == pytest.approx(16.0)

    def test_returns_none_when_no_prior_date(self) -> None:
        sorted_dates = [date(2024, 1, 10), date(2024, 1, 11)]
        vix_dict = {d: 20.0 for d in sorted_dates}
        result = _find_prev_vix(date(2024, 1, 9), sorted_dates, vix_dict)
        assert result is None

    def test_returns_none_for_target_equal_to_first_date(self) -> None:
        sorted_dates = [date(2024, 1, 10)]
        vix_dict = {date(2024, 1, 10): 20.0}
        result = _find_prev_vix(date(2024, 1, 10), sorted_dates, vix_dict)
        # bisect_left returns 0, idx = -1 -> None
        assert result is None

    def test_skips_weekend_gap(self) -> None:
        """Friday VIX used for Monday bars."""
        sorted_dates = [date(2024, 1, 12)]  # Friday
        vix_dict = {date(2024, 1, 12): 19.5}
        result = _find_prev_vix(date(2024, 1, 15), sorted_dates, vix_dict)  # Monday
        assert result == pytest.approx(19.5)
