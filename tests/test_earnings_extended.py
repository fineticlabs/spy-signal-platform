"""Extended tests for src.filters.earnings_calendar — fetch, cache, orchestrator."""

from __future__ import annotations

import json
import sys
from datetime import date
from unittest.mock import MagicMock

import pandas as pd
import pytest


@pytest.fixture()
def mock_yf(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Inject a mock yfinance module so the local import inside _fetch picks it up."""
    fake_yf = MagicMock()
    monkeypatch.setitem(sys.modules, "yfinance", fake_yf)
    return fake_yf


class TestFetchEarningsDates:
    """Tests for _fetch_earnings_dates (yfinance mocked)."""

    def test_returns_sorted_dates(self, mock_yf: MagicMock) -> None:
        """Mocked yfinance earnings returns sorted date list."""
        from src.filters.earnings_calendar import _fetch_earnings_dates

        idx = pd.DatetimeIndex(
            [
                pd.Timestamp("2024-04-25", tz="America/New_York"),
                pd.Timestamp("2024-01-24", tz="America/New_York"),
            ]
        )
        mock_ticker = MagicMock()
        mock_ticker.get_earnings_dates.return_value = pd.DataFrame({"EPS": [1.0, 2.0]}, index=idx)
        mock_yf.Ticker.return_value = mock_ticker

        result = _fetch_earnings_dates("AAPL", limit=50)

        assert result == [date(2024, 1, 24), date(2024, 4, 25)]

    def test_exception_returns_empty_list(self, mock_yf: MagicMock) -> None:
        """yfinance exception returns []."""
        from src.filters.earnings_calendar import _fetch_earnings_dates

        mock_ticker = MagicMock()
        mock_ticker.get_earnings_dates.side_effect = RuntimeError("API error")
        mock_yf.Ticker.return_value = mock_ticker

        result = _fetch_earnings_dates("AAPL")

        assert result == []

    def test_empty_df_returns_empty_list(self, mock_yf: MagicMock) -> None:
        """Empty earnings DataFrame returns []."""
        from src.filters.earnings_calendar import _fetch_earnings_dates

        mock_ticker = MagicMock()
        mock_ticker.get_earnings_dates.return_value = pd.DataFrame()
        mock_yf.Ticker.return_value = mock_ticker

        result = _fetch_earnings_dates("AAPL")

        assert result == []


class TestEarningsCacheIO:
    """Tests for load_earnings_cache / save_earnings_cache."""

    def test_load_existing_cache(
        self, tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """File exists returns dict."""
        import src.filters.earnings_calendar as mod

        cache_file = tmp_path / "earnings.json"  # type: ignore[operator]
        data = {"AAPL": ["2024-01-24", "2024-04-25"]}
        cache_file.write_text(json.dumps(data))
        monkeypatch.setattr(mod, "_CACHE_PATH", cache_file)

        result = mod.load_earnings_cache()

        assert result == data

    def test_load_missing_cache_returns_empty(
        self, tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Missing file returns {}."""
        import src.filters.earnings_calendar as mod

        cache_file = tmp_path / "nonexistent.json"  # type: ignore[operator]
        monkeypatch.setattr(mod, "_CACHE_PATH", cache_file)

        result = mod.load_earnings_cache()

        assert result == {}

    def test_save_writes_valid_json(
        self, tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Writes valid JSON to disk."""
        import src.filters.earnings_calendar as mod

        cache_file = tmp_path / "earnings.json"  # type: ignore[operator]
        monkeypatch.setattr(mod, "_CACHE_PATH", cache_file)

        data = {"TSLA": ["2024-01-24"]}
        mod.save_earnings_cache(data)

        assert cache_file.exists()
        loaded = json.loads(cache_file.read_text())
        assert loaded == data


class TestGetEarningsDates:
    """Tests for get_earnings_dates orchestrator."""

    def test_cache_hit_returns_from_cache(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Cache hit returns cached dates without fetching."""
        import src.filters.earnings_calendar as mod

        cache = {"AAPL": ["2024-01-24", "2024-04-25"]}
        monkeypatch.setattr(mod, "load_earnings_cache", lambda: cache)
        fetch_called = False

        def _fake_fetch(*a: object, **kw: object) -> list[date]:
            nonlocal fetch_called
            fetch_called = True
            return []

        monkeypatch.setattr(mod, "_fetch_earnings_dates", _fake_fetch)

        result = mod.get_earnings_dates("AAPL", use_cache=True, refresh=False)

        assert not fetch_called
        assert result == {date(2024, 1, 24), date(2024, 4, 25)}

    def test_cache_miss_fetches_and_caches(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Cache miss fetches and saves to cache."""
        import src.filters.earnings_calendar as mod

        monkeypatch.setattr(mod, "load_earnings_cache", lambda: {})
        monkeypatch.setattr(mod, "_fetch_earnings_dates", lambda sym, **kw: [date(2024, 7, 10)])
        saved: list[dict[str, list[str]]] = []
        monkeypatch.setattr(mod, "save_earnings_cache", lambda c: saved.append(c))

        result = mod.get_earnings_dates("MSFT", use_cache=True, refresh=False)

        assert result == {date(2024, 7, 10)}
        assert len(saved) == 1
        assert "MSFT" in saved[0]

    def test_refresh_refetches(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """refresh=True re-fetches even when cached."""
        import src.filters.earnings_calendar as mod

        cache = {"AAPL": ["2024-01-24"]}
        monkeypatch.setattr(mod, "load_earnings_cache", lambda: cache)
        monkeypatch.setattr(
            mod, "_fetch_earnings_dates", lambda sym, **kw: [date(2024, 1, 24), date(2024, 4, 25)]
        )
        monkeypatch.setattr(mod, "save_earnings_cache", lambda c: None)

        result = mod.get_earnings_dates("AAPL", use_cache=True, refresh=True)

        assert date(2024, 4, 25) in result


class TestPrefetchEarningsCache:
    """Tests for prefetch_earnings_cache."""

    def test_skips_cached_symbols(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Already-cached symbols are not re-fetched."""
        import src.filters.earnings_calendar as mod

        cache = {"AAPL": ["2024-01-24"]}
        monkeypatch.setattr(mod, "load_earnings_cache", lambda: dict(cache))
        fetched_symbols: list[str] = []

        def _fake_fetch(sym: str, **kw: object) -> list[date]:
            fetched_symbols.append(sym)
            return [date(2024, 7, 10)]

        monkeypatch.setattr(mod, "_fetch_earnings_dates", _fake_fetch)
        monkeypatch.setattr(mod, "save_earnings_cache", lambda c: None)

        mod.prefetch_earnings_cache(["AAPL", "TSLA"])

        assert "AAPL" not in fetched_symbols
        assert "TSLA" in fetched_symbols

    def test_fetches_uncached_symbols(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Uncached symbols are fetched and cache is saved."""
        import src.filters.earnings_calendar as mod

        monkeypatch.setattr(mod, "load_earnings_cache", lambda: {})
        monkeypatch.setattr(mod, "_fetch_earnings_dates", lambda sym, **kw: [date(2024, 3, 1)])
        saved: list[dict[str, list[str]]] = []
        monkeypatch.setattr(mod, "save_earnings_cache", lambda c: saved.append(dict(c)))

        mod.prefetch_earnings_cache(["GOOG", "META"])

        assert len(saved) == 1
        assert "GOOG" in saved[0]
        assert "META" in saved[0]
