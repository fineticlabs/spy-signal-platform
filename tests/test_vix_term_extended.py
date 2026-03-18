"""Extended tests for src.filters.vix_term_structure — fetch, save, orchestrator."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pandas as pd
import pytest


@pytest.fixture()
def mock_yf(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Inject a mock yfinance module so the local import inside fetch picks it up."""
    fake_yf = MagicMock()
    monkeypatch.setitem(sys.modules, "yfinance", fake_yf)
    return fake_yf


class TestFetchVixTermStructure:
    """Tests for fetch_vix_term_structure (yfinance mocked)."""

    def test_returns_df_with_expected_columns(self, mock_yf: MagicMock) -> None:
        """Mocked yfinance returns DF with vix, vix3m, ratio columns."""
        from src.filters.vix_term_structure import fetch_vix_term_structure

        idx = pd.date_range("2024-01-02", periods=3, freq="B")
        mock_yf.download.side_effect = [
            pd.DataFrame({"Close": [15.0, 16.0, 14.0]}, index=idx),
            pd.DataFrame({"Close": [18.0, 19.0, 17.0]}, index=idx),
        ]

        df = fetch_vix_term_structure(start="2024-01-02", end="2024-01-05")

        assert list(df.columns) == ["vix", "vix3m", "ratio"]
        assert len(df) == 3
        assert pytest.approx(df["ratio"].iloc[0]) == 15.0 / 18.0

    def test_empty_yfinance_returns_empty_df(self, mock_yf: MagicMock) -> None:
        """Empty yfinance download returns empty DF."""
        from src.filters.vix_term_structure import fetch_vix_term_structure

        mock_yf.download.return_value = pd.DataFrame()

        df = fetch_vix_term_structure()

        assert df.empty
        assert list(df.columns) == ["vix", "vix3m", "ratio"]

    def test_multiindex_columns_flattened(self, mock_yf: MagicMock) -> None:
        """MultiIndex columns from yfinance are flattened correctly."""
        from src.filters.vix_term_structure import fetch_vix_term_structure

        idx = pd.date_range("2024-01-02", periods=2, freq="B")
        vix_df = pd.DataFrame(
            {("Close", "^VIX"): [15.0, 16.0], ("Open", "^VIX"): [14.5, 15.5]},
            index=idx,
        )
        vix_df.columns = pd.MultiIndex.from_tuples(vix_df.columns)

        vix3m_df = pd.DataFrame(
            {("Close", "^VIX3M"): [18.0, 19.0], ("Open", "^VIX3M"): [17.5, 18.5]},
            index=idx,
        )
        vix3m_df.columns = pd.MultiIndex.from_tuples(vix3m_df.columns)

        mock_yf.download.side_effect = [vix_df, vix3m_df]

        df = fetch_vix_term_structure(start="2024-01-02", end="2024-01-04")

        assert "vix" in df.columns
        assert "ratio" in df.columns
        assert len(df) == 2


class TestSaveVixTermStructureCache:
    """Tests for save_vix_term_structure_cache (file I/O)."""

    def test_creates_parquet_file(
        self, tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Saves DataFrame to parquet at the configured cache path."""
        import src.filters.vix_term_structure as mod

        cache_file = tmp_path / "vix_cache.parquet"  # type: ignore[operator]
        monkeypatch.setattr(mod, "_CACHE_PATH", cache_file)

        idx = pd.date_range("2024-01-02", periods=2, freq="B")
        df = pd.DataFrame(
            {"vix": [15.0, 16.0], "vix3m": [18.0, 19.0], "ratio": [0.83, 0.84]}, index=idx
        )

        mod.save_vix_term_structure_cache(df)

        assert cache_file.exists()
        loaded = pd.read_parquet(cache_file)
        assert len(loaded) == 2
        assert list(loaded.columns) == ["vix", "vix3m", "ratio"]


class TestGetVixTermStructure:
    """Tests for get_vix_term_structure orchestrator."""

    def test_uses_cache_when_available(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Returns cached data without calling fetch."""
        import src.filters.vix_term_structure as mod

        idx = pd.date_range("2024-01-02", periods=2, freq="B")
        cached_df = pd.DataFrame(
            {"vix": [15.0, 16.0], "vix3m": [18.0, 19.0], "ratio": [0.83, 0.84]}, index=idx
        )

        monkeypatch.setattr(mod, "load_vix_term_structure_cache", lambda: cached_df)
        fetch_called = False

        def _fake_fetch(*a: object, **kw: object) -> pd.DataFrame:
            nonlocal fetch_called
            fetch_called = True
            return pd.DataFrame()

        monkeypatch.setattr(mod, "fetch_vix_term_structure", _fake_fetch)

        result = mod.get_vix_term_structure(use_cache=True, refresh=False)

        assert not fetch_called
        assert len(result) == 2

    def test_fetches_when_no_cache(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Fetches from yfinance when cache is empty."""
        import src.filters.vix_term_structure as mod

        empty_df = pd.DataFrame(columns=["vix", "vix3m", "ratio"])
        idx = pd.date_range("2024-01-02", periods=1, freq="B")
        fetched_df = pd.DataFrame({"vix": [15.0], "vix3m": [18.0], "ratio": [0.83]}, index=idx)

        monkeypatch.setattr(mod, "load_vix_term_structure_cache", lambda: empty_df)
        monkeypatch.setattr(mod, "fetch_vix_term_structure", lambda **kw: fetched_df)
        monkeypatch.setattr(mod, "save_vix_term_structure_cache", lambda df: None)

        result = mod.get_vix_term_structure(use_cache=True, refresh=False)

        assert len(result) == 1

    def test_refresh_forces_refetch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """refresh=True bypasses cache and re-fetches."""
        import src.filters.vix_term_structure as mod

        idx = pd.date_range("2024-01-02", periods=2, freq="B")
        cached_df = pd.DataFrame(
            {"vix": [15.0, 16.0], "vix3m": [18.0, 19.0], "ratio": [0.83, 0.84]}, index=idx
        )

        idx2 = pd.date_range("2024-01-02", periods=3, freq="B")
        fetched_df = pd.DataFrame(
            {"vix": [15.0, 16.0, 17.0], "vix3m": [18.0, 19.0, 20.0], "ratio": [0.83, 0.84, 0.85]},
            index=idx2,
        )

        monkeypatch.setattr(mod, "load_vix_term_structure_cache", lambda: cached_df)
        monkeypatch.setattr(mod, "fetch_vix_term_structure", lambda **kw: fetched_df)
        monkeypatch.setattr(mod, "save_vix_term_structure_cache", lambda df: None)

        result = mod.get_vix_term_structure(use_cache=True, refresh=True)

        assert len(result) == 3
