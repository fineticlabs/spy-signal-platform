"""Comprehensive tests for filter modules: economic calendar, VIX term structure, earnings calendar."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

# ── Economic Calendar Tests ─────────────────────────────────────────────────


class TestIsHighImpactDay:
    """Tests for economic_calendar.is_high_impact_day()."""

    def test_fomc_date_is_high_impact(self) -> None:
        from src.filters.economic_calendar import is_high_impact_day

        assert is_high_impact_day(date(2024, 1, 31)) is True

    def test_random_date_is_not_high_impact(self) -> None:
        from src.filters.economic_calendar import is_high_impact_day

        # A Saturday — never a release day
        assert is_high_impact_day(date(2024, 2, 17)) is False

    def test_cpi_date_is_high_impact(self) -> None:
        from src.filters.economic_calendar import is_high_impact_day

        assert is_high_impact_day(date(2024, 1, 11)) is True

    def test_ppi_date_is_high_impact(self) -> None:
        from src.filters.economic_calendar import is_high_impact_day

        assert is_high_impact_day(date(2024, 1, 12)) is True

    def test_nfp_first_friday_is_high_impact(self) -> None:
        from src.filters.economic_calendar import is_high_impact_day

        # 2024-01-05 is the first Friday of January 2024
        assert is_high_impact_day(date(2024, 1, 5)) is True


class TestGetEventTypes:
    """Tests for economic_calendar.get_event_types()."""

    def test_fomc_date_returns_fomc(self) -> None:
        from src.filters.economic_calendar import get_event_types

        events = get_event_types(date(2024, 1, 31))
        assert events == ["FOMC"]

    def test_normal_date_returns_empty(self) -> None:
        from src.filters.economic_calendar import get_event_types

        events = get_event_types(date(2024, 2, 17))
        assert events == []

    def test_cpi_date_returns_cpi(self) -> None:
        from src.filters.economic_calendar import get_event_types

        events = get_event_types(date(2024, 1, 11))
        assert "CPI" in events

    def test_multiple_events_same_day(self) -> None:
        """CPI 2020-06-10 and FOMC 2020-06-10 overlap."""
        from src.filters.economic_calendar import get_event_types

        events = get_event_types(date(2020, 6, 10))
        assert "FOMC" in events
        assert "CPI" in events

    def test_event_order_is_fomc_nfp_cpi_ppi(self) -> None:
        """When multiple events exist, order is FOMC, NFP, CPI, PPI."""
        from src.filters.economic_calendar import get_event_types

        # 2020-06-10 has FOMC and CPI
        events = get_event_types(date(2020, 6, 10))
        fomc_idx = events.index("FOMC")
        cpi_idx = events.index("CPI")
        assert fomc_idx < cpi_idx


class TestNfpDates:
    """Tests for NFP first-Friday-of-month logic."""

    def test_first_friday_jan_2024(self) -> None:
        """Jan 1 2024 is Monday, so first Friday is Jan 5."""
        from src.filters.economic_calendar import _NFP_DATES

        assert date(2024, 1, 5) in _NFP_DATES

    def test_first_friday_march_2024(self) -> None:
        """Mar 1 2024 is Friday, so first Friday is Mar 1."""
        from src.filters.economic_calendar import _NFP_DATES

        assert date(2024, 3, 1) in _NFP_DATES

    def test_first_friday_feb_2024(self) -> None:
        """Feb 1 2024 is Thursday, so first Friday is Feb 2."""
        from src.filters.economic_calendar import _NFP_DATES

        assert date(2024, 2, 2) in _NFP_DATES

    def test_second_friday_is_not_nfp(self) -> None:
        from src.filters.economic_calendar import _NFP_DATES

        # Second Friday of Jan 2024 is Jan 12
        assert date(2024, 1, 12) not in _NFP_DATES


class TestComputeEconBlockedArray:
    """Tests for economic_calendar.compute_econ_blocked_array()."""

    def test_returns_boolean_array(self) -> None:
        from src.filters.economic_calendar import compute_econ_blocked_array

        idx = pd.DatetimeIndex(
            ["2024-01-31 14:30:00", "2024-01-31 14:31:00", "2024-02-17 14:30:00"],
            tz="UTC",
        )
        result = compute_econ_blocked_array(idx)
        assert result.dtype == bool
        assert len(result) == 3

    def test_fomc_day_bars_are_blocked(self) -> None:
        from src.filters.economic_calendar import compute_econ_blocked_array

        # 2024-01-31 is FOMC — these UTC times map to ET trading hours
        idx = pd.DatetimeIndex(
            ["2024-01-31 14:30:00", "2024-01-31 15:00:00"],
            tz="UTC",
        )
        result = compute_econ_blocked_array(idx)
        assert result[0] is np.True_
        assert result[1] is np.True_

    def test_normal_day_bars_not_blocked(self) -> None:
        from src.filters.economic_calendar import compute_econ_blocked_array

        idx = pd.DatetimeIndex(
            ["2024-02-17 14:30:00", "2024-02-17 15:00:00"],
            tz="UTC",
        )
        result = compute_econ_blocked_array(idx)
        assert not result[0]
        assert not result[1]

    def test_mixed_days(self) -> None:
        from src.filters.economic_calendar import compute_econ_blocked_array

        idx = pd.DatetimeIndex(
            [
                "2024-01-31 14:30:00",  # FOMC day — blocked
                "2024-02-01 14:30:00",  # normal — not blocked
            ],
            tz="UTC",
        )
        result = compute_econ_blocked_array(idx)
        assert result[0] is np.True_
        assert not result[1]

    def test_empty_index(self) -> None:
        from src.filters.economic_calendar import compute_econ_blocked_array

        idx = pd.DatetimeIndex([], tz="UTC")
        result = compute_econ_blocked_array(idx)
        assert len(result) == 0

    def test_tz_naive_index_is_handled(self) -> None:
        """Index without tz info should be localized to UTC internally."""
        from src.filters.economic_calendar import compute_econ_blocked_array

        idx = pd.DatetimeIndex(["2024-01-31 14:30:00", "2024-02-01 14:30:00"])
        result = compute_econ_blocked_array(idx)
        assert len(result) == 2
        # FOMC day bar should be blocked
        assert result[0] is np.True_


# ── VIX Term Structure Tests ────────────────────────────────────────────────


class TestClassifyTermStructure:
    """Tests for vix_term_structure.classify_term_structure()."""

    def test_deep_contango(self) -> None:
        from src.filters.vix_term_structure import classify_term_structure

        assert classify_term_structure(0.80) == "CONTANGO"

    def test_backwardation(self) -> None:
        from src.filters.vix_term_structure import classify_term_structure

        assert classify_term_structure(1.05) == "BACKWARDATION"

    def test_normal_range_returns_none(self) -> None:
        from src.filters.vix_term_structure import classify_term_structure

        assert classify_term_structure(0.90) is None

    def test_nan_returns_none(self) -> None:
        from src.filters.vix_term_structure import classify_term_structure

        assert classify_term_structure(float("nan")) is None

    def test_at_contango_threshold_returns_none(self) -> None:
        """Exactly 0.85 is NOT contango (needs to be strictly less)."""
        from src.filters.vix_term_structure import classify_term_structure

        assert classify_term_structure(0.85) is None

    def test_at_backwardation_threshold_returns_none(self) -> None:
        """Exactly 1.00 is NOT backwardation (needs to be strictly greater)."""
        from src.filters.vix_term_structure import classify_term_structure

        assert classify_term_structure(1.00) is None

    def test_just_below_contango_threshold(self) -> None:
        from src.filters.vix_term_structure import classify_term_structure

        assert classify_term_structure(0.849) == "CONTANGO"

    def test_just_above_backwardation_threshold(self) -> None:
        from src.filters.vix_term_structure import classify_term_structure

        assert classify_term_structure(1.001) == "BACKWARDATION"


class TestVixTermStructureConstants:
    """Verify VIX term structure threshold constants."""

    def test_contango_threshold(self) -> None:
        from src.filters.vix_term_structure import CONTANGO_THRESHOLD

        assert CONTANGO_THRESHOLD == 0.85

    def test_backwardation_threshold(self) -> None:
        from src.filters.vix_term_structure import BACKWARDATION_THRESHOLD

        assert BACKWARDATION_THRESHOLD == 1.00


class TestComputeVixTermStructureArray:
    """Tests for vix_term_structure.compute_vix_term_structure_array()."""

    @staticmethod
    def _make_vts_df(dates: list[str], ratios: list[float]) -> pd.DataFrame:
        """Helper: build a VTS DataFrame with given dates and ratios."""
        idx = pd.DatetimeIndex(dates)
        return pd.DataFrame(
            {"vix": [20.0] * len(dates), "vix3m": [25.0] * len(dates), "ratio": ratios},
            index=idx,
        )

    def test_correct_length(self) -> None:
        from src.filters.vix_term_structure import compute_vix_term_structure_array

        vts_df = self._make_vts_df(
            ["2024-01-29", "2024-01-30", "2024-01-31"],
            [0.80, 0.90, 1.05],
        )
        idx = pd.DatetimeIndex(
            ["2024-01-31 14:30:00", "2024-01-31 14:31:00", "2024-01-31 14:32:00"],
            tz="UTC",
        )
        result = compute_vix_term_structure_array(idx, vts_df)
        assert len(result) == 3

    def test_nan_for_missing_data(self) -> None:
        from src.filters.vix_term_structure import compute_vix_term_structure_array

        # Empty VTS DataFrame — all output should be NaN
        vts_df = pd.DataFrame(columns=["vix", "vix3m", "ratio"])
        idx = pd.DatetimeIndex(["2024-01-31 14:30:00"], tz="UTC")
        result = compute_vix_term_structure_array(idx, vts_df)
        assert np.isnan(result[0])

    def test_one_day_shift(self) -> None:
        """Bar on day D should use ratio from day D-1."""
        from src.filters.vix_term_structure import compute_vix_term_structure_array

        vts_df = self._make_vts_df(
            ["2024-01-29", "2024-01-30", "2024-01-31"],
            [0.80, 0.90, 1.05],
        )
        # Bar on 2024-01-31 ET should use ratio from 2024-01-30 (0.90)
        idx = pd.DatetimeIndex(["2024-01-31 14:30:00"], tz="UTC")
        result = compute_vix_term_structure_array(idx, vts_df)
        assert result[0] == pytest.approx(0.90)

    def test_first_day_is_nan(self) -> None:
        """The first day in the VTS has no D-1, so should be NaN."""
        from src.filters.vix_term_structure import compute_vix_term_structure_array

        vts_df = self._make_vts_df(["2024-01-29"], [0.80])
        idx = pd.DatetimeIndex(["2024-01-29 14:30:00"], tz="UTC")
        result = compute_vix_term_structure_array(idx, vts_df)
        assert np.isnan(result[0])

    def test_values_mapped_correctly(self) -> None:
        """Multiple bars across two days get correct shifted ratios."""
        from src.filters.vix_term_structure import compute_vix_term_structure_array

        vts_df = self._make_vts_df(
            ["2024-01-29", "2024-01-30", "2024-01-31"],
            [0.80, 0.90, 1.05],
        )
        idx = pd.DatetimeIndex(
            [
                "2024-01-30 14:30:00",  # D=Jan30, uses D-1=Jan29 ratio=0.80
                "2024-01-30 14:31:00",  # same day, same ratio
                "2024-01-31 14:30:00",  # D=Jan31, uses D-1=Jan30 ratio=0.90
            ],
            tz="UTC",
        )
        result = compute_vix_term_structure_array(idx, vts_df)
        assert result[0] == pytest.approx(0.80)
        assert result[1] == pytest.approx(0.80)
        assert result[2] == pytest.approx(0.90)

    def test_empty_index(self) -> None:
        from src.filters.vix_term_structure import compute_vix_term_structure_array

        vts_df = self._make_vts_df(["2024-01-29"], [0.80])
        idx = pd.DatetimeIndex([], tz="UTC")
        result = compute_vix_term_structure_array(idx, vts_df)
        assert len(result) == 0

    def test_missing_ratio_column(self) -> None:
        from src.filters.vix_term_structure import compute_vix_term_structure_array

        vts_df = pd.DataFrame(
            {"vix": [20.0], "vix3m": [25.0]}, index=pd.DatetimeIndex(["2024-01-29"])
        )
        idx = pd.DatetimeIndex(["2024-01-29 14:30:00"], tz="UTC")
        result = compute_vix_term_structure_array(idx, vts_df)
        assert np.isnan(result[0])


class TestLoadVixTermStructureCache:
    """Tests for vix_term_structure.load_vix_term_structure_cache()."""

    def test_returns_empty_when_no_cache(
        self, tmp_path: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from pathlib import Path

        import src.filters.vix_term_structure as mod

        # Point cache path to a non-existent file
        monkeypatch.setattr(mod, "_CACHE_PATH", Path("/tmp/nonexistent_vix_cache_test.parquet"))  # noqa: S108
        result = mod.load_vix_term_structure_cache()
        assert isinstance(result, pd.DataFrame)
        assert result.empty
        assert list(result.columns) == ["vix", "vix3m", "ratio"]


# ── Earnings Calendar Tests ─────────────────────────────────────────────────


class TestIsEarningsBlackout:
    """Tests for earnings_calendar.is_earnings_blackout()."""

    def test_earnings_day_is_blocked(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.filters import earnings_calendar as mod

        earnings_dates = {date(2024, 1, 25)}
        monkeypatch.setattr(
            mod,
            "get_earnings_dates",
            lambda symbol, **kwargs: earnings_dates,
        )

        assert mod.is_earnings_blackout("AAPL", date(2024, 1, 25)) is True

    def test_day_after_earnings_is_blocked(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.filters import earnings_calendar as mod

        earnings_dates = {date(2024, 1, 25)}  # Thursday
        monkeypatch.setattr(
            mod,
            "get_earnings_dates",
            lambda symbol, **kwargs: earnings_dates,
        )

        # Day after (Friday)
        assert mod.is_earnings_blackout("AAPL", date(2024, 1, 26)) is True

    def test_two_days_after_not_blocked(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.filters import earnings_calendar as mod

        earnings_dates = {date(2024, 1, 25)}  # Thursday
        monkeypatch.setattr(
            mod,
            "get_earnings_dates",
            lambda symbol, **kwargs: earnings_dates,
        )

        # Two days after (Saturday — but also not in blackout)
        assert mod.is_earnings_blackout("AAPL", date(2024, 1, 27)) is False

    def test_day_before_not_blocked(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.filters import earnings_calendar as mod

        earnings_dates = {date(2024, 1, 25)}
        monkeypatch.setattr(
            mod,
            "get_earnings_dates",
            lambda symbol, **kwargs: earnings_dates,
        )

        assert mod.is_earnings_blackout("AAPL", date(2024, 1, 24)) is False


class TestGetEarningsBlackoutDates:
    """Tests for earnings_calendar.get_earnings_blackout_dates()."""

    def test_basic_blackout_window(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Earnings on Wednesday → Wed + Thu blocked."""
        from src.filters import earnings_calendar as mod

        # 2024-01-24 is Wednesday
        earnings_dates = {date(2024, 1, 24)}
        monkeypatch.setattr(
            mod,
            "get_earnings_dates",
            lambda symbol, **kwargs: earnings_dates,
        )

        result = mod.get_earnings_blackout_dates("AAPL")
        assert date(2024, 1, 24) in result  # earnings day
        assert date(2024, 1, 25) in result  # day after
        assert date(2024, 1, 26) not in result  # two days after

    def test_friday_earnings_blocks_monday(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Earnings on Friday → Friday + Saturday + Monday blocked."""
        from src.filters import earnings_calendar as mod

        # 2024-01-26 is Friday
        earnings_dates = {date(2024, 1, 26)}
        monkeypatch.setattr(
            mod,
            "get_earnings_dates",
            lambda symbol, **kwargs: earnings_dates,
        )

        result = mod.get_earnings_blackout_dates("AAPL")
        assert date(2024, 1, 26) in result  # Friday (earnings day)
        assert date(2024, 1, 27) in result  # Saturday (day after)
        assert date(2024, 1, 29) in result  # Monday (Friday + 3 days)

    def test_empty_earnings(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.filters import earnings_calendar as mod

        monkeypatch.setattr(
            mod,
            "get_earnings_dates",
            lambda symbol, **kwargs: set(),
        )

        result = mod.get_earnings_blackout_dates("AAPL")
        assert len(result) == 0

    def test_thursday_earnings_no_monday_block(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Earnings on Thursday → Thu + Fri only, no Monday."""
        from src.filters import earnings_calendar as mod

        # 2024-01-25 is Thursday
        earnings_dates = {date(2024, 1, 25)}
        monkeypatch.setattr(
            mod,
            "get_earnings_dates",
            lambda symbol, **kwargs: earnings_dates,
        )

        result = mod.get_earnings_blackout_dates("AAPL")
        assert date(2024, 1, 25) in result  # Thursday
        assert date(2024, 1, 26) in result  # Friday
        assert date(2024, 1, 29) not in result  # Monday — not blocked


class TestComputeEarningsBlockedArray:
    """Tests for earnings_calendar.compute_earnings_blocked_array()."""

    def test_returns_boolean_array(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.filters import earnings_calendar as mod

        monkeypatch.setattr(
            mod,
            "get_earnings_blackout_dates",
            lambda symbol, **kwargs: {date(2024, 1, 25)},
        )

        idx = pd.DatetimeIndex(
            ["2024-01-25 14:30:00", "2024-01-26 14:30:00"],
            tz="UTC",
        )
        result = mod.compute_earnings_blocked_array(idx, "AAPL")
        assert result.dtype == bool
        assert len(result) == 2

    def test_blocked_on_blackout_day(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.filters import earnings_calendar as mod

        monkeypatch.setattr(
            mod,
            "get_earnings_blackout_dates",
            lambda symbol, **kwargs: {date(2024, 1, 25), date(2024, 1, 26)},
        )

        idx = pd.DatetimeIndex(
            [
                "2024-01-25 14:30:00",  # blocked
                "2024-01-26 14:30:00",  # blocked
                "2024-01-27 14:30:00",  # not blocked
            ],
            tz="UTC",
        )
        result = mod.compute_earnings_blocked_array(idx, "AAPL")
        assert result[0] is np.True_
        assert result[1] is np.True_
        assert not result[2]

    def test_empty_blackout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.filters import earnings_calendar as mod

        monkeypatch.setattr(
            mod,
            "get_earnings_blackout_dates",
            lambda symbol, **kwargs: set(),
        )

        idx = pd.DatetimeIndex(
            ["2024-01-25 14:30:00", "2024-01-26 14:30:00"],
            tz="UTC",
        )
        result = mod.compute_earnings_blocked_array(idx, "AAPL")
        assert not result.any()

    def test_empty_index(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.filters import earnings_calendar as mod

        monkeypatch.setattr(
            mod,
            "get_earnings_blackout_dates",
            lambda symbol, **kwargs: {date(2024, 1, 25)},
        )

        idx = pd.DatetimeIndex([], tz="UTC")
        result = mod.compute_earnings_blocked_array(idx, "AAPL")
        assert len(result) == 0

    def test_no_real_api_calls(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Ensure yfinance is never called during tests."""
        from src.filters import earnings_calendar as mod

        call_tracker: list[str] = []

        def mock_fetch(symbol: str, limit: int = 100) -> list[date]:
            call_tracker.append(symbol)
            return []

        monkeypatch.setattr(mod, "_fetch_earnings_dates", mock_fetch)
        monkeypatch.setattr(
            mod,
            "get_earnings_blackout_dates",
            lambda symbol, **kwargs: set(),
        )

        idx = pd.DatetimeIndex(["2024-01-25 14:30:00"], tz="UTC")
        mod.compute_earnings_blocked_array(idx, "AAPL")

        # _fetch_earnings_dates should NOT have been called because we mocked
        # get_earnings_blackout_dates directly
        assert len(call_tracker) == 0
