"""Tests for candlestick quality filters (engulfing, pin bar, inside bar)."""

from __future__ import annotations

from src.strategies.candlestick_filters import (
    has_inside_bar_compression,
    is_engulfing,
    is_pin_bar_rejection,
)

# ── Engulfing bar tests ─────────────────────────────────────────────────────


class TestIsEngulfing:
    def test_long_engulfing_green_candle_covers_prior(self) -> None:
        """Green candle body fully covers prior body, body > 60% range."""
        assert is_engulfing(
            curr_open=100.0,
            curr_high=105.0,
            curr_low=99.5,
            curr_close=104.5,
            prev_open=102.0,
            prev_close=101.0,
            direction="LONG",
        )

    def test_short_engulfing_red_candle_covers_prior(self) -> None:
        """Red candle body fully covers prior body, body > 60% range."""
        assert is_engulfing(
            curr_open=104.0,
            curr_high=104.5,
            curr_low=99.0,
            curr_close=99.5,
            prev_open=101.0,
            prev_close=102.5,
            direction="SHORT",
        )

    def test_long_engulfing_rejected_when_red(self) -> None:
        """LONG engulfing requires green candle (close > open)."""
        assert not is_engulfing(
            curr_open=104.0,
            curr_high=105.0,
            curr_low=99.0,
            curr_close=100.0,
            prev_open=102.0,
            prev_close=101.0,
            direction="LONG",
        )

    def test_not_engulfing_when_body_too_small(self) -> None:
        """Reject if body < 60% of range (too much wick)."""
        # body = 1.0, range = 5.0, ratio = 20% < 60%
        assert not is_engulfing(
            curr_open=100.0,
            curr_high=105.0,
            curr_low=100.0,
            curr_close=101.0,
            prev_open=100.5,
            prev_close=100.5,
            direction="LONG",
        )

    def test_not_engulfing_when_body_doesnt_cover_prior(self) -> None:
        """Reject if current body doesn't fully cover prior body."""
        assert not is_engulfing(
            curr_open=101.0,
            curr_high=104.0,
            curr_low=100.5,
            curr_close=103.5,
            prev_open=100.0,
            prev_close=102.0,
            direction="LONG",
        )

    def test_zero_range_returns_false(self) -> None:
        """Doji (zero range) is not engulfing."""
        assert not is_engulfing(
            curr_open=100.0,
            curr_high=100.0,
            curr_low=100.0,
            curr_close=100.0,
            prev_open=100.0,
            prev_close=100.0,
            direction="LONG",
        )


# ── Pin bar rejection tests ─────────────────────────────────────────────────


class TestIsPinBarRejection:
    def test_bearish_pin_blocks_long(self) -> None:
        """Long upper wick > 2x body, body in lower 33% -> blocks LONG."""
        # body = 0.5 (100.0-100.5), upper wick = 4.5 (100.5-105.0) > 2*0.5
        # body_bot=100.0, lower_zone = 100.0 + 0.33*5.0 = 101.65 -> 100.0 < 101.65
        assert is_pin_bar_rejection(
            curr_open=100.5,
            curr_high=105.0,
            curr_low=100.0,
            curr_close=100.0,
            direction="LONG",
        )

    def test_bullish_pin_blocks_short(self) -> None:
        """Long lower wick > 2x body, body in upper 33% -> blocks SHORT."""
        # body = 0.5 (104.5-105.0), lower wick = 4.5 (100.0-104.5) > 2*0.5
        # body_top=105.0, upper_zone = 105.0 - 0.33*5.0 = 103.35 -> 105.0 > 103.35
        assert is_pin_bar_rejection(
            curr_open=104.5,
            curr_high=105.0,
            curr_low=100.0,
            curr_close=105.0,
            direction="SHORT",
        )

    def test_strong_body_not_pin_bar(self) -> None:
        """Candle with body > 60% of range is NOT a pin bar (valid breakout)."""
        # body = 4.0 (100-104), range = 5.0, upper wick = 1.0
        # upper wick 1.0 < 2 * 4.0 = 8.0 -> NOT pin bar
        assert not is_pin_bar_rejection(
            curr_open=100.0,
            curr_high=105.0,
            curr_low=100.0,
            curr_close=104.0,
            direction="LONG",
        )

    def test_pin_bar_does_not_block_wrong_direction(self) -> None:
        """Bearish pin bar should NOT block SHORT (only blocks LONG)."""
        assert not is_pin_bar_rejection(
            curr_open=100.5,
            curr_high=105.0,
            curr_low=100.0,
            curr_close=100.0,
            direction="SHORT",
        )

    def test_zero_range_returns_false(self) -> None:
        """Doji with no range is not a pin bar."""
        assert not is_pin_bar_rejection(
            curr_open=100.0,
            curr_high=100.0,
            curr_low=100.0,
            curr_close=100.0,
            direction="LONG",
        )


# ── Inside bar compression tests ────────────────────────────────────────────


class TestHasInsideBarCompression:
    def test_inside_bar_in_lookback(self) -> None:
        """Detect inside bar when bar's range is within prior bar's range."""
        highs = [105.0, 103.0, 104.0, 102.0]  # bar[1] inside bar[0]: 103<105, low check below
        lows = [100.0, 101.0, 99.0, 101.5]  # bar[1] inside bar[0]: 101>100
        # bar index 1: high=103 < 105 (prev high), low=101 > 100 (prev low) -> inside bar!
        assert has_inside_bar_compression(highs, lows, lookback=3)

    def test_no_inside_bar(self) -> None:
        """No compression when all bars expand range."""
        highs = [100.0, 101.0, 102.0, 103.0]
        lows = [99.0, 98.0, 97.0, 96.0]
        assert not has_inside_bar_compression(highs, lows, lookback=3)

    def test_inside_bar_at_end_of_lookback(self) -> None:
        """Inside bar at position -1 (last bar before breakout) is detected."""
        highs = [105.0, 106.0, 107.0, 106.5]
        lows = [100.0, 99.0, 98.0, 98.5]
        # bar[3]: 106.5 < 107.0 and 98.5 > 98.0 -> inside bar
        assert has_inside_bar_compression(highs, lows, lookback=3)

    def test_insufficient_data_returns_false(self) -> None:
        """Not enough bars for lookback -> False."""
        highs = [105.0, 103.0]
        lows = [100.0, 101.0]
        assert not has_inside_bar_compression(highs, lows, lookback=3)

    def test_exact_minimum_data(self) -> None:
        """Exactly lookback+1 bars should work."""
        highs = [105.0, 103.0, 104.0, 106.0]  # bar[1]: 103<105 inside
        lows = [100.0, 101.0, 99.0, 98.0]  # bar[1]: 101>100 inside
        assert has_inside_bar_compression(highs, lows, lookback=3)
