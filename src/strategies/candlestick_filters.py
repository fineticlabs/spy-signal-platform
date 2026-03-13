"""Candlestick quality filters for ORB breakout confirmation.

Three filters based on "The Candlestick Trading Bible":

1. **Engulfing bar** (booster): breakout candle body engulfs prior candle body
   and body > 60% of total range. Adds +1 confidence.
2. **Pin bar rejection** (blocker): breakout candle has a long wick > 2x body
   against the trade direction. Blocks the trade entirely.
3. **Inside bar compression** (booster): any of the 3 candles before the breakout
   is an inside bar (range fully within prior candle). Adds +1 confidence.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

# Engulfing: body must be > this fraction of total candle range
_ENGULFING_BODY_RATIO = 0.60

# Pin bar: wick must be > this multiple of body size
_PIN_BAR_WICK_MULT = 2.0

# Pin bar: body must be in this fraction of the range (upper/lower third)
_PIN_BAR_BODY_ZONE = 1.0 / 3.0


def is_engulfing(
    curr_open: float,
    curr_high: float,
    curr_low: float,
    curr_close: float,
    prev_open: float,
    prev_close: float,
    direction: str,
) -> bool:
    """Check if the current candle is an engulfing bar confirming the breakout.

    For LONG: candle closes green (close > open), body engulfs prior body,
    and body > 60% of range.
    For SHORT: candle closes red (close < open), body engulfs prior body,
    and body > 60% of range.

    Args:
        curr_open:  Current bar open.
        curr_high:  Current bar high.
        curr_low:   Current bar low.
        curr_close: Current bar close.
        prev_open:  Previous bar open.
        prev_close: Previous bar close.
        direction:  "LONG" or "SHORT".

    Returns:
        True if the breakout candle is an engulfing bar.
    """
    curr_body = abs(curr_close - curr_open)
    curr_range = curr_high - curr_low
    if curr_range <= 0:
        return False

    # Body must be > 60% of total range (strong conviction candle)
    if curr_body / curr_range < _ENGULFING_BODY_RATIO:
        return False

    prev_body_top = max(prev_open, prev_close)
    prev_body_bot = min(prev_open, prev_close)
    curr_body_top = max(curr_open, curr_close)
    curr_body_bot = min(curr_open, curr_close)

    # Current body must completely cover the previous body
    if curr_body_top < prev_body_top or curr_body_bot > prev_body_bot:
        return False

    if direction == "LONG":
        return curr_close > curr_open  # green candle
    return curr_close < curr_open  # red candle for SHORT


def is_pin_bar_rejection(
    curr_open: float,
    curr_high: float,
    curr_low: float,
    curr_close: float,
    direction: str,
) -> bool:
    """Check if the breakout candle is a pin bar rejecting the breakout direction.

    For LONG breakouts: a bearish pin bar has upper wick > 2x body AND body
    in the lower 33% of range (market rejected the highs).
    For SHORT breakouts: a bullish pin bar has lower wick > 2x body AND body
    in the upper 33% of range (market rejected the lows).

    Args:
        curr_open:  Current bar open.
        curr_high:  Current bar high.
        curr_low:   Current bar low.
        curr_close: Current bar close.
        direction:  "LONG" or "SHORT".

    Returns:
        True if the candle is a pin bar that should BLOCK the trade.
    """
    curr_range = curr_high - curr_low
    if curr_range <= 0:
        return False

    body = abs(curr_close - curr_open)
    body_top = max(curr_open, curr_close)
    body_bot = min(curr_open, curr_close)
    upper_wick = curr_high - body_top
    lower_wick = body_bot - curr_low

    # Dojis (zero body) are ambiguous — only reject if the opposing wick
    # dominates (>= 70% of range), otherwise allow the trade through.
    if body <= 0:
        if direction == "LONG":
            return upper_wick > 0.7 * curr_range
        return lower_wick > 0.7 * curr_range

    if direction == "LONG":
        # Bearish pin: long upper wick poking above ORB high, small body near low
        return (
            upper_wick > _PIN_BAR_WICK_MULT * body
            and body_bot < curr_low + _PIN_BAR_BODY_ZONE * curr_range
        )

    # SHORT: bullish pin: long lower wick poking below ORB low, small body near high
    return (
        lower_wick > _PIN_BAR_WICK_MULT * body
        and body_top > curr_high - _PIN_BAR_BODY_ZONE * curr_range
    )


def has_inside_bar_compression(
    highs: list[float] | np.ndarray,
    lows: list[float] | np.ndarray,
    lookback: int = 3,
) -> bool:
    """Check if any of the last ``lookback`` candles before the breakout is an inside bar.

    An inside bar has its high below the previous candle's high AND its low
    above the previous candle's low (range fully contained). This compression
    pattern indicates energy building up before the breakout.

    Args:
        highs: Array of high prices ending at the bar BEFORE the breakout candle.
               Must have at least ``lookback + 1`` elements.
        lows:  Array of low prices ending at the bar BEFORE the breakout candle.
               Must have at least ``lookback + 1`` elements.
        lookback: Number of candles to check (default 3).

    Returns:
        True if inside bar compression detected.
    """
    n = len(highs)
    if n < lookback + 1:
        return False

    return any(highs[i] < highs[i - 1] and lows[i] > lows[i - 1] for i in range(n - lookback, n))
