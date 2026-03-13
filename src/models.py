from __future__ import annotations

from datetime import datetime  # noqa: TCH003
from decimal import Decimal  # noqa: TCH003
from enum import StrEnum

import structlog
from pydantic import BaseModel, Field, field_validator

logger = structlog.get_logger(__name__)


class TimeFrame(StrEnum):
    ONE_MIN = "1Min"
    FIVE_MIN = "5Min"
    FIFTEEN_MIN = "15Min"
    THIRTY_MIN = "30Min"
    DAILY = "1Day"


class Bar(BaseModel):
    """OHLCV bar with VWAP for a single symbol and timeframe."""

    symbol: str = Field(..., description="Ticker symbol, e.g. 'SPY'")
    timeframe: TimeFrame = Field(default=TimeFrame.ONE_MIN, description="Bar timeframe")
    timestamp: datetime = Field(..., description="Bar open time, timezone-aware UTC")
    open: Decimal = Field(..., description="Open price")
    high: Decimal = Field(..., description="High price")
    low: Decimal = Field(..., description="Low price")
    close: Decimal = Field(..., description="Close price")
    volume: int = Field(..., ge=0, description="Volume in shares")
    vwap: Decimal = Field(..., description="Volume-weighted average price")

    @field_validator("timestamp")
    @classmethod
    def must_be_timezone_aware(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware (UTC)")
        return v

    @field_validator("high")
    @classmethod
    def high_gte_low(cls, v: Decimal, info: object) -> Decimal:
        # info.data available in pydantic v2
        data = getattr(info, "data", {})
        low = data.get("low")
        if low is not None and v < low:
            raise ValueError(f"high ({v}) must be >= low ({low})")
        return v

    @field_validator("open", "high", "low", "close", "vwap")
    @classmethod
    def must_be_positive(cls, v: Decimal) -> Decimal:
        if v <= 0:
            raise ValueError(f"price field must be positive, got {v}")
        return v

    model_config = {"frozen": True}


class IndicatorSnapshot(BaseModel):
    """Point-in-time values for all active indicators.

    All fields are ``Optional[Decimal]``; a ``None`` value means the indicator
    has not yet accumulated enough bars to produce a result.
    """

    ema9: Decimal | None = Field(default=None, description="EMA(9) of close price")
    ema20: Decimal | None = Field(default=None, description="EMA(20) of close price")
    ema50: Decimal | None = Field(default=None, description="EMA(50) of close price")
    rsi: Decimal | None = Field(default=None, description="RSI(14) oscillator (0-100)")
    macd: Decimal | None = Field(default=None, description="MACD line (fast EMA - slow EMA)")
    macd_signal: Decimal | None = Field(default=None, description="MACD signal line")
    macd_histogram: Decimal | None = Field(default=None, description="MACD histogram")
    bb_upper: Decimal | None = Field(default=None, description="Bollinger upper band")
    bb_middle: Decimal | None = Field(default=None, description="Bollinger middle (SMA)")
    bb_lower: Decimal | None = Field(default=None, description="Bollinger lower band")
    atr: Decimal | None = Field(default=None, description="ATR(14) volatility measure")
    vwap: Decimal | None = Field(default=None, description="Session VWAP")


class LevelSnapshot(BaseModel):
    """Point-in-time snapshot of all key price levels.

    All price fields are ``Optional[Decimal]``; ``None`` means the level has
    not yet been established (e.g. ORB before 9:35 ET).
    All boolean flags default to ``False``.
    """

    # Opening Range Breakout — 5-min
    orb_high: Decimal | None = Field(default=None, description="5-min ORB high")
    orb_low: Decimal | None = Field(default=None, description="5-min ORB low")
    orb_midpoint: Decimal | None = Field(default=None, description="5-min ORB midpoint")
    orb_range: Decimal | None = Field(default=None, description="5-min ORB width")
    orb_complete: bool = Field(default=False, description="True once 9:35 ET passed")

    # Opening Range Breakout — 15-min
    orb15_high: Decimal | None = Field(default=None, description="15-min ORB high")
    orb15_low: Decimal | None = Field(default=None, description="15-min ORB low")
    orb15_complete: bool = Field(default=False, description="True once 9:45 ET passed")

    # Session VWAP + bands
    vwap: Decimal | None = Field(default=None, description="Session VWAP")
    vwap_upper_1: Decimal | None = Field(default=None, description="VWAP + 1-sigma")
    vwap_lower_1: Decimal | None = Field(default=None, description="VWAP - 1-sigma")
    vwap_upper_2: Decimal | None = Field(default=None, description="VWAP + 2-sigma")
    vwap_lower_2: Decimal | None = Field(default=None, description="VWAP - 2-sigma")

    # Dynamic intraday levels
    high_of_day: Decimal | None = Field(default=None, description="Session high so far")
    low_of_day: Decimal | None = Field(default=None, description="Session low so far")
    last_price: Decimal | None = Field(default=None, description="Most recent close")

    # Previous day levels
    prev_day_high: Decimal | None = Field(default=None, description="PDH")
    prev_day_low: Decimal | None = Field(default=None, description="PDL")
    prev_day_close: Decimal | None = Field(default=None, description="PDC")

    # Premarket levels
    premarket_high: Decimal | None = Field(default=None, description="Premarket high")
    premarket_low: Decimal | None = Field(default=None, description="Premarket low")
