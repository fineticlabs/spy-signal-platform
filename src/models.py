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
