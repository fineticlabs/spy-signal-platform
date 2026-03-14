from __future__ import annotations

from datetime import datetime
from decimal import Decimal
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


class Direction(StrEnum):
    LONG = "LONG"
    SHORT = "SHORT"


class Regime(StrEnum):
    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    RANGING = "RANGING"
    CHOPPY = "CHOPPY"


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

    # First-5-min relative volume
    rvol: Decimal | None = Field(
        default=None,
        description="First-5-min RVOL (today's opening volume / 20-day avg)",
    )

    # Prior-day volume profile levels
    vp_poc: Decimal | None = Field(default=None, description="Prior-day Volume Profile POC")
    vp_vah: Decimal | None = Field(default=None, description="Prior-day Value Area High")
    vp_val: Decimal | None = Field(default=None, description="Prior-day Value Area Low")
    vp_hvn: Decimal | None = Field(default=None, description="Nearest prior-day HVN")
    vp_lvn: Decimal | None = Field(default=None, description="Nearest prior-day LVN")

    # VIX term structure
    vix_term_ratio: Decimal | None = Field(
        default=None,
        description="VIX/VIX3M ratio (prior day): <0.85 contango, >1.0 backwardation",
    )


class Signal(BaseModel):
    """Trading signal produced by a strategy when entry conditions are met."""

    symbol: str = Field(default="SPY", description="Ticker symbol that generated this signal")
    direction: Direction = Field(..., description="Trade direction: LONG or SHORT")
    strategy_name: str = Field(..., description="Name of the strategy that generated this signal")
    entry_price: Decimal = Field(..., description="Suggested entry price")
    stop_price: Decimal = Field(..., description="Initial stop-loss price")
    target_price: Decimal = Field(..., description="Initial profit target price")
    risk_reward_ratio: Decimal = Field(
        ..., description="Reward-to-risk ratio (target/stop distance)"
    )
    confidence_score: int = Field(
        ..., ge=1, le=5, description="Signal confidence from 1 (low) to 5 (high)"
    )
    reason: str = Field(..., description="Human-readable explanation of why the signal fired")
    timeframe: TimeFrame = Field(..., description="Bar timeframe that triggered the signal")
    regime: Regime = Field(..., description="Market regime at signal time")
    vix: Decimal | None = Field(default=None, description="VIX level at signal time")
    adx: Decimal | None = Field(default=None, description="ADX value at signal time")
    indicators_snapshot: IndicatorSnapshot | None = Field(
        default=None, description="Full indicator state at signal time"
    )
    levels_snapshot: LevelSnapshot | None = Field(
        default=None, description="Full level state at signal time"
    )
    timestamp: datetime = Field(..., description="Timestamp of the bar that triggered the signal")
    tags: list[str] = Field(default_factory=list, description="Quality tags: ENGULFING, COMPRESSED")

    model_config = {"frozen": True}


class TradeResult(BaseModel):
    """Outcome of a completed trade, used for risk state tracking."""

    signal: Signal = Field(..., description="The signal that initiated the trade")
    pnl: Decimal = Field(..., description="Realized P&L in dollars (negative = loss)")
    timestamp: datetime = Field(..., description="Time the trade was closed (UTC)")

    model_config = {"frozen": True}


class RiskDecision(BaseModel):
    """Result of a pre-trade risk check from :class:`~src.risk.manager.RiskManager`."""

    approved: bool = Field(..., description="True if the trade is approved to proceed")
    reason: str = Field(..., description="Human-readable explanation of the decision")
    position_size: int = Field(
        default=0, ge=0, description="Approved position size in shares (0 if rejected)"
    )

    model_config = {"frozen": True}
