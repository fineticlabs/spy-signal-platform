"""Streaming (incremental) indicator computation using talipp.

Each class wraps a talipp indicator and provides a consistent interface:

- ``update(bar)``  — feed one completed :class:`~src.models.Bar`
- ``value``        — current indicator value(s); ``None`` if not yet ready
- ``ready``        — ``True`` once the look-back window is satisfied

These classes are safe to use inside an asyncio event loop because all talipp
operations are synchronous CPU work with no I/O.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

import structlog
from talipp.indicators import ATR, EMA, MACD, RSI
from talipp.ohlcv import OHLCV

if TYPE_CHECKING:
    from src.models import Bar

logger = structlog.get_logger(__name__)


class StreamingEMA:
    """Exponential Moving Average updated one bar at a time.

    Args:
        period: EMA look-back window (e.g. 9, 20, 50).
    """

    def __init__(self, period: int) -> None:
        self._period = period
        self._indicator: EMA = EMA(period=period)

    def update(self, bar: Bar) -> None:
        """Feed a completed bar into the EMA."""
        self._indicator.add(float(bar.close))

    @property
    def value(self) -> Decimal | None:
        """Current EMA, or ``None`` if not enough bars have been seen."""
        vals = self._indicator.output_values
        if not vals or vals[-1] is None:
            return None
        return Decimal(str(vals[-1]))

    @property
    def ready(self) -> bool:
        """``True`` once the look-back window is satisfied."""
        return self.value is not None

    def __repr__(self) -> str:
        return f"StreamingEMA(period={self._period}, value={self.value})"


class StreamingRSI:
    """Relative Strength Index updated one bar at a time.

    Args:
        period: RSI look-back window (default 14).
    """

    def __init__(self, period: int = 14) -> None:
        self._period = period
        self._indicator: RSI = RSI(period=period)

    def update(self, bar: Bar) -> None:
        """Feed a completed bar into the RSI."""
        self._indicator.add(float(bar.close))

    @property
    def value(self) -> Decimal | None:
        """Current RSI value (0-100), or ``None`` if not yet ready."""
        vals = self._indicator.output_values
        if not vals or vals[-1] is None:
            return None
        return Decimal(str(vals[-1]))

    @property
    def ready(self) -> bool:
        return self.value is not None

    def __repr__(self) -> str:
        return f"StreamingRSI(period={self._period}, value={self.value})"


class StreamingATR:
    """Average True Range updated one OHLCV bar at a time.

    Args:
        period: ATR smoothing window (default 14).
    """

    def __init__(self, period: int = 14) -> None:
        self._period = period
        self._indicator: ATR = ATR(period=period)

    def update(self, bar: Bar) -> None:
        """Feed a completed bar into the ATR."""
        ohlcv = OHLCV(
            open=float(bar.open),
            high=float(bar.high),
            low=float(bar.low),
            close=float(bar.close),
            volume=float(bar.volume),
        )
        self._indicator.add(ohlcv)

    @property
    def value(self) -> Decimal | None:
        """Current ATR, or ``None`` if not enough bars have been seen."""
        vals = self._indicator.output_values
        if not vals or vals[-1] is None:
            return None
        return Decimal(str(vals[-1]))

    @property
    def ready(self) -> bool:
        return self.value is not None

    def __repr__(self) -> str:
        return f"StreamingATR(period={self._period}, value={self.value})"


class MACDValue:
    """Container for a single MACD snapshot."""

    def __init__(self, macd: Decimal, signal: Decimal, histogram: Decimal) -> None:
        self.macd = macd
        self.signal = signal
        self.histogram = histogram

    def __repr__(self) -> str:
        return f"MACDValue(macd={self.macd}, signal={self.signal}, histogram={self.histogram})"


class StreamingMACD:
    """Moving Average Convergence/Divergence updated one bar at a time.

    Args:
        fast:   Fast EMA period (default 12).
        slow:   Slow EMA period (default 26).
        signal: Signal EMA period (default 9).
    """

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9) -> None:
        self._fast = fast
        self._slow = slow
        self._signal = signal
        self._indicator: MACD = MACD(
            fast_period=fast,
            slow_period=slow,
            signal_period=signal,
        )

    def update(self, bar: Bar) -> None:
        """Feed a completed bar into the MACD."""
        self._indicator.add(float(bar.close))

    @property
    def value(self) -> MACDValue | None:
        """Current MACD snapshot, or ``None`` if not yet ready."""
        vals = self._indicator.output_values
        if not vals or vals[-1] is None:
            return None
        v = vals[-1]
        if v.macd is None or v.signal is None or v.histogram is None:
            return None
        return MACDValue(
            macd=Decimal(str(v.macd)),
            signal=Decimal(str(v.signal)),
            histogram=Decimal(str(v.histogram)),
        )

    @property
    def ready(self) -> bool:
        return self.value is not None

    def __repr__(self) -> str:
        return f"StreamingMACD(fast={self._fast}, slow={self._slow}, signal={self._signal}, value={self.value})"
