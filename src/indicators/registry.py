"""IndicatorRegistry — manages a collection of streaming indicators.

Usage::

    registry = IndicatorRegistry()
    registry.register("ema9", StreamingEMA(9))
    registry.register("rsi14", StreamingRSI(14))
    registry.register("atr14", StreamingATR(14))

    # For each incoming bar:
    registry.update_all(bar)
    snapshot = registry.get_snapshot()
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import structlog

from src.indicators.streaming import MACDValue
from src.models import IndicatorSnapshot

if TYPE_CHECKING:
    from src.models import Bar

logger = structlog.get_logger(__name__)


@runtime_checkable
class StreamingIndicator(Protocol):
    """Minimal interface required by :class:`IndicatorRegistry`."""

    def update(self, bar: Bar) -> None: ...

    @property
    def ready(self) -> bool: ...


class IndicatorRegistry:
    """Container that keeps a named set of streaming indicators in sync.

    Calling :meth:`update_all` feeds every registered indicator a new bar in
    registration order.  :meth:`get_snapshot` returns an
    :class:`~src.models.IndicatorSnapshot` with all current values.
    """

    def __init__(self) -> None:
        self._indicators: dict[str, StreamingIndicator] = {}

    def register(self, name: str, indicator: StreamingIndicator) -> None:
        """Add *indicator* under *name*.

        If *name* already exists the previous indicator is replaced and a
        warning is logged.

        Args:
            name:      Unique identifier (used as the snapshot field key).
            indicator: Any object satisfying the :class:`StreamingIndicator`
                       protocol.
        """
        if name in self._indicators:
            logger.warning("indicator_replaced", name=name)
        self._indicators[name] = indicator
        logger.debug("indicator_registered", name=name, indicator=repr(indicator))

    def update_all(self, bar: Bar) -> None:
        """Push *bar* into every registered indicator.

        Args:
            bar: A fully completed :class:`~src.models.Bar`.
        """
        for name, indicator in self._indicators.items():
            try:
                indicator.update(bar)
            except Exception as exc:
                logger.error("indicator_update_failed", name=name, error=str(exc))

    def get_snapshot(self) -> IndicatorSnapshot:
        """Return an :class:`~src.models.IndicatorSnapshot` of all current values.

        Only indicators registered under the canonical names defined in
        :class:`~src.models.IndicatorSnapshot` are included.  Unknown names are
        logged and skipped.

        Returns:
            A frozen snapshot with ``None`` for any indicator not yet ready.
        """
        values: dict[str, Decimal | None] = {}
        _known = {
            "ema9",
            "ema20",
            "ema50",
            "rsi",
            "macd",
            "macd_signal",
            "macd_histogram",
            "bb_upper",
            "bb_middle",
            "bb_lower",
            "atr",
            "vwap",
        }

        for name, indicator in self._indicators.items():
            if name not in _known:
                logger.debug("snapshot_unknown_indicator", name=name)
                continue
            if not indicator.ready:
                values[name] = None
                continue

            raw = getattr(indicator, "value", None)
            # StreamingMACD returns a MACDValue object — expand it
            if isinstance(raw, MACDValue):
                values["macd"] = raw.macd
                values["macd_signal"] = raw.signal
                values["macd_histogram"] = raw.histogram
            else:
                values[name] = Decimal(str(raw)) if raw is not None else None

        return IndicatorSnapshot(**values)

    @property
    def names(self) -> list[str]:
        """Registered indicator names in registration order."""
        return list(self._indicators.keys())

    def __len__(self) -> int:
        return len(self._indicators)

    def __repr__(self) -> str:
        return f"IndicatorRegistry(indicators={self.names})"
