"""Abstract base class for all trading strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.models import Bar, IndicatorSnapshot, LevelSnapshot, Signal
    from src.strategies.regime import RegimeDetector


class Strategy(ABC):
    """Abstract interface that every trading strategy must implement.

    The :meth:`evaluate` method is called once per completed bar.  It returns
    a :class:`~src.models.Signal` when entry conditions are met, or ``None``
    when the strategy is waiting or filtered out.
    """

    @abstractmethod
    def evaluate(
        self,
        bar: Bar,
        indicators: IndicatorSnapshot,
        levels: LevelSnapshot,
        regime: RegimeDetector,
    ) -> Signal | None:
        """Evaluate the current bar and return a signal if conditions are met.

        Args:
            bar:        The most recently completed bar.
            indicators: Point-in-time indicator values.
            levels:     Point-in-time key price level values.
            regime:     Current market regime detector.

        Returns:
            A :class:`~src.models.Signal` if all entry conditions are satisfied,
            otherwise ``None``.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable strategy name."""

    @abstractmethod
    def required_indicators(self) -> list[str]:
        """Indicator names this strategy depends on (keys in IndicatorSnapshot)."""
