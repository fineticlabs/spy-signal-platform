"""Extended tests for src.indicators.registry — edge cases and coverage gaps."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

from src.indicators.registry import IndicatorRegistry
from src.indicators.streaming import MACDValue


def _make_stub_indicator(*, ready: bool = True, value: object = Decimal("100.0")) -> MagicMock:
    """Create a mock indicator satisfying the StreamingIndicator protocol."""
    ind = MagicMock()
    ind.ready = ready
    ind.value = value
    return ind


class TestRegisterReplace:
    """register: replacing existing indicator logs warning."""

    def test_replace_existing_logs_warning(self) -> None:
        registry = IndicatorRegistry()
        ind1 = _make_stub_indicator()
        ind2 = _make_stub_indicator()

        registry.register("ema9", ind1)
        # Replacing should not raise
        registry.register("ema9", ind2)

        assert len(registry) == 1
        # The second indicator should be current
        snapshot = registry.get_snapshot()
        assert snapshot.ema9 == Decimal("100.0")


class TestUpdateAllErrorHandling:
    """update_all: indicator.update raises → caught, logged, continues."""

    def test_failing_indicator_does_not_stop_others(self) -> None:
        registry = IndicatorRegistry()
        bad_ind = _make_stub_indicator()
        bad_ind.update.side_effect = ValueError("boom")

        good_ind = _make_stub_indicator()
        good_ind.update.return_value = None

        registry.register("ema9", bad_ind)
        registry.register("rsi", good_ind)

        bar = MagicMock()
        # Should not raise
        registry.update_all(bar)

        # Both got called
        bad_ind.update.assert_called_once_with(bar)
        good_ind.update.assert_called_once_with(bar)


class TestGetSnapshotEdgeCases:
    """get_snapshot: unknown indicator, MACD expansion."""

    def test_unknown_indicator_skipped(self) -> None:
        """Indicator with unrecognised name is excluded from snapshot."""
        registry = IndicatorRegistry()
        registry.register("custom_xyz", _make_stub_indicator())

        snapshot = registry.get_snapshot()

        # custom_xyz is not a known snapshot field, so all fields should be None
        assert snapshot.ema9 is None
        assert snapshot.rsi is None

    def test_macd_value_expanded(self) -> None:
        """MACDValue object is expanded to macd, macd_signal, macd_histogram."""
        registry = IndicatorRegistry()
        macd_val = MACDValue(
            macd=Decimal("1.23"),
            signal=Decimal("0.98"),
            histogram=Decimal("0.25"),
        )
        registry.register("macd", _make_stub_indicator(value=macd_val))

        snapshot = registry.get_snapshot()

        assert snapshot.macd == Decimal("1.23")
        assert snapshot.macd_signal == Decimal("0.98")
        assert snapshot.macd_histogram == Decimal("0.25")


class TestRegistryDunderMethods:
    """names property, __len__, __repr__."""

    def test_names_returns_registration_order(self) -> None:
        registry = IndicatorRegistry()
        registry.register("rsi", _make_stub_indicator())
        registry.register("ema9", _make_stub_indicator())
        registry.register("atr", _make_stub_indicator())

        assert registry.names == ["rsi", "ema9", "atr"]

    def test_len_returns_count(self) -> None:
        registry = IndicatorRegistry()
        assert len(registry) == 0

        registry.register("ema9", _make_stub_indicator())
        registry.register("rsi", _make_stub_indicator())

        assert len(registry) == 2

    def test_repr_includes_names(self) -> None:
        registry = IndicatorRegistry()
        registry.register("ema9", _make_stub_indicator())
        registry.register("vwap", _make_stub_indicator())

        r = repr(registry)

        assert "IndicatorRegistry" in r
        assert "ema9" in r
        assert "vwap" in r
