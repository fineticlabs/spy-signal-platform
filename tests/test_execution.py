"""Tests for the Alpaca execution module."""

from __future__ import annotations

from datetime import time
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.execution.alpaca_executor import AlpacaExecutor
from src.models import Direction

# ── Helpers ──────────────────────────────────────────────────────────────────


def _mock_account(
    buying_power: str = "100000",
    equity: str = "100000",
    cash: str = "100000",
    account_number: str = "PA12345678",
    daytrade_count: int = 0,
    pattern_day_trader: bool = False,
) -> SimpleNamespace:
    """Create a fake Alpaca TradeAccount."""
    return SimpleNamespace(
        buying_power=buying_power,
        equity=equity,
        cash=cash,
        account_number=account_number,
        daytrade_count=daytrade_count,
        pattern_day_trader=pattern_day_trader,
    )


def _mock_position(
    symbol: str = "SPY",
    qty: str = "100",
    side: str = "long",
    avg_entry_price: str = "480.00",
    current_price: str = "482.00",
    unrealized_pl: str = "200.00",
) -> SimpleNamespace:
    """Create a fake Alpaca Position."""
    return SimpleNamespace(
        symbol=symbol,
        qty=qty,
        side=side,
        avg_entry_price=avg_entry_price,
        current_price=current_price,
        unrealized_pl=unrealized_pl,
    )


def _mock_order(order_id: str = "test-order-123") -> SimpleNamespace:
    """Create a fake Alpaca Order."""
    return SimpleNamespace(
        id=order_id,
        symbol="SPY",
        side="buy",
        qty="100",
        order_class="bracket",
        status="accepted",
    )


def _make_executor() -> AlpacaExecutor:
    """Create an AlpacaExecutor with a mocked TradingClient."""
    with patch("src.execution.alpaca_executor.TradingClient"):
        executor = AlpacaExecutor(
            api_key="test-key",
            secret_key="test-secret",  # noqa: S106 — test-only credential
            paper=True,
        )
    executor._client = MagicMock()
    return executor


# ── Test: submit_bracket_order ───────────────────────────────────────────────


class TestSubmitBracketOrder:
    def test_long_bracket_order_submitted(self) -> None:
        """A LONG signal submits a BUY bracket order."""
        executor = _make_executor()
        executor._client.get_account.return_value = _mock_account()
        executor._client.submit_order.return_value = _mock_order()

        with patch.object(executor, "_check_market_hours", return_value=True):
            order = executor.submit_bracket_order(
                symbol="SPY",
                direction=Direction.LONG,
                qty=100,
                stop_price=Decimal("478.00"),
                target_price=Decimal("486.00"),
            )

        assert order is not None
        assert order.id == "test-order-123"
        call_args = executor._client.submit_order.call_args[0][0]
        assert call_args.side.value == "buy"
        assert call_args.order_class.value == "bracket"
        assert call_args.symbol == "SPY"
        assert call_args.qty == 100

    def test_short_bracket_order_submitted(self) -> None:
        """A SHORT signal submits a SELL bracket order."""
        executor = _make_executor()
        executor._client.get_account.return_value = _mock_account()
        executor._client.submit_order.return_value = _mock_order()

        with patch.object(executor, "_check_market_hours", return_value=True):
            order = executor.submit_bracket_order(
                symbol="SPY",
                direction=Direction.SHORT,
                qty=50,
                stop_price=Decimal("485.00"),
                target_price=Decimal("475.00"),
            )

        assert order is not None
        call_args = executor._client.submit_order.call_args[0][0]
        assert call_args.side.value == "sell"

    def test_order_rejected_outside_market_hours(self) -> None:
        """Order is rejected when market is closed."""
        executor = _make_executor()

        with patch.object(executor, "_check_market_hours", return_value=False):
            order = executor.submit_bracket_order(
                symbol="SPY",
                direction=Direction.LONG,
                qty=100,
                stop_price=Decimal("478.00"),
                target_price=Decimal("486.00"),
            )

        assert order is None
        executor._client.submit_order.assert_not_called()

    def test_order_rejected_insufficient_buying_power(self) -> None:
        """Order is rejected when buying power is too low."""
        executor = _make_executor()
        executor._client.get_account.return_value = _mock_account(buying_power="500")

        with patch.object(executor, "_check_market_hours", return_value=True):
            order = executor.submit_bracket_order(
                symbol="SPY",
                direction=Direction.LONG,
                qty=100,
                stop_price=Decimal("478.00"),
                target_price=Decimal("486.00"),
            )

        assert order is None
        executor._client.submit_order.assert_not_called()

    def test_order_rejected_not_paper_account(self) -> None:
        """Order is rejected when paper=True but account is not paper."""
        executor = _make_executor()
        executor._client.get_account.return_value = _mock_account(
            account_number="LV12345678"  # not PA prefix
        )

        with (
            patch.object(executor, "_check_market_hours", return_value=True),
            patch.object(executor, "_cap_to_buying_power", return_value=100),
        ):
            order = executor.submit_bracket_order(
                symbol="SPY",
                direction=Direction.LONG,
                qty=100,
                stop_price=Decimal("478.00"),
                target_price=Decimal("486.00"),
            )

        assert order is None
        executor._client.submit_order.assert_not_called()

    def test_api_error_returns_none(self) -> None:
        """API errors are caught and return None."""
        from alpaca.common.exceptions import APIError

        executor = _make_executor()
        executor._client.get_account.return_value = _mock_account()
        executor._client.submit_order.side_effect = APIError("Insufficient qty")

        with patch.object(executor, "_check_market_hours", return_value=True):
            order = executor.submit_bracket_order(
                symbol="SPY",
                direction=Direction.LONG,
                qty=100,
                stop_price=Decimal("478.00"),
                target_price=Decimal("486.00"),
            )

        assert order is None

    def test_bracket_order_includes_stop_and_target(self) -> None:
        """Verify take_profit and stop_loss are set on the order request."""
        executor = _make_executor()
        executor._client.get_account.return_value = _mock_account()
        executor._client.submit_order.return_value = _mock_order()

        with patch.object(executor, "_check_market_hours", return_value=True):
            executor.submit_bracket_order(
                symbol="TSLA",
                direction=Direction.LONG,
                qty=25,
                stop_price=Decimal("170.00"),
                target_price=Decimal("180.00"),
            )

        call_args = executor._client.submit_order.call_args[0][0]
        assert call_args.take_profit is not None
        assert call_args.take_profit.limit_price == 180.0
        assert call_args.stop_loss is not None
        assert call_args.stop_loss.stop_price == 170.0


# ── Test: flatten_all_positions ──────────────────────────────────────────────


class TestFlattenAllPositions:
    def test_flatten_closes_all_positions(self) -> None:
        """All open positions are closed."""
        executor = _make_executor()
        executor._client.get_all_positions.return_value = [
            _mock_position(symbol="SPY", qty="100"),
            _mock_position(symbol="TSLA", qty="50"),
        ]
        executor._client.close_position.return_value = _mock_order()

        closed = executor.flatten_all_positions()

        assert closed == 2
        assert executor._client.close_position.call_count == 2

    def test_flatten_no_positions(self) -> None:
        """No action taken when there are no positions."""
        executor = _make_executor()
        executor._client.get_all_positions.return_value = []

        closed = executor.flatten_all_positions()

        assert closed == 0
        executor._client.close_position.assert_not_called()

    def test_flatten_handles_partial_failure(self) -> None:
        """If one position fails to close, others still close."""
        from alpaca.common.exceptions import APIError

        executor = _make_executor()
        executor._client.get_all_positions.return_value = [
            _mock_position(symbol="SPY"),
            _mock_position(symbol="TSLA"),
        ]
        executor._client.close_position.side_effect = [
            _mock_order(),
            APIError("Position not found"),
        ]

        closed = executor.flatten_all_positions()

        assert closed == 1


# ── Test: cancel_open_orders ─────────────────────────────────────────────────


class TestCancelOpenOrders:
    def test_cancel_open_orders(self) -> None:
        """All open orders are cancelled."""
        executor = _make_executor()
        executor._client.get_orders.return_value = [_mock_order(), _mock_order()]

        cancelled = executor.cancel_open_orders()

        assert cancelled == 2
        executor._client.cancel_orders.assert_called_once()

    def test_cancel_no_orders(self) -> None:
        """No action when there are no open orders."""
        executor = _make_executor()
        executor._client.get_orders.return_value = []

        cancelled = executor.cancel_open_orders()

        assert cancelled == 0
        executor._client.cancel_orders.assert_not_called()


# ── Test: safety checks ─────────────────────────────────────────────────────


class TestSafetyChecks:
    def test_market_hours_during_session(self) -> None:
        """Market hours check passes during regular session."""
        executor = _make_executor()
        with patch.object(executor, "_now_et_time", return_value=time(10, 30)):
            assert executor._check_market_hours() is True

    def test_market_hours_before_open(self) -> None:
        """Market hours check fails before open."""
        executor = _make_executor()
        with patch.object(executor, "_now_et_time", return_value=time(9, 0)):
            assert executor._check_market_hours() is False

    def test_market_hours_after_close(self) -> None:
        """Market hours check fails after close."""
        executor = _make_executor()
        with patch.object(executor, "_now_et_time", return_value=time(16, 1)):
            assert executor._check_market_hours() is False

    def test_buying_power_sufficient_no_downsize(self) -> None:
        """When buying power is ample, qty is returned unchanged."""
        executor = _make_executor()
        executor._client.get_account.return_value = _mock_account(buying_power="200000")

        # 100 shares * $480 = $48K, half BP = $100K → fits
        result = executor._cap_to_buying_power("SPY", qty=100, entry_price=Decimal("480.00"))
        assert result == 100

    def test_buying_power_downsizes_to_half(self) -> None:
        """When qty exceeds 50% BP, it's capped to half of buying power."""
        executor = _make_executor()
        executor._client.get_account.return_value = _mock_account(buying_power="100000")

        # 873 shares * $645 = $563K >> $50K (half of $100K) → downsized
        result = executor._cap_to_buying_power("SPY", qty=873, entry_price=Decimal("645.00"))
        # max_qty = floor(50000 / 645) = 77
        assert result == 77

    def test_buying_power_below_minimum_returns_zero(self) -> None:
        """Below _MIN_BUYING_POWER ($1000), returns 0."""
        executor = _make_executor()
        executor._client.get_account.return_value = _mock_account(buying_power="500")

        result = executor._cap_to_buying_power("SPY", qty=1, entry_price=Decimal("50.00"))
        assert result == 0

    def test_paper_account_assertion_passes(self) -> None:
        """Paper assertion passes for PA-prefixed accounts."""
        executor = _make_executor()
        executor._client.get_account.return_value = _mock_account(account_number="PA12345678")

        assert executor._assert_paper_account() is True

    def test_paper_account_assertion_fails_live(self) -> None:
        """Paper assertion fails for non-PA accounts."""
        executor = _make_executor()
        executor._client.get_account.return_value = _mock_account(account_number="LV98765432")

        assert executor._assert_paper_account() is False

    def test_is_flatten_time_before(self) -> None:
        """is_flatten_time returns False before 3:55 PM ET."""
        executor = _make_executor()
        with patch.object(executor, "_now_et_time", return_value=time(15, 50)):
            assert executor.is_flatten_time() is False

    def test_is_flatten_time_at(self) -> None:
        """is_flatten_time returns True at 3:55 PM ET."""
        executor = _make_executor()
        with patch.object(executor, "_now_et_time", return_value=time(15, 55)):
            assert executor.is_flatten_time() is True


# ── Test: config integration ─────────────────────────────────────────────────


class TestExecutionModeConfig:
    def test_alerts_only_is_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Default execution_mode is alerts_only."""
        monkeypatch.delenv("EXECUTION_MODE", raising=False)
        monkeypatch.delenv("TRADING_MODE", raising=False)
        monkeypatch.delenv("DB_PATH", raising=False)
        from src.config import AppSettings

        settings = AppSettings(_env_file=None)  # type: ignore[call-arg]
        assert settings.execution_mode == "alerts_only"

    def test_paper_trade_mode_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """execution_mode can be set to paper_trade via env."""
        monkeypatch.setenv("EXECUTION_MODE", "paper_trade")
        from src.config import AppSettings

        settings = AppSettings(_env_file=None)  # type: ignore[call-arg]
        assert settings.execution_mode == "paper_trade"

    def test_invalid_execution_mode_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Invalid execution_mode raises ValueError."""
        monkeypatch.setenv("EXECUTION_MODE", "yolo_trade")
        from src.config import AppSettings

        with pytest.raises(ValueError, match="execution_mode"):
            AppSettings(_env_file=None)  # type: ignore[call-arg]
