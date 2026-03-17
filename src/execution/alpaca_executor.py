"""Alpaca bracket-order executor for paper and live trading.

Places market-entry bracket orders (entry + stop-loss + take-profit) via
the alpaca-py ``TradingClient``.  Includes safety checks for market hours,
buying power, and paper-mode assertion.

Usage
-----
The executor is wired into the main signal loop in ``src/main.py``.  When
``execution_mode`` is ``paper_trade`` or ``live_trade``, approved signals
are routed here *in addition to* the Telegram alert.

The executor is **not** responsible for risk management — that is handled
upstream by :class:`~src.risk.manager.RiskManager`.
"""

from __future__ import annotations

from datetime import datetime, time
from decimal import Decimal
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

import structlog
from alpaca.common.exceptions import APIError
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderClass, OrderSide, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import (
    ClosePositionRequest,
    GetOrdersRequest,
    MarketOrderRequest,
    StopLossRequest,
    TakeProfitRequest,
)

from src.models import Direction

if TYPE_CHECKING:
    from alpaca.trading.models import Order, Position, TradeAccount

logger = structlog.get_logger(__name__)

_ET = ZoneInfo("America/New_York")

# Market hours in ET
_MARKET_OPEN = time(9, 30)
_MARKET_CLOSE = time(16, 0)
_FLATTEN_TIME = time(15, 55)

# Minimum buying power required to place an order (safety buffer)
_MIN_BUYING_POWER = Decimal("1000")


class AlpacaExecutor:
    """Places bracket orders on Alpaca and manages EOD flattening.

    Args:
        api_key:    Alpaca API key.
        secret_key: Alpaca secret key.
        paper:      If ``True``, use the paper trading endpoint.  The executor
                    will refuse to operate if ``paper=False`` and the account
                    is not a live account (safety check).
    """

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        paper: bool = True,
    ) -> None:
        self._paper = paper
        self._client = TradingClient(
            api_key=api_key,
            secret_key=secret_key,
            paper=paper,
        )
        logger.info("executor_initialized", paper=paper)

    # ── public interface ──────────────────────────────────────────────────────

    def submit_bracket_order(
        self,
        symbol: str,
        direction: Direction,
        qty: int,
        stop_price: Decimal,
        target_price: Decimal,
    ) -> Order | None:
        """Submit a bracket order (market entry + stop-loss + take-profit).

        Args:
            symbol:       Ticker symbol (e.g. ``"SPY"``).
            direction:    ``Direction.LONG`` or ``Direction.SHORT``.
            qty:          Number of shares.
            stop_price:   Stop-loss price.
            target_price: Take-profit limit price.

        Returns:
            The parent :class:`Order` on success, or ``None`` if a safety
            check failed or the API returned an error.
        """
        # ── safety checks ─────────────────────────────────────────────────
        if not self._check_market_hours():
            logger.warning("order_rejected_market_closed", symbol=symbol)
            return None

        if not self._check_buying_power(qty, stop_price):
            logger.warning("order_rejected_buying_power", symbol=symbol, qty=qty)
            return None

        if self._paper and not self._assert_paper_account():
            logger.error("order_rejected_not_paper", symbol=symbol)
            return None

        # ── build bracket order ───────────────────────────────────────────
        side = OrderSide.BUY if direction == Direction.LONG else OrderSide.SELL
        order_data = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=side,
            time_in_force=TimeInForce.DAY,
            order_class=OrderClass.BRACKET,
            take_profit=TakeProfitRequest(limit_price=float(target_price)),
            stop_loss=StopLossRequest(stop_price=float(stop_price)),
        )

        try:
            order: Order = self._client.submit_order(order_data)
            logger.info(
                "bracket_order_submitted",
                symbol=symbol,
                direction=str(direction),
                qty=qty,
                order_id=str(order.id),
                stop=str(stop_price),
                target=str(target_price),
            )
            return order
        except APIError as exc:
            logger.error(
                "bracket_order_failed",
                symbol=symbol,
                direction=str(direction),
                qty=qty,
                error=str(exc),
            )
            return None

    def flatten_all_positions(self) -> int:
        """Close all open positions (EOD flatten).

        Returns:
            Number of positions that were closed (or attempted).
        """
        try:
            positions = self.get_positions()
        except APIError as exc:
            logger.error("flatten_get_positions_failed", error=str(exc))
            return 0

        if not positions:
            logger.info("flatten_no_positions")
            return 0

        closed = 0
        for pos in positions:
            try:
                self._client.close_position(
                    symbol_or_asset_id=pos.symbol,
                    close_options=ClosePositionRequest(qty=str(abs(int(pos.qty)))),
                )
                logger.info(
                    "position_closed",
                    symbol=pos.symbol,
                    qty=str(pos.qty),
                    side=pos.side,
                )
                closed += 1
            except APIError as exc:
                logger.error(
                    "position_close_failed",
                    symbol=pos.symbol,
                    error=str(exc),
                )
        return closed

    def cancel_open_orders(self) -> int:
        """Cancel all open orders.

        Returns:
            Number of orders cancelled.
        """
        try:
            orders = self._client.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.OPEN))
            if not orders:
                return 0
            self._client.cancel_orders()
            logger.info("open_orders_cancelled", count=len(orders))
            return len(orders)
        except APIError as exc:
            logger.error("cancel_orders_failed", error=str(exc))
            return 0

    def get_account(self) -> TradeAccount:
        """Return the current Alpaca account state."""
        return self._client.get_account()

    def get_positions(self) -> list[Position]:
        """Return all open positions."""
        return self._client.get_all_positions()

    def is_flatten_time(self) -> bool:
        """Return ``True`` if current ET time is at or past 3:55 PM."""
        return self._now_et_time() >= _FLATTEN_TIME

    # ── safety checks ─────────────────────────────────────────────────────────

    def _now_et_time(self) -> time:
        """Return the current wall-clock time in ET.  Override in tests."""
        return datetime.now(_ET).time()

    def _check_market_hours(self) -> bool:
        """Return ``True`` if the market is currently open (9:30-16:00 ET)."""
        now_et = self._now_et_time()
        return _MARKET_OPEN <= now_et < _MARKET_CLOSE

    def _check_buying_power(self, qty: int, reference_price: Decimal) -> bool:
        """Return ``True`` if account has enough buying power for the order."""
        try:
            account = self._client.get_account()
            buying_power = Decimal(str(account.buying_power))
            required = Decimal(str(qty)) * reference_price
            if buying_power < max(required, _MIN_BUYING_POWER):
                logger.warning(
                    "insufficient_buying_power",
                    buying_power=str(buying_power),
                    required=str(required),
                )
                return False
            return True
        except APIError as exc:
            logger.error("buying_power_check_failed", error=str(exc))
            return False

    def _assert_paper_account(self) -> bool:
        """Verify the connected account is a paper account.

        This is a critical safety check — prevents accidentally trading
        real money when ``paper=True`` is configured.
        """
        try:
            account = self._client.get_account()
            # Alpaca paper accounts have account_number starting with "PA"
            # and the base URL contains "paper"
            is_paper = str(getattr(account, "account_number", "")).startswith("PA")
            if not is_paper:
                logger.error(
                    "paper_assertion_failed",
                    account_number=str(getattr(account, "account_number", "?"))[:4],
                )
            return is_paper
        except APIError as exc:
            logger.error("paper_assertion_check_failed", error=str(exc))
            return False
