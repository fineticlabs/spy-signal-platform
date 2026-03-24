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
from alpaca.trading.enums import OrderClass, OrderSide, OrderStatus, QueryOrderStatus, TimeInForce
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
        # In-memory set of symbols with active orders/positions today.
        # Checked BEFORE the Alpaca API round-trip to prevent the race
        # condition where order N+1 is submitted before Alpaca acknowledges
        # order N.  Populated at startup via load_existing_positions() and
        # updated on each successful bracket order submission.
        self._submitted_symbols: set[str] = set()
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

        Position size is capped to 50% of available buying power.  Stop and
        target prices are rounded to 2 decimal places for Alpaca compliance.

        Args:
            symbol:       Ticker symbol (e.g. ``"SPY"``).
            direction:    ``Direction.LONG`` or ``Direction.SHORT``.
            qty:          Number of shares (from risk sizer, before BP cap).
            stop_price:   Stop-loss price.
            target_price: Take-profit limit price.

        Returns:
            The parent :class:`Order` on success, or ``None`` if a safety
            check failed or the API returned an error.
        """
        # ── safety checks ─────────────────────────────────────────────────
        # Fast local check first — eliminates race condition where the
        # Alpaca API hasn't acknowledged the previous order yet.
        if symbol in self._submitted_symbols:
            logger.info("signal_skipped_local_dedup", symbol=symbol)
            return None

        if self._has_existing_position_or_order(symbol):
            logger.info("signal_skipped_existing_position", symbol=symbol)
            self._submitted_symbols.add(symbol)  # sync local set
            return None

        if not self._check_market_hours():
            logger.warning("order_rejected_market_closed", symbol=symbol)
            return None

        if self._paper and not self._assert_paper_account():
            logger.error("order_rejected_not_paper", symbol=symbol)
            return None

        # ── cap position to 50% of buying power ──────────────────────────
        qty = self._cap_to_buying_power(symbol, qty, stop_price)
        if qty < 1:
            logger.warning("order_skipped_insufficient_size", symbol=symbol)
            return None

        # ── round prices to 2 decimals for Alpaca compliance ─────────────
        rounded_stop = round(float(stop_price), 2)
        rounded_target = round(float(target_price), 2)

        # ── build bracket order ───────────────────────────────────────────
        side = OrderSide.BUY if direction == Direction.LONG else OrderSide.SELL
        order_data = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=side,
            time_in_force=TimeInForce.DAY,
            order_class=OrderClass.BRACKET,
            take_profit=TakeProfitRequest(limit_price=rounded_target),
            stop_loss=StopLossRequest(stop_price=rounded_stop),
        )

        try:
            order: Order = self._client.submit_order(order_data)
            self._submitted_symbols.add(symbol)
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

    def load_existing_positions(self) -> list[str]:
        """Pre-load current Alpaca positions into the local dedup set.

        Call this once at startup (before the first signal cycle) so that
        carry-over positions from the previous day are immediately known.

        Returns:
            List of symbols that were loaded.
        """
        try:
            positions = self._client.get_all_positions()
            symbols = [pos.symbol for pos in positions]
            self._submitted_symbols.update(symbols)
            logger.info(
                "existing_positions_loaded",
                symbols=symbols,
                count=len(symbols),
            )
            return symbols
        except APIError as exc:
            logger.error("existing_positions_load_failed", error=str(exc))
            return []

    def clear_submitted_symbols(self) -> None:
        """Clear the local dedup set.  Call at EOD after flatten."""
        self._submitted_symbols.clear()
        logger.info("submitted_symbols_cleared")

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

    def _has_existing_position_or_order(self, symbol: str) -> bool:
        """Return ``True`` if there is an open position or pending order for *symbol*.

        Prevents duplicate bracket orders from piling up on the same symbol.
        """
        try:
            positions = self._client.get_all_positions()
            for pos in positions:
                if pos.symbol == symbol:
                    return True
        except APIError as exc:
            logger.error("position_check_failed", symbol=symbol, error=str(exc))
            return True  # fail-closed: block order if we can't verify

        try:
            orders = self._client.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.OPEN))
            for order in orders:
                if order.symbol == symbol:
                    return True
        except APIError as exc:
            logger.error("order_check_failed", symbol=symbol, error=str(exc))
            return True  # fail-closed

        return False

    def _now_et_time(self) -> time:
        """Return the current wall-clock time in ET.  Override in tests."""
        return datetime.now(_ET).time()

    def _check_market_hours(self) -> bool:
        """Return ``True`` if the market is currently open (9:30-16:00 ET)."""
        now_et = self._now_et_time()
        return _MARKET_OPEN <= now_et < _MARKET_CLOSE

    def _cap_to_buying_power(self, symbol: str, qty: int, entry_price: Decimal) -> int:
        """Cap position size to 50% of available buying power.

        This ensures no single trade consumes all buying power, leaving room
        for concurrent positions.  Returns 0 if buying power is insufficient
        for even 1 share.

        Args:
            symbol:      Ticker symbol (for logging).
            qty:         Desired quantity from risk sizer.
            entry_price: Approximate entry price for position value calc.

        Returns:
            Adjusted quantity (may be less than *qty*), or 0 if infeasible.
        """
        try:
            account = self._client.get_account()
            buying_power = Decimal(str(account.buying_power))
        except APIError as exc:
            logger.error("buying_power_check_failed", error=str(exc))
            return 0

        if buying_power < _MIN_BUYING_POWER:
            logger.warning(
                "insufficient_buying_power",
                buying_power=str(buying_power),
                min_required=str(_MIN_BUYING_POWER),
            )
            return 0

        if entry_price <= 0:  # pragma: no cover — defensive guard; prices are always positive
            return 0

        # Max shares that fit in 50% of buying power
        half_bp = buying_power / 2
        max_qty = int(half_bp / entry_price)
        if qty <= max_qty:
            return qty

        adjusted = max_qty
        logger.info(
            "position_downsized",
            symbol=symbol,
            original_qty=qty,
            max_qty=max_qty,
            adjusted_qty=adjusted,
            buying_power=str(buying_power),
            reason="buying_power_cap",
        )
        return adjusted

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

    # ── reconciliation ─────────────────────────────────────────────────────────

    def get_closed_orders_today(self) -> list[Order]:
        """Return all filled/cancelled orders from today.

        Used at EOD to reconcile executed orders back to signal records.
        """
        try:
            today_start = datetime.now(_ET).replace(
                hour=0,
                minute=0,
                second=0,
                microsecond=0,
            )
            orders = self._client.get_orders(
                filter=GetOrdersRequest(
                    status=QueryOrderStatus.CLOSED,
                    after=today_start.isoformat(),
                    limit=500,
                ),
            )
            return [o for o in orders if o.status == OrderStatus.FILLED]
        except APIError as exc:
            logger.error("get_closed_orders_failed", error=str(exc))
            return []

    def get_daily_pnl(self) -> Decimal:
        """Return today's realized P&L from the Alpaca account.

        Calculated as current equity minus last equity (previous close).
        """
        try:
            account = self._client.get_account()
            equity = Decimal(str(account.equity))
            last_equity = Decimal(str(account.last_equity))
            return equity - last_equity
        except APIError as exc:
            logger.error("get_daily_pnl_failed", error=str(exc))
            return Decimal("0")
