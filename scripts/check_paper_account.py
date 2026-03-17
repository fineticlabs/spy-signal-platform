#!/usr/bin/env python
"""Check Alpaca paper trading account status and open positions.

Usage
-----
    python scripts/check_paper_account.py

Reads ALPACA_API_KEY and ALPACA_SECRET_KEY from .env and displays:
- Account equity, buying power, cash
- Open positions with unrealized P&L
- Open orders
"""

from __future__ import annotations

import sys
from decimal import Decimal
from pathlib import Path

from dotenv import load_dotenv

# ── path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv(Path(__file__).parent.parent / ".env")

from src.config import get_alpaca_settings  # noqa: E402
from src.execution.alpaca_executor import AlpacaExecutor  # noqa: E402


def main() -> None:
    """Display paper account status."""
    settings = get_alpaca_settings()

    executor = AlpacaExecutor(
        api_key=settings.api_key,
        secret_key=settings.secret_key,
        paper=True,
    )

    # ── Account info ──────────────────────────────────────────────────────────
    account = executor.get_account()
    equity = Decimal(str(account.equity))
    buying_power = Decimal(str(account.buying_power))
    cash = Decimal(str(account.cash))

    print(f"\n{'=' * 60}")
    print("  Alpaca Paper Account Status")
    print(f"{'=' * 60}")
    print(f"  Account:       {account.account_number}")
    print(f"  Equity:        ${equity:,.2f}")
    print(f"  Cash:          ${cash:,.2f}")
    print(f"  Buying Power:  ${buying_power:,.2f}")
    print(f"  Day Trade Count: {account.daytrade_count}")
    print(f"  PDT Flag:      {account.pattern_day_trader}")

    # ── Open positions ────────────────────────────────────────────────────────
    positions = executor.get_positions()
    print(f"\n  Open Positions: {len(positions)}")
    if positions:
        print(f"  {'-' * 56}")
        print(f"  {'Symbol':<8} {'Qty':>6} {'Side':<6} {'Entry$':>10} {'Current$':>10} {'P&L':>10}")
        print(f"  {'-' * 56}")
        total_unrealized = Decimal("0")
        for pos in positions:
            unrealized = Decimal(str(pos.unrealized_pl))
            total_unrealized += unrealized
            print(
                f"  {pos.symbol:<8} {pos.qty:>6} {pos.side:<6} "
                f"${Decimal(str(pos.avg_entry_price)):>9,.2f} "
                f"${Decimal(str(pos.current_price)):>9,.2f} "
                f"${unrealized:>+9,.2f}"
            )
        print(f"  {'-' * 56}")
        print(f"  {'Total unrealized P&L:':>40} ${total_unrealized:>+9,.2f}")

    # ── Open orders ───────────────────────────────────────────────────────────
    from alpaca.trading.enums import QueryOrderStatus
    from alpaca.trading.requests import GetOrdersRequest

    orders = executor._client.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.OPEN))
    print(f"\n  Open Orders: {len(orders)}")
    if orders:
        print(f"  {'-' * 56}")
        for order in orders:
            print(
                f"  {order.symbol:<8} {order.side:<5} {order.qty:>6} "
                f"{order.order_class or 'simple':<10} {order.status}"
            )

    print(f"\n{'=' * 60}\n")


if __name__ == "__main__":
    main()
