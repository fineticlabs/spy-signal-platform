"""Fixed-fractional position sizing using Decimal arithmetic."""

from __future__ import annotations

from decimal import Decimal


def calculate_position_size(
    account_size: Decimal,
    risk_pct: Decimal,
    entry: Decimal,
    stop: Decimal,
) -> int:
    """Calculate the number of shares to trade given a fixed risk fraction.

    The position size is the largest whole number of shares such that the
    total dollar risk (entry-to-stop distance * shares) does not exceed
    ``account_size * risk_pct / 100``.

    Args:
        account_size: Total account value in dollars.
        risk_pct:     Percentage of account to risk (e.g. ``Decimal("1.0")`` = 1%).
        entry:        Planned entry price per share.
        stop:         Initial stop-loss price per share.

    Returns:
        Number of shares to trade, floored to the nearest whole share.
        Returns ``0`` if the stop distance is zero or negative (impossible sizing).
    """
    risk_per_share = abs(entry - stop)
    if risk_per_share <= 0:
        return 0
    dollar_risk = account_size * risk_pct / Decimal("100")
    return int(dollar_risk / risk_per_share)  # floor toward zero
