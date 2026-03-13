"""Script to verify the Telegram bot is configured and can send messages.

Usage::

    python scripts/test_telegram.py

Requires TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env or environment.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from decimal import Decimal

from src.alerts.telegram import TelegramAlerter
from src.config import get_telegram_settings
from src.models import (
    Direction,
    IndicatorSnapshot,
    LevelSnapshot,
    Regime,
    RiskDecision,
    Signal,
    TimeFrame,
)


def _sample_signal() -> Signal:
    return Signal(
        direction=Direction.LONG,
        strategy_name="ORB-5min",
        entry_price=Decimal("486.00"),
        stop_price=Decimal("483.00"),
        target_price=Decimal("492.00"),
        risk_reward_ratio=Decimal("2.00"),
        confidence_score=3,
        reason="Test signal from scripts/test_telegram.py",
        timeframe=TimeFrame.ONE_MIN,
        regime=Regime.TRENDING_UP,
        vix=Decimal("18.5"),
        adx=Decimal("28.0"),
        indicators_snapshot=IndicatorSnapshot(atr=Decimal("2.00")),
        levels_snapshot=LevelSnapshot(
            orb_high=Decimal("485.00"),
            orb_low=Decimal("480.00"),
            orb_complete=True,
            vwap=Decimal("483.50"),
            prev_day_high=Decimal("490.00"),
            prev_day_low=Decimal("478.00"),
        ),
        timestamp=datetime.now(UTC),
    )


async def main() -> None:
    settings = get_telegram_settings()
    alerter = TelegramAlerter(
        bot_token=settings.bot_token,
        chat_id=settings.chat_id,
    )

    signal = _sample_signal()
    decision = RiskDecision(approved=True, reason="Test delivery", position_size=166)

    print("Sending test signal alert...")
    ok = await alerter.send_signal(signal, decision)
    print(f"Signal alert: {'OK' if ok else 'FAILED'}")

    print("Sending test risk warning...")
    ok = await alerter.send_risk_warning(
        "This is a test risk warning from scripts/test_telegram.py"
    )
    print(f"Risk warning: {'OK' if ok else 'FAILED'}")


if __name__ == "__main__":
    asyncio.run(main())
