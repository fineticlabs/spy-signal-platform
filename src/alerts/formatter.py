"""Telegram MarkdownV2 message formatters for trading alerts."""

from __future__ import annotations

from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

if TYPE_CHECKING:
    from decimal import Decimal

    from src.models import RiskDecision, Signal, TradeResult

_ET = ZoneInfo("America/New_York")

# Characters that must be escaped in Telegram MarkdownV2 (outside formatting entities)
_MD2_SPECIAL = r"\_*[]()~`>#+-=|{}.!"


def _md2(text: str) -> str:
    """Escape a string for safe use in Telegram MarkdownV2."""
    result = ""
    for char in text:
        if char in _MD2_SPECIAL:
            result += f"\\{char}"
        else:
            result += char
    return result


def _price(value: Decimal | None) -> str:
    """Format a Decimal price with 2 decimal places, MarkdownV2-escaped."""
    if value is None:
        return _md2("N/A")
    return _md2(f"{value:.2f}")


def _stars(n: int) -> str:
    """Return star unicode string for a confidence score 1-5."""
    filled = max(0, min(n, 5))
    return "⭐" * filled + "☆" * (5 - filled)


def format_signal_alert(signal: Signal, risk_decision: RiskDecision) -> str:
    """Format a trading signal as a Telegram MarkdownV2 message.

    Args:
        signal:        The signal to format.
        risk_decision: The approved risk decision (provides position size).

    Returns:
        A MarkdownV2-formatted string ready to send via ``Bot.send_message``.
    """
    direction_emoji = "🟢" if signal.direction == "LONG" else "🔴"
    time_et = signal.timestamp.astimezone(_ET).strftime("%H:%M ET")

    tag_str = " ".join(f"[{_md2(t)}]" for t in signal.tags) if signal.tags else ""
    header = f"{direction_emoji} *{_md2(str(signal.direction))}* — {_md2(signal.strategy_name)}"
    if tag_str:
        header += f" {tag_str}"

    lines: list[str] = [
        header,
        "",
        f"*{_md2(signal.symbol)} @ {_price(signal.entry_price)}*",
        f"Entry:  {_price(signal.entry_price)}",
        f"Stop:   {_price(signal.stop_price)}",
        f"Target: {_price(signal.target_price)}",
        f"R:R: {_price(signal.risk_reward_ratio)} \\| Confidence: {_stars(signal.confidence_score)}",
        f"Shares: {_md2(str(risk_decision.position_size))}",
        "",
        f"Regime: {_md2(str(signal.regime))} \\| VIX: {_price(signal.vix)} \\| ADX: {_price(signal.adx)}",
        f"Time: {_md2(time_et)}",
    ]

    # Optional key levels
    lvl = signal.levels_snapshot
    if lvl is not None:
        if lvl.orb_high is not None and lvl.orb_low is not None:
            lines.append(f"ORB: {_price(lvl.orb_low)} \\- {_price(lvl.orb_high)}")
        if lvl.vwap is not None:
            lines.append(f"VWAP: {_price(lvl.vwap)}")
        if lvl.prev_day_high is not None and lvl.prev_day_low is not None:
            lines.append(f"PDH/PDL: {_price(lvl.prev_day_high)} / {_price(lvl.prev_day_low)}")
        if lvl.rvol is not None:
            lines.append(f"RVOL: {_price(lvl.rvol)}x")
        if lvl.vp_poc is not None:
            va_str = f"{_price(lvl.vp_val)} \\- {_price(lvl.vp_vah)}"
            lines.append(f"VP POC: {_price(lvl.vp_poc)} \\| VA: {va_str}")
        if lvl.vix_term_ratio is not None:
            ratio_str = _md2(f"{lvl.vix_term_ratio:.2f}")
            lines.append(f"VIX Term: {ratio_str}")

    lines += ["", f"_{_md2(signal.reason)}_"]

    return "\n".join(lines)


def format_earnings_blackout_alert(symbol: str, earnings_date: str) -> str:
    """Format an earnings blackout notification as a Telegram MarkdownV2 message.

    Sent when a ticker is blocked from ORB trading due to nearby earnings.

    Args:
        symbol:        Ticker symbol being blocked.
        earnings_date: Human-readable date of the earnings announcement.

    Returns:
        MarkdownV2-formatted string.
    """
    return (
        f"🚫 *EARNINGS BLACKOUT*\n\n"
        f"{_md2(symbol)} blocked from ORB trading\\.\n"
        f"Earnings: {_md2(earnings_date)}\n"
        f"Resumes day after next\\."
    )


def format_risk_alert(message: str) -> str:
    """Format a risk-management warning as a Telegram MarkdownV2 message.

    Args:
        message: Plain-text warning message.

    Returns:
        MarkdownV2-formatted string.
    """
    return f"⚠️ *Risk Warning*\n\n{_md2(message)}"


def format_daily_summary(trades: list[TradeResult]) -> str:
    """Format an end-of-day trade summary as a Telegram MarkdownV2 message.

    Args:
        trades: All completed trades for the session.

    Returns:
        MarkdownV2-formatted string.
    """
    if not trades:
        return "📊 *Daily Summary*\n\nNo trades today\\."

    total_pnl = sum(t.pnl for t in trades)
    wins = sum(1 for t in trades if t.pnl >= 0)
    losses = len(trades) - wins
    win_rate = wins / len(trades) * 100

    pnl_sign = "\\+" if total_pnl >= 0 else ""
    pnl_str = f"{pnl_sign}\\${_md2(f'{total_pnl:.2f}')}"

    return "\n".join(
        [
            "📊 *Daily Summary*",
            "",
            f"Trades: {len(trades)}",
            f"P&L: {pnl_str}",
            f"Wins: {wins} \\| Losses: {losses}",
            f"Win Rate: {_md2(f'{win_rate:.1f}')}%",
        ]
    )
