#!/usr/bin/env python
"""Backtest the ORB strategy for a date range using the live pipeline modules.

Uses the same streaming indicators, ORB tracker, regime detector, and strategy
logic as the live scanner — NOT the backtesting.py engine — so results reflect
what the live scanner would have produced.

Usage
-----
    python scripts/backtest_week.py
    python scripts/backtest_week.py --start 2026-03-17 --end 2026-03-20
    python scripts/backtest_week.py --telegram
"""

from __future__ import annotations

import argparse
import json
import sys
import time as time_mod
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

import structlog

# ── path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_alpaca_settings
from src.indicators.registry import IndicatorRegistry
from src.indicators.streaming import (
    StreamingADX,
    StreamingATR,
    StreamingEMA,
    StreamingMACD,
    StreamingRSI,
)
from src.levels import LevelManager
from src.main import _VIX_FALLBACK
from src.models import Bar, Direction, TimeFrame
from src.risk.position_sizing import calculate_position_size
from src.strategies.orb import ORBStrategy
from src.strategies.regime import RegimeDetector

logger = structlog.get_logger(__name__)

_ET = ZoneInfo("America/New_York")
_ACCOUNT_SIZE = Decimal("200000")
_RISK_PCT = Decimal("1.0")
_BUYING_POWER = Decimal("200000")
_BP_CAP_FRACTION = Decimal("0.5")  # max 50% of BP per trade


# ── Alpaca REST fetch ────────────────────────────────────────────────────────


def _fetch_day_bars(
    symbol: str,
    day: date,
) -> list[Bar]:
    """Fetch 1-min bars for a single symbol on a single trading day."""
    from alpaca.data.enums import DataFeed
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame as AlpacaTimeFrame

    settings = get_alpaca_settings()
    client = StockHistoricalDataClient(
        api_key=settings.api_key,
        secret_key=settings.secret_key,
    )

    # Market hours in UTC: 9:30-16:00 ET
    et_open = datetime.combine(day, time(9, 30), tzinfo=_ET)
    et_close = datetime.combine(day, time(16, 0), tzinfo=_ET)
    start_utc = et_open.astimezone(UTC)
    end_utc = et_close.astimezone(UTC)

    feed_map = {"iex": DataFeed.IEX, "sip": DataFeed.SIP}
    feed = feed_map.get(settings.feed, DataFeed.IEX)

    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=AlpacaTimeFrame.Minute,
        start=start_utc,
        end=end_utc,
        feed=feed,
    )

    response = client.get_stock_bars(request)
    raw_bars = (
        response.data.get(symbol, []) if hasattr(response, "data") else response.get(symbol, [])
    )

    bars: list[Bar] = []
    for raw in raw_bars:
        if any(getattr(raw, f) <= 0 for f in ("open", "high", "low", "close")):
            continue
        ts = raw.timestamp
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        bars.append(
            Bar(
                symbol=symbol,
                timeframe=TimeFrame.ONE_MIN,
                timestamp=ts,
                open=Decimal(str(raw.open)),
                high=Decimal(str(raw.high)),
                low=Decimal(str(raw.low)),
                close=Decimal(str(raw.close)),
                volume=int(raw.volume),
                vwap=Decimal(str(raw.vwap)) if raw.vwap is not None else Decimal(str(raw.close)),
            )
        )

    return bars


# ── Pipeline ─────────────────────────────────────────────────────────────────


def _build_registry() -> IndicatorRegistry:
    registry = IndicatorRegistry()
    registry.register("ema9", StreamingEMA(9))
    registry.register("ema20", StreamingEMA(20))
    registry.register("ema50", StreamingEMA(50))
    registry.register("rsi", StreamingRSI(14))
    registry.register("macd", StreamingMACD())
    registry.register("atr", StreamingATR(14))
    registry.register("adx", StreamingADX(14))
    return registry


def _cap_position(qty: int, entry: Decimal) -> int:
    """Cap position to 50% of buying power."""
    if entry <= 0:
        return 0
    max_qty = int(_BUYING_POWER * _BP_CAP_FRACTION / entry)
    return min(qty, max_qty)


# ── Signal outcome evaluation ────────────────────────────────────────────────


def _evaluate_outcome(
    signal_dir: str,
    entry: Decimal,
    stop: Decimal,
    target: Decimal,
    bars_after: list[Bar],
) -> tuple[str, Decimal]:
    """Determine if target or stop was hit first. Returns (outcome, exit_price)."""
    for bar in bars_after:
        h = float(bar.high)
        lo = float(bar.low)

        if signal_dir == "LONG":
            if h >= float(target):
                return "winner", target
            if lo <= float(stop):
                return "loser", stop
        else:
            if lo <= float(target):
                return "winner", target
            if h >= float(stop):
                return "loser", stop

    # Neither hit — open at close
    if bars_after:
        return "open", bars_after[-1].close
    return "open", entry


# ── Day simulation ───────────────────────────────────────────────────────────


def _simulate_day(
    day: date,
    symbols: list[str],
) -> dict:
    """Run the ORB strategy for one trading day using the live pipeline modules."""
    # Skip Monday
    if day.weekday() == 0:
        return {
            "date": day.isoformat(),
            "day_name": "Monday",
            "skipped": True,
            "reason": "Monday filter",
            "signals": [],
        }

    logger.info("simulating_day", date=day.isoformat())
    day_signals: list[dict] = []
    orb_formed = 0

    for sym_idx, symbol in enumerate(symbols):
        # Rate limit: ~3 symbols/second to stay under 200/min
        if sym_idx > 0 and sym_idx % 3 == 0:
            time_mod.sleep(1.0)

        try:
            bars = _fetch_day_bars(symbol, day)
        except Exception as exc:
            logger.warning("fetch_failed", symbol=symbol, date=day.isoformat(), error=str(exc))
            continue

        if len(bars) < 10:
            continue

        # Build fresh pipeline for this symbol+day
        registry = _build_registry()
        levels = LevelManager(db=None, symbol=symbol)
        regime = RegimeDetector()
        strategy = ORBStrategy()

        signal_indices: list[int] = []

        for i, bar in enumerate(bars):
            registry.update_all(bar)
            snapshot = registry.get_snapshot()
            levels.update(bar)
            level_snapshot = levels.get_levels()

            # Update regime (same logic as _process_bar in main.py)
            if snapshot.atr is not None:
                trending_up = None
                if snapshot.ema9 is not None and snapshot.ema20 is not None:
                    trending_up = snapshot.ema9 > snapshot.ema20
                adx_val = snapshot.adx
                vix_val = regime.vix_level or _VIX_FALLBACK
                regime.update(vix=vix_val, adx=adx_val, trending_up=trending_up)

            # Evaluate strategy
            signal = strategy.evaluate(bar, snapshot, level_snapshot, regime)
            if signal is not None:
                signal_indices.append(i)

                # Position sizing
                qty = calculate_position_size(
                    _ACCOUNT_SIZE, _RISK_PCT, signal.entry_price, signal.stop_price
                )
                qty = _cap_position(qty, signal.entry_price)

                # Evaluate outcome from subsequent bars
                bars_after = bars[i + 1 :]
                outcome, exit_price = _evaluate_outcome(
                    str(signal.direction),
                    signal.entry_price,
                    signal.stop_price,
                    signal.target_price,
                    bars_after,
                )

                # P&L
                if signal.direction == Direction.LONG:
                    pnl = float(exit_price - signal.entry_price) * qty
                else:
                    pnl = float(signal.entry_price - exit_price) * qty

                day_signals.append(
                    {
                        "symbol": symbol,
                        "direction": str(signal.direction),
                        "entry_price": str(signal.entry_price),
                        "stop_price": str(signal.stop_price),
                        "target_price": str(signal.target_price),
                        "entry_time": bar.timestamp.astimezone(_ET).strftime("%H:%M:%S"),
                        "qty": qty,
                        "outcome": outcome,
                        "exit_price": str(exit_price),
                        "pnl": round(pnl, 2),
                        "confidence": signal.confidence_score,
                        "regime": str(signal.regime),
                        "adx": str(signal.adx) if signal.adx else "N/A",
                    }
                )

        if levels._orb.is_complete:
            orb_formed += 1

    winners = [s for s in day_signals if s["outcome"] == "winner"]
    losers = [s for s in day_signals if s["outcome"] == "loser"]
    open_trades = [s for s in day_signals if s["outcome"] == "open"]
    n_long = sum(1 for s in day_signals if s["direction"] == "LONG")
    n_short = sum(1 for s in day_signals if s["direction"] == "SHORT")
    total_pnl = sum(s["pnl"] for s in day_signals)

    return {
        "date": day.isoformat(),
        "day_name": day.strftime("%A"),
        "skipped": False,
        "orb_formed": orb_formed,
        "total_tickers": len(symbols),
        "signals": len(day_signals),
        "long": n_long,
        "short": n_short,
        "winners": len(winners),
        "losers": len(losers),
        "open": len(open_trades),
        "win_rate": round(len(winners) / (len(winners) + len(losers)) * 100, 1)
        if (len(winners) + len(losers)) > 0
        else 0,
        "total_pnl": round(total_pnl, 2),
        "top_winner": max(day_signals, key=lambda s: s["pnl"]) if day_signals else None,
        "top_loser": min(day_signals, key=lambda s: s["pnl"]) if day_signals else None,
        "trades": day_signals,
    }


# ── Output ───────────────────────────────────────────────────────────────────


def _print_day(result: dict) -> None:
    """Print a day's results to terminal."""
    print(f"\n=== {result['day_name']}, {result['date']} ===")

    if result.get("skipped"):
        print(f"  Skipped: {result.get('reason', 'N/A')}")
        return

    print(f"  ORB formed: {result['orb_formed']}/{result['total_tickers']} tickers")
    print(f"  Signals: {result['signals']} (LONG: {result['long']}, SHORT: {result['short']})")
    print(f"  Winners: {result['winners']} | Losers: {result['losers']} | Open: {result['open']}")
    print(f"  Win rate: {result['win_rate']}%")
    print(f"  Theoretical P&L: ${result['total_pnl']:+,.2f}")

    if result.get("top_winner"):
        tw = result["top_winner"]
        print(f"  Top winner: {tw['symbol']} {tw['direction']} ${tw['pnl']:+,.2f}")
    if result.get("top_loser"):
        tl = result["top_loser"]
        print(f"  Top loser: {tl['symbol']} {tl['direction']} ${tl['pnl']:+,.2f}")


def _print_weekly_summary(results: list[dict]) -> None:
    """Print the weekly summary."""
    active_days = [r for r in results if not r.get("skipped")]
    if not active_days:
        print("\n=== WEEKLY SUMMARY ===")
        print("  No active trading days.")
        return

    total_signals = sum(r["signals"] for r in active_days)
    total_winners = sum(r["winners"] for r in active_days)
    total_losers = sum(r["losers"] for r in active_days)
    total_pnl = sum(r["total_pnl"] for r in active_days)
    decided = total_winners + total_losers
    win_rate = round(total_winners / decided * 100, 1) if decided > 0 else 0

    best_day = max(active_days, key=lambda r: r["total_pnl"])
    worst_day = min(active_days, key=lambda r: r["total_pnl"])

    print("\n" + "=" * 50)
    print("  WEEKLY SUMMARY")
    print("=" * 50)
    print(f"  Total signals: {total_signals}")
    print(f"  Win rate: {win_rate}% ({total_winners}/{decided})")
    print(f"  Total theoretical P&L: ${total_pnl:+,.2f}")
    print(f"  Best day: {best_day['day_name']} ${best_day['total_pnl']:+,.2f}")
    print(f"  Worst day: {worst_day['day_name']} ${worst_day['total_pnl']:+,.2f}")
    print("=" * 50)


# ── Telegram ─────────────────────────────────────────────────────────────────


def _send_telegram_summary(results: list[dict]) -> None:
    """Send weekly summary to Telegram."""
    import httpx

    from src.config import get_telegram_settings

    active_days = [r for r in results if not r.get("skipped")]
    if not active_days:
        return

    total_signals = sum(r["signals"] for r in active_days)
    total_winners = sum(r["winners"] for r in active_days)
    total_losers = sum(r["losers"] for r in active_days)
    total_pnl = sum(r["total_pnl"] for r in active_days)
    decided = total_winners + total_losers
    win_rate = round(total_winners / decided * 100, 1) if decided > 0 else 0

    day_lines = []
    for r in active_days:
        emoji = "🟢" if r["total_pnl"] >= 0 else "🔴"
        day_lines.append(
            f"  {emoji} {r['day_name']}: ${r['total_pnl']:+,.2f} ({r['signals']} signals)"
        )

    start_date = active_days[0]["date"]
    end_date = active_days[-1]["date"]

    message = (
        f"📊 Weekly Backtest ({start_date} to {end_date})\n\n"
        f"Signals: {total_signals} | Win rate: {win_rate}%\n"
        f"P&L: ${total_pnl:+,.2f}\n\n" + "\n".join(day_lines)
    )

    settings = get_telegram_settings()
    resp = httpx.post(
        f"https://api.telegram.org/bot{settings.bot_token}/sendMessage",
        json={"chat_id": settings.chat_id, "text": message},
        timeout=10,
    )
    if resp.status_code == 200:
        logger.info("telegram_summary_sent")
    else:
        logger.warning("telegram_summary_failed", status=resp.status_code)


# ── Main ─────────────────────────────────────────────────────────────────────


def _get_symbols() -> list[str]:
    try:
        from src.config import get_app_settings

        return get_app_settings().symbols
    except Exception:
        return ["SPY", "QQQ", "MSFT", "AMD", "TSLA", "AMZN"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest ORB strategy for a date range")
    parser.add_argument("--start", type=str, default=None, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, default=None, help="End date YYYY-MM-DD")
    parser.add_argument("--telegram", action="store_true", help="Send summary via Telegram")
    parser.add_argument("--symbols", type=str, default=None, help="Comma-separated symbols")
    args = parser.parse_args()

    # Default: this week Tue-Fri
    end_date = date.fromisoformat(args.end) if args.end else date.today()
    if args.start:
        start_date = date.fromisoformat(args.start)
    else:
        # Go back to Tuesday of this week
        start_date = end_date - timedelta(days=end_date.weekday() - 1)
        if start_date > end_date:
            start_date = end_date

    symbols = args.symbols.split(",") if args.symbols else _get_symbols()

    print(f"Backtesting {start_date} to {end_date}")
    print(f"Symbols: {len(symbols)} ({', '.join(symbols[:5])}{'...' if len(symbols) > 5 else ''})")
    print(
        f"Account: ${_ACCOUNT_SIZE:,.0f} | Risk: {_RISK_PCT}% | BP cap: {_BP_CAP_FRACTION * 100}%"
    )

    # Generate trading days
    trading_days: list[date] = []
    current = start_date
    while current <= end_date:
        if current.weekday() < 5:  # Mon-Fri
            trading_days.append(current)
        current += timedelta(days=1)

    results: list[dict] = []
    for day in trading_days:
        result = _simulate_day(day, symbols)
        results.append(result)
        _print_day(result)

    _print_weekly_summary(results)

    # Save to JSON
    output_path = Path(f"data/backtest_{start_date}_to_{end_date}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    if args.telegram:
        _send_telegram_summary(results)


if __name__ == "__main__":
    main()
