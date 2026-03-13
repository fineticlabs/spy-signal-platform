"""CLI script: download the last N days of 1-min bars for one or more tickers."""

from __future__ import annotations

import argparse
import sys

import structlog

from src.config import get_app_settings
from src.ingestion.historical import fetch_historical_bars
from src.models import TimeFrame
from src.storage.database import BarDatabase

logger = structlog.get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill historical 1-min bars from Alpaca into SQLite."
    )
    parser.add_argument(
        "--days",
        type=int,
        default=90,
        help="Number of calendar days to fetch (default: 90)",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default=None,
        help="Single ticker symbol to backfill (overridden by --symbols if both given)",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="Comma-separated ticker symbols to backfill (e.g. SPY,QQQ,IWM)",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Override SQLite database path from .env",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    db_path = args.db_path or get_app_settings().db_path

    # --symbols beats --symbol beats default SPY
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    elif args.symbol:
        symbols = [args.symbol.strip().upper()]
    else:
        symbols = ["SPY"]

    print(f"Backfilling {args.days} days for {', '.join(symbols)} → {db_path}")

    for symbol in symbols:
        print(f"\n── {symbol} ──")
        with BarDatabase(db_path=db_path) as db:
            bars = fetch_historical_bars(
                symbol=symbol,
                days=args.days,
                timeframe=TimeFrame.ONE_MIN,
                db=db,
            )
            total = db.count_bars(symbol=symbol, timeframe=TimeFrame.ONE_MIN)
            earliest_bars = db.query_bars(symbol=symbol, timeframe=TimeFrame.ONE_MIN, limit=1)

        earliest = earliest_bars[0].timestamp if earliest_bars else None
        print(f"  Fetched {len(bars):,} bars this run.  Total in DB: {total:,}")
        if earliest:
            print(f"  Earliest bar in DB: {earliest.strftime('%Y-%m-%d %H:%M UTC')}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
