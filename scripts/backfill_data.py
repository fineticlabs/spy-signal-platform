"""CLI script: download the last N days of 1-min SPY bars and store in SQLite."""

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
        description="Backfill historical SPY 1-min bars from Alpaca into SQLite."
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
        default="SPY",
        help="Ticker symbol to backfill (default: SPY)",
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

    print(f"Backfilling {args.days} days of {args.symbol} 1-min bars → {db_path}")

    with BarDatabase(db_path=db_path) as db:
        bars = fetch_historical_bars(
            symbol=args.symbol,
            days=args.days,
            timeframe=TimeFrame.ONE_MIN,
            db=db,
        )
        total = db.count_bars(symbol=args.symbol, timeframe=TimeFrame.ONE_MIN)

    print(f"Done.  Fetched {len(bars):,} bars this run.  Total in DB: {total:,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
