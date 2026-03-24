#!/usr/bin/env python
"""Portfolio-level backtest: shared capital, position limits, signal ranking.

Unlike ``run_backtest.py`` which runs 26 independent per-ticker backtests,
this simulates a single portfolio with:
- Shared capital pool (default $200K)
- Max 3 concurrent positions (enforced)
- Max 5 trades per day (enforced)
- Signal ranking when multiple candidates fire on the same bar
- Sector concentration limits (max 2 same-sector positions)

This gives a realistic estimate of what the strategy would produce under
capital constraints, unlike the standard backtest's $1.3M implied capital.

Usage
-----
    python scripts/run_portfolio_backtest.py --cash 200000
"""

from __future__ import annotations

import argparse
import sys
import warnings
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import structlog
import talib

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtest.data_loader import load_bars
from src.models import TimeFrame
from src.risk.manager import SECTOR_MAP
from src.storage.database import BarDatabase

warnings.filterwarnings("ignore", category=RuntimeWarning)

logger = structlog.get_logger(__name__)
_ET = ZoneInfo("America/New_York")

# ── Strategy constants (matching engine.py) ──────────────────────────────────
_ORB_BARS = 5
_ATR_PERIOD = 14
_ATR_MULTIPLIER = 1.5
_RISK_MULTIPLIER = 2.0
_VOL_MULTIPLIER = 1.5
_VOL_WINDOW = 20
_DAILY_ADX_MIN = 25.0
_REALIZED_VOL_MAX = 0.18
_MIN_ORB_PCT = 0.0015
_GAP_THRESHOLD_PCT = 0.3
_WINDOW_START = time(9, 35)
_WINDOW_END = time(10, 0)
_FORCE_FLAT = time(15, 55)
_MAX_TRADES_PER_DAY = 5
_MAX_CONCURRENT = 3
_MAX_SAME_SECTOR = 2
_POSITION_SCALE = 0.25


@dataclass
class OpenPosition:
    """Tracks an open bracket position."""

    symbol: str
    direction: str  # "LONG" or "SHORT"
    entry: float
    stop: float
    target: float
    size: float  # notional dollar value
    entry_time: datetime


@dataclass
class ClosedTrade:
    """Completed trade result."""

    symbol: str
    direction: str
    entry: float
    exit_price: float
    pnl: float
    entry_time: datetime
    exit_time: datetime


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Portfolio-level ORB backtest with capital constraints"
    )
    parser.add_argument(
        "--cash", type=float, default=200_000.0, help="Starting capital (default: 200000)"
    )
    parser.add_argument("--out-dir", default="docs/portfolio_backtest", help="Output directory")
    parser.add_argument(
        "--symbols",
        default="SPY,QQQ,MSFT,AMD,TSLA,AMZN,UBER,SMCI,SHOP,PLTR,NFLX,MSTR,SNOW,ARM,DASH,PYPL,INTC,MU,HOOD,DKNG,SOXL,ROKU,TQQQ,BA,MRVL,META",
    )
    return parser.parse_args()


def main() -> None:
    """Run portfolio-level backtest."""
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    symbols = [s.strip().upper() for s in args.symbols.split(",")]

    db = BarDatabase()
    db.connect()

    # Load all 1-min bars per symbol
    print(f"Loading bars for {len(symbols)} symbols...")
    all_bars: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        df = load_bars(db, symbol=sym, timeframe=TimeFrame.ONE_MIN)
        if not df.empty:
            all_bars[sym] = df
            print(f"  {sym}: {len(df):,} bars ({df.index[0].date()} to {df.index[-1].date()})")

    db.close()

    if not all_bars:
        print("No data loaded.")
        sys.exit(1)

    # Find common date range
    min_date = max(df.index[0] for df in all_bars.values())
    max_date = min(df.index[-1] for df in all_bars.values())
    print(f"\nCommon range: {min_date.date()} to {max_date.date()}")

    # Pre-compute daily indicators for SPY (ADX, realized vol)
    spy_df = all_bars.get("SPY")
    if spy_df is None:
        print("SPY data required for regime filters.")
        sys.exit(1)

    spy_et = spy_df.copy()
    spy_et.index = spy_et.index.tz_convert(_ET)
    spy_daily = (
        spy_et.resample("1D")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna()
    )

    daily_adx = talib.ADX(
        spy_daily["high"].values, spy_daily["low"].values, spy_daily["close"].values, timeperiod=14
    )
    daily_returns = spy_daily["close"].pct_change()
    realized_vol = daily_returns.rolling(20).std() * np.sqrt(252)

    # Build date→indicator lookup (shifted 1 day = no lookahead)
    adx_lookup: dict[date, float] = {}
    vol_lookup: dict[date, float] = {}
    dates_list = list(spy_daily.index)
    for i in range(1, len(dates_list)):
        d = dates_list[i].date()
        if i - 1 < len(daily_adx) and not np.isnan(daily_adx[i - 1]):
            adx_lookup[d] = float(daily_adx[i - 1])
        if i - 1 < len(realized_vol) and not np.isnan(realized_vol.iloc[i - 1]):
            vol_lookup[d] = float(realized_vol.iloc[i - 1])

    # Pre-compute ORB ranges per symbol per day
    print("\nPre-computing ORB ranges...")
    # symbol → date → (orb_high, orb_low)
    orb_cache: dict[str, dict[date, tuple[float, float]]] = defaultdict(dict)
    for sym, df in all_bars.items():
        df_et = df.copy()
        df_et.index = df_et.index.tz_convert(_ET)
        for d, day_df in df_et.groupby(df_et.index.date):
            orb_bars = day_df.between_time("09:30", "09:34")
            if len(orb_bars) >= _ORB_BARS:
                orb_cache[sym][d] = (float(orb_bars["high"].max()), float(orb_bars["low"].min()))

    # ── Portfolio simulation ─────────────────────────────────────────────────
    print(f"\nSimulating portfolio (${args.cash:,.0f}, max {_MAX_CONCURRENT} concurrent)...")

    equity = args.cash
    open_positions: list[OpenPosition] = []
    closed_trades: list[ClosedTrade] = []
    equity_curve: list[float] = []
    signals_generated = 0
    signals_rejected_position_limit = 0
    signals_rejected_sector = 0

    # Get all unique trading days across all symbols
    all_dates: set[date] = set()
    for df in all_bars.values():
        df_et = df.copy()
        df_et.index = df_et.index.tz_convert(_ET)
        all_dates.update(df_et.index.date)
    trading_days = sorted(all_dates)

    for day in trading_days:
        # Skip Mondays
        if day.weekday() == 0:
            continue

        # Check regime filters
        d_adx = adx_lookup.get(day)
        d_vol = vol_lookup.get(day)
        if d_adx is not None and d_adx <= _DAILY_ADX_MIN:
            continue
        if d_vol is not None and d_vol >= _REALIZED_VOL_MAX:
            continue

        daily_trade_count = 0

        # Process bars in chronological order within the ORB window
        # Collect candidate signals across all symbols for each bar minute
        for minute_offset in range(5, 30):  # 9:35 to 9:59
            if daily_trade_count >= _MAX_TRADES_PER_DAY:
                break

            # Check/close existing positions first
            for pos in list(open_positions):
                sym_df = all_bars.get(pos.symbol)
                if sym_df is None:
                    continue
                df_et = sym_df.copy()
                df_et.index = df_et.index.tz_convert(_ET)
                bar_time = datetime.combine(day, time(9, 30 + minute_offset)).replace(tzinfo=_ET)
                mask = (df_et.index >= bar_time) & (df_et.index < bar_time + timedelta(minutes=1))
                bars = df_et[mask]
                if bars.empty:
                    continue
                bar = bars.iloc[0]
                if pos.direction == "LONG":
                    if float(bar["high"]) >= pos.target:
                        pnl = (pos.target - pos.entry) * (pos.size / pos.entry)
                        closed_trades.append(
                            ClosedTrade(
                                pos.symbol,
                                pos.direction,
                                pos.entry,
                                pos.target,
                                pnl,
                                pos.entry_time,
                                bar_time,
                            )
                        )
                        open_positions.remove(pos)
                        equity += pnl
                    elif float(bar["low"]) <= pos.stop:
                        pnl = (pos.stop - pos.entry) * (pos.size / pos.entry)
                        closed_trades.append(
                            ClosedTrade(
                                pos.symbol,
                                pos.direction,
                                pos.entry,
                                pos.stop,
                                pnl,
                                pos.entry_time,
                                bar_time,
                            )
                        )
                        open_positions.remove(pos)
                        equity += pnl
                else:
                    if float(bar["low"]) <= pos.target:
                        pnl = (pos.entry - pos.target) * (pos.size / pos.entry)
                        closed_trades.append(
                            ClosedTrade(
                                pos.symbol,
                                pos.direction,
                                pos.entry,
                                pos.target,
                                pnl,
                                pos.entry_time,
                                bar_time,
                            )
                        )
                        open_positions.remove(pos)
                        equity += pnl
                    elif float(bar["high"]) >= pos.stop:
                        pnl = (pos.entry - pos.stop) * (pos.size / pos.entry)
                        closed_trades.append(
                            ClosedTrade(
                                pos.symbol,
                                pos.direction,
                                pos.entry,
                                pos.stop,
                                pnl,
                                pos.entry_time,
                                bar_time,
                            )
                        )
                        open_positions.remove(pos)
                        equity += pnl

            if len(open_positions) >= _MAX_CONCURRENT:
                continue
            if daily_trade_count >= _MAX_TRADES_PER_DAY:
                break

            # Collect candidates across all symbols for this minute
            candidates: list[
                tuple[str, str, float, float, float, float]
            ] = []  # (sym, dir, entry, stop, target, score)

            for sym in all_bars:
                orb = orb_cache.get(sym, {}).get(day)
                if orb is None:
                    continue
                orb_high, orb_low = orb

                # ORB range check
                if orb_low <= 0:
                    continue
                orb_range_pct = (orb_high - orb_low) / orb_low
                if orb_range_pct < _MIN_ORB_PCT:
                    continue

                # Already in position for this symbol?
                if any(p.symbol == sym for p in open_positions):
                    continue

                sym_df = all_bars.get(sym)
                if sym_df is None:
                    continue
                df_et = sym_df.copy()
                df_et.index = df_et.index.tz_convert(_ET)
                bar_time = datetime.combine(day, time(9, 30 + minute_offset)).replace(tzinfo=_ET)
                prev_time = bar_time - timedelta(minutes=1)

                curr_mask = (df_et.index >= bar_time) & (
                    df_et.index < bar_time + timedelta(minutes=1)
                )
                prev_mask = (df_et.index >= prev_time) & (
                    df_et.index < prev_time + timedelta(minutes=1)
                )
                curr_bars = df_et[curr_mask]
                prev_bars = df_et[prev_mask]

                if curr_bars.empty or prev_bars.empty:
                    continue

                close = float(curr_bars.iloc[0]["close"])
                prev_close = float(prev_bars.iloc[0]["close"])
                volume = float(curr_bars.iloc[0]["volume"])

                # Volume filter (simplified: compare to a fixed threshold)
                if volume < 100_000:
                    continue

                # ATR approximation (use ORB range as proxy)
                atr_proxy = orb_high - orb_low
                if atr_proxy <= 0:
                    continue

                # 2-bar confirmation + direction
                if close > orb_high and prev_close > orb_high:
                    direction = "LONG"
                    entry = close
                    stop = entry - _ATR_MULTIPLIER * atr_proxy
                    target = entry + _RISK_MULTIPLIER * _ATR_MULTIPLIER * atr_proxy
                elif close < orb_low and prev_close < orb_low:
                    direction = "SHORT"
                    entry = close
                    stop = entry + _ATR_MULTIPLIER * atr_proxy
                    target = entry - _RISK_MULTIPLIER * _ATR_MULTIPLIER * atr_proxy
                else:
                    continue

                signals_generated += 1
                # Score: ADX strength + ORB range width
                score = (d_adx or 25) / 100 * 0.4 + orb_range_pct * 100 * 0.3 + 0.3
                candidates.append((sym, direction, entry, stop, target, score))

            # Rank candidates by score, take top N within limits
            candidates.sort(key=lambda c: c[5], reverse=True)

            for sym, direction, entry, stop, target, _score in candidates:
                if len(open_positions) >= _MAX_CONCURRENT:
                    signals_rejected_position_limit += 1
                    continue
                if daily_trade_count >= _MAX_TRADES_PER_DAY:
                    signals_rejected_position_limit += 1
                    break

                # Sector limit
                sym_sector = SECTOR_MAP.get(sym, "other")
                same_sector = sum(
                    1 for p in open_positions if SECTOR_MAP.get(p.symbol, "other") == sym_sector
                )
                if same_sector >= _MAX_SAME_SECTOR:
                    signals_rejected_sector += 1
                    continue

                pos_size = equity * _POSITION_SCALE
                bar_time = datetime.combine(day, time(9, 30 + minute_offset)).replace(tzinfo=_ET)
                open_positions.append(
                    OpenPosition(sym, direction, entry, stop, target, pos_size, bar_time)
                )
                daily_trade_count += 1

        # Force flat at 15:55 ET
        for pos in list(open_positions):
            sym_df = all_bars.get(pos.symbol)
            if sym_df is None:
                continue
            df_et = sym_df.copy()
            df_et.index = df_et.index.tz_convert(_ET)
            close_time = datetime.combine(day, _FORCE_FLAT).replace(tzinfo=_ET)
            mask = (df_et.index >= close_time) & (df_et.index < close_time + timedelta(minutes=1))
            bars = df_et[mask]
            if bars.empty:
                # Use last available bar
                day_bars = df_et[df_et.index.date == day]
                if day_bars.empty:
                    continue
                exit_price = float(day_bars.iloc[-1]["close"])
            else:
                exit_price = float(bars.iloc[0]["close"])

            if pos.direction == "LONG":
                pnl = (exit_price - pos.entry) * (pos.size / pos.entry)
            else:
                pnl = (pos.entry - exit_price) * (pos.size / pos.entry)
            closed_trades.append(
                ClosedTrade(
                    pos.symbol,
                    pos.direction,
                    pos.entry,
                    exit_price,
                    pnl,
                    pos.entry_time,
                    close_time,
                )
            )
            equity += pnl
            open_positions.remove(pos)

        equity_curve.append(equity)

    # ── Results ──────────────────────────────────────────────────────────────
    total_trades = len(closed_trades)
    if total_trades == 0:
        print("\nNo trades generated.")
        sys.exit(0)

    pnl_series = pd.Series([t.pnl for t in closed_trades])
    winners = pnl_series[pnl_series > 0]
    losers = pnl_series[pnl_series < 0]
    gross_profit = float(winners.sum()) if len(winners) else 0
    gross_loss = abs(float(losers.sum())) if len(losers) else 0.001
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    win_rate = len(winners) / total_trades * 100
    net_pnl = gross_profit - abs(float(losers.sum())) if len(losers) else gross_profit

    eq_series = pd.Series(equity_curve)
    peak = eq_series.cummax()
    drawdown = (eq_series - peak) / peak
    max_dd = float(drawdown.min()) * 100

    sharpe = (
        float(pnl_series.mean() / pnl_series.std() * np.sqrt(252)) if pnl_series.std() > 0 else 0
    )

    print(f"\n{'=' * 60}")
    print("  Portfolio Backtest Results")
    print(
        f"  Capital: ${args.cash:,.0f} | Max {_MAX_CONCURRENT} concurrent | Max {_MAX_TRADES_PER_DAY}/day"
    )
    print(f"{'=' * 60}")
    print(f"  Trades:         {total_trades}")
    print(f"  Win Rate:       {win_rate:.1f}%")
    print(f"  Profit Factor:  {pf:.3f}")
    print(f"  Sharpe Ratio:   {sharpe:.2f}")
    print(f"  Net P&L:        ${net_pnl:,.2f}")
    print(f"  Max Drawdown:   {max_dd:.1f}%")
    print(f"  Signals:        {signals_generated}")
    print(f"  Rejected (pos): {signals_rejected_position_limit}")
    print(f"  Rejected (sec): {signals_rejected_sector}")
    print(f"  Final Equity:   ${equity:,.2f}")
    print(f"{'=' * 60}")

    # Save results
    trades_df = pd.DataFrame(
        [
            {
                "symbol": t.symbol,
                "direction": t.direction,
                "entry": t.entry,
                "exit": t.exit_price,
                "pnl": t.pnl,
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
            }
            for t in closed_trades
        ]
    )
    trades_df.to_csv(out_dir / "portfolio_trades.csv", index=False)
    print(f"\nTrade log: {out_dir / 'portfolio_trades.csv'}")

    # Save equity curve
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(eq_series.values, linewidth=0.8)
        ax.set_title(f"Portfolio Equity Curve (PF={pf:.2f}, {total_trades} trades)")
        ax.set_ylabel("Equity ($)")
        ax.set_xlabel("Trading Days")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "portfolio_equity.png", dpi=150)
        print(f"Equity curve: {out_dir / 'portfolio_equity.png'}")
    except Exception as exc:
        print(f"Could not save equity curve: {exc}")


if __name__ == "__main__":
    main()
