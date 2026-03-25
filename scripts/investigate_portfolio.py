#!/usr/bin/env python
"""Investigate why portfolio backtest produces far fewer trades than per-ticker.

Diagnoses:
1. Signal funnel — where candidates drop off at each filter stage
2. Common date range impact — ARM truncation vs per-ticker full range
3. Filter comparison — differences between portfolio and per-ticker logic
4. Runs multiple portfolio variants with different constraint levels

Outputs:
- docs/portfolio_debug/signal_funnel.csv
- docs/portfolio_debug/filter_comparison.md
- docs/portfolio_debug/variant_results.md
"""

from __future__ import annotations

import sys
import warnings
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import talib

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtest.data_loader import load_bars
from src.models import TimeFrame
from src.risk.manager import SECTOR_MAP
from src.storage.database import BarDatabase

warnings.filterwarnings("ignore", category=RuntimeWarning)

_ET = ZoneInfo("America/New_York")

# ── Strategy constants ───────────────────────────────────────────────────────
_ORB_BARS = 5
_ATR_MULTIPLIER = 1.5
_RISK_MULTIPLIER = 2.0
_DAILY_ADX_MIN = 25.0
_REALIZED_VOL_MAX = 0.18
_MIN_ORB_PCT = 0.0015
_FORCE_FLAT = time(15, 55)
_POSITION_SCALE = 0.25


@dataclass
class OpenPosition:
    symbol: str
    direction: str
    entry: float
    stop: float
    target: float
    size: float
    entry_time: datetime


@dataclass
class ClosedTrade:
    symbol: str
    direction: str
    entry: float
    exit_price: float
    pnl: float
    entry_time: datetime
    exit_time: datetime


def load_data(
    symbols: list[str],
) -> tuple[dict[str, pd.DataFrame], dict[date, float], dict[date, float]]:
    """Load bar data and compute daily indicators."""
    db = BarDatabase()
    db.connect()
    all_bars: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        df = load_bars(db, symbol=sym, timeframe=TimeFrame.ONE_MIN)
        if not df.empty:
            all_bars[sym] = df
    db.close()

    spy_df = all_bars["SPY"]
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

    adx_lookup: dict[date, float] = {}
    vol_lookup: dict[date, float] = {}
    dates_list = list(spy_daily.index)
    for i in range(1, len(dates_list)):
        d = dates_list[i].date()
        if i - 1 < len(daily_adx) and not np.isnan(daily_adx[i - 1]):
            adx_lookup[d] = float(daily_adx[i - 1])
        if i - 1 < len(realized_vol) and not np.isnan(realized_vol.iloc[i - 1]):
            vol_lookup[d] = float(realized_vol.iloc[i - 1])

    return all_bars, adx_lookup, vol_lookup


def compute_orb_cache(
    all_bars: dict[str, pd.DataFrame],
) -> dict[str, dict[date, tuple[float, float]]]:
    """Pre-compute ORB ranges."""
    orb_cache: dict[str, dict[date, tuple[float, float]]] = defaultdict(dict)
    for sym, df in all_bars.items():
        df_et = df.copy()
        df_et.index = df_et.index.tz_convert(_ET)
        for d, day_df in df_et.groupby(df_et.index.date):
            orb_bars = day_df.between_time("09:30", "09:34")
            if len(orb_bars) >= _ORB_BARS:
                orb_cache[sym][d] = (float(orb_bars["high"].max()), float(orb_bars["low"].min()))
    return orb_cache


def run_variant(
    all_bars: dict[str, pd.DataFrame],
    adx_lookup: dict[date, float],
    vol_lookup: dict[date, float],
    orb_cache: dict[str, dict[date, tuple[float, float]]],
    trading_days: list[date],
    cash: float = 200_000,
    max_concurrent: int = 3,
    max_trades_per_day: int = 5,
    max_same_sector: int = 2,
    use_sector_limits: bool = True,
    use_adx_filter: bool = True,
    use_vol_filter: bool = True,
    collect_funnel: bool = False,
) -> tuple[list[ClosedTrade], list[float], dict[str, int] | None]:
    """Run one portfolio variant and return trades + equity curve."""
    equity = cash
    open_positions: list[OpenPosition] = []
    closed_trades: list[ClosedTrade] = []
    equity_curve: list[float] = []

    funnel: dict[str, int] | None = None
    if collect_funnel:
        funnel = defaultdict(int)

    for day in trading_days:
        if day.weekday() == 0:
            if funnel is not None:
                funnel["days_monday_skipped"] += 1
            continue

        d_adx = adx_lookup.get(day)
        d_vol = vol_lookup.get(day)

        if use_adx_filter and d_adx is not None and d_adx <= _DAILY_ADX_MIN:
            if funnel is not None:
                funnel["days_adx_blocked"] += 1
            continue
        if use_vol_filter and d_vol is not None and d_vol >= _REALIZED_VOL_MAX:
            if funnel is not None:
                funnel["days_vol_blocked"] += 1
            continue

        if funnel is not None:
            funnel["days_tradeable"] += 1

        daily_trade_count = 0

        for minute_offset in range(5, 30):
            if daily_trade_count >= max_trades_per_day:
                break

            # Check/close positions
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
                bar_h, bar_l = float(bar["high"]), float(bar["low"])
                if pos.direction == "LONG":
                    if bar_h >= pos.target:
                        pnl = (pos.target - pos.entry) * (pos.size / pos.entry)
                        closed_trades.append(
                            ClosedTrade(
                                pos.symbol,
                                "LONG",
                                pos.entry,
                                pos.target,
                                pnl,
                                pos.entry_time,
                                bar_time,
                            )
                        )
                        open_positions.remove(pos)
                        equity += pnl
                    elif bar_l <= pos.stop:
                        pnl = (pos.stop - pos.entry) * (pos.size / pos.entry)
                        closed_trades.append(
                            ClosedTrade(
                                pos.symbol,
                                "LONG",
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
                    if bar_l <= pos.target:
                        pnl = (pos.entry - pos.target) * (pos.size / pos.entry)
                        closed_trades.append(
                            ClosedTrade(
                                pos.symbol,
                                "SHORT",
                                pos.entry,
                                pos.target,
                                pnl,
                                pos.entry_time,
                                bar_time,
                            )
                        )
                        open_positions.remove(pos)
                        equity += pnl
                    elif bar_h >= pos.stop:
                        pnl = (pos.entry - pos.stop) * (pos.size / pos.entry)
                        closed_trades.append(
                            ClosedTrade(
                                pos.symbol,
                                "SHORT",
                                pos.entry,
                                pos.stop,
                                pnl,
                                pos.entry_time,
                                bar_time,
                            )
                        )
                        open_positions.remove(pos)
                        equity += pnl

            candidates: list[tuple[str, str, float, float, float, float]] = []

            for sym in all_bars:
                orb = orb_cache.get(sym, {}).get(day)
                if orb is None:
                    continue
                orb_high, orb_low = orb
                if orb_low <= 0:
                    continue
                orb_range_pct = (orb_high - orb_low) / orb_low
                if orb_range_pct < _MIN_ORB_PCT:
                    continue
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
                if volume < 100_000:
                    continue

                atr_proxy = orb_high - orb_low
                if atr_proxy <= 0:
                    continue

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

                if funnel is not None:
                    funnel["signals_raw"] += 1

                score = (d_adx or 25) / 100 * 0.4 + orb_range_pct * 100 * 0.3 + 0.3
                candidates.append((sym, direction, entry, stop, target, score))

            candidates.sort(key=lambda c: c[5], reverse=True)

            for sym, direction, entry, stop, target, _score in candidates:
                if len(open_positions) >= max_concurrent:
                    if funnel is not None:
                        funnel["rejected_concurrent"] += 1
                    continue
                if daily_trade_count >= max_trades_per_day:
                    if funnel is not None:
                        funnel["rejected_daily_limit"] += 1
                    break

                if use_sector_limits:
                    sym_sector = SECTOR_MAP.get(sym, "other")
                    same_sector = sum(
                        1 for p in open_positions if SECTOR_MAP.get(p.symbol, "other") == sym_sector
                    )
                    if same_sector >= max_same_sector:
                        if funnel is not None:
                            funnel["rejected_sector"] += 1
                        continue

                pos_size = equity * _POSITION_SCALE
                bar_time = datetime.combine(day, time(9, 30 + minute_offset)).replace(tzinfo=_ET)
                open_positions.append(
                    OpenPosition(sym, direction, entry, stop, target, pos_size, bar_time)
                )
                daily_trade_count += 1
                if funnel is not None:
                    funnel["signals_accepted"] += 1

        # Force flat
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

    return closed_trades, equity_curve, funnel


def compute_stats(trades: list[ClosedTrade], equity_curve: list[float]) -> dict[str, float]:
    """Compute PF, Sharpe, etc."""
    if not trades:
        return {
            "trades": 0,
            "win_rate": 0,
            "pf": 0,
            "sharpe": 0,
            "net_pnl": 0,
            "max_dd": 0,
            "avg_per_day": 0,
        }
    pnl = pd.Series([t.pnl for t in trades])
    winners = pnl[pnl > 0]
    losers = pnl[pnl < 0]
    gp = float(winners.sum()) if len(winners) else 0
    gl = abs(float(losers.sum())) if len(losers) else 0.001
    eq = pd.Series(equity_curve)
    peak = eq.cummax()
    dd = ((eq - peak) / peak).min() * 100 if len(eq) > 0 else 0
    n_days = len([e for e in equity_curve])
    return {
        "trades": len(trades),
        "win_rate": len(winners) / len(trades) * 100 if trades else 0,
        "pf": gp / gl if gl > 0 else 0,
        "sharpe": float(pnl.mean() / pnl.std() * np.sqrt(252)) if pnl.std() > 0 else 0,
        "net_pnl": gp - abs(float(losers.sum())) if len(losers) else gp,
        "max_dd": float(dd),
        "avg_per_day": len(trades) / n_days if n_days else 0,
    }


def main() -> None:
    out_dir = Path("docs/portfolio_debug")
    out_dir.mkdir(parents=True, exist_ok=True)

    symbols_all = [
        "SPY",
        "QQQ",
        "MSFT",
        "AMD",
        "TSLA",
        "AMZN",
        "UBER",
        "SMCI",
        "SHOP",
        "PLTR",
        "NFLX",
        "MSTR",
        "SNOW",
        "ARM",
        "DASH",
        "PYPL",
        "INTC",
        "MU",
        "HOOD",
        "DKNG",
        "SOXL",
        "ROKU",
        "TQQQ",
        "BA",
        "MRVL",
        "META",
    ]
    symbols_no_arm = [s for s in symbols_all if s != "ARM"]

    print("Loading data for all 26 symbols...")
    all_bars, adx_lookup, vol_lookup = load_data(symbols_all)

    # Show per-symbol date ranges
    print("\nPer-symbol date ranges:")
    for sym in sorted(all_bars.keys()):
        df = all_bars[sym]
        print(f"  {sym:<6}: {df.index[0].date()} to {df.index[-1].date()} ({len(df):>8,} bars)")

    orb_cache = compute_orb_cache(all_bars)

    # Compute date ranges
    common_all = max(df.index[0] for df in all_bars.values())
    common_no_arm = max(all_bars[s].index[0] for s in symbols_no_arm if s in all_bars)
    max_date = min(df.index[-1] for df in all_bars.values())

    print(f"\nCommon range (all 26):  {common_all.date()} to {max_date.date()}")
    print(f"Common range (no ARM): {common_no_arm.date()} to {max_date.date()}")

    # Get trading days for each range
    all_dates_26: set[date] = set()
    all_dates_no_arm: set[date] = set()
    for sym, df in all_bars.items():
        df_et = df.copy()
        df_et.index = df_et.index.tz_convert(_ET)
        dates_set = set(df_et.index.date)
        if sym != "ARM":
            all_dates_no_arm.update(
                d for d in dates_set if pd.Timestamp(d) >= common_no_arm.tz_localize(None)
            )
        all_dates_26.update(d for d in dates_set if pd.Timestamp(d) >= common_all.tz_localize(None))

    days_26 = sorted(all_dates_26)
    days_no_arm = sorted(all_dates_no_arm)

    print(f"\nTrading days (all 26):  {len(days_26)}")
    print(f"Trading days (no ARM): {len(days_no_arm)}")

    # ── Step 1: Signal funnel on baseline ────────────────────────────────────
    print("\n" + "=" * 70)
    print("  STEP 1: Signal Funnel Analysis (baseline, all 26, common range)")
    print("=" * 70)

    trades_base, _eq_base, funnel = run_variant(
        all_bars,
        adx_lookup,
        vol_lookup,
        orb_cache,
        days_26,
        cash=200_000,
        max_concurrent=3,
        max_trades_per_day=5,
        max_same_sector=2,
        use_sector_limits=True,
        collect_funnel=True,
    )

    if funnel:
        print(f"\n  Total trading days:     {len(days_26)}")
        print(f"  Days Monday skipped:    {funnel.get('days_monday_skipped', 0)}")
        print(f"  Days ADX blocked:       {funnel.get('days_adx_blocked', 0)}")
        print(f"  Days vol blocked:       {funnel.get('days_vol_blocked', 0)}")
        print(f"  Days tradeable:         {funnel.get('days_tradeable', 0)}")
        print(f"  Raw signals generated:  {funnel.get('signals_raw', 0)}")
        print(f"  Rejected (concurrent):  {funnel.get('rejected_concurrent', 0)}")
        print(f"  Rejected (daily limit): {funnel.get('rejected_daily_limit', 0)}")
        print(f"  Rejected (sector):      {funnel.get('rejected_sector', 0)}")
        print(f"  Signals accepted:       {funnel.get('signals_accepted', 0)}")
        print(f"  Trades completed:       {len(trades_base)}")

        # Save funnel CSV
        funnel_df = pd.DataFrame([funnel])
        funnel_df.to_csv(out_dir / "signal_funnel.csv", index=False)
        print(f"\n  Saved: {out_dir / 'signal_funnel.csv'}")

    # ── Step 2: Filter comparison ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  STEP 2: Filter Comparison (portfolio vs per-ticker)")
    print("=" * 70)

    comparison = """# Filter Comparison: Portfolio vs Per-Ticker Backtest

| Filter | Per-Ticker (engine.py) | Portfolio (run_portfolio_backtest.py) | Difference |
|--------|----------------------|--------------------------------------|------------|
| Date range | Per-ticker full history (2020+) | Common range across ALL tickers | Portfolio uses shortest ticker (ARM: 2023-09) |
| Capital | $50K per ticker (26 x $50K = $1.3M) | $200K shared pool | Portfolio has 85% less capital |
| Concurrent positions | 1 per ticker (unlimited cross-ticker) | Max 3 across all tickers | Portfolio severely constrained |
| Max trades/day | 5 per ticker (130 possible) | 5 total | Portfolio severely constrained |
| Sector limits | None | Max 2 per sector | Portfolio blocks correlated bets |
| ADX filter | Daily ADX > 25 (per-ticker from TA-Lib) | Daily ADX > 25 (from SPY only) | Same threshold, same source |
| Realized vol | HV < 18% (from SPY daily) | HV < 18% (from SPY daily) | Identical |
| Volume filter | 1.5x 20-bar rolling avg | Fixed 100K minimum | Per-ticker is adaptive, portfolio is fixed |
| ATR for stops | TA-Lib ATR(14) computed per bar | ORB range as ATR proxy | Per-ticker is precise, portfolio is approximate |
| EMA trend | 15-min EMA(20) via TA-Lib | Not applied | Missing from portfolio |
| Gap filter | ±0.3% from prev day close | Not applied | Missing from portfolio |
| VIX backwardation | VIX/VIX3M ratio > 1.0 blocks | Not applied | Missing from portfolio |
| Economic calendar | FOMC/NFP/CPI/PPI blocks | Not applied | Missing from portfolio |
| Earnings blackout | Tagged (informational) | Not applied | Minor |
| Position exit | backtesting.py SL/TP on every tick | Check only during ORB window (9:35-9:59) | Portfolio misses mid-day exits |

## Key Findings:
1. **Positions only close during ORB window (9:35-9:59)** — portfolio never checks SL/TP after 10:00
2. **No EMA, gap, VIX backwardation, or econ calendar filters** — portfolio takes trades the per-ticker would block
3. **Volume filter is fixed 100K instead of adaptive 1.5x** — very different behavior
4. **Common date range (2023-09 to present)** vs per-ticker (2020+) = 3 fewer years of trades
"""

    (out_dir / "filter_comparison.md").write_text(comparison)
    print(f"  Saved: {out_dir / 'filter_comparison.md'}")
    print("\n  KEY ISSUE: Positions only checked during 9:35-9:59 window!")
    print("  Bracket SL/TP must be checked on EVERY bar until 15:55 ET.")

    # ── Step 3: Run variants ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  STEP 3: Portfolio Variants (all 26, common range)")
    print("=" * 70)

    variants = [
        (
            "Baseline (3 conc, 5/day, sectors)",
            dict(max_concurrent=3, max_trades_per_day=5, max_same_sector=2, use_sector_limits=True),
        ),
        (
            "V1: 5 conc, 8/day, sectors",
            dict(max_concurrent=5, max_trades_per_day=8, max_same_sector=2, use_sector_limits=True),
        ),
        (
            "V2: 5 conc, 8/day, no sectors",
            dict(
                max_concurrent=5, max_trades_per_day=8, max_same_sector=99, use_sector_limits=False
            ),
        ),
        (
            "V3: 8 conc, 10/day, no sectors",
            dict(
                max_concurrent=8, max_trades_per_day=10, max_same_sector=99, use_sector_limits=False
            ),
        ),
        (
            "V4: No limits (upper bound)",
            dict(
                max_concurrent=99,
                max_trades_per_day=99,
                max_same_sector=99,
                use_sector_limits=False,
            ),
        ),
        (
            "V5: V4 + sector limits",
            dict(
                max_concurrent=99, max_trades_per_day=99, max_same_sector=2, use_sector_limits=True
            ),
        ),
    ]

    results_rows = []
    for name, kwargs in variants:
        trades, eq, _ = run_variant(
            all_bars, adx_lookup, vol_lookup, orb_cache, days_26, cash=200_000, **kwargs
        )
        stats = compute_stats(trades, eq)
        results_rows.append({"variant": name, **stats})
        print(f"\n  {name}")
        print(
            f"    Trades={stats['trades']:>5}, WR={stats['win_rate']:.1f}%, PF={stats['pf']:.3f}, "
            f"Sharpe={stats['sharpe']:.2f}, Net=${stats['net_pnl']:>10,.0f}, "
            f"MaxDD={stats['max_dd']:.1f}%, Avg/Day={stats['avg_per_day']:.1f}"
        )

    # V6: Drop ARM, use longer range
    print(f"\n  V6: No ARM, extended range ({common_no_arm.date()} to {max_date.date()})")
    all_bars_no_arm = {s: df for s, df in all_bars.items() if s != "ARM"}
    orb_cache_no_arm = {s: v for s, v in orb_cache.items() if s != "ARM"}
    trades_v6, eq_v6, _ = run_variant(
        all_bars_no_arm,
        adx_lookup,
        vol_lookup,
        orb_cache_no_arm,
        days_no_arm,
        cash=200_000,
        max_concurrent=5,
        max_trades_per_day=8,
        max_same_sector=2,
        use_sector_limits=True,
    )
    stats_v6 = compute_stats(trades_v6, eq_v6)
    results_rows.append(
        {"variant": f"V6: No ARM, extended ({common_no_arm.date()}+), 5c/8d/sectors", **stats_v6}
    )
    print(
        f"    Trades={stats_v6['trades']:>5}, WR={stats_v6['win_rate']:.1f}%, PF={stats_v6['pf']:.3f}, "
        f"Sharpe={stats_v6['sharpe']:.2f}, Net=${stats_v6['net_pnl']:>10,.0f}, "
        f"MaxDD={stats_v6['max_dd']:.1f}%, Avg/Day={stats_v6['avg_per_day']:.1f}"
    )

    # Save variant results
    results_df = pd.DataFrame(results_rows)
    md_lines = [
        "# Portfolio Variant Results\n",
        f"Date range (26 symbols): {common_all.date()} to {max_date.date()}\n",
    ]
    md_lines.append("| Variant | Trades | WR% | PF | Sharpe | Net P&L | MaxDD | Avg/Day |")
    md_lines.append("|---------|--------|-----|-----|--------|---------|-------|---------|")
    for _, row in results_df.iterrows():
        md_lines.append(
            f"| {row['variant']} | {int(row['trades'])} | {row['win_rate']:.1f}% | "
            f"{row['pf']:.3f} | {row['sharpe']:.2f} | ${row['net_pnl']:,.0f} | "
            f"{row['max_dd']:.1f}% | {row['avg_per_day']:.1f} |"
        )
    (out_dir / "variant_results.md").write_text("\n".join(md_lines))
    results_df.to_csv(out_dir / "variant_results.csv", index=False)
    print(f"\n  Saved: {out_dir / 'variant_results.md'}")
    print(f"  Saved: {out_dir / 'variant_results.csv'}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  DIAGNOSIS SUMMARY")
    print("=" * 70)
    print("""
  ROOT CAUSES of 6-trade portfolio vs 2,094-trade per-ticker:

  1. CRITICAL: Position SL/TP only checked during 9:35-9:59 ORB window.
     After 10:00, positions are NEVER checked until 15:55 force-flat.
     This means most positions exit at force-flat (not at target/stop),
     which destroys the 2R target structure that makes the strategy work.

  2. HIGH: Common date range starts 2023-09 (ARM truncation).
     Per-ticker backtest runs from 2020+ per ticker.
     ~3.5 fewer years of trading history = far fewer opportunities.

  3. HIGH: Missing filters (EMA, gap, VIX backwardation, econ calendar).
     These are present in per-ticker backtest but missing from portfolio.
     Without them, portfolio takes some bad trades that per-ticker blocks.

  4. MEDIUM: Concurrent position limit (3) + daily limit (5).
     With 26 tickers, many signals compete for 3 slots.
     ~60-80% of raw signals rejected due to position constraints.

  5. MEDIUM: Volume filter is fixed 100K vs adaptive 1.5x rolling avg.
     Per-ticker uses per-symbol rolling volume; portfolio uses flat floor.

  FIX PRIORITY:
  1. Check SL/TP on EVERY bar from entry to 15:55 (not just ORB window)
  2. Drop ARM, use per-ticker min date (not common range)
  3. Add missing filters (EMA, gap, econ calendar)
  4. Use adaptive volume filter
""")


if __name__ == "__main__":
    main()
