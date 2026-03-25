#!/usr/bin/env python
"""Portfolio-level backtest: shared capital, position limits, signal ranking.

Unlike ``run_backtest.py`` which runs 26 independent per-ticker backtests,
this simulates a single portfolio with:
- Shared capital pool (default $200K)
- Max concurrent positions (default 3)
- Max trades per day (default 5)
- Signal ranking when multiple candidates fire
- Sector concentration limits
- Bar-by-bar ORB window: entries and exits interleaved (slot recycling)
- Per-bar ATR(14) shifted by 1 (matching engine.py)
- 15-min EMA(20) shifted by 1 bar (no lookahead)
- Post-ORB exit sweep on all bars through 15:54

Usage
-----
    python scripts/run_portfolio_backtest.py --cash 200000
    python scripts/run_portfolio_backtest.py --max-concurrent 5 --max-daily 8
"""

from __future__ import annotations

import argparse
import sys
import warnings
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import structlog
import talib

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtest.data_loader import load_bars
from src.filters.economic_calendar import compute_econ_blocked_array
from src.filters.vix_term_structure import (
    compute_vix_term_structure_array,
    get_vix_term_structure,
)
from src.levels.kalman_levels import StreamingKalman
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
_POSITION_SCALE = 0.25
_EMA15M_PERIOD = 20
_VIX_SPOT_MAX = 25.0  # VIX close >= 25 → skip all entries (matching live _VIX_MAX)
_RISK_PCT = 1.0  # 1% of account risked per trade (matching live)
_SLIPPAGE_PER_SHARE = 0.02  # $0.02/share per side

# Trading window — 9:35-10:00 ET only (matching engine.py)
_WINDOW_START = time(9, 35)
_WINDOW_END = time(10, 0)
_FORCE_FLAT = time(15, 55)


@dataclass
class Position:
    symbol: str
    direction: str
    entry: float
    stop: float
    target: float
    size: float
    entry_time: datetime


@dataclass
class Trade:
    symbol: str
    direction: str
    entry: float
    exit_price: float
    pnl: float
    entry_time: datetime
    exit_time: datetime
    exit_reason: str


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Portfolio-level ORB backtest")
    p.add_argument("--cash", type=float, default=200_000)
    p.add_argument("--max-concurrent", type=int, default=3)
    p.add_argument("--max-daily", type=int, default=5)
    p.add_argument("--max-sector", type=int, default=2)
    p.add_argument("--no-adx", action="store_true", help="Disable ADX filter")
    p.add_argument("--no-monday", action="store_true", help="Disable Monday block")
    p.add_argument("--out-dir", default="docs/portfolio_backtest")
    p.add_argument(
        "--symbols",
        default="SPY,QQQ,MSFT,AMD,TSLA,AMZN,UBER,SMCI,SHOP,PLTR,NFLX,MSTR,SNOW,ARM,DASH,PYPL,INTC,MU,HOOD,DKNG,SOXL,ROKU,TQQQ,BA,MRVL,META",
    )
    return p.parse_args()


def _check_exits(
    open_positions: list[Position],
    day_bars: dict[str, dict[date, pd.DataFrame]],
    day: date,
    after_time: datetime,
    before_time_str: str,
    stats_counters: dict[str, int],
    closed_trades: list[Trade],
) -> float:
    """Check SL/TP for all open positions on bars in a time range. Returns equity delta."""
    equity_delta = 0.0
    for pos in list(open_positions):
        pos_day_bars = day_bars.get(pos.symbol, {}).get(day)
        if pos_day_bars is None:
            continue
        after_entry = pos_day_bars[pos_day_bars.index > after_time]
        market_bars = after_entry.between_time("09:30", before_time_str)
        for _, bar_row in market_bars.iterrows():
            bar_o = float(bar_row["open"])
            bar_h = float(bar_row["high"])
            bar_l = float(bar_row["low"])
            bar_c = float(bar_row["close"])
            btime = bar_row.name

            hit_target = False
            hit_stop = False
            if pos.direction == "LONG":
                hit_target = bar_h >= pos.target
                hit_stop = bar_l <= pos.stop
            else:
                hit_target = bar_l <= pos.target
                hit_stop = bar_h >= pos.stop

            if hit_target and hit_stop:
                favor = bar_c > bar_o if pos.direction == "LONG" else bar_c < bar_o
                if favor:
                    hit_stop = False
                else:
                    hit_target = False

            if hit_target:
                pnl = (
                    (pos.target - pos.entry)
                    if pos.direction == "LONG"
                    else (pos.entry - pos.target)
                ) * pos.size
                closed_trades.append(
                    Trade(
                        pos.symbol,
                        pos.direction,
                        pos.entry,
                        pos.target,
                        pnl,
                        pos.entry_time,
                        btime,
                        "target",
                    )
                )
                open_positions.remove(pos)
                equity_delta += pnl
                stats_counters["exits_target"] += 1
                break
            elif hit_stop:
                pnl = (
                    (pos.stop - pos.entry) if pos.direction == "LONG" else (pos.entry - pos.stop)
                ) * pos.size
                closed_trades.append(
                    Trade(
                        pos.symbol,
                        pos.direction,
                        pos.entry,
                        pos.stop,
                        pnl,
                        pos.entry_time,
                        btime,
                        "stop",
                    )
                )
                open_positions.remove(pos)
                equity_delta += pnl
                stats_counters["exits_stop"] += 1
                break
    return equity_delta


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    symbols = [s.strip().upper() for s in args.symbols.split(",")]
    max_concurrent = args.max_concurrent
    max_daily = args.max_daily
    max_sector = args.max_sector

    db = BarDatabase()
    db.connect()

    # ── Load data ────────────────────────────────────────────────────────────
    print(f"Loading bars for {len(symbols)} symbols...")
    raw_bars: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        df = load_bars(db, symbol=sym, timeframe=TimeFrame.ONE_MIN)
        if not df.empty:
            raw_bars[sym] = df
            print(f"  {sym}: {len(df):,} bars ({df.index[0].date()} to {df.index[-1].date()})")
    db.close()

    # ── Pre-index: convert to ET, build day→bars lookup for speed ────────
    print("\nPre-indexing bars by date...")
    day_bars: dict[str, dict[date, pd.DataFrame]] = defaultdict(dict)
    all_trading_dates: set[date] = set()

    for sym, df in raw_bars.items():
        df_et = df.copy()
        df_et.index = df_et.index.tz_convert(_ET)
        for d, grp in df_et.groupby(df_et.index.date):
            day_bars[sym][d] = grp
            all_trading_dates.add(d)

    trading_days = sorted(all_trading_dates)
    print(f"  {len(trading_days)} unique trading days across all symbols")

    # ── Pre-compute regime indicators (SPY daily ADX + realized vol) ──────
    spy_df = raw_bars.get("SPY")
    if spy_df is None:
        print("SPY data required.")
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

    # Shifted 1 day (no lookahead)
    adx_lookup: dict[date, float] = {}
    vol_lookup: dict[date, float] = {}
    dl = list(spy_daily.index)
    for i in range(1, len(dl)):
        d = dl[i].date()
        if not np.isnan(daily_adx[i - 1]):
            adx_lookup[d] = float(daily_adx[i - 1])
        if not np.isnan(realized_vol.iloc[i - 1]):
            vol_lookup[d] = float(realized_vol.iloc[i - 1])

    # ── FIX #1: Per-bar ATR(14) shifted by 1, indexed by (symbol, bar_index) ──
    # engine.py: atr_raw = talib.ATR(H, L, C, 14); self.atr = np.roll(atr_raw, 1)
    # We store the shifted ATR directly in each symbol's ET DataFrame.
    print("Pre-computing per-bar ATR(14) shifted by 1 (matching engine.py)...")
    # Store as column in day_bars DataFrames
    for sym, df in raw_bars.items():
        h = df["high"].astype(float).values
        lo = df["low"].astype(float).values
        c = df["close"].astype(float).values
        if len(c) < _ATR_PERIOD + 1:
            continue
        atr_raw = talib.ATR(h, lo, c, timeperiod=_ATR_PERIOD)
        atr_shifted = np.roll(atr_raw, 1)
        atr_shifted[0] = np.nan

        # Also compute rolling avg vol shifted by 1 (matching engine.py)
        vol_series = pd.Series(df["volume"].astype(float).values)
        avg_vol_raw = vol_series.rolling(_VOL_WINDOW).mean().to_numpy()
        avg_vol_shifted = np.roll(avg_vol_raw, 1)
        avg_vol_shifted[0] = np.nan

        df_et = df.copy()
        df_et.index = df_et.index.tz_convert(_ET)
        df_et["_atr"] = atr_shifted
        df_et["_avg_vol"] = avg_vol_shifted

        # Re-split into day_bars with the new columns
        for d, grp in df_et.groupby(df_et.index.date):
            day_bars[sym][d] = grp

    # ── Pre-compute ORB ranges ───────────────────────────────────────────
    print("Pre-computing ORB ranges...")
    orb_cache: dict[str, dict[date, tuple[float, float]]] = defaultdict(dict)

    for sym in raw_bars:
        for d, grp in day_bars.get(sym, {}).items():
            orb_bars_df = grp.between_time("09:30", "09:34")
            if len(orb_bars_df) >= _ORB_BARS:
                orb_cache[sym][d] = (
                    float(orb_bars_df["high"].max()),
                    float(orb_bars_df["low"].min()),
                )

    # ── FIX #3: 15-min EMA(20) shifted by 1, mapped to 1-min bars ───────
    # Matching engine.py: _compute_15m_ema() → resample 15min, EMA, shift(1), ffill to 1min
    print("Pre-computing 15-min EMA(20) shifted by 1 bar (matching engine.py)...")
    for sym, _df in raw_bars.items():
        df_et = None
        # Reconstruct full ET-indexed series from day_bars
        all_day_dfs = []
        for d in sorted(day_bars.get(sym, {}).keys()):
            all_day_dfs.append(day_bars[sym][d])
        if not all_day_dfs:
            continue
        df_et = pd.concat(all_day_dfs)
        close_series = df_et["close"].astype(float)

        close_15m = close_series.resample("15min").last().dropna()
        if len(close_15m) < _EMA15M_PERIOD:
            continue

        ema_raw = talib.EMA(close_15m.to_numpy(dtype=float), timeperiod=_EMA15M_PERIOD)
        ema_15m = pd.Series(ema_raw, index=close_15m.index)
        ema_shifted = ema_15m.shift(1)
        ema_1m = ema_shifted.reindex(df_et.index, method="ffill")

        # Add as column to day_bars
        for d, grp in df_et.groupby(df_et.index.date):
            grp = grp.copy()
            grp["_ema15m"] = ema_1m.reindex(grp.index)
            day_bars[sym][d] = grp

    # ── Pre-compute gap cache ────────────────────────────────────────────
    print("Pre-computing gap, VIX term structure, econ calendar...")
    gap_cache: dict[str, dict[date, float]] = defaultdict(dict)

    for sym in raw_bars:
        prev_close_val = None
        for d in sorted(day_bars.get(sym, {}).keys()):
            grp = day_bars[sym][d]
            first_bar = grp.between_time("09:30", "09:30")
            if not first_bar.empty and prev_close_val is not None:
                session_open = float(first_bar.iloc[0]["open"])
                gap_pct = (session_open - prev_close_val) / prev_close_val * 100
                gap_cache[sym][d] = gap_pct
            last_bar = grp.between_time("15:55", "15:59")
            if not last_bar.empty:
                prev_close_val = float(last_bar.iloc[-1]["close"])
            elif len(grp) > 0:
                prev_close_val = float(grp.iloc[-1]["close"])

    # VIX term structure (for backwardation filter)
    vts_df = get_vix_term_structure()
    vts_lookup: dict[date, float] = {}
    if not vts_df.empty:
        vts_arr = compute_vix_term_structure_array(spy_df.index, vts_df)
        spy_et_for_vts = spy_df.copy()
        spy_et_for_vts.index = spy_et_for_vts.index.tz_convert(_ET)
        spy_et_for_vts["_vts"] = vts_arr
        for d, grp in spy_et_for_vts.groupby(spy_et_for_vts.index.date):
            vals = grp["_vts"].dropna()
            if len(vals) > 0:
                vts_lookup[d] = float(vals.iloc[0])
        print(f"  VIX term structure: {len(vts_lookup)} days")

    # VIX spot level: D-1 VIX close for >= 25 hard block (matching live _VIX_MAX)
    vix_spot_lookup: dict[date, float] = {}
    if not vts_df.empty and "vix" in vts_df.columns:
        vix_shifted = vts_df["vix"].shift(1)
        for d_ts, v in vix_shifted.items():
            if not np.isnan(float(v)):
                d_val = d_ts.date() if hasattr(d_ts, "date") else d_ts
                vix_spot_lookup[d_val] = float(v)
        print(f"  VIX spot lookup: {len(vix_spot_lookup)} days")

    # Economic calendar
    econ_blocked_arr = compute_econ_blocked_array(spy_df.index)
    econ_blocked_dates: set[date] = set()
    spy_et_for_econ = spy_df.copy()
    spy_et_for_econ.index = spy_et_for_econ.index.tz_convert(_ET)
    spy_et_for_econ["_econ"] = econ_blocked_arr
    for d, grp in spy_et_for_econ.groupby(spy_et_for_econ.index.date):
        if grp["_econ"].max() > 0.5:
            econ_blocked_dates.add(d)
    print(f"  Econ blocked days: {len(econ_blocked_dates)}")

    # ── Portfolio simulation ─────────────────────────────────────────────────
    # FIX #2: Bar-by-bar during ORB window (slot recycling), then batch exits.
    # For each minute in 9:35-9:59:
    #   1. Check SL/TP on open positions (bars since last check through current minute)
    #   2. If slots open, generate and accept new entries
    # After ORB window:
    #   3. Batch check SL/TP on all remaining bars (10:00-15:54)
    #   4. Force flat at 15:55
    print(
        f"\nSimulating portfolio (${args.cash:,.0f}, max {max_concurrent} conc, {max_daily}/day)..."
    )

    equity = args.cash
    open_positions: list[Position] = []
    closed_trades: list[Trade] = []
    equity_curve: list[float] = []
    stats_counters: dict[str, int] = defaultdict(int)

    adx_enabled = not args.no_adx
    monday_enabled = not args.no_monday

    for day in trading_days:
        if monday_enabled and day.weekday() == 0:
            stats_counters["days_monday"] += 1
            continue

        d_adx = adx_lookup.get(day)
        d_vol = vol_lookup.get(day)
        if adx_enabled and d_adx is not None and d_adx <= _DAILY_ADX_MIN:
            stats_counters["days_adx_blocked"] += 1
            continue
        if d_vol is not None and d_vol >= _REALIZED_VOL_MAX:
            stats_counters["days_vol_blocked"] += 1
            continue
        if day in econ_blocked_dates:
            stats_counters["days_econ_blocked"] += 1
            continue
        vts_ratio = vts_lookup.get(day)
        if vts_ratio is not None and vts_ratio > 1.0:
            stats_counters["days_backwardation_blocked"] += 1
            continue
        vix_spot = vix_spot_lookup.get(day)
        if vix_spot is not None and vix_spot >= _VIX_SPOT_MAX:
            stats_counters["days_vix_blocked"] += 1
            continue

        stats_counters["days_tradeable"] += 1
        daily_trade_count = 0

        # Initialize Kalman filters for each symbol on this day
        kalman_state: dict[str, StreamingKalman] = {}
        for sym in raw_bars:
            orb = orb_cache.get(sym, {}).get(day)
            if orb is None:
                continue
            sym_day = day_bars.get(sym, {}).get(day)
            if sym_day is None:
                continue
            # ATR at 9:34 (shifted by 1) for Kalman init
            orb_end = sym_day.between_time("09:34", "09:34")
            if orb_end.empty or "_atr" not in sym_day.columns:
                continue
            sym_atr = float(orb_end.iloc[0]["_atr"])
            if np.isnan(sym_atr) or sym_atr <= 0:
                continue
            orb_bars_df = sym_day.between_time("09:30", "09:34")
            if len(orb_bars_df) >= _ORB_BARS:
                km = StreamingKalman()
                km.reset([float(r["close"]) for _, r in orb_bars_df.iterrows()], sym_atr)
                kalman_state[sym] = km

        # ── ORB window: bar-by-bar with interleaved exits + entries ──────
        # Track the last time we checked exits (to avoid re-scanning bars)
        for minute_offset in range(5, 30):
            bar_time = datetime.combine(day, time(9, 30 + minute_offset)).replace(tzinfo=_ET)

            # Step A: Check SL/TP on open positions for bars up to current minute
            for pos in list(open_positions):
                pos_day_bars = day_bars.get(pos.symbol, {}).get(day)
                if pos_day_bars is None:
                    continue
                # Check bars from entry (or last checked) through current minute
                check_from = pos.entry_time
                after_entry = pos_day_bars[pos_day_bars.index > check_from]
                # Only check up to current bar (inclusive)
                in_range = after_entry[after_entry.index <= bar_time]
                if in_range.empty:
                    continue

                for _, bar_row in in_range.iterrows():
                    bar_o = float(bar_row["open"])
                    bar_h = float(bar_row["high"])
                    bar_l = float(bar_row["low"])
                    bar_c = float(bar_row["close"])
                    btime = bar_row.name

                    hit_target = False
                    hit_stop = False
                    if pos.direction == "LONG":
                        hit_target = bar_h >= pos.target
                        hit_stop = bar_l <= pos.stop
                    else:
                        hit_target = bar_l <= pos.target
                        hit_stop = bar_h >= pos.stop

                    if hit_target and hit_stop:
                        favor = bar_c > bar_o if pos.direction == "LONG" else bar_c < bar_o
                        if favor:
                            hit_stop = False
                        else:
                            hit_target = False

                    if hit_target:
                        pnl = (
                            (pos.target - pos.entry)
                            if pos.direction == "LONG"
                            else (pos.entry - pos.target)
                        ) * pos.size - _SLIPPAGE_PER_SHARE * pos.size
                        closed_trades.append(
                            Trade(
                                pos.symbol,
                                pos.direction,
                                pos.entry,
                                pos.target,
                                pnl,
                                pos.entry_time,
                                btime,
                                "target",
                            )
                        )
                        open_positions.remove(pos)
                        equity += pnl
                        stats_counters["exits_target"] += 1
                        break
                    elif hit_stop:
                        pnl = (
                            (pos.stop - pos.entry)
                            if pos.direction == "LONG"
                            else (pos.entry - pos.stop)
                        ) * pos.size - _SLIPPAGE_PER_SHARE * pos.size
                        closed_trades.append(
                            Trade(
                                pos.symbol,
                                pos.direction,
                                pos.entry,
                                pos.stop,
                                pnl,
                                pos.entry_time,
                                btime,
                                "stop",
                            )
                        )
                        open_positions.remove(pos)
                        equity += pnl
                        stats_counters["exits_stop"] += 1
                        break

            # Step B: Generate entry signals if slots available
            if daily_trade_count >= max_daily or len(open_positions) >= max_concurrent:
                continue  # `continue` not `break` — slots may open on next minute

            candidates: list[tuple[str, str, float, float, float, float]] = []

            for sym in raw_bars:
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

                sym_day = day_bars.get(sym, {}).get(day)
                if sym_day is None:
                    continue

                # Find current and previous bar
                target_time_val = time(9, 30 + minute_offset)
                prev_time_val = time(9, 29 + minute_offset)
                curr = sym_day.between_time(target_time_val, target_time_val)
                prev = sym_day.between_time(prev_time_val, prev_time_val)
                if curr.empty or prev.empty:
                    continue

                close_val = float(curr.iloc[0]["close"])
                prev_close = float(curr.iloc[0]["close"] if prev.empty else prev.iloc[0]["close"])
                if not prev.empty:
                    prev_close = float(prev.iloc[0]["close"])
                volume = float(curr.iloc[0]["volume"])

                # FIX #1: Per-bar rolling avg volume shifted by 1 (matching engine.py)
                if "_avg_vol" not in curr.columns:
                    continue
                avg_vol = float(curr.iloc[0]["_avg_vol"])
                if np.isnan(avg_vol) or avg_vol <= 0:
                    continue
                if volume < avg_vol * _VOL_MULTIPLIER:
                    continue

                # FIX #1: Per-bar ATR(14) shifted by 1 (matching engine.py)
                if "_atr" not in curr.columns:
                    continue
                atr = float(curr.iloc[0]["_atr"])
                if np.isnan(atr) or atr <= 0:
                    continue

                # Feed bar into Kalman filter
                km = kalman_state.get(sym)
                if km is not None and km.ready:
                    km.update(close_val)
                kalman_mult = km.multiplier if km is not None and km.ready else 1.0

                # FIX #3: 15-min EMA shifted by 1 (no lookahead)
                ema_val = None
                if "_ema15m" in curr.columns:
                    ev = curr.iloc[0]["_ema15m"]
                    if not np.isnan(ev):
                        ema_val = float(ev)

                if ema_val is not None:
                    if close_val > orb_high and close_val <= ema_val:
                        continue  # LONG blocked: below EMA
                    if close_val < orb_low and close_val >= ema_val:
                        continue  # SHORT blocked: above EMA

                # Gap classification filter (matching backtest ±0.3%)
                gap_pct = gap_cache.get(sym, {}).get(day)
                if gap_pct is not None:
                    if close_val > orb_high and gap_pct < -_GAP_THRESHOLD_PCT:
                        continue  # gap down blocks LONG
                    if close_val < orb_low and gap_pct > _GAP_THRESHOLD_PCT:
                        continue  # gap up blocks SHORT

                # 2-bar confirmation — Kalman widens stop, target uses base ATR
                base_atr_risk = _ATR_MULTIPLIER * atr
                adaptive_stop = base_atr_risk * kalman_mult
                if close_val > orb_high and prev_close > orb_high:
                    direction = "LONG"
                    entry = close_val
                    stop_price = entry - adaptive_stop
                    target_price = entry + _RISK_MULTIPLIER * base_atr_risk
                elif close_val < orb_low and prev_close < orb_low:
                    direction = "SHORT"
                    entry = close_val
                    stop_price = entry + adaptive_stop
                    target_price = entry - _RISK_MULTIPLIER * base_atr_risk
                else:
                    continue

                stats_counters["signals_raw"] += 1
                score = (d_adx or 25) / 100 * 0.4 + orb_range_pct * 100 * 0.3 + 0.3
                candidates.append((sym, direction, entry, stop_price, target_price, score))

            candidates.sort(key=lambda c: c[5], reverse=True)

            for sym, direction, entry, stop_price, target_price, _score in candidates:
                if len(open_positions) >= max_concurrent:
                    stats_counters["rejected_concurrent"] += 1
                    continue
                if daily_trade_count >= max_daily:
                    stats_counters["rejected_daily"] += 1
                    break
                sym_sector = SECTOR_MAP.get(sym, "other")
                same_sector = sum(
                    1 for p in open_positions if SECTOR_MAP.get(p.symbol, "other") == sym_sector
                )
                if same_sector >= max_sector:
                    stats_counters["rejected_sector"] += 1
                    continue

                # Risk-based position sizing (matching live: position_sizing.py)
                # shares = (equity * 1% / risk_per_share) * 0.25, whole shares
                risk_per_share = abs(entry - stop_price)
                if risk_per_share <= 0:
                    continue
                dollar_risk = equity * _RISK_PCT / 100.0
                shares = int(dollar_risk / risk_per_share * _POSITION_SCALE)
                if shares <= 0:
                    continue
                # Safety cap: position value <= 25% of equity
                pos_value = shares * entry
                max_pos_value = equity * _POSITION_SCALE
                if pos_value > max_pos_value:
                    shares = int(max_pos_value / entry)
                    if shares <= 0:
                        continue

                # Apply slippage to entry fill
                if direction == "LONG":
                    fill_entry = entry + _SLIPPAGE_PER_SHARE
                else:
                    fill_entry = entry - _SLIPPAGE_PER_SHARE

                open_positions.append(
                    Position(
                        sym,
                        direction,
                        fill_entry,
                        stop_price,
                        target_price,
                        float(shares),
                        bar_time,
                    )
                )
                daily_trade_count += 1
                stats_counters["signals_accepted"] += 1

        # ── Post-ORB: batch check SL/TP on bars 10:00-15:54 ─────────────
        orb_end_time = datetime.combine(day, time(9, 59)).replace(tzinfo=_ET)
        for pos in list(open_positions):
            pos_day_bars = day_bars.get(pos.symbol, {}).get(day)
            if pos_day_bars is None:
                continue
            # Check from after ORB window (or entry, whichever is later)
            check_from = max(pos.entry_time, orb_end_time)
            after = pos_day_bars[pos_day_bars.index > check_from]
            market_bars = after.between_time("09:30", "15:54")
            for _, bar_row in market_bars.iterrows():
                bar_o = float(bar_row["open"])
                bar_h = float(bar_row["high"])
                bar_l = float(bar_row["low"])
                bar_c = float(bar_row["close"])
                btime = bar_row.name

                hit_target = False
                hit_stop = False
                if pos.direction == "LONG":
                    hit_target = bar_h >= pos.target
                    hit_stop = bar_l <= pos.stop
                else:
                    hit_target = bar_l <= pos.target
                    hit_stop = bar_h >= pos.stop

                if hit_target and hit_stop:
                    favor = bar_c > bar_o if pos.direction == "LONG" else bar_c < bar_o
                    if favor:
                        hit_stop = False
                    else:
                        hit_target = False

                if hit_target:
                    pnl = (
                        (pos.target - pos.entry)
                        if pos.direction == "LONG"
                        else (pos.entry - pos.target)
                    ) * pos.size - _SLIPPAGE_PER_SHARE * pos.size
                    closed_trades.append(
                        Trade(
                            pos.symbol,
                            pos.direction,
                            pos.entry,
                            pos.target,
                            pnl,
                            pos.entry_time,
                            btime,
                            "target",
                        )
                    )
                    open_positions.remove(pos)
                    equity += pnl
                    stats_counters["exits_target"] += 1
                    break
                elif hit_stop:
                    pnl = (
                        (pos.stop - pos.entry)
                        if pos.direction == "LONG"
                        else (pos.entry - pos.stop)
                    ) * pos.size - _SLIPPAGE_PER_SHARE * pos.size
                    closed_trades.append(
                        Trade(
                            pos.symbol,
                            pos.direction,
                            pos.entry,
                            pos.stop,
                            pnl,
                            pos.entry_time,
                            btime,
                            "stop",
                        )
                    )
                    open_positions.remove(pos)
                    equity += pnl
                    stats_counters["exits_stop"] += 1
                    break

        # ── Force flat at 15:55 ET ───────────────────────────────────────
        for pos in list(open_positions):
            pos_day = day_bars.get(pos.symbol, {}).get(day)
            if pos_day is None:
                continue
            flat_bars = pos_day.between_time("15:55", "15:55")
            if flat_bars.empty:
                late_bars = pos_day.between_time("15:50", "15:59")
                if late_bars.empty:
                    continue
                exit_price = float(late_bars.iloc[-1]["close"])
            else:
                exit_price = float(flat_bars.iloc[0]["close"])

            if pos.direction == "LONG":
                pnl = (exit_price - pos.entry) * pos.size - _SLIPPAGE_PER_SHARE * pos.size
            else:
                pnl = (pos.entry - exit_price) * pos.size - _SLIPPAGE_PER_SHARE * pos.size
            closed_trades.append(
                Trade(
                    pos.symbol,
                    pos.direction,
                    pos.entry,
                    exit_price,
                    pnl,
                    pos.entry_time,
                    datetime.combine(day, time(15, 55)).replace(tzinfo=_ET),
                    "flat",
                )
            )
            equity += pnl
            open_positions.remove(pos)
            stats_counters["exits_flat"] += 1

        equity_curve.append(equity)

    # ── Results ──────────────────────────────────────────────────────────────
    total = len(closed_trades)
    if total == 0:
        print("\nNo trades generated.")
        print(f"  Stats: {dict(stats_counters)}")
        sys.exit(0)

    pnl_s = pd.Series([t.pnl for t in closed_trades])
    winners = pnl_s[pnl_s > 0]
    losers = pnl_s[pnl_s < 0]
    gp = float(winners.sum()) if len(winners) else 0
    gl = abs(float(losers.sum())) if len(losers) else 0.001
    pf = gp / gl
    wr = len(winners) / total * 100
    net = gp - abs(float(losers.sum())) if len(losers) else gp
    eq = pd.Series(equity_curve)
    dd = float(((eq - eq.cummax()) / eq.cummax()).min()) * 100 if len(eq) > 1 else 0
    sharpe = float(pnl_s.mean() / pnl_s.std() * np.sqrt(252)) if pnl_s.std() > 0 else 0

    # Trades per day distribution
    days_with_trades: dict[date, int] = defaultdict(int)
    for t in closed_trades:
        d = t.entry_time.date() if hasattr(t.entry_time, "date") else t.entry_time
        if isinstance(d, datetime):
            d = d.date()
        days_with_trades[d] += 1
    trade_counts = (
        pd.Series(list(days_with_trades.values())) if days_with_trades else pd.Series([0])
    )

    print(f"\n{'=' * 65}")
    print("  Portfolio Backtest Results")
    print(
        f"  ${args.cash:,.0f} | {max_concurrent} concurrent | {max_daily}/day | {max_sector}/sector"
    )
    print(f"{'=' * 65}")
    print(f"  Trades:       {total:>6}")
    print(f"  Win Rate:     {wr:>5.1f}%")
    print(f"  Profit Factor:{pf:>6.3f}")
    print(f"  Sharpe:       {sharpe:>6.2f}")
    print(f"  Net P&L:      ${net:>10,.2f}")
    print(f"  Max Drawdown: {dd:>5.1f}%")
    print(f"  Final Equity: ${equity:>10,.2f}")
    print(
        f"  Exits: target={stats_counters['exits_target']}, "
        f"stop={stats_counters['exits_stop']}, flat={stats_counters['exits_flat']}"
    )
    print(
        f"  Signals: raw={stats_counters['signals_raw']}, accepted={stats_counters['signals_accepted']}"
    )
    print(
        f"  Rejected: conc={stats_counters['rejected_concurrent']}, "
        f"daily={stats_counters['rejected_daily']}, sector={stats_counters['rejected_sector']}"
    )
    print(
        f"  Days: tradeable={stats_counters['days_tradeable']}, "
        f"adx_blocked={stats_counters['days_adx_blocked']}, "
        f"vol_blocked={stats_counters['days_vol_blocked']}, "
        f"monday={stats_counters['days_monday']}"
    )
    print(
        f"  Trades/day: mean={trade_counts.mean():.1f}, max={trade_counts.max()}, "
        f"dist={dict(trade_counts.value_counts().sort_index())}"
    )
    print(f"{'=' * 65}")

    # Save
    trades_df = pd.DataFrame(
        [
            {
                "symbol": t.symbol,
                "direction": t.direction,
                "entry": t.entry,
                "exit": t.exit_price,
                "pnl": round(t.pnl, 2),
                "exit_reason": t.exit_reason,
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
            }
            for t in closed_trades
        ]
    )
    trades_df.to_csv(out_dir / "portfolio_trades.csv", index=False)
    print(f"\nTrade log: {out_dir / 'portfolio_trades.csv'}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(eq.values, linewidth=0.8)
        ax.set_title(f"Portfolio Equity (PF={pf:.2f}, {total} trades, ${net:+,.0f})")
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
