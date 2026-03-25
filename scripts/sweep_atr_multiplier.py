#!/usr/bin/env python
"""Sweep ATR stop multipliers: backtesting.py vs bar H/L execution.

For each ATR multiplier [1.5, 2, 3, 4, 5], runs:
1. Per-ticker backtest via backtesting.py (the framework)
2. Per-ticker backtest via bar H/L checking (our sim)

Shows where the two PFs converge — the "honest zone" where bar H/L PF > 1.0.

Usage:
    python scripts/sweep_atr_multiplier.py
"""

from __future__ import annotations

import sys
import warnings
from datetime import date
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import talib

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtest.data_loader import load_bars
from src.backtest.engine import run_backtest
from src.filters.economic_calendar import compute_econ_blocked_array
from src.filters.vix_term_structure import (
    BACKWARDATION_THRESHOLD,
    compute_vix_term_structure_array,
    get_vix_term_structure,
)
from src.models import TimeFrame
from src.storage.database import BarDatabase

warnings.filterwarnings("ignore", category=RuntimeWarning)

_ET = ZoneInfo("America/New_York")
_ORB_BARS = 5
_VOL_MULTIPLIER = 1.5
_VOL_WINDOW = 20
_DAILY_ADX_MIN = 25.0
_REALIZED_VOL_MAX = 0.18
_MIN_ORB_PCT = 0.0015
_GAP_THRESHOLD_PCT = 0.3
_MAX_TRADES_PER_DAY = 5
_POSITION_SCALE = 0.25
_RISK_MULT = 2.0  # fixed R:R


def run_barhl_ticker(
    df_1min: pd.DataFrame,
    symbol: str,
    cash: float,
    atr_mult: float,
    econ_blocked: np.ndarray | None = None,
    vts_arr: np.ndarray | None = None,
) -> dict[str, float]:
    """Run one ticker with bar H/L execution at given ATR multiplier."""
    df_et = df_1min.copy()
    df_et.index = df_et.index.tz_convert(_ET)

    h = df_1min["high"].astype(float).values
    lo = df_1min["low"].astype(float).values
    c = df_1min["close"].astype(float).values
    v = df_1min["volume"].astype(float).values

    atr_raw = talib.ATR(h, lo, c, timeperiod=14)
    atr_shifted = np.roll(atr_raw, 1)
    atr_shifted[0] = np.nan

    avg_vol_raw = pd.Series(v).rolling(_VOL_WINDOW).mean().values
    avg_vol_shifted = np.roll(avg_vol_raw, 1)
    avg_vol_shifted[0] = np.nan

    # Daily ADX + realized vol
    daily = (
        df_et.resample("1D")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        .dropna()
    )
    daily_adx_arr = np.full(len(daily), np.nan)
    daily_rv_arr = np.full(len(daily), np.nan)
    if len(daily) >= 15:
        daily_adx_arr = talib.ADX(
            daily["high"].values, daily["low"].values, daily["close"].values, timeperiod=14
        )
        daily_rv_arr = (daily["close"].pct_change().rolling(20).std() * np.sqrt(252)).values

    adx_lk: dict[date, float] = {}
    rv_lk: dict[date, float] = {}
    dl = list(daily.index)
    for i in range(1, len(dl)):
        d = dl[i].date()
        if not np.isnan(daily_adx_arr[i - 1]):
            adx_lk[d] = float(daily_adx_arr[i - 1])
        if not np.isnan(daily_rv_arr[i - 1]):
            rv_lk[d] = float(daily_rv_arr[i - 1])

    # 15-min EMA
    bars_15m = df_et.resample("15min").agg({"close": "last"}).dropna()
    ema_lk: dict[date, float] = {}
    if len(bars_15m) >= 21:
        from talipp.indicators import EMA as _EMA

        ema = _EMA(period=20)
        for ts, row in bars_15m.iterrows():
            ema.add(float(row["close"]))
            if ema.output_values and ema.output_values[-1] is not None:
                ema_lk[ts.date()] = float(ema.output_values[-1])

    # Gap
    gap_lk: dict[date, float] = {}
    prev_c = None
    for d, grp in df_et.groupby(df_et.index.date):
        fb = grp.between_time("09:30", "09:30")
        if not fb.empty and prev_c is not None:
            gap_lk[d] = (float(fb.iloc[0]["open"]) - prev_c) / prev_c * 100
        lb = grp.between_time("15:55", "15:59")
        prev_c = (
            float(lb.iloc[-1]["close"])
            if not lb.empty
            else (float(grp.iloc[-1]["close"]) if len(grp) else prev_c)
        )

    econ_dates: set[date] = set()
    if econ_blocked is not None:
        for i, ts in enumerate(df_et.index):
            if econ_blocked[i] > 0.5:
                econ_dates.add(ts.date())
    vts_lk: dict[date, float] = {}
    if vts_arr is not None:
        for i, ts in enumerate(df_et.index):
            if not np.isnan(vts_arr[i]):
                vts_lk[ts.date()] = float(vts_arr[i])

    orb_cache: dict[date, tuple[float, float]] = {}
    day_bars: dict[date, pd.DataFrame] = {}
    for d, grp in df_et.groupby(df_et.index.date):
        day_bars[d] = grp
        ob = grp.between_time("09:30", "09:34")
        if len(ob) >= _ORB_BARS:
            orb_cache[d] = (float(ob["high"].max()), float(ob["low"].min()))

    equity = cash
    trades = 0
    winners = 0
    gp = 0.0
    gl = 0.0
    tgt_exits = 0
    stp_exits = 0

    for day in sorted(day_bars.keys()):
        if day.weekday() == 0:
            continue
        da = adx_lk.get(day)
        dr = rv_lk.get(day)
        if da is not None and da <= _DAILY_ADX_MIN:
            continue
        if dr is not None and dr >= _REALIZED_VOL_MAX:
            continue
        if day in econ_dates:
            continue
        vts = vts_lk.get(day)
        if vts is not None and vts > BACKWARDATION_THRESHOLD:
            continue

        orb = orb_cache.get(day)
        if orb is None:
            continue
        oh, ol = orb
        if ol <= 0 or (oh - ol) / ol < _MIN_ORB_PCT:
            continue

        grp = day_bars[day]
        sig_bars = grp.between_time("09:35", "09:59")
        all_bars = grp.between_time("09:30", "15:54")
        dt = 0

        for bi in range(1, len(sig_bars)):
            if dt >= _MAX_TRADES_PER_DAY:
                break
            curr = sig_bars.iloc[bi]
            prev = sig_bars.iloc[bi - 1]
            ct = curr.name
            cv = float(curr["close"])
            pv = float(prev["close"])
            vol = float(curr["volume"])

            oi = df_et.index.get_loc(ct)
            if isinstance(oi, slice):
                oi = oi.start
            atr_v = atr_shifted[oi] if oi < len(atr_shifted) else np.nan
            av = avg_vol_shifted[oi] if oi < len(avg_vol_shifted) else np.nan
            if np.isnan(atr_v) or atr_v <= 0 or np.isnan(av) or av <= 0:
                continue
            if vol < av * _VOL_MULTIPLIER:
                continue

            ev = ema_lk.get(day)
            if ev is not None:
                if cv > oh and cv <= ev:
                    continue
                if cv < ol and cv >= ev:
                    continue
            gp_v = gap_lk.get(day)
            if gp_v is not None:
                if cv > oh and gp_v < -_GAP_THRESHOLD_PCT:
                    continue
                if cv < ol and gp_v > _GAP_THRESHOLD_PCT:
                    continue

            base = atr_mult * atr_v
            if cv > oh and pv > oh:
                d = "L"
                entry = cv
                stop = entry - base
                target = entry + _RISK_MULT * base
            elif cv < ol and pv < ol:
                d = "S"
                entry = cv
                stop = entry + base
                target = entry - _RISK_MULT * base
            else:
                continue

            # Bar H/L exit
            after = all_bars[all_bars.index > ct]
            ep = None
            er = "flat"
            for _, cb in after.iterrows():
                bh, bl = float(cb["high"]), float(cb["low"])
                bo, bc = float(cb["open"]), float(cb["close"])
                ht = (bh >= target) if d == "L" else (bl <= target)
                hs = (bl <= stop) if d == "L" else (bh >= stop)
                if ht and hs:
                    favor = (bc > bo) if d == "L" else (bc < bo)
                    if favor:
                        hs = False
                    else:
                        ht = False
                if ht:
                    ep = target
                    er = "target"
                    break
                elif hs:
                    ep = stop
                    er = "stop"
                    break

            if ep is None:
                fb = grp.between_time("15:55", "15:59")
                ep = float(fb.iloc[0]["close"]) if not fb.empty else float(grp.iloc[-1]["close"])
                er = "flat"

            ps = equity * _POSITION_SCALE
            pnl = ((ep - entry) if d == "L" else (entry - ep)) * (ps / entry)
            trades += 1
            if pnl > 0:
                winners += 1
                gp += pnl
            else:
                gl += abs(pnl)
            if er == "target":
                tgt_exits += 1
            elif er == "stop":
                stp_exits += 1
            equity += pnl
            dt += 1

    return {
        "trades": trades,
        "winners": winners,
        "gp": gp,
        "gl": gl,
        "tgt": tgt_exits,
        "stp": stp_exits,
    }


def main() -> None:
    db = BarDatabase()
    db.connect()

    symbols = [
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
    cash = 200_000.0
    atr_mults = [1.5, 2.0, 3.0, 4.0, 5.0]

    # Load VTS once
    vts_df = get_vix_term_structure()

    # Load all ticker data once
    print("Loading data...")
    ticker_data: dict[str, tuple[pd.DataFrame, np.ndarray, np.ndarray | None]] = {}
    for sym in symbols:
        df = load_bars(db, symbol=sym, timeframe=TimeFrame.ONE_MIN)
        if df.empty:
            continue
        econ = compute_econ_blocked_array(df.index)
        vts_arr = compute_vix_term_structure_array(df.index, vts_df) if not vts_df.empty else None
        ticker_data[sym] = (df, econ, vts_arr)
        print(f"  {sym}: {len(df):,} bars")
    db.close()

    print(f"\nSweeping ATR multipliers: {atr_mults}")
    print(f"Tickers: {len(ticker_data)}, Cash: ${cash:,.0f}")
    print()

    # Run backtesting.py framework for each multiplier
    print("Running backtesting.py framework sweep...")
    framework_results: dict[float, dict[str, float]] = {}
    for mult in atr_mults:
        total_trades = 0
        total_gp = 0.0
        total_gl = 0.0
        for sym, (df, _econ, _vts) in ticker_data.items():
            try:
                stats = run_backtest(
                    df, cash=cash, slippage=0.02, atr_mult=mult, risk_mult=_RISK_MULT, symbol=sym
                )
                t = int(stats["# Trades"])
                if t > 0:
                    total_trades += t
                    tr = stats["_trades"]
                    w = tr[tr["PnL"] > 0]["PnL"].sum()
                    lo_sum = abs(tr[tr["PnL"] <= 0]["PnL"].sum())
                    total_gp += w
                    total_gl += lo_sum
            except Exception as exc:
                print(f"    {sym} failed: {exc}")
        pf = total_gp / total_gl if total_gl > 0 else 0
        wr = 0  # Can't easily get from aggregate
        framework_results[mult] = {"trades": total_trades, "pf": pf, "net": total_gp - total_gl}
        print(
            f"  ATR {mult}x: {total_trades} trades, PF={pf:.3f}, Net=${total_gp - total_gl:+,.0f}"
        )

    # Run bar H/L for each multiplier
    print("\nRunning bar H/L execution sweep...")
    barhl_results: dict[float, dict[str, float]] = {}
    for mult in atr_mults:
        agg = {"trades": 0, "winners": 0, "gp": 0.0, "gl": 0.0, "tgt": 0, "stp": 0}
        for sym, (df, econ, vts_arr) in ticker_data.items():
            r = run_barhl_ticker(df, sym, cash, atr_mult=mult, econ_blocked=econ, vts_arr=vts_arr)
            for k in agg:
                agg[k] += r[k]
        pf = agg["gp"] / agg["gl"] if agg["gl"] > 0 else 0
        wr = agg["winners"] / agg["trades"] * 100 if agg["trades"] > 0 else 0
        barhl_results[mult] = {
            "trades": agg["trades"],
            "pf": pf,
            "wr": wr,
            "net": agg["gp"] - agg["gl"],
            "tgt_pct": agg["tgt"] / agg["trades"] * 100 if agg["trades"] > 0 else 0,
            "stp_pct": agg["stp"] / agg["trades"] * 100 if agg["trades"] > 0 else 0,
        }
        print(
            f"  ATR {mult}x: {agg['trades']} trades, PF={pf:.3f}, WR={wr:.1f}%, "
            f"Net=${agg['gp'] - agg['gl']:+,.0f}, "
            f"Tgt={agg['tgt']}({agg['tgt']/agg['trades']*100:.0f}%) "
            f"Stp={agg['stp']}({agg['stp']/agg['trades']*100:.0f}%)"
            if agg["trades"] > 0
            else f"  ATR {mult}x: 0 trades"
        )

    # Summary table
    print(f"\n{'=' * 90}")
    print("  ATR MULTIPLIER SWEEP — backtesting.py vs Bar H/L")
    print(
        f"  {'ATR Mult':>8} | {'Framework PF':>12} {'Frmwk Trades':>13} {'Frmwk Net':>12} | "
        f"{'BarHL PF':>8} {'BarHL Trades':>12} {'BarHL WR':>8} {'BarHL Net':>11} {'Tgt%':>5} {'Stp%':>5}"
    )
    print(f"  {'-' * 86}")
    for mult in atr_mults:
        fw = framework_results[mult]
        bh = barhl_results[mult]
        marker = " <<<" if bh["pf"] > 1.0 and fw["pf"] < 2.0 else ""
        print(
            f"  {mult:>7.1f}x | {fw['pf']:>11.3f} {int(fw['trades']):>13,} ${fw['net']:>+11,.0f} | "
            f"{bh['pf']:>7.3f} {int(bh['trades']):>12,} {bh.get('wr', 0):>7.1f}% ${bh['net']:>+10,.0f} "
            f"{bh.get('tgt_pct', 0):>4.0f}% {bh.get('stp_pct', 0):>4.0f}%{marker}"
        )
    print(f"{'=' * 90}")
    print("  <<< = 'honest zone' where bar H/L PF > 1.0 AND framework PF < 2.0")


if __name__ == "__main__":
    main()
