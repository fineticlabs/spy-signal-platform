#!/usr/bin/env python
"""Phase 4: Compare three portfolio configurations with annual breakdowns.

1. CURRENT: baseline (all filters, 3 concurrent, 5/day)
2. OPTIMIZED-A: ADX OFF, concurrent=5, daily=8
3. OPTIMIZED-B: ADX OFF, concurrent=5, daily=8, Monday OFF

Outputs to docs/portfolio_investigation/phase4_optimization/
"""

from __future__ import annotations

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
_WINDOW_START = time(9, 35)
_WINDOW_END = time(10, 0)
_FORCE_FLAT = time(15, 55)

SYMBOLS = [
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


def load_all_data():
    """Load bars and pre-compute all caches."""
    db = BarDatabase()
    db.connect()
    raw_bars: dict[str, pd.DataFrame] = {}
    for sym in SYMBOLS:
        df = load_bars(db, symbol=sym, timeframe=TimeFrame.ONE_MIN)
        if not df.empty:
            raw_bars[sym] = df
    db.close()

    day_bars: dict[str, dict[date, pd.DataFrame]] = defaultdict(dict)
    all_trading_dates: set[date] = set()
    for sym, df in raw_bars.items():
        df_et = df.copy()
        df_et.index = df_et.index.tz_convert(_ET)
        for d, grp in df_et.groupby(df_et.index.date):
            day_bars[sym][d] = grp
            all_trading_dates.add(d)
    trading_days = sorted(all_trading_dates)

    spy_df = raw_bars["SPY"]
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
    dl = list(spy_daily.index)
    for i in range(1, len(dl)):
        d = dl[i].date()
        if not np.isnan(daily_adx[i - 1]):
            adx_lookup[d] = float(daily_adx[i - 1])
        if not np.isnan(realized_vol.iloc[i - 1]):
            vol_lookup[d] = float(realized_vol.iloc[i - 1])

    # Per-bar ATR + avg vol shifted by 1
    for sym, df in raw_bars.items():
        h = df["high"].astype(float).values
        lo = df["low"].astype(float).values
        c = df["close"].astype(float).values
        if len(c) < _ATR_PERIOD + 1:
            continue
        atr_raw = talib.ATR(h, lo, c, timeperiod=_ATR_PERIOD)
        atr_shifted = np.roll(atr_raw, 1)
        atr_shifted[0] = np.nan
        vol_s = pd.Series(df["volume"].astype(float).values)
        avg_vol_raw = vol_s.rolling(_VOL_WINDOW).mean().to_numpy()
        avg_vol_shifted = np.roll(avg_vol_raw, 1)
        avg_vol_shifted[0] = np.nan
        df_et = df.copy()
        df_et.index = df_et.index.tz_convert(_ET)
        df_et["_atr"] = atr_shifted
        df_et["_avg_vol"] = avg_vol_shifted
        for d, grp in df_et.groupby(df_et.index.date):
            day_bars[sym][d] = grp

    orb_cache: dict[str, dict[date, tuple[float, float]]] = defaultdict(dict)
    for sym in raw_bars:
        for d, grp in day_bars.get(sym, {}).items():
            ob = grp.between_time("09:30", "09:34")
            if len(ob) >= _ORB_BARS:
                orb_cache[sym][d] = (float(ob["high"].max()), float(ob["low"].min()))

    # 15-min EMA shifted by 1 bar
    for sym in raw_bars:
        all_day_dfs = [day_bars[sym][d] for d in sorted(day_bars.get(sym, {}).keys())]
        if not all_day_dfs:
            continue
        df_et = pd.concat(all_day_dfs)
        cs = df_et["close"].astype(float)
        c15 = cs.resample("15min").last().dropna()
        if len(c15) < _EMA15M_PERIOD:
            continue
        ema_raw = talib.EMA(c15.to_numpy(dtype=float), timeperiod=_EMA15M_PERIOD)
        ema_15m = pd.Series(ema_raw, index=c15.index)
        ema_1m = ema_15m.shift(1).reindex(df_et.index, method="ffill")
        for d, grp in df_et.groupby(df_et.index.date):
            grp = grp.copy()
            grp["_ema15m"] = ema_1m.reindex(grp.index)
            day_bars[sym][d] = grp

    gap_cache: dict[str, dict[date, float]] = defaultdict(dict)
    for sym in raw_bars:
        prev_c = None
        for d in sorted(day_bars.get(sym, {}).keys()):
            grp = day_bars[sym][d]
            fb = grp.between_time("09:30", "09:30")
            if not fb.empty and prev_c is not None:
                gap_cache[sym][d] = (float(fb.iloc[0]["open"]) - prev_c) / prev_c * 100
            lb = grp.between_time("15:55", "15:59")
            if not lb.empty:
                prev_c = float(lb.iloc[-1]["close"])
            elif len(grp) > 0:
                prev_c = float(grp.iloc[-1]["close"])

    vts_df = get_vix_term_structure()
    vts_lookup: dict[date, float] = {}
    if not vts_df.empty:
        vts_arr = compute_vix_term_structure_array(spy_df.index, vts_df)
        tmp = spy_df.copy()
        tmp.index = tmp.index.tz_convert(_ET)
        tmp["_vts"] = vts_arr
        for d, grp in tmp.groupby(tmp.index.date):
            vals = grp["_vts"].dropna()
            if len(vals) > 0:
                vts_lookup[d] = float(vals.iloc[0])

    econ_arr = compute_econ_blocked_array(spy_df.index)
    econ_blocked: set[date] = set()
    tmp2 = spy_df.copy()
    tmp2.index = tmp2.index.tz_convert(_ET)
    tmp2["_e"] = econ_arr
    for d, grp in tmp2.groupby(tmp2.index.date):
        if grp["_e"].max() > 0.5:
            econ_blocked.add(d)

    return {
        "raw_bars": raw_bars,
        "day_bars": day_bars,
        "trading_days": trading_days,
        "adx_lookup": adx_lookup,
        "vol_lookup": vol_lookup,
        "orb_cache": orb_cache,
        "gap_cache": gap_cache,
        "vts_lookup": vts_lookup,
        "econ_blocked": econ_blocked,
    }


def run_sim(
    data: dict,
    cash: float = 200_000,
    max_concurrent: int = 3,
    max_daily: int = 5,
    max_sector: int = 2,
    adx_filter: bool = True,
    vol_filter: bool = True,
    monday_filter: bool = True,
    econ_filter: bool = True,
    vts_filter: bool = True,
    sector_filter: bool = True,
) -> tuple[list[Trade], list[float], dict[str, int]]:
    raw_bars = data["raw_bars"]
    day_bars = data["day_bars"]
    trading_days = data["trading_days"]
    adx_lookup = data["adx_lookup"]
    vol_lookup = data["vol_lookup"]
    orb_cache = data["orb_cache"]
    gap_cache = data["gap_cache"]
    vts_lookup = data["vts_lookup"]
    econ_blocked = data["econ_blocked"]

    equity = cash
    open_positions: list[Position] = []
    closed_trades: list[Trade] = []
    equity_curve: list[float] = []
    stats: dict[str, int] = defaultdict(int)

    for day in trading_days:
        if monday_filter and day.weekday() == 0:
            stats["days_monday"] += 1
            continue
        d_adx = adx_lookup.get(day)
        d_vol = vol_lookup.get(day)
        if adx_filter and d_adx is not None and d_adx <= _DAILY_ADX_MIN:
            stats["days_adx_blocked"] += 1
            continue
        if vol_filter and d_vol is not None and d_vol >= _REALIZED_VOL_MAX:
            stats["days_vol_blocked"] += 1
            continue
        if econ_filter and day in econ_blocked:
            stats["days_econ_blocked"] += 1
            continue
        vts_ratio = vts_lookup.get(day)
        if vts_filter and vts_ratio is not None and vts_ratio > 1.0:
            stats["days_backwardation"] += 1
            continue

        stats["days_tradeable"] += 1
        daily_trade_count = 0

        kalman_state: dict[str, StreamingKalman] = {}
        for sym in raw_bars:
            orb = orb_cache.get(sym, {}).get(day)
            if orb is None:
                continue
            sym_day = day_bars.get(sym, {}).get(day)
            if sym_day is None or "_atr" not in sym_day.columns:
                continue
            oe = sym_day.between_time("09:34", "09:34")
            if oe.empty:
                continue
            sa = float(oe.iloc[0]["_atr"])
            if np.isnan(sa) or sa <= 0:
                continue
            ob = sym_day.between_time("09:30", "09:34")
            if len(ob) >= _ORB_BARS:
                km = StreamingKalman()
                km.reset([float(r["close"]) for _, r in ob.iterrows()], sa)
                kalman_state[sym] = km

        # ORB window bar-by-bar
        for minute_offset in range(5, 30):
            bar_time = datetime.combine(day, time(9, 30 + minute_offset)).replace(tzinfo=_ET)

            # Exits
            for pos in list(open_positions):
                pdb = day_bars.get(pos.symbol, {}).get(day)
                if pdb is None:
                    continue
                after = pdb[(pdb.index > pos.entry_time) & (pdb.index <= bar_time)]
                for _, br in after.iterrows():
                    bo, bh, bl, bc = (
                        float(br["open"]),
                        float(br["high"]),
                        float(br["low"]),
                        float(br["close"]),
                    )
                    ht = (bh >= pos.target) if pos.direction == "LONG" else (bl <= pos.target)
                    hs = (bl <= pos.stop) if pos.direction == "LONG" else (bh >= pos.stop)
                    if ht and hs:
                        fav = bc > bo if pos.direction == "LONG" else bc < bo
                        hs, ht = (False, True) if fav else (True, False)
                    if ht:
                        pnl = (
                            (pos.target - pos.entry)
                            if pos.direction == "LONG"
                            else (pos.entry - pos.target)
                        ) * (pos.size / pos.entry)
                        closed_trades.append(
                            Trade(
                                pos.symbol,
                                pos.direction,
                                pos.entry,
                                pos.target,
                                pnl,
                                pos.entry_time,
                                br.name,
                                "target",
                            )
                        )
                        open_positions.remove(pos)
                        equity += pnl
                        stats["exits_target"] += 1
                        break
                    elif hs:
                        pnl = (
                            (pos.stop - pos.entry)
                            if pos.direction == "LONG"
                            else (pos.entry - pos.stop)
                        ) * (pos.size / pos.entry)
                        closed_trades.append(
                            Trade(
                                pos.symbol,
                                pos.direction,
                                pos.entry,
                                pos.stop,
                                pnl,
                                pos.entry_time,
                                br.name,
                                "stop",
                            )
                        )
                        open_positions.remove(pos)
                        equity += pnl
                        stats["exits_stop"] += 1
                        break

            # Entries
            if daily_trade_count >= max_daily or len(open_positions) >= max_concurrent:
                continue
            cands: list[tuple[str, str, float, float, float, float]] = []
            for sym in raw_bars:
                orb = orb_cache.get(sym, {}).get(day)
                if orb is None:
                    continue
                oh, ol = orb
                if ol <= 0:
                    continue
                orp = (oh - ol) / ol
                if orp < _MIN_ORB_PCT:
                    continue
                if any(p.symbol == sym for p in open_positions):
                    continue
                sd = day_bars.get(sym, {}).get(day)
                if sd is None:
                    continue
                tt = time(9, 30 + minute_offset)
                pt = time(9, 29 + minute_offset)
                cur = sd.between_time(tt, tt)
                prv = sd.between_time(pt, pt)
                if cur.empty or prv.empty:
                    continue
                cv = float(cur.iloc[0]["close"])
                pv = float(prv.iloc[0]["close"])
                vol = float(cur.iloc[0]["volume"])
                if "_avg_vol" not in cur.columns:
                    continue
                av = float(cur.iloc[0]["_avg_vol"])
                if np.isnan(av) or av <= 0 or vol < av * _VOL_MULTIPLIER:
                    continue
                if "_atr" not in cur.columns:
                    continue
                atr = float(cur.iloc[0]["_atr"])
                if np.isnan(atr) or atr <= 0:
                    continue
                km = kalman_state.get(sym)
                if km is not None and km.ready:
                    km.update(cv)
                kml = km.multiplier if km is not None and km.ready else 1.0
                if "_ema15m" in cur.columns:
                    ev = cur.iloc[0]["_ema15m"]
                    if not np.isnan(ev):
                        if cv > oh and cv <= ev:
                            continue
                        if cv < ol and cv >= ev:
                            continue
                gp = gap_cache.get(sym, {}).get(day)
                if gp is not None:
                    if cv > oh and gp < -_GAP_THRESHOLD_PCT:
                        continue
                    if cv < ol and gp > _GAP_THRESHOLD_PCT:
                        continue
                bar = _ATR_MULTIPLIER * atr
                ads = bar * kml
                if cv > oh and pv > oh:
                    di, en, sl, tp = "LONG", cv, cv - ads, cv + _RISK_MULTIPLIER * bar
                elif cv < ol and pv < ol:
                    di, en, sl, tp = "SHORT", cv, cv + ads, cv - _RISK_MULTIPLIER * bar
                else:
                    continue
                sc = (d_adx or 25) / 100 * 0.4 + orp * 100 * 0.3 + 0.3
                cands.append((sym, di, en, sl, tp, sc))

            cands.sort(key=lambda c: c[5], reverse=True)
            for sym, di, en, sl, tp, _ in cands:
                if len(open_positions) >= max_concurrent:
                    continue
                if daily_trade_count >= max_daily:
                    break
                if sector_filter:
                    sec = SECTOR_MAP.get(sym, "other")
                    if (
                        sum(1 for p in open_positions if SECTOR_MAP.get(p.symbol, "other") == sec)
                        >= max_sector
                    ):
                        continue
                open_positions.append(
                    Position(sym, di, en, sl, tp, equity * _POSITION_SCALE, bar_time)
                )
                daily_trade_count += 1
                stats["signals_accepted"] += 1

        # Post-ORB batch exits
        oet = datetime.combine(day, time(9, 59)).replace(tzinfo=_ET)
        for pos in list(open_positions):
            pdb = day_bars.get(pos.symbol, {}).get(day)
            if pdb is None:
                continue
            cf = max(pos.entry_time, oet)
            after = pdb[pdb.index > cf]
            mkt = after.between_time("09:30", "15:54")
            for _, br in mkt.iterrows():
                bo, bh, bl, bc = (
                    float(br["open"]),
                    float(br["high"]),
                    float(br["low"]),
                    float(br["close"]),
                )
                ht = (bh >= pos.target) if pos.direction == "LONG" else (bl <= pos.target)
                hs = (bl <= pos.stop) if pos.direction == "LONG" else (bh >= pos.stop)
                if ht and hs:
                    fav = bc > bo if pos.direction == "LONG" else bc < bo
                    hs, ht = (False, True) if fav else (True, False)
                if ht:
                    pnl = (
                        (pos.target - pos.entry)
                        if pos.direction == "LONG"
                        else (pos.entry - pos.target)
                    ) * (pos.size / pos.entry)
                    closed_trades.append(
                        Trade(
                            pos.symbol,
                            pos.direction,
                            pos.entry,
                            pos.target,
                            pnl,
                            pos.entry_time,
                            br.name,
                            "target",
                        )
                    )
                    open_positions.remove(pos)
                    equity += pnl
                    stats["exits_target"] += 1
                    break
                elif hs:
                    pnl = (
                        (pos.stop - pos.entry)
                        if pos.direction == "LONG"
                        else (pos.entry - pos.stop)
                    ) * (pos.size / pos.entry)
                    closed_trades.append(
                        Trade(
                            pos.symbol,
                            pos.direction,
                            pos.entry,
                            pos.stop,
                            pnl,
                            pos.entry_time,
                            br.name,
                            "stop",
                        )
                    )
                    open_positions.remove(pos)
                    equity += pnl
                    stats["exits_stop"] += 1
                    break

        # Force flat
        for pos in list(open_positions):
            pd_ = day_bars.get(pos.symbol, {}).get(day)
            if pd_ is None:
                continue
            fb = pd_.between_time("15:55", "15:55")
            if fb.empty:
                lb = pd_.between_time("15:50", "15:59")
                if lb.empty:
                    continue
                ep = float(lb.iloc[-1]["close"])
            else:
                ep = float(fb.iloc[0]["close"])
            pnl = ((ep - pos.entry) if pos.direction == "LONG" else (pos.entry - ep)) * (
                pos.size / pos.entry
            )
            closed_trades.append(
                Trade(
                    pos.symbol,
                    pos.direction,
                    pos.entry,
                    ep,
                    pnl,
                    pos.entry_time,
                    datetime.combine(day, time(15, 55)).replace(tzinfo=_ET),
                    "flat",
                )
            )
            equity += pnl
            open_positions.remove(pos)
            stats["exits_flat"] += 1

        equity_curve.append(equity)

    return closed_trades, equity_curve, dict(stats)


def compute_metrics(trades: list[Trade], eq_curve: list[float]) -> dict:
    if not trades:
        return {
            "trades": 0,
            "wr": 0,
            "pf": 0,
            "sharpe": 0,
            "net": 0,
            "max_dd": 0,
            "stop_rate": 0,
            "target_rate": 0,
            "flat_rate": 0,
        }
    pnl_s = pd.Series([t.pnl for t in trades])
    w = pnl_s[pnl_s > 0]
    lo = pnl_s[pnl_s < 0]
    gp = float(w.sum()) if len(w) else 0
    gl = abs(float(lo.sum())) if len(lo) else 0.001
    eq = pd.Series(eq_curve)
    dd = float(((eq - eq.cummax()) / eq.cummax()).min()) * 100 if len(eq) > 1 else 0
    total = len(trades)
    return {
        "trades": total,
        "wr": len(w) / total * 100,
        "pf": gp / gl,
        "sharpe": float(pnl_s.mean() / pnl_s.std() * np.sqrt(252)) if pnl_s.std() > 0 else 0,
        "net": gp - abs(float(lo.sum())) if len(lo) else gp,
        "max_dd": dd,
        "stop_rate": sum(1 for t in trades if t.exit_reason == "stop") / total * 100,
        "target_rate": sum(1 for t in trades if t.exit_reason == "target") / total * 100,
        "flat_rate": sum(1 for t in trades if t.exit_reason == "flat") / total * 100,
    }


def annual_breakdown(trades: list[Trade]) -> pd.DataFrame:
    """Compute per-year metrics."""
    if not trades:
        return pd.DataFrame()
    rows = []
    by_year: dict[int, list[Trade]] = defaultdict(list)
    for t in trades:
        yr = t.entry_time.year if hasattr(t.entry_time, "year") else t.entry_time
        by_year[yr].append(t)
    for yr in sorted(by_year.keys()):
        yts = by_year[yr]
        pnls = pd.Series([t.pnl for t in yts])
        w = pnls[pnls > 0]
        lo = pnls[pnls < 0]
        gp = float(w.sum()) if len(w) else 0
        gl = abs(float(lo.sum())) if len(lo) else 0.001
        rows.append(
            {
                "year": yr,
                "trades": len(yts),
                "wr": len(w) / len(yts) * 100 if yts else 0,
                "pf": gp / gl,
                "net": gp - abs(float(lo.sum())) if len(lo) else gp,
                "avg_pnl": float(pnls.mean()),
                "stops": sum(1 for t in yts if t.exit_reason == "stop"),
                "targets": sum(1 for t in yts if t.exit_reason == "target"),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    out_dir = Path("docs/portfolio_investigation/phase4_optimization")
    out_dir.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []

    def log(msg: str) -> None:
        print(msg)
        lines.append(msg)

    log("=" * 80)
    log("PHASE 4: PORTFOLIO CONFIGURATION COMPARISON")
    log("=" * 80)

    log("\nLoading data...")
    data = load_all_data()
    log(f"Loaded {len(data['raw_bars'])} symbols, {len(data['trading_days'])} trading days\n")

    configs = {
        "CURRENT": {
            "desc": "All filters ON, 3 concurrent, 5/day, 2/sector",
            "kwargs": {},
        },
        "OPTIMIZED-A": {
            "desc": "ADX OFF, 5 concurrent, 8/day, 2/sector",
            "kwargs": {"adx_filter": False, "max_concurrent": 5, "max_daily": 8},
        },
        "OPTIMIZED-B": {
            "desc": "ADX OFF, Monday OFF, 5 concurrent, 8/day, 2/sector",
            "kwargs": {
                "adx_filter": False,
                "monday_filter": False,
                "max_concurrent": 5,
                "max_daily": 8,
            },
        },
    }

    all_results: dict[str, dict] = {}
    all_trades: dict[str, list[Trade]] = {}
    all_eq: dict[str, list[float]] = {}

    for name, cfg in configs.items():
        log(f"Running {name}: {cfg['desc']}...")
        trades, eq, _st = run_sim(data, **cfg["kwargs"])
        m = compute_metrics(trades, eq)
        all_results[name] = m
        all_trades[name] = trades
        all_eq[name] = eq

        # Trades/day dist
        d_counts: dict[date, int] = defaultdict(int)
        for t in trades:
            d_ = t.entry_time.date() if hasattr(t.entry_time, "date") else t.entry_time
            if isinstance(d_, datetime):
                d_ = d_.date()
            d_counts[d_] += 1
        tc = pd.Series(list(d_counts.values())) if d_counts else pd.Series([0])
        m["trades_per_day_mean"] = float(tc.mean())
        m["trades_per_day_max"] = int(tc.max())
        m["trades_per_day_dist"] = dict(tc.value_counts().sort_index())

        log(
            f"  Trades={m['trades']}, WR={m['wr']:.1f}%, PF={m['pf']:.3f}, "
            f"Sharpe={m['sharpe']:.2f}, Net=${m['net']:,.0f}\n"
        )

    # ── Summary comparison ───────────────────────────────────────────────────
    log("=" * 80)
    log("SUMMARY COMPARISON")
    log("=" * 80)

    header_labels = list(configs.keys())
    log(f"\n  {'Metric':<25}" + "".join(f" {h:>18}" for h in header_labels))
    log("  " + "-" * (25 + 18 * len(header_labels)))

    metric_rows = [
        ("Trades", "trades", "d"),
        ("Win Rate %", "wr", ".1f"),
        ("Profit Factor", "pf", ".3f"),
        ("Sharpe", "sharpe", ".2f"),
        ("Net PnL", "net", "$"),
        ("Max Drawdown %", "max_dd", ".1f"),
        ("Stop Rate %", "stop_rate", ".1f"),
        ("Target Rate %", "target_rate", ".1f"),
        ("Flat Rate %", "flat_rate", ".1f"),
        ("Trades/Day (mean)", "trades_per_day_mean", ".1f"),
        ("Trades/Day (max)", "trades_per_day_max", "d"),
    ]

    for label, key, fmt in metric_rows:
        parts = [f"  {label:<25}"]
        for name in header_labels:
            val = all_results[name][key]
            if fmt == "$":
                parts.append(f" ${val:>17,.0f}")
            elif fmt == "d":
                parts.append(f" {int(val):>18}")
            else:
                parts.append(f" {val:>18{fmt}}")
        log("".join(parts))

    # Trades/day distribution
    log("\n  Trades/day distribution:")
    for name in header_labels:
        log(f"    {name}: {all_results[name]['trades_per_day_dist']}")

    # ── Annual breakdown ─────────────────────────────────────────────────────
    for name in header_labels:
        log(f"\n{'=' * 80}")
        log(f"ANNUAL BREAKDOWN: {name}")
        log(f"  {configs[name]['desc']}")
        log("=" * 80)

        ab = annual_breakdown(all_trades[name])
        if ab.empty:
            log("  No trades.")
            continue

        log(
            f"\n  {'Year':>6} {'Trades':>7} {'WR%':>6} {'PF':>7} {'Net PnL':>12} "
            f"{'Avg PnL':>10} {'Stops':>6} {'Targets':>8}"
        )
        log("  " + "-" * 70)
        for _, row in ab.iterrows():
            log(
                f"  {int(row['year']):>6} {int(row['trades']):>7} {row['wr']:>5.1f}% "
                f"{row['pf']:>7.3f} ${row['net']:>11,.0f} ${row['avg_pnl']:>9,.0f} "
                f"{int(row['stops']):>6} {int(row['targets']):>8}"
            )

        # Totals
        total_trades = int(ab["trades"].sum())
        total_net = float(ab["net"].sum())
        total_stops = int(ab["stops"].sum())
        total_targets = int(ab["targets"].sum())
        log("  " + "-" * 70)
        log(
            f"  {'TOTAL':>6} {total_trades:>7} {all_results[name]['wr']:>5.1f}% "
            f"{all_results[name]['pf']:>7.3f} ${total_net:>11,.0f} "
            f"${total_net/total_trades:>9,.0f} {total_stops:>6} {total_targets:>8}"
        )

        ab.to_csv(out_dir / f"annual_{name.lower().replace('-', '_')}.csv", index=False)

    # ── Save trade logs ──────────────────────────────────────────────────────
    for name in header_labels:
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
                for t in all_trades[name]
            ]
        )
        fname = f"trades_{name.lower().replace('-', '_')}.csv"
        trades_df.to_csv(out_dir / fname, index=False)

    # ── Equity curves ────────────────────────────────────────────────────────
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(14, 6))
        for name in header_labels:
            eq = pd.Series(all_eq[name])
            m = all_results[name]
            ax.plot(eq.values, linewidth=1.0, label=f"{name} (PF={m['pf']:.2f}, ${m['net']:+,.0f})")
        ax.set_title("Portfolio Equity Curves — Configuration Comparison")
        ax.set_ylabel("Equity ($)")
        ax.set_xlabel("Trading Days")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)
        ax.axhline(y=200_000, color="gray", linestyle="--", alpha=0.5, label="Starting Capital")
        fig.tight_layout()
        fig.savefig(out_dir / "equity_curves_comparison.png", dpi=150)
        log(f"\nEquity curve: {out_dir / 'equity_curves_comparison.png'}")
    except Exception as exc:
        log(f"\nCould not save equity curves: {exc}")

    # Save report
    (out_dir / "comparison_report.txt").write_text("\n".join(lines))
    log(f"Full report: {out_dir / 'comparison_report.txt'}")


if __name__ == "__main__":
    main()
