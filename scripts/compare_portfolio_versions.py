#!/usr/bin/env python
"""Phase 3: Before/after comparison + filter ablation on the FIXED portfolio backtest.

Imports the core simulation from run_portfolio_backtest.py's pre-computation,
then runs variants with different filter settings.

Outputs to docs/portfolio_investigation/phase3_results/
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

# Strategy constants (matching engine.py)
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
    """Load bars and pre-compute all caches (shared across variants)."""
    db = BarDatabase()
    db.connect()
    raw_bars: dict[str, pd.DataFrame] = {}
    for sym in SYMBOLS:
        df = load_bars(db, symbol=sym, timeframe=TimeFrame.ONE_MIN)
        if not df.empty:
            raw_bars[sym] = df
    db.close()

    # Day bars index
    day_bars: dict[str, dict[date, pd.DataFrame]] = defaultdict(dict)
    all_trading_dates: set[date] = set()
    for sym, df in raw_bars.items():
        df_et = df.copy()
        df_et.index = df_et.index.tz_convert(_ET)
        for d, grp in df_et.groupby(df_et.index.date):
            day_bars[sym][d] = grp
            all_trading_dates.add(d)
    trading_days = sorted(all_trading_dates)

    # SPY daily ADX + realized vol
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
        vol_series = pd.Series(df["volume"].astype(float).values)
        avg_vol_raw = vol_series.rolling(_VOL_WINDOW).mean().to_numpy()
        avg_vol_shifted = np.roll(avg_vol_raw, 1)
        avg_vol_shifted[0] = np.nan

        df_et = df.copy()
        df_et.index = df_et.index.tz_convert(_ET)
        df_et["_atr"] = atr_shifted
        df_et["_avg_vol"] = avg_vol_shifted
        for d, grp in df_et.groupby(df_et.index.date):
            day_bars[sym][d] = grp

    # ORB cache
    orb_cache: dict[str, dict[date, tuple[float, float]]] = defaultdict(dict)
    for sym in raw_bars:
        for d, grp in day_bars.get(sym, {}).items():
            orb_bars_df = grp.between_time("09:30", "09:34")
            if len(orb_bars_df) >= _ORB_BARS:
                orb_cache[sym][d] = (
                    float(orb_bars_df["high"].max()),
                    float(orb_bars_df["low"].min()),
                )

    # 15-min EMA shifted by 1 bar
    for sym, _df in raw_bars.items():
        all_day_dfs = [day_bars[sym][d] for d in sorted(day_bars.get(sym, {}).keys())]
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
        for d, grp in df_et.groupby(df_et.index.date):
            grp = grp.copy()
            grp["_ema15m"] = ema_1m.reindex(grp.index)
            day_bars[sym][d] = grp

    # Gap cache
    gap_cache: dict[str, dict[date, float]] = defaultdict(dict)
    for sym in raw_bars:
        prev_close_val = None
        for d in sorted(day_bars.get(sym, {}).keys()):
            grp = day_bars[sym][d]
            first_bar = grp.between_time("09:30", "09:30")
            if not first_bar.empty and prev_close_val is not None:
                session_open = float(first_bar.iloc[0]["open"])
                gap_cache[sym][d] = (session_open - prev_close_val) / prev_close_val * 100
            last_bar = grp.between_time("15:55", "15:59")
            if not last_bar.empty:
                prev_close_val = float(last_bar.iloc[-1]["close"])
            elif len(grp) > 0:
                prev_close_val = float(grp.iloc[-1]["close"])

    # VIX + econ
    vts_df = get_vix_term_structure()
    vts_lookup: dict[date, float] = {}
    if not vts_df.empty:
        vts_arr = compute_vix_term_structure_array(spy_df.index, vts_df)
        spy_et_vts = spy_df.copy()
        spy_et_vts.index = spy_et_vts.index.tz_convert(_ET)
        spy_et_vts["_vts"] = vts_arr
        for d, grp in spy_et_vts.groupby(spy_et_vts.index.date):
            vals = grp["_vts"].dropna()
            if len(vals) > 0:
                vts_lookup[d] = float(vals.iloc[0])

    econ_blocked_arr = compute_econ_blocked_array(spy_df.index)
    econ_blocked_dates: set[date] = set()
    spy_et_econ = spy_df.copy()
    spy_et_econ.index = spy_et_econ.index.tz_convert(_ET)
    spy_et_econ["_econ"] = econ_blocked_arr
    for d, grp in spy_et_econ.groupby(spy_et_econ.index.date):
        if grp["_econ"].max() > 0.5:
            econ_blocked_dates.add(d)

    return {
        "raw_bars": raw_bars,
        "day_bars": day_bars,
        "trading_days": trading_days,
        "adx_lookup": adx_lookup,
        "vol_lookup": vol_lookup,
        "orb_cache": orb_cache,
        "gap_cache": gap_cache,
        "vts_lookup": vts_lookup,
        "econ_blocked_dates": econ_blocked_dates,
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
    """Run the FIXED portfolio sim with configurable filters."""
    raw_bars = data["raw_bars"]
    day_bars = data["day_bars"]
    trading_days = data["trading_days"]
    adx_lookup = data["adx_lookup"]
    vol_lookup = data["vol_lookup"]
    orb_cache = data["orb_cache"]
    gap_cache = data["gap_cache"]
    vts_lookup = data["vts_lookup"]
    econ_blocked_dates = data["econ_blocked_dates"]

    equity = cash
    open_positions: list[Position] = []
    closed_trades: list[Trade] = []
    equity_curve: list[float] = []
    stats: dict[str, int] = defaultdict(int)

    for day in trading_days:
        if monday_filter and day.weekday() == 0:
            continue
        d_adx = adx_lookup.get(day)
        d_vol = vol_lookup.get(day)
        if adx_filter and d_adx is not None and d_adx <= _DAILY_ADX_MIN:
            continue
        if vol_filter and d_vol is not None and d_vol >= _REALIZED_VOL_MAX:
            continue
        if econ_filter and day in econ_blocked_dates:
            continue
        vts_ratio = vts_lookup.get(day)
        if vts_filter and vts_ratio is not None and vts_ratio > 1.0:
            continue

        daily_trade_count = 0

        # Kalman init
        kalman_state: dict[str, StreamingKalman] = {}
        for sym in raw_bars:
            orb = orb_cache.get(sym, {}).get(day)
            if orb is None:
                continue
            sym_day = day_bars.get(sym, {}).get(day)
            if sym_day is None or "_atr" not in sym_day.columns:
                continue
            orb_end = sym_day.between_time("09:34", "09:34")
            if orb_end.empty:
                continue
            sym_atr = float(orb_end.iloc[0]["_atr"])
            if np.isnan(sym_atr) or sym_atr <= 0:
                continue
            orb_bars_df = sym_day.between_time("09:30", "09:34")
            if len(orb_bars_df) >= _ORB_BARS:
                km = StreamingKalman()
                km.reset([float(r["close"]) for _, r in orb_bars_df.iterrows()], sym_atr)
                kalman_state[sym] = km

        # ORB window: bar-by-bar with interleaved exits + entries
        for minute_offset in range(5, 30):
            bar_time = datetime.combine(day, time(9, 30 + minute_offset)).replace(tzinfo=_ET)

            # Step A: check exits
            for pos in list(open_positions):
                pos_day_bars = day_bars.get(pos.symbol, {}).get(day)
                if pos_day_bars is None:
                    continue
                after = pos_day_bars[
                    (pos_day_bars.index > pos.entry_time) & (pos_day_bars.index <= bar_time)
                ]
                for _, bar_row in after.iterrows():
                    bar_o, bar_h, bar_l, bar_c = (
                        float(bar_row["open"]),
                        float(bar_row["high"]),
                        float(bar_row["low"]),
                        float(bar_row["close"]),
                    )
                    hit_tp = (
                        (bar_h >= pos.target) if pos.direction == "LONG" else (bar_l <= pos.target)
                    )
                    hit_sl = (bar_l <= pos.stop) if pos.direction == "LONG" else (bar_h >= pos.stop)
                    if hit_tp and hit_sl:
                        favor = bar_c > bar_o if pos.direction == "LONG" else bar_c < bar_o
                        hit_sl, hit_tp = (False, True) if favor else (True, False)
                    if hit_tp:
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
                                bar_row.name,
                                "target",
                            )
                        )
                        open_positions.remove(pos)
                        equity += pnl
                        stats["exits_target"] += 1
                        break
                    elif hit_sl:
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
                                bar_row.name,
                                "stop",
                            )
                        )
                        open_positions.remove(pos)
                        equity += pnl
                        stats["exits_stop"] += 1
                        break

            # Step B: entries
            if daily_trade_count >= max_daily or len(open_positions) >= max_concurrent:
                continue

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
                target_t = time(9, 30 + minute_offset)
                prev_t = time(9, 29 + minute_offset)
                curr = sym_day.between_time(target_t, target_t)
                prev = sym_day.between_time(prev_t, prev_t)
                if curr.empty or prev.empty:
                    continue
                close_val = float(curr.iloc[0]["close"])
                prev_close = float(prev.iloc[0]["close"])
                volume = float(curr.iloc[0]["volume"])

                if "_avg_vol" not in curr.columns:
                    continue
                avg_vol = float(curr.iloc[0]["_avg_vol"])
                if np.isnan(avg_vol) or avg_vol <= 0 or volume < avg_vol * _VOL_MULTIPLIER:
                    continue
                if "_atr" not in curr.columns:
                    continue
                atr = float(curr.iloc[0]["_atr"])
                if np.isnan(atr) or atr <= 0:
                    continue

                km = kalman_state.get(sym)
                if km is not None and km.ready:
                    km.update(close_val)
                kalman_mult = km.multiplier if km is not None and km.ready else 1.0

                if "_ema15m" in curr.columns:
                    ev = curr.iloc[0]["_ema15m"]
                    if not np.isnan(ev):
                        if close_val > orb_high and close_val <= ev:
                            continue
                        if close_val < orb_low and close_val >= ev:
                            continue

                gap_pct = gap_cache.get(sym, {}).get(day)
                if gap_pct is not None:
                    if close_val > orb_high and gap_pct < -_GAP_THRESHOLD_PCT:
                        continue
                    if close_val < orb_low and gap_pct > _GAP_THRESHOLD_PCT:
                        continue

                base_atr_risk = _ATR_MULTIPLIER * atr
                adaptive_stop = base_atr_risk * kalman_mult
                if close_val > orb_high and prev_close > orb_high:
                    d_ = "LONG"
                    e_ = close_val
                    sl_ = e_ - adaptive_stop
                    tp_ = e_ + _RISK_MULTIPLIER * base_atr_risk
                elif close_val < orb_low and prev_close < orb_low:
                    d_ = "SHORT"
                    e_ = close_val
                    sl_ = e_ + adaptive_stop
                    tp_ = e_ - _RISK_MULTIPLIER * base_atr_risk
                else:
                    continue

                score = (d_adx or 25) / 100 * 0.4 + orb_range_pct * 100 * 0.3 + 0.3
                candidates.append((sym, d_, e_, sl_, tp_, score))

            candidates.sort(key=lambda c: c[5], reverse=True)
            for sym, d_, e_, sl_, tp_, _ in candidates:
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
                    Position(sym, d_, e_, sl_, tp_, equity * _POSITION_SCALE, bar_time)
                )
                daily_trade_count += 1

        # Post-ORB: batch check SL/TP
        orb_end_time = datetime.combine(day, time(9, 59)).replace(tzinfo=_ET)
        for pos in list(open_positions):
            pos_day_bars = day_bars.get(pos.symbol, {}).get(day)
            if pos_day_bars is None:
                continue
            check_from = max(pos.entry_time, orb_end_time)
            after = pos_day_bars[pos_day_bars.index > check_from]
            market = after.between_time("09:30", "15:54")
            for _, bar_row in market.iterrows():
                bar_o, bar_h, bar_l, bar_c = (
                    float(bar_row["open"]),
                    float(bar_row["high"]),
                    float(bar_row["low"]),
                    float(bar_row["close"]),
                )
                hit_tp = (bar_h >= pos.target) if pos.direction == "LONG" else (bar_l <= pos.target)
                hit_sl = (bar_l <= pos.stop) if pos.direction == "LONG" else (bar_h >= pos.stop)
                if hit_tp and hit_sl:
                    favor = bar_c > bar_o if pos.direction == "LONG" else bar_c < bar_o
                    hit_sl, hit_tp = (False, True) if favor else (True, False)
                if hit_tp:
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
                            bar_row.name,
                            "target",
                        )
                    )
                    open_positions.remove(pos)
                    equity += pnl
                    stats["exits_target"] += 1
                    break
                elif hit_sl:
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
                            bar_row.name,
                            "stop",
                        )
                    )
                    open_positions.remove(pos)
                    equity += pnl
                    stats["exits_stop"] += 1
                    break

        # Force flat
        for pos in list(open_positions):
            pos_day = day_bars.get(pos.symbol, {}).get(day)
            if pos_day is None:
                continue
            flat = pos_day.between_time("15:55", "15:55")
            if flat.empty:
                late = pos_day.between_time("15:50", "15:59")
                if late.empty:
                    continue
                exit_p = float(late.iloc[-1]["close"])
            else:
                exit_p = float(flat.iloc[0]["close"])
            pnl = ((exit_p - pos.entry) if pos.direction == "LONG" else (pos.entry - exit_p)) * (
                pos.size / pos.entry
            )
            closed_trades.append(
                Trade(
                    pos.symbol,
                    pos.direction,
                    pos.entry,
                    exit_p,
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


def compute_metrics(trades: list[Trade], equity_curve: list[float]) -> dict:
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
    losses = pnl_s[pnl_s < 0]
    gp = float(w.sum()) if len(w) else 0
    gl = abs(float(losses.sum())) if len(losses) else 0.001
    eq = pd.Series(equity_curve)
    dd = float(((eq - eq.cummax()) / eq.cummax()).min()) * 100 if len(eq) > 1 else 0
    total = len(trades)
    return {
        "trades": total,
        "wr": len(w) / total * 100,
        "pf": gp / gl,
        "sharpe": float(pnl_s.mean() / pnl_s.std() * np.sqrt(252)) if pnl_s.std() > 0 else 0,
        "net": gp - abs(float(losses.sum())) if len(losses) else gp,
        "max_dd": dd,
        "stop_rate": sum(1 for t in trades if t.exit_reason == "stop") / total * 100,
        "target_rate": sum(1 for t in trades if t.exit_reason == "target") / total * 100,
        "flat_rate": sum(1 for t in trades if t.exit_reason == "flat") / total * 100,
    }


def main() -> None:
    out_dir = Path("docs/portfolio_investigation/phase3_results")
    out_dir.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []

    def log(msg: str) -> None:
        print(msg)
        lines.append(msg)

    log("=" * 80)
    log("PHASE 3: FILTER ABLATION ON FIXED VERSION")
    log("=" * 80)

    log("\nLoading data...")
    data = load_all_data()
    log(f"Loaded {len(data['raw_bars'])} symbols, {len(data['trading_days'])} trading days")

    variants = {
        "Baseline (all on)": {},
        "ADX filter OFF": {"adx_filter": False},
        "Monday block OFF": {"monday_filter": False},
        "Vol filter OFF": {"vol_filter": False},
        "Sector limit OFF": {"sector_filter": False},
        "Econ filter OFF": {"econ_filter": False},
        "VTS filter OFF": {"vts_filter": False},
        "Concurrent=5": {"max_concurrent": 5},
        "Concurrent=5 + Daily=8": {"max_concurrent": 5, "max_daily": 8},
        "ALL filters OFF": {
            "adx_filter": False,
            "vol_filter": False,
            "monday_filter": False,
            "econ_filter": False,
            "vts_filter": False,
            "sector_filter": False,
        },
    }

    results = []
    for name, kwargs in variants.items():
        log(f"\n  Running: {name}...")
        trades, eq, _st = run_sim(data, **kwargs)
        m = compute_metrics(trades, eq)
        results.append({"variant": name, **m})
        # Trades/day dist
        d_counts: dict[date, int] = defaultdict(int)
        for t in trades:
            d_ = t.entry_time.date() if hasattr(t.entry_time, "date") else t.entry_time
            if isinstance(d_, datetime):
                d_ = d_.date()
            d_counts[d_] += 1
        tc = pd.Series(list(d_counts.values())) if d_counts else pd.Series([0])
        log(
            f"    Trades={m['trades']}, WR={m['wr']:.1f}%, PF={m['pf']:.3f}, Sharpe={m['sharpe']:.2f}, "
            f"Net=${m['net']:,.0f}, DD={m['max_dd']:.1f}%, SL={m['stop_rate']:.0f}%, "
            f"TP={m['target_rate']:.0f}%, Trades/day={tc.mean():.1f} (max {tc.max()})"
        )

    # ── Before/After comparison ──────────────────────────────────────────────
    log("\n" + "=" * 80)
    log("BEFORE/AFTER COMPARISON")
    log("=" * 80)

    old = {
        "trades": 648,
        "wr": 39.4,
        "pf": 0.891,
        "sharpe": -0.76,
        "net": -10337,
        "max_dd": -5.2,
        "stop_rate": 60.3,
        "target_rate": 39.2,
        "flat_rate": 0.5,
    }
    new = results[0]  # Baseline

    log(f"\n  {'Metric':<25} {'OLD (buggy)':>15} {'NEW (fixed)':>15} {'Delta':>12}")
    log("  " + "-" * 70)
    for label, key in [
        ("Trades", "trades"),
        ("Win Rate %", "wr"),
        ("Profit Factor", "pf"),
        ("Sharpe", "sharpe"),
        ("Net PnL $", "net"),
        ("Max Drawdown %", "max_dd"),
        ("Stop Rate %", "stop_rate"),
        ("Target Rate %", "target_rate"),
    ]:
        o, n = old[key], new[key]
        d = n - o
        if key == "net":
            log(f"  {label:<25} ${o:>14,.0f} ${n:>14,.0f} ${d:>+11,.0f}")
        elif key == "trades":
            log(f"  {label:<25} {o:>15} {n:>15} {d:>+12}")
        else:
            log(f"  {label:<25} {o:>15.1f} {n:>15.1f} {d:>+12.1f}")

    # ── Ablation table ───────────────────────────────────────────────────────
    log("\n" + "=" * 80)
    log("FILTER ABLATION TABLE")
    log("=" * 80)
    log(
        f"\n  {'Variant':<30} {'Trds':>5} {'WR%':>6} {'PF':>6} {'Shrp':>6} {'Net PnL':>12} "
        f"{'DD%':>6} {'SL%':>5} {'TP%':>5}"
    )
    log("  " + "-" * 90)
    for r in results:
        log(
            f"  {r['variant']:<30} {r['trades']:>5} {r['wr']:>5.1f}% {r['pf']:>6.3f} "
            f"{r['sharpe']:>6.2f} ${r['net']:>11,.0f} {r['max_dd']:>5.1f}% "
            f"{r['stop_rate']:>4.0f}% {r['target_rate']:>4.0f}%"
        )

    # Save
    pd.DataFrame(results).to_csv(out_dir / "filter_ablation.csv", index=False)
    (out_dir / "comparison_report.txt").write_text("\n".join(lines))
    log(f"\nResults saved to {out_dir}/")


if __name__ == "__main__":
    main()
