#!/usr/bin/env python
"""Deep investigation: why portfolio backtest underperforms individual ticker backtest.

Phase 1 analysis covering all 7 investigation areas:
1. Signal selection audit (accepted vs rejected PnL)
2. Filter impact analysis (toggle each filter independently)
3. Signal selection mechanism documentation
4. Exit analysis (stop-to-target ratio per ticker)
5. Timing analysis (hour + day-of-week breakdown)
6. Position sizing check
7. Ticker concentration analysis

Outputs to docs/portfolio_investigation/
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
OUT_DIR = Path("docs/portfolio_investigation")

# ── Strategy constants (matching engine.py) ──────────────────────────────────
_ORB_BARS = 5
_ATR_MULTIPLIER = 1.5
_RISK_MULTIPLIER = 2.0
_VOL_MULTIPLIER = 1.5
_VOL_WINDOW = 20
_DAILY_ADX_MIN = 25.0
_REALIZED_VOL_MAX = 0.18
_MIN_ORB_PCT = 0.0015
_GAP_THRESHOLD_PCT = 0.3
_POSITION_SCALE = 0.25

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


@dataclass
class SignalCandidate:
    """A signal that was generated, whether accepted or rejected."""

    symbol: str
    direction: str
    entry: float
    stop: float
    target: float
    score: float
    day: date
    minute_offset: int
    accepted: bool = False
    reject_reason: str = ""
    hypothetical_pnl: float | None = None


def load_all_data():
    """Load bars, compute daily indicators, caches."""
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

    # ORB cache
    orb_cache: dict[str, dict[date, tuple[float, float]]] = defaultdict(dict)
    atr_cache: dict[str, dict[date, float]] = defaultdict(dict)
    vol_avg_cache: dict[str, dict[date, float]] = defaultdict(dict)

    for sym, df in raw_bars.items():
        h = df["high"].astype(float).values
        lo = df["low"].astype(float).values
        c = df["close"].astype(float).values
        if len(c) >= 15:
            atr_1min = talib.ATR(h, lo, c, timeperiod=14)
            df_et = df.copy()
            df_et.index = df_et.index.tz_convert(_ET)
            df_et["_atr"] = atr_1min
            for d, grp in df_et.groupby(df_et.index.date):
                orb_end = grp.between_time("09:34", "09:34")
                if not orb_end.empty:
                    val = orb_end.iloc[0]["_atr"]
                    if not np.isnan(val):
                        atr_cache[sym][d] = float(val)

        for d, grp in day_bars.get(sym, {}).items():
            orb_bars_df = grp.between_time("09:30", "09:34")
            if len(orb_bars_df) >= _ORB_BARS:
                orb_cache[sym][d] = (
                    float(orb_bars_df["high"].max()),
                    float(orb_bars_df["low"].min()),
                )
            pre_orb = grp.between_time("09:30", "09:59")
            if len(pre_orb) >= 5:
                vol_avg_cache[sym][d] = float(pre_orb["volume"].iloc[:20].mean())

    # EMA 15m cache (FIX: only use values up to current bar, not end-of-day)
    ema15m_cache: dict[str, dict[date, dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
    for sym, df in raw_bars.items():
        df_et = df.copy()
        df_et.index = df_et.index.tz_convert(_ET)
        bars_15m = df_et.resample("15min").agg({"close": "last"}).dropna()
        if len(bars_15m) >= 21:
            from talipp.indicators import EMA as _EMA

            ema = _EMA(period=20)
            for ts, row in bars_15m.iterrows():
                ema.add(float(row["close"]))
                if ema.output_values and ema.output_values[-1] is not None:
                    d = ts.date()
                    minute = ts.hour * 60 + ts.minute
                    ema15m_cache[sym][d][minute] = float(ema.output_values[-1])

    # Gap cache
    gap_cache: dict[str, dict[date, float]] = defaultdict(dict)
    for sym, df in raw_bars.items():
        df_et = df.copy()
        df_et.index = df_et.index.tz_convert(_ET)
        prev_close_val = None
        for d, grp in df_et.groupby(df_et.index.date):
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

    # VIX term structure
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

    # Econ calendar
    econ_blocked_arr = compute_econ_blocked_array(spy_df.index)
    econ_blocked_dates: set[date] = set()
    spy_et_for_econ = spy_df.copy()
    spy_et_for_econ.index = spy_et_for_econ.index.tz_convert(_ET)
    spy_et_for_econ["_econ"] = econ_blocked_arr
    for d, grp in spy_et_for_econ.groupby(spy_et_for_econ.index.date):
        if grp["_econ"].max() > 0.5:
            econ_blocked_dates.add(d)

    return (
        raw_bars,
        day_bars,
        trading_days,
        adx_lookup,
        vol_lookup,
        orb_cache,
        atr_cache,
        vol_avg_cache,
        ema15m_cache,
        gap_cache,
        vts_lookup,
        econ_blocked_dates,
    )


def compute_hypothetical_pnl(
    signal: SignalCandidate,
    day_bars: dict[str, dict[date, pd.DataFrame]],
) -> float | None:
    """Simulate what would have happened if this signal was accepted."""
    sym_day = day_bars.get(signal.symbol, {}).get(signal.day)
    if sym_day is None:
        return None

    entry_time = datetime.combine(signal.day, time(9, 30 + signal.minute_offset)).replace(
        tzinfo=_ET
    )
    after_entry = sym_day[sym_day.index > entry_time]
    market_bars = after_entry.between_time("09:30", "15:54")

    for _, bar_row in market_bars.iterrows():
        bar_h = float(bar_row["high"])
        bar_l = float(bar_row["low"])
        bar_c = float(bar_row["close"])
        bar_o = float(bar_row["open"])

        hit_target = False
        hit_stop = False

        if signal.direction == "LONG":
            hit_target = bar_h >= signal.target
            hit_stop = bar_l <= signal.stop
        else:
            hit_target = bar_l <= signal.target
            hit_stop = bar_h >= signal.stop

        if hit_target and hit_stop:
            favor = bar_c > bar_o if signal.direction == "LONG" else bar_c < bar_o
            if favor:
                hit_stop = False
            else:
                hit_target = False

        size = 200_000 * _POSITION_SCALE
        if hit_target:
            if signal.direction == "LONG":
                return (signal.target - signal.entry) * (size / signal.entry)
            else:
                return (signal.entry - signal.target) * (size / signal.entry)
        elif hit_stop:
            if signal.direction == "LONG":
                return (signal.stop - signal.entry) * (size / signal.entry)
            else:
                return (signal.entry - signal.stop) * (size / signal.entry)

    # Force flat at 15:55
    flat_bars = sym_day.between_time("15:55", "15:55")
    if flat_bars.empty:
        late_bars = sym_day.between_time("15:50", "15:59")
        if late_bars.empty:
            return None
        exit_price = float(late_bars.iloc[-1]["close"])
    else:
        exit_price = float(flat_bars.iloc[0]["close"])

    size = 200_000 * _POSITION_SCALE
    if signal.direction == "LONG":
        return (exit_price - signal.entry) * (size / signal.entry)
    else:
        return (signal.entry - exit_price) * (size / signal.entry)


def run_portfolio_sim(
    raw_bars,
    day_bars,
    trading_days,
    adx_lookup,
    vol_lookup,
    orb_cache,
    atr_cache,
    vol_avg_cache,
    ema15m_cache,
    gap_cache,
    vts_lookup,
    econ_blocked_dates,
    cash=200_000,
    max_concurrent=3,
    max_daily=5,
    max_sector=2,
    adx_filter=True,
    vol_filter=True,
    monday_filter=True,
    econ_filter=True,
    vts_filter=True,
    sector_filter=True,
    collect_signals=False,
    compute_rejected_pnl=False,
) -> tuple[list[Trade], list[float], dict[str, int], list[SignalCandidate]]:
    """Run portfolio sim with configurable filters. Returns trades, equity, counters, signals."""
    equity = cash
    open_positions: list[Position] = []
    closed_trades: list[Trade] = []
    equity_curve: list[float] = []
    stats: dict[str, int] = defaultdict(int)
    all_signals: list[SignalCandidate] = []

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
        if econ_filter and day in econ_blocked_dates:
            stats["days_econ_blocked"] += 1
            continue
        vts_ratio = vts_lookup.get(day)
        if vts_filter and vts_ratio is not None and vts_ratio > 1.0:
            stats["days_backwardation"] += 1
            continue

        stats["days_tradeable"] += 1
        daily_trade_count = 0

        # Kalman filters
        kalman_state: dict[str, StreamingKalman] = {}
        for sym in raw_bars:
            orb = orb_cache.get(sym, {}).get(day)
            sym_atr = atr_cache.get(sym, {}).get(day, 0)
            if orb is not None and sym_atr > 0:
                sym_day = day_bars.get(sym, {}).get(day)
                if sym_day is not None:
                    orb_bars_df = sym_day.between_time("09:30", "09:34")
                    if len(orb_bars_df) >= _ORB_BARS:
                        km = StreamingKalman()
                        km.reset([float(r["close"]) for _, r in orb_bars_df.iterrows()], sym_atr)
                        kalman_state[sym] = km

        # Phase 1: Generate signals (9:35-9:59)
        for minute_offset in range(5, 30):
            if daily_trade_count >= max_daily or len(open_positions) >= max_concurrent:
                break

            candidates: list[SignalCandidate] = []
            bar_time = datetime.combine(day, time(9, 30 + minute_offset)).replace(tzinfo=_ET)

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

                target_time = time(9, 30 + minute_offset)
                prev_time_val = time(9, 29 + minute_offset)
                curr = sym_day.between_time(target_time, target_time)
                prev = sym_day.between_time(prev_time_val, prev_time_val)
                if curr.empty or prev.empty:
                    continue

                close_val = float(curr.iloc[0]["close"])
                prev_close = float(prev.iloc[0]["close"])
                volume = float(curr.iloc[0]["volume"])

                avg_vol = vol_avg_cache.get(sym, {}).get(day, 100_000)
                if volume < avg_vol * _VOL_MULTIPLIER:
                    continue

                atr = atr_cache.get(sym, {}).get(day, orb_high - orb_low)
                if atr <= 0:
                    continue

                km = kalman_state.get(sym)
                if km is not None and km.ready:
                    km.update(close_val)
                kalman_mult = km.multiplier if km is not None and km.ready else 1.0

                # EMA filter (use latest available BEFORE current bar)
                ema_vals = ema15m_cache.get(sym, {}).get(day, {})
                current_minute = 9 * 60 + 30 + minute_offset
                ema_val = None
                for m in sorted(ema_vals.keys()):
                    if m < current_minute:
                        ema_val = ema_vals[m]
                    else:
                        break
                # If no intraday EMA, use previous day's last EMA
                if ema_val is None:
                    prev_days = sorted(d for d in ema15m_cache.get(sym, {}) if d < day)
                    if prev_days:
                        prev_ema_vals = ema15m_cache[sym][prev_days[-1]]
                        if prev_ema_vals:
                            ema_val = prev_ema_vals[max(prev_ema_vals.keys())]

                if ema_val is not None:
                    if close_val > orb_high and close_val <= ema_val:
                        continue
                    if close_val < orb_low and close_val >= ema_val:
                        continue

                # Gap filter
                gap_pct = gap_cache.get(sym, {}).get(day)
                if gap_pct is not None:
                    if close_val > orb_high and gap_pct < -_GAP_THRESHOLD_PCT:
                        continue
                    if close_val < orb_low and gap_pct > _GAP_THRESHOLD_PCT:
                        continue

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

                stats["signals_raw"] += 1
                score = (d_adx or 25) / 100 * 0.4 + orb_range_pct * 100 * 0.3 + 0.3
                candidates.append(
                    SignalCandidate(
                        symbol=sym,
                        direction=direction,
                        entry=entry,
                        stop=stop_price,
                        target=target_price,
                        score=score,
                        day=day,
                        minute_offset=minute_offset,
                    )
                )

            candidates.sort(key=lambda c: c.score, reverse=True)

            for sig in candidates:
                if len(open_positions) >= max_concurrent:
                    sig.reject_reason = "concurrent_limit"
                    stats["rejected_concurrent"] += 1
                elif daily_trade_count >= max_daily:
                    sig.reject_reason = "daily_limit"
                    stats["rejected_daily"] += 1
                elif sector_filter:
                    sym_sector = SECTOR_MAP.get(sig.symbol, "other")
                    same_sector = sum(
                        1 for p in open_positions if SECTOR_MAP.get(p.symbol, "other") == sym_sector
                    )
                    if same_sector >= max_sector:
                        sig.reject_reason = "sector_limit"
                        stats["rejected_sector"] += 1
                    else:
                        sig.accepted = True
                else:
                    sig.accepted = True

                if sig.accepted:
                    pos_size = equity * _POSITION_SCALE
                    open_positions.append(
                        Position(
                            sig.symbol,
                            sig.direction,
                            sig.entry,
                            sig.stop,
                            sig.target,
                            pos_size,
                            bar_time,
                        )
                    )
                    daily_trade_count += 1
                    stats["signals_accepted"] += 1

            if collect_signals:
                all_signals.extend(candidates)

        # Phase 2: SL/TP check on all bars
        for pos in list(open_positions):
            pos_day_bars = day_bars.get(pos.symbol, {}).get(day)
            if pos_day_bars is None:
                continue
            after_entry = pos_day_bars[pos_day_bars.index > pos.entry_time]
            market_bars = after_entry.between_time("09:30", "15:54")
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
                    if pos.direction == "LONG":
                        pnl = (pos.target - pos.entry) * (pos.size / pos.entry)
                    else:
                        pnl = (pos.entry - pos.target) * (pos.size / pos.entry)
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
                    stats["exits_target"] += 1
                    break
                elif hit_stop:
                    if pos.direction == "LONG":
                        pnl = (pos.stop - pos.entry) * (pos.size / pos.entry)
                    else:
                        pnl = (pos.entry - pos.stop) * (pos.size / pos.entry)
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
                    stats["exits_stop"] += 1
                    break

        # Phase 3: Force flat
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
                pnl = (exit_price - pos.entry) * (pos.size / pos.entry)
            else:
                pnl = (pos.entry - exit_price) * (pos.size / pos.entry)
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
            stats["exits_flat"] += 1

        equity_curve.append(equity)

    return closed_trades, equity_curve, dict(stats), all_signals


def compute_stats(trades: list[Trade]) -> dict[str, float]:
    """Compute standard backtest statistics."""
    if not trades:
        return {"trades": 0, "wr": 0, "pf": 0, "sharpe": 0, "net": 0}
    pnl_s = pd.Series([t.pnl for t in trades])
    winners = pnl_s[pnl_s > 0]
    losers = pnl_s[pnl_s < 0]
    gp = float(winners.sum()) if len(winners) else 0
    gl = abs(float(losers.sum())) if len(losers) else 0.001
    return {
        "trades": len(trades),
        "wr": len(winners) / len(trades) * 100,
        "pf": gp / gl,
        "sharpe": float(pnl_s.mean() / pnl_s.std() * np.sqrt(252)) if pnl_s.std() > 0 else 0,
        "net": gp - abs(float(losers.sum())) if len(losers) else gp,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output_lines: list[str] = []

    def log(msg: str) -> None:
        print(msg)
        output_lines.append(msg)

    log("=" * 80)
    log("PORTFOLIO BACKTEST DEEP INVESTIGATION")
    log("=" * 80)

    log("\nLoading data...")
    data = load_all_data()
    (
        raw_bars,
        day_bars,
        trading_days,
        _adx_lookup,
        _vol_lookup,
        _orb_cache,
        _atr_cache,
        _vol_avg_cache,
        _ema15m_cache,
        _gap_cache,
        _vts_lookup,
        _econ_blocked_dates,
    ) = data

    log(f"Loaded {len(raw_bars)} symbols, {len(trading_days)} trading days")

    # Load individual backtest results for cross-reference
    indiv_csv = Path("docs/backtest_results.csv")
    indiv_df = pd.read_csv(indiv_csv) if indiv_csv.exists() else pd.DataFrame()
    log(f"Individual backtest: {len(indiv_df)} trades loaded")

    # ══════════════════════════════════════════════════════════════════════════
    # INVESTIGATION 1: Signal Selection Audit
    # ══════════════════════════════════════════════════════════════════════════
    log("\n" + "=" * 80)
    log("1. SIGNAL SELECTION AUDIT: Accepted vs Rejected PnL")
    log("=" * 80)

    trades_base, _eq_base, stats_base, signals_base = run_portfolio_sim(
        *data,
        collect_signals=True,
        compute_rejected_pnl=True,
    )
    base_stats = compute_stats(trades_base)

    log(
        f"\nBaseline: {base_stats['trades']} trades, WR={base_stats['wr']:.1f}%, "
        f"PF={base_stats['pf']:.3f}, Net=${base_stats['net']:,.0f}"
    )
    log(f"Counters: {stats_base}")

    # Compute hypothetical PnL for rejected signals
    accepted = [s for s in signals_base if s.accepted]
    rejected = [s for s in signals_base if not s.accepted]

    log(f"\nTotal signals: {len(signals_base)}")
    log(f"Accepted: {len(accepted)}")
    log(f"Rejected: {len(rejected)}")

    # Compute hypothetical PnL for rejected signals
    log("\nComputing hypothetical PnL for rejected signals (this may take a moment)...")
    reject_pnls = []
    for sig in rejected:
        hyp_pnl = compute_hypothetical_pnl(sig, day_bars)
        if hyp_pnl is not None:
            sig.hypothetical_pnl = hyp_pnl
            reject_pnls.append(hyp_pnl)

    if reject_pnls:
        rej_s = pd.Series(reject_pnls)
        rej_winners = rej_s[rej_s > 0]
        rej_losers = rej_s[rej_s < 0]
        rej_gp = float(rej_winners.sum()) if len(rej_winners) else 0
        rej_gl = abs(float(rej_losers.sum())) if len(rej_losers) else 0.001

        log("\nRejected signals hypothetical performance:")
        log(f"  Total: {len(reject_pnls)}")
        log(f"  Winners: {len(rej_winners)} ({len(rej_winners)/len(reject_pnls)*100:.1f}%)")
        log(f"  Losers: {len(rej_losers)} ({len(rej_losers)/len(reject_pnls)*100:.1f}%)")
        log(f"  PF: {rej_gp/rej_gl:.3f}")
        log(f"  Net hypothetical PnL: ${rej_gp - abs(float(rej_losers.sum())):,.0f}")
        log(f"  Avg rejected PnL: ${rej_s.mean():,.2f}")

        # Break down by rejection reason
        for reason in ["concurrent_limit", "daily_limit", "sector_limit"]:
            r_sigs = [
                s for s in rejected if s.reject_reason == reason and s.hypothetical_pnl is not None
            ]
            if r_sigs:
                r_pnls = pd.Series([s.hypothetical_pnl for s in r_sigs])
                r_win = r_pnls[r_pnls > 0]
                r_lose = r_pnls[r_pnls < 0]
                r_gp = float(r_win.sum()) if len(r_win) else 0
                r_gl = abs(float(r_lose.sum())) if len(r_lose) else 0.001
                log(f"\n  Rejected by {reason}:")
                log(
                    f"    Count: {len(r_sigs)}, WR: {len(r_win)/len(r_sigs)*100:.1f}%, "
                    f"PF: {r_gp/r_gl:.3f}, Net: ${r_gp-abs(float(r_lose.sum())):,.0f}"
                )

        # Top rejected tickers by opportunity cost
        log("\n  Opportunity cost by ticker (rejected signals):")
        ticker_opp: dict[str, list[float]] = defaultdict(list)
        for sig in rejected:
            if sig.hypothetical_pnl is not None:
                ticker_opp[sig.symbol].append(sig.hypothetical_pnl)
        rows = []
        for sym, pnls in sorted(ticker_opp.items(), key=lambda x: sum(x[1]), reverse=True):
            s = pd.Series(pnls)
            rows.append(
                {
                    "ticker": sym,
                    "rejected": len(pnls),
                    "hyp_wr": f"{len(s[s>0])/len(s)*100:.0f}%",
                    "hyp_net": f"${sum(pnls):,.0f}",
                }
            )
        opp_df = pd.DataFrame(rows)
        log(opp_df.to_string(index=False))

    # Save signal audit
    sig_rows = []
    for s in signals_base:
        sig_rows.append(
            {
                "symbol": s.symbol,
                "direction": s.direction,
                "day": s.day,
                "minute": s.minute_offset + 30,
                "entry": s.entry,
                "score": round(s.score, 4),
                "accepted": s.accepted,
                "reject_reason": s.reject_reason,
                "hypothetical_pnl": round(s.hypothetical_pnl, 2)
                if s.hypothetical_pnl is not None
                else None,
            }
        )
    pd.DataFrame(sig_rows).to_csv(OUT_DIR / "signal_audit.csv", index=False)
    log(f"\nSaved: {OUT_DIR / 'signal_audit.csv'}")

    # ══════════════════════════════════════════════════════════════════════════
    # INVESTIGATION 2: Filter Impact Analysis
    # ══════════════════════════════════════════════════════════════════════════
    log("\n" + "=" * 80)
    log("2. FILTER IMPACT ANALYSIS")
    log("=" * 80)

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

    filter_results = []
    for name, kwargs in variants.items():
        trades, _eq, st, _ = run_portfolio_sim(*data, **kwargs)
        s = compute_stats(trades)
        filter_results.append({"variant": name, **s, "stats": st})
        log(f"\n  {name}:")
        log(
            f"    Trades={s['trades']}, WR={s['wr']:.1f}%, PF={s['pf']:.3f}, "
            f"Sharpe={s['sharpe']:.2f}, Net=${s['net']:,.0f}"
        )

    # Comparison table
    log("\n  Filter Impact Comparison:")
    log(f"  {'Variant':<35} {'Trades':>6} {'WR%':>6} {'PF':>6} {'Sharpe':>7} {'Net PnL':>12}")
    log("  " + "-" * 75)
    for r in filter_results:
        log(
            f"  {r['variant']:<35} {r['trades']:>6} {r['wr']:>5.1f}% {r['pf']:>6.3f} "
            f"{r['sharpe']:>7.2f} ${r['net']:>11,.0f}"
        )

    pd.DataFrame([{k: v for k, v in r.items() if k != "stats"} for r in filter_results]).to_csv(
        OUT_DIR / "filter_impact.csv", index=False
    )

    # ══════════════════════════════════════════════════════════════════════════
    # INVESTIGATION 3: Signal Selection Mechanism
    # ══════════════════════════════════════════════════════════════════════════
    log("\n" + "=" * 80)
    log("3. SIGNAL SELECTION MECHANISM")
    log("=" * 80)

    log("""
  Current ranking formula:
    score = (daily_ADX / 100) * 0.4 + (orb_range_pct * 100) * 0.3 + 0.3

  Components:
    - ADX weight (0.4): Same for ALL tickers on same day (SPY-based). Ranges ~0.10-0.20.
    - ORB range weight (0.3): Varies by ticker. Wider ORB = higher score.
      This FAVORS volatile tickers (TQQQ 3x, SOXL 3x, MSTR, TSLA).
    - Fixed base (0.3): Constant for all signals.

  CRITICAL ISSUE: The score is dominated by ORB range, which systematically
  selects the most volatile tickers. 3x leveraged ETFs (TQQQ, SOXL) have
  the widest ORB ranges, but their individual PFs are among the lowest:
    TQQQ: PF=1.03, SOXL: PF=1.21

  Meanwhile, high-PF tickers with narrower ranges get deprioritized:
    SMCI: PF=1.90, AMZN: PF=1.76, PYPL: PF=1.71

  SECOND ISSUE: Phase 1/Phase 2 separation.
  The minute loop (Phase 1, lines 313-424) BREAKs when max_concurrent is hit.
  Phase 2 (SL/TP checking, lines 429-504) runs AFTER Phase 1 completes.
  This means if 3 positions are opened at 9:35, the loop breaks at 9:36
  even though a position might exit at 9:37 and free a slot.
  Effective max trades/day = min(max_concurrent, max_daily) = 3, NOT 5.
""")

    # Verify: how many trades per day?
    days_with_trades: dict[date, int] = defaultdict(int)
    for t in trades_base:
        d = t.entry_time.date() if hasattr(t.entry_time, "date") else t.entry_time
        if isinstance(d, datetime):
            d = d.date()
        days_with_trades[d] += 1

    trade_counts = pd.Series(list(days_with_trades.values()))
    log("  Trades per day distribution:")
    log(f"    Mean: {trade_counts.mean():.1f}")
    log(f"    Max:  {trade_counts.max()}")
    log(f"    Distribution: {dict(trade_counts.value_counts().sort_index())}")

    # ══════════════════════════════════════════════════════════════════════════
    # INVESTIGATION 4: Exit Analysis
    # ══════════════════════════════════════════════════════════════════════════
    log("\n" + "=" * 80)
    log("4. EXIT ANALYSIS: Stop-to-Target Ratio")
    log("=" * 80)

    # Portfolio exit breakdown
    port_exits = defaultdict(lambda: defaultdict(int))
    port_pnl_by_ticker = defaultdict(list)
    for t in trades_base:
        port_exits[t.symbol][t.exit_reason] += 1
        port_pnl_by_ticker[t.symbol].append(t.pnl)

    log(
        f"\n  Portfolio exits: target={stats_base.get('exits_target', 0)}, "
        f"stop={stats_base.get('exits_stop', 0)}, flat={stats_base.get('exits_flat', 0)}"
    )
    total_exits = sum(stats_base.get(f"exits_{r}", 0) for r in ["target", "stop", "flat"])
    if total_exits > 0:
        log(f"  Stop rate: {stats_base.get('exits_stop', 0)/total_exits*100:.1f}%")
        log(f"  Flat rate: {stats_base.get('exits_flat', 0)/total_exits*100:.1f}%")

    # Individual backtest exit breakdown per ticker
    if not indiv_df.empty and "ExitPrice" in indiv_df.columns:
        log("\n  Per-ticker exit analysis (individual vs portfolio):")
        log(
            f"  {'Ticker':<8} {'Indiv Trades':>12} {'Indiv SL%':>10} {'Port Trades':>12} "
            f"{'Port SL%':>10} {'Port Flat%':>10}"
        )
        log("  " + "-" * 70)

        for sym in sorted(
            set(
                list(port_exits.keys())
                + list(indiv_df["ticker"].unique() if "ticker" in indiv_df.columns else [])
            )
        ):
            # Individual
            if "ticker" in indiv_df.columns:
                ind_sym = indiv_df[indiv_df["ticker"] == sym]
                ind_total = len(ind_sym)
                if ind_total > 0 and "SL" in ind_sym.columns and "ExitPrice" in ind_sym.columns:
                    # SL hit = ExitPrice equals SL
                    ind_sl = len(ind_sym[abs(ind_sym["ExitPrice"] - ind_sym["SL"]) < 0.01])
                    ind_sl_pct = ind_sl / ind_total * 100
                else:
                    ind_total = 0
                    ind_sl_pct = 0
            else:
                ind_total = 0
                ind_sl_pct = 0

            # Portfolio
            pe = port_exits.get(sym, {})
            p_total = sum(pe.values())
            p_sl = pe.get("stop", 0)
            p_flat = pe.get("flat", 0)
            p_sl_pct = p_sl / p_total * 100 if p_total > 0 else 0
            p_flat_pct = p_flat / p_total * 100 if p_total > 0 else 0

            if ind_total > 0 or p_total > 0:
                log(
                    f"  {sym:<8} {ind_total:>12} {ind_sl_pct:>9.1f}% {p_total:>12} "
                    f"{p_sl_pct:>9.1f}% {p_flat_pct:>9.1f}%"
                )

    # ══════════════════════════════════════════════════════════════════════════
    # INVESTIGATION 5: Timing Analysis
    # ══════════════════════════════════════════════════════════════════════════
    log("\n" + "=" * 80)
    log("5. TIMING ANALYSIS: Hour and Day-of-Week Breakdown")
    log("=" * 80)

    # Portfolio trades by entry hour
    hour_stats: dict[int, list[float]] = defaultdict(list)
    dow_stats: dict[int, list[float]] = defaultdict(list)
    dow_names = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri"}

    for t in trades_base:
        et = t.entry_time
        if hasattr(et, "hour"):
            hour_stats[et.hour].append(t.pnl)
        if hasattr(et, "weekday"):
            dow_stats[et.weekday()].append(t.pnl)

    log("\n  Entry hour breakdown (ET):")
    log(f"  {'Hour':>6} {'Trades':>7} {'WR%':>6} {'PF':>6} {'Net':>10}")
    log("  " + "-" * 40)
    for h in sorted(hour_stats.keys()):
        pnls = pd.Series(hour_stats[h])
        w = pnls[pnls > 0]
        losses = pnls[pnls < 0]
        gp = float(w.sum()) if len(w) else 0
        gl = abs(float(losses.sum())) if len(losses) else 0.001
        log(
            f"  {h:>6} {len(pnls):>7} {len(w)/len(pnls)*100:>5.1f}% {gp/gl:>6.3f} ${gp-abs(float(losses.sum())) if len(losses) else gp:>9,.0f}"
        )

    log("\n  Day-of-week breakdown:")
    log(f"  {'Day':>6} {'Trades':>7} {'WR%':>6} {'PF':>6} {'Net':>10}")
    log("  " + "-" * 40)
    for d in sorted(dow_stats.keys()):
        pnls = pd.Series(dow_stats[d])
        w = pnls[pnls > 0]
        losses = pnls[pnls < 0]
        gp = float(w.sum()) if len(w) else 0
        gl = abs(float(losses.sum())) if len(losses) else 0.001
        log(
            f"  {dow_names.get(d, str(d)):>6} {len(pnls):>7} {len(w)/len(pnls)*100:>5.1f}% "
            f"{gp/gl:>6.3f} ${gp-abs(float(losses.sum())) if len(losses) else gp:>9,.0f}"
        )

    # ══════════════════════════════════════════════════════════════════════════
    # INVESTIGATION 6: Position Sizing Check
    # ══════════════════════════════════════════════════════════════════════════
    log("\n" + "=" * 80)
    log("6. POSITION SIZING CHECK")
    log("=" * 80)

    log(f"""
  Portfolio position sizing:
    pos_size = equity * _POSITION_SCALE = ${200_000 * _POSITION_SCALE:,.0f} (at start)
    shares = pos_size / entry_price
    PnL = (exit - entry) * shares

  Individual backtest position sizing:
    cash = $50,000 per ticker (backtesting.py)
    size = position_scale_factor = 0.25 (fraction of equity)
    => $12,500 per trade per ticker

  Portfolio: $50,000 per trade (25% of $200K)
  Individual: $12,500 per trade (25% of $50K)

  Position size is 4x larger in portfolio, BUT portfolio has $200K total
  vs individual which effectively deploys $50K * 26 = $1.3M across tickers.

  The stops and targets are calibrated identically (same ATR multiplier,
  same risk multiplier). The DOLLAR PnL per trade scales linearly with
  position size, so this isn't a bug — just different capital deployment.

  KEY FINDING: No position sizing bug. The difference is portfolio
  constraint (3 concurrent * $50K = $150K deployed) vs individual
  (26 tickers * $12.5K = $325K deployed simultaneously).
""")

    # ══════════════════════════════════════════════════════════════════════════
    # INVESTIGATION 7: Ticker Concentration
    # ══════════════════════════════════════════════════════════════════════════
    log("\n" + "=" * 80)
    log("7. TICKER CONCENTRATION ANALYSIS")
    log("=" * 80)

    # Portfolio trade distribution
    port_ticker_dist: dict[str, dict] = {}
    for sym in SYMBOLS:
        sym_trades = [t for t in trades_base if t.symbol == sym]
        if sym_trades:
            pnls = pd.Series([t.pnl for t in sym_trades])
            w = pnls[pnls > 0]
            losses = pnls[pnls < 0]
            gp = float(w.sum()) if len(w) else 0
            gl = abs(float(losses.sum())) if len(losses) else 0.001
            port_ticker_dist[sym] = {
                "trades": len(sym_trades),
                "wr": len(w) / len(sym_trades) * 100,
                "pf": gp / gl,
                "net": gp - abs(float(losses.sum())) if len(losses) else gp,
            }
        else:
            port_ticker_dist[sym] = {"trades": 0, "wr": 0, "pf": 0, "net": 0}

    # Individual backtest PF per ticker
    indiv_pf: dict[str, float] = {}
    indiv_trades_count: dict[str, int] = {}
    if not indiv_df.empty and "ticker" in indiv_df.columns and "PnL" in indiv_df.columns:
        for sym in SYMBOLS:
            sym_df = indiv_df[indiv_df["ticker"] == sym]
            if len(sym_df) > 0:
                pnl = sym_df["PnL"]
                w = pnl[pnl > 0]
                losses = pnl[pnl < 0]
                gp = float(w.sum()) if len(w) else 0
                gl = abs(float(losses.sum())) if len(losses) else 0.001
                indiv_pf[sym] = gp / gl
                indiv_trades_count[sym] = len(sym_df)

    log(
        f"\n  {'Ticker':<8} {'Indiv':>6} {'Indiv':>6} {'Port':>6} {'Port':>6} {'Port':>6} {'Port':>10} {'Delta':>6}"
    )
    log(f"  {'':8} {'Trds':>6} {'PF':>6} {'Trds':>6} {'WR%':>6} {'PF':>6} {'Net':>10} {'PF':>6}")
    log("  " + "-" * 65)

    for sym in sorted(SYMBOLS, key=lambda s: indiv_pf.get(s, 0), reverse=True):
        ipf = indiv_pf.get(sym, 0)
        it = indiv_trades_count.get(sym, 0)
        pd_ = port_ticker_dist.get(sym, {})
        ppf = pd_.get("pf", 0)
        pt = pd_.get("trades", 0)
        pwr = pd_.get("wr", 0)
        pnet = pd_.get("net", 0)
        delta = ppf - ipf if pt > 0 else float("nan")
        log(
            f"  {sym:<8} {it:>6} {ipf:>6.2f} {pt:>6} {pwr:>5.1f}% {ppf:>6.2f} ${pnet:>9,.0f} "
            f"{'N/A' if np.isnan(delta) else f'{delta:>+.2f}':>6}"
        )

    # Save concentration data
    conc_rows = []
    for sym in SYMBOLS:
        conc_rows.append(
            {
                "ticker": sym,
                "indiv_trades": indiv_trades_count.get(sym, 0),
                "indiv_pf": round(indiv_pf.get(sym, 0), 3),
                "port_trades": port_ticker_dist.get(sym, {}).get("trades", 0),
                "port_pf": round(port_ticker_dist.get(sym, {}).get("pf", 0), 3),
                "port_net": round(port_ticker_dist.get(sym, {}).get("net", 0), 2),
                "sector": SECTOR_MAP.get(sym, "other"),
            }
        )
    pd.DataFrame(conc_rows).to_csv(OUT_DIR / "ticker_concentration.csv", index=False)

    # ══════════════════════════════════════════════════════════════════════════
    # INVESTIGATION 3b: Score vs Individual PF correlation
    # ══════════════════════════════════════════════════════════════════════════
    log("\n  Signal score vs individual PF correlation:")
    accepted_scores_by_ticker: dict[str, list[float]] = defaultdict(list)
    rejected_scores_by_ticker: dict[str, list[float]] = defaultdict(list)
    for sig in signals_base:
        if sig.accepted:
            accepted_scores_by_ticker[sig.symbol].append(sig.score)
        else:
            rejected_scores_by_ticker[sig.symbol].append(sig.score)

    log(
        f"\n  {'Ticker':<8} {'Indiv PF':>9} {'Avg Score':>10} {'Accepted':>9} {'Rejected':>9} {'Accept%':>8}"
    )
    log("  " + "-" * 60)
    for sym in sorted(SYMBOLS, key=lambda s: indiv_pf.get(s, 0), reverse=True):
        a_scores = accepted_scores_by_ticker.get(sym, [])
        r_scores = rejected_scores_by_ticker.get(sym, [])
        all_scores = a_scores + r_scores
        avg_score = np.mean(all_scores) if all_scores else 0
        total = len(a_scores) + len(r_scores)
        accept_rate = len(a_scores) / total * 100 if total > 0 else 0
        log(
            f"  {sym:<8} {indiv_pf.get(sym, 0):>9.2f} {avg_score:>10.4f} {len(a_scores):>9} "
            f"{len(r_scores):>9} {accept_rate:>7.1f}%"
        )

    # ══════════════════════════════════════════════════════════════════════════
    # SAVE FULL REPORT
    # ══════════════════════════════════════════════════════════════════════════
    report_path = OUT_DIR / "investigation_report.txt"
    report_path.write_text("\n".join(output_lines))
    log(f"\nFull report saved to: {report_path}")


if __name__ == "__main__":
    main()
