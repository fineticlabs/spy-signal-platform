#!/usr/bin/env python
"""Per-ticker backtest using bar H/L stop/target checking.

Replicates the per-ticker backtest signal generation (same filters, ATR,
walk-forward windows) but replaces backtesting.py's framework with the
same bar H/L execution model used in run_portfolio_backtest.py.

This answers: "Is the per-ticker edge real, or is it an artifact of
backtesting.py's execution model?"

Usage:
    python scripts/run_backtest_barhl.py --cash 200000
"""

from __future__ import annotations

import argparse
import sys
import warnings
from dataclasses import dataclass, field
from datetime import date, time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import talib

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtest.data_loader import load_bars
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

# Strategy constants (matching engine.py exactly)
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
_MAX_TRADES_PER_DAY = 5
_POSITION_SCALE = 0.25
_WINDOW_START = time(9, 35)
_WINDOW_END = time(10, 0)
_FORCE_FLAT = time(15, 55)


@dataclass
class TickerResult:
    symbol: str
    trades: int = 0
    winners: int = 0
    gross_profit: float = 0
    gross_loss: float = 0
    pnl_list: list[float] = field(default_factory=list)
    target_exits: int = 0
    stop_exits: int = 0
    flat_exits: int = 0


def _run_ticker_barhl(
    df_1min: pd.DataFrame,
    symbol: str,
    cash: float,
    econ_blocked: np.ndarray | None = None,
    vts_arr: np.ndarray | None = None,
) -> TickerResult:
    """Run one ticker using bar H/L execution model."""
    result = TickerResult(symbol=symbol)

    df_et = df_1min.copy()
    df_et.index = df_et.index.tz_convert(_ET)

    # Pre-compute indicators on full series (matching engine.py:init)
    h = df_1min["high"].astype(float).values
    lo = df_1min["low"].astype(float).values
    c = df_1min["close"].astype(float).values
    v = df_1min["volume"].astype(float).values

    atr_raw = talib.ATR(h, lo, c, timeperiod=_ATR_PERIOD)
    atr_shifted = np.roll(atr_raw, 1)
    atr_shifted[0] = np.nan

    avg_vol_raw = pd.Series(v).rolling(_VOL_WINDOW).mean().values
    avg_vol_shifted = np.roll(avg_vol_raw, 1)
    avg_vol_shifted[0] = np.nan

    # Daily ADX + realized vol (from SPY-level, but for per-ticker we compute per-ticker)
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
        daily_returns = daily["close"].pct_change()
        daily_rv_arr = (daily_returns.rolling(20).std() * np.sqrt(252)).values

    # Build date lookups (shifted 1 day)
    adx_lookup: dict[date, float] = {}
    rv_lookup: dict[date, float] = {}
    dl = list(daily.index)
    for i in range(1, len(dl)):
        d = dl[i].date()
        if not np.isnan(daily_adx_arr[i - 1]):
            adx_lookup[d] = float(daily_adx_arr[i - 1])
        if not np.isnan(daily_rv_arr[i - 1]):
            rv_lookup[d] = float(daily_rv_arr[i - 1])

    # Compute 15-min EMA(20)
    bars_15m = df_et.resample("15min").agg({"close": "last"}).dropna()
    ema15m_lookup: dict[date, float] = {}
    if len(bars_15m) >= 21:
        from talipp.indicators import EMA as _EMA

        ema = _EMA(period=20)
        for ts, row in bars_15m.iterrows():
            ema.add(float(row["close"]))
            if ema.output_values and ema.output_values[-1] is not None:
                ema15m_lookup[ts.date()] = float(ema.output_values[-1])

    # Gap classification
    gap_lookup: dict[date, float] = {}
    prev_close_val = None
    for d, grp in df_et.groupby(df_et.index.date):
        first_bar = grp.between_time("09:30", "09:30")
        if not first_bar.empty and prev_close_val is not None:
            so = float(first_bar.iloc[0]["open"])
            gap_lookup[d] = (so - prev_close_val) / prev_close_val * 100
        last_bar = grp.between_time("15:55", "15:59")
        if not last_bar.empty:
            prev_close_val = float(last_bar.iloc[-1]["close"])
        elif len(grp) > 0:
            prev_close_val = float(grp.iloc[-1]["close"])

    # Econ blocked dates
    econ_dates: set[date] = set()
    if econ_blocked is not None:
        for i, ts in enumerate(df_et.index):
            if econ_blocked[i] > 0.5:
                econ_dates.add(ts.date())

    # VTS lookup
    vts_lookup: dict[date, float] = {}
    if vts_arr is not None:
        for i, ts in enumerate(df_et.index):
            if not np.isnan(vts_arr[i]):
                vts_lookup[ts.date()] = float(vts_arr[i])

    # Pre-compute ORB per day
    orb_cache: dict[date, tuple[float, float]] = {}
    for d, grp in df_et.groupby(df_et.index.date):
        orb_bars = grp.between_time("09:30", "09:34")
        if len(orb_bars) >= _ORB_BARS:
            orb_cache[d] = (float(orb_bars["high"].max()), float(orb_bars["low"].min()))

    # Pre-index bars by date
    day_bars: dict[date, pd.DataFrame] = {}
    for d, grp in df_et.groupby(df_et.index.date):
        day_bars[d] = grp

    # Simulation
    equity = cash
    trading_days = sorted(day_bars.keys())

    for day in trading_days:
        if day.weekday() == 0:
            continue
        d_adx = adx_lookup.get(day)
        d_rv = rv_lookup.get(day)
        if d_adx is not None and d_adx <= _DAILY_ADX_MIN:
            continue
        if d_rv is not None and d_rv >= _REALIZED_VOL_MAX:
            continue
        if day in econ_dates:
            continue
        vts = vts_lookup.get(day)
        if vts is not None and vts > BACKWARDATION_THRESHOLD:
            continue

        orb = orb_cache.get(day)
        if orb is None:
            continue
        orb_high, orb_low = orb
        if orb_low <= 0:
            continue
        orb_range_pct = (orb_high - orb_low) / orb_low
        if orb_range_pct < _MIN_ORB_PCT:
            continue

        daily_trades = 0
        grp = day_bars[day]

        # Get bars in signal window
        signal_bars = grp.between_time("09:35", "09:59")
        if len(signal_bars) < 2:
            continue

        # Get all bars for SL/TP checking
        all_market_bars = grp.between_time("09:30", "15:54")

        # Get ATR and avg_vol at the signal bars using the original index positions
        for bar_idx in range(1, len(signal_bars)):
            if daily_trades >= _MAX_TRADES_PER_DAY:
                break

            curr_bar = signal_bars.iloc[bar_idx]
            prev_bar = signal_bars.iloc[bar_idx - 1]
            curr_time = curr_bar.name

            close_val = float(curr_bar["close"])
            prev_close = float(prev_bar["close"])
            volume = float(curr_bar["volume"])

            # Find the position in the original DataFrame for ATR/vol lookup
            orig_idx = df_et.index.get_loc(curr_time)
            if isinstance(orig_idx, slice):
                orig_idx = orig_idx.start
            atr_val = atr_shifted[orig_idx] if orig_idx < len(atr_shifted) else np.nan
            avg_vol_val = avg_vol_shifted[orig_idx] if orig_idx < len(avg_vol_shifted) else np.nan

            if np.isnan(atr_val) or atr_val <= 0:
                continue
            if np.isnan(avg_vol_val) or avg_vol_val <= 0:
                continue
            if volume < avg_vol_val * _VOL_MULTIPLIER:
                continue

            # EMA trend filter
            ema_val = ema15m_lookup.get(day)
            if ema_val is not None:
                if close_val > orb_high and close_val <= ema_val:
                    continue
                if close_val < orb_low and close_val >= ema_val:
                    continue

            # Gap filter
            gap = gap_lookup.get(day)
            if gap is not None:
                if close_val > orb_high and gap < -_GAP_THRESHOLD_PCT:
                    continue
                if close_val < orb_low and gap > _GAP_THRESHOLD_PCT:
                    continue

            # 2-bar confirmation
            base_risk = _ATR_MULTIPLIER * atr_val
            if close_val > orb_high and prev_close > orb_high:
                direction = "LONG"
                entry = close_val
                stop = entry - base_risk
                target = entry + _RISK_MULTIPLIER * base_risk
            elif close_val < orb_low and prev_close < orb_low:
                direction = "SHORT"
                entry = close_val
                stop = entry + base_risk
                target = entry - _RISK_MULTIPLIER * base_risk
            else:
                continue

            # Bar H/L execution: check all bars after entry
            after_entry = all_market_bars[all_market_bars.index > curr_time]
            exit_price = None
            exit_reason = "flat"

            for _, check_bar in after_entry.iterrows():
                bh = float(check_bar["high"])
                bl = float(check_bar["low"])
                bo = float(check_bar["open"])
                bc = float(check_bar["close"])

                hit_tgt = (bh >= target) if direction == "LONG" else (bl <= target)
                hit_stp = (bl <= stop) if direction == "LONG" else (bh >= stop)

                if hit_tgt and hit_stp:
                    favor = (bc > bo) if direction == "LONG" else (bc < bo)
                    if favor:
                        hit_stp = False
                    else:
                        hit_tgt = False

                if hit_tgt:
                    exit_price = target
                    exit_reason = "target"
                    break
                elif hit_stp:
                    exit_price = stop
                    exit_reason = "stop"
                    break

            # Force flat
            if exit_price is None:
                flat_bars = grp.between_time("15:55", "15:59")
                if not flat_bars.empty:
                    exit_price = float(flat_bars.iloc[0]["close"])
                else:
                    exit_price = float(grp.iloc[-1]["close"])
                exit_reason = "flat"

            # PnL
            pos_size = equity * _POSITION_SCALE
            if direction == "LONG":
                pnl = (exit_price - entry) * (pos_size / entry)
            else:
                pnl = (entry - exit_price) * (pos_size / entry)

            result.trades += 1
            result.pnl_list.append(pnl)
            if pnl > 0:
                result.winners += 1
                result.gross_profit += pnl
            else:
                result.gross_loss += abs(pnl)

            if exit_reason == "target":
                result.target_exits += 1
            elif exit_reason == "stop":
                result.stop_exits += 1
            else:
                result.flat_exits += 1

            equity += pnl
            daily_trades += 1

    return result


def main() -> None:
    p = argparse.ArgumentParser(description="Per-ticker backtest with bar H/L execution")
    p.add_argument("--cash", type=float, default=200_000)
    p.add_argument(
        "--symbols",
        default="SPY,QQQ,MSFT,AMD,TSLA,AMZN,UBER,SMCI,SHOP,PLTR,NFLX,MSTR,SNOW,ARM,DASH,PYPL,INTC,MU,HOOD,DKNG,SOXL,ROKU,TQQQ,BA,MRVL,META",
    )
    args = p.parse_args()
    symbols = [s.strip() for s in args.symbols.split(",")]

    db = BarDatabase()
    db.connect()

    # Load VTS
    vts_df = get_vix_term_structure()

    print(f"Per-ticker backtest (bar H/L mode), ${args.cash:,.0f}")
    print("=" * 85)

    all_results: list[TickerResult] = []
    for sym in symbols:
        df = load_bars(db, symbol=sym, timeframe=TimeFrame.ONE_MIN)
        if df.empty:
            continue

        econ = compute_econ_blocked_array(df.index)
        vts_arr = compute_vix_term_structure_array(df.index, vts_df) if not vts_df.empty else None

        result = _run_ticker_barhl(df, sym, args.cash, econ_blocked=econ, vts_arr=vts_arr)
        all_results.append(result)

        wr = result.winners / result.trades * 100 if result.trades > 0 else 0
        pf = result.gross_profit / result.gross_loss if result.gross_loss > 0 else 0
        net = result.gross_profit - result.gross_loss
        print(
            f"  {sym:<6} {result.trades:>4} trades  {wr:>5.1f}% WR  PF={pf:>5.3f}  ${net:>+9,.0f}  "
            f"(tgt={result.target_exits} stp={result.stop_exits} flat={result.flat_exits})"
        )

    db.close()

    # Aggregate
    total_trades = sum(r.trades for r in all_results)
    total_winners = sum(r.winners for r in all_results)
    total_gp = sum(r.gross_profit for r in all_results)
    total_gl = sum(r.gross_loss for r in all_results)
    total_tgt = sum(r.target_exits for r in all_results)
    total_stp = sum(r.stop_exits for r in all_results)
    total_flat = sum(r.flat_exits for r in all_results)
    all_pnl = []
    for r in all_results:
        all_pnl.extend(r.pnl_list)

    wr = total_winners / total_trades * 100 if total_trades > 0 else 0
    pf = total_gp / total_gl if total_gl > 0 else 0
    net = total_gp - total_gl
    pnl_s = pd.Series(all_pnl)
    sharpe = (
        float(pnl_s.mean() / pnl_s.std() * np.sqrt(252))
        if len(pnl_s) > 1 and pnl_s.std() > 0
        else 0
    )

    print("=" * 85)
    print(f"  AGGREGATE: {total_trades} trades, {wr:.1f}% WR, PF={pf:.3f}, Sharpe={sharpe:.2f}")
    print(
        f"  Net PnL: ${net:+,.0f}, Avg: ${net/total_trades:+,.0f}/trade" if total_trades > 0 else ""
    )
    print(
        f"  Exits: target={total_tgt} ({total_tgt/total_trades*100:.0f}%), "
        f"stop={total_stp} ({total_stp/total_trades*100:.0f}%), "
        f"flat={total_flat} ({total_flat/total_trades*100:.0f}%)"
        if total_trades > 0
        else ""
    )
    print(
        f"  Avg winner: ${total_gp/total_winners:,.0f}, "
        f"Avg loser: ${-total_gl/(total_trades-total_winners):,.0f}, "
        f"W/L ratio: {(total_gp/total_winners)/(total_gl/(total_trades-total_winners)):.2f}"
        if total_winners > 0 and total_trades > total_winners
        else ""
    )
    print("=" * 85)


if __name__ == "__main__":
    main()
