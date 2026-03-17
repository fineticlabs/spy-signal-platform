---
description: Trading engine, backtest, and strategy rules — loaded for backtest/strategy/levels code
globs:
  - src/backtest/**
  - src/strategies/**
  - src/levels/**
---

# Trading Engine Rules

## Walk-Forward Backtest
- 60-day in-sample (IS) window, 20-day out-of-sample (OOS) window. These are intentional — do not change without backtest re-validation.
- HMM trains on IS window, predicts regime on OOS window. Falls back to NORMAL (1.0) on training failure.
- Kalman filter computes adaptive stop multiplier per bar. Clamped to [0.90, 1.50].
- All 11 filters applied in order: ATR, EMA, ORB range, gap, realized vol, daily ADX, RVOL, economic calendar, earnings blackout, volume profile, VIX term structure.
- Backtest results must be reproducible — use fixed random seeds (`np.random.default_rng(42)`).

## Position Sizing
- 1% account risk per trade ($500 on $50K account).
- `position_size = risk_dollars / (entry_price - stop_price)`, floored to whole shares.
- SOXL/TQQQ: 3x leveraged — effective risk is 3x, so reduce position size by 1/3.

## ORB Strategy
- Opening range = first 5 completed 1-min bars (9:30-9:35 ET).
- Entry: price breaks ORB high/low with volume > 1.5x 20-bar average.
- Stop: 1.5x ATR(14) from entry, adjusted by Kalman multiplier.
- Target: 2R (2x risk distance).
- No trades during lunch chop (11:30-13:30 ET) unless confluence > 4/5.
- No new trades after 15:45 ET. All positions flat by 15:55 ET.
- Monday filter ON — no trades on Mondays.

## No-Lookahead Rules
- Never use bar close price before bar is complete.
- Daily indicators (ADX, VIX term structure) use D-1 values (shifted by 1 day).
- ATR values shifted by 1 bar before use in signals.
- ORB high/low only available after bar 5 completes.
