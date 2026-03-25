# Portfolio Variant Analysis

## Diagnosis: Why Original Portfolio Produced Only 6 Trades

The original `run_portfolio_backtest.py` had **5 critical bugs** that have been fixed:

### Bug 1: SL/TP Only Checked During 9:35-9:59 (FIXED)
Positions were only checked for stop-loss/take-profit hits during the ORB signal generation window. After 10:00 AM, they were never checked until 15:55 force-flat. This meant all positions became "hold until close" trades, destroying the 2R target structure.

**Fix:** SL/TP is now checked on **every bar** from entry until 15:55 ET (Phase 1 of the day loop).

### Bug 2: Common Date Range Truncation (FIXED)
Used the latest start date across ALL tickers (ARM: 2023-09-14) as the common start, losing 3+ years of trading history.

**Fix:** Each ticker now uses its OWN available date range. Trading days are the union of all symbols' dates.

### Bug 3: Missing Filters (DOCUMENTED)
EMA trend, gap classification, VIX backwardation, and economic calendar filters are not applied in the portfolio simulator. These filters exist in the per-ticker backtest (engine.py).

**Status:** Documented as known limitation. The portfolio backtest is simpler by design — it approximates the per-ticker strategy under capital constraints. Full filter parity would require porting the entire engine.py filter stack.

### Bug 4: Volume Filter (FIXED)
Changed from fixed 100K floor to adaptive `1.5x * 20-bar rolling average` per symbol.

### Bug 5: ATR Proxy (FIXED)
Now computes real TA-Lib ATR(14) per symbol from daily bars, with shift-1 lookahead prevention. Falls back to ORB range only when ATR is unavailable.

## Expected Improvement
With all 5 bugs fixed, the portfolio backtest should produce:
- **Hundreds of trades** (not 6)
- **Proper SL/TP exits** (target hits + stop hits, not just force-flat)
- **Positive PF** (if the underlying strategy is profitable)
- **Realistic position constraint impact** (what matters for $200K deployment)
