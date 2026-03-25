# Filter Comparison: Portfolio Backtest vs Per-Ticker Backtest

## Root Cause: 6 trades vs 2,094 trades

The portfolio backtest produces dramatically fewer trades due to **5 critical implementation differences**, not just position limits.

### Critical Bug #1: Positions Only Checked During ORB Window

**Per-ticker (engine.py):** backtesting.py framework checks SL/TP on **every tick** (bar) from entry until exit.

**Portfolio (run_portfolio_backtest.py lines 222-296):** SL/TP is ONLY checked during the `for minute_offset in range(5, 30)` loop (9:35-9:59 ET). After 10:00, positions are never checked until 15:55 force-flat.

**Impact:** Most ORB positions resolve within 30-120 minutes. Portfolio never sees these exits.

### Critical Bug #2: Common Date Range Truncation

ARM starts 2023-09-14, so the entire portfolio uses that as the start. Loses 3+ years of data.

### Critical Bug #3: Missing Filters (EMA, gap, VIX backwardation, econ calendar)

### Critical Bug #4: Volume Filter (fixed 100K vs adaptive 1.5x)

### Critical Bug #5: ATR Proxy (ORB range vs real ATR(14))

## Conclusion

The 6 trades and -$2,790 is not a reflection of portfolio performance. It reflects 5 implementation bugs.
