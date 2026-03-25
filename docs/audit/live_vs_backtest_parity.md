# Live vs Backtest Parity Audit

**Date**: 2026-03-24
**Auditor**: Claude Code
**Scope**: Signal generation, SL/TP computation, filter parity, position monitoring

## 1. Position Monitoring — No Phase Ordering Bug in Live

The live system does **NOT** have the portfolio backtest's phase ordering bug.

**Why:** The live system uses **Alpaca server-side bracket orders**, not client-side SL/TP checking.

When a signal fires:
1. `orb.py:evaluate()` computes entry, stop, target prices (line 385-401)
2. `main.py:_process_bar()` passes signal to risk manager (line 547)
3. If approved, `alpaca_executor.py:submit_bracket_order()` submits an `OrderClass.BRACKET` order (line 148)
4. Alpaca's server monitors the SL/TP legs continuously, independent of our scanner

The bracket order contains:
```python
order_class=OrderClass.BRACKET,
take_profit=TakeProfitRequest(limit_price=rounded_target),
stop_loss=StopLossRequest(stop_price=rounded_stop),
```

**Conclusion**: Alpaca handles SL/TP server-side on every tick. Our scanner doesn't need to poll. **No phase ordering issue exists in live.**

## 2. SL/TP Calculation Comparison

### Backtest (engine.py:976-998)
```python
kalman_m = float(self.kalman_mult[-1])           # from Kalman filter
base_atr_risk = self.atr_mult * atr              # 1.5 * ATR(14)
adaptive_atr_stop = base_atr_risk * kalman_m     # Kalman-scaled stop
# LONG:
stop  = entry - adaptive_atr_stop                # Kalman-widened
target = entry + dynamic_risk_mult * base_atr_risk  # Fixed 2.0R on base ATR
```
- ATR: TA-Lib ATR(14) on full 1-min series, shifted 1 bar
- Stop multiplier: 1.5 × ATR × Kalman (0.90-1.50)
- Target multiplier: 2.0 × 1.5 × ATR (fixed, not Kalman-scaled)

### Live (orb.py:386-401)
```python
km = kalman_stop_mult if kalman_stop_mult is not None else Decimal("1.0")
adaptive_atr_stop = _ATR_MULTIPLIER * atr * km   # 1.5 * ATR * kalman
base_atr_risk = _ATR_MULTIPLIER * atr             # 1.5 * ATR
# LONG:
stop  = entry - adaptive_atr_stop                 # Kalman-widened
target = entry + _RISK_MULTIPLIER * base_atr_risk  # 2.0 * 1.5 * ATR
```
- ATR: talipp StreamingATR(14) on 1-min bars
- Stop multiplier: 1.5 × ATR × Kalman (but **kalman_stop_mult is always None** — see Finding #1)
- Target multiplier: 2.0 × 1.5 × ATR

### Portfolio Sim (run_portfolio_backtest.py:317-325)
```python
atr = atr_cache.get(sym, {}).get(day, orb_high - orb_low)
stop  = entry - _ATR_MULTIPLIER * atr             # 1.5 * ATR, NO Kalman
target = entry + _RISK_MULTIPLIER * _ATR_MULTIPLIER * atr  # 2.0 * 1.5 * ATR
```
- ATR: TA-Lib ATR(14) on full 1-min series, value at 9:34 ET
- Stop multiplier: 1.5 × ATR (no Kalman)
- Target multiplier: 3.0 × ATR

## 3. Critical Finding #1: Kalman Stop Multiplier Never Passed to Live Strategy

**File**: `src/main.py:530`
```python
signal = pipeline.strategy.evaluate(bar, indicator_snapshot, level_snapshot, pipeline.regime)
```

The `evaluate()` method accepts `kalman_stop_mult` as an optional parameter, but `_process_bar()` **never passes it**. The default is `None`, so `km = Decimal("1.0")` always.

**Impact**: Live stops are fixed at 1.5 × ATR. Backtest stops are 1.5 × ATR × Kalman (0.90-1.50). In volatile conditions, backtest widens stops to avoid noise stop-outs; live does not.

**Severity**: HIGH — stop distances differ by up to ±50% in volatile conditions.

## 4. Critical Finding #2: VIX Term Ratio Never Passed to Live Strategy

**File**: `src/main.py:530` — same call. `vix_term_ratio` defaults to `None`.

The VIX backwardation check in `orb.py:262-271` checks `if vix_term_ratio is not None`, but since it's never passed, the check is **never executed** in live.

**However**: VIX backwardation data is not available in the live scanner (no real-time VIX3M feed). The check correctly defaults to "no data = allow". This is a known limitation, not a bug.

**Severity**: MEDIUM — documented known limitation.

## 5. Filter Parity Table

| Filter | backtesting.py | run_portfolio_backtest.py | Live Scanner | Notes |
|--------|:-:|:-:|:-:|-------|
| Monday exclusion | YES | YES | YES | All aligned |
| Trading window 9:35-10:00 | YES | YES | YES | All aligned |
| Economic calendar | YES | NO | YES | Portfolio missing |
| VIX backwardation | YES | NO | YES* | *Live: never triggered (no VIX3M data) |
| HMM regime (tag only) | YES (tag) | NO | NO | Not available in live |
| Earnings (tag only) | YES (tag) | NO | YES (tag) | Portfolio missing |
| Max trades/day | YES (5) | YES (5) | YES (5) | All aligned |
| ORB range minimum | YES (0.15%) | YES (0.15%) | YES (0.15%) | All aligned |
| Realized vol < 18% | YES | YES | YES | All aligned |
| Daily ADX > 25 | YES (daily) | YES (daily) | YES (daily) | All aligned via daily_adx |
| Volume 1.5x | YES (adaptive) | YES (adaptive) | YES (adaptive) | All aligned |
| 15-min EMA trend | YES | NO | YES | Portfolio missing |
| Gap classification | YES (±0.3%) | NO | YES (±0.3%) | Portfolio missing |
| 2-bar confirmation | YES | YES | YES | All aligned |
| VIX < 25 | NO (uses HV) | NO | YES | Live has VIX gate, backtest uses HV |
| Low volume 0.5x reject | NO | NO | YES | Live-only hard gate |
| Kalman adaptive stops | YES | NO | NO* | *Live has the code but kalman=None |
| Volume profile (tag) | YES (tag) | NO | YES (tag) | All informational |
| Sector limits | NO | YES | YES | Portfolio + live only |
| Max concurrent positions | NO | YES (3) | YES (3) | Portfolio + live only |

## 6. Parity Status (Updated 2026-03-24)

### RESOLVED
1. **Kalman stop mult** — StreamingKalman tracker added to LevelManager. Wired into LevelSnapshot → evaluate() → bracket order. Live now uses adaptive 0.90-1.50x multiplier matching backtest.
2. **VIX term ratio** — Fetched from yfinance (^VIX / ^VIX3M) at startup + daily reset. Wired through `_vix_term_ratio_cache` → evaluate(). Backwardation filter now fires in live.
3. **Portfolio sim missing filters** — Added: 15-min EMA trend, gap ±0.3%, VIX backwardation, economic calendar. Portfolio sim now applies same filter stack as per-ticker backtest.
4. **HMM regime** — Confirmed tag-only in backtest (lines 837-843). No blocking, no impact. Documented and skipped.

### REMAINING KNOWN GAPS (low impact)
5. **Live has VIX < 25 gate, backtest uses realized vol** — Different mechanisms for the same goal. Both effective.
6. **Live has low-volume 0.5x hard reject** — Extra safety gate in live not in backtest. Slightly more conservative.
7. **Portfolio sim SL/TP resolution** — Checks bar H/L only; backtesting.py may simulate intra-bar tick resolution. This is why portfolio PF is lower than per-ticker PF.
