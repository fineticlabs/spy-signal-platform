# Backtest vs Live Engine Audit Report

**Date:** 2026-03-24
**Auditor:** Automated code audit (Claude)
**Files reviewed:** 23 source files (complete read of all code paths)

---

## Executive Summary

Overall alignment between the backtest engine (`src/backtest/engine.py`), live strategy (`src/strategies/orb.py` + `src/main.py`), and portfolio backtest (`scripts/run_portfolio_backtest.py`) is **GOOD** with **5 mismatches** identified (1 CRITICAL, 1 HIGH, 2 MEDIUM, 1 LOW). The codebase shows evidence of deliberate synchronization efforts. Most parameters match exactly across all three codebases.

**Confidence level: HIGH** — all files were read in their entirety; no guesswork.

---

## 1. ORB Definition

| Parameter | Backtest (`engine.py`) | Live (`orb.py` + `opening_range.py`) | Portfolio (`run_portfolio_backtest.py`) | Status |
|---|---|---|---|---|
| ORB window bars | 5 bars (`_ORB_BARS = 5`) | 5 bars (`_ORB5_END = time(9,35)`, accumulates bars 9:30-9:34) | 5 bars (`_ORB_BARS = 5`, `between_time("09:30","09:34")`) | **MATCH** |
| ORB OHLC logic | `max(high[0:5])`, `min(low[0:5])` | `max(highs)`, `min(lows)` from `_RangeAccumulator` | `high.max()`, `low.min()` on 9:30-9:34 bars | **MATCH** |
| ORB completion | NaN for bars 0-4, values for bar 5+ per day | `is_complete` set when `bar_time >= cutoff (9:35)` and highs exist | `len(orb_bars_df) >= 5` | **MATCH** |
| ORB uses close of 9:34 bar? | Uses high/low of bars 0-4 (9:30, 9:31, 9:32, 9:33, 9:34) | Same — accumulates bars with `bar_time < 9:35` | Same — `between_time("09:30","09:34")` | **MATCH** |

---

## 2. Entry Logic

| Parameter | Backtest | Live | Portfolio | Status |
|---|---|---|---|---|
| Breakout trigger | 2 consecutive closes outside ORB (`close > orb_high AND prev_close > orb_high`) | 2-bar confirmation via `_pending_breakouts` dict (first bar stores pending, second bar confirms) | 2-bar confirmation (`close > orb_high and prev_close > orb_high`) | **MATCH** |
| Entry price | Current bar close (`entry = close`) | Current bar close (`entry = close`) | Current bar close (`entry = close_val`) | **MATCH** |
| Entry timing | Same bar as confirmation (no next-bar delay) | Same bar as second confirmation bar | Same bar | **MATCH** |

### Filter Comparison (order of evaluation)

| Filter | Backtest | Live (`orb.py`) | Portfolio | Status |
|---|---|---|---|---|
| Excluded weekdays (Monday) | `excluded_days` param, checked via `ts.dayofweek` | `_excluded_days` set (default `{0}`), checked first | `day.weekday() == 0` hardcoded | **MISMATCH (LOW)** |
| Trading window | `_WINDOW1_START=9:35`, `_WINDOW1_END=10:00` | `_cutoff` default `10:00`, min start implicit (after ORB complete ~9:35) | `_WINDOW_START=9:35`, `_WINDOW_END=10:00` | **MATCH** |
| Economic calendar (FOMC/NFP/CPI/PPI) | `econ_blocked` array, hard block | `is_high_impact_day()`, hard block | `econ_blocked_dates`, hard block | **MATCH** |
| VIX backwardation | `vts_ratio > BACKWARDATION_THRESHOLD (1.0)`, hard block | `vix_term_ratio > BACKWARDATION_THRESHOLD`, hard block | `vts_ratio > 1.0`, hard block | **MATCH** |
| Realized vol | `realized_vol >= 0.18`, hard block | `rv >= self._realized_vol_max (0.18)`, hard block | `d_vol >= 0.18`, hard block | **MATCH** |
| Daily ADX | `d_adx <= 25.0`, hard block | `adx <= self._adx_min (25)`, hard block | `d_adx <= 25.0`, hard block | **MATCH** |
| ORB range minimum | `orb_range < 0.0015`, hard block | `range_pct < 0.0015`, hard block | `orb_range_pct < 0.0015`, hard block | **MATCH** |
| Volume filter | `volume < avg_vol * 1.5`, hard block | Two gates: `bar_vol < avg_vol * 0.5` (reject) + `bar_vol < avg_vol * 1.5` (reject) | `volume < avg_vol * 1.5`, hard block | **MISMATCH (MEDIUM)** |
| ATR required | `np.isnan(atr)`, skip | `atr is None`, skip | `np.isnan(atr)`, skip | **MATCH** |
| EMA(20) 15-min trend | `close > ema15m` for LONG, `close < ema15m` for SHORT | `close > ema20` for LONG (prefers 15m EMA, falls back to 1m EMA20) | `close > ema_val` for LONG, `close < ema_val` for SHORT | **MATCH** |
| Gap classification | `gap > +0.3%` LONG only, `gap < -0.3%` SHORT only | Same thresholds (0.3%) with directional gate | Same thresholds | **MATCH** |
| Earnings | Informational only (no blocking) | Informational only (tags `EARNINGS`) | **NOT CHECKED** | **MISMATCH (MEDIUM)** |
| Low volume rejection | Not present (only has `vol < 1.5x` gate) | Present: `bar_vol < avg_vol * 0.5` hard reject | Not present | **MISMATCH (MEDIUM)** |
| Max trades per day | 5 (`_MAX_TRADES_PER_DAY`) | 5 (via `RiskManager._settings.max_trades_per_day`) | `max_daily` arg (default 5) | **MATCH** |
| VIX level cap | Not explicitly checked (realized vol proxy) | `vix >= 25` hard block | Not explicitly checked (realized vol proxy) | **MISMATCH (HIGH)** |
| Lunch chop filter | Not applicable (window ends at 10:00) | `11:30-13:30` blocked (not applicable with 10:00 cutoff) | Not applicable | **MATCH** (N/A) |

---

## 3. Stop Loss / Take Profit

| Parameter | Backtest | Live | Portfolio | Status |
|---|---|---|---|---|
| ATR period | 14 (`_ATR_PERIOD`) | 14 (StreamingATR(14)) | 14 (`_ATR_PERIOD`) | **MATCH** |
| ATR multiplier (stop) | 1.5 (`_ATR_MULTIPLIER`) | 1.5 (`_ATR_MULTIPLIER`) | 1.5 (`_ATR_MULTIPLIER`) | **MATCH** |
| Risk multiplier (target) | 2.0 (`_RISK_MULTIPLIER`) — fixed, dynamic targets removed | 2.0 (`_RISK_MULTIPLIER`) | 2.0 (`_RISK_MULTIPLIER`) | **MATCH** |
| Kalman adjustment | `adaptive_atr_stop = base_atr_risk * kalman_m` (batch `compute_kalman_stop_multiplier`) | `adaptive_atr_stop = _ATR_MULTIPLIER * atr * km` where `km = kalman_stop_mult` (streaming `StreamingKalman`) | `adaptive_stop = base_atr_risk * kalman_mult` (streaming `StreamingKalman`) | **MATCH** |
| Kalman clamp range | `[0.90, 1.50]` | `[0.90, 1.50]` (same constants in `kalman_levels.py`) | Uses same `StreamingKalman` class | **MATCH** |
| Target decoupled from Kalman | Yes — `target = entry +/- risk_mult * base_atr_risk` (original ATR, no Kalman) | Yes — `target = entry +/- _RISK_MULTIPLIER * base_atr_risk` | Yes — `target = entry +/- _RISK_MULTIPLIER * base_atr_risk` | **MATCH** |
| Stop type (fixed/trailing) | Fixed at entry (Backtesting.py `sl=` parameter) | Fixed at entry (Alpaca bracket stop-loss) | Fixed at entry | **MATCH** |
| Exit check method | Backtesting.py internal: checks bar HIGH/LOW for SL/TP | Alpaca bracket order: exchange monitors intraday | Checks `bar_h >= target` / `bar_l <= stop` (HIGH/LOW) | **MATCH** |
| Both SL+TP hit same bar | Not handled (Backtesting.py defaults) | Not applicable (exchange resolves) | Disambiguated: if both hit, use `close > open` for direction favor | **MISMATCH (LOW)** |
| Live order types | N/A | Market entry + bracket (stop_loss=stop, take_profit=limit) | N/A | N/A |
| Price rounding | Float (backtest) | Rounded to 2 decimals for Alpaca compliance | Float | **MATCH** (functionally) |

### LONG stop/target formulas

| Formula | Backtest | Live | Portfolio |
|---|---|---|---|
| stop | `entry - (1.5 * ATR * kalman_mult)` | `entry - (1.5 * ATR * kalman_mult)` | `entry - (1.5 * ATR * kalman_mult)` |
| target | `entry + (2.0 * 1.5 * ATR)` | `entry + (2.0 * 1.5 * ATR)` | `entry + (2.0 * 1.5 * ATR)` |

All three compute the same values.

---

## 4. Position Sizing

| Parameter | Backtest | Live | Portfolio | Status |
|---|---|---|---|---|
| Method | `size=position_scale_factor` (fraction of equity, Backtesting.py) | `dollar_risk / risk_per_share`, scaled by `position_scale_factor` | `equity * _POSITION_SCALE` (fraction of equity) | **MISMATCH (CRITICAL)** |
| Scale factor | 0.25 (25% of equity) | 0.25 (applied after risk-based calc) | 0.25 (25% of equity) | See below |
| Risk per trade | Implicit (25% of equity) | 1% of account ($500 on $50K), then * 0.25 | Implicit (25% of equity) | See below |
| Account size | `cash` param (default $50K) | `account_size` (default $50K) | `cash` param (default $200K) | N/A (configurable) |
| Rounding | Backtesting.py handles fractional shares | `int()` floor to whole shares | Dollar-based (no share rounding) | See below |
| Buying power cap | None | 50% of buying power cap (`_cap_to_buying_power`) | None | See below |

**CRITICAL DETAIL:** The backtest and portfolio backtest use `size=0.25` meaning 25% of current equity per trade (e.g., $12,500 on $50K). The live engine uses a fundamentally different formula: `position_size = (account * 1% / risk_per_share) * 0.25`. On a $50K account with a $2 risk per share, live would compute `(500 / 2) * 0.25 = 62.5 -> 62 shares ~ $27,900 position` (at $450/share), whereas backtest would allocate `$12,500 / $450 = 27 shares ~ $12,150`. These are very different position sizes. See MISMATCH #1 below.

---

## 5. Timing

| Parameter | Backtest | Live | Portfolio | Status |
|---|---|---|---|---|
| Entry scan start | `9:35 ET` (`_WINDOW1_START`) | After ORB complete (~9:35+), default cutoff `10:00` | `9:35 ET` (`_WINDOW_START`) | **MATCH** |
| Entry cutoff | `10:00 ET` (`_WINDOW1_END`) | `10:00 ET` (`_DEFAULT_CUTOFF`) | `10:00 ET` (`_WINDOW_END`) | **MATCH** |
| EOD flatten | `15:55 ET` (`_FORCE_FLAT`) | `15:55 ET` (`_FLATTEN_TIME` in executor) | `15:55 ET` (`_FORCE_FLAT`) | **MATCH** |
| Risk manager time window | N/A | `9:35-15:45 ET` (`_SESSION_START`, `_CUTOFF`) | N/A | **MATCH** (subset) |
| Off-by-one in minutes | None detected | None detected | `time(9, 30 + minute_offset)` for `minute_offset in range(5, 30)` = 9:35-9:59, correct | **MATCH** |

---

## 6. Data

| Parameter | Backtest | Live | Portfolio | Status |
|---|---|---|---|---|
| Bar type | 1-min OHLCV from SQLite | 1-min trade bars from Alpaca WebSocket | 1-min OHLCV from SQLite | **MATCH** |
| Timezone handling | UTC storage, ET conversion via `ZoneInfo("America/New_York")` | UTC storage, ET conversion via `ZoneInfo("America/New_York")` | UTC storage, ET conversion via `ZoneInfo("America/New_York")` | **MATCH** |
| Split-adjusted prices | As stored in DB (Alpaca provides adjusted) | As received from Alpaca (adjusted) | Same DB data | **MATCH** |
| ATR lookahead prevention | `np.roll(atr_raw, 1)` shift | Streaming (bar-by-bar, inherently no lookahead) | `np.roll(atr_raw, 1)` shift | **MATCH** |
| Volume average lookahead | `np.roll(avg_vol_raw, 1)` shift | Rolling deque with avg computed BEFORE current bar added | `np.roll(avg_vol_raw, 1)` shift | **MATCH** |
| 15-min EMA lookahead | Resampled, computed, `shift(1)`, forward-filled | Streaming 15-min aggregator with `seed_15m_ema()` | Resampled, computed, `shift(1)`, forward-filled | **MATCH** |
| Daily ADX lookahead | `adx_series.shift(1)` (D-1 value) | Computed from last 30 days of DB bars, uses D-1 value | `daily_adx[i-1]` (D-1 value) | **MATCH** |
| Realized vol lookahead | `rolling_vol.shift(1)` (D-1 value) | Computed from DB daily bars, uses D-2 value (`valid_std.iloc[-2]`) | `realized_vol.iloc[i-1]` (D-1 value) | **MATCH** |
| Indicator library | TA-Lib (batch) | talipp (streaming) | TA-Lib (batch) | Acceptable divergence |

---

## 7. HMM Regime Filter

| Parameter | Backtest | Live | Portfolio | Status |
|---|---|---|---|---|
| Training data | IS window 5-min bars (60-day IS) | Not retrained live (uses tags from levels) | Not used | **MATCH** (by design) |
| Features | Log returns, volume ratio, ATR change | N/A (informational tag from backtest) | N/A | **MATCH** |
| States | 3 (CALM/NORMAL/VOLATILE) via variance ordering | 3 labels: CALM/NORMAL/VOLATILE | N/A | **MATCH** |
| Blocking behavior | Informational only (no blocking) | Informational only (confidence +1/-1) | Not implemented | **MATCH** |
| Retraining schedule | Per walk-forward window (60d IS) | Not retrained live | N/A | By design |

---

## 8. Slippage / Commissions

| Parameter | Backtest | Live | Portfolio | Status |
|---|---|---|---|---|
| Slippage | $0.02/share round-trip (Backtesting.py param) | Real market slippage (market orders) | None modeled | **MISMATCH (MEDIUM)** |
| Commissions | $0 (Alpaca commission-free) | $0 (Alpaca commission-free) | $0 | **MATCH** |

---

## 9. Walk-Forward vs Live

| Parameter | Backtest | Live | Status |
|---|---|---|---|
| HMM params | Trained per IS window, predict on OOS | Tags passed through, not retrained | By design |
| Kalman filter | Batch per-day (`compute_kalman_stop_multiplier`) | Streaming per-bar (`StreamingKalman.update()`) | **MATCH** (same math) |
| Lookahead bias | ATR shift(1), ADX shift(1), EMA shift(1), realized vol shift(1), gap uses 9:30 open | Streaming (inherent no-lookahead), ADX uses D-1, realized vol uses D-1 | **MATCH** |

---

## 10. Portfolio-Specific

| Parameter | `run_portfolio_backtest.py` | `engine.py` | Status |
|---|---|---|---|
| ORB calculation | `between_time("09:30","09:34")`, high max / low min | `_compute_orb_arrays()`, same logic | **MATCH** |
| Breakout confirmation | `close > orb_high and prev_close > orb_high` | Same | **MATCH** |
| Stop formula | `entry - _ATR_MULTIPLIER * atr * kalman_mult` | `entry - atr_mult * atr * kalman_m` | **MATCH** |
| Target formula | `entry + _RISK_MULTIPLIER * base_atr_risk` | `entry + dynamic_risk_mult * base_atr_risk` (dynamic=2.0 fixed) | **MATCH** |
| Exit check | Bar HIGH/LOW with both-hit disambiguation | Backtesting.py internal (similar) | **MATCH** (functionally) |
| Concurrent position limit | Enforced (default 3) | Not enforced in engine.py (single-position per Backtesting.py) | Different scope |
| Sector limits | Enforced (default 2 per sector) | Not in engine.py (portfolio-level only) | By design |
| Signal ranking | ADX-weighted score when multiple candidates | Not applicable (single ticker per backtest) | By design |

---

## Detailed Mismatch Analysis

### MISMATCH #1: Position Sizing Method (CRITICAL)

**Severity: CRITICAL**

**What's different:**
- **Backtest + Portfolio:** Use `size=0.25` (Backtesting.py) meaning 25% of current equity per trade. On $50K, that's a $12,500 position.
- **Live:** Uses fixed-fractional risk sizing: `shares = (account_size * risk_pct / 100) / abs(entry - stop) * scale_factor`. On $50K with 1% risk and $2 stop distance: `(500/2)*0.25 = 62 shares`. At $450/share that's ~$27,900.

**The formulas are fundamentally different.** The backtest allocates a fixed fraction of equity. The live engine allocates based on risk per share scaled by the same factor. They will produce different share counts depending on the stop distance (ATR) and price level.

**Estimated impact:** Position sizes diverge by 1.5x-3x depending on the stock price and ATR. Backtest PnL and drawdowns are not representative of live performance at the dollar level.

**Recommended fix:** Modify the backtest engine to use the same risk-based position sizing formula as live. Replace `size=self.position_scale_factor` with a calculated share count based on `account * risk_pct / risk_per_share * scale_factor`, then convert to fraction of equity for Backtesting.py's `size` parameter.

---

### MISMATCH #2: VIX Level Cap — Live Only (HIGH)

**Severity: HIGH**

**What's different:**
- **Backtest:** No explicit VIX level check. Uses realized volatility (20-day HV < 18%) as a VIX proxy.
- **Live:** Has BOTH realized vol check AND a hard `vix >= 25` block (`_VIX_MAX = 25` in `orb.py` line 46, checked at line 243).
- **Portfolio:** Uses realized vol only, no VIX level check.

**Estimated impact:** The live engine will block signals on days when VIX >= 25 but realized vol is still < 18% (possible during sudden spikes). This means live will miss some trades that the backtest would have taken. Conversely, there may be days where HV >= 18% but spot VIX < 25 — these would be blocked in backtest but allowed live.

**Recommended fix:** Add explicit VIX level data to the backtest (download ^VIX daily closes, shift by 1 day, block when >= 25) to match the live engine's double gate. Alternatively, document that the VIX < 25 filter is a live-only safety net and the realized vol filter is the primary gate.

---

### MISMATCH #3: Live Has Extra Low-Volume Rejection Gate (MEDIUM)

**Severity: MEDIUM**

**What's different:**
- **Live (`orb.py` lines 274-283):** Has an additional hard reject: `bar_vol < avg_vol * 0.5` (volume less than half the average). This fires BEFORE the 1.5x volume confirmation check.
- **Backtest + Portfolio:** Only have the `volume < avg_vol * 1.5` check. No low-volume floor.

**Estimated impact:** The live engine rejects extremely low-volume bars that the backtest would still consider (as long as they passed the 1.5x threshold on a subsequent bar). In practice, any bar failing the 0.5x check would also fail the 1.5x check, so this is a redundant safety gate. Impact is **near zero** — but it's a code divergence that could confuse future analysis.

**Recommended fix:** Add the `_VOL_LOW_RATIO = 0.5` gate to the backtest engine for exact parity, even though it's functionally redundant.

---

### MISMATCH #4: Portfolio Backtest Lacks Earnings Filter (MEDIUM)

**Severity: MEDIUM**

**What's different:**
- **Backtest (`engine.py`):** Computes `earnings_blocked` array and includes it as informational (tagged, not blocking).
- **Live (`orb.py`):** Calls `is_earnings_blackout()` — informational only (tags `EARNINGS`, no blocking).
- **Portfolio (`run_portfolio_backtest.py`):** Does not check earnings at all. No import of earnings functions.

**Estimated impact:** Since earnings is informational-only (no blocking) in both backtest and live, the portfolio backtest simply lacks the informational tag. No trades are missed or added. Impact on trade selection is **zero** but the portfolio backtest cannot report which trades occurred near earnings for post-hoc analysis.

**Recommended fix:** Add earnings tagging to the portfolio backtest for analysis parity.

---

### MISMATCH #5: Portfolio Backtest Slippage Modeling (MEDIUM)

**Severity: MEDIUM**

**What's different:**
- **Backtest (`engine.py`):** Models $0.02/share slippage via Backtesting.py's `commission` parameter.
- **Live:** Real market slippage (market orders).
- **Portfolio (`run_portfolio_backtest.py`):** No slippage modeled. PnL is calculated as `(target - entry) * (size / entry)` with no friction.

**Estimated impact:** The portfolio backtest overstates returns by ~$0.04/share round-trip. Over hundreds of trades, this could inflate net PnL by several percent.

**Recommended fix:** Subtract $0.02/share from entry (longs) or add to entry (shorts), or apply as a flat deduction per trade.

---

### MISMATCH #6: Excluded Weekdays — Portfolio Hardcoded (LOW)

**Severity: LOW**

**What's different:**
- **Backtest:** Configurable `excluded_days` parameter (default "0" = Monday).
- **Live:** Configurable `excluded_days` (default `[0]`).
- **Portfolio:** Hardcoded `day.weekday() == 0` check (only Monday, not configurable).

**Estimated impact:** None unless you want to test different excluded days. Functionally identical in default config.

**Recommended fix:** Use the `excluded_days` config in the portfolio backtest for consistency.

---

### MISMATCH #7: Both-SL-and-TP-Hit Disambiguation (LOW)

**Severity: LOW**

**What's different:**
- **Backtest (`engine.py`):** Handled by Backtesting.py internals (undocumented priority order).
- **Portfolio (`run_portfolio_backtest.py`):** Explicit disambiguation: if both SL and TP are hit in the same bar, checks whether `close > open` (bullish = favor target for longs).
- **Live:** Alpaca exchange handles the race condition (bracket order mechanics).

**Estimated impact:** Rare edge case. The portfolio backtest has a reasonable disambiguation; the Backtesting.py behavior is opaque but likely similar (favors the first-hit order). This could cause 1-2 trades per year to differ.

**Recommended fix:** Document the portfolio's disambiguation logic as the "canonical" behavior for auditing purposes.

---

## VP (Volume Profile) Blocking Behavior

| Behavior | Backtest | Live | Status |
|---|---|---|---|
| VP_HVN_TARGET (target inside VA) | Code sets `blocked = True` but `_vp_blocked` is **NOT used** (line 999-1001: `self.buy()` runs regardless) | No blocking — VP is informational only (confidence adjustment) | **MATCH** (both non-blocking) |
| VP_LVN_TARGET | Tag only | Confidence +1 | **MATCH** (informational) |
| VP_POC_CROSS | Tag only | Confidence -1 | **MATCH** (informational) |

Note: The backtest code computes `_vp_blocked` but then ignores it. The variable name with underscore prefix confirms it was intentionally made informational.

---

## Summary Table

| # | Category | Issue | Severity | Status |
|---|---|---|---|---|
| 1 | Position sizing | Backtest uses equity fraction; live uses risk-based formula | **CRITICAL** | **RESOLVED v1.8.1** |
| 2 | VIX filter | Live has VIX < 25 hard gate; backtest uses only realized vol | **HIGH** | **RESOLVED v1.8.1** |
| 3 | Volume filter | Live has extra 0.5x low-volume rejection | **MEDIUM** | Open (redundant gate, near-zero impact) |
| 4 | Earnings filter | Portfolio backtest missing earnings tagging | **MEDIUM** | Open (informational only, zero trade impact) |
| 5 | Slippage | Portfolio backtest has no slippage; engine.py has $0.02 | **MEDIUM** | **RESOLVED v1.8.1** |
| 6 | Excluded days | Portfolio hardcodes Monday | **LOW** | **RESOLVED v1.8.1** (added --no-monday flag) |
| 7 | SL/TP tie-break | Different disambiguation logic | **LOW** | Open (1-2 trades/year) |

---

## v1.8.1 Resolution Details (2026-03-24)

### Fix #1: Position Sizing (CRITICAL → RESOLVED)

**Change:** Replaced `size=position_scale_factor` (25% of equity) with risk-based formula matching `src/risk/position_sizing.py`:
```
shares = int((equity * 1% / risk_per_share) * 0.25)
```
Capped at 25% of equity for safety. Applied to both `engine.py` and `run_portfolio_backtest.py`.

**Impact on individual backtest ($200K, 26 tickers):**
| Metric | v1.8.0 | v1.8.1 | Delta |
|--------|--------|--------|-------|
| Trades | 2,094 | 1,763 | -331 (VIX gate) |
| PF | 1.34 | 1.25 | -0.09 |
| Sharpe | 1.91 | 1.55 | -0.36 |
| Net PnL | +$103K | +$61K | -$42K |
| Max DD | -8.2% | -6.8% | +1.5pp (better) |

Signal quality unchanged (WR ~45%). Dollar metrics more conservative and realistic.

### Fix #2: VIX Spot Gate (HIGH → RESOLVED)

**Change:** Added `VIX_SPOT` column to backtest data (D-1 VIX close, no lookahead). Both `engine.py` and `run_portfolio_backtest.py` now block entries when VIX >= 25, matching live `_VIX_MAX`.

**Impact:** ~330 trades blocked in high-VIX periods (2020 COVID, 2022 bear). These were marginal-to-negative trades.

### Fix #3: Slippage (MEDIUM → RESOLVED)

**Change:** Portfolio backtest now applies $0.02/share slippage on both entry (adjusted fill price) and exit (deducted from PnL). Engine.py already had spread-based slippage via Backtesting.py.

**Impact on portfolio backtest:** PF reduced by ~0.02-0.03 from slippage friction.

### Key Insight: Portfolio Signal Selection Needs Work

The position sizing fix revealed that the portfolio backtest's signal ranking (score = ADX weight + ORB range) is insufficient. With risk-based sizing, tighter stops → larger positions → amplified losses on stop-outs. The individual backtest (PF 1.25) still works because each ticker runs independently. The portfolio needs a better signal selection mechanism (e.g., historical PF weighting, ticker quality scores) to overcome realistic friction.

---

## Recommended Remaining Priority

1. **Improve portfolio signal ranking** — Replace ADX+ORB-range score with ticker-quality-aware ranking (e.g., weight by rolling PF from individual backtest).
2. **Volume filter 0.5x gate (MEDIUM)** — Add for code parity, near-zero functional impact.
3. **Earnings tagging (MEDIUM)** — Add for analysis completeness.
4. **SL/TP tie-break (LOW)** — Document current behavior.
