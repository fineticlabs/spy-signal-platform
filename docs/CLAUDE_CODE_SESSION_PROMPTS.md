# Claude Code Session Prompts — Feed These In Order

Copy-paste each prompt into Claude Code (VS Code) one session at a time.
Wait for each session to complete and verify the output before moving to the next.

---

## SESSION 1: Project Scaffolding + Data Ingestion

```
Read CLAUDE.md first. Then:

1. Create all __init__.py files for every package in the repo structure
2. Create src/config.py using pydantic-settings to load from .env:
   - AlpacaSettings (api_key, secret_key, base_url, feed)
   - TelegramSettings (bot_token, chat_id)
   - RiskSettings (account_size, risk_per_trade_pct, max_daily_loss_pct, max_trades_per_day)
   - AppSettings (log_level, trading_mode, db_path)
3. Create src/models.py with Pydantic models:
   - Bar (timestamp, open, high, low, close, volume, vwap — all Decimal except volume/timestamp)
   - TimeFrame enum (ONE_MIN, FIVE_MIN, FIFTEEN_MIN, THIRTY_MIN, DAILY)
4. Create src/ingestion/historical.py:
   - Function to fetch N days of 1-min SPY bars from Alpaca REST API
   - Store results in SQLite (create table if not exists)
   - Handle pagination (Alpaca returns max 10000 bars per request)
5. Create src/ingestion/websocket.py:
   - Async WebSocket listener that connects to Alpaca's real-time stream
   - Receives 1-min bars for SPY, converts to Bar model
   - Puts bars onto an asyncio.Queue for downstream consumers
6. Create src/storage/database.py:
   - SQLite connection with WAL mode
   - Table creation for bars (symbol, timeframe, timestamp, OHLCV, vwap)
   - Insert and query functions
7. Create scripts/backfill_data.py:
   - CLI script that downloads last 90 days of 1-min SPY bars
   - Shows progress with bar count
8. Create tests/test_ingestion.py:
   - Test that Bar model validates correctly
   - Test that config loads from env vars
   - Test database create/insert/query cycle

Run ruff and mypy after creating all files. Fix any issues.
```

---

## SESSION 2: Indicator Engine

```
Read CLAUDE.md first. Build the indicator computation layer.

1. Create src/indicators/batch.py:
   - Functions using TA-Lib that take a pandas DataFrame of bars and return indicator values:
   - calculate_ema(df, period) → Series — for periods 9, 20, 50
   - calculate_rsi(df, period=14) → Series
   - calculate_macd(df, fast=12, slow=26, signal=9) → tuple of 3 Series
   - calculate_bollinger(df, period=20, std=2) → tuple of 3 Series (upper, middle, lower)
   - calculate_atr(df, period=14) → Series
   - calculate_vwap(df) → Series — cumulative from session open
   - Each function: typed, docstring explaining what it measures, handles NaN at start

2. Create src/indicators/streaming.py:
   - Classes wrapping talipp for incremental computation as bars arrive:
   - StreamingEMA(period)
   - StreamingRSI(period=14)
   - StreamingATR(period=14)
   - StreamingMACD(fast=12, slow=26, signal=9)
   - Each has an .update(bar: Bar) method and .value property
   - Each has a .ready property that returns False until enough bars exist

3. Create src/indicators/registry.py:
   - IndicatorRegistry class that manages a set of active indicators
   - .register(name, indicator) and .update_all(bar) and .get_snapshot() → dict
   - get_snapshot returns current values of all registered indicators as a Pydantic model

4. Create src/models.py updates — add:
   - IndicatorSnapshot model with fields for each indicator value (all Optional[Decimal])

5. Create tests/test_indicators.py:
   - Test EMA(9) against hand-calculated values for a known 20-bar sequence
   - Test RSI(14) produces values between 0-100
   - Test ATR(14) is always positive
   - Test VWAP calculation matches manual formula: cumsum(price*volume) / cumsum(volume)
   - Test streaming vs batch indicators produce same values (within floating point tolerance)

Run ruff, mypy, and pytest after. Fix all issues.
```

---

## SESSION 3: Levels Engine

```
Read CLAUDE.md first. Build the key price levels module.

1. Create src/levels/opening_range.py:
   - OpeningRangeTracker class
   - Accumulates bars from 9:30-9:35 ET (first 5 completed 1-min bars)
   - Exposes: orb_high, orb_low, orb_midpoint, orb_range, is_complete
   - Also track 15-min ORB (9:30-9:45) as secondary reference
   - Reset daily at 9:30

2. Create src/levels/daily_levels.py:
   - PreviousDayLevels: loads yesterday's high, low, close from SQLite
   - PremarketLevels: tracks high/low from 4:00-9:30 ET (if premarket data available from Alpaca)
   - If no premarket data, skip gracefully (log warning, don't crash)

3. Create src/levels/vwap.py:
   - SessionVWAP class:
   - Incrementally computes VWAP from 9:30 ET
   - Also computes VWAP ± 1σ and ± 2σ bands (standard deviation of price from VWAP, volume-weighted)
   - Exposes: vwap, upper_1, lower_1, upper_2, lower_2
   - Reset daily at 9:30

4. Create src/levels/dynamic.py:
   - DayTracker class
   - Tracks: high_of_day, low_of_day, last_price
   - Updates on every bar

5. Create src/levels/__init__.py:
   - LevelManager class that orchestrates all level trackers
   - .update(bar) → updates all trackers
   - .get_levels() → LevelSnapshot Pydantic model with all current levels

6. Add LevelSnapshot model to src/models.py

7. Create tests/test_levels.py:
   - Test ORB correctly identifies high/low from first 5 bars
   - Test ORB is_complete returns False before 9:35 ET
   - Test VWAP formula matches manual calculation
   - Test HOD/LOD tracking updates correctly

Run ruff, mypy, pytest. Fix all issues.
```

---

## SESSION 4: Strategy #1 — Opening Range Breakout

```
Read CLAUDE.md first. Implement the ORB strategy.

1. Create src/strategies/base.py:
   - Abstract Strategy class with methods:
   - .evaluate(bar, indicators, levels, regime) → Optional[Signal]
   - .name property
   - .required_indicators() → list of indicator names

2. Create src/strategies/regime.py:
   - RegimeDetector class
   - Classifies current market into: TRENDING_UP, TRENDING_DOWN, RANGING, CHOPPY
   - Uses: ADX on 15-min (>25 = trending, <15 = choppy)
   - Uses: VIX level (will need to fetch VIX quote from Alpaca)
   - Exposes: current_regime, vix_level, adx_value, is_tradeable

3. Create src/strategies/orb.py:
   - ORBStrategy(Strategy):
   - Entry conditions:
     - ORB is complete (5 bars done)
     - LONG: current close > ORB high AND volume > 1.5x 20-bar average volume
     - SHORT: current close < ORB low AND volume > 1.5x 20-bar average volume
   - Filters:
     - regime.vix_level < 25
     - regime.adx_value > 20
     - current time not in lunch chop (11:30-13:30 ET)
     - current time before 15:45 ET
   - Stop: entry_price -/+ 1.5 * ATR(14) on 5-min
   - Target: entry_price +/- 2 * risk_distance
   - Output Signal model

4. Update src/models.py — add:
   - Signal model: direction (LONG/SHORT), strategy_name, entry_price, stop_price, target_price,
     risk_reward_ratio, confidence_score (1-5), reason (str), timeframe, regime, vix, adx,
     indicators_snapshot, levels_snapshot, timestamp
   - Regime enum: TRENDING_UP, TRENDING_DOWN, RANGING, CHOPPY

5. Create tests/test_strategies.py:
   - Test ORB fires LONG when price breaks above ORB high with sufficient volume
   - Test ORB fires SHORT when price breaks below ORB low with sufficient volume
   - Test ORB does NOT fire during lunch chop
   - Test ORB does NOT fire when VIX > 25
   - Test ORB does NOT fire before ORB is complete
   - Test stop and target are calculated correctly relative to ATR

Run ruff, mypy, pytest. Fix all issues.
```

---

## SESSION 5: Risk Manager

```
Read CLAUDE.md first. Build the risk management gate.

1. Create src/risk/manager.py:
   - RiskManager class
   - Pre-trade check: .approve(signal: Signal) → RiskDecision (approved/rejected + reason)
   - Checks (in order, fail-fast):
     a. Is trading allowed today? (not past daily loss limit)
     b. Is current time in allowed window? (9:35-15:45 ET, not lunch chop unless confidence >= 4)
     c. Is daily trade count under limit? (max from config)
     d. Is consecutive loss count under 3?
     e. Is cooldown active? (15 min after 2 consecutive losses)
     f. Is risk/reward >= 1.5?
     g. Does position size stay within 1% account risk?
   - Every check result logged with full context

2. Create src/risk/position_sizing.py:
   - calculate_position_size(account_size, risk_pct, entry, stop) → int (shares)
   - Uses Decimal math, rounds down to whole shares
   - Returns 0 if stop is too close (would require > max position)

3. Create src/risk/cooldown.py:
   - CooldownTracker class
   - Tracks: consecutive_losses, daily_pnl, daily_trade_count, last_loss_time
   - .record_trade(result: TradeResult) → updates state
   - .is_cooled_down() → bool (15 min since 2nd consecutive loss)
   - .is_tilted() → bool (3+ consecutive losses)
   - .reset_daily() → called at 9:30 ET

4. Add RiskDecision and TradeResult models to src/models.py

5. Create tests/test_risk.py:
   - Test approval of valid signal with clean risk state
   - Test rejection when daily loss limit hit
   - Test rejection when max trades reached
   - Test rejection during cooldown period
   - Test rejection for poor R:R ratio
   - Test position sizing calculation matches manual formula
   - Test position sizing returns 0 for impossible stop distance

Run ruff, mypy, pytest. Fix all issues.
```

---

## SESSION 6: Alerting

```
Read CLAUDE.md first. Build the alert dispatch system.

1. Create src/alerts/formatter.py:
   - format_signal_alert(signal: Signal, risk_decision: RiskDecision) → str
   - Rich Telegram MarkdownV2 formatted message with:
     - 🟢/🔴 emoji for direction
     - Symbol, price, strategy name
     - Entry, stop, target prices
     - Risk/reward ratio
     - Confidence score (stars or number)
     - VIX level, regime
     - Time
     - Key levels (VWAP, ORB, PDH/PDL)
     - Reason summary
   - format_risk_alert(message: str) → str for daily limit warnings
   - format_daily_summary(trades: list[TradeResult]) → str for end-of-day recap

2. Create src/alerts/telegram.py:
   - TelegramAlerter class (async)
   - Uses python-telegram-bot to send messages
   - .send_signal(signal) — formatted signal alert
   - .send_risk_warning(message) — risk limit notifications
   - .send_daily_summary(trades) — end of day summary
   - Retry logic: 3 attempts with exponential backoff
   - Error logging (don't crash the system if Telegram fails)

3. Create src/alerts/dispatcher.py:
   - AlertDispatcher class
   - Receives approved signals from risk manager
   - Dispatches to all configured channels (Telegram for MVP)
   - Logs every alert sent with timestamp and delivery status
   - Rate limiting: max 1 alert per 30 seconds to avoid spam

4. Create scripts/test_telegram.py:
   - Simple script that sends a test alert to verify bot is configured correctly
   - Should send a sample signal message and confirm delivery

5. Create tests/test_alerts.py:
   - Test formatter produces valid MarkdownV2 (escape special chars)
   - Test dispatcher respects rate limiting
   - Test dispatcher handles Telegram failure gracefully (logs error, doesn't crash)

Run ruff, mypy, pytest. Fix all issues.
```

---

## SESSION 7: Orchestrator + Integration

```
Read CLAUDE.md first. Wire everything together.

1. Create src/main.py:
   - Main asyncio orchestrator
   - Startup sequence:
     a. Load config from .env
     b. Initialize SQLite database
     c. Load previous day levels from DB
     d. Initialize all indicator streaming engines
     e. Initialize level trackers
     f. Initialize ORB strategy
     g. Initialize regime detector
     h. Initialize risk manager
     i. Initialize alert dispatcher
     j. Connect Alpaca WebSocket
   - Main loop:
     a. Receive bar from WebSocket queue
     b. Update indicators
     c. Update levels
     d. Update regime
     e. Evaluate strategy
     f. If signal: run through risk manager
     g. If approved: dispatch alert
     h. Log everything to SQLite
   - Graceful shutdown on SIGINT/SIGTERM
   - Daily reset at 9:30 ET (clear levels, reset risk counters)
   - End-of-day summary at 16:05 ET

2. Create src/api/routes.py:
   - FastAPI app with endpoints:
   - GET /health — system status
   - GET /state — current indicators, levels, regime, risk state
   - GET /signals — recent signals (last 24h)
   - GET /trades — trade log
   - POST /trades — manually log a trade result (for tracking)

3. Update src/main.py to start FastAPI in background (uvicorn on port 8000)

4. Create tests/test_integration.py:
   - Test that a sequence of bars flows through the full pipeline:
     bars → indicators → levels → strategy → risk → alert
   - Use mock bars that simulate an ORB breakout scenario
   - Verify a signal is produced with correct properties

Run ruff, mypy, pytest. Fix all issues.
```

---

## SESSION 8: Backtesting

```
Read CLAUDE.md first. Build the backtesting framework.

1. Create src/backtest/data_loader.py:
   - Load historical bars from SQLite into pandas DataFrame
   - Resample 1-min bars to 5-min and 15-min using proper OHLCV aggregation
   - Split into walk-forward windows (60-day IS, 20-day OOS)

2. Create src/backtest/engine.py:
   - ORBBacktest class using Backtesting.py framework
   - Implements the ORB strategy as a Backtesting.py Strategy subclass
   - Properly handles:
     - Opening range calculation (no lookahead — only uses completed bars)
     - Volume filter
     - Time-of-day filter
     - ATR-based stops
     - Fixed-R targets
   - Slippage model: $0.02 per share round-trip
   - Commission: $0 (Alpaca is commission-free)

3. Create src/backtest/metrics.py:
   - Calculate from backtest results:
     - Win rate, loss rate
     - Profit factor (gross profit / gross loss)
     - Expectancy (avg_win * win_rate - avg_loss * loss_rate)
     - Max drawdown (peak-to-trough)
     - Sharpe ratio (annualized)
     - Average winner, average loser
     - Reward-to-risk realized
     - Trades per day average
     - Time-of-day performance breakdown
   - Print formatted summary table
   - Save equity curve as PNG

4. Create scripts/run_backtest.py:
   - CLI script that:
     - Loads historical data
     - Runs walk-forward backtest
     - Prints metric summary
     - Saves equity curve to docs/equity_curve.png
     - Saves full trade log to docs/backtest_results.csv

5. Create tests/test_backtest.py:
   - Test that backtest engine doesn't use future data (shift check)
   - Test metrics calculations against known values
   - Test walk-forward window generation produces correct date ranges

Run ruff, mypy, pytest. Fix all issues.
```

---

## SESSION 9: Streamlit Dashboard

```
Read CLAUDE.md first. Build the monitoring dashboard.

1. Create src/dashboard/app.py:
   - Streamlit app that polls FastAPI endpoints every 5 seconds
   - Layout:
     - Header: SPY price, daily change, VIX level, current regime badge
     - Row 1: Key levels table (VWAP, ORB H/L, PDH/PDL/PDC, HOD/LOD)
     - Row 2: Indicator gauges (RSI, ADX, ATR) + EMA alignment status
     - Row 3: Active signals list with color-coded confidence
     - Row 4: Recent alerts log (scrollable, last 20)
     - Row 5: Daily P&L tracker + trade count + risk status
     - Sidebar: System health (WS connected, last bar time, data freshness)
   - Use plotly for any charts
   - Color code: green for bullish, red for bearish, yellow for neutral
   - Show "NO TRADE ZONE" banner during lunch chop or when risk limits are hit

Run ruff and verify the dashboard starts with `streamlit run src/dashboard/app.py`.
```

---

## NOTES FOR ALL SESSIONS

- Always read CLAUDE.md before starting
- Always run `ruff check`, `ruff format`, and `mypy` before finishing
- Always run `pytest` and fix any failures
- Use Decimal for all prices, never float
- Use structlog for all logging
- Every new file needs type annotations on every function
- Every public function needs a docstring
- When in doubt, refer to CLAUDE.md for architecture decisions
