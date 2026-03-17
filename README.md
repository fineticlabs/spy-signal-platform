# SPY Intraday Signal Platform

A personal intraday trading signal platform that monitors 26 US stocks and ETFs, detects breakout setups using statistical models, and sends Telegram alerts for manual execution. In paper_trade mode, it also places bracket orders on Alpaca's paper trading API for automated forward-testing. Built as a personal tool, not a product.

## What is this?

Every trading day at 9:30 AM ET, markets open and the first few minutes of trading establish an "opening range" for each stock. This platform watches those opening ranges across 26 liquid tickers, detects when price breaks out with conviction, runs the setup through a series of quality filters, and sends a Telegram alert with exact entry, stop-loss, and target prices. You decide whether to take the trade -- or let the platform place paper orders automatically for forward-testing.

## How it works

The core strategy is an **Opening Range Breakout (ORB)**:

1. **Opening range** -- The first 5 completed 1-minute bars (9:30-9:35 ET) establish the high and low of the opening range for each ticker.
2. **Breakout detection** -- If price closes above the opening range high (long) or below the opening range low (short) with volume at least 1.5x the 20-bar average, a potential signal is generated.
3. **Quality filters** -- The signal passes through a dozen filters (trend alignment, regime detection, volatility, economic calendar, and more) that reject setups likely to fail.
4. **Alert and execution** -- Surviving signals are sent to Telegram with entry price, stop-loss, profit target, confidence score, and tags explaining why the signal fired. In paper_trade mode, a bracket order (market entry + stop-loss + take-profit) is simultaneously submitted to Alpaca's paper trading API.
5. **Exit** -- Target is a fixed multiple of the risk distance (typically 2R). All positions are flat by 3:55 PM ET.

## Performance (backtest)

Walk-forward out-of-sample backtest across 26 tickers, 5.6 years (2020-2026), $50,000 starting capital:

| Metric | Value |
|---|---|
| Profit Factor | 1.423 |
| Sharpe Ratio | 2.270 |
| Win Rate | 47.0% |
| Expectancy | $59.07 / trade |
| Net Profit | $117,018 |
| Annual Yield | 23.8% (CAGR on $50K starting capital) |
| Max Drawdown | -6.23% |
| Total Trades | 1,981 (~5 signals / day) |

All 26 tickers are individually profitable. Results validated via Monte Carlo permutation test (p=0.0000, 10,000 permutations) and Combinatorial Purged Cross-Validation (15/15 paths profitable on original universe).

**Disclaimer:** Past performance does not guarantee future results. Backtest assumes $0.02/share slippage, zero commissions (Alpaca), and does not account for liquidity constraints or partial fills.

### First live day

March 17, 2026 -- first day running in paper_trade mode against Alpaca paper API:

| Metric | Value |
|---|---|
| Signals | 9 |
| Wins / Losses | 3W / 6L |
| Win Rate | 33.3% |
| Net P&L | +$137.21 |

Despite a sub-50% win rate, positive expectancy from the 2R target structure produced a net gain. Full replay available via `python scripts/replay_day.py --date 2026-03-17`.

## Execution modes

The platform supports three execution modes, controlled by the `EXECUTION_MODE` environment variable:

| Mode | Behavior |
|---|---|
| `alerts_only` (default) | Sends Telegram alerts only. No orders placed. |
| `paper_trade` | Sends Telegram alerts and places bracket orders on Alpaca paper trading API. |
| `live_trade` | Sends Telegram alerts and places bracket orders on Alpaca live API. Use with caution. |

Safety checks enforced in paper_trade and live_trade modes:
- Market hours gate (9:30-16:00 ET)
- Minimum buying power check ($1,000)
- Paper account assertion in paper_trade mode (account number must start with "PA")
- EOD flatten at 3:55 PM ET (cancel open orders, close all positions)

## Technical stack

| Component | Technology |
|---|---|
| Language | Python 3.12+ with full type annotations |
| Market data | Alpaca Markets API (1-min bars, WebSocket streaming) |
| Execution | alpaca-py TradingClient (bracket orders: market entry + stop-loss + take-profit) |
| Indicators | TA-Lib (batch), talipp (streaming), numpy, pandas |
| Backtesting | Backtesting.py with walk-forward OOS engine |
| Regime detection | Hidden Markov Model (hmmlearn) -- 3-state GaussianHMM trained on rolling 60-day windows |
| Adaptive stops | Kalman filter (filterpy) -- innovation-based stop sizing, widens in volatile regimes |
| Volume profile | Prior-day POC, Value Area (VAH/VAL), HVN/LVN levels |
| VIX term structure | VIX/VIX3M ratio for contango/backwardation regime tagging |
| Storage | SQLite (WAL mode) for bar data, DuckDB for analytics |
| Alerts | Telegram Bot API (python-telegram-bot) |
| Automation | launchd (macOS) -- weekday auto-start/stop with daily P&L summary to Telegram |
| Dashboard | Streamlit + Plotly |
| API | FastAPI + Uvicorn |
| Data models | Pydantic v2 (no raw dicts anywhere) |
| ML (tested, rejected) | LightGBM signal scorer -- AUC 0.487, no predictive value |

## Telegram alerts

Real-time alerts are sent to Telegram whenever a signal fires during market hours. Each alert includes the ticker, direction, entry price, stop-loss, target, confidence score, and filter tags.

After market close, a daily P&L summary is delivered to Telegram automatically (when using launchd or `auto_stop.sh`). The summary includes a trade-by-trade breakdown with ticker, direction, share count, entry/exit prices, outcome, and P&L, followed by totals for signals, wins, losses, win rate, and net P&L.

## Active filters

The strategy applies these filters before generating a signal:

| Filter | Type | Description |
|---|---|---|
| Trading window | Blocking | 9:35-10:00 ET only (momentum exhausts after 25 min) |
| Monday | Blocking | Skip Mondays (PF 1.125 vs 1.5+ other days) |
| 15-min EMA trend | Blocking | Long only above EMA(20), short only below |
| Consecutive candle | Blocking | Requires 2 consecutive closes beyond ORB level |
| Gap filter | Blocking | Gap-up = long only, gap-down = short only |
| Realized volatility | Blocking | Skip when 20-day HV > 18% |
| Daily ADX | Blocking | Skip when daily ADX(14) < 25 |
| Economic calendar | Blocking | No trades on FOMC, NFP, CPI, PPI days |
| Volume Profile HVN | Blocking | Skip when target lands inside Value Area |
| VIX term structure | Blocking | Skip backwardation days (VIX/VIX3M > 1.00) |
| Dynamic targets | Active | ORB range percentile adjusts R:R (1.5x-2.5x) |
| Kalman adaptive stops | Active | Innovation-based multiplier (0.90-1.50x ATR) |
| HMM regime | Informational | Tags: HMM_VOLATILE (+1), HMM_CALM (-1 confidence) |
| SPY market direction | Informational | Tags: SPY_ALIGNED (+1), SPY_CONFLICT (-2 confidence) |
| Earnings proximity | Informational | Tags ticker with EARNINGS when near announcement |
| VIX contango | Informational | Tags: CONTANGO (+1 confidence) |
| RVOL | Informational | Tags: HIGH_RVOL (+1), LOW_RVOL (-2 confidence) |

## Tickers

**Original 15:** SPY, QQQ, MSFT, AMD, TSLA, AMZN, UBER, SMCI, SHOP, PLTR, NFLX, MSTR, SNOW, ARM, DASH

**Expansion 11:** PYPL, INTC, MU, HOOD, DKNG, SOXL*, ROKU, TQQQ*, BA, MRVL, META

*\* 3x leveraged ETFs (SOXL = semiconductors, TQQQ = Nasdaq-100)*

## Project structure

```
spy-signal-platform/
├── src/
│   ├── main.py              # Asyncio orchestrator entry point
│   ├── config.py            # Pydantic Settings from .env
│   ├── models.py            # Shared Pydantic models (Bar, Signal, etc.)
│   ├── ingestion/           # Alpaca WebSocket + REST data fetching
│   ├── indicators/          # TA-Lib batch + talipp streaming indicators
│   ├── levels/              # VWAP, ORB, daily levels, Kalman filter
│   ├── strategies/          # ORB strategy, regime detection, HMM
│   ├── filters/             # Economic calendar, earnings, VIX term structure
│   ├── signals/             # Confluence scoring and signal explanation
│   ├── risk/                # Position sizing, cooldown, daily loss limits
│   ├── alerts/              # Telegram bot integration and formatting
│   ├── execution/           # Alpaca bracket order execution (paper/live)
│   ├── storage/             # SQLite database layer
│   ├── backtest/            # Walk-forward engine, data loader, metrics
│   ├── api/                 # FastAPI endpoints
│   └── dashboard/           # Streamlit UI
├── scripts/                 # CLI tools (backtest, replay, backfill, etc.)
├── tests/                   # pytest suite (448 tests)
├── config/                  # Strategy parameters (settings.yaml)
├── data/                    # SQLite database, earnings cache
└── docs/                    # Equity curves, trade logs, design docs
```

## Quick start

### Prerequisites

- Python 3.12+
- TA-Lib C library (`brew install ta-lib` on macOS, `apt install libta-lib-dev` on Linux)
- Alpaca account (free): https://app.alpaca.markets
- Telegram bot (free): message @BotFather on Telegram

### Setup

```bash
git clone <this-repo>
cd spy-signal-platform

python -m venv .venv
source .venv/bin/activate

make dev                    # Install all dependencies
cp .env.example .env        # Edit with your Alpaca + Telegram credentials
make backfill               # Download historical bar data
make test-telegram          # Verify Telegram alerts work
make test                   # Run tests
```

### Enable paper trading

To forward-test signals with paper orders:

```bash
# In .env, set:
EXECUTION_MODE=paper_trade

# Verify paper account connectivity:
python scripts/check_paper_account.py

# Start the scanner (signals will alert AND place paper orders):
make run
```

### Run

```bash
make run                    # Start the live signal platform
make dashboard              # Start Streamlit dashboard (separate terminal)
make backtest               # Run the walk-forward backtest
```

## Automation (Mac)

The platform can run fully automated on macOS using launchd. Two scheduled jobs handle the entire daily workflow:

- **6:25 AM PT (9:25 AM ET) Mon--Fri:** `auto_start.sh` backfills recent market data, then starts the live scanner. The scanner monitors all 26 tickers during market hours and sends real-time Telegram alerts when ORB setups fire.
- **1:05 PM PT (4:05 PM ET) Mon--Fri:** `auto_stop.sh` stops the scanner, backfills end-of-day data, runs `replay_day.py` to evaluate the day's trades, and sends a formatted daily P&L summary to Telegram with a trade-by-trade breakdown including signals, wins, losses, win rate, and net P&L.

### Setup automation

```bash
bash scripts/install_launchd.sh
```

This installs two launchd jobs (`com.fineticlabs.spy-scanner-start` and `com.fineticlabs.spy-scanner-stop`) that auto-start and auto-stop the scanner on weekdays.

**IMPORTANT:** Your Mac must stay awake during market hours for the scheduled jobs to fire. Go to System Settings > Energy > turn on "Prevent automatic sleeping when display is off."

To remove automation:

```bash
bash scripts/uninstall_launchd.sh
```

### Manual start/stop

If you prefer to run the scanner manually without launchd automation:

```bash
bash scripts/auto_start.sh                         # Start scanner (backfill + launch)
bash scripts/auto_stop.sh                           # Stop scanner + get daily report on Telegram

# Or run directly
make run                                            # Terminal 1: start live scanner
make dashboard                                      # Terminal 2: start Streamlit dashboard

# Replay any past trading day
python scripts/replay_day.py --date 2026-03-17
```

### Monitoring

```bash
cat logs/scanner.pid                                # Check scanner PID
ps aux | grep "src.main"                            # Check if scanner process is running
cat logs/scanner_$(date +%Y-%m-%d).log              # View today's scanner log
```

Telegram alerts arrive in real-time during market hours whenever an ORB setup fires. After market close, a daily P&L summary is delivered to Telegram automatically (when using launchd or `auto_stop.sh`).

### Troubleshooting

- **Scanner not starting:** Check `logs/scanner_YYYY-MM-DD.log` for errors. Common causes are missing `.env` credentials or an unreachable Alpaca API.
- **No Telegram messages:** Verify `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` in `.env`. Run `make test-telegram` to confirm the bot works.
- **Stale PID file:** If the scanner crashed without cleanup, delete `logs/scanner.pid` and restart with `bash scripts/auto_start.sh`.
- **Mac was asleep during market hours:** Run `python scripts/backfill_data.py` to catch up on missed bars, then `python scripts/replay_day.py --date YYYY-MM-DD` to see what signals you missed.

## Scripts

| Script | Description |
|---|---|
| `run_backtest.py` | Full 26-ticker walk-forward backtest with per-ticker summary, combined metrics, equity curve, and trade log CSV. |
| `replay_day.py` | Replay a single trading day to see what signals would have fired, with entry/stop/target and P&L outcome. |
| `backfill_data.py` | Download historical 1-min bars from Alpaca for one or more tickers. |
| `check_paper_account.py` | Display paper account equity, buying power, open positions, and open orders. |
| `run_monte_carlo.py` | Monte Carlo permutation test (10,000 shuffles) to validate that backtest results are not due to chance. |
| `run_cpcv.py` | Combinatorial Purged Cross-Validation to test strategy robustness across all possible train/test path combinations. |
| `train_ml_scorer.py` | LightGBM signal scorer experiment with Optuna tuning and SHAP analysis. Tested and rejected (AUC 0.487). |
| `test_telegram.py` | Send a test message to verify Telegram bot configuration. |
| `auto_start.sh` | Activate venv, backfill recent data, start the live scanner with PID tracking. |
| `auto_stop.sh` | Stop the scanner, backfill EOD data, replay the day, send Telegram daily summary. |
| `install_launchd.sh` | Install launchd plists for automated weekday scheduling (6:25 AM / 1:05 PM PT). |
| `uninstall_launchd.sh` | Unload and remove the launchd plists. |

## Development

```bash
make lint                   # Run ruff linter
make format                 # Auto-format code
make test                   # Fast unit tests (448 tests)
make test-all               # All tests including slow/integration
make check                  # lint + typecheck + test
```
