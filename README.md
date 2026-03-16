# SPY Intraday Signal Platform

A personal intraday trading signal platform that monitors 15 US stocks and ETFs, detects breakout setups using statistical models, and sends Telegram alerts for manual execution. It does not auto-trade or manage money. Built as a personal tool, not a product.

## What is this?

Every trading day at 9:30 AM ET, markets open and the first few minutes of trading establish an "opening range" for each stock. This platform watches those opening ranges across 15 liquid tickers, detects when price breaks out with conviction, runs the setup through a series of quality filters, and sends a Telegram alert with exact entry, stop-loss, and target prices. You decide whether to take the trade.

The system is a discretionary trading assistant: it analyzes, scores, and alerts. It never places orders.

## How it works

The core strategy is an **Opening Range Breakout (ORB)**:

1. **Opening range** -- The first 5 completed 1-minute bars (9:30-9:35 ET) establish the high and low of the opening range for each ticker.
2. **Breakout detection** -- If price closes above the opening range high (long) or below the opening range low (short) with volume at least 1.5x the 20-bar average, a potential signal is generated.
3. **Quality filters** -- The signal passes through a dozen filters (trend alignment, regime detection, volatility, economic calendar, and more) that reject setups likely to fail.
4. **Alert** -- Surviving signals are sent to Telegram with entry price, stop-loss, profit target, confidence score, and tags explaining why the signal fired.
5. **Exit** -- Target is a fixed multiple of the risk distance (typically 2R). All positions are flat by 3:55 PM ET.

## Performance (backtest)

Walk-forward out-of-sample backtest across 15 tickers, 5.6 years (2020-2026), $50,000 starting capital:

| Metric | Value |
|---|---|
| Profit Factor | 1.514 |
| Sharpe Ratio | 2.586 |
| Win Rate | 48.3% |
| Expectancy | $69.82 / trade |
| Net Profit | $73,662 |
| Max Drawdown | -6.23% |
| Total Trades | 1,055 (~3 signals / day) |

All 15 tickers are individually profitable. Results validated via Monte Carlo permutation test (p=0.0000, 10,000 permutations) and Combinatorial Purged Cross-Validation (15/15 paths profitable).

**Disclaimer:** Past performance does not guarantee future results. Backtest assumes $0.02/share slippage, zero commissions (Alpaca), and does not account for liquidity constraints or partial fills.

## Technical stack

| Component | Technology |
|---|---|
| Language | Python 3.12+ with full type annotations |
| Market data | Alpaca Markets API (1-min bars, WebSocket streaming) |
| Indicators | TA-Lib (batch), talipp (streaming), numpy, pandas |
| Backtesting | Backtesting.py with walk-forward OOS engine |
| Regime detection | Hidden Markov Model (hmmlearn) -- 3-state GaussianHMM trained on rolling 60-day windows |
| Adaptive stops | Kalman filter (filterpy) -- innovation-based stop sizing, widens in volatile regimes |
| Volume profile | Prior-day POC, Value Area (VAH/VAL), HVN/LVN levels |
| VIX term structure | VIX/VIX3M ratio for contango/backwardation regime tagging |
| Storage | SQLite (WAL mode) for bar data, DuckDB for analytics |
| Alerts | Telegram Bot API (python-telegram-bot) |
| Dashboard | Streamlit + Plotly |
| API | FastAPI + Uvicorn |
| Data models | Pydantic v2 (no raw dicts anywhere) |
| ML (tested, rejected) | LightGBM signal scorer -- AUC 0.487, no predictive value |

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

SPY, QQQ, MSFT, AMD, TSLA, AMZN, UBER, SMCI, SHOP, PLTR, NFLX, MSTR, SNOW, ARM, DASH

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
│   ├── storage/             # SQLite database layer
│   ├── backtest/            # Walk-forward engine, data loader, metrics
│   ├── api/                 # FastAPI endpoints
│   └── dashboard/           # Streamlit UI
├── scripts/                 # CLI tools (backtest, replay, backfill, etc.)
├── tests/                   # pytest suite
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

### Run

```bash
make run                    # Start the live signal platform
make dashboard              # Start Streamlit dashboard (separate terminal)
make backtest               # Run the walk-forward backtest
```

## Scripts

| Script | Description |
|---|---|
| `run_backtest.py` | Full 15-ticker walk-forward backtest with per-ticker summary, combined metrics, equity curve, and trade log CSV. |
| `replay_day.py` | Replay a single trading day to see what signals would have fired, with entry/stop/target and P&L outcome. |
| `backfill_data.py` | Download historical 1-min bars from Alpaca for one or more tickers. |
| `run_monte_carlo.py` | Monte Carlo permutation test (10,000 shuffles) to validate that backtest results are not due to chance. |
| `run_cpcv.py` | Combinatorial Purged Cross-Validation to test strategy robustness across all possible train/test path combinations. |
| `train_ml_scorer.py` | LightGBM signal scorer experiment with Optuna tuning and SHAP analysis. Tested and rejected (AUC 0.487). |
| `test_telegram.py` | Send a test message to verify Telegram bot configuration. |

## Development

```bash
make lint                   # Run ruff linter
make format                 # Auto-format code
make typecheck              # Run mypy
make test                   # Fast unit tests
make test-all               # All tests including slow/integration
make check                  # lint + typecheck + test
```
