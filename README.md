# SPY Intraday Signal Platform

![Python 3.12](https://img.shields.io/badge/python-3.12-blue)
![Tests](https://img.shields.io/badge/tests-1%2C478%20passed-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-97.8%25-brightgreen)

An intraday ORB (Opening Range Breakout) trading signal platform that monitors 26 US stocks and ETFs, detects breakout setups using statistical models, and places bracket orders on Alpaca's paper trading API. Includes HMM regime detection, Kalman adaptive stops, walk-forward validated backtest engine, and Telegram alerts.

## How it works

1. **Opening range** -- First 5 completed 1-min bars (9:30-9:35 ET) establish the high/low for each ticker.
2. **Breakout detection** -- Price must close beyond the ORB level for 2 consecutive bars with volume >= 1.5x the 20-bar average.
3. **Quality filters** -- Signal passes through 15+ filters (trend alignment, regime detection, volatility, gap classification, economic calendar).
4. **Alert and execution** -- Surviving signals are sent to Telegram and (in paper_trade mode) a bracket order is submitted to Alpaca.
5. **Exit** -- Target is 2R (2x risk distance). All positions flat by 3:55 PM ET.

## Backtest performance

Walk-forward OOS across 26 tickers, 5.6 years (2020-2026), $50K starting capital:

| Metric | Value |
|---|---|
| Profit Factor | 1.41 |
| Sharpe Ratio | 2.21 |
| Win Rate | 47.0% |
| Expectancy | $59 / trade |
| Max Drawdown | -6.2% |
| OOS Trades | 1,997 |

Validated via Monte Carlo permutation test (p=0.0000) and CPCV (15/15 paths profitable).

## Architecture

```
src/
├── main.py              # Asyncio orchestrator (TaskGroup: WS, bar loop, scheduler, API)
├── config.py            # Pydantic Settings from .env
├── models.py            # Bar, Signal, TimeFrame, Direction, Regime (all Pydantic)
├── ingestion/           # Alpaca WS streaming + REST historical + bar aggregation
├── indicators/          # talipp (live streaming) + TA-Lib (batch backtest)
├── levels/              # VWAP, ORB, daily levels, HOD/LOD, Kalman adaptive stops
├── strategies/          # ORB breakout, HMM regime, candlestick filters
├── filters/             # Economic calendar, earnings blackout, VIX term structure
├── signals/             # Confluence scorer + human-readable explainer
├── risk/                # Position sizing, cooldown/tilt, pre-trade risk gate
├── execution/           # Alpaca bracket order executor (paper/live)
├── alerts/              # Telegram dispatch + message formatting
├── storage/             # SQLite (WAL mode) + named query functions
├── backtest/            # Walk-forward engine, volume profile, Monte Carlo, CPCV
├── api/                 # FastAPI internal endpoints
└── dashboard/           # Streamlit UI
```

## Filters (live + backtest aligned)

| Filter | Type | Description |
|---|---|---|
| Trading window | Blocking | 9:35-10:00 ET only (configurable `signal_cutoff_et`) |
| 2-bar confirmation | Blocking | Requires 2 consecutive closes beyond ORB level |
| EMA trend alignment | Blocking | Long only above EMA(20), short only below |
| Gap classification | Blocking | Gap > +0.3%: long only; gap < -0.3%: short only |
| Daily ADX | Blocking | ADX(14) > 25 required (configurable `adx_min_threshold`) |
| VIX backwardation | Blocking | Blocks when VIX/VIX3M > 1.00 |
| ORB range minimum | Blocking | Range must be >= 0.15% of price |
| Volume confirmation | Blocking | Bar volume >= 1.5x 20-bar average |
| Monday exclusion | Blocking | No trades on Mondays (configurable `excluded_days`) |
| Economic calendar | Blocking | No trades on FOMC, NFP, CPI, PPI days |
| VIX level | Blocking | VIX < 25 required |
| Dynamic targets | Active | ORB range percentile adjusts R:R (1.5-2.5x) in backtest |
| Kalman adaptive stops | Active | Innovation-based multiplier (0.90-1.50x ATR) |
| HMM regime | Informational | Tags: HMM_VOLATILE (+1), HMM_CALM (-1 confidence) |
| SPY market direction | Informational | Tags: SPY_ALIGNED (+1), SPY_CONFLICT (-2) |

## Risk management

| Rule | Implementation |
|---|---|
| Position sizing | 1% account risk, 0.25x scale factor |
| Per-trade stop | 1.5x ATR(14) with Kalman adjustment |
| Per-symbol max loss | $500/day per symbol (configurable `max_symbol_loss`) |
| Portfolio daily max loss | 3% of equity |
| Max concurrent positions | 3 (enforced in risk manager) |
| Max trades per day | 5 |
| Cooldown | 15 min after 2 consecutive losses; done for day after 3 |
| Cooldown persistence | Saved to DB, survives scanner restarts |
| Bracket orders | Market entry + stop-loss + take-profit (all flat by 15:55 ET) |

## System hardening

| Feature | Description |
|---|---|
| Graceful shutdown | SIGTERM/SIGINT drains in-flight bars, runs reconciliation, sends Telegram summary |
| WS reconnect | Automatic re-subscription on disconnect with Telegram alerts |
| Atomic EOD flatten | Retry up to 3x per position, verify final state, per-symbol dedup clear |
| Startup sync | Loads both open positions AND pending orders into dedup set |
| Error logging | All `contextlib.suppress(Exception)` replaced with logged `try/except` |
| Bar queue overflow | Drops oldest bar on full queue instead of blocking WS callback |
| VIX fallback warning | One-time Telegram alert when VIX feed is unavailable |
| Preflight validation | 15 checks: Python, Alpaca, Telegram, DB, schema, account, WS, pipeline, tests |

## Quick start

### Prerequisites

- Python 3.12+
- TA-Lib C library (`brew install ta-lib`)
- [Alpaca account](https://app.alpaca.markets) (free paper trading)
- [Telegram bot](https://t.me/BotFather) (free)

### Setup

```bash
git clone <this-repo>
cd spy-signal-platform

python -m venv .venv
source .venv/bin/activate
make dev                    # Install all dependencies

cp .env.example .env        # Edit with your credentials
```

Required `.env` variables:
```
ALPACA_API_KEY=...
ALPACA_SECRET_KEY=...
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...
EXECUTION_MODE=paper_trade
ALPACA_EXPECTED_ACCOUNT=PA...   # optional, for preflight verification
```

### Validate

```bash
python scripts/preflight_check.py   # 15-check system validation
```

### Run

```bash
make run                    # Start live scanner
make dashboard              # Start Streamlit dashboard (separate terminal)
make backtest               # Run walk-forward backtest
```

### Automation (macOS)

```bash
bash scripts/install_launchd.sh     # Auto-start 9:25 ET, auto-stop 16:05 ET
bash scripts/uninstall_launchd.sh   # Remove automation
```

Manual start/stop:
```bash
bash scripts/auto_start.sh          # Backfill + start scanner
bash scripts/auto_stop.sh           # Flatten + reconcile + Telegram summary
```

## Execution modes

| Mode | Behavior |
|---|---|
| `alerts_only` | Telegram alerts only. No orders placed. |
| `paper_trade` | Alerts + bracket orders on Alpaca paper API. |
| `live_trade` | Alerts + bracket orders on Alpaca live API. |

## Tickers (26)

**Original 15:** SPY, QQQ, MSFT, AMD, TSLA, AMZN, UBER, SMCI, SHOP, PLTR, NFLX, MSTR, SNOW, ARM, DASH

**Expansion 11:** PYPL, INTC, MU, HOOD, DKNG, SOXL\*, ROKU, TQQQ\*, BA, MRVL, META

*\* 3x leveraged ETFs -- position sizing accounts for leverage*

## Scripts

| Script | Description |
|---|---|
| `preflight_check.py` | 15-check preflight validation (Python, Alpaca, Telegram, DB, schema, etc.) |
| `run_backtest.py` | Walk-forward backtest with per-ticker summary, equity curve, trade log |
| `replay_day.py` | Replay a single trading day with signal-by-signal P&L |
| `backfill_data.py` | Download historical 1-min bars from Alpaca |
| `backtest_week.py` | Backtest the current week's data |
| `check_paper_account.py` | Display paper account equity, positions, and orders |
| `run_monte_carlo.py` | Monte Carlo permutation test (10,000 shuffles) |
| `run_cpcv.py` | Combinatorial Purged Cross-Validation |
| `train_ml_scorer.py` | LightGBM scorer experiment (tested, rejected -- AUC 0.487) |
| `test_telegram.py` | Send a test message to verify Telegram config |
| `auto_start.sh` | Backfill + start scanner with PID tracking |
| `auto_stop.sh` | Stop scanner + EOD flatten + reconciliation + Telegram summary |
| `health_check.sh` | Verify scanner process, WS connection, bar flow, regime data |
| `install_launchd.sh` | Install macOS launchd plists for weekday scheduling |
| `uninstall_launchd.sh` | Remove launchd automation |
| `stop.sh` | Emergency stop (kill scanner process) |

## Test suite

```bash
make test                   # 1,478 tests, ~64s
make lint                   # ruff check + format
```

- **1,478 tests** across unit, integration, and parity tests
- Integration tests: pipeline wiring, bar flow, executor wiring, daily lifecycle, websocket watchdog
- Filter parity tests: verify live filters match backtest engine
- System hardening tests: cooldown persistence, schema validation, per-symbol loss, queue overflow

## Tech stack

| Component | Technology |
|---|---|
| Language | Python 3.12, full type annotations, `from __future__ import annotations` |
| Data | Alpaca Markets API (IEX/SIP websocket + REST) |
| Execution | alpaca-py `TradingClient` (bracket orders) |
| Indicators | talipp (streaming), TA-Lib (batch backtest) |
| Backtesting | backtesting.py with custom walk-forward engine |
| Regime | hmmlearn `GaussianHMM` (3-state, 60-day rolling) |
| Stops | filterpy `KalmanFilter` (innovation-based adaptive sizing) |
| Storage | SQLite (WAL mode) |
| Logging | structlog (structured JSON) |
| Alerts | Telegram Bot API |
| Models | Pydantic v2 (no raw dicts) |
| Scheduling | launchd (macOS) |
| Dashboard | Streamlit + Plotly |
| API | FastAPI + Uvicorn |

## Release history

| Version | Date | Description |
|---|---|---|
| v1.7.0 | 2026-03-24 | Filter parity + system hardening (Phase 1 + Phase 2) |
| v1.6.0 | 2026-03-21 | Production bug fixes (Monday filter, dedup, overnight positions, EOD reconciliation) |
| v1.5.0 | 2026-03-17 | Position scale factor, paper trading execution mode |
| v1.4.0 | 2026-03-16 | Ticker expansion to 26, per-symbol pipelines |

---

*Disclaimer: This is a personal trading tool, not financial advice. Past backtest performance does not guarantee future results. Use at your own risk.*
