# SPY Intraday Signal Platform

Personal-use intraday SPY trading signal and alerting system. Analyzes the market, identifies high-quality setups, calculates levels, detects signals, scores confidence, and sends Telegram alerts. **Does NOT place trades automatically.**

## Quick Start

### Prerequisites
- Python 3.12+
- TA-Lib C library (`brew install ta-lib` on Mac, `apt install libta-lib-dev` on Linux)
- Alpaca account (free): https://app.alpaca.markets
- Telegram bot (free): message @BotFather on Telegram

### Setup
```bash
git clone <this-repo>
cd spy-signal-platform
cp .env.example .env
# Edit .env with your Alpaca and Telegram credentials

# Install dependencies
make dev

# Download historical data (last 90 days)
make backfill

# Verify Telegram is working
make test-telegram

# Run tests
make test
```

### Run
```bash
# Start the live signal platform
make run

# Start the dashboard (separate terminal)
make dashboard

# Run a backtest
make backtest
```

## Architecture
```
Alpaca WebSocket → Ingestion → Indicators → Levels → Strategy → Risk Gate → Telegram Alerts
                                                                              ↓
                                                                  FastAPI → Streamlit Dashboard
```

## MVP Strategies
1. **Opening Range Breakout (ORB)** — trades the break of the first 5-minute range with volume confirmation
2. **VWAP Pullback** — trades pullbacks to VWAP in trending conditions (v2)

## Risk Rules
- Max 1% account risk per trade
- Max 3% daily loss limit
- Max 5 trades per day
- 15-min cooldown after 2 consecutive losses
- No trades during lunch chop (11:30-1:30 ET)
- Flat by 3:55 PM ET

## Development
```bash
make lint       # Run linter
make format     # Auto-format code
make typecheck  # Run type checker
make test       # Run tests
make check      # All of the above
```
