# SPY Intraday Signal Platform

Python 3.12 intraday ORB trading signal platform. 26 tickers, Alpaca API data, Telegram alerts, SQLite storage. **NOT an auto-trading bot** — generates alerts for manual execution only.

## Commands
```bash
make run                                    # Start live scanner
make dashboard                              # Launch Streamlit dashboard
pytest tests/ -v --tb=short                 # Run full test suite (1,478 tests)
ruff check src/ tests/ && ruff format src/ tests/  # Lint + format
python scripts/run_backtest.py              # Walk-forward backtest (all tickers)
python scripts/replay_day.py --date DATE    # Replay a single trading day
python scripts/backfill_data.py             # Backfill historical bars from Alpaca
```

## Architecture
```
src/
├── main.py              # Asyncio orchestrator (TaskGroup: WS, bar loop, signal loop, API)
├── config.py            # Pydantic Settings (.env) — AppSettings, AlpacaSettings, RiskSettings
├── models.py            # Bar, Signal, TimeFrame, Direction, Regime (all Pydantic)
├── ingestion/           # Alpaca WS streaming + REST historical + bar aggregation
├── indicators/          # talipp (live streaming) + TA-Lib (batch backtest)
├── levels/              # VWAP, ORB, daily levels, HOD/LOD, Kalman adaptive stops
├── strategies/          # ORB breakout, HMM regime, candlestick filters, failed breakout
├── filters/             # Economic calendar, earnings blackout, VIX term structure
├── signals/             # Confluence scorer + human-readable explainer
├── risk/                # Position sizing (1% risk), cooldown/tilt, pre-trade gate
├── alerts/              # Telegram dispatch + message formatting
├── storage/             # SQLite (WAL mode) + named query functions
├── backtest/            # Walk-forward engine, volume profile, monte carlo, ML features
├── api/                 # FastAPI internal endpoints
└── dashboard/           # Streamlit UI
```

## Standards
- Type hints on ALL functions (params + return). `from __future__ import annotations` in every file.
- Google-style docstrings on public functions and classes.
- Absolute imports only (`from src.models import Bar`, never relative).
- `structlog` for all logging — never `print()`. Per-module: `logger = structlog.get_logger(__name__)`.
- All timestamps UTC-aware. All prices as `Decimal`. Pydantic models for all data — no raw dicts.
- Max function length ~50 lines. No bare `except:`. No mutable default args. `pathlib` over `os.path`.
- Conventional commits: `feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `chore:`.
- Never commit `.env`, API keys, or `.pkl` model files.
- Run `ruff check` + `pytest` before every commit.

## Gotchas (Project-Specific Landmines)
- **TimeFrame enum required**: `db.query_bars(timeframe=TimeFrame.ONE_MIN)` — never pass string `"1Min"`.
- **DataFeed enum required**: `StockDataStream(feed=DataFeed.IEX)` — never pass string `"iex"`. See `_FEED_MAP` in `websocket.py`.
- **SPY/QQQ earnings warning**: No earnings dates in yfinance — "symbol may be delisted" warning is expected and benign.
- **HMM RuntimeWarnings**: `divide by zero`, `overflow` during training are benign — suppress with `warnings.filterwarnings`.
- **MSFT window 3 HMM covars failure**: Known issue, handled gracefully (falls back to default regime 1.0).
- **SOXL/TQQQ are 3x leveraged ETFs**: Position sizing must account for 3x leverage (reduce size).
- **Monday filter is ON**: No trades on Mondays (PF 1.125 too weak). Enforced in backtest engine.
- **Walk-forward 60-day IS window**: Intentional — captures current regime, not diluted history.
- **Kalman stop multiplier clamped [0.90, 1.50]**: Values outside this range indicate bugs.
- **ORB = first 5 bars (9:30-9:35 ET)**: Not configurable without backtest re-validation.
- **Signal cutoff 10:00 ET**: Default matches backtest window (9:35-10:00). Configurable via `signal_cutoff_et`.
- **Lunch chop 11:30-13:30 ET**: No new trades unless confluence > 4/5 (only applies if cutoff extended past 10:00).

## Risk Rules (Non-Negotiable)
- Max 1% account risk per trade ($500 on $50K), scaled by 0.25x position_scale_factor
- Max 3% daily loss -> stop trading for the day
- Max $500 loss per symbol per day -> block new entries for that symbol
- Max 5 trades per day, max 3 concurrent positions (enforced)
- 2 consecutive losses -> 15-min cooldown; 3 consecutive -> done for the day
- Cooldown state persisted to DB (survives scanner restarts)
- No new signals after 10:00 ET (configurable signal_cutoff_et); all flat by 15:55 ET
- Minimum 1.5:1 reward-to-risk or signal is rejected
