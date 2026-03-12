# SPY Intraday Signal Platform

## What This Project Is
Personal-use intraday SPY trading signal and alerting system.
**NOT an auto-trading bot.** Generates alerts for manual execution only.
Discretionary trader assistant — analyze, score, alert. Never place orders.

## Tech Stack (Do Not Deviate)
- **Python 3.12+** with full type annotations everywhere
- **asyncio** event loop for all real-time processing
- **Alpaca API** (`alpaca-py`) for WebSocket market data + REST historical
- **TA-Lib** for batch indicator computation (backtesting)
- **talipp** for streaming incremental indicator updates (live)
- **SQLite** with WAL mode for operational storage
- **DuckDB** for analytical queries and backtesting
- **FastAPI** for internal API layer (serves dashboard + logs)
- **Streamlit** for dashboard UI
- **Telegram Bot API** (`python-telegram-bot`) for alerts
- **Pydantic v2** for ALL data models — no raw dicts anywhere
- **Ruff** for linting + formatting
- **pytest** for tests, **pytest-asyncio** for async tests

## Architecture
```
Alpaca WS → AsyncIngestion → BarBuffer → IndicatorEngine(talipp)
                                              ↓
                                    LevelTracker (VWAP, ORB, PDH/PDL)
                                              ↓
                                    StrategyEngine (ORB, VWAP pullback)
                                              ↓
                                    RegimeFilter (VIX, ADX)
                                              ↓
                                    SignalScorer (confluence)
                                              ↓
                                    RiskManager (pre-trade gate)
                                              ↓
                                    AlertDispatcher (Telegram)
                                              ↓
                                    SQLite (trade log) + FastAPI → Streamlit
```

## Repo Structure
```
spy-signal-platform/
├── CLAUDE.md              # You are reading this
├── pyproject.toml         # All deps, ruff, mypy, pytest config
├── Makefile               # Common commands
├── .env.example           # Required env vars template
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── main.py            # Asyncio orchestrator entry point
│   ├── config.py          # Pydantic Settings from .env
│   ├── models.py          # Shared Pydantic models (Bar, Signal, Alert, etc.)
│   ├── ingestion/         # Alpaca WS + REST data fetching
│   │   ├── __init__.py
│   │   ├── websocket.py   # Real-time bar streaming
│   │   ├── historical.py  # Backfill historical bars
│   │   └── bar_buffer.py  # Multi-timeframe bar aggregation
│   ├── indicators/        # Indicator computation
│   │   ├── __init__.py
│   │   ├── streaming.py   # talipp-based live indicators
│   │   ├── batch.py       # TA-Lib-based batch indicators
│   │   └── registry.py    # Indicator registry/factory
│   ├── levels/            # Key price levels
│   │   ├── __init__.py
│   │   ├── vwap.py        # VWAP + deviation bands
│   │   ├── opening_range.py # ORB high/low
│   │   ├── daily_levels.py  # PDH/PDL/PDC, premarket H/L
│   │   └── dynamic.py     # HOD/LOD tracking
│   ├── strategies/        # Strategy implementations
│   │   ├── __init__.py
│   │   ├── base.py        # Abstract strategy interface
│   │   ├── orb.py         # Opening Range Breakout
│   │   ├── vwap_pullback.py # VWAP pullback continuation
│   │   └── regime.py      # Regime detection (VIX + ADX)
│   ├── signals/           # Signal scoring engine
│   │   ├── __init__.py
│   │   ├── scorer.py      # Confluence scoring
│   │   └── explainer.py   # Human-readable signal explanations
│   ├── risk/              # Risk management
│   │   ├── __init__.py
│   │   ├── manager.py     # Pre-trade risk checks
│   │   ├── position_sizing.py # Fixed fractional sizing
│   │   └── cooldown.py    # Loss cooldown / tilt detection
│   ├── alerts/            # Notification dispatch
│   │   ├── __init__.py
│   │   ├── telegram.py    # Telegram bot integration
│   │   ├── formatter.py   # Alert message formatting
│   │   └── dispatcher.py  # Multi-channel dispatch
│   ├── storage/           # Database layer
│   │   ├── __init__.py
│   │   ├── database.py    # SQLite connection + migrations
│   │   ├── models.py      # SQLAlchemy/raw SQL table definitions
│   │   └── queries.py     # Named query functions
│   ├── api/               # FastAPI endpoints
│   │   ├── __init__.py
│   │   └── routes.py      # REST endpoints for dashboard
│   ├── dashboard/         # Streamlit UI
│   │   ├── __init__.py
│   │   └── app.py         # Main dashboard
│   └── backtest/          # Backtesting framework
│       ├── __init__.py
│       ├── engine.py      # Backtesting.py wrappers
│       ├── data_loader.py # Load historical bars for backtest
│       └── metrics.py     # Performance metrics calculation
├── config/
│   └── settings.yaml      # Strategy parameters, thresholds
├── tests/
│   ├── conftest.py        # Shared fixtures
│   ├── test_indicators.py
│   ├── test_levels.py
│   ├── test_strategies.py
│   ├── test_risk.py
│   └── test_signals.py
├── scripts/
│   ├── backfill_data.py   # One-time historical data download
│   ├── run_backtest.py    # CLI backtest runner
│   └── test_telegram.py   # Verify Telegram bot works
└── docs/
    ├── DESIGN.md           # Full design document
    └── STRATEGIES.md       # Strategy documentation
```

## Coding Standards (ENFORCE THESE ALWAYS)

### Type Safety
- Every function has full type annotations including return types
- Use `from __future__ import annotations` at top of every file
- All data structures are Pydantic BaseModel, never raw dicts
- Use `Decimal` for all price/money values, never float
- Use `datetime` with timezone-aware UTC everywhere

### Naming
- Classes: PascalCase (e.g., `BarBuffer`, `SignalScorer`)
- Functions/methods: snake_case (e.g., `calculate_vwap`)
- Constants: UPPER_SNAKE (e.g., `MAX_DAILY_TRADES`)
- Private: prefix with underscore (e.g., `_validate_bar`)
- Files: snake_case matching primary class/function

### Error Handling
- Never use bare `except:` — always catch specific exceptions
- All external API calls wrapped in try/except with structured logging
- Use custom exception classes in `src/exceptions.py`
- Fail loudly during development, gracefully in production

### Logging
- Use `structlog` for all logging (JSON structured output)
- Every module gets its own logger: `logger = structlog.get_logger(__name__)`
- Log levels: DEBUG for indicator values, INFO for signals, WARNING for risk rejections, ERROR for failures
- Include context in every log: timestamp, symbol, timeframe, indicator values

### Async Patterns
- Use `asyncio.Queue` for inter-component messaging
- Never use `time.sleep()` — always `asyncio.sleep()`
- Use `asyncio.TaskGroup` for concurrent operations
- Graceful shutdown via signal handlers (SIGINT, SIGTERM)

### Testing
- Every public function has at least one test
- Use `pytest-asyncio` for async tests
- Use `freezegun` or manual datetime injection for time-dependent tests
- Test indicators against known calculated values
- Test strategies against known bar sequences with expected outcomes

## MVP Strategy Rules

### Strategy 1: Opening Range Breakout (ORB)
- Opening range = first 5 completed 1-min bars (9:30-9:35 ET)
- LONG: price breaks above ORB high with volume > 1.5x 20-bar avg
- SHORT: price breaks below ORB low with volume > 1.5x 20-bar avg
- Stop: 1.5x ATR(14) on 5-min bars from entry
- Target: 2x risk distance (2R)
- Filters: VIX < 25, ADX(14) > 20 on 15-min, not lunch chop (11:30-13:30)
- Exit: hit target, hit stop, or 15:55 ET (forced flat)

### Strategy 2: VWAP Pullback (build after ORB is validated)
- Bias: determined by 15-min trend (price above/below VWAP + EMA20)
- Setup: price pulls back to VWAP ± 0.5 ATR zone on 5-min
- Trigger: rejection candle (hammer/engulfing) on 1-min at VWAP zone
- Stop: below/above VWAP by 1x ATR
- Target: previous HOD/LOD or 2R, whichever is closer
- Filters: same as ORB + RSI not extreme (30-70 range on 5-min)

### Regime Filter (gates ALL strategies)
- VIX < 15: low vol — use ORB on narrow ranges, expect small moves
- VIX 15-25: normal — both strategies active, full sizing
- VIX > 25: high vol — reduce to half size OR sit out entirely
- ADX < 15 on 15-min: choppy regime — no trend trades, consider mean reversion only
- ADX > 25 on 15-min: trending — favor breakout/continuation setups

## Risk Rules (NON-NEGOTIABLE)
- Max 1% account risk per trade
- Max 3% daily loss → stop trading for the day
- Max 5 trades per day
- After 2 consecutive losses → 15-min mandatory cooldown
- After 3 consecutive losses → done for the day
- No new trades after 15:45 ET
- All positions flat by 15:55 ET
- Minimum 1.5:1 reward-to-risk or signal is rejected
- Lunch chop zone (11:30-13:30 ET) → no new trades unless confluence > 4/5

## Common Mistakes to Avoid
- DO NOT use `pandas-ta` (supply chain compromised) — use TA-Lib or talipp
- DO NOT use float for prices — use Decimal
- DO NOT compute indicators on incomplete bars
- DO NOT reference bar close price before bar is complete (lookahead bias)
- DO NOT use bare `except:` blocks
- DO NOT store secrets in code — always from .env
- DO NOT build multiple strategies simultaneously — finish and validate one first
- DO NOT skip tests "to move faster"
- DO NOT use global state — pass dependencies explicitly or use DI
