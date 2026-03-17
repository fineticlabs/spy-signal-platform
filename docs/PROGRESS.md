# Project Progress Tracker

## Current State (March 2026)
- **Tickers**: 26 (SPY, QQQ, MSFT, AMD, TSLA, AMZN, UBER, SMCI, SHOP, PLTR, NFLX, MSTR, SNOW, ARM, DASH, PYPL, INTC, MU, HOOD, DKNG, SOXL, ROKU, TQQQ, BA, MRVL, META)
- **Strategy**: ORB breakout with 11 confluence filters + HMM regime + Kalman adaptive stops
- **Backtest Results**: PF 1.423, Sharpe 2.270, ~$117K net P&L, 1981 trades (walk-forward OOS)
- **Test Suite**: 410+ tests, all passing
- **Automation**: launchd scheduling active (6:25 AM PT start, 1:05 PM PT stop, weekdays only)
- **Alerts**: Telegram bot with daily summary reports (win/loss breakdown, P&L)

## Completed Build Sessions

### Core Platform (Sessions 1-9)
1. Project scaffolding + data ingestion (Alpaca WS + REST, SQLite storage)
2. Indicator engine (talipp streaming + TA-Lib batch, EMA/RSI/MACD/BB/ATR/VWAP)
3. Key price levels (VWAP bands, ORB high/low, PDH/PDL/PDC, HOD/LOD)
4. ORB strategy implementation (entry/exit logic, regime filter, time filters)
5. Signal scoring engine (confluence scoring + human-readable explanations)
6. Risk management (position sizing, cooldown/tilt, pre-trade gate, 1% risk)
7. Telegram alerts (formatted messages, multi-channel dispatch)
8. FastAPI + Streamlit dashboard (trade log, equity curve, signal history)
9. Backtesting framework (Backtesting.py wrapper, walk-forward engine)

### Improvements (Sessions 10-21)
10. Walk-forward OOS validation with rolling windows
11. HMM regime detection (3-state: calm/normal/volatile) on 5-min bars
12. Kalman filter adaptive stop sizing (innovation-based multiplier)
13. Volume profile (POC, VAH/VAL, HVN/LVN) as confluence filter
14. VIX term structure filter (contango/backwardation from VIX/VIX3M ratio)
15. Economic calendar blocking (FOMC, NFP, CPI, PPI days)
16. Earnings blackout filter (yfinance-based, +1 day window)
17. Candlestick pattern filters (engulfing, hammer, doji compression)
18. Failed breakout detection (ORB break then reversal)
19. Ticker expansion: 8 -> 15 -> 26 tickers with per-ticker walk-forward
20. Monte Carlo simulation + CPCV cross-validation
21. ML feature extraction (SHAP importance, gradient boosted scorer)

### Operational (Sessions 22-24)
22. launchd automation (auto_start.sh, auto_stop.sh, install/uninstall scripts)
23. Live scanner crash fix (DataFeed enum, not string, for Alpaca WS)
24. Exhaustive test suite (410+ tests across 15 test files)

## Lessons Learned (Gotchas)
- `TimeFrame` enum must be used for all `db.query_bars()` calls — strings cause `'str' has no attribute 'value'` crash
- `DataFeed` enum must be used for `StockDataStream(feed=...)` — same crash pattern
- SPY/QQQ have no earnings dates in yfinance — "may be delisted" warning is benign
- HMM training produces RuntimeWarnings (divide by zero, overflow) — benign, suppress them
- MSFT window 3 HMM covariance failure — falls back to default regime gracefully
- Monday trades have PF 1.125 — too weak, filtered out
- 60-day IS window is optimal — longer dilutes regime signal, shorter overfits
- SOXL/TQQQ 3x leverage requires reduced position sizing
- Kalman multiplier must be clamped [0.90, 1.50] — unclamped values cause stop blow-outs
- Volume profile with tick_size < 0.05 is prohibitively slow on 5+ year backtests
- Walk-forward must process tickers independently — cross-ticker regime leakage is real

## Next Steps
- [ ] Live observation period: run scanner for 2 weeks, compare signals to manual trades
- [ ] Account scaling: validate risk rules at $100K, $250K account sizes
- [ ] VWAP pullback strategy: build after ORB is validated in live
- [ ] Gap fade strategy: morning gap reversal on high-RVOL days
- [ ] Multi-timeframe confirmation: 15-min trend alignment for 1-min entries
- [ ] Broker integration research: evaluate Alpaca paper trading API for signal verification
