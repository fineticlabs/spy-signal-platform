Run a backtest for the specified strategy. Steps:

1. Load historical 1-min SPY bars from SQLite (or fetch from Alpaca if not cached)
2. Resample to required timeframes (5-min, 15-min)
3. Compute all indicators needed by the strategy
4. Run the strategy through Backtesting.py with:
   - Slippage: $0.02 round-trip per share
   - No lookahead bias (signals execute on next-bar-open)
   - Walk-forward: 60-day in-sample, 20-day out-of-sample
5. Output metrics: win rate, profit factor, max drawdown, expectancy, Sharpe, avg winner, avg loser, total trades
6. Save equity curve plot to docs/
7. Compare results to buy-and-hold baseline

Strategy to backtest: $ARGUMENTS
