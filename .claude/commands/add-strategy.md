Add a new trading strategy to the platform. The strategy must:

1. Inherit from the base Strategy class in src/strategies/base.py
2. Define clear entry conditions, exit conditions, stop logic, and target logic
3. Include a regime filter check (VIX/ADX)
4. Include time-of-day filter (no lunch chop, no late-day entries)
5. Output a Signal Pydantic model with: direction, entry_price, stop_price, target_price, confidence_score, reason_text, invalidation_conditions
6. Have at least 3 unit tests: one for valid long entry, one for valid short entry, one for filter rejection
7. Be registered in the strategy engine so the orchestrator picks it up
8. Include a docstring with: plain-English description, when it works, when it fails, required indicators

Strategy to add: $ARGUMENTS
