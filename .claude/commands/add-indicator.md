Add a new indicator to the platform. The indicator must:

1. Have both a streaming version (using talipp in src/indicators/streaming.py) and a batch version (using TA-Lib in src/indicators/batch.py)
2. Be registered in src/indicators/registry.py
3. Have a Pydantic model for its output
4. Include at least 2 unit tests in tests/test_indicators.py verifying output against hand-calculated values
5. Use Decimal for any price-based values
6. Include docstring explaining: what it measures, whether lagging/leading, and best intraday use case

Indicator to add: $ARGUMENTS
