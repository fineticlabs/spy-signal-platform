---
description: Testing standards and practices — always loaded
globs:
---

# Testing Standards

- Every new public function needs at least one test. No exceptions.
- Mock all external APIs (Alpaca, Telegram, yfinance) — never call real APIs in tests.
- Use fixtures in `tests/conftest.py` for common setup (`sample_bar`, `sample_signal`, `tmp_db`, `make_bar()`, `make_1min_df()`).
- Test edge cases: empty DataFrames, NaN values, single-bar data, zero volume, timezone-naive inputs.
- Use `pytest-asyncio` for async tests. Use `freezegun` or manual datetime injection for time-dependent tests.
- Test indicators against known hand-calculated values.
- Test strategies against known bar sequences with expected outcomes.
- Run `ruff check src/ tests/` and `pytest tests/ -v --tb=short` before declaring any work complete.
- Use `monkeypatch` for env vars in config tests — never read real `.env` in tests.
- Naming: `test_<function>_<scenario>` (e.g., `test_calculate_vwap_single_bar_equals_typical_price`).
- Group related tests in classes: `class TestCalculateVwap:`.
- No `# noqa` suppressions without a comment explaining why.
