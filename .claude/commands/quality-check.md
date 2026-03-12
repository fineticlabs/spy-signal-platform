Run the full quality check suite on the codebase:

1. Run `ruff check src/ tests/` — fix any linting errors
2. Run `ruff format src/ tests/` — ensure consistent formatting
3. Run `mypy src/` — fix any type errors
4. Run `pytest -x tests/` — ensure all tests pass
5. Check that no .env secrets are committed (grep for API keys)
6. Check that all new functions have type annotations
7. Check that all Pydantic models use Decimal for prices, not float

Report results and fix any issues found. Do not skip any step.
