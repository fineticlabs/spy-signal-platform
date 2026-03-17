---
description: Python code style standards — always loaded
globs:
---

# Python Code Style

- Type hints required on ALL functions (parameters and return type).
- `from __future__ import annotations` at the top of every Python file.
- Google-style docstrings on all public functions and classes. Private functions need docstrings only if non-obvious.
- Import order: stdlib -> third-party -> local. Ruff enforces this (`isort` rules enabled).
- Absolute imports only: `from src.models import Bar`. Never use relative imports.
- No mutable default arguments. Use `None` + assign inside the function body.
- Use `pathlib.Path` over `os.path` for all file path operations.
- Use f-strings over `.format()` or `%` formatting.
- Use `Decimal` for all price/money values. Never use `float` for financial data.
- All data structures are Pydantic `BaseModel` — no raw dicts for structured data.
- Constants in `UPPER_SNAKE_CASE`. Classes in `PascalCase`. Functions/methods in `snake_case`.
- Private functions/methods prefixed with `_`.
- Max function length ~50 lines. If longer, refactor into helper functions.
- Prefer early returns over deeply nested conditionals.
