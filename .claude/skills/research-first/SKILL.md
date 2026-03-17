---
name: research-first
description: Research the codebase thoroughly before making any changes. Read docs, find related files, understand dependencies, then propose a plan.
---

# Research-First Workflow

Before writing ANY code, complete these steps in order:

## 1. Read Context
- Read `CLAUDE.md` at the project root.
- Read any subdirectory `CLAUDE.md` files relevant to the change.
- Read `docs/PROGRESS.md` for current project state and known issues.

## 2. Find Related Code
- Use `Grep` and `Glob` to find ALL files related to the change.
- Search for the function/class name, related imports, and callers.
- Check tests that exercise the code being changed.

## 3. Understand Dependencies
- Read the existing code and tests thoroughly.
- Identify all callers of functions being modified.
- Identify side effects (database writes, API calls, file I/O).
- Check if the change affects the backtest engine, live scanner, or both.

## 4. Present Plan
- Describe what will change and why.
- List files to be modified/created.
- Identify risks and edge cases.
- **Wait for user approval before writing any code.**

## 5. Implement (TDD)
- Write or update tests FIRST that define the expected behavior.
- Then implement the production code to make tests pass.
- Run `ruff check` and `ruff format` on changed files.

## 6. Verify
- Run `pytest tests/ -v --tb=short` — all tests must pass.
- Run `ruff check src/ tests/` — no lint errors.
- Summarize what changed and any follow-up work needed.
