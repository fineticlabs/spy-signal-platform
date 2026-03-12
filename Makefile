.PHONY: install dev lint format typecheck test test-all backfill run dashboard help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install production dependencies
	pip install -e .

dev: ## Install dev + backtest dependencies
	pip install -e ".[dev,backtest]"
	pre-commit install

lint: ## Run ruff linter
	ruff check src/ tests/ scripts/

format: ## Auto-format code with ruff
	ruff format src/ tests/ scripts/
	ruff check --fix src/ tests/ scripts/

typecheck: ## Run mypy type checker
	mypy src/

test: ## Run fast unit tests only
	pytest -m "not slow and not integration and not backtest" tests/

test-all: ## Run all tests including slow/integration
	pytest tests/

test-cov: ## Run tests with coverage report
	pytest --cov=src --cov-report=html tests/

backfill: ## Download historical SPY data
	python scripts/backfill_data.py

run: ## Start the live signal platform
	python -m src.main

dashboard: ## Start the Streamlit dashboard
	streamlit run src/dashboard/app.py

backtest: ## Run backtest for ORB strategy
	python scripts/run_backtest.py

test-telegram: ## Send a test alert to Telegram
	python scripts/test_telegram.py

clean: ## Remove build artifacts and caches
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

check: lint typecheck test ## Run all checks (lint + types + tests)
