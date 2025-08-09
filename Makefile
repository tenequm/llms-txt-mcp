# llms-txt-mcp Development Makefile

.PHONY: help check fix ci format lint type test test-fast clean install

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'

check: ## Run all checks (format, lint, test)
	@uv run ruff format --check .
	@uv run ruff check .
	@uv run pytest

fix: ## Auto-fix format and lint issues, then run checks
	@uv run ruff format .
	@uv run ruff check --fix .

ci: ## Run checks in CI mode (verbose output)
	@uv run ruff format --check .
	@uv run ruff check .
	@uv run pytest -v

format: ## Format code with ruff
	@uv run ruff format .

lint: ## Lint code with ruff
	@uv run ruff check .

type: ## Type check with mypy
	@uv run mypy src/

test: ## Run all tests
	@uv run pytest

test-fast: ## Run tests without slow/performance tests
	@uv run pytest -m "not performance"

clean: ## Clean cache and temporary files
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true

install: ## Install all dependencies
	@uv sync --all-extras

# Default target
.DEFAULT_GOAL := help