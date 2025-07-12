.PHONY: help install install-dev test test-unit test-integration test-benchmark lint format type-check security clean build upload pre-commit setup-hooks

# Python executable from virtual environment
PYTHON = /Users/ajpri/Summer/AI_Audio/.venv/bin/python

# Default target
help:
	@echo "Available commands:"
	@echo "  install        Install package dependencies"
	@echo "  install-dev    Install package and development dependencies"
	@echo "  test           Run all tests"
	@echo "  test-unit      Run unit tests only"
	@echo "  test-integration Run integration tests only"
	@echo "  test-benchmark Run benchmark tests"
	@echo "  lint           Run code linting"
	@echo "  format         Format code with black"
	@echo "  type-check     Run type checking with mypy"
	@echo "  security       Run security checks"
	@echo "  clean          Clean up generated files"
	@echo "  build          Build package"
	@echo "  pre-commit     Run pre-commit hooks"
	@echo "  setup-hooks    Install pre-commit hooks"

# Installation
install:
	$(PYTHON) -m pip install -r requirements.txt

install-dev:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -r requirements-dev.txt

# Testing
test:
	$(PYTHON) -m pytest tests/ -v

test-unit:
	$(PYTHON) -m pytest tests/ -v -m "unit"

test-integration:
	$(PYTHON) -m pytest tests/ -v -m "integration"

test-benchmark:
	$(PYTHON) -m pytest tests/benchmark.py -v -s

test-coverage:
	$(PYTHON) -m pytest tests/ --cov=ai_audio_detector --cov-report=html --cov-report=term-missing

# Code quality
lint:
	$(PYTHON) -m flake8 .
	$(PYTHON) -m bandit -r ai_audio_detector/ example_usage.py --skip B110,B112,B311,B404,B602

format:
	$(PYTHON) -m black .

type-check:
	$(PYTHON) -m mypy ai_audio_detector/ --ignore-missing-imports

security:
	$(PYTHON) -m bandit -r ai_audio_detector/ example_usage.py --skip B110,B112,B311 --severity-level medium
	@echo "⚠️  Safety scan skipped (requires authentication)"
	@echo "To run safety scan, install and authenticate: pip install safety && safety auth login"

# Pre-commit
pre-commit:
	pre-commit run --all-files

setup-hooks:
	pre-commit install
	pre-commit install --hook-type pre-push

# Cleanup
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .tox/

# Building
build: clean
	$(PYTHON) -m build

build-check:
	$(PYTHON) -m build
	$(PYTHON) -m twine check dist/*

# CI/CD simulation
ci-test: install-dev lint type-check security test

# Development workflow
dev-setup: install-dev setup-hooks
	@echo "Development environment setup complete!"
	@echo "Run 'make test' to verify everything works."

# Quick development checks
quick-check: format lint test-unit

# Full checks before commit
full-check: format lint type-check security test
