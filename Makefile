# Makefile for protein-diffusion-design-lab

.PHONY: help install install-dev test test-unit test-integration test-e2e test-gpu test-performance lint format security clean docs build

# Default target
help:
	@echo "Available commands:"
	@echo "  install        Install production dependencies"
	@echo "  install-dev    Install development dependencies"
	@echo "  test           Run all tests"
	@echo "  test-unit      Run unit tests only"
	@echo "  test-integration Run integration tests"
	@echo "  test-e2e       Run end-to-end tests"
	@echo "  test-gpu       Run GPU tests"
	@echo "  test-performance Run performance benchmarks"
	@echo "  lint           Run all linting checks"
	@echo "  format         Format code with black and isort"
	@echo "  security       Run security scans"
	@echo "  clean          Clean build artifacts"
	@echo "  docs           Build documentation"
	@echo "  build          Build distribution packages"

# Installation targets
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pre-commit install

# Test targets
test:
	pytest

test-unit:
	pytest tests/unit/ -m "unit or not (slow or integration or e2e or gpu or performance)"

test-integration:
	pytest tests/integration/ -m integration

test-e2e:
	pytest tests/e2e/ -m e2e

test-gpu:
	pytest -m gpu

test-performance:
	pytest tests/performance/ -m performance --tb=no

test-fast:
	pytest -m "not (slow or gpu or performance)" --tb=line

# Code quality targets
lint:
	flake8 src/ tests/
	mypy src/
	bandit -r src/

format:
	black src/ tests/
	isort src/ tests/

format-check:
	black --check src/ tests/
	isort --check-only src/ tests/

# Security targets
security:
	bandit -r src/ -f json -o .bandit-report.json
	safety check --json --output .safety-report.json
	pip-audit --format=json --output=.pip-audit-report.json

# Cleanup targets
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	rm -rf .tox/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Documentation targets
docs:
	cd docs && make html

docs-clean:
	cd docs && make clean

# Build targets
build: clean
	python -m build

build-wheel:
	python -m build --wheel

build-sdist:
	python -m build --sdist

# Development targets
dev-setup: install-dev
	@echo "Development environment ready!"
	@echo "Run 'make test' to verify installation"

# CI targets (for GitHub Actions)
ci-test:
	pytest --cov=src/protein_diffusion --cov-report=xml --junitxml=pytest.xml

ci-lint:
	black --check src/ tests/
	isort --check-only src/ tests/
	flake8 src/ tests/
	mypy src/

ci-security:
	bandit -r src/
	safety check
	pip-audit