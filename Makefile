# Makefile for protein-diffusion-design-lab
# Provides convenient shortcuts for common development tasks

.PHONY: help install install-dev test test-coverage lint format security clean docker-build docker-run docs

# Default target
help:
	@echo "Available targets:"
	@echo "  install      - Install package dependencies"
	@echo "  install-dev  - Install development dependencies"
	@echo "  test         - Run tests"
	@echo "  test-coverage - Run tests with coverage reporting"
	@echo "  lint         - Run linting checks"
	@echo "  format       - Format code with black and isort"
	@echo "  security     - Run security checks with bandit"
	@echo "  clean        - Clean build artifacts and cache"
	@echo "  docker-build - Build Docker image"
	@echo "  docker-run   - Run application in Docker"
	@echo "  docs         - Build documentation"

# Installation targets
install:
	pip install -e .

install-dev:
	pip install -e .
	pip install -r requirements-dev.txt
	pre-commit install

# Testing targets
test:
	pytest tests/ -v

test-coverage:
	pytest --cov=src --cov-report=term-missing --cov-report=html tests/

test-integration:
	pytest tests/integration/ -v -m integration

test-unit:
	pytest tests/unit/ -v -m "not slow"

test-gpu:
	pytest tests/ -v -m gpu

# Code quality targets
lint:
	black --check src/ tests/
	isort --check-only src/ tests/
	flake8 src/ tests/
	mypy src/

format:
	black src/ tests/
	isort src/ tests/

security:
	bandit -r src/ -f json -o bandit-report.json
	safety check --json --output safety-report.json || true

# Docker targets
docker-build:
	docker-compose build protein-diffusion

docker-run:
	docker-compose up protein-diffusion

docker-dev:
	docker-compose --profile dev up protein-diffusion-dev

docker-test:
	docker-compose --profile test run test

# Documentation targets
docs:
	cd docs && make html

docs-serve:
	cd docs/_build/html && python -m http.server 8000

# Cleanup targets
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .tox/

# CI/CD simulation
ci: lint security test-coverage
	@echo "CI pipeline completed successfully"

# Quick development workflow
dev-setup: install-dev
	@echo "Development environment setup complete"

# Performance testing
benchmark:
	pytest tests/ -v -m performance --benchmark-only