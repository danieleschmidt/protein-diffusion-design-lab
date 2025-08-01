# GitHub Actions CI/CD Setup

This document provides templates and configuration for setting up comprehensive GitHub Actions workflows.

## Required Workflows

### 1. Main CI/CD Pipeline

Create `.github/workflows/ci.yml`:

```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11]
        
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
        
    - name: Cache pip packages
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements*.txt') }}
        
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e .[dev]
        
    - name: Lint with flake8
      run: |
        flake8 src/ tests/ --count --select=E9,F63,F7,F82 --show-source --statistics
        flake8 src/ tests/ --count --max-complexity=10 --max-line-length=88 --statistics
        
    - name: Type check with mypy
      run: mypy src/
      
    - name: Security check with bandit
      run: bandit -r src/ -f json
      
    - name: Test with pytest
      run: |
        pytest tests/ --cov=src --cov-report=xml --cov-report=term-missing
        
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
        name: codecov-umbrella

  security:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Run GitGuardian scan
      uses: GitGuardian/ggshield-action@v1.20.0
      env:
        GITHUB_PUSH_BEFORE_SHA: ${{ github.event.before }}
        GITHUB_PUSH_BASE_SHA: ${{ github.event.base }}
        GITHUB_PULL_BASE_SHA: ${{ github.event.pull_request.base.sha }}
        GITHUB_DEFAULT_BRANCH: ${{ github.event.repository.default_branch }}
        GITGUARDIAN_API_KEY: ${{ secrets.GITGUARDIAN_API_KEY }}
        
    - name: Dependency vulnerability check
      run: |
        pip install safety
        safety check --json --output safety-report.json
        
  docker:
    runs-on: ubuntu-latest
    needs: [test, security]
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3
      
    - name: Login to DockerHub
      uses: docker/login-action@v3
      with:
        username: ${{ secrets.DOCKERHUB_USERNAME }}
        password: ${{ secrets.DOCKERHUB_TOKEN }}
        
    - name: Build and push
      uses: docker/build-push-action@v5
      with:
        context: .
        push: true
        tags: |
          yourusername/protein-diffusion:latest
          yourusername/protein-diffusion:${{ github.sha }}
        cache-from: type=gha
        cache-to: type=gha,mode=max
```

### 2. Automated Dependency Updates

Create `.github/workflows/dependency-update.yml`:

```yaml
name: Dependency Updates

on:
  schedule:
    - cron: '0 2 * * 1'  # Weekly on Monday
  workflow_dispatch:

jobs:
  update-dependencies:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
      with:
        token: ${{ secrets.GITHUB_TOKEN }}
        
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: 3.9
        
    - name: Install pip-tools
      run: pip install pip-tools
      
    - name: Update requirements
      run: |
        pip-compile --upgrade requirements.in
        pip-compile --upgrade requirements-dev.in
        
    - name: Create Pull Request
      uses: peter-evans/create-pull-request@v5
      with:
        token: ${{ secrets.GITHUB_TOKEN }}
        commit-message: "chore: update dependencies"
        title: "Automated dependency updates"
        body: |
          This PR updates dependencies to their latest versions.
          
          Please review the changes and ensure all tests pass.
        branch: automated/dependency-updates
        delete-branch: true
```

### 3. Release Automation

Create `.github/workflows/release.yml`:

```yaml
name: Release

on:
  push:
    tags:
      - 'v*'

jobs:
  release:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: 3.9
        
    - name: Install build dependencies
      run: |
        python -m pip install --upgrade pip
        pip install build twine
        
    - name: Build package
      run: python -m build
      
    - name: Publish to PyPI
      env:
        TWINE_USERNAME: __token__
        TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
      run: twine upload dist/*
      
    - name: Create GitHub Release
      uses: softprops/action-gh-release@v1
      with:
        files: dist/*
        generate_release_notes: true
```

## Setup Instructions

1. **Create directories**: `mkdir -p .github/workflows`
2. **Copy workflow files** to `.github/workflows/`
3. **Configure secrets** in repository settings:
   - `DOCKERHUB_USERNAME`
   - `DOCKERHUB_TOKEN`
   - `GITGUARDIAN_API_KEY`
   - `PYPI_API_TOKEN`
4. **Enable Dependabot** in repository settings
5. **Configure branch protection** rules for main branch

## Branch Protection Rules

Recommended settings for main branch:
- Require pull request reviews (1 reviewer minimum)
- Require status checks to pass (CI/CD workflow)
- Require branches to be up to date before merging
- Restrict pushes to users with admin access
- Allow force pushes: disabled
- Allow deletions: disabled

## Monitoring and Alerts

Consider adding:
- **Codecov** for coverage reporting
- **Dependabot** for automated dependency updates
- **CodeQL** for advanced security analysis
- **Performance regression** testing in CI