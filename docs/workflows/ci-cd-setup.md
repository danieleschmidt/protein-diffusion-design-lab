# CI/CD Workflow Setup Guide

This document provides comprehensive GitHub Actions workflows for the protein-diffusion-design-lab project.

## Overview

The CI/CD pipeline includes:
- **Continuous Integration**: Automated testing, linting, and security scanning
- **Continuous Deployment**: Automated releases and container publishing
- **Quality Gates**: Code coverage, security checks, and performance monitoring

## Required GitHub Actions Workflows

### 1. Main CI Workflow (`.github/workflows/ci.yml`)

```yaml
name: CI Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - run: pip install -r requirements-dev.txt
      - run: make lint

  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - run: pip install bandit safety
      - run: make security

  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.9', '3.10', '3.11']
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
      - run: pip install -r requirements-dev.txt
      - run: make test-coverage
      - uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml

  gpu-test:
    runs-on: self-hosted
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4
      - run: make test-gpu
```

### 2. Release Workflow (`.github/workflows/release.yml`)

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
      - uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - run: pip install build twine
      - run: python -m build
      - uses: pypa/gh-action-pypi-publish@release/v1
        with:
          password: ${{ secrets.PYPI_API_TOKEN }}
      - uses: actions/create-release@v1
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        with:
          tag_name: ${{ github.ref }}
          release_name: Release ${{ github.ref }}
          draft: false
          prerelease: false
```

### 3. Container Build Workflow (`.github/workflows/docker.yml`)

```yaml
name: Docker Build

on:
  push:
    branches: [ main ]
    tags: [ 'v*' ]
  pull_request:
    branches: [ main ]

jobs:
  docker:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: docker/setup-buildx-action@v3
      - uses: docker/login-action@v3
        if: github.event_name != 'pull_request'
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      - uses: docker/build-push-action@v5
        with:
          context: .
          push: ${{ github.event_name != 'pull_request' }}
          tags: |
            ghcr.io/${{ github.repository }}:latest
            ghcr.io/${{ github.repository }}:${{ github.sha }}
```

## Setup Instructions

### 1. Create Workflow Files

Copy the above workflows to `.github/workflows/` directory in your repository.

### 2. Configure Secrets

Add the following secrets to your GitHub repository:

- `PYPI_API_TOKEN`: For PyPI package publishing
- `CODECOV_TOKEN`: For code coverage reporting (optional)

### 3. Self-Hosted Runner Setup (for GPU tests)

```bash
# On your GPU-enabled machine
curl -o actions-runner-linux-x64-2.311.0.tar.gz -L https://github.com/actions/runner/releases/download/v2.311.0/actions-runner-linux-x64-2.311.0.tar.gz
tar xzf ./actions-runner-linux-x64-2.311.0.tar.gz
./config.sh --url https://github.com/YOUR_ORG/protein-diffusion-design-lab --token YOUR_TOKEN
sudo ./svc.sh install
sudo ./svc.sh start
```

### 4. Branch Protection Rules

Configure the following branch protection rules for `main`:

- Require pull request reviews
- Require status checks to pass before merging:
  - `lint`
  - `security`
  - `test (3.9)`
  - `test (3.10)`
  - `test (3.11)`
- Require branches to be up to date before merging
- Restrict pushes that create files larger than 100MB

## Quality Gates

### Code Coverage
- Minimum 80% coverage required
- Coverage reports uploaded to Codecov
- Coverage checks fail if coverage drops

### Security Scanning
- Bandit for Python security issues
- Safety for dependency vulnerabilities
- Secret scanning enabled

### Performance Testing
- Benchmark tests run on releases
- Performance regression detection
- GPU utilization monitoring

## Deployment Strategy

### Development Environment
- Automatic deployment to staging on `develop` branch
- Integration tests run against staging

### Production Environment
- Manual approval required for production deployment
- Blue-green deployment strategy
- Rollback capability maintained

## Monitoring and Alerts

### Build Status
- Slack notifications for failed builds
- Email alerts for security issues
- Dashboard for build metrics

### Performance Monitoring
- Model inference time tracking
- Memory usage monitoring
- GPU utilization alerts

## Troubleshooting

### Common Issues

1. **GPU Tests Failing**
   - Check self-hosted runner status
   - Verify CUDA installation
   - Check GPU memory availability

2. **Security Scan False Positives**
   - Add exceptions to `bandit.yaml`
   - Update Safety policy
   - Document security decisions

3. **Slow Build Times**
   - Enable build caching
   - Optimize Docker layers
   - Use matrix builds efficiently

### Debug Commands

```bash
# Local CI simulation
make ci

# Debug specific workflow
act -W .github/workflows/ci.yml

# Check workflow syntax
github-workflow-validator .github/workflows/
```

## Metrics and KPIs

Track the following metrics:
- Build success rate (target: >95%)
- Average build time (target: <15 minutes)
- Test coverage (target: >80%)
- Security vulnerability count (target: 0 high/critical)
- Deployment frequency (target: daily for dev, weekly for prod)