# CI Workflow Template

Template for continuous integration workflow at `.github/workflows/ci.yml`

## Workflow Configuration

```yaml
name: CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]
  workflow_dispatch:

env:
  PYTHON_VERSION: "3.9"
  POETRY_VERSION: "1.6.1"

jobs:
  code-quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          
      - name: Cache dependencies
        uses: actions/cache@v3
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements*.txt') }}
          
      - name: Install dependencies
        run: |
          pip install -e ".[dev]"
          
      - name: Run black
        run: black --check src/ tests/
        
      - name: Run isort
        run: isort --check-only src/ tests/
        
      - name: Run flake8
        run: flake8 src/ tests/
        
      - name: Run mypy
        run: mypy src/

  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          
      - name: Install dependencies
        run: pip install bandit safety
        
      - name: Run bandit
        run: bandit -r src/ -f json -o bandit-report.json
        
      - name: Run safety
        run: safety check --json --output safety-report.json
        
      - name: Upload security reports
        uses: actions/upload-artifact@v3
        with:
          name: security-reports
          path: "*-report.json"

  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
        python-version: ["3.9", "3.10", "3.11"]
        exclude:
          - os: windows-latest
            python-version: "3.11"
          - os: macos-latest  
            python-version: "3.11"
            
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
          
      - name: Install dependencies
        run: |
          pip install -e ".[dev]"
          
      - name: Run unit tests
        run: |
          pytest tests/unit/ -v --cov=src/protein_diffusion --cov-report=xml
          
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
          flags: unittests
          name: codecov-${{ matrix.os }}-py${{ matrix.python-version }}

  integration-tests:
    runs-on: ubuntu-latest
    needs: [code-quality, security]
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          
      - name: Install dependencies
        run: pip install -e ".[dev]"
        
      - name: Run integration tests
        run: pytest tests/integration/ -v -m integration
        
  gpu-tests:
    runs-on: [self-hosted, gpu]  # Requires self-hosted GPU runner
    if: github.event_name != 'pull_request' || contains(github.event.pull_request.labels.*.name, 'gpu-tests')
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          
      - name: Install dependencies
        run: pip install -e ".[dev]"
        
      - name: Run GPU tests
        run: pytest tests/ -v -m gpu
```

## Key Features

- **Multi-OS testing**: Ubuntu, Windows, macOS
- **Multi-Python testing**: 3.9, 3.10, 3.11
- **Comprehensive linting**: black, isort, flake8, mypy
- **Security scanning**: bandit, safety
- **Coverage reporting**: pytest-cov with CodeCov integration
- **GPU testing**: Conditional GPU tests with self-hosted runners

## Required Secrets

- `CODECOV_TOKEN`: For coverage reporting

## Implementation Notes

1. Adjust Python versions based on project requirements
2. Configure self-hosted GPU runner for ML model testing
3. Add custom pytest markers as needed
4. Consider adding performance benchmarking jobs

## Troubleshooting

**Issue**: GPU tests fail on PR from forks
**Solution**: GPU tests only run on main branch pushes and labeled PRs

**Issue**: Windows tests fail due to path issues
**Solution**: Use cross-platform path handling in test fixtures

**Issue**: Coverage reports are inconsistent
**Solution**: Ensure all test runs upload coverage with unique flags