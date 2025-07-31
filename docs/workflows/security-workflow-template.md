# Security Workflow Template

Template for comprehensive security scanning at `.github/workflows/security.yml`

## Workflow Configuration

```yaml
name: Security Scan

on:
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM UTC
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
    paths:
      - 'requirements*.txt'
      - 'pyproject.toml'
      - 'Dockerfile'
  workflow_dispatch:

jobs:
  dependency-scan:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: "3.9"
          
      - name: Install dependencies
        run: |
          pip install safety pip-audit
          pip install -e .
          
      - name: Run safety check
        run: |
          safety check --json --output safety-report.json
          safety check  # Also run with stdout for workflow logs
        continue-on-error: true
        
      - name: Run pip-audit
        run: |
          pip-audit --format=json --output=pip-audit-report.json
          pip-audit  # Also run with stdout for workflow logs
        continue-on-error: true
        
      - name: Upload vulnerability reports
        uses: actions/upload-artifact@v3
        with:
          name: vulnerability-reports
          path: "*-report.json"

  sast-scan:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: "3.9"
          
      - name: Run Bandit security scan
        run: |
          pip install bandit[toml]
          bandit -r src/ -f json -o bandit-report.json
          bandit -r src/  # Also run with stdout for workflow logs
        continue-on-error: true
        
      - name: Run semgrep
        uses: returntocorp/semgrep-action@v1
        with:
          config: >-
            p/security-audit
            p/python
            p/bandit
        continue-on-error: true
        
      - name: Upload SAST reports
        uses: actions/upload-artifact@v3
        with:
          name: sast-reports
          path: bandit-report.json

  container-scan:
    runs-on: ubuntu-latest
    if: hashFiles('Dockerfile') != ''
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Build Docker image
        run: docker build -t protein-diffusion:scan .
        
      - name: Run Trivy vulnerability scanner
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: 'protein-diffusion:scan'
          format: 'sarif'
          output: 'trivy-results.sarif'
          
      - name: Upload Trivy scan results
        uses: github/codeql-action/upload-sarif@v2
        with:
          sarif_file: 'trivy-results.sarif'

  license-check:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: "3.9"
          
      - name: Install pip-licenses
        run: pip install pip-licenses
        
      - name: Install project dependencies
        run: pip install -e .
        
      - name: Check licenses
        run: |
          pip-licenses --format=json --output-file=licenses.json
          pip-licenses --fail-on="GPL v3;AGPL v3"  # Fail on copyleft licenses
          
      - name: Upload license report
        uses: actions/upload-artifact@v3
        with:
          name: license-report
          path: licenses.json

  sbom-generation:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: "3.9"
          
      - name: Install cyclonedx-bom
        run: pip install cyclonedx-bom
        
      - name: Install project dependencies
        run: pip install -e .
        
      - name: Generate SBOM
        run: |
          cyclonedx-py requirements -r requirements.txt -o sbom.json
          
      - name: Upload SBOM
        uses: actions/upload-artifact@v3
        with:
          name: sbom
          path: sbom.json
          
      - name: Attach SBOM to release
        if: startsWith(github.ref, 'refs/tags/')
        uses: softprops/action-gh-release@v1
        with:
          files: sbom.json

  secrets-scan:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0  # Full history for better secret detection
          
      - name: Run GitGuardian scan
        uses: GitGuardian/ggshield-action@v1.22.0
        env:
          GITHUB_PUSH_BEFORE_SHA: ${{ github.event.before }}
          GITHUB_PUSH_BASE_SHA: ${{ github.event.base }}
          GITHUB_PULL_BASE_SHA: ${{ github.event.pull_request.base.sha }}
          GITHUB_DEFAULT_BRANCH: ${{ github.event.repository.default_branch }}
          GITGUARDIAN_API_KEY: ${{ secrets.GITGUARDIAN_API_KEY }}

  supply-chain-scan:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Run Scorecard analysis
        uses: ossf/scorecard-action@v2.2.0
        with:
          results_file: results.sarif
          results_format: sarif
          publish_results: true
          
      - name: Upload Scorecard results
        uses: github/codeql-action/upload-sarif@v2
        with:
          sarif_file: results.sarif

  security-summary:
    runs-on: ubuntu-latest
    needs: [dependency-scan, sast-scan, license-check, secrets-scan]
    if: always()
    
    steps:
      - name: Download all artifacts
        uses: actions/download-artifact@v3
        
      - name: Create security summary
        run: |
          echo "# Security Scan Summary" >> $GITHUB_STEP_SUMMARY
          echo "Date: $(date)" >> $GITHUB_STEP_SUMMARY
          echo "" >> $GITHUB_STEP_SUMMARY
          
          # Add results from each scan
          if [ -f vulnerability-reports/safety-report.json ]; then
            echo "## Dependency Vulnerabilities" >> $GITHUB_STEP_SUMMARY
            # Parse and summarize safety report
          fi
          
          if [ -f sast-reports/bandit-report.json ]; then
            echo "## SAST Findings" >> $GITHUB_STEP_SUMMARY
            # Parse and summarize bandit report
          fi
```

## Key Features

- **Daily scheduled scans**: Automated security monitoring
- **Multi-tool approach**: safety, bandit, semgrep, trivy
- **SBOM generation**: Software Bill of Materials for supply chain security
- **License compliance**: Automated license checking
- **Secret detection**: GitGuardian integration
- **Supply chain security**: OSSF Scorecard analysis

## Required Secrets

- `GITGUARDIAN_API_KEY`: For secret scanning

## Configuration Files

Create `.bandit` configuration:

```yaml
# .bandit
exclude_dirs:
  - tests/
  - docs/

skips:
  - B101  # Skip assert_used test for test files
```

## Implementation Notes

1. Adjust license restrictions based on project policy
2. Configure SBOM generation for releases
3. Set up security issue templates for vulnerability reports
4. Consider adding security.txt file for responsible disclosure

## Troubleshooting

**Issue**: Too many false positives from bandit
**Solution**: Configure `.bandit` file to skip irrelevant checks

**Issue**: License check fails on development dependencies
**Solution**: Separate license checking for production vs dev dependencies

**Issue**: SBOM generation fails
**Solution**: Ensure all dependencies are properly installed before generation