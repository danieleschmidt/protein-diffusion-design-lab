#!/bin/bash
# Security scanning script for protein-diffusion-design-lab
set -euo pipefail

echo "🔒 Running comprehensive security scan..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Create reports directory
mkdir -p reports/security

# Function to print status
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 1. Run Bandit for Python security issues
print_status "Running Bandit security scan..."
if command -v bandit &> /dev/null; then
    bandit -r src/ -f json -o reports/security/bandit-report.json --config bandit.yaml || {
        print_warning "Bandit found security issues - check reports/security/bandit-report.json"
    }
    bandit -r src/ -f txt --config bandit.yaml || true
else
    print_error "Bandit not installed. Run: pip install bandit"
    exit 1
fi

# 2. Run Safety for dependency vulnerabilities
print_status "Running Safety dependency scan..."
if command -v safety &> /dev/null; then
    safety check --json --output reports/security/safety-report.json --policy-file .safety-policy.json || {
        print_warning "Safety found vulnerabilities - check reports/security/safety-report.json"
    }
    safety check --policy-file .safety-policy.json || true
else
    print_error "Safety not installed. Run: pip install safety"
    exit 1
fi

# 3. Check for secrets with GitLeaks (if available)
print_status "Checking for secrets..."
if command -v gitleaks &> /dev/null; then
    gitleaks detect --source . --report-format json --report-path reports/security/gitleaks-report.json || {
        print_warning "GitLeaks found potential secrets - check reports/security/gitleaks-report.json"
    }
else
    print_warning "GitLeaks not available - install from: https://github.com/zricethezav/gitleaks"
fi

# 4. Check file permissions
print_status "Checking file permissions..."
find . -type f -perm /022 -not -path "./.git/*" -not -path "./.*" | while read -r file; do
    print_warning "File has overly permissive permissions: $file"
done

# 5. Check for hardcoded secrets (simple regex)
print_status "Scanning for potential hardcoded secrets..."
{
    echo "# Potential Secrets Scan Results"
    echo "Generated on: $(date)"
    echo ""
    
    # API keys
    grep -r -i "api[_-]key\s*=" src/ 2>/dev/null | head -10 || true
    
    # Passwords
    grep -r -i "password\s*=" src/ 2>/dev/null | head -10 || true
    
    # Tokens
    grep -r -i "token\s*=" src/ 2>/dev/null | head -10 || true
    
    # Database URLs
    grep -r -i "database.*://" src/ 2>/dev/null | head -10 || true
    
} > reports/security/secrets-scan.txt

# 6. Generate summary report
print_status "Generating security summary..."
{
    echo "# Security Scan Summary"
    echo "Generated on: $(date)"
    echo ""
    
    echo "## Bandit Results"
    if [[ -f reports/security/bandit-report.json ]]; then
        python3 -c "
import json
try:
    with open('reports/security/bandit-report.json') as f:
        data = json.load(f)
    print(f'Total issues: {len(data.get(\"results\", []))}')
    severity_counts = {}
    for result in data.get('results', []):
        severity = result.get('issue_severity', 'UNKNOWN')
        severity_counts[severity] = severity_counts.get(severity, 0) + 1
    for severity, count in severity_counts.items():
        print(f'{severity}: {count}')
except Exception as e:
    print(f'Error reading Bandit report: {e}')
        "
    fi
    
    echo ""
    echo "## Safety Results"
    if [[ -f reports/security/safety-report.json ]]; then
        python3 -c "
import json
try:
    with open('reports/security/safety-report.json') as f:
        data = json.load(f)
    vulnerabilities = data.get('vulnerabilities', [])
    print(f'Total vulnerabilities: {len(vulnerabilities)}')
    if vulnerabilities:
        print('Affected packages:')
        for vuln in vulnerabilities[:5]:  # Show first 5
            print(f'  - {vuln.get(\"package_name\", \"unknown\")}: {vuln.get(\"vulnerability_id\", \"unknown\")}')
except Exception as e:
    print(f'Error reading Safety report: {e}')
        "
    fi
    
    echo ""
    echo "## Recommendations"
    echo "1. Review all HIGH and CRITICAL severity issues"
    echo "2. Update vulnerable dependencies"
    echo "3. Remove any hardcoded secrets"
    echo "4. Consider implementing additional security measures"
    
} > reports/security/summary.md

print_status "Security scan complete! Check reports/security/ for detailed results."

# Exit with error if critical issues found
if [[ -f reports/security/bandit-report.json ]]; then
    critical_issues=$(python3 -c "
import json
try:
    with open('reports/security/bandit-report.json') as f:
        data = json.load(f)
    critical = sum(1 for r in data.get('results', []) if r.get('issue_severity') == 'HIGH')
    print(critical)
except:
    print(0)
    ")
    
    if [[ $critical_issues -gt 0 ]]; then
        print_error "Found $critical_issues critical security issues!"
        exit 1
    fi
fi

print_status "No critical security issues found ✅"