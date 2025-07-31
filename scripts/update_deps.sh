#!/bin/bash
set -e

# Dependency update script for protein-diffusion-design-lab

echo "📦 Updating dependencies for Protein Diffusion Design Lab..."

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "🔧 Activating virtual environment..."
    source venv/bin/activate
fi

# Backup current dependencies
echo "💾 Backing up current dependencies..."
pip freeze > requirements_backup_$(date +%Y%m%d_%H%M%S).txt

# Update pip first
echo "📈 Updating pip..."
pip install --upgrade pip

# Update setuptools and wheel
echo "🛠️ Updating build tools..."
pip install --upgrade setuptools wheel

# Dry run to check for conflicts
echo "🔍 Checking for dependency conflicts..."
pip-compile --dry-run requirements.txt || echo "⚠️ Potential conflicts detected"

# Update all dependencies
echo "🔄 Updating all dependencies..."
pip install --upgrade -r requirements.txt

# Update development dependencies
echo "🛠️ Updating development dependencies..."
pip install --upgrade -r requirements-dev.txt

# Reinstall package in development mode
echo "🔧 Reinstalling package..."
pip install -e ".[dev]"

# Update pre-commit hooks
echo "🔒 Updating pre-commit hooks..."
pre-commit autoupdate
pre-commit install

# Run security audit
echo "🔒 Running security audit..."
pip-audit --desc --output pip-audit-report.txt || echo "⚠️ Security issues found - check pip-audit-report.txt"

# Run tests to ensure everything still works
echo "🧪 Running tests to verify updates..."
if make test-fast; then
    echo "✅ Tests passed after update"
else
    echo "❌ Tests failed after update - review changes"
    exit 1
fi

# Generate new requirements files with pinned versions
echo "📝 Generating updated requirements files..."
pip freeze > requirements_updated.txt

# Show differences
echo "📊 Dependency changes:"
if command -v diff &> /dev/null; then
    diff requirements_backup_*.txt requirements_updated.txt || echo "Differences shown above"
fi

# Cleanup
rm requirements_updated.txt

echo ""
echo "✅ Dependency update complete!"
echo ""
echo "📋 What was updated:"
echo "  • Core Python packages"
echo "  • Development tools"
echo "  • Pre-commit hooks"
echo "  • Security patches"
echo ""
echo "📝 Next steps:"
echo "  1. Review pip-audit-report.txt for security issues"
echo "  2. Test thoroughly before committing changes"
echo "  3. Update CI/CD if any major version changes"
echo "  4. Document any breaking changes"