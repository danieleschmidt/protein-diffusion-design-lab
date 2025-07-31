#!/bin/bash
set -e

# Development environment setup script for protein-diffusion-design-lab

echo "🧬 Setting up Protein Diffusion Design Lab development environment..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.9"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python $required_version or higher is required. Found: $python_version"
    exit 1
fi

echo "✅ Python version check passed: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "📈 Upgrading pip..."
pip install --upgrade pip

# Install development dependencies
echo "🛠️ Installing development dependencies..."
pip install -e ".[dev]"

# Install pre-commit hooks
echo "🔒 Setting up pre-commit hooks..."
pre-commit install

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p data/{raw,processed}
mkdir -p weights
mkdir -p outputs
mkdir -p logs

# Download sample data (placeholder)
echo "📊 Setting up sample data..."
# wget -O data/sample_protein.pdb "https://files.rcsb.org/download/1AKE.pdb" || echo "⚠️ Sample data download failed"

# Verify installation
echo "🧪 Verifying installation..."
if python -c "import protein_diffusion; print('✅ Package import successful')" 2>/dev/null; then
    echo "✅ Package installation verified"
else
    echo "❌ Package import failed - installation may be incomplete"
fi

# Run quick tests
echo "🧪 Running quick tests..."
if make test-fast > /dev/null 2>&1; then
    echo "✅ Quick tests passed"
else
    echo "⚠️ Some tests failed - check configuration"
fi

# Check GPU availability
echo "🎮 Checking GPU availability..."
if python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')" 2>/dev/null; then
    if python -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
        gpu_count=$(python -c "import torch; print(torch.cuda.device_count())")
        echo "✅ GPU support detected: $gpu_count device(s)"
    else
        echo "⚠️ CUDA not available - CPU-only mode"
    fi
else
    echo "❌ PyTorch import failed"
fi

# Check external tools (placeholder)
echo "🔧 Checking external tools..."

# Display helpful information
echo ""
echo "🎉 Development environment setup complete!"
echo ""
echo "📋 Next steps:"
echo "  1. Activate the environment: source venv/bin/activate"
echo "  2. Run tests: make test"
echo "  3. Start development server: streamlit run app.py"
echo "  4. Open in VS Code: code ."
echo ""
echo "🛠️ Available commands:"
echo "  make help           - Show all available commands"
echo "  make test           - Run all tests"
echo "  make test-fast      - Run quick tests"
echo "  make lint           - Check code quality"
echo "  make format         - Format code"
echo "  make security       - Run security scans"
echo ""
echo "📚 Documentation:"
echo "  docs/DEVELOPMENT.md - Development guide"
echo "  docs/ARCHITECTURE.md - System architecture"
echo "  README.md          - Project overview"