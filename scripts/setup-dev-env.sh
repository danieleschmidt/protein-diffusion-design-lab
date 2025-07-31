#!/bin/bash
# Development environment setup script for protein-diffusion-design-lab
set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Check if running from project root
if [[ ! -f "pyproject.toml" ]]; then
    log_error "Please run this script from the project root directory"
    exit 1
fi

log_info "Setting up development environment for Protein Diffusion Design Lab"

# Step 1: Check Python version
log_step "Checking Python version..."
python_version=$(python3 --version 2>&1 | grep -oE '3\.[0-9]+' || echo "unknown")
if [[ "$python_version" < "3.9" ]]; then
    log_error "Python 3.9 or higher is required. Found: $python_version"
    exit 1
fi
log_info "Python version: $python_version ✓"

# Step 2: Check for CUDA (optional)
log_step "Checking for CUDA..."
if command -v nvidia-smi &> /dev/null; then
    cuda_version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -1)
    log_info "CUDA detected: Driver version $cuda_version ✓"
    export CUDA_AVAILABLE=1
else
    log_warn "CUDA not detected. GPU acceleration will not be available."
    export CUDA_AVAILABLE=0
fi

# Step 3: Create virtual environment
log_step "Creating virtual environment..."
if [[ ! -d "venv" ]]; then
    python3 -m venv venv
    log_info "Virtual environment created ✓"
else
    log_info "Virtual environment already exists ✓"
fi

# Step 4: Activate virtual environment
log_step "Activating virtual environment..."
source venv/bin/activate || {
    log_error "Failed to activate virtual environment"
    exit 1
}
log_info "Virtual environment activated ✓"

# Step 5: Upgrade pip and install build tools
log_step "Upgrading pip and installing build tools..."
pip install --upgrade pip setuptools wheel
log_info "Build tools updated ✓"

# Step 6: Install development dependencies
log_step "Installing development dependencies..."
pip install -r requirements-dev.txt
log_info "Development dependencies installed ✓"

# Step 7: Install package in development mode
log_step "Installing package in development mode..."
pip install -e .
log_info "Package installed in development mode ✓"

# Step 8: Setup pre-commit hooks
log_step "Setting up pre-commit hooks..."
pre-commit install --hook-type pre-commit --hook-type pre-push
pre-commit install --hook-type commit-msg
log_info "Pre-commit hooks installed ✓"

# Step 9: Create necessary directories
log_step "Creating project directories..."
mkdir -p logs data outputs weights reports/security
log_info "Project directories created ✓"

# Step 10: Setup Git LFS (if available)
log_step "Setting up Git LFS..."
if command -v git-lfs &> /dev/null; then
    git lfs install
    git lfs track "*.ckpt" "*.pth" "*.h5" "*.hdf5" "*.pkl"
    log_info "Git LFS configured ✓"
else
    log_warn "Git LFS not available. Large model files won't be tracked efficiently."
fi

# Step 11: Setup Docker environment
log_step "Checking Docker setup..."
if command -v docker &> /dev/null; then
    if docker info &> /dev/null; then
        log_info "Docker is available and running ✓"
        
        # Check for Docker Compose
        if command -v docker-compose &> /dev/null; then
            log_info "Docker Compose is available ✓"
        else
            log_warn "Docker Compose not found. Install for full container support."
        fi
        
        # Check for NVIDIA Docker (if CUDA available)
        if [[ "$CUDA_AVAILABLE" == "1" ]]; then
            if docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi &> /dev/null; then
                log_info "NVIDIA Docker support verified ✓"
            else
                log_warn "NVIDIA Docker not properly configured. GPU containers may not work."
            fi
        fi
    else
        log_warn "Docker daemon not running. Start Docker for container support."
    fi
else
    log_warn "Docker not found. Install Docker for container support."
fi

# Step 12: Run initial tests
log_step "Running initial test suite..."
if python -m pytest tests/unit/ -v --tb=short; then
    log_info "Initial tests passed ✓"
else
    log_warn "Some tests failed. Review output above."
fi

# Step 13: Setup IDE configurations
log_step "Setting up IDE configurations..."

# VS Code settings
if [[ -d ".vscode" ]] || command -v code &> /dev/null; then
    mkdir -p .vscode
    cat > .vscode/settings.json << 'EOF'
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": false,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.sortImports.args": ["--profile", "black"],
    "[python]": {
        "editor.formatOnSave": true,
        "editor.codeActionsOnSave": {
            "source.organizeImports": true
        }
    },
    "files.exclude": {
        "**/__pycache__": true,
        "**/.pytest_cache": true,
        "**/.mypy_cache": true,
        "**/logs": true,
        "**/outputs": true
    }
}
EOF
    log_info "VS Code settings configured ✓"
fi

# Step 14: Create .env template
log_step "Creating environment template..."
cat > .env.template << 'EOF'
# Environment configuration template
# Copy to .env and fill in your values

# Model configuration
MODEL_PATH=weights/boltz-1b.ckpt
DEVICE=cuda  # or cpu
BATCH_SIZE=32

# Data paths
DATA_DIR=data
OUTPUT_DIR=outputs
LOG_DIR=logs

# API configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=false

# Monitoring
ENABLE_METRICS=true
METRICS_PORT=8080

# Security
SECRET_KEY=your-secret-key-here
ALLOWED_HOSTS=localhost,127.0.0.1
EOF
log_info "Environment template created ✓"

# Step 15: Display next steps
log_step "Development environment setup complete!"

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                    SETUP COMPLETE                             ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo "1. Copy .env.template to .env and configure your settings"
echo "2. Download model weights: python scripts/download_weights.py"
echo "3. Run tests: make test"
echo "4. Start development server: make docker-dev"
echo "5. Access docs: make docs && open docs/_build/html/index.html"
echo ""
echo -e "${BLUE}Useful commands:${NC}"
echo "• make help          - Show all available commands"
echo "• make format        - Format code with black and isort"
echo "• make lint          - Run linting checks"
echo "• make test          - Run test suite"
echo "• make security      - Run security scans"
echo "• make ci            - Run full CI pipeline locally"
echo ""
echo -e "${BLUE}Resources:${NC}"
echo "• Development guide: DEVELOPMENT.md"
echo "• Contributing guide: CONTRIBUTING.md"
echo "• API documentation: docs/"
echo "• Monitoring setup: monitoring/"
echo ""

# Deactivate virtual environment for clean state
deactivate || true

log_info "Development environment is ready! Activate with: source venv/bin/activate"