# Development Guide

This guide provides comprehensive information for developers working on the protein-diffusion-design-lab project.

## Quick Start

### Prerequisites

- Python 3.9+
- CUDA 11.0+ (for GPU support)
- Git
- 16GB+ RAM (32GB recommended for development)

### Development Environment Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/protein-diffusion-design-lab.git
cd protein-diffusion-design-lab

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Set up pre-commit hooks
pre-commit install

# Verify installation
make test-fast
```

## Project Structure

```
protein-diffusion-design-lab/
├── src/
│   └── protein_diffusion/          # Main package
│       ├── __init__.py
│       ├── core/                   # Core business logic
│       ├── models/                 # ML models
│       ├── data/                   # Data handling
│       ├── utils/                  # Utilities
│       └── cli/                    # Command-line interface
├── tests/                          # Test suite
│   ├── unit/                       # Unit tests
│   ├── integration/                # Integration tests
│   ├── e2e/                       # End-to-end tests
│   └── performance/               # Performance benchmarks
├── docs/                          # Documentation
├── config/                        # Configuration files
├── scripts/                       # Utility scripts
└── examples/                      # Usage examples
```

## Development Workflow

### 1. Feature Development

```bash
# Create feature branch
git checkout -b feature/new-feature-name

# Make changes and commit frequently
git add .
git commit -m "feat: add new feature"

# Push and create PR
git push origin feature/new-feature-name
```

### 2. Code Quality Checks

```bash
# Run all quality checks
make lint

# Format code
make format

# Run security scans
make security
```

### 3. Testing Strategy

```bash
# Run fast tests during development
make test-fast

# Run specific test categories
make test-unit           # Unit tests only
make test-integration    # Integration tests
make test-performance    # Performance benchmarks

# Run all tests before PR
make test
```

## Code Standards

### Python Style Guide

We follow PEP 8 with some modifications:

- **Line length**: 88 characters (Black default)
- **Imports**: isort with Black profile
- **Type hints**: Required for all public functions
- **Docstrings**: Google style docstrings

### Example Function

```python
from typing import List, Optional
import torch

def generate_protein_scaffold(
    motif: str,
    num_samples: int = 100,
    temperature: float = 0.8,
    device: Optional[torch.device] = None,
) -> List[ProteinStructure]:
    """Generate protein scaffolds using diffusion model.
    
    Args:
        motif: Target structural motif (e.g., "HELIX_SHEET_HELIX")
        num_samples: Number of scaffolds to generate
        temperature: Sampling temperature for diversity control
        device: PyTorch device (auto-detected if None)
        
    Returns:
        List of generated protein structures
        
    Raises:
        ValueError: If motif format is invalid
        RuntimeError: If model fails to generate samples
        
    Example:
        >>> scaffolds = generate_protein_scaffold(
        ...     motif="HELIX_SHEET_HELIX",
        ...     num_samples=10,
        ...     temperature=0.8
        ... )
        >>> len(scaffolds)
        10
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    # Implementation here
    pass
```

### Error Handling

```python
# Use specific exception types
class ProteinDiffusionError(Exception):
    """Base exception for protein diffusion errors."""
    pass

class InvalidMotifError(ProteinDiffusionError):
    """Raised when motif format is invalid."""
    pass

class ModelLoadError(ProteinDiffusionError):
    """Raised when model fails to load."""
    pass

# Proper error handling
try:
    result = risky_operation()
except SpecificError as e:
    logger.error(f"Operation failed: {e}")
    raise
```

## Testing Guidelines

### Test Structure

```python
import pytest
from unittest.mock import Mock, patch
from protein_diffusion import ProteinDiffuser

class TestProteinDiffuser:
    """Test suite for ProteinDiffuser class."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.diffuser = ProteinDiffuser()
        
    def test_initialization(self):
        """Test diffuser initializes correctly."""
        assert self.diffuser is not None
        assert hasattr(self.diffuser, 'model')
        
    @pytest.mark.parametrize("temperature", [0.1, 0.5, 1.0])
    def test_temperature_values(self, temperature):
        """Test different temperature values."""
        result = self.diffuser.generate(
            motif="HELIX", 
            temperature=temperature
        )
        assert result is not None
        
    @pytest.mark.gpu
    def test_gpu_acceleration(self):
        """Test GPU-accelerated generation."""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")
            
        # GPU-specific test logic
        pass
        
    @pytest.mark.slow
    def test_large_batch_generation(self):
        """Test generation with large batch sizes."""
        # Long-running test
        pass
```

### Mock Usage

```python
# Mock external dependencies
@patch('protein_diffusion.external.AutoDockWrapper')
def test_docking_integration(self, mock_autodock):
    """Test docking integration with mocked AutoDock."""
    mock_autodock.return_value.dock.return_value = -12.5
    
    result = self.ranker.calculate_affinity(structure, target)
    assert result == -12.5
    mock_autodock.return_value.dock.assert_called_once()
```

## Performance Optimization

### Profiling

```python
# Use cProfile for performance analysis
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Code to profile
result = expensive_operation()

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)
```

### Memory Management

```python
# Efficient tensor operations
import torch

# Use context managers for GPU memory
with torch.cuda.device(0):
    # GPU operations
    pass

# Clear cache explicitly
torch.cuda.empty_cache()

# Use memory-efficient implementations
def memory_efficient_operation(large_tensor):
    """Example of memory-efficient tensor operation."""
    # Process in chunks to avoid OOM
    chunk_size = 1000
    results = []
    
    for i in range(0, len(large_tensor), chunk_size):
        chunk = large_tensor[i:i+chunk_size]
        result = process_chunk(chunk)
        results.append(result)
        
        # Clear intermediate results
        del chunk
        
    return torch.cat(results)
```

## Configuration Management

### Environment Variables

```bash
# Development environment
export PROTEIN_DIFFUSION_LOG_LEVEL=DEBUG
export PROTEIN_DIFFUSION_CACHE_DIR=./dev_cache
export PROTEIN_DIFFUSION_MODEL_PATH=./dev_models

# Testing environment
export PROTEIN_DIFFUSION_TESTING=true
export PROTEIN_DIFFUSION_USE_MOCK_MODELS=true
```

### Configuration Files

```yaml
# config/development.yaml
model:
  checkpoint_path: "dev_weights/small_model.ckpt"
  device: "cpu"  # Use CPU for development
  batch_size: 4

logging:
  level: "DEBUG"
  format: "detailed"

cache:
  enabled: true
  max_size: "1GB"
```

## Debugging

### Logging Setup

```python
import logging
from protein_diffusion.utils import get_logger

# Get configured logger
logger = get_logger(__name__)

# Use structured logging
logger.info(
    "Generated scaffolds",
    extra={
        "motif": motif,
        "num_samples": len(scaffolds),
        "generation_time": elapsed_time
    }
)
```

### Debug Configuration

```python
# Debug mode settings
import torch

# Enable anomaly detection
torch.autograd.set_detect_anomaly(True)

# Set deterministic behavior for debugging
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Manual seed for reproducibility
torch.manual_seed(42)
```

## Documentation

### Code Documentation

```python
class ProteinStructure:
    """Represents a protein structure with sequence and coordinates.
    
    This class provides a unified interface for protein structure
    data, supporting various file formats and manipulation operations.
    
    Attributes:
        sequence: Amino acid sequence string
        coordinates: 3D atomic coordinates
        metadata: Additional structure information
        
    Example:
        >>> structure = ProteinStructure.from_pdb("protein.pdb")
        >>> print(structure.sequence)
        MKWVTFISLLLLFSSAYSRGV...
    """
    
    def __init__(self, sequence: str, coordinates: torch.Tensor):
        """Initialize protein structure.
        
        Args:
            sequence: Amino acid sequence
            coordinates: Atomic coordinates tensor [N_atoms, 3]
        """
        self.sequence = sequence
        self.coordinates = coordinates
```

### Adding Examples

```python
# examples/basic_usage.py
"""Basic usage example for protein-diffusion-design-lab."""

from protein_diffusion import ProteinDiffuser, AffinityRanker

def main():
    """Demonstrate basic protein design workflow."""
    # Initialize components
    diffuser = ProteinDiffuser()
    ranker = AffinityRanker()
    
    # Generate scaffolds
    scaffolds = diffuser.generate(
        motif="HELIX_SHEET_HELIX",
        num_samples=10
    )
    
    # Rank by affinity
    ranked = ranker.rank(scaffolds, target="target.pdb")
    
    # Save top candidates
    for i, protein in enumerate(ranked[:3]):
        protein.to_pdb(f"candidate_{i}.pdb")

if __name__ == "__main__":
    main()
```

## Continuous Integration

### Local CI Simulation

```bash
# Simulate CI pipeline locally
make ci-lint        # Linting checks
make ci-test        # Test execution
make ci-security    # Security scans

# Build wheel locally
make build-wheel
```

### GitHub Actions Integration

The project uses GitHub Actions for CI/CD. Key workflows:

- **CI**: Runs on every push and PR
- **Security**: Daily vulnerability scans
- **Release**: Automated releases on version tags
- **Docs**: Documentation deployment

## Troubleshooting

### Common Development Issues

**Issue**: Import errors after installation
```bash
# Solution: Reinstall in development mode
pip uninstall protein-diffusion-design-lab
pip install -e ".[dev]"
```

**Issue**: GPU out of memory during development
```python
# Solution: Use smaller batch sizes
config.model.batch_size = 2  # Reduce from default 32
torch.cuda.empty_cache()     # Clear GPU cache
```

**Issue**: Tests fail with model loading errors
```bash
# Solution: Use mock models for testing
export PROTEIN_DIFFUSION_USE_MOCK_MODELS=true
make test
```

### Getting Help

1. **Documentation**: Check docs/ directory
2. **Issues**: Search GitHub issues
3. **Discussions**: Use GitHub discussions for questions
4. **Contributing**: See CONTRIBUTING.md for guidelines

## Release Process

### Version Management

```bash
# Update version in pyproject.toml
# Create release notes in CHANGELOG.md
# Tag release
git tag v0.2.0
git push origin v0.2.0

# GitHub Actions will handle the rest
```

This development guide should help you get productive quickly while maintaining code quality and consistency.