# Contributing to Protein Diffusion Design Lab

We welcome contributions! This document provides guidelines for contributing to the project.

## 🚀 Quick Start

1. **Fork and clone** the repository
2. **Set up** your development environment
3. **Make changes** following our coding standards
4. **Submit** a pull request

## 📋 Development Setup

### Prerequisites

- Python 3.9+
- CUDA 11.0+ (for GPU support)
- Git

### Environment Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/protein-diffusion-design-lab.git
cd protein-diffusion-design-lab

# Create development environment
conda create -n protein-diffusion-dev python=3.9
conda activate protein-diffusion-dev

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

## 🛠️ Development Workflow

### Code Style

We use automated formatting and linting:

```bash
# Format code
black src/ tests/
isort src/ tests/

# Lint code
flake8 src/ tests/
mypy src/

# Run all quality checks
pre-commit run --all-files
```

### Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/protein_diffusion --cov-report=html

# Run specific test categories
pytest -m "not slow"        # Skip slow tests
pytest -m integration       # Run integration tests
pytest -m gpu               # Run GPU tests (requires CUDA)
```

### Documentation

```bash
# Build documentation locally
cd docs/
make html

# View at docs/_build/html/index.html
```

## 📝 Code Standards

### Python Code

- **PEP 8** compliance (enforced by black/flake8)
- **Type hints** for all public functions
- **Docstrings** in Google style
- **Maximum line length**: 88 characters

### Example Function

```python
def generate_protein_scaffold(
    motif: str,
    num_samples: int = 100,
    temperature: float = 0.8,
) -> List[ProteinStructure]:
    """Generate protein scaffolds using diffusion model.
    
    Args:
        motif: Target structural motif (e.g., "HELIX_SHEET_HELIX")
        num_samples: Number of scaffolds to generate
        temperature: Sampling temperature for diversity control
        
    Returns:
        List of generated protein structures
        
    Raises:
        ValueError: If motif format is invalid
        RuntimeError: If model fails to generate samples
    """
    # Implementation here
    pass
```

### Commit Messages

Use conventional commits format:

```
type(scope): brief description

Longer description if needed

- feat: new feature
- fix: bug fix
- docs: documentation changes
- style: formatting changes
- refactor: code restructuring
- test: adding tests
- chore: maintenance tasks
```

## 🧪 Testing Guidelines

### Test Categories

- **Unit tests**: Fast, isolated component tests
- **Integration tests**: Multi-component interaction tests  
- **GPU tests**: CUDA-dependent functionality tests
- **Slow tests**: Long-running validation tests

### Test Structure

```python
import pytest
from protein_diffusion import ProteinDiffuser

class TestProteinDiffuser:
    def test_initialization(self):
        """Test diffuser initializes correctly."""
        diffuser = ProteinDiffuser()
        assert diffuser is not None
        
    @pytest.mark.gpu
    def test_gpu_generation(self):
        """Test GPU-accelerated generation."""
        # GPU-specific test
        pass
        
    @pytest.mark.slow
    def test_large_batch_generation(self):
        """Test generation with large batch sizes."""
        # Long-running test
        pass
```

## 📚 Documentation

### README Updates

- Keep installation instructions current
- Update example code when APIs change
- Maintain performance benchmarks

### API Documentation

- Use Google-style docstrings
- Include code examples
- Document all parameters and return values
- Note any limitations or assumptions

## 🐛 Bug Reports

Use the issue template with:

- **Environment details** (Python version, CUDA version, OS)
- **Minimal reproduction case**
- **Expected vs actual behavior**
- **Error messages** (full stack traces)

## ✨ Feature Requests

For new features, please:

- **Check existing issues** for similar requests
- **Describe the use case** and motivation
- **Provide implementation ideas** if possible
- **Consider backwards compatibility**

## 🔍 Code Review Process

1. **Automated checks** must pass (CI/CD)
2. **Manual review** by maintainers
3. **Testing** on multiple environments
4. **Documentation** updates if needed

### Review Checklist

- [ ] Code follows style guidelines
- [ ] Tests added for new functionality
- [ ] Documentation updated
- [ ] No breaking changes (or properly documented)
- [ ] Performance implications considered

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

## 🤝 Community

- **Be respectful** and inclusive
- **Ask questions** if anything is unclear
- **Help others** in discussions and reviews
- **Share knowledge** through documentation and examples

Thank you for contributing to advancing open-source protein design! 🧬