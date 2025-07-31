# Development Guide

This comprehensive guide covers the development workflow, tooling, and best practices for the Protein Diffusion Design Lab project.

## Quick Start

```bash
# Clone and setup
git clone https://github.com/yourusername/protein-diffusion-design-lab.git
cd protein-diffusion-design-lab

# Setup development environment
make dev-setup

# Run tests
make test

# Start development server
make docker-dev
```

## Development Environment

### Prerequisites

- Python 3.9+
- Docker & Docker Compose
- CUDA 11.0+ (for GPU acceleration)
- Git with LFS (for model weights)

### Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements-dev.txt
pre-commit install
```

### IDE Configuration

#### VS Code Settings
```json
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
    }
}
```

## Development Workflow

### 1. Feature Development

```bash
# Create feature branch
git checkout -b feature/new-feature

# Make changes, following conventions
# ... code changes ...

# Run full development workflow
make ci

# Commit and push
git add .
git commit -m "feat: add new feature"
git push origin feature/new-feature
```

### 2. Code Quality Checks

```bash
# Format code
make format

# Run linting
make lint

# Security scan
make security

# Full test suite
make test-coverage
```

### 3. Testing Strategy

#### Unit Tests
- Location: `tests/unit/`
- Command: `pytest tests/unit/ -v`
- Coverage: Aim for >90%

#### Integration Tests
- Location: `tests/integration/`
- Command: `pytest tests/integration/ -v -m integration`
- Requires: GPU access for ML tests

#### Performance Tests
- Command: `pytest tests/ -v -m performance --benchmark-only`
- Monitors: Inference time, memory usage, throughput

### 4. Docker Development

```bash
# Build development image
make docker-build

# Run with live reload
make docker-dev

# Run tests in container
make docker-test
```

## Architecture Overview

```
src/protein_diffusion/
├── core/           # Core ML models and algorithms
├── data/           # Data processing and loaders
├── models/         # Model definitions
├── utils/          # Utilities and helpers
├── api/            # API endpoints
└── ui/             # Streamlit interface
```

## Code Standards

### Python Style

- **Formatter**: Black (line length: 88)
- **Import sorting**: isort with black profile
- **Linting**: flake8 with bandit for security
- **Type hints**: Required for all public functions
- **Docstrings**: Google style

Example:
```python
def generate_protein_scaffold(
    motif: str, 
    temperature: float = 0.8,
    num_samples: int = 100
) -> List[ProteinStructure]:
    """Generate protein scaffolds using diffusion model.
    
    Args:
        motif: Target structural motif
        temperature: Sampling temperature for diversity
        num_samples: Number of samples to generate
        
    Returns:
        List of generated protein structures
        
    Raises:
        ValueError: If motif is invalid
        RuntimeError: If GPU memory insufficient
    """
    pass
```

### Git Conventions

#### Commit Messages
Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add new diffusion sampling method
fix: resolve GPU memory leak in inference
docs: update API documentation
refactor: simplify data preprocessing pipeline
test: add integration tests for docking module
```

#### Branch Naming
- `feat/feature-name` - New features
- `fix/bug-description` - Bug fixes
- `docs/topic` - Documentation updates
- `refactor/component` - Code refactoring

## Testing Guidelines

### Writing Tests

```python
import pytest
from protein_diffusion import ProteinDiffuser

class TestProteinDiffuser:
    @pytest.fixture
    def diffuser(self):
        return ProteinDiffuser(device='cpu')
    
    def test_generate_basic(self, diffuser):
        """Test basic scaffold generation."""
        scaffolds = diffuser.generate("HELIX", num_samples=1)
        assert len(scaffolds) == 1
        assert scaffolds[0].is_valid()
    
    @pytest.mark.gpu
    def test_generate_gpu(self):
        """Test GPU-accelerated generation."""
        diffuser = ProteinDiffuser(device='cuda')
        scaffolds = diffuser.generate("HELIX", num_samples=10)
        assert len(scaffolds) == 10
    
    @pytest.mark.slow
    def test_large_batch_generation(self, diffuser):
        """Test large batch processing."""
        scaffolds = diffuser.generate("HELIX", num_samples=1000)
        assert len(scaffolds) == 1000
```

### Test Markers
- `@pytest.mark.unit` - Unit tests (fast)
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.gpu` - Requires GPU
- `@pytest.mark.slow` - Long-running tests
- `@pytest.mark.security` - Security-related tests

## Performance Optimization

### Profiling

```bash
# Memory profiling
python -m memory_profiler scripts/profile_memory.py

# Line profiling
kernprof -l -v scripts/profile_inference.py

# CPU profiling with py-spy
py-spy record -o profile.svg -- python scripts/run_inference.py
```

### GPU Optimization

```python
# Use torch.compile for PyTorch 2.0+
model = torch.compile(model, mode="max-autotune")

# Mixed precision training
with torch.autocast(device_type='cuda'):
    output = model(input_data)

# Gradient checkpointing for memory efficiency
model.gradient_checkpointing_enable()
```

## Security Best Practices

### Code Security
- Never commit secrets or API keys
- Use environment variables for configuration
- Validate all inputs
- Use safe YAML loading: `yaml.safe_load()`
- Regular dependency updates

### Model Security
- Validate model inputs/outputs
- Implement rate limiting
- Monitor for adversarial inputs
- Secure model serving endpoints

## Debugging

### Common Issues

1. **CUDA Out of Memory**
   ```python
   # Reduce batch size or use gradient checkpointing
   torch.cuda.empty_cache()
   ```

2. **Slow Inference**
   ```python
   # Use TorchScript or torch.compile
   model = torch.jit.script(model)
   ```

3. **Import Errors**
   ```bash
   # Check PYTHONPATH
   export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
   ```

### Debug Commands

```bash
# Interactive debugging
python -m pdb script.py

# Memory debugging
python -m tracemalloc script.py

# GPU debugging
nvidia-smi -l 1
```

## Release Process

### Version Management
- Use semantic versioning: `MAJOR.MINOR.PATCH`
- Update `pyproject.toml` version
- Create git tags: `v1.0.0`

### Release Checklist
- [ ] All tests passing
- [ ] Documentation updated
- [ ] Security scan clean
- [ ] Performance benchmarks run
- [ ] Docker images built
- [ ] Release notes prepared

```bash
# Create release
git tag -a v1.0.0 -m "Release v1.0.0"
git push origin v1.0.0

# Build and publish
python -m build
twine upload dist/*
```

## Monitoring and Observability

### Local Development
```bash
# Start monitoring stack
cd monitoring
docker-compose -f docker-compose.monitoring.yml up -d

# Access dashboards
# Grafana: http://localhost:3000 (admin/admin)
# Prometheus: http://localhost:9090
```

### Metrics to Track
- Inference latency (p50, p95, p99)
- GPU utilization and memory
- Request rate and error rate
- Model quality metrics (diversity, validity)

## Contributing

### Pull Request Process
1. Fork the repository
2. Create feature branch
3. Make changes with tests
4. Run full CI pipeline locally
5. Submit PR with clear description
6. Address review feedback
7. Squash and merge

### Code Review Guidelines
- Focus on correctness, performance, maintainability
- Check test coverage
- Verify security implications
- Ensure documentation updates

## Resources

- [PyTorch Best Practices](https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [Python Security Guide](https://python-security.readthedocs.io/)
- [ML Engineering Guide](https://huyenchip.com/ml-interviews-book/)