"""Test package initialization."""

import pytest


@pytest.mark.unit
def test_package_version():
    """Test that package version is accessible."""
    from protein_diffusion import __version__, __author__, __email__
    
    assert __version__ == "0.1.0"
    assert __author__ == "Daniel Schmidt"
    assert isinstance(__email__, str)


@pytest.mark.unit 
def test_package_imports():
    """Test basic package imports work."""
    import protein_diffusion
    
    assert hasattr(protein_diffusion, '__version__')
    assert hasattr(protein_diffusion, '__author__')
    assert hasattr(protein_diffusion, '__email__')