"""Pytest configuration and shared fixtures."""

import pytest
import torch
import tempfile
import shutil
from pathlib import Path


@pytest.fixture(scope="session")
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture(scope="session")
def device():
    """Get the appropriate device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def sample_protein_sequence():
    """Sample protein sequence for testing."""
    return "MKWVTFISLLLLFSSAYSRGVFRRDTHKSEIAHRFKDLGEEHFKGLVLIAFSQYLQQCPFDEHVKLVNELTEFAKTCVADESHAGCEKSLHTLFGDELCKVASLRETYGDMADCCEKQEPERNECFLSHKDDSPDL"


@pytest.fixture
def sample_motif():
    """Sample structural motif for testing.""" 
    return "HELIX_SHEET_HELIX"


@pytest.fixture
def mock_pdb_file(temp_dir):
    """Create a mock PDB file for testing."""
    pdb_content = """HEADER    PROTEIN                                 01-JAN-25   TEST            
ATOM      1  N   ALA A   1      20.154  16.967  15.691  1.00 10.00           N  
ATOM      2  CA  ALA A   1      20.154  18.379  16.107  1.00 10.00           C  
ATOM      3  C   ALA A   1      21.618  18.807  16.381  1.00 10.00           C  
ATOM      4  O   ALA A   1      22.041  19.920  16.070  1.00 10.00           O  
END                                                                             
"""
    pdb_path = temp_dir / "test.pdb"
    pdb_path.write_text(pdb_content)
    return pdb_path


@pytest.fixture
def skip_if_no_gpu():
    """Skip test if GPU is not available."""
    if not torch.cuda.is_available():
        pytest.skip("GPU not available")


# Pytest markers for test categorization
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with -m 'not slow')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "gpu: marks tests that require GPU")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")