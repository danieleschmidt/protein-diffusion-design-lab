#!/usr/bin/env python3
"""
Test runner for the protein diffusion project.

This script runs tests and quality checks in the correct order.
"""

import sys
import subprocess
from pathlib import Path
import argparse


def run_command(cmd, description, check=True):
    """Run a command and handle output."""
    print(f"\n🧪 {description}")
    print("=" * 60)
    
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=False, 
            text=True,
            check=check
        )
        
        if result.returncode == 0:
            print(f"✅ {description} - PASSED")
        else:
            print(f"❌ {description} - FAILED (exit code: {result.returncode})")
            
        return result.returncode == 0
        
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - FAILED: {e}")
        return False
    except Exception as e:
        print(f"❌ {description} - ERROR: {e}")
        return False


def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(description="Run tests and quality checks")
    parser.add_argument("--unit-only", action="store_true", 
                       help="Run only unit tests")
    parser.add_argument("--fast", action="store_true", 
                       help="Skip slow tests")
    parser.add_argument("--no-quality", action="store_true", 
                       help="Skip quality gates")
    parser.add_argument("--coverage", action="store_true", 
                       help="Run with coverage reporting")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Verbose output")
    
    args = parser.parse_args()
    
    project_root = Path(__file__).parent
    
    print("🚀 Running Protein Diffusion Test Suite")
    print("=" * 60)
    print(f"Project root: {project_root}")
    print(f"Python version: {sys.version}")
    
    # Check if pytest is available
    try:
        import pytest
        print(f"✅ pytest version: {pytest.__version__}")
    except ImportError:
        print("❌ pytest not found. Installing...")
        if not run_command("pip install pytest pytest-cov pytest-mock", "Installing pytest"):
            print("❌ Failed to install pytest. Exiting.")
            return 1
        import pytest
    
    success_count = 0
    total_tests = 0
    
    # Build test command
    test_cmd = ["python", "-m", "pytest"]
    
    if args.verbose:
        test_cmd.append("-v")
    else:
        test_cmd.append("-q")
    
    if args.coverage:
        test_cmd.extend([
            "--cov=src/protein_diffusion",
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov"
        ])
    
    if args.fast:
        test_cmd.extend(["-m", "not slow"])
    
    # Add test paths
    if args.unit_only:
        test_cmd.append("tests/unit/")
    else:
        test_cmd.append("tests/")
    
    test_cmd_str = " ".join(test_cmd)
    
    # Run tests
    total_tests += 1
    if run_command(test_cmd_str, "Running pytest"):
        success_count += 1
    
    # Run quality gates (unless skipped)
    if not args.no_quality:
        total_tests += 1
        if run_command("python3 run_quality_gates.py", "Running quality gates", check=False):
            success_count += 1
    
    # Run basic import test
    total_tests += 1
    import_test_cmd = 'python -c "import src.protein_diffusion; print(\'✅ Package imports successfully\')"'
    if run_command(import_test_cmd, "Testing package imports"):
        success_count += 1
    
    # Check for security issues with bandit (if available)
    try:
        import bandit
        total_tests += 1
        bandit_cmd = "python -m bandit -r src/ -f json || true"
        if run_command(bandit_cmd, "Running security scan (bandit)", check=False):
            success_count += 1
    except ImportError:
        print("⚠️  bandit not available, skipping security scan")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print(f"Tests passed: {success_count}/{total_tests}")
    print(f"Success rate: {(success_count/total_tests)*100:.1f}%")
    
    if success_count == total_tests:
        print("🎉 All tests passed!")
        return 0
    else:
        print(f"❌ {total_tests - success_count} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())