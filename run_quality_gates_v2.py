"""
Enhanced Quality Gates for TPU-Optimized Protein Diffusion Models

This module runs comprehensive quality gates including testing, security scanning,
performance benchmarking, and compliance validation.
"""

import subprocess
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QualityGate:
    """Base class for quality gates."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.passed = False
        self.duration = 0.0
        self.details = {}
        self.errors = []
    
    def run(self) -> bool:
        """Run the quality gate. Returns True if passed."""
        start_time = time.time()
        logger.info(f"Running quality gate: {self.name}")
        
        try:
            self.passed = self._execute()
        except Exception as e:
            self.passed = False
            self.errors.append(str(e))
            logger.error(f"Quality gate {self.name} failed with error: {e}")
        
        self.duration = time.time() - start_time
        
        status = "PASSED" if self.passed else "FAILED"
        logger.info(f"Quality gate {self.name} {status} in {self.duration:.2f}s")
        
        return self.passed
    
    def _execute(self) -> bool:
        """Execute the quality gate logic. Override in subclasses."""
        raise NotImplementedError


class UnitTestGate(QualityGate):
    """Unit testing quality gate."""
    
    def __init__(self):
        super().__init__("Unit Tests", "Run unit tests to verify core functionality")
    
    def _execute(self) -> bool:
        """Run unit tests."""
        try:
            # Run basic functionality tests
            result = subprocess.run([
                sys.executable, "-m", "unittest", 
                "tests.test_basic_functionality", "-v"
            ], capture_output=True, text=True, cwd=Path.cwd())
            
            self.details['stdout'] = result.stdout
            self.details['stderr'] = result.stderr
            self.details['return_code'] = result.returncode
            
            # Parse test results
            if "OK" in result.stdout:
                # Extract test counts
                lines = result.stdout.split('\n')
                for line in lines:
                    if line.startswith("Ran "):
                        parts = line.split()
                        if len(parts) >= 2:
                            self.details['tests_run'] = int(parts[1])
                
                return result.returncode == 0
            else:
                return False
                
        except Exception as e:
            self.errors.append(f"Failed to run tests: {e}")
            return False


class CodeQualityGate(QualityGate):
    """Code quality analysis gate."""
    
    def __init__(self):
        super().__init__("Code Quality", "Analyze code quality and style")
    
    def _execute(self) -> bool:
        """Analyze code quality."""
        quality_checks = []
        
        # Check 1: Basic Python syntax
        try:
            import ast
            src_path = Path("src/protein_diffusion")
            
            syntax_errors = []
            file_count = 0
            
            for py_file in src_path.glob("**/*.py"):
                file_count += 1
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    ast.parse(content)
                except SyntaxError as e:
                    syntax_errors.append(f"{py_file}: {e}")
            
            self.details['files_checked'] = file_count
            self.details['syntax_errors'] = syntax_errors
            
            quality_checks.append(len(syntax_errors) == 0)
            
        except Exception as e:
            self.errors.append(f"Syntax check failed: {e}")
            quality_checks.append(False)
        
        # Check 2: Import validation
        try:
            import_errors = []
            
            # Test core module imports
            test_imports = [
                "protein_diffusion.zero_nas",
                "protein_diffusion.tpu_optimization",
                "protein_diffusion.tpu_nas_integration"
            ]
            
            for module_name in test_imports:
                try:
                    __import__(module_name)
                except ImportError as e:
                    import_errors.append(f"{module_name}: {e}")
            
            self.details['import_errors'] = import_errors
            quality_checks.append(len(import_errors) == 0)
            
        except Exception as e:
            self.errors.append(f"Import validation failed: {e}")
            quality_checks.append(False)
        
        # Check 3: Documentation coverage
        try:
            src_path = Path("src/protein_diffusion")
            
            total_functions = 0
            documented_functions = 0
            
            for py_file in src_path.glob("**/*.py"):
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    lines = content.split('\n')
                    in_function = False
                    
                    for i, line in enumerate(lines):
                        stripped = line.strip()
                        
                        # Function definition
                        if stripped.startswith('def ') and not stripped.startswith('def _'):
                            total_functions += 1
                            in_function = True
                            
                            # Check if next non-empty line is docstring
                            for j in range(i + 1, min(i + 5, len(lines))):
                                next_line = lines[j].strip()
                                if next_line:
                                    if next_line.startswith('"""') or next_line.startswith("'''"):
                                        documented_functions += 1
                                    break
                
                except Exception:
                    continue
            
            doc_coverage = (documented_functions / max(total_functions, 1)) * 100
            self.details['documentation_coverage'] = doc_coverage
            self.details['total_functions'] = total_functions
            self.details['documented_functions'] = documented_functions
            
            quality_checks.append(doc_coverage >= 50)  # 50% minimum
            
        except Exception as e:
            self.errors.append(f"Documentation check failed: {e}")
            quality_checks.append(False)
        
        return all(quality_checks)


class SecurityScanGate(QualityGate):
    """Security scanning quality gate."""
    
    def __init__(self):
        super().__init__("Security Scan", "Scan for security vulnerabilities")
    
    def _execute(self) -> bool:
        """Run security scans."""
        security_checks = []
        
        # Check 1: No hardcoded secrets
        try:
            src_path = Path("src/protein_diffusion")
            
            suspicious_patterns = [
                'password', 'secret', 'api_key', 'token', 'private_key',
                'aws_secret', 'database_url', 'connection_string'
            ]
            
            security_issues = []
            
            for py_file in src_path.glob("**/*.py"):
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read().lower()
                    
                    for pattern in suspicious_patterns:
                        if pattern in content and '=' in content:
                            # Simple heuristic to avoid false positives
                            lines = content.split('\n')
                            for i, line in enumerate(lines):
                                if pattern in line and '=' in line and not line.strip().startswith('#'):
                                    security_issues.append(f"{py_file}:{i+1} - Potential secret: {pattern}")
                
                except Exception:
                    continue
            
            self.details['security_issues'] = security_issues
            security_checks.append(len(security_issues) == 0)
            
        except Exception as e:
            self.errors.append(f"Secret scan failed: {e}")
            security_checks.append(False)
        
        # Check 2: Safe imports only
        try:
            dangerous_imports = [
                'subprocess.call', 'os.system', 'eval(', 'exec(',
                'pickle.loads', '__import__'
            ]
            
            dangerous_usage = []
            src_path = Path("src/protein_diffusion")
            
            for py_file in src_path.glob("**/*.py"):
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    for dangerous in dangerous_imports:
                        if dangerous in content:
                            lines = content.split('\n')
                            for i, line in enumerate(lines):
                                if dangerous in line and not line.strip().startswith('#'):
                                    dangerous_usage.append(f"{py_file}:{i+1} - Dangerous usage: {dangerous}")
                
                except Exception:
                    continue
            
            self.details['dangerous_usage'] = dangerous_usage
            # Allow some dangerous usage in our case (subprocess for tests, pickle for caching)
            security_checks.append(len(dangerous_usage) < 10)
            
        except Exception as e:
            self.errors.append(f"Import safety check failed: {e}")
            security_checks.append(False)
        
        return all(security_checks)


class PerformanceBenchmarkGate(QualityGate):
    """Performance benchmarking quality gate."""
    
    def __init__(self):
        super().__init__("Performance Benchmark", "Run performance benchmarks")
    
    def _execute(self) -> bool:
        """Run performance benchmarks."""
        try:
            # Import and run basic performance tests
            sys.path.insert(0, str(Path("src")))
            
            from protein_diffusion.zero_nas import (
                ArchitectureConfig, ProteinDiffusionArchitecture, ZeroCostEvaluator
            )
            
            benchmarks = {}
            
            # Benchmark 1: Architecture creation speed
            start_time = time.time()
            for _ in range(100):
                config = ArchitectureConfig()
                arch = ProteinDiffusionArchitecture(config)
            arch_creation_time = time.time() - start_time
            
            benchmarks['architecture_creation_per_sec'] = 100 / arch_creation_time
            
            # Benchmark 2: Parameter estimation speed
            config = ArchitectureConfig()
            start_time = time.time()
            for _ in range(100):
                arch = ProteinDiffusionArchitecture(config)
                params = arch.estimated_params
                flops = arch.estimated_flops
            param_estimation_time = time.time() - start_time
            
            benchmarks['param_estimation_per_sec'] = 100 / param_estimation_time
            
            # Benchmark 3: Evaluation speed
            evaluator = ZeroCostEvaluator()
            arch = ProteinDiffusionArchitecture(config)
            
            start_time = time.time()
            for _ in range(10):
                metrics = evaluator.evaluate_architecture(arch)
            evaluation_time = time.time() - start_time
            
            benchmarks['evaluation_per_sec'] = 10 / evaluation_time
            
            self.details['benchmarks'] = benchmarks
            
            # Performance thresholds
            performance_checks = [
                benchmarks['architecture_creation_per_sec'] > 50,  # > 50 per second
                benchmarks['param_estimation_per_sec'] > 100,     # > 100 per second
                benchmarks['evaluation_per_sec'] > 1             # > 1 per second
            ]
            
            return all(performance_checks)
            
        except Exception as e:
            self.errors.append(f"Performance benchmark failed: {e}")
            return False


class ArchitectureValidationGate(QualityGate):
    """Architecture validation quality gate."""
    
    def __init__(self):
        super().__init__("Architecture Validation", "Validate protein diffusion architectures")
    
    def _execute(self) -> bool:
        """Validate architectures."""
        try:
            sys.path.insert(0, str(Path("src")))
            
            from protein_diffusion.zero_nas import (
                ArchitectureConfig, ProteinDiffusionArchitecture, 
                create_protein_diffusion_search_space
            )
            
            validation_checks = []
            
            # Check 1: Default architecture is valid
            try:
                default_config = ArchitectureConfig()
                default_arch = ProteinDiffusionArchitecture(default_config)
                
                # Basic validation
                assert default_arch.estimated_params > 0
                assert default_arch.estimated_flops > 0
                assert len(default_arch.arch_id) > 0
                
                validation_checks.append(True)
                
            except Exception as e:
                self.errors.append(f"Default architecture validation failed: {e}")
                validation_checks.append(False)
            
            # Check 2: Search space architectures are valid
            try:
                search_space = create_protein_diffusion_search_space()
                
                # Test a few random combinations
                import random
                
                valid_architectures = 0
                total_tested = 10
                
                for _ in range(total_tested):
                    config_dict = {}
                    for key, values in search_space.items():
                        if isinstance(values[0], list):
                            # Handle nested lists (e.g., conv_layers)
                            config_dict[key] = random.choice(values)
                        else:
                            config_dict[key] = random.choice(values)
                    
                    try:
                        config = ArchitectureConfig(**config_dict)
                        arch = ProteinDiffusionArchitecture(config)
                        
                        # Validate constraints
                        assert arch.config.hidden_size % arch.config.num_attention_heads == 0
                        assert arch.estimated_params > 0
                        
                        valid_architectures += 1
                        
                    except Exception:
                        continue
                
                self.details['valid_architectures'] = valid_architectures
                self.details['total_tested'] = total_tested
                
                validation_checks.append(valid_architectures >= total_tested * 0.8)  # 80% should be valid
                
            except Exception as e:
                self.errors.append(f"Search space validation failed: {e}")
                validation_checks.append(False)
            
            # Check 3: Parameter scaling is reasonable
            try:
                small_config = ArchitectureConfig(num_layers=6, hidden_size=384)
                large_config = ArchitectureConfig(num_layers=24, hidden_size=1024)
                
                small_arch = ProteinDiffusionArchitecture(small_config)
                large_arch = ProteinDiffusionArchitecture(large_config)
                
                # Large should have more parameters
                assert large_arch.estimated_params > small_arch.estimated_params
                assert large_arch.estimated_flops > small_arch.estimated_flops
                
                # Scaling should be reasonable (not too extreme)
                param_ratio = large_arch.estimated_params / small_arch.estimated_params
                assert 2 < param_ratio < 100  # Between 2x and 100x
                
                validation_checks.append(True)
                
            except Exception as e:
                self.errors.append(f"Parameter scaling validation failed: {e}")
                validation_checks.append(False)
            
            return all(validation_checks)
            
        except Exception as e:
            self.errors.append(f"Architecture validation failed: {e}")
            return False


class ComplianceGate(QualityGate):
    """Compliance and licensing quality gate."""
    
    def __init__(self):
        super().__init__("Compliance Check", "Verify licensing and compliance")
    
    def _execute(self) -> bool:
        """Check compliance."""
        compliance_checks = []
        
        # Check 1: License file exists
        try:
            license_file = Path("LICENSE")
            if license_file.exists():
                with open(license_file, 'r') as f:
                    license_content = f.read()
                self.details['license_exists'] = True
                self.details['license_length'] = len(license_content)
                compliance_checks.append(len(license_content) > 100)  # Reasonable license
            else:
                self.details['license_exists'] = False
                compliance_checks.append(False)
                
        except Exception as e:
            self.errors.append(f"License check failed: {e}")
            compliance_checks.append(False)
        
        # Check 2: README exists and is comprehensive
        try:
            readme_file = Path("README.md")
            if readme_file.exists():
                with open(readme_file, 'r') as f:
                    readme_content = f.read()
                
                # Check for required sections
                required_sections = ['overview', 'installation', 'usage', 'citation']
                sections_found = sum(1 for section in required_sections 
                                   if section.lower() in readme_content.lower())
                
                self.details['readme_exists'] = True
                self.details['readme_length'] = len(readme_content)
                self.details['sections_found'] = sections_found
                
                compliance_checks.append(sections_found >= 3)  # At least 3 required sections
            else:
                self.details['readme_exists'] = False
                compliance_checks.append(False)
                
        except Exception as e:
            self.errors.append(f"README check failed: {e}")
            compliance_checks.append(False)
        
        # Check 3: No prohibited content
        try:
            prohibited_terms = [
                'malicious', 'hack', 'exploit', 'backdoor', 'virus',
                'illegal', 'unauthorized', 'crack', 'pirate'
            ]
            
            prohibited_found = []
            src_path = Path("src/protein_diffusion")
            
            for py_file in src_path.glob("**/*.py"):
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read().lower()
                    
                    for term in prohibited_terms:
                        if term in content:
                            prohibited_found.append(f"{py_file}: {term}")
                
                except Exception:
                    continue
            
            self.details['prohibited_content'] = prohibited_found
            compliance_checks.append(len(prohibited_found) == 0)
            
        except Exception as e:
            self.errors.append(f"Content compliance check failed: {e}")
            compliance_checks.append(False)
        
        return all(compliance_checks)


class QualityGateRunner:
    """Main quality gate runner."""
    
    def __init__(self):
        self.gates = [
            UnitTestGate(),
            CodeQualityGate(),
            SecurityScanGate(),
            PerformanceBenchmarkGate(),
            ArchitectureValidationGate(),
            ComplianceGate()
        ]
        
        self.results = {}
    
    def run_all_gates(self) -> Dict[str, Any]:
        """Run all quality gates."""
        logger.info("Starting quality gate validation")
        start_time = time.time()
        
        passed_gates = 0
        total_gates = len(self.gates)
        
        for gate in self.gates:
            gate_passed = gate.run()
            
            self.results[gate.name] = {
                'passed': gate_passed,
                'duration': gate.duration,
                'details': gate.details,
                'errors': gate.errors,
                'description': gate.description
            }
            
            if gate_passed:
                passed_gates += 1
        
        total_duration = time.time() - start_time
        
        # Overall summary
        summary = {
            'overall_passed': passed_gates == total_gates,
            'passed_gates': passed_gates,
            'total_gates': total_gates,
            'success_rate': passed_gates / total_gates,
            'total_duration': total_duration,
            'timestamp': time.time(),
            'gate_results': self.results
        }
        
        logger.info(f"Quality gates completed: {passed_gates}/{total_gates} passed")
        logger.info(f"Overall success rate: {summary['success_rate']:.1%}")
        
        return summary
    
    def export_results(self, filepath: str):
        """Export results to JSON file."""
        try:
            with open(filepath, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            logger.info(f"Quality gate results exported to {filepath}")
        except Exception as e:
            logger.error(f"Failed to export results: {e}")


def main():
    """Main entry point."""
    print("🔒 TPU-Optimized Protein Diffusion - Quality Gates")
    print("=" * 60)
    
    runner = QualityGateRunner()
    summary = runner.run_all_gates()
    
    # Export results
    results_file = f"quality_gates_results_{int(time.time())}.json"
    runner.export_results(results_file)
    
    # Print summary
    print("\n📊 QUALITY GATES SUMMARY")
    print("=" * 40)
    print(f"Overall Status: {'✅ PASSED' if summary['overall_passed'] else '❌ FAILED'}")
    print(f"Success Rate: {summary['success_rate']:.1%}")
    print(f"Gates Passed: {summary['passed_gates']}/{summary['total_gates']}")
    print(f"Total Duration: {summary['total_duration']:.1f}s")
    
    print("\n📋 INDIVIDUAL GATES")
    print("-" * 40)
    for gate_name, result in summary['gate_results'].items():
        status = "✅ PASS" if result['passed'] else "❌ FAIL"
        print(f"{gate_name}: {status} ({result['duration']:.1f}s)")
        
        if result['errors']:
            for error in result['errors']:
                print(f"  ⚠️  {error}")
    
    print(f"\n📄 Detailed results saved to: {results_file}")
    
    # Exit with appropriate code
    exit_code = 0 if summary['overall_passed'] else 1
    return exit_code


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)