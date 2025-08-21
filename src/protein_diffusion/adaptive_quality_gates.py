#!/usr/bin/env python3
"""
Adaptive Quality Gates for Protein Diffusion Design Lab

Self-contained quality validation system that adapts to environment
without requiring external dependencies.
"""

import sys
import os
import subprocess
import time
import json
import logging
import importlib
import platform
import gc
import warnings
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum

# Suppress warnings during quality gate execution
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('adaptive_quality_gates.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)


class GateStatus(Enum):
    """Quality gate status."""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    WARNING = "warning"


@dataclass
class QualityGateResult:
    """Results from a quality gate execution."""
    name: str
    status: GateStatus
    execution_time: float = 0.0
    output: str = ""
    error_output: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class AdaptiveQualityGateRunner:
    """Self-contained quality gate runner that adapts to environment."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results: Dict[str, QualityGateResult] = {}
        self.system_info = self._gather_system_info()
        self.execution_start_time = 0
        
        if verbose:
            logging.getLogger().setLevel(logging.DEBUG)
    
    def _gather_system_info(self) -> Dict[str, Any]:
        """Gather basic system information without external dependencies."""
        try:
            info = {
                'platform': platform.platform(),
                'python_version': platform.python_version(),
                'cpu_count': os.cpu_count(),
                'working_directory': os.getcwd(),
                'python_executable': sys.executable
            }
            
            # Try to get memory info (fallback if psutil unavailable)
            try:
                import psutil
                info['memory_gb'] = psutil.virtual_memory().total / (1024**3)
                info['available_memory_gb'] = psutil.virtual_memory().available / (1024**3)
                info['current_memory_mb'] = psutil.Process().memory_info().rss / (1024**2)
            except ImportError:
                info['memory_info'] = 'unavailable (psutil not installed)'
            
        except Exception as e:
            info = {'platform': 'unknown', 'error': f'failed to gather system info: {e}'}
        
        return info
    
    def _run_gate(self, name: str, description: str, gate_function: callable, 
                  required: bool = True, timeout: int = 300) -> QualityGateResult:
        """Run a single quality gate."""
        result = QualityGateResult(name=name, status=GateStatus.RUNNING)
        self.results[name] = result
        
        logger.info(f"🔄 Running: {name} - {description}")
        start_time = time.time()
        
        try:
            success = gate_function(result)
            result.execution_time = time.time() - start_time
            result.status = GateStatus.PASSED if success else GateStatus.FAILED
            
            if success:
                logger.info(f"✅ {name}: PASSED ({result.execution_time:.2f}s)")
            else:
                if required:
                    logger.error(f"❌ {name}: FAILED ({result.execution_time:.2f}s)")
                else:
                    logger.warning(f"⚠️ {name}: FAILED (optional) ({result.execution_time:.2f}s)")
            
        except Exception as e:
            result.execution_time = time.time() - start_time
            result.status = GateStatus.FAILED
            result.error_output = str(e)
            logger.error(f"💥 {name}: ERROR - {e} ({result.execution_time:.2f}s)")
        
        return result
    
    # ===== GATE IMPLEMENTATIONS =====
    
    def _gate_system_health(self, result: QualityGateResult) -> bool:
        """Check basic system health."""
        try:
            # Check Python version
            if sys.version_info < (3, 9):
                result.warnings.append(f"Python {sys.version_info} detected, 3.9+ recommended")
            
            # Check working directory
            if not os.path.exists('src'):
                result.warnings.append("Source directory 'src' not found")
                return False
            
            # Check basic file structure
            required_files = ['README.md', 'src/protein_diffusion/__init__.py']
            missing_files = [f for f in required_files if not os.path.exists(f)]
            
            if missing_files:
                result.warnings.extend([f"Missing file: {f}" for f in missing_files])
            
            result.metrics.update({
                'python_version': sys.version,
                'platform': platform.platform(),
                'cpu_count': os.cpu_count(),
                'missing_files': len(missing_files)
            })
            
            return len(missing_files) == 0
            
        except Exception as e:
            result.error_output = str(e)
            return False
    
    def _gate_import_validation(self, result: QualityGateResult) -> bool:
        """Validate core module imports."""
        try:
            import_tests = [
                ('sys', None),
                ('os', None),
                ('json', None),
                ('pathlib.Path', 'Path'),
                ('protein_diffusion', None),
            ]
            
            successful_imports = 0
            failed_imports = []
            
            for module_name, attr_name in import_tests:
                try:
                    module = importlib.import_module(module_name)
                    if attr_name and not hasattr(module, attr_name):
                        failed_imports.append(f"{module_name}.{attr_name}")
                    else:
                        successful_imports += 1
                except ImportError as e:
                    failed_imports.append(f"{module_name}: {str(e)}")
            
            # Test core protein_diffusion imports
            try:
                from protein_diffusion import ProteinDiffuser, ProteinDiffuserConfig
                successful_imports += 1
            except ImportError as e:
                failed_imports.append(f"protein_diffusion core: {str(e)}")
            
            total_tests = len(import_tests) + 1
            success_rate = successful_imports / total_tests
            
            result.metrics.update({
                'successful_imports': successful_imports,
                'failed_imports': len(failed_imports),
                'total_tests': total_tests,
                'success_rate': success_rate
            })
            
            if failed_imports:
                result.warnings.extend(failed_imports)
            
            return success_rate >= 0.8  # Require 80% success rate
            
        except Exception as e:
            result.error_output = str(e)
            return False
    
    def _gate_basic_functionality(self, result: QualityGateResult) -> bool:
        """Test basic system functionality."""
        try:
            # Try simplified diffuser first for compatibility
            try:
                from protein_diffusion.simplified_diffuser import (
                    SimplifiedProteinDiffuser, SimplifiedDiffuserConfig
                )
                
                # Test configuration creation
                config = SimplifiedDiffuserConfig()
                config.num_samples = 1
                config.max_length = 16
                
                # Test diffuser initialization
                diffuser = SimplifiedProteinDiffuser(config)
                
                # Test minimal generation
                start_time = time.time()
                results = diffuser.generate(
                    num_samples=1,
                    max_length=8,
                    progress=False
                )
                generation_time = time.time() - start_time
                
                result.metrics.update({
                    'generation_time_seconds': generation_time,
                    'num_results': len(results) if results else 0,
                    'config_created': True,
                    'diffuser_initialized': True,
                    'method': 'simplified_diffuser'
                })
                
                # Success if we got at least one result
                return results and len(results) > 0
                
            except ImportError:
                # Fallback to main diffuser interface
                from protein_diffusion import ProteinDiffuser, ProteinDiffuserConfig
                
                # Test configuration creation
                config = ProteinDiffuserConfig()
                config.num_samples = 1
                config.max_length = 16
                
                # Test diffuser initialization
                diffuser = ProteinDiffuser(config)
                
                # Test minimal generation
                start_time = time.time()
                results = diffuser.generate(
                    num_samples=1,
                    max_length=8,
                    progress=False
                )
                generation_time = time.time() - start_time
                
                result.metrics.update({
                    'generation_time_seconds': generation_time,
                    'num_results': len(results) if results else 0,
                    'config_created': True,
                    'diffuser_initialized': True,
                    'method': 'main_diffuser'
                })
                
                # Success if we got at least one result
                return results and len(results) > 0
            
        except Exception as e:
            result.error_output = str(e)
            return False
    
    def _gate_code_structure(self, result: QualityGateResult) -> bool:
        """Check code structure and quality."""
        try:
            structure_score = 0
            max_score = 5
            
            # Check Python syntax in core files
            core_files = [
                'src/protein_diffusion/__init__.py',
                'src/protein_diffusion/diffuser.py',
                'src/protein_diffusion/models.py'
            ]
            
            syntax_errors = 0
            files_checked = 0
            
            for file_path in core_files:
                if os.path.exists(file_path):
                    files_checked += 1
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            compile(f.read(), file_path, 'exec')
                        structure_score += 1
                    except SyntaxError:
                        syntax_errors += 1
                        result.warnings.append(f"Syntax error in {file_path}")
            
            # Check for proper package structure
            if os.path.exists('src/protein_diffusion/__init__.py'):
                structure_score += 1
            
            # Check for tests directory
            if os.path.exists('tests'):
                structure_score += 1
            
            result.metrics.update({
                'structure_score': structure_score,
                'max_score': max_score,
                'files_checked': files_checked,
                'syntax_errors': syntax_errors,
                'structure_percentage': (structure_score / max_score) * 100
            })
            
            return structure_score >= 3  # Require at least 60%
            
        except Exception as e:
            result.error_output = str(e)
            return False
    
    def _gate_security_basic(self, result: QualityGateResult) -> bool:
        """Basic security checks without external tools."""
        try:
            security_score = 5  # Start with perfect score, subtract for issues
            security_issues = []
            
            # Check for dangerous patterns in Python files
            py_files = list(Path('src').rglob('*.py'))
            dangerous_patterns = [
                ('exec(', 'Use of exec() function'),
                ('__import__', 'Dynamic imports'),
                ('shell=True', 'Shell execution enabled')
            ]
            
            files_checked = 0
            for py_file in py_files[:10]:  # Check first 10 files
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        files_checked += 1
                        
                        for pattern, description in dangerous_patterns:
                            if pattern in content:
                                security_issues.append(f"{py_file}: {description}")
                                security_score -= 1
                                break  # One issue per file to avoid double counting
                except:
                    continue
            
            # Check for sensitive files that shouldn't be tracked
            sensitive_patterns = [
                '*.key', '*.pem', '*.p12', '*.pfx',
                '.env', 'secrets.json', 'credentials.json'
            ]
            
            for pattern in sensitive_patterns:
                if list(Path('.').glob(pattern)):
                    security_issues.append(f"Sensitive files found: {pattern}")
                    security_score -= 1
            
            result.metrics.update({
                'security_score': max(0, security_score),
                'max_score': 5,
                'files_checked': files_checked,
                'security_issues': len(security_issues),
                'security_percentage': max(0, security_score) / 5 * 100
            })
            
            if security_issues:
                result.warnings.extend(security_issues[:5])  # Limit warnings
            
            return security_score >= 3  # Require at least 60%
            
        except Exception as e:
            result.error_output = str(e)
            return False
    
    def _gate_performance_baseline(self, result: QualityGateResult) -> bool:
        """Measure basic performance metrics."""
        try:
            # Use simplified diffuser for performance testing
            try:
                from protein_diffusion.simplified_diffuser import (
                    SimplifiedProteinDiffuser, SimplifiedDiffuserConfig
                )
                
                # Measure initialization time
                init_start = time.time()
                config = SimplifiedDiffuserConfig()
                config.num_samples = 2
                config.max_length = 24
                diffuser = SimplifiedProteinDiffuser(config)
                init_time = time.time() - init_start
                
                # Measure generation time
                gen_start = time.time()
                results = diffuser.generate(
                    num_samples=2,
                    max_length=12,
                    progress=False
                )
                gen_time = time.time() - gen_start
                
                method = 'simplified_diffuser'
                
            except ImportError:
                from protein_diffusion import ProteinDiffuser, ProteinDiffuserConfig
                
                # Measure initialization time
                init_start = time.time()
                config = ProteinDiffuserConfig()
                config.num_samples = 2
                config.max_length = 24
                diffuser = ProteinDiffuser(config)
                init_time = time.time() - init_start
                
                # Measure generation time
                gen_start = time.time()
                results = diffuser.generate(
                    num_samples=2,
                    max_length=12,
                    progress=False
                )
                gen_time = time.time() - gen_start
                
                method = 'main_diffuser'
            
            # Basic memory check (if available)
            memory_info = "unavailable"
            try:
                import psutil
                memory_mb = psutil.Process().memory_info().rss / (1024**2)
                memory_info = f"{memory_mb:.1f}MB"
            except ImportError:
                memory_mb = 0
            
            result.metrics.update({
                'initialization_time_seconds': init_time,
                'generation_time_seconds': gen_time,
                'memory_usage': memory_info,
                'sequences_generated': len(results) if results else 0,
                'time_per_sequence': gen_time / max(len(results), 1) if results else float('inf')
            })
            
            # Performance thresholds (generous for compatibility)
            performance_ok = (
                init_time < 60 and  # 1 minute initialization
                gen_time < 120 and  # 2 minutes generation
                results and len(results) > 0  # Actually generated something
            )
            
            if not performance_ok:
                if init_time >= 60:
                    result.warnings.append(f"Slow initialization: {init_time:.1f}s")
                if gen_time >= 120:
                    result.warnings.append(f"Slow generation: {gen_time:.1f}s")
                if not results or len(results) == 0:
                    result.warnings.append("No sequences generated")
            
            return performance_ok
            
        except Exception as e:
            result.error_output = str(e)
            return False
    
    def _gate_research_validation(self, result: QualityGateResult) -> bool:
        """Validate research components."""
        try:
            research_score = 0
            max_score = 4
            
            # Check for research directory structure
            research_components = [
                ('research', 'Research directory'),
                ('research/results', 'Results directory'),
                ('notebooks', 'Jupyter notebooks'),
                ('docs', 'Documentation')
            ]
            
            for path, description in research_components:
                if os.path.exists(path):
                    research_score += 1
                else:
                    result.warnings.append(f"Missing {description}: {path}")
            
            # Check for advanced modules
            advanced_modules = [
                'src/protein_diffusion/novel_architectures.py',
                'src/protein_diffusion/research_innovations.py',
                'src/protein_diffusion/advanced_generation.py'
            ]
            
            advanced_count = sum(1 for module in advanced_modules if os.path.exists(module))
            
            result.metrics.update({
                'research_score': research_score,
                'max_score': max_score,
                'advanced_modules': advanced_count,
                'research_percentage': (research_score / max_score) * 100
            })
            
            return research_score >= 2  # Require at least 50% research infrastructure
            
        except Exception as e:
            result.error_output = str(e)
            return False
    
    def _gate_deployment_readiness(self, result: QualityGateResult) -> bool:
        """Check deployment readiness."""
        try:
            deployment_score = 0
            max_score = 6
            
            # Check deployment infrastructure
            deployment_components = [
                ('Dockerfile', 'Docker containerization'),
                ('requirements.txt', 'Python dependencies'),
                ('pyproject.toml', 'Project configuration'),
                ('deployment', 'Deployment scripts'),
                ('k8s', 'Kubernetes configuration'),
                ('monitoring', 'Monitoring setup')
            ]
            
            for path, description in deployment_components:
                if os.path.exists(path):
                    deployment_score += 1
                else:
                    result.warnings.append(f"Missing {description}: {path}")
            
            result.metrics.update({
                'deployment_score': deployment_score,
                'max_score': max_score,
                'deployment_percentage': (deployment_score / max_score) * 100
            })
            
            return deployment_score >= 3  # Require at least 50% deployment readiness
            
        except Exception as e:
            result.error_output = str(e)
            return False
    
    def _gate_stress_test(self, result: QualityGateResult) -> bool:
        """Run basic stress test."""
        try:
            # Use simplified diffuser for stress testing
            try:
                from protein_diffusion.simplified_diffuser import (
                    SimplifiedProteinDiffuser, SimplifiedDiffuserConfig
                )
                
                stress_start = time.time()
                successful_runs = 0
                total_runs = 3  # Reduced for faster execution
                
                config = SimplifiedDiffuserConfig()
                config.num_samples = 1
                config.max_length = 16
                
                method = 'simplified_diffuser'
                
            except ImportError:
                from protein_diffusion import ProteinDiffuser, ProteinDiffuserConfig
                
                stress_start = time.time()
                successful_runs = 0
                total_runs = 3  # Reduced for faster execution
                
                config = ProteinDiffuserConfig()
                config.num_samples = 1
                config.max_length = 16
                
                method = 'main_diffuser'
            
            # Run stress test iterations
            for i in range(total_runs):
                try:
                    if method == 'simplified_diffuser':
                        diffuser = SimplifiedProteinDiffuser(config)
                    else:
                        diffuser = ProteinDiffuser(config)
                        
                    results = diffuser.generate(
                        num_samples=1,
                        max_length=8,
                        progress=False
                    )
                    if results and len(results) > 0:
                        successful_runs += 1
                    
                    # Cleanup
                    del diffuser
                    gc.collect()
                    
                except Exception as e:
                    result.warnings.append(f"Stress test iteration {i+1} failed: {str(e)[:100]}")
            
            stress_time = time.time() - stress_start
            success_rate = successful_runs / total_runs
            
            result.metrics.update({
                'stress_test_time_seconds': stress_time,
                'successful_runs': successful_runs,
                'total_runs': total_runs,
                'success_rate': success_rate
            })
            
            return success_rate >= 0.67  # Require at least 67% success rate
            
        except Exception as e:
            result.error_output = str(e)
            return False
    
    # ===== EXECUTION ENGINE =====
    
    def run_all_gates(self) -> Dict[str, Any]:
        """Run all adaptive quality gates."""
        logger.info("🚀 Starting Adaptive Quality Gates Execution")
        logger.info("=" * 70)
        logger.info(f"🖥️ System: {self.system_info.get('platform', 'unknown')}")
        logger.info(f"🐍 Python: {self.system_info.get('python_version', 'unknown')}")
        logger.info(f"📁 Working Directory: {self.system_info.get('working_directory', 'unknown')}")
        logger.info("=" * 70)
        
        self.execution_start_time = time.time()
        
        # Define gates to run
        gates = [
            ("system_health", "System health and environment check", self._gate_system_health, True, 30),
            ("import_validation", "Core module import validation", self._gate_import_validation, True, 60),
            ("basic_functionality", "Basic system functionality test", self._gate_basic_functionality, True, 120),
            ("code_structure", "Code structure and syntax validation", self._gate_code_structure, False, 90),
            ("security_basic", "Basic security vulnerability check", self._gate_security_basic, True, 120),
            ("performance_baseline", "Performance baseline measurement", self._gate_performance_baseline, False, 300),
            ("research_validation", "Research components validation", self._gate_research_validation, False, 60),
            ("deployment_readiness", "Deployment readiness check", self._gate_deployment_readiness, False, 60),
            ("stress_test", "Basic stress testing", self._gate_stress_test, False, 180),
        ]
        
        logger.info(f"📊 Running {len(gates)} adaptive quality gates")
        
        # Run gates sequentially
        for i, (name, description, gate_function, required, timeout) in enumerate(gates, 1):
            logger.info(f"🔄 Gate {i}/{len(gates)}: {name}")
            self._run_gate(name, description, gate_function, required, timeout)
        
        # Compile results
        return self._compile_results()
    
    def _compile_results(self) -> Dict[str, Any]:
        """Compile final results."""
        total_time = time.time() - self.execution_start_time
        
        # Count results by status
        status_counts = {status.value: 0 for status in GateStatus}
        required_failed = 0
        optional_failed = 0
        total_gates = len(self.results)
        
        for result in self.results.values():
            status_counts[result.status.value] += 1
            
            if result.status == GateStatus.FAILED:
                # Determine if this was a required gate (simplified heuristic)
                required_gates = {'system_health', 'import_validation', 'basic_functionality', 'security_basic'}
                if result.name in required_gates:
                    required_failed += 1
                else:
                    optional_failed += 1
        
        # Overall success
        overall_success = required_failed == 0
        
        # Generate recommendations
        recommendations = self._generate_recommendations()
        
        compiled_results = {
            'timestamp': time.time(),
            'execution_time_seconds': total_time,
            'system_info': self.system_info,
            'overall_success': overall_success,
            'total_gates': total_gates,
            'status_counts': status_counts,
            'required_failed': required_failed,
            'optional_failed': optional_failed,
            'gate_results': {name: {
                'status': result.status.value,
                'execution_time': result.execution_time,
                'metrics': result.metrics,
                'warnings': result.warnings,
                'has_error': bool(result.error_output),
                'error_output': result.error_output[:500] if result.error_output else ""
            } for name, result in self.results.items()},
            'recommendations': recommendations
        }
        
        return compiled_results
    
    def _generate_recommendations(self) -> List[str]:
        """Generate improvement recommendations based on results."""
        recommendations = []
        
        # Check specific failure patterns
        for name, result in self.results.items():
            if result.status == GateStatus.FAILED:
                if name == 'import_validation':
                    recommendations.append("Install missing dependencies: check requirements.txt")
                elif name == 'basic_functionality':
                    recommendations.append("Debug core functionality: check diffuser initialization")
                elif name == 'security_basic':
                    recommendations.append("Address security issues: review code for unsafe patterns")
                elif name == 'performance_baseline':
                    recommendations.append("Optimize performance: check memory usage and generation speed")
        
        # System-level recommendations
        if sys.version_info < (3, 9):
            recommendations.append("Upgrade Python to 3.9+ for better compatibility")
        
        if not os.path.exists('tests'):
            recommendations.append("Add test directory structure for better quality assurance")
        
        return recommendations
    
    def print_summary(self, results: Dict[str, Any]) -> None:
        """Print execution summary."""
        logger.info("=" * 70)
        logger.info("📊 Adaptive Quality Gates Summary")
        logger.info("=" * 70)
        
        # Overall status
        if results['overall_success']:
            logger.info("🎉 ALL REQUIRED QUALITY GATES PASSED!")
        else:
            logger.error("💥 SOME REQUIRED QUALITY GATES FAILED!")
        
        # Statistics
        logger.info(f"⏱️ Total Time: {results['execution_time_seconds']:.2f}s")
        logger.info(f"📊 Total Gates: {results['total_gates']}")
        
        # Status breakdown
        status_counts = results['status_counts']
        logger.info(f"✅ Passed: {status_counts.get('passed', 0)}")
        logger.info(f"❌ Failed: {status_counts.get('failed', 0)}")
        logger.info(f"⏭️ Skipped: {status_counts.get('skipped', 0)}")
        logger.info(f"🔴 Required Failed: {results['required_failed']}")
        logger.info(f"🟡 Optional Failed: {results['optional_failed']}")
        
        # Failed gates
        failed_gates = [
            name for name, details in results['gate_results'].items()
            if details['status'] == 'failed'
        ]
        
        if failed_gates:
            logger.info("\n❌ Failed Gates:")
            for gate_name in failed_gates:
                gate_details = results['gate_results'][gate_name]
                logger.info(f"  • {gate_name} ({gate_details['execution_time']:.2f}s)")
        
        # Recommendations
        if results['recommendations']:
            logger.info("\n💡 Recommendations:")
            for rec in results['recommendations']:
                logger.info(f"  • {rec}")
        
        logger.info("=" * 70)


def main():
    """Main entry point for adaptive quality gates."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Adaptive Quality Gates for Protein Diffusion Design Lab")
    parser.add_argument('--output', default='adaptive_quality_gates_results.json',
                       help='Output file for results')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Create and run adaptive quality gates
    runner = AdaptiveQualityGateRunner(verbose=args.verbose)
    
    try:
        results = runner.run_all_gates()
        runner.print_summary(results)
        
        # Save results
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"📄 Results saved to {args.output}")
        
        # Exit with appropriate code
        return 0 if results['overall_success'] else 1
        
    except KeyboardInterrupt:
        logger.warning("⚠️ Quality gates interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"💥 Quality gates failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())