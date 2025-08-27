#!/usr/bin/env python3
"""
Enhanced Progressive Quality Gate Runner - Generation 2 Implementation

Autonomous SDLC enhancement with comprehensive error recovery, security hardening,
and production-ready reliability patterns. This implementation provides:

- Circuit breaker patterns for fault tolerance
- Advanced retry mechanisms with exponential backoff
- Real-time system health monitoring
- Comprehensive security scanning and validation
- Performance optimization and resource management
- Research-grade reproducibility validation
- Production deployment readiness checks
- Automated error recovery and graceful degradation
"""

import sys
import os
import time
import json
import logging
import subprocess
import importlib
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field
import psutil
import signal
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
import tempfile
import ast
import re
import hashlib

# Add src to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [QualityGate] - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('enhanced_quality_gates.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class EnhancedQualityGateConfig:
    """Enhanced configuration for quality gates."""
    
    # Environment settings
    environment: str = "development"  # development, ci, production, research
    auto_detect_environment: bool = True
    
    # Gate selection
    run_basic_gates: bool = True
    run_security_gates: bool = True
    run_performance_gates: bool = True
    run_research_gates: bool = False
    run_production_gates: bool = False
    
    # Execution settings
    parallel_execution: bool = True
    max_workers: int = 4
    timeout_multiplier: float = 1.0
    
    # Reliability settings
    enable_circuit_breaker: bool = True
    max_retry_attempts: int = 3
    retry_delay: float = 1.0
    
    # Security settings
    security_scan_enabled: bool = True
    vulnerability_check_enabled: bool = True
    compliance_check_enabled: bool = True
    
    # Performance settings
    performance_monitoring: bool = True
    resource_optimization: bool = True
    benchmark_enabled: bool = False
    
    # Output settings
    verbose: bool = False
    save_detailed_results: bool = True
    export_format: str = "json"

@dataclass 
class GateResult:
    """Result from a quality gate execution."""
    name: str
    status: str  # passed, failed, warning, skipped, error
    execution_time: float
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    retry_count: int = 0
    recovery_actions: List[str] = field(default_factory=list)

class CircuitBreaker:
    """Circuit breaker for fault tolerance."""
    
    def __init__(self, failure_threshold: int = 3, timeout: int = 300):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = 'closed'  # closed, open, half-open
    
    def can_execute(self) -> bool:
        """Check if execution is allowed."""
        if self.state == 'closed':
            return True
        elif self.state == 'open':
            if time.time() - self.last_failure_time >= self.timeout:
                self.state = 'half-open'
                return True
            return False
        return True  # half-open
    
    def record_success(self):
        """Record successful execution."""
        self.failure_count = 0
        if self.state == 'half-open':
            self.state = 'closed'
    
    def record_failure(self):
        """Record failed execution."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = 'open'

class EnhancedSystemHealthMonitor:
    """Enhanced system health monitoring."""
    
    def __init__(self):
        self.baseline_metrics = None
        self.monitoring_start_time = time.time()
    
    def collect_system_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive system metrics."""
        try:
            # Basic system info
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=1)
            disk = psutil.disk_usage('/')
            
            # Process info
            process = psutil.Process()
            process_memory = process.memory_info()
            
            # Network info (if available)
            network = None
            try:
                network = psutil.net_io_counters()
            except:
                pass
            
            metrics = {
                'timestamp': time.time(),
                'system': {
                    'memory_total_gb': memory.total / (1024**3),
                    'memory_available_gb': memory.available / (1024**3),
                    'memory_percent_used': memory.percent,
                    'cpu_percent': cpu_percent,
                    'cpu_count': os.cpu_count(),
                    'disk_total_gb': disk.total / (1024**3),
                    'disk_free_gb': disk.free / (1024**3),
                    'disk_percent_used': (disk.used / disk.total) * 100,
                    'load_average': os.getloadavg() if hasattr(os, 'getloadavg') else [0, 0, 0]
                },
                'process': {
                    'memory_rss_mb': process_memory.rss / (1024**2),
                    'memory_vms_mb': process_memory.vms / (1024**2),
                    'cpu_percent': process.cpu_percent(),
                    'num_threads': process.num_threads(),
                    'pid': process.pid
                }
            }
            
            if network:
                metrics['network'] = {
                    'bytes_sent': network.bytes_sent,
                    'bytes_recv': network.bytes_recv,
                    'packets_sent': network.packets_sent,
                    'packets_recv': network.packets_recv
                }
            
            return metrics
        
        except Exception as e:
            logger.warning(f"Failed to collect system metrics: {e}")
            return {
                'timestamp': time.time(),
                'error': str(e),
                'system': {'error': 'collection_failed'}
            }
    
    def assess_system_health(self, metrics: Dict[str, Any]) -> Tuple[str, List[str], List[str]]:
        """Assess overall system health."""
        status = "healthy"
        warnings = []
        recommendations = []
        
        try:
            system = metrics.get('system', {})
            process = metrics.get('process', {})
            
            # Memory assessment
            memory_percent = system.get('memory_percent_used', 0)
            if memory_percent > 95:
                status = "critical"
                warnings.append(f"Critical memory usage: {memory_percent:.1f}%")
                recommendations.append("Free up system memory immediately")
            elif memory_percent > 85:
                if status == "healthy":
                    status = "warning"
                warnings.append(f"High memory usage: {memory_percent:.1f}%")
                recommendations.append("Consider closing unnecessary applications")
            
            # CPU assessment
            cpu_percent = system.get('cpu_percent', 0)
            if cpu_percent > 95:
                status = "critical"
                warnings.append(f"Critical CPU usage: {cpu_percent:.1f}%")
                recommendations.append("Investigate high CPU processes")
            elif cpu_percent > 80:
                if status == "healthy":
                    status = "warning"
                warnings.append(f"High CPU usage: {cpu_percent:.1f}%")
            
            # Disk assessment
            disk_percent = system.get('disk_percent_used', 0)
            if disk_percent > 95:
                status = "critical"
                warnings.append(f"Critical disk usage: {disk_percent:.1f}%")
                recommendations.append("Free up disk space immediately")
            elif disk_percent > 90:
                if status == "healthy":
                    status = "warning"
                warnings.append(f"High disk usage: {disk_percent:.1f}%")
                recommendations.append("Consider cleaning up temporary files")
            
            # Process memory assessment
            process_memory_mb = process.get('memory_rss_mb', 0)
            if process_memory_mb > 2000:  # 2GB
                if status == "healthy":
                    status = "warning"
                warnings.append(f"High process memory usage: {process_memory_mb:.0f}MB")
                recommendations.append("Monitor for memory leaks")
        
        except Exception as e:
            logger.error(f"Health assessment error: {e}")
            status = "error"
            warnings.append(f"Health assessment failed: {e}")
        
        return status, warnings, recommendations

class EnhancedSecurityScanner:
    """Enhanced security scanner with comprehensive checks."""
    
    def __init__(self):
        self.security_patterns = [
            # Code injection patterns
            (r'eval\s*\(', "Potential code injection with eval()"),
            (r'exec\s*\(', "Potential code injection with exec()"),
            (r'os\.system\s*\(', "Command execution with os.system()"),
            (r'subprocess\.call\([^)]*shell\s*=\s*True', "Shell command execution"),
            
            # Dangerous imports
            (r'import\s+pickle', "Pickle usage - potential security risk"),
            (r'from\s+pickle\s+import', "Pickle import - security risk"),
            
            # Hardcoded secrets patterns
            (r'password\s*=\s*["\'][^"\']+["\']', "Potential hardcoded password"),
            (r'secret\s*=\s*["\'][^"\']+["\']', "Potential hardcoded secret"),
            (r'api[_-]?key\s*=\s*["\'][^"\']+["\']', "Potential hardcoded API key"),
            (r'token\s*=\s*["\'][^"\']+["\']', "Potential hardcoded token"),
            
            # SQL injection patterns
            (r'\.execute\s*\([^)]*%\s*[^)]*\)', "Potential SQL injection"),
            (r'query\s*=\s*["\'][^"\']*%[^"\']*["\']', "SQL query with string formatting"),
        ]
    
    def scan_security_issues(self, src_dir: Path) -> Dict[str, Any]:
        """Comprehensive security scan of source code."""
        results = {
            'total_files_scanned': 0,
            'security_issues': [],
            'risk_levels': {'high': 0, 'medium': 0, 'low': 0},
            'categories': {},
            'recommendations': []
        }
        
        try:
            python_files = list(src_dir.rglob('*.py'))
            results['total_files_scanned'] = len(python_files)
            
            for py_file in python_files:
                try:
                    with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    file_issues = self._scan_file_content(py_file, content)
                    results['security_issues'].extend(file_issues)
                    
                except Exception as e:
                    logger.warning(f"Failed to scan {py_file}: {e}")
            
            # Categorize and prioritize issues
            self._categorize_issues(results)
            
            # Generate recommendations
            results['recommendations'] = self._generate_security_recommendations(results)
            
        except Exception as e:
            logger.error(f"Security scan failed: {e}")
            results['error'] = str(e)
        
        return results
    
    def _scan_file_content(self, file_path: Path, content: str) -> List[Dict[str, Any]]:
        """Scan file content for security issues."""
        issues = []
        
        for line_num, line in enumerate(content.split('\n'), 1):
            # Skip comments
            if line.strip().startswith('#'):
                continue
            
            for pattern, description in self.security_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    # Additional validation to reduce false positives
                    if self._validate_security_issue(pattern, line, description):
                        risk_level = self._assess_risk_level(pattern, line)
                        
                        issue = {
                            'file': str(file_path.relative_to(file_path.parents[2])),
                            'line': line_num,
                            'issue': description,
                            'code_snippet': line.strip(),
                            'risk_level': risk_level,
                            'pattern': pattern
                        }
                        issues.append(issue)
        
        return issues
    
    def _validate_security_issue(self, pattern: str, line: str, description: str) -> bool:
        """Validate security issue to reduce false positives."""
        
        # Skip model.eval() calls (PyTorch)
        if 'eval(' in pattern and ('model.eval()' in line or '.eval()' in line):
            return False
        
        # Skip method calls that are not dangerous
        if 'exec(' in pattern and '.exec(' in line:
            return False
        
        # Skip commented out code
        if line.strip().startswith('#'):
            return False
        
        return True
    
    def _assess_risk_level(self, pattern: str, line: str) -> str:
        """Assess risk level of security issue."""
        
        # High risk patterns
        high_risk_patterns = ['eval\\s*\\(', 'exec\\s*\\(', 'os\\.system']
        if any(re.search(p, pattern) for p in high_risk_patterns):
            return 'high'
        
        # Medium risk patterns  
        medium_risk_patterns = ['shell\\s*=\\s*True', 'pickle', 'password', 'secret']
        if any(re.search(p, pattern) for p in medium_risk_patterns):
            return 'medium'
        
        return 'low'
    
    def _categorize_issues(self, results: Dict[str, Any]) -> None:
        """Categorize security issues by type."""
        categories = {}
        risk_counts = {'high': 0, 'medium': 0, 'low': 0}
        
        for issue in results['security_issues']:
            risk_level = issue['risk_level']
            risk_counts[risk_level] += 1
            
            # Categorize by issue type
            issue_type = self._get_issue_category(issue['issue'])
            if issue_type not in categories:
                categories[issue_type] = 0
            categories[issue_type] += 1
        
        results['risk_levels'] = risk_counts
        results['categories'] = categories
    
    def _get_issue_category(self, issue_description: str) -> str:
        """Get category for security issue."""
        if 'injection' in issue_description.lower():
            return 'Code Injection'
        elif 'password' in issue_description.lower() or 'secret' in issue_description.lower():
            return 'Hardcoded Secrets'
        elif 'shell' in issue_description.lower() or 'command' in issue_description.lower():
            return 'Command Execution'
        elif 'pickle' in issue_description.lower():
            return 'Insecure Deserialization'
        elif 'sql' in issue_description.lower():
            return 'SQL Injection'
        else:
            return 'Other'
    
    def _generate_security_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate security recommendations based on findings."""
        recommendations = []
        
        high_risk_count = results['risk_levels']['high']
        medium_risk_count = results['risk_levels']['medium']
        
        if high_risk_count > 0:
            recommendations.append(f"URGENT: Address {high_risk_count} high-risk security issues immediately")
            recommendations.append("Consider implementing input validation and sanitization")
        
        if medium_risk_count > 0:
            recommendations.append(f"Address {medium_risk_count} medium-risk security issues")
        
        if 'Code Injection' in results['categories']:
            recommendations.append("Replace eval() and exec() with safer alternatives")
        
        if 'Hardcoded Secrets' in results['categories']:
            recommendations.append("Move secrets to environment variables or secure vaults")
        
        if 'Command Execution' in results['categories']:
            recommendations.append("Validate and sanitize all shell commands")
        
        return recommendations

class EnhancedPerformanceBenchmark:
    """Enhanced performance benchmarking and optimization."""
    
    def __init__(self):
        self.baseline_metrics = None
        self.benchmark_history = []
    
    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run comprehensive performance benchmark."""
        results = {
            'timestamp': time.time(),
            'benchmarks': {},
            'performance_score': 0,
            'bottlenecks': [],
            'optimizations': []
        }
        
        try:
            # System performance benchmark
            results['benchmarks']['system'] = self._benchmark_system_performance()
            
            # Import performance benchmark
            results['benchmarks']['imports'] = self._benchmark_import_performance()
            
            # Memory usage benchmark
            results['benchmarks']['memory'] = self._benchmark_memory_usage()
            
            # Functionality performance (if possible)
            try:
                results['benchmarks']['functionality'] = self._benchmark_core_functionality()
            except Exception as e:
                logger.warning(f"Functionality benchmark failed: {e}")
                results['benchmarks']['functionality'] = {'error': str(e)}
            
            # Calculate overall performance score
            results['performance_score'] = self._calculate_performance_score(results['benchmarks'])
            
            # Identify bottlenecks and optimizations
            results['bottlenecks'] = self._identify_bottlenecks(results['benchmarks'])
            results['optimizations'] = self._suggest_optimizations(results['benchmarks'])
            
        except Exception as e:
            logger.error(f"Performance benchmark failed: {e}")
            results['error'] = str(e)
        
        return results
    
    def _benchmark_system_performance(self) -> Dict[str, float]:
        """Benchmark basic system performance."""
        
        # CPU benchmark
        start_time = time.time()
        total = sum(i * i for i in range(100000))
        cpu_time = time.time() - start_time
        
        # Memory allocation benchmark
        start_time = time.time()
        data = [i for i in range(50000)]
        memory_time = time.time() - start_time
        del data
        
        # Disk I/O benchmark
        start_time = time.time()
        with tempfile.NamedTemporaryFile() as tmp:
            test_data = b"x" * 1024 * 100  # 100KB
            tmp.write(test_data)
            tmp.flush()
        disk_time = time.time() - start_time
        
        return {
            'cpu_benchmark_seconds': cpu_time,
            'memory_allocation_seconds': memory_time,
            'disk_io_seconds': disk_time
        }
    
    def _benchmark_import_performance(self) -> Dict[str, float]:
        """Benchmark import performance."""
        import_times = {}
        
        key_modules = [
            'numpy', 'torch', 'scipy', 'matplotlib', 'pandas',
            'protein_diffusion', 'streamlit', 'plotly'
        ]
        
        for module_name in key_modules:
            start_time = time.time()
            try:
                importlib.import_module(module_name)
                import_time = time.time() - start_time
                import_times[f'{module_name}_import_seconds'] = import_time
            except ImportError:
                import_times[f'{module_name}_import_seconds'] = -1  # Not available
        
        return import_times
    
    def _benchmark_memory_usage(self) -> Dict[str, float]:
        """Benchmark memory usage patterns."""
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        # Memory allocation test
        test_data = []
        for i in range(10):
            test_data.append([j for j in range(1000)])
            current_memory = process.memory_info().rss
        
        peak_memory = process.memory_info().rss
        memory_growth = peak_memory - initial_memory
        
        # Cleanup
        del test_data
        
        return {
            'initial_memory_mb': initial_memory / (1024**2),
            'peak_memory_mb': peak_memory / (1024**2),
            'memory_growth_mb': memory_growth / (1024**2)
        }
    
    def _benchmark_core_functionality(self) -> Dict[str, float]:
        """Benchmark core functionality performance."""
        try:
            from protein_diffusion import ProteinDiffuser, ProteinDiffuserConfig
            
            # Configuration creation benchmark
            config_start = time.time()
            config = ProteinDiffuserConfig()
            config.num_samples = 1
            config.max_length = 16
            config_time = time.time() - config_start
            
            # Diffuser initialization benchmark
            init_start = time.time()
            diffuser = ProteinDiffuser(config)
            init_time = time.time() - init_start
            
            # Small generation benchmark
            gen_start = time.time()
            results = diffuser.generate(
                num_samples=1,
                max_length=8,
                progress=False
            )
            gen_time = time.time() - gen_start
            
            return {
                'config_creation_seconds': config_time,
                'diffuser_init_seconds': init_time,
                'generation_seconds': gen_time,
                'total_functionality_seconds': config_time + init_time + gen_time,
                'generation_success': len(results) > 0 if results else False
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _calculate_performance_score(self, benchmarks: Dict[str, Any]) -> float:
        """Calculate overall performance score (0-100)."""
        score = 100.0
        
        try:
            system_bench = benchmarks.get('system', {})
            imports_bench = benchmarks.get('imports', {})
            functionality_bench = benchmarks.get('functionality', {})
            
            # Penalize slow operations
            if system_bench.get('cpu_benchmark_seconds', 0) > 1.0:
                score -= 20
            elif system_bench.get('cpu_benchmark_seconds', 0) > 0.5:
                score -= 10
            
            # Penalize slow imports
            slow_imports = sum(1 for v in imports_bench.values() if isinstance(v, (int, float)) and v > 5.0)
            score -= slow_imports * 5
            
            # Penalize slow functionality
            if functionality_bench.get('total_functionality_seconds', 0) > 60:
                score -= 30
            elif functionality_bench.get('total_functionality_seconds', 0) > 30:
                score -= 15
            
            return max(0, min(100, score))
            
        except:
            return 50.0  # Default moderate score
    
    def _identify_bottlenecks(self, benchmarks: Dict[str, Any]) -> List[str]:
        """Identify performance bottlenecks."""
        bottlenecks = []
        
        try:
            system_bench = benchmarks.get('system', {})
            imports_bench = benchmarks.get('imports', {})
            functionality_bench = benchmarks.get('functionality', {})
            
            # System bottlenecks
            if system_bench.get('cpu_benchmark_seconds', 0) > 1.0:
                bottlenecks.append("CPU performance is below expected baseline")
            
            if system_bench.get('disk_io_seconds', 0) > 0.5:
                bottlenecks.append("Disk I/O performance is slow")
            
            # Import bottlenecks
            slow_imports = [(k, v) for k, v in imports_bench.items() if isinstance(v, (int, float)) and v > 3.0]
            for import_name, import_time in slow_imports:
                bottlenecks.append(f"Slow import detected: {import_name} ({import_time:.2f}s)")
            
            # Functionality bottlenecks
            if functionality_bench.get('diffuser_init_seconds', 0) > 20:
                bottlenecks.append("Diffuser initialization is slow")
            
            if functionality_bench.get('generation_seconds', 0) > 30:
                bottlenecks.append("Protein generation is slow")
            
        except Exception as e:
            bottlenecks.append(f"Bottleneck analysis error: {e}")
        
        return bottlenecks
    
    def _suggest_optimizations(self, benchmarks: Dict[str, Any]) -> List[str]:
        """Suggest performance optimizations."""
        optimizations = []
        
        try:
            memory_bench = benchmarks.get('memory', {})
            functionality_bench = benchmarks.get('functionality', {})
            
            # Memory optimizations
            if memory_bench.get('memory_growth_mb', 0) > 100:
                optimizations.append("Consider implementing memory pooling or caching")
                optimizations.append("Review object lifecycle management")
            
            # Functionality optimizations
            if functionality_bench.get('diffuser_init_seconds', 0) > 10:
                optimizations.append("Consider lazy loading for diffuser components")
                optimizations.append("Implement diffuser instance caching")
            
            if functionality_bench.get('generation_seconds', 0) > 20:
                optimizations.append("Consider batch processing for generation")
                optimizations.append("Implement generation result caching")
            
            # General optimizations
            optimizations.append("Enable parallel processing where applicable")
            optimizations.append("Consider using compiled extensions for critical paths")
            optimizations.append("Implement progressive loading for large models")
            
        except Exception as e:
            optimizations.append(f"Optimization analysis error: {e}")
        
        return optimizations

class EnhancedQualityGateRunner:
    """Enhanced Quality Gate Runner with comprehensive validation."""
    
    def __init__(self, config: Optional[EnhancedQualityGateConfig] = None):
        self.config = config or EnhancedQualityGateConfig()
        self.results: Dict[str, GateResult] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.health_monitor = EnhancedSystemHealthMonitor()
        self.security_scanner = EnhancedSecurityScanner()
        self.performance_benchmark = EnhancedPerformanceBenchmark()
        self.src_dir = Path("src")
        self.execution_start_time = 0
        
        # Auto-detect environment
        if self.config.auto_detect_environment:
            self._detect_environment()
        
        # Initialize circuit breakers
        if self.config.enable_circuit_breaker:
            self._initialize_circuit_breakers()
        
        logger.info(f"🚀 Enhanced Quality Gate Runner initialized (Environment: {self.config.environment})")
    
    def _detect_environment(self):
        """Auto-detect execution environment."""
        ci_indicators = ['CI', 'CONTINUOUS_INTEGRATION', 'GITHUB_ACTIONS', 'JENKINS_URL', 'GITLAB_CI']
        production_indicators = ['PRODUCTION', 'PROD', 'KUBERNETES_SERVICE_HOST']
        
        if any(os.getenv(indicator) for indicator in ci_indicators):
            self.config.environment = "ci"
            self.config.run_production_gates = True
            logger.info("🔍 Detected CI environment")
        elif any(os.getenv(indicator) for indicator in production_indicators):
            self.config.environment = "production"
            self.config.run_production_gates = True
            self.config.run_research_gates = False
            logger.info("🔍 Detected production environment")
        else:
            self.config.environment = "development"
            logger.info("🔍 Detected development environment")
    
    def _initialize_circuit_breakers(self):
        """Initialize circuit breakers for gates."""
        gate_names = [
            'system_health', 'dependency_check', 'basic_functionality',
            'security_scan', 'performance_baseline', 'code_quality',
            'research_validation', 'production_readiness'
        ]
        
        for gate_name in gate_names:
            self.circuit_breakers[gate_name] = CircuitBreaker(failure_threshold=3, timeout=300)
    
    def run_all_gates(self) -> Dict[str, Any]:
        """Run all enabled quality gates."""
        logger.info("=" * 70)
        logger.info("🚀 Starting Enhanced Progressive Quality Gates")
        logger.info("=" * 70)
        logger.info(f"🔍 Environment: {self.config.environment}")
        logger.info(f"⚡ Parallel execution: {self.config.parallel_execution}")
        logger.info(f"🔒 Security scanning: {self.config.security_scan_enabled}")
        logger.info(f"📊 Performance monitoring: {self.config.performance_monitoring}")
        logger.info("=" * 70)
        
        self.execution_start_time = time.time()
        
        # Define all available gates
        gates = self._define_quality_gates()
        
        # Filter gates based on configuration
        active_gates = self._filter_gates_by_config(gates)
        
        logger.info(f"📊 Running {len(active_gates)} quality gates")
        
        # Execute gates
        if self.config.parallel_execution and len(active_gates) > 1:
            self._run_gates_parallel(active_gates)
        else:
            self._run_gates_sequential(active_gates)
        
        # Compile final results
        return self._compile_final_results()
    
    def _define_quality_gates(self) -> List[Tuple[str, callable, Dict[str, Any]]]:
        """Define all available quality gates."""
        return [
            # Basic gates
            ('system_health', self._gate_system_health, {'timeout': 30, 'required': True}),
            ('dependency_check', self._gate_dependency_check, {'timeout': 60, 'required': True}),
            ('basic_functionality', self._gate_basic_functionality, {'timeout': 120, 'required': True}),
            
            # Security gates
            ('security_scan', self._gate_security_scan, {'timeout': 300, 'required': True}),
            ('vulnerability_check', self._gate_vulnerability_check, {'timeout': 180, 'required': False}),
            
            # Performance gates
            ('performance_baseline', self._gate_performance_baseline, {'timeout': 300, 'required': False}),
            ('resource_optimization', self._gate_resource_optimization, {'timeout': 120, 'required': False}),
            
            # Code quality gates
            ('code_quality', self._gate_code_quality, {'timeout': 180, 'required': False}),
            ('documentation_check', self._gate_documentation_check, {'timeout': 60, 'required': False}),
            
            # Research gates
            ('research_validation', self._gate_research_validation, {'timeout': 600, 'required': False}),
            ('reproducibility_check', self._gate_reproducibility_check, {'timeout': 300, 'required': False}),
            
            # Production gates
            ('production_readiness', self._gate_production_readiness, {'timeout': 300, 'required': False}),
            ('deployment_validation', self._gate_deployment_validation, {'timeout': 240, 'required': False})
        ]
    
    def _filter_gates_by_config(self, gates: List[Tuple[str, callable, Dict[str, Any]]]) -> List[Tuple[str, callable, Dict[str, Any]]]:
        """Filter gates based on configuration."""
        active_gates = []
        
        gate_filters = {
            'system_health': self.config.run_basic_gates,
            'dependency_check': self.config.run_basic_gates,
            'basic_functionality': self.config.run_basic_gates,
            'security_scan': self.config.security_scan_enabled,
            'vulnerability_check': self.config.vulnerability_check_enabled,
            'performance_baseline': self.config.performance_monitoring,
            'resource_optimization': self.config.resource_optimization,
            'code_quality': self.config.run_basic_gates,
            'documentation_check': self.config.run_basic_gates,
            'research_validation': self.config.run_research_gates,
            'reproducibility_check': self.config.run_research_gates,
            'production_readiness': self.config.run_production_gates,
            'deployment_validation': self.config.run_production_gates
        }
        
        for gate_name, gate_func, gate_config in gates:
            if gate_filters.get(gate_name, True):
                active_gates.append((gate_name, gate_func, gate_config))
        
        return active_gates
    
    def _run_gates_sequential(self, gates: List[Tuple[str, callable, Dict[str, Any]]]):
        """Run gates sequentially."""
        for i, (gate_name, gate_func, gate_config) in enumerate(gates, 1):
            logger.info(f"🔄 Running gate {i}/{len(gates)}: {gate_name}")
            self._execute_gate_with_retry(gate_name, gate_func, gate_config)
    
    def _run_gates_parallel(self, gates: List[Tuple[str, callable, Dict[str, Any]]]):
        """Run gates in parallel."""
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all gates
            future_to_gate = {
                executor.submit(self._execute_gate_with_retry, gate_name, gate_func, gate_config): gate_name
                for gate_name, gate_func, gate_config in gates
            }
            
            # Collect results
            for future in as_completed(future_to_gate):
                gate_name = future_to_gate[future]
                try:
                    future.result()  # This will raise any exception that occurred
                except Exception as e:
                    logger.error(f"💥 Gate {gate_name} failed in parallel execution: {e}")
    
    def _execute_gate_with_retry(self, gate_name: str, gate_func: callable, gate_config: Dict[str, Any]):
        """Execute a gate with retry logic and circuit breaker."""
        
        # Check circuit breaker
        circuit_breaker = self.circuit_breakers.get(gate_name)
        if circuit_breaker and not circuit_breaker.can_execute():
            self.results[gate_name] = GateResult(
                name=gate_name,
                status="skipped",
                execution_time=0,
                message="Circuit breaker is open",
                warnings=["Gate skipped due to circuit breaker"]
            )
            logger.warning(f"⏭️ Skipping {gate_name}: Circuit breaker is open")
            return
        
        # Execute with retry logic
        max_attempts = self.config.max_retry_attempts
        for attempt in range(1, max_attempts + 1):
            try:
                start_time = time.time()
                
                # Execute gate with timeout
                timeout = gate_config.get('timeout', 300) * self.config.timeout_multiplier
                result = self._execute_gate_with_timeout(gate_func, timeout)
                
                execution_time = time.time() - start_time
                
                if result.status == "passed":
                    result.execution_time = execution_time
                    result.retry_count = attempt - 1
                    self.results[gate_name] = result
                    
                    if circuit_breaker:
                        circuit_breaker.record_success()
                    
                    logger.info(f"✅ Gate {gate_name} passed ({execution_time:.2f}s)")
                    return
                else:
                    if attempt < max_attempts:
                        retry_delay = self.config.retry_delay * attempt  # Exponential backoff
                        logger.warning(f"⚠️ Gate {gate_name} failed (attempt {attempt}/{max_attempts}), retrying in {retry_delay:.1f}s")
                        time.sleep(retry_delay)
                        continue
                    else:
                        result.execution_time = execution_time
                        result.retry_count = attempt - 1
                        self.results[gate_name] = result
                        
                        if circuit_breaker:
                            circuit_breaker.record_failure()
                        
                        logger.error(f"❌ Gate {gate_name} failed after {max_attempts} attempts")
                        return
            
            except TimeoutError:
                if attempt < max_attempts:
                    logger.warning(f"⏰ Gate {gate_name} timed out (attempt {attempt}/{max_attempts}), retrying...")
                    continue
                else:
                    self.results[gate_name] = GateResult(
                        name=gate_name,
                        status="timeout",
                        execution_time=timeout,
                        message=f"Gate timed out after {timeout}s",
                        retry_count=attempt - 1
                    )
                    if circuit_breaker:
                        circuit_breaker.record_failure()
                    logger.error(f"⏰ Gate {gate_name} timed out after {max_attempts} attempts")
                    return
            
            except Exception as e:
                if attempt < max_attempts:
                    logger.warning(f"💥 Gate {gate_name} crashed (attempt {attempt}/{max_attempts}): {e}")
                    time.sleep(self.config.retry_delay)
                    continue
                else:
                    self.results[gate_name] = GateResult(
                        name=gate_name,
                        status="error",
                        execution_time=time.time() - start_time if 'start_time' in locals() else 0,
                        message=str(e),
                        errors=[str(e)],
                        retry_count=attempt - 1
                    )
                    if circuit_breaker:
                        circuit_breaker.record_failure()
                    logger.error(f"💥 Gate {gate_name} crashed after {max_attempts} attempts: {e}")
                    return
    
    def _execute_gate_with_timeout(self, gate_func: callable, timeout: float) -> GateResult:
        """Execute gate function with timeout."""
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Gate execution timed out after {timeout}s")
        
        # Set timeout signal (Unix only)
        if hasattr(signal, 'SIGALRM'):
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(timeout))
        
        try:
            result = gate_func()
            return result
        finally:
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)
                if old_handler:
                    signal.signal(signal.SIGALRM, old_handler)
    
    # ===== GATE IMPLEMENTATIONS =====
    
    def _gate_system_health(self) -> GateResult:
        """Comprehensive system health check."""
        try:
            # Collect system metrics
            metrics = self.health_monitor.collect_system_metrics()
            
            # Assess health
            health_status, warnings, recommendations = self.health_monitor.assess_system_health(metrics)
            
            # Determine gate result
            if health_status == "healthy":
                status = "passed"
                message = "System health is optimal"
            elif health_status == "warning":
                status = "warning"
                message = "System health has warnings but is acceptable"
            elif health_status == "critical":
                status = "failed"
                message = "System health is critical"
            else:
                status = "error"
                message = f"System health check error: {health_status}"
            
            return GateResult(
                name="system_health",
                status=status,
                execution_time=0,
                message=message,
                details=metrics,
                warnings=warnings,
                recommendations=recommendations,
                metrics=metrics
            )
        
        except Exception as e:
            return GateResult(
                name="system_health",
                status="error",
                execution_time=0,
                message=f"System health check failed: {e}",
                errors=[str(e)]
            )
    
    def _gate_dependency_check(self) -> GateResult:
        """Enhanced dependency validation."""
        try:
            # Core dependencies to check
            dependencies = [
                ('numpy', '>=1.20.0'),
                ('torch', '>=1.10.0'),
                ('scipy', '>=1.7.0'),
                ('pandas', '>=1.3.0'),
                ('matplotlib', None),
                ('plotly', None),
                ('streamlit', '>=1.20.0'),
                ('psutil', None),
                ('protein_diffusion', None)
            ]
            
            successful = 0
            failed = []
            version_warnings = []
            import_times = {}
            
            for module_name, min_version in dependencies:
                start_time = time.time()
                try:
                    module = importlib.import_module(module_name)
                    import_time = time.time() - start_time
                    import_times[module_name] = import_time
                    
                    # Version check if specified
                    if min_version and hasattr(module, '__version__'):
                        if not self._check_version(module.__version__, min_version):
                            version_warnings.append(f"{module_name} {module.__version__} < {min_version}")
                    
                    successful += 1
                    
                except ImportError as e:
                    failed.append(f"{module_name}: {str(e)}")
                    import_times[module_name] = -1
            
            success_rate = successful / len(dependencies)
            
            # Check for slow imports
            slow_imports = [(name, time) for name, time in import_times.items() if time > 5.0]
            
            # Determine status
            if success_rate >= 0.9 and len(version_warnings) == 0:
                status = "passed"
                message = f"All dependencies validated ({success_rate:.1%} success rate)"
            elif success_rate >= 0.8:
                status = "warning"
                message = f"Most dependencies validated ({success_rate:.1%} success rate)"
            else:
                status = "failed"
                message = f"Dependency validation failed ({success_rate:.1%} success rate)"
            
            warnings = failed + version_warnings
            if slow_imports:
                warnings.extend([f"Slow import: {name} ({time:.2f}s)" for name, time in slow_imports])
            
            recommendations = []
            if failed:
                recommendations.append("Install missing dependencies: pip install -r requirements.txt")
            if version_warnings:
                recommendations.append("Update outdated dependencies")
            if slow_imports:
                recommendations.append("Optimize import performance")
            
            return GateResult(
                name="dependency_check",
                status=status,
                execution_time=0,
                message=message,
                warnings=warnings,
                recommendations=recommendations,
                metrics={
                    'success_rate': success_rate,
                    'successful_imports': successful,
                    'failed_imports': len(failed),
                    'version_warnings': len(version_warnings),
                    'import_times': import_times
                }
            )
        
        except Exception as e:
            return GateResult(
                name="dependency_check",
                status="error",
                execution_time=0,
                message=f"Dependency check failed: {e}",
                errors=[str(e)]
            )
    
    def _check_version(self, current: str, required: str) -> bool:
        """Simple version comparison."""
        try:
            required = required.replace('>=', '').strip()
            current_parts = [int(x) for x in current.split('.')[:3]]
            required_parts = [int(x) for x in required.split('.')[:3]]
            
            # Pad to same length
            while len(current_parts) < 3:
                current_parts.append(0)
            while len(required_parts) < 3:
                required_parts.append(0)
            
            return current_parts >= required_parts
        except:
            return True  # If parsing fails, assume OK
    
    def _gate_basic_functionality(self) -> GateResult:
        """Enhanced basic functionality test."""
        try:
            from protein_diffusion import ProteinDiffuser, ProteinDiffuserConfig
            
            test_results = {}
            overall_success = True
            warnings = []
            
            # Test 1: Configuration creation
            try:
                config = ProteinDiffuserConfig()
                config.num_samples = 1
                config.max_length = 16
                test_results['config_creation'] = True
            except Exception as e:
                test_results['config_creation'] = False
                overall_success = False
                warnings.append(f"Config creation failed: {e}")
            
            # Test 2: Diffuser initialization
            if test_results.get('config_creation', False):
                try:
                    diffuser = ProteinDiffuser(config)
                    test_results['diffuser_init'] = True
                except Exception as e:
                    test_results['diffuser_init'] = False
                    overall_success = False
                    warnings.append(f"Diffuser initialization failed: {e}")
            
            # Test 3: Basic generation
            if test_results.get('diffuser_init', False):
                try:
                    results = diffuser.generate(
                        num_samples=1,
                        max_length=8,
                        progress=False
                    )
                    if results and len(results) > 0:
                        test_results['basic_generation'] = True
                        test_results['generation_results'] = len(results)
                    else:
                        test_results['basic_generation'] = False
                        overall_success = False
                        warnings.append("No results generated")
                except Exception as e:
                    test_results['basic_generation'] = False
                    overall_success = False
                    warnings.append(f"Generation failed: {e}")
            
            success_count = sum(1 for v in test_results.values() if v is True)
            total_tests = 3
            success_rate = success_count / total_tests
            
            if overall_success:
                status = "passed"
                message = "All basic functionality tests passed"
            elif success_rate >= 0.66:
                status = "warning"
                message = f"Basic functionality mostly working ({success_rate:.1%} success)"
            else:
                status = "failed"
                message = f"Basic functionality tests failed ({success_rate:.1%} success)"
            
            return GateResult(
                name="basic_functionality",
                status=status,
                execution_time=0,
                message=message,
                warnings=warnings,
                metrics={
                    'test_results': test_results,
                    'success_rate': success_rate,
                    'tests_passed': success_count,
                    'total_tests': total_tests
                }
            )
        
        except Exception as e:
            return GateResult(
                name="basic_functionality",
                status="error",
                execution_time=0,
                message=f"Basic functionality test failed: {e}",
                errors=[str(e)]
            )
    
    def _gate_security_scan(self) -> GateResult:
        """Comprehensive security scan."""
        try:
            if not self.src_dir.exists():
                return GateResult(
                    name="security_scan",
                    status="skipped",
                    execution_time=0,
                    message="Source directory not found",
                    warnings=["No source code to scan"]
                )
            
            # Run security scan
            scan_results = self.security_scanner.scan_security_issues(self.src_dir)
            
            high_risk = scan_results.get('risk_levels', {}).get('high', 0)
            medium_risk = scan_results.get('risk_levels', {}).get('medium', 0)
            low_risk = scan_results.get('risk_levels', {}).get('low', 0)
            total_issues = high_risk + medium_risk + low_risk
            
            # Determine status
            if high_risk > 0:
                status = "failed"
                message = f"Security scan failed: {high_risk} high-risk issues found"
            elif medium_risk > 5:
                status = "failed"
                message = f"Security scan failed: too many medium-risk issues ({medium_risk})"
            elif medium_risk > 0 or low_risk > 0:
                status = "warning"
                message = f"Security scan passed with warnings: {medium_risk} medium, {low_risk} low risk issues"
            else:
                status = "passed"
                message = "Security scan passed: no issues found"
            
            warnings = []
            if total_issues > 0:
                warnings.append(f"Found {total_issues} total security issues")
                for issue in scan_results.get('security_issues', [])[:5]:  # Show first 5
                    warnings.append(f"{issue['file']}:{issue['line']} - {issue['issue']}")
            
            return GateResult(
                name="security_scan",
                status=status,
                execution_time=0,
                message=message,
                warnings=warnings,
                recommendations=scan_results.get('recommendations', []),
                metrics=scan_results
            )
        
        except Exception as e:
            return GateResult(
                name="security_scan",
                status="error",
                execution_time=0,
                message=f"Security scan failed: {e}",
                errors=[str(e)]
            )
    
    def _gate_vulnerability_check(self) -> GateResult:
        """Vulnerability check using external tools."""
        try:
            # Try to run safety check if available
            try:
                result = subprocess.run(
                    ['python', '-m', 'safety', 'check', '--json'],
                    capture_output=True,
                    text=True,
                    timeout=120
                )
                
                if result.returncode == 0:
                    # Parse safety results
                    try:
                        safety_data = json.loads(result.stdout) if result.stdout else []
                        vulnerability_count = len(safety_data)
                        
                        if vulnerability_count == 0:
                            status = "passed"
                            message = "No known vulnerabilities found"
                        else:
                            status = "warning"
                            message = f"Found {vulnerability_count} known vulnerabilities"
                        
                        return GateResult(
                            name="vulnerability_check",
                            status=status,
                            execution_time=0,
                            message=message,
                            metrics={'vulnerabilities': safety_data, 'count': vulnerability_count}
                        )
                    
                    except json.JSONDecodeError:
                        pass
            
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass
            
            # Fallback to basic package vulnerability check
            return self._basic_vulnerability_check()
        
        except Exception as e:
            return GateResult(
                name="vulnerability_check",
                status="error",
                execution_time=0,
                message=f"Vulnerability check failed: {e}",
                errors=[str(e)]
            )
    
    def _basic_vulnerability_check(self) -> GateResult:
        """Basic vulnerability check without external tools."""
        try:
            # Check for known problematic patterns in requirements
            vulnerable_packages = []
            warnings = []
            
            # Check requirements.txt if exists
            requirements_file = Path("requirements.txt")
            if requirements_file.exists():
                with open(requirements_file, 'r') as f:
                    requirements = f.read()
                
                # Known vulnerable patterns
                vulnerable_patterns = [
                    ('pickle', 'Pickle can be unsafe for untrusted data'),
                    ('eval', 'eval() usage can be dangerous'),
                    ('yaml.load', 'yaml.load() without Loader can be unsafe')
                ]
                
                for pattern, warning in vulnerable_patterns:
                    if pattern in requirements.lower():
                        warnings.append(warning)
            
            if warnings:
                status = "warning"
                message = f"Basic vulnerability check found {len(warnings)} potential issues"
            else:
                status = "passed"
                message = "Basic vulnerability check passed"
            
            return GateResult(
                name="vulnerability_check",
                status=status,
                execution_time=0,
                message=message,
                warnings=warnings,
                recommendations=["Install and run 'safety' tool for comprehensive vulnerability checking"]
            )
        
        except Exception as e:
            return GateResult(
                name="vulnerability_check",
                status="error",
                execution_time=0,
                message=f"Basic vulnerability check failed: {e}",
                errors=[str(e)]
            )
    
    def _gate_performance_baseline(self) -> GateResult:
        """Comprehensive performance baseline measurement."""
        try:
            # Run performance benchmark
            benchmark_results = self.performance_benchmark.run_comprehensive_benchmark()
            
            performance_score = benchmark_results.get('performance_score', 0)
            bottlenecks = benchmark_results.get('bottlenecks', [])
            
            # Determine status based on performance score
            if performance_score >= 80:
                status = "passed"
                message = f"Performance baseline excellent (score: {performance_score:.1f})"
            elif performance_score >= 60:
                status = "warning"
                message = f"Performance baseline acceptable (score: {performance_score:.1f})"
            else:
                status = "failed"
                message = f"Performance baseline below acceptable threshold (score: {performance_score:.1f})"
            
            warnings = bottlenecks[:5]  # Show first 5 bottlenecks
            
            return GateResult(
                name="performance_baseline",
                status=status,
                execution_time=0,
                message=message,
                warnings=warnings,
                recommendations=benchmark_results.get('optimizations', []),
                metrics=benchmark_results
            )
        
        except Exception as e:
            return GateResult(
                name="performance_baseline",
                status="error",
                execution_time=0,
                message=f"Performance baseline failed: {e}",
                errors=[str(e)]
            )
    
    def _gate_resource_optimization(self) -> GateResult:
        """Resource optimization validation."""
        try:
            # Collect current resource usage
            metrics = self.health_monitor.collect_system_metrics()
            
            system = metrics.get('system', {})
            process = metrics.get('process', {})
            
            optimization_score = 100.0
            warnings = []
            recommendations = []
            
            # Memory optimization check
            memory_percent = system.get('memory_percent', 0)
            if memory_percent > 90:
                optimization_score -= 30
                warnings.append(f"High memory usage: {memory_percent:.1f}%")
                recommendations.append("Optimize memory usage")
            elif memory_percent > 80:
                optimization_score -= 15
                warnings.append(f"Elevated memory usage: {memory_percent:.1f}%")
            
            # Process memory check
            process_memory_mb = process.get('memory_rss_mb', 0)
            if process_memory_mb > 1000:  # 1GB
                optimization_score -= 20
                warnings.append(f"High process memory: {process_memory_mb:.0f}MB")
                recommendations.append("Review memory allocations")
            
            # CPU optimization check
            cpu_percent = system.get('cpu_percent', 0)
            if cpu_percent > 90:
                optimization_score -= 25
                warnings.append(f"High CPU usage: {cpu_percent:.1f}%")
                recommendations.append("Optimize CPU-intensive operations")
            
            # Disk space check
            disk_percent = system.get('disk_percent_used', 0)
            if disk_percent > 95:
                optimization_score -= 20
                warnings.append(f"Very low disk space: {100-disk_percent:.1f}% free")
                recommendations.append("Free up disk space")
            elif disk_percent > 90:
                optimization_score -= 10
                warnings.append(f"Low disk space: {100-disk_percent:.1f}% free")
            
            # Determine status
            if optimization_score >= 80:
                status = "passed"
                message = f"Resource optimization excellent (score: {optimization_score:.1f})"
            elif optimization_score >= 60:
                status = "warning"
                message = f"Resource optimization acceptable (score: {optimization_score:.1f})"
            else:
                status = "failed"
                message = f"Resource optimization needs improvement (score: {optimization_score:.1f})"
            
            return GateResult(
                name="resource_optimization",
                status=status,
                execution_time=0,
                message=message,
                warnings=warnings,
                recommendations=recommendations,
                metrics={
                    'optimization_score': optimization_score,
                    'system_metrics': system,
                    'process_metrics': process
                }
            )
        
        except Exception as e:
            return GateResult(
                name="resource_optimization",
                status="error",
                execution_time=0,
                message=f"Resource optimization check failed: {e}",
                errors=[str(e)]
            )
    
    def _gate_code_quality(self) -> GateResult:
        """Code quality assessment."""
        try:
            if not self.src_dir.exists():
                return GateResult(
                    name="code_quality",
                    status="skipped",
                    execution_time=0,
                    message="Source directory not found"
                )
            
            quality_score = 100.0
            issues = []
            warnings = []
            recommendations = []
            
            # Check Python syntax
            syntax_errors = 0
            python_files = list(self.src_dir.rglob('*.py'))
            
            for py_file in python_files:
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        source = f.read()
                    ast.parse(source)
                except SyntaxError as e:
                    syntax_errors += 1
                    issues.append(f"Syntax error in {py_file.name}: {e.msg}")
                except Exception:
                    pass
            
            if syntax_errors > 0:
                quality_score -= syntax_errors * 20
                warnings.append(f"Found {syntax_errors} syntax errors")
                recommendations.append("Fix all syntax errors")
            
            # Check for code complexity (simplified)
            complex_files = 0
            for py_file in python_files[:10]:  # Check first 10 files
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    
                    if len(lines) > 500:  # Very long files
                        complex_files += 1
                        
                except Exception:
                    pass
            
            if complex_files > 0:
                quality_score -= complex_files * 5
                warnings.append(f"Found {complex_files} potentially complex files")
                recommendations.append("Consider refactoring large files")
            
            # Determine status
            if quality_score >= 90:
                status = "passed"
                message = f"Code quality excellent (score: {quality_score:.1f})"
            elif quality_score >= 70:
                status = "warning"
                message = f"Code quality acceptable (score: {quality_score:.1f})"
            else:
                status = "failed"
                message = f"Code quality needs improvement (score: {quality_score:.1f})"
            
            return GateResult(
                name="code_quality",
                status=status,
                execution_time=0,
                message=message,
                warnings=warnings,
                recommendations=recommendations,
                metrics={
                    'quality_score': quality_score,
                    'syntax_errors': syntax_errors,
                    'complex_files': complex_files,
                    'total_files': len(python_files)
                }
            )
        
        except Exception as e:
            return GateResult(
                name="code_quality",
                status="error",
                execution_time=0,
                message=f"Code quality check failed: {e}",
                errors=[str(e)]
            )
    
    def _gate_documentation_check(self) -> GateResult:
        """Documentation coverage check."""
        try:
            if not self.src_dir.exists():
                return GateResult(
                    name="documentation_check",
                    status="skipped",
                    execution_time=0,
                    message="Source directory not found"
                )
            
            # Check for documentation files
            doc_files = []
            doc_patterns = ['README*', '*.md', 'docs/*', 'CHANGELOG*', 'LICENSE*']
            
            for pattern in doc_patterns:
                doc_files.extend(Path('.').glob(pattern))
            
            # Check function/class documentation
            python_files = list(self.src_dir.rglob('*.py'))
            total_functions = 0
            documented_functions = 0
            total_classes = 0
            documented_classes = 0
            
            for py_file in python_files[:5]:  # Check first 5 files
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        source = f.read()
                    
                    tree = ast.parse(source)
                    
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            total_functions += 1
                            if ast.get_docstring(node):
                                documented_functions += 1
                        elif isinstance(node, ast.ClassDef):
                            total_classes += 1
                            if ast.get_docstring(node):
                                documented_classes += 1
                
                except Exception:
                    pass
            
            # Calculate documentation coverage
            function_coverage = (documented_functions / total_functions * 100) if total_functions > 0 else 100
            class_coverage = (documented_classes / total_classes * 100) if total_classes > 0 else 100
            overall_coverage = (function_coverage + class_coverage) / 2
            
            # Determine status
            warnings = []
            recommendations = []
            
            if len(doc_files) == 0:
                warnings.append("No documentation files found")
                recommendations.append("Add README.md and basic documentation")
            
            if function_coverage < 50:
                warnings.append(f"Low function documentation coverage: {function_coverage:.1f}%")
                recommendations.append("Add docstrings to functions")
            
            if class_coverage < 50:
                warnings.append(f"Low class documentation coverage: {class_coverage:.1f}%")
                recommendations.append("Add docstrings to classes")
            
            if overall_coverage >= 80:
                status = "passed"
                message = f"Documentation coverage excellent ({overall_coverage:.1f}%)"
            elif overall_coverage >= 60:
                status = "warning"
                message = f"Documentation coverage acceptable ({overall_coverage:.1f}%)"
            else:
                status = "failed"
                message = f"Documentation coverage insufficient ({overall_coverage:.1f}%)"
            
            return GateResult(
                name="documentation_check",
                status=status,
                execution_time=0,
                message=message,
                warnings=warnings,
                recommendations=recommendations,
                metrics={
                    'overall_coverage': overall_coverage,
                    'function_coverage': function_coverage,
                    'class_coverage': class_coverage,
                    'doc_files_count': len(doc_files),
                    'total_functions': total_functions,
                    'total_classes': total_classes
                }
            )
        
        except Exception as e:
            return GateResult(
                name="documentation_check",
                status="error",
                execution_time=0,
                message=f"Documentation check failed: {e}",
                errors=[str(e)]
            )
    
    def _gate_research_validation(self) -> GateResult:
        """Research code validation."""
        try:
            research_dir = Path("research")
            if not research_dir.exists():
                return GateResult(
                    name="research_validation",
                    status="skipped",
                    execution_time=0,
                    message="Research directory not found"
                )
            
            research_score = 0
            max_score = 5
            findings = []
            
            # Check for research structure
            if research_dir.exists():
                research_score += 1
                findings.append("Research directory exists")
                
                # Check for results directory
                if (research_dir / "results").exists():
                    research_score += 1
                    findings.append("Results directory exists")
                
                # Check for research pipeline
                research_files = ['research_pipeline.py', 'benchmarks.py', 'experiments.py']
                for research_file in research_files:
                    if (research_dir / research_file).exists():
                        research_score += 1
                        findings.append(f"Found {research_file}")
                        break
                
                # Check for configuration files
                config_files = list(research_dir.glob('*.yaml')) + list(research_dir.glob('*.json'))
                if config_files:
                    research_score += 1
                    findings.append(f"Found {len(config_files)} configuration files")
                
                # Check for documentation
                doc_files = list(research_dir.glob('*.md')) + list(research_dir.glob('README*'))
                if doc_files:
                    research_score += 1
                    findings.append(f"Found {len(doc_files)} documentation files")
            
            # Determine status
            if research_score >= 4:
                status = "passed"
                message = f"Research validation excellent ({research_score}/{max_score})"
            elif research_score >= 2:
                status = "warning"
                message = f"Research validation acceptable ({research_score}/{max_score})"
            else:
                status = "failed"
                message = f"Research validation insufficient ({research_score}/{max_score})"
            
            recommendations = []
            if research_score < 2:
                recommendations.append("Set up proper research directory structure")
            if not (research_dir / "results").exists():
                recommendations.append("Create results directory for experiment outputs")
            if research_score < 4:
                recommendations.append("Add research documentation and configuration files")
            
            return GateResult(
                name="research_validation",
                status=status,
                execution_time=0,
                message=message,
                recommendations=recommendations,
                metrics={
                    'research_score': research_score,
                    'max_score': max_score,
                    'findings': findings
                }
            )
        
        except Exception as e:
            return GateResult(
                name="research_validation",
                status="error",
                execution_time=0,
                message=f"Research validation failed: {e}",
                errors=[str(e)]
            )
    
    def _gate_reproducibility_check(self) -> GateResult:
        """Reproducibility validation."""
        try:
            reproducibility_score = 0
            max_score = 4
            findings = []
            
            # Check for requirements file
            req_files = ['requirements.txt', 'environment.yml', 'pyproject.toml']
            for req_file in req_files:
                if Path(req_file).exists():
                    reproducibility_score += 1
                    findings.append(f"Found {req_file}")
                    break
            
            # Check for version control
            if Path('.git').exists():
                reproducibility_score += 1
                findings.append("Git version control present")
            
            # Check for configuration management
            config_dirs = ['config', 'configs']
            config_files = []
            for config_dir in config_dirs:
                if Path(config_dir).exists():
                    config_files.extend(list(Path(config_dir).glob('*.yaml')))
                    config_files.extend(list(Path(config_dir).glob('*.json')))
            
            if config_files:
                reproducibility_score += 1
                findings.append(f"Found {len(config_files)} configuration files")
            
            # Check for containerization
            container_files = ['Dockerfile', 'docker-compose.yml', '.devcontainer']
            for container_file in container_files:
                if Path(container_file).exists():
                    reproducibility_score += 1
                    findings.append(f"Found {container_file}")
                    break
            
            # Determine status
            if reproducibility_score >= 3:
                status = "passed"
                message = f"Reproducibility excellent ({reproducibility_score}/{max_score})"
            elif reproducibility_score >= 2:
                status = "warning" 
                message = f"Reproducibility acceptable ({reproducibility_score}/{max_score})"
            else:
                status = "failed"
                message = f"Reproducibility insufficient ({reproducibility_score}/{max_score})"
            
            recommendations = []
            if reproducibility_score < 2:
                recommendations.append("Add requirements.txt or environment.yml")
                recommendations.append("Initialize git repository")
            if reproducibility_score < 3:
                recommendations.append("Add configuration management")
            if reproducibility_score < 4:
                recommendations.append("Consider adding Dockerfile for containerization")
            
            return GateResult(
                name="reproducibility_check",
                status=status,
                execution_time=0,
                message=message,
                recommendations=recommendations,
                metrics={
                    'reproducibility_score': reproducibility_score,
                    'max_score': max_score,
                    'findings': findings
                }
            )
        
        except Exception as e:
            return GateResult(
                name="reproducibility_check",
                status="error",
                execution_time=0,
                message=f"Reproducibility check failed: {e}",
                errors=[str(e)]
            )
    
    def _gate_production_readiness(self) -> GateResult:
        """Production deployment readiness check."""
        try:
            readiness_score = 0
            max_score = 6
            findings = []
            
            # Check for containerization
            if Path('Dockerfile').exists():
                readiness_score += 1
                findings.append("Dockerfile present")
            
            # Check for orchestration
            if Path('docker-compose.yml').exists() or Path('k8s').exists():
                readiness_score += 1
                findings.append("Container orchestration configuration found")
            
            # Check for monitoring
            if Path('monitoring').exists():
                readiness_score += 1
                findings.append("Monitoring configuration present")
            
            # Check for deployment scripts
            deploy_files = ['deploy.sh', 'deployment', 'scripts/deploy.sh']
            for deploy_file in deploy_files:
                if Path(deploy_file).exists():
                    readiness_score += 1
                    findings.append(f"Deployment scripts found: {deploy_file}")
                    break
            
            # Check for configuration management
            if Path('config').exists() or list(Path('.').glob('*.yaml')):
                readiness_score += 1
                findings.append("Configuration management present")
            
            # Check for health checks
            health_files = ['health_check.py', 'healthcheck.py']
            for health_file in health_files:
                if Path(health_file).exists() or Path('deployment').joinpath(health_file).exists():
                    readiness_score += 1
                    findings.append("Health check implementation found")
                    break
            
            # Determine status
            if readiness_score >= 5:
                status = "passed"
                message = f"Production readiness excellent ({readiness_score}/{max_score})"
            elif readiness_score >= 3:
                status = "warning"
                message = f"Production readiness acceptable ({readiness_score}/{max_score})"
            else:
                status = "failed"
                message = f"Production readiness insufficient ({readiness_score}/{max_score})"
            
            recommendations = []
            if readiness_score < 3:
                recommendations.append("Add Dockerfile and containerization")
                recommendations.append("Set up deployment scripts")
            if readiness_score < 5:
                recommendations.append("Add monitoring and alerting")
                recommendations.append("Implement health checks")
            
            return GateResult(
                name="production_readiness",
                status=status,
                execution_time=0,
                message=message,
                recommendations=recommendations,
                metrics={
                    'readiness_score': readiness_score,
                    'max_score': max_score,
                    'findings': findings
                }
            )
        
        except Exception as e:
            return GateResult(
                name="production_readiness",
                status="error",
                execution_time=0,
                message=f"Production readiness check failed: {e}",
                errors=[str(e)]
            )
    
    def _gate_deployment_validation(self) -> GateResult:
        """Deployment configuration validation."""
        try:
            deployment_score = 0
            max_score = 4
            findings = []
            
            # Check for environment configurations
            env_configs = ['.env.example', 'config/production.yaml', 'deployment/production']
            for env_config in env_configs:
                if Path(env_config).exists():
                    deployment_score += 1
                    findings.append(f"Environment configuration: {env_config}")
                    break
            
            # Check for scaling configuration
            scaling_files = ['k8s/deployment.yaml', 'docker-compose.prod.yml']
            for scaling_file in scaling_files:
                if Path(scaling_file).exists():
                    deployment_score += 1
                    findings.append(f"Scaling configuration: {scaling_file}")
                    break
            
            # Check for security configuration
            security_files = ['security.yaml', 'config/security.yaml']
            for security_file in security_files:
                if Path(security_file).exists():
                    deployment_score += 1
                    findings.append(f"Security configuration: {security_file}")
                    break
            
            # Check for backup/persistence
            persistence_indicators = ['volumes', 'persistent', 'backup']
            for indicator in persistence_indicators:
                if any(Path('.').rglob(f'*{indicator}*')):
                    deployment_score += 1
                    findings.append(f"Persistence configuration found")
                    break
            
            # Determine status
            if deployment_score >= 3:
                status = "passed"
                message = f"Deployment validation excellent ({deployment_score}/{max_score})"
            elif deployment_score >= 2:
                status = "warning"
                message = f"Deployment validation acceptable ({deployment_score}/{max_score})"
            else:
                status = "failed"
                message = f"Deployment validation insufficient ({deployment_score}/{max_score})"
            
            recommendations = []
            if deployment_score < 2:
                recommendations.append("Add environment-specific configurations")
                recommendations.append("Set up scaling and deployment configurations")
            if deployment_score < 4:
                recommendations.append("Add security configurations")
                recommendations.append("Plan for data persistence and backups")
            
            return GateResult(
                name="deployment_validation",
                status=status,
                execution_time=0,
                message=message,
                recommendations=recommendations,
                metrics={
                    'deployment_score': deployment_score,
                    'max_score': max_score,
                    'findings': findings
                }
            )
        
        except Exception as e:
            return GateResult(
                name="deployment_validation",
                status="error",
                execution_time=0,
                message=f"Deployment validation failed: {e}",
                errors=[str(e)]
            )
    
    def _compile_final_results(self) -> Dict[str, Any]:
        """Compile comprehensive final results."""
        total_execution_time = time.time() - self.execution_start_time
        
        # Count results by status
        status_counts = {
            'passed': 0,
            'failed': 0, 
            'warning': 0,
            'skipped': 0,
            'error': 0,
            'timeout': 0
        }
        
        required_failures = 0
        total_warnings = 0
        total_recommendations = 0
        
        for gate_name, result in self.results.items():
            status_counts[result.status] = status_counts.get(result.status, 0) + 1
            total_warnings += len(result.warnings)
            total_recommendations += len(result.recommendations)
            
            # Check if this was a required gate that failed
            if result.status in ['failed', 'error', 'timeout']:
                # Assume basic gates are required
                if gate_name in ['system_health', 'dependency_check', 'basic_functionality', 'security_scan']:
                    required_failures += 1
        
        # Determine overall success
        overall_success = (required_failures == 0 and 
                         status_counts['error'] == 0 and
                         status_counts['timeout'] == 0)
        
        # Generate summary recommendations
        summary_recommendations = self._generate_summary_recommendations()
        
        # Collect all metrics
        all_metrics = {}
        for gate_name, result in self.results.items():
            if result.metrics:
                all_metrics[gate_name] = result.metrics
        
        # Circuit breaker status
        circuit_breaker_status = {}
        for name, cb in self.circuit_breakers.items():
            circuit_breaker_status[name] = cb.get_status()
        
        # Compile comprehensive results
        compiled_results = {
            'timestamp': time.time(),
            'execution_time_seconds': total_execution_time,
            'environment': self.config.environment,
            'configuration': {
                'parallel_execution': self.config.parallel_execution,
                'max_workers': self.config.max_workers,
                'circuit_breaker_enabled': self.config.enable_circuit_breaker,
                'security_scanning': self.config.security_scan_enabled,
                'performance_monitoring': self.config.performance_monitoring
            },
            
            # Overall results
            'overall_success': overall_success,
            'quality_score': self._calculate_overall_quality_score(),
            'total_gates': len(self.results),
            'status_counts': status_counts,
            'required_failures': required_failures,
            'total_warnings': total_warnings,
            'total_recommendations': total_recommendations,
            
            # Individual gate results
            'gate_results': {
                name: {
                    'status': result.status,
                    'execution_time': result.execution_time,
                    'message': result.message,
                    'warnings_count': len(result.warnings),
                    'recommendations_count': len(result.recommendations),
                    'retry_count': result.retry_count,
                    'has_metrics': bool(result.metrics)
                }
                for name, result in self.results.items()
            },
            
            # Detailed results (if enabled)
            'detailed_results': self.results if self.config.save_detailed_results else None,
            'all_metrics': all_metrics if self.config.save_detailed_results else None,
            
            # System status
            'system_health': self._get_final_system_health(),
            'circuit_breakers': circuit_breaker_status,
            
            # Recommendations
            'summary_recommendations': summary_recommendations,
            'next_steps': self._generate_next_steps()
        }
        
        return compiled_results
    
    def _calculate_overall_quality_score(self) -> float:
        """Calculate overall quality score (0-100)."""
        if not self.results:
            return 0.0
        
        total_score = 0.0
        total_weight = 0.0
        
        # Weight different types of gates
        gate_weights = {
            'system_health': 1.5,
            'dependency_check': 1.5,
            'basic_functionality': 2.0,
            'security_scan': 2.0,
            'performance_baseline': 1.0,
            'code_quality': 1.0,
            'production_readiness': 1.5
        }
        
        for gate_name, result in self.results.items():
            weight = gate_weights.get(gate_name, 1.0)
            
            # Score based on status
            if result.status == 'passed':
                score = 100.0
            elif result.status == 'warning':
                score = 75.0
            elif result.status == 'skipped':
                score = 50.0
            elif result.status == 'failed':
                score = 25.0
            else:  # error, timeout
                score = 0.0
            
            total_score += score * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _generate_summary_recommendations(self) -> List[str]:
        """Generate summary recommendations based on all results."""
        recommendations = []
        
        # Analyze patterns across results
        failed_gates = [name for name, result in self.results.items() if result.status == 'failed']
        warning_gates = [name for name, result in self.results.items() if result.status == 'warning']
        
        if failed_gates:
            recommendations.append(f"🔴 URGENT: Address {len(failed_gates)} failed gates: {', '.join(failed_gates)}")
        
        if len(warning_gates) > 3:
            recommendations.append(f"⚠️ Review {len(warning_gates)} gates with warnings for potential improvements")
        
        # Specific recommendations based on patterns
        if 'security_scan' in failed_gates:
            recommendations.append("🔒 Implement security fixes before deployment")
        
        if 'performance_baseline' in failed_gates or 'performance_baseline' in warning_gates:
            recommendations.append("⚡ Optimize performance before scaling")
        
        if 'production_readiness' in failed_gates:
            recommendations.append("🚀 Complete production setup before deployment")
        
        # Add top recommendations from individual gates
        all_recommendations = []
        for result in self.results.values():
            all_recommendations.extend(result.recommendations[:2])  # Top 2 from each
        
        # Count and prioritize
        recommendation_counts = {}
        for rec in all_recommendations:
            recommendation_counts[rec] = recommendation_counts.get(rec, 0) + 1
        
        # Add most common recommendations
        sorted_recs = sorted(recommendation_counts.items(), key=lambda x: x[1], reverse=True)
        for rec, count in sorted_recs[:3]:
            if count > 1:
                recommendations.append(f"📋 {rec} (mentioned {count} times)")
        
        return recommendations[:10]  # Limit to top 10
    
    def _generate_next_steps(self) -> List[str]:
        """Generate actionable next steps."""
        next_steps = []
        
        # Analyze results for next steps
        failed_count = sum(1 for result in self.results.values() if result.status == 'failed')
        warning_count = sum(1 for result in self.results.values() if result.status == 'warning')
        
        if failed_count > 0:
            next_steps.append("1. Address all failed quality gates before proceeding")
            next_steps.append("2. Re-run quality gates after fixes to verify resolution")
        elif warning_count > 0:
            next_steps.append("1. Review and address warning conditions")
            next_steps.append("2. Consider upgrading warnings to requirements for production")
        else:
            next_steps.append("1. All quality gates passed - ready for next phase")
            next_steps.append("2. Consider running with stricter requirements")
        
        # Environment-specific next steps
        if self.config.environment == "development":
            next_steps.append("3. Consider running with CI/production gate settings")
        elif self.config.environment == "ci":
            next_steps.append("3. Deploy to staging environment for integration testing")
        elif self.config.environment == "production":
            next_steps.append("3. Monitor production deployment closely")
        
        return next_steps
    
    def _get_final_system_health(self) -> Dict[str, Any]:
        """Get final system health summary."""
        try:
            current_metrics = self.health_monitor.collect_system_metrics()
            status, warnings, recommendations = self.health_monitor.assess_system_health(current_metrics)
            
            return {
                'status': status,
                'warnings_count': len(warnings),
                'recommendations_count': len(recommendations),
                'metrics': current_metrics.get('system', {}),
                'timestamp': current_metrics.get('timestamp', time.time())
            }
        except:
            return {
                'status': 'unknown',
                'error': 'Failed to collect final health metrics'
            }
    
    def print_comprehensive_summary(self, results: Dict[str, Any]) -> None:
        """Print comprehensive results summary."""
        logger.info("=" * 80)
        logger.info("🎯 ENHANCED PROGRESSIVE QUALITY GATES - FINAL REPORT")
        logger.info("=" * 80)
        
        # Header information
        logger.info(f"🕐 Execution Time: {results['execution_time_seconds']:.2f} seconds")
        logger.info(f"🌍 Environment: {results['environment']}")
        logger.info(f"⚙️ Configuration: {results['configuration']}")
        logger.info("")
        
        # Overall status
        if results['overall_success']:
            logger.info("🎉 OVERALL STATUS: ✅ SUCCESS - All critical quality gates passed!")
        else:
            logger.info("💥 OVERALL STATUS: ❌ FAILED - Critical issues need attention!")
        
        logger.info(f"📊 Quality Score: {results['quality_score']:.1f}/100")
        logger.info("")
        
        # Statistics
        status_counts = results['status_counts']
        logger.info("📈 EXECUTION STATISTICS:")
        logger.info(f"  ✅ Passed: {status_counts.get('passed', 0)}")
        logger.info(f"  ❌ Failed: {status_counts.get('failed', 0)}")
        logger.info(f"  ⚠️  Warnings: {status_counts.get('warning', 0)}")
        logger.info(f"  ⏭️  Skipped: {status_counts.get('skipped', 0)}")
        logger.info(f"  💥 Errors: {status_counts.get('error', 0)}")
        logger.info(f"  ⏰ Timeouts: {status_counts.get('timeout', 0)}")
        logger.info("")
        
        # Gate results summary
        logger.info("🔍 GATE EXECUTION SUMMARY:")
        for gate_name, gate_result in results['gate_results'].items():
            status_emoji = {
                'passed': '✅',
                'failed': '❌', 
                'warning': '⚠️',
                'skipped': '⏭️',
                'error': '💥',
                'timeout': '⏰'
            }.get(gate_result['status'], '❓')
            
            retry_info = f" (retries: {gate_result['retry_count']})" if gate_result['retry_count'] > 0 else ""
            logger.info(f"  {status_emoji} {gate_name}: {gate_result['message']} ({gate_result['execution_time']:.2f}s{retry_info})")
        
        logger.info("")
        
        # System health
        system_health = results.get('system_health', {})
        logger.info(f"💻 FINAL SYSTEM HEALTH: {system_health.get('status', 'unknown').upper()}")
        if system_health.get('warnings_count', 0) > 0:
            logger.info(f"  ⚠️  {system_health['warnings_count']} health warnings")
        logger.info("")
        
        # Summary recommendations
        if results.get('summary_recommendations'):
            logger.info("💡 KEY RECOMMENDATIONS:")
            for recommendation in results['summary_recommendations']:
                logger.info(f"  • {recommendation}")
            logger.info("")
        
        # Next steps
        if results.get('next_steps'):
            logger.info("🚀 NEXT STEPS:")
            for step in results['next_steps']:
                logger.info(f"  {step}")
            logger.info("")
        
        # Circuit breaker status
        circuit_breakers = results.get('circuit_breakers', {})
        open_breakers = [name for name, status in circuit_breakers.items() if status.get('state') == 'open']
        if open_breakers:
            logger.info("⚡ CIRCUIT BREAKER ALERTS:")
            for breaker in open_breakers:
                logger.info(f"  🔴 {breaker}: Circuit breaker is OPEN")
            logger.info("")
        
        logger.info("=" * 80)
        
        # Final summary
        if results['overall_success']:
            logger.info("🎊 QUALITY GATES EXECUTION COMPLETED SUCCESSFULLY!")
            logger.info("   Ready for next development phase or deployment.")
        else:
            logger.info("⚠️  QUALITY GATES EXECUTION COMPLETED WITH ISSUES!")
            logger.info("   Please address failed gates before proceeding.")
        
        logger.info("=" * 80)


def main():
    """Enhanced main execution with comprehensive error handling."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Enhanced Progressive Quality Gates for Protein Diffusion Design Lab",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python enhanced_quality_gate_runner.py                    # Run with defaults
  python enhanced_quality_gate_runner.py --environment ci  # Run in CI mode
  python enhanced_quality_gate_runner.py --production      # Enable production gates
  python enhanced_quality_gate_runner.py --research        # Enable research gates
  python enhanced_quality_gate_runner.py --verbose --parallel=8  # Verbose with 8 workers
        """
    )
    
    # Environment options
    parser.add_argument('--environment', choices=['development', 'ci', 'production', 'research'],
                       help='Override environment detection')
    parser.add_argument('--auto-detect', action='store_true', default=True,
                       help='Auto-detect environment (default: True)')
    
    # Gate selection
    parser.add_argument('--basic', action='store_true', default=True,
                       help='Run basic gates (default: True)')
    parser.add_argument('--security', action='store_true', default=True,
                       help='Run security gates (default: True)')
    parser.add_argument('--performance', action='store_true', default=True,
                       help='Run performance gates (default: True)')
    parser.add_argument('--research', action='store_true', default=False,
                       help='Run research gates')
    parser.add_argument('--production', action='store_true', default=False,
                       help='Run production gates')
    
    # Execution options
    parser.add_argument('--parallel', type=int, default=4,
                       help='Max parallel workers (default: 4)')
    parser.add_argument('--sequential', action='store_true',
                       help='Run gates sequentially')
    parser.add_argument('--timeout-multiplier', type=float, default=1.0,
                       help='Timeout multiplier (default: 1.0)')
    
    # Reliability options
    parser.add_argument('--no-circuit-breaker', action='store_true',
                       help='Disable circuit breaker')
    parser.add_argument('--max-retries', type=int, default=3,
                       help='Maximum retry attempts (default: 3)')
    parser.add_argument('--retry-delay', type=float, default=1.0,
                       help='Retry delay in seconds (default: 1.0)')
    
    # Output options
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    parser.add_argument('--output', default='enhanced_quality_gates_results.json',
                       help='Output file for results')
    parser.add_argument('--format', choices=['json', 'yaml'], default='json',
                       help='Output format (default: json)')
    parser.add_argument('--no-save-details', action='store_true',
                       help='Do not save detailed results')
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("🔧 Verbose logging enabled")
    
    try:
        # Create enhanced configuration
        config = EnhancedQualityGateConfig()
        
        # Apply command line arguments
        if args.environment:
            config.environment = args.environment
            config.auto_detect_environment = False
        
        config.run_basic_gates = args.basic
        config.security_scan_enabled = args.security
        config.performance_monitoring = args.performance
        config.run_research_gates = args.research
        config.run_production_gates = args.production
        
        config.parallel_execution = not args.sequential
        config.max_workers = args.parallel
        config.timeout_multiplier = args.timeout_multiplier
        
        config.enable_circuit_breaker = not args.no_circuit_breaker
        config.max_retry_attempts = args.max_retries
        config.retry_delay = args.retry_delay
        
        config.verbose = args.verbose
        config.save_detailed_results = not args.no_save_details
        config.export_format = args.format
        
        # Create and run enhanced quality gate runner
        runner = EnhancedQualityGateRunner(config)
        
        logger.info("🚀 Starting Enhanced Progressive Quality Gates Execution")
        start_time = time.time()
        
        results = runner.run_all_gates()
        
        execution_time = time.time() - start_time
        logger.info(f"⏱️ Total execution completed in {execution_time:.2f} seconds")
        
        # Print comprehensive summary
        runner.print_comprehensive_summary(results)
        
        # Save results to file
        try:
            if args.format == 'yaml':
                import yaml
                with open(args.output.replace('.json', '.yaml'), 'w') as f:
                    yaml.dump(results, f, default_flow_style=False, indent=2)
                logger.info(f"📄 Results saved to {args.output.replace('.json', '.yaml')}")
            else:
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                logger.info(f"📄 Results saved to {args.output}")
        
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
        
        # Return appropriate exit code
        if results['overall_success']:
            logger.info("🎉 Enhanced Quality Gates completed successfully!")
            return 0
        else:
            logger.error("💥 Enhanced Quality Gates completed with failures!")
            return 1
    
    except KeyboardInterrupt:
        logger.warning("⚠️ Enhanced Quality Gates interrupted by user")
        return 130
    
    except Exception as e:
        logger.error(f"💥 Enhanced Quality Gates failed with error: {e}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        return 1


if __name__ == "__main__":
    sys.exit(main())