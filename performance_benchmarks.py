"""
Performance Benchmarking Suite for Protein Diffusion Design Lab

This module provides comprehensive performance benchmarking including:
- Generation speed and throughput testing
- Memory usage profiling
- Scalability benchmarking
- Cache performance analysis
- Model inference optimization
"""

import time
import psutil
import gc
import threading
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import json
import statistics

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


@dataclass
class BenchmarkResult:
    """Results from a benchmark test."""
    test_name: str
    duration: float
    throughput: float  # operations per second
    memory_usage: Dict[str, float]  # MB
    cpu_usage: float  # percentage
    gpu_usage: Optional[float] = None  # percentage
    error_rate: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemMetrics:
    """System resource metrics."""
    timestamp: float
    cpu_percent: float
    memory_mb: float
    memory_percent: float
    gpu_memory_mb: Optional[float] = None
    gpu_utilization: Optional[float] = None


class ResourceMonitor:
    """Monitor system resources during benchmarks."""
    
    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.metrics = []
        self.monitoring = False
        self.monitor_thread = None
    
    def start(self):
        """Start resource monitoring."""
        self.monitoring = True
        self.metrics = []
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop(self) -> List[SystemMetrics]:
        """Stop monitoring and return collected metrics."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        return self.metrics.copy()
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                # CPU and memory
                cpu_percent = psutil.cpu_percent()
                memory = psutil.virtual_memory()
                
                # GPU metrics (if available)
                gpu_memory_mb = None
                gpu_utilization = None
                
                if TORCH_AVAILABLE and torch.cuda.is_available():
                    try:
                        gpu_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
                        gpu_utilization = torch.cuda.utilization()
                    except Exception:
                        pass
                
                metric = SystemMetrics(
                    timestamp=time.time(),
                    cpu_percent=cpu_percent,
                    memory_mb=memory.used / 1024 / 1024,
                    memory_percent=memory.percent,
                    gpu_memory_mb=gpu_memory_mb,
                    gpu_utilization=gpu_utilization
                )
                
                self.metrics.append(metric)
                time.sleep(self.interval)
                
            except Exception as e:
                print(f"Monitoring error: {e}")
                break
    
    def get_peak_usage(self) -> Dict[str, float]:
        """Get peak resource usage from monitoring period."""
        if not self.metrics:
            return {}
        
        return {
            'peak_cpu_percent': max(m.cpu_percent for m in self.metrics),
            'peak_memory_mb': max(m.memory_mb for m in self.metrics),
            'peak_memory_percent': max(m.memory_percent for m in self.metrics),
            'peak_gpu_memory_mb': max((m.gpu_memory_mb for m in self.metrics if m.gpu_memory_mb), default=0),
            'peak_gpu_utilization': max((m.gpu_utilization for m in self.metrics if m.gpu_utilization), default=0),
            'avg_cpu_percent': statistics.mean(m.cpu_percent for m in self.metrics),
            'avg_memory_mb': statistics.mean(m.memory_mb for m in self.metrics),
        }


class ProteinGenerationBenchmark:
    """Benchmark protein generation performance."""
    
    def __init__(self):
        self.results = []
    
    def benchmark_generation_speed(self, generator_func: Callable, test_configs: List[Dict]) -> List[BenchmarkResult]:
        """Benchmark generation speed with different configurations."""
        results = []
        
        for config in test_configs:
            print(f"Benchmarking config: {config}")
            
            monitor = ResourceMonitor()
            monitor.start()
            
            start_time = time.time()
            errors = 0
            
            try:
                # Run generation
                result = generator_func(**config)
                
                if result is None or (isinstance(result, list) and len(result) == 0):
                    errors += 1
                
            except Exception as e:
                print(f"Generation error: {e}")
                errors += 1
                result = None
            
            duration = time.time() - start_time
            metrics = monitor.stop()
            peak_usage = monitor.get_peak_usage()
            
            # Calculate throughput
            num_samples = config.get('num_samples', 1)
            throughput = num_samples / duration if duration > 0 else 0
            
            benchmark_result = BenchmarkResult(
                test_name=f"generation_{config.get('num_samples', 1)}samples",
                duration=duration,
                throughput=throughput,
                memory_usage=peak_usage,
                cpu_usage=peak_usage.get('avg_cpu_percent', 0),
                gpu_usage=peak_usage.get('peak_gpu_utilization'),
                error_rate=errors / 1,
                metadata={'config': config, 'result_count': len(result) if result else 0}
            )
            
            results.append(benchmark_result)
            
            # Cleanup
            gc.collect()
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return results
    
    def benchmark_batch_processing(self, processor_func: Callable, batch_sizes: List[int]) -> List[BenchmarkResult]:
        """Benchmark batch processing performance."""
        results = []
        
        for batch_size in batch_sizes:
            print(f"Benchmarking batch size: {batch_size}")
            
            # Create test requests
            requests = [
                {'num_samples': 5, 'max_length': 100, 'temperature': 1.0}
                for _ in range(batch_size)
            ]
            
            monitor = ResourceMonitor()
            monitor.start()
            
            start_time = time.time()
            errors = 0
            
            try:
                results_batch = processor_func(requests)
                errors = sum(1 for r in results_batch if not r or len(r) == 0)
            except Exception as e:
                print(f"Batch processing error: {e}")
                errors = batch_size
                results_batch = []
            
            duration = time.time() - start_time
            metrics = monitor.stop()
            peak_usage = monitor.get_peak_usage()
            
            # Calculate throughput (requests per second)
            throughput = batch_size / duration if duration > 0 else 0
            
            benchmark_result = BenchmarkResult(
                test_name=f"batch_processing_{batch_size}requests",
                duration=duration,
                throughput=throughput,
                memory_usage=peak_usage,
                cpu_usage=peak_usage.get('avg_cpu_percent', 0),
                gpu_usage=peak_usage.get('peak_gpu_utilization'),
                error_rate=errors / batch_size,
                metadata={'batch_size': batch_size, 'successful_requests': batch_size - errors}
            )
            
            results.append(benchmark_result)
            
            # Cleanup
            gc.collect()
        
        return results
    
    def benchmark_concurrent_processing(self, generator_func: Callable, thread_counts: List[int]) -> List[BenchmarkResult]:
        """Benchmark concurrent processing performance."""
        results = []
        
        for thread_count in thread_counts:
            print(f"Benchmarking {thread_count} concurrent threads")
            
            monitor = ResourceMonitor()
            monitor.start()
            
            start_time = time.time()
            errors = 0
            completed_tasks = 0
            
            def worker_task():
                try:
                    result = generator_func(num_samples=2, max_length=50)
                    return result is not None and len(result) > 0
                except Exception:
                    return False
            
            with ThreadPoolExecutor(max_workers=thread_count) as executor:
                futures = [executor.submit(worker_task) for _ in range(thread_count * 2)]
                
                for future in futures:
                    try:
                        if future.result():
                            completed_tasks += 1
                        else:
                            errors += 1
                    except Exception:
                        errors += 1
            
            duration = time.time() - start_time
            metrics = monitor.stop()
            peak_usage = monitor.get_peak_usage()
            
            # Calculate throughput (tasks per second)
            total_tasks = thread_count * 2
            throughput = total_tasks / duration if duration > 0 else 0
            
            benchmark_result = BenchmarkResult(
                test_name=f"concurrent_{thread_count}threads",
                duration=duration,
                throughput=throughput,
                memory_usage=peak_usage,
                cpu_usage=peak_usage.get('avg_cpu_percent', 0),
                gpu_usage=peak_usage.get('peak_gpu_utilization'),
                error_rate=errors / total_tasks,
                metadata={'thread_count': thread_count, 'completed_tasks': completed_tasks, 'total_tasks': total_tasks}
            )
            
            results.append(benchmark_result)
            
            # Cleanup
            gc.collect()
        
        return results


class CacheBenchmark:
    """Benchmark caching performance."""
    
    def benchmark_cache_operations(self, cache_instance, operation_counts: List[int]) -> List[BenchmarkResult]:
        """Benchmark cache set/get operations."""
        results = []
        
        for count in operation_counts:
            print(f"Benchmarking {count} cache operations")
            
            # Prepare test data
            test_data = [(f"key_{i}", f"value_{i}" * 100) for i in range(count)]
            
            # Benchmark SET operations
            monitor = ResourceMonitor()
            monitor.start()
            
            start_time = time.time()
            
            for key, value in test_data:
                try:
                    cache_instance.set(key, value)
                except Exception as e:
                    print(f"Cache set error: {e}")
            
            set_duration = time.time() - start_time
            set_metrics = monitor.stop()
            set_peak_usage = monitor.get_peak_usage()
            
            # Benchmark GET operations
            monitor = ResourceMonitor()
            monitor.start()
            
            start_time = time.time()
            hits = 0
            misses = 0
            
            for key, _ in test_data:
                try:
                    result = cache_instance.get(key)
                    if result is not None:
                        hits += 1
                    else:
                        misses += 1
                except Exception as e:
                    print(f"Cache get error: {e}")
                    misses += 1
            
            get_duration = time.time() - start_time
            get_metrics = monitor.stop()
            get_peak_usage = monitor.get_peak_usage()
            
            # SET results
            set_result = BenchmarkResult(
                test_name=f"cache_set_{count}ops",
                duration=set_duration,
                throughput=count / set_duration if set_duration > 0 else 0,
                memory_usage=set_peak_usage,
                cpu_usage=set_peak_usage.get('avg_cpu_percent', 0),
                metadata={'operation': 'set', 'count': count}
            )
            results.append(set_result)
            
            # GET results
            get_result = BenchmarkResult(
                test_name=f"cache_get_{count}ops",
                duration=get_duration,
                throughput=count / get_duration if get_duration > 0 else 0,
                memory_usage=get_peak_usage,
                cpu_usage=get_peak_usage.get('avg_cpu_percent', 0),
                metadata={'operation': 'get', 'count': count, 'hits': hits, 'misses': misses, 'hit_rate': hits / count}
            )
            results.append(get_result)
        
        return results


class SecurityBenchmark:
    """Benchmark security operations performance."""
    
    def benchmark_validation_performance(self, validator, test_inputs: List[str]) -> BenchmarkResult:
        """Benchmark validation performance."""
        print(f"Benchmarking validation with {len(test_inputs)} inputs")
        
        monitor = ResourceMonitor()
        monitor.start()
        
        start_time = time.time()
        successful_validations = 0
        errors = 0
        
        for test_input in test_inputs:
            try:
                result = validator.validate(test_input)
                if result.is_valid:
                    successful_validations += 1
            except Exception as e:
                errors += 1
        
        duration = time.time() - start_time
        metrics = monitor.stop()
        peak_usage = monitor.get_peak_usage()
        
        return BenchmarkResult(
            test_name="validation_performance",
            duration=duration,
            throughput=len(test_inputs) / duration if duration > 0 else 0,
            memory_usage=peak_usage,
            cpu_usage=peak_usage.get('avg_cpu_percent', 0),
            error_rate=errors / len(test_inputs),
            metadata={
                'total_inputs': len(test_inputs),
                'successful_validations': successful_validations,
                'errors': errors
            }
        )
    
    def benchmark_authentication_performance(self, auth_manager, request_count: int) -> BenchmarkResult:
        """Benchmark authentication performance."""
        print(f"Benchmarking authentication with {request_count} requests")
        
        # Generate test API key
        api_key = auth_manager.generate_api_key("test_user", "user")
        
        monitor = ResourceMonitor()
        monitor.start()
        
        start_time = time.time()
        successful_auths = 0
        
        for _ in range(request_count):
            try:
                context = auth_manager.validate_api_key(api_key)
                if context is not None:
                    successful_auths += 1
            except Exception:
                pass
        
        duration = time.time() - start_time
        metrics = monitor.stop()
        peak_usage = monitor.get_peak_usage()
        
        return BenchmarkResult(
            test_name="authentication_performance",
            duration=duration,
            throughput=request_count / duration if duration > 0 else 0,
            memory_usage=peak_usage,
            cpu_usage=peak_usage.get('avg_cpu_percent', 0),
            metadata={
                'request_count': request_count,
                'successful_auths': successful_auths,
                'auth_rate': successful_auths / request_count
            }
        )


class BenchmarkRunner:
    """Main benchmark runner that coordinates all tests."""
    
    def __init__(self):
        self.results = []
        self.start_time = None
        self.end_time = None
    
    def run_all_benchmarks(self, test_config: Dict[str, Any] = None) -> Dict[str, List[BenchmarkResult]]:
        """Run comprehensive benchmark suite."""
        if test_config is None:
            test_config = self._get_default_config()
        
        print("Starting comprehensive benchmark suite...")
        self.start_time = time.time()
        
        all_results = {}
        
        # Mock generators for testing (replace with actual implementations)
        mock_generator = self._create_mock_generator()
        mock_processor = self._create_mock_processor()
        mock_cache = self._create_mock_cache()
        mock_validator = self._create_mock_validator()
        mock_auth_manager = self._create_mock_auth_manager()
        
        # 1. Generation Performance Benchmarks
        print("\n=== Generation Performance Benchmarks ===")
        gen_benchmark = ProteinGenerationBenchmark()
        
        gen_configs = test_config.get('generation_configs', [
            {'num_samples': 1, 'max_length': 50},
            {'num_samples': 5, 'max_length': 100},
            {'num_samples': 10, 'max_length': 200}
        ])
        
        all_results['generation_speed'] = gen_benchmark.benchmark_generation_speed(
            mock_generator, gen_configs
        )
        
        batch_sizes = test_config.get('batch_sizes', [1, 5, 10, 20])
        all_results['batch_processing'] = gen_benchmark.benchmark_batch_processing(
            mock_processor, batch_sizes
        )
        
        thread_counts = test_config.get('thread_counts', [1, 2, 4, 8])
        all_results['concurrent_processing'] = gen_benchmark.benchmark_concurrent_processing(
            mock_generator, thread_counts
        )
        
        # 2. Cache Performance Benchmarks
        print("\n=== Cache Performance Benchmarks ===")
        cache_benchmark = CacheBenchmark()
        
        operation_counts = test_config.get('cache_operation_counts', [100, 1000, 5000])
        all_results['cache_performance'] = cache_benchmark.benchmark_cache_operations(
            mock_cache, operation_counts
        )
        
        # 3. Security Performance Benchmarks
        print("\n=== Security Performance Benchmarks ===")
        security_benchmark = SecurityBenchmark()
        
        # Generate test sequences
        test_sequences = [
            "MKLLVLGLFTLVLLGLVGLAL" * (i + 1) for i in range(100)
        ]
        
        all_results['validation_performance'] = [security_benchmark.benchmark_validation_performance(
            mock_validator, test_sequences
        )]
        
        all_results['authentication_performance'] = [security_benchmark.benchmark_authentication_performance(
            mock_auth_manager, 1000
        )]
        
        self.end_time = time.time()
        print(f"\nBenchmark suite completed in {self.end_time - self.start_time:.2f} seconds")
        
        return all_results
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default benchmark configuration."""
        return {
            'generation_configs': [
                {'num_samples': 1, 'max_length': 50},
                {'num_samples': 5, 'max_length': 100},
                {'num_samples': 10, 'max_length': 200}
            ],
            'batch_sizes': [1, 5, 10],
            'thread_counts': [1, 2, 4],
            'cache_operation_counts': [100, 1000]
        }
    
    def _create_mock_generator(self):
        """Create mock generator for testing."""
        def mock_generate(**kwargs):
            time.sleep(0.1)  # Simulate processing time
            num_samples = kwargs.get('num_samples', 1)
            return [{'sequence': 'MKLLVLGLFT', 'confidence': 0.8} for _ in range(num_samples)]
        return mock_generate
    
    def _create_mock_processor(self):
        """Create mock batch processor for testing."""
        def mock_process(requests):
            time.sleep(0.05 * len(requests))  # Simulate batch processing
            return [[{'sequence': 'MKLLVLGLFT'}] for _ in requests]
        return mock_process
    
    def _create_mock_cache(self):
        """Create mock cache for testing."""
        class MockCache:
            def __init__(self):
                self.storage = {}
            
            def set(self, key, value):
                self.storage[key] = value
            
            def get(self, key):
                return self.storage.get(key)
        
        return MockCache()
    
    def _create_mock_validator(self):
        """Create mock validator for testing."""
        class MockValidator:
            def validate(self, sequence):
                time.sleep(0.001)  # Simulate validation time
                class Result:
                    is_valid = len(sequence) > 5 and sequence.isalpha()
                return Result()
        
        return MockValidator()
    
    def _create_mock_auth_manager(self):
        """Create mock authentication manager for testing."""
        class MockAuthManager:
            def generate_api_key(self, user_id, level):
                return f"api_key_{user_id}_{level}"
            
            def validate_api_key(self, api_key):
                time.sleep(0.001)  # Simulate auth time
                return {"user_id": "test"} if api_key.startswith("api_key_") else None
        
        return MockAuthManager()
    
    def generate_report(self, results: Dict[str, List[BenchmarkResult]]) -> str:
        """Generate comprehensive benchmark report."""
        report = []
        report.append("# Protein Diffusion Design Lab - Performance Benchmark Report")
        report.append(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total benchmark time: {self.end_time - self.start_time:.2f} seconds\n")
        
        for category, category_results in results.items():
            report.append(f"## {category.replace('_', ' ').title()}")
            
            for result in category_results:
                report.append(f"\n### {result.test_name}")
                report.append(f"- **Duration**: {result.duration:.3f} seconds")
                report.append(f"- **Throughput**: {result.throughput:.2f} ops/sec")
                report.append(f"- **CPU Usage**: {result.cpu_usage:.1f}%")
                report.append(f"- **Peak Memory**: {result.memory_usage.get('peak_memory_mb', 0):.1f} MB")
                
                if result.gpu_usage:
                    report.append(f"- **GPU Usage**: {result.gpu_usage:.1f}%")
                
                if result.error_rate > 0:
                    report.append(f"- **Error Rate**: {result.error_rate:.1%}")
                
                if result.metadata:
                    report.append("- **Additional Metrics**:")
                    for key, value in result.metadata.items():
                        if isinstance(value, float):
                            report.append(f"  - {key}: {value:.3f}")
                        else:
                            report.append(f"  - {key}: {value}")
        
        # Performance recommendations
        report.append("\n## Performance Recommendations")
        report.append(self._generate_recommendations(results))
        
        return '\n'.join(report)
    
    def _generate_recommendations(self, results: Dict[str, List[BenchmarkResult]]) -> str:
        """Generate performance recommendations based on results."""
        recommendations = []
        
        # Analyze generation performance
        gen_results = results.get('generation_speed', [])
        if gen_results:
            avg_throughput = statistics.mean(r.throughput for r in gen_results)
            if avg_throughput < 1.0:
                recommendations.append("- Consider optimizing generation algorithms for better throughput")
            
            high_memory_tests = [r for r in gen_results if r.memory_usage.get('peak_memory_mb', 0) > 1000]
            if high_memory_tests:
                recommendations.append("- High memory usage detected, consider implementing memory optimization")
        
        # Analyze cache performance
        cache_results = results.get('cache_performance', [])
        if cache_results:
            cache_get_results = [r for r in cache_results if 'get' in r.test_name]
            if cache_get_results:
                avg_hit_rate = statistics.mean(
                    r.metadata.get('hit_rate', 0) for r in cache_get_results
                )
                if avg_hit_rate < 0.8:
                    recommendations.append("- Cache hit rate below 80%, consider cache optimization")
        
        # Analyze concurrent performance
        concurrent_results = results.get('concurrent_processing', [])
        if concurrent_results:
            throughputs = [r.throughput for r in concurrent_results]
            if len(throughputs) > 1 and throughputs[-1] < throughputs[0] * 2:
                recommendations.append("- Concurrent processing scaling suboptimal, investigate bottlenecks")
        
        if not recommendations:
            recommendations.append("- No major performance issues detected")
        
        return '\n'.join(recommendations)
    
    def save_results(self, results: Dict[str, List[BenchmarkResult]], output_file: str = "benchmark_results.json"):
        """Save benchmark results to file."""
        # Convert results to JSON-serializable format
        serializable_results = {}
        
        for category, category_results in results.items():
            serializable_results[category] = []
            for result in category_results:
                result_dict = {
                    'test_name': result.test_name,
                    'duration': result.duration,
                    'throughput': result.throughput,
                    'memory_usage': result.memory_usage,
                    'cpu_usage': result.cpu_usage,
                    'gpu_usage': result.gpu_usage,
                    'error_rate': result.error_rate,
                    'metadata': result.metadata
                }
                serializable_results[category].append(result_dict)
        
        with open(output_file, 'w') as f:
            json.dump({
                'timestamp': time.time(),
                'benchmark_duration': self.end_time - self.start_time if self.end_time else 0,
                'results': serializable_results
            }, f, indent=2)
        
        print(f"Benchmark results saved to {output_file}")


def main():
    """Run benchmark suite."""
    runner = BenchmarkRunner()
    
    # Run benchmarks
    results = runner.run_all_benchmarks()
    
    # Generate and save report
    report = runner.generate_report(results)
    
    with open("benchmark_report.md", "w") as f:
        f.write(report)
    
    runner.save_results(results)
    
    print("\nBenchmark completed!")
    print("- Detailed report: benchmark_report.md")
    print("- Raw results: benchmark_results.json")


if __name__ == "__main__":
    main()