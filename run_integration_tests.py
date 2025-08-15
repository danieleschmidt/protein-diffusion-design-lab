#!/usr/bin/env python3
"""
Comprehensive Integration Tests for Protein Diffusion Design Lab.

This script runs end-to-end integration tests across all new modules
to validate the autonomous SDLC implementation.
"""

import sys
import time
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_complete_workflow():
    """Test complete protein design workflow."""
    print("🧪 Testing Complete Workflow Integration...")
    
    try:
        from protein_diffusion.workflow_manager import ProteinDesignWorkflow, WorkflowConfig
        from protein_diffusion.quality_gates import QualityGateManager, run_quality_gates
        from protein_diffusion.error_recovery import get_global_recovery_manager
        from protein_diffusion.robust_monitoring import get_global_monitoring_manager
        from protein_diffusion.advanced_caching import get_global_cache_manager
        from protein_diffusion.performance_optimization import get_global_performance_optimizer
        
        # Initialize workflow
        config = WorkflowConfig(
            output_dir="./test_output",
            experiment_name="integration_test"
        )
        workflow = ProteinDesignWorkflow(config)
        
        # Test basic workflow without actual model execution
        print("  ✅ Workflow components initialized")
        
        # Test monitoring integration
        monitoring = get_global_monitoring_manager()
        dashboard = monitoring.get_monitoring_dashboard()
        print(f"  ✅ Monitoring dashboard: {len(dashboard)} metrics")
        
        # Test caching integration
        cache_manager = get_global_cache_manager()
        cache_manager.put("test_key", {"test": "data"})
        cached_data = cache_manager.get("test_key")
        assert cached_data == {"test": "data"}
        print("  ✅ Caching system functional")
        
        # Test error recovery
        recovery_manager = get_global_recovery_manager()
        health = recovery_manager.get_circuit_breaker_status()
        print(f"  ✅ Error recovery system: {len(health)} circuit breakers")
        
        # Test performance optimization
        performance_optimizer = get_global_performance_optimizer()
        perf_summary = performance_optimizer.get_performance_summary()
        print(f"  ✅ Performance optimization: {perf_summary['status']}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Workflow integration test failed: {e}")
        return False


def test_quality_gates():
    """Test quality gate system."""
    print("🛡️ Testing Quality Gates System...")
    
    try:
        from protein_diffusion.quality_gates import QualityGateManager, QualityGateConfig
        
        # Create test context
        test_context = {
            "generated_sequences": [
                {"sequence": "MKLLILTCLVAVALARPKHPIPWDQAITVAYASRALGRGLVVMAQDGNRGGKFHPWTVNQGPLKDYICQAYDMGTTTEVPGTMGMLRRRSNVWSCLPRLLCERVAAPNLDPEGFVVAVPIPVYEAWDFGDPKLNLRQNTVAVTCTGVQTLAVRGRVGNLLSNGVPIGRGLPHIPSKGSGATFEFIGSDLKAELATDQAGVLQVDVQQVEACWFASQGGGVDTDYTGQPWDGGKPTVTGAMCGAFSCRHDGKRDVRVGTAAGVGGGYCSDGDGPVKPVVSNPNQALAFGLSEAGSRRLHPFTTARQGAGSM", "confidence": 0.85, "length": 250},
                {"sequence": "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY", "confidence": 0.75, "length": 60}
            ],
            "generation_stats": {
                "generation_time": 45.2,
                "success_rate": 0.9,
                "successful_generations": 18,
                "avg_sequence_length": 155.5,
                "avg_confidence": 0.8
            },
            "ranking_stats": {
                "ranking_time": 12.3,
                "total_ranked": 18,
                "diversity_score": 0.72
            }
        }
        
        # Initialize quality gates
        config = QualityGateConfig(save_reports=False, verbose_logging=False)
        manager = QualityGateManager(config)
        
        # Run quality gates
        results = manager.execute_all_gates(test_context)
        
        print(f"  ✅ Executed {results['summary']['total_gates']} quality gates")
        print(f"  ✅ Results: {results['summary']['passed']} passed, {results['summary']['failed']} failed")
        print(f"  ✅ Overall status: {results['overall_result'].value}")
        
        return results['overall_result'].value in ['PASS', 'WARNING']
        
    except Exception as e:
        print(f"  ❌ Quality gates test failed: {e}")
        return False


def test_error_recovery():
    """Test error recovery mechanisms."""
    print("🔄 Testing Error Recovery System...")
    
    try:
        from protein_diffusion.error_recovery import (
            ErrorRecoveryManager, CircuitBreakerConfig, RetryConfig, 
            with_error_recovery, ErrorSeverity
        )
        
        # Initialize error recovery
        recovery_manager = ErrorRecoveryManager()
        
        # Register circuit breaker
        cb_config = CircuitBreakerConfig(failure_threshold=3, recovery_timeout=30.0)
        circuit_breaker = recovery_manager.register_circuit_breaker("test_service", cb_config)
        print("  ✅ Circuit breaker registered")
        
        # Register retry manager
        retry_config = RetryConfig(max_attempts=3, base_delay=0.1)
        retry_manager = recovery_manager.register_retry_manager("test_operation", retry_config)
        print("  ✅ Retry manager registered")
        
        # Test error handling
        def failing_function():
            raise ValueError("Test error")
        
        context = {
            "original_function": failing_function,
            "args": (),
            "kwargs": {},
            "operation_name": "test_operation"
        }
        
        # Handle error (should attempt recovery)
        result = recovery_manager.handle_error(
            ValueError("Test error"), 
            context, 
            "test_operation", 
            ErrorSeverity.MEDIUM
        )
        
        print(f"  ✅ Error recovery attempted: {result.get('strategy')}")
        
        # Get metrics
        metrics = recovery_manager.get_error_metrics()
        print(f"  ✅ Error metrics: {metrics.total_errors} total errors tracked")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error recovery test failed: {e}")
        return False


def test_caching_system():
    """Test hierarchical caching system."""
    print("⚡ Testing Advanced Caching System...")
    
    try:
        from protein_diffusion.advanced_caching import (
            HierarchicalCacheManager, CacheConfig, cached
        )
        
        # Initialize cache manager
        config = CacheConfig(
            l1_max_size=100,
            l2_cache_dir="./test_cache",
            background_cleanup=False
        )
        cache_manager = HierarchicalCacheManager(config)
        
        # Test basic operations
        test_data = {"protein_sequence": "MKLLILTCLVAVAL", "score": 0.95}
        
        # Put and get
        cache_manager.put("test_protein_1", test_data)
        retrieved_data = cache_manager.get("test_protein_1")
        assert retrieved_data == test_data
        print("  ✅ Basic cache operations work")
        
        # Test cache hierarchy
        for i in range(5):
            cache_manager.put(f"protein_{i}", {"seq": f"TEST{i}", "score": i * 0.1})
        
        # Get stats
        stats = cache_manager.get_comprehensive_stats()
        print(f"  ✅ L1 cache: {stats['l1_memory']['size']} items")
        print(f"  ✅ L2 cache: {stats['l2_ssd']['size']} items")
        print(f"  ✅ Hit rate: {stats['overall']['hit_rate']:.2f}")
        
        # Test decorator
        @cached(cache_manager, ttl=60.0)
        def expensive_computation(x):
            time.sleep(0.01)  # Simulate work
            return x * 2
        
        # First call (cache miss)
        start_time = time.time()
        result1 = expensive_computation(42)
        time1 = time.time() - start_time
        
        # Second call (cache hit)
        start_time = time.time()
        result2 = expensive_computation(42)
        time2 = time.time() - start_time
        
        assert result1 == result2 == 84
        print(f"  ✅ Cached function: {time1:.4f}s vs {time2:.4f}s (speedup: {time1/time2:.1f}x)")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Caching system test failed: {e}")
        return False


def test_performance_optimization():
    """Test performance optimization."""
    print("🚀 Testing Performance Optimization...")
    
    try:
        from protein_diffusion.performance_optimization import (
            PerformanceOptimizer, PerformanceConfig, OptimizationStrategy,
            optimize_performance
        )
        
        # Initialize performance optimizer
        config = PerformanceConfig(
            max_workers=2,
            enable_gpu=False,  # Disable GPU for testing
            default_batch_size=4
        )
        optimizer = PerformanceOptimizer(config)
        optimizer.start_optimization()
        
        # Test function optimization
        @optimize_performance(strategy=OptimizationStrategy.SEQUENTIAL)
        def test_computation(items):
            return [x * 2 for x in items]
        
        test_data = list(range(10))
        result = test_computation(test_data)
        expected = [x * 2 for x in test_data]
        assert result == expected
        print("  ✅ Function optimization works")
        
        # Test batch processing
        def batch_processor(batch):
            return [{"input": item, "output": item * 3} for item in batch]
        
        items = list(range(20))
        batch_results = optimizer.process_batch_optimized(items, batch_processor, batch_size=5)
        assert len(batch_results) == 20
        print(f"  ✅ Batch processing: {len(batch_results)} items processed")
        
        # Get performance summary
        summary = optimizer.get_performance_summary()
        print(f"  ✅ Performance status: {summary['status']}")
        
        optimizer.stop_optimization()
        return True
        
    except Exception as e:
        print(f"  ❌ Performance optimization test failed: {e}")
        return False


def test_deployment_readiness():
    """Test deployment readiness."""
    print("🚀 Testing Deployment Readiness...")
    
    try:
        from protein_diffusion.deployment_manager import (
            DeploymentManager, DeploymentConfig, EnvironmentType
        )
        
        # Test configuration generation
        config = DeploymentConfig(
            environment=EnvironmentType.DEVELOPMENT,
            app_name="protein-diffusion-test",
            version="1.0.0-test"
        )
        
        deployment_manager = DeploymentManager(config)
        
        # Generate Kubernetes manifests
        from protein_diffusion.deployment_manager import KubernetesManager
        k8s_manager = KubernetesManager(config)
        manifests = k8s_manager.generate_manifests()
        
        print(f"  ✅ Generated {len(manifests)} Kubernetes manifests")
        
        # Validate manifest structure
        required_manifests = ["deployment.yaml", "service.yaml", "configmap.yaml"]
        for manifest in required_manifests:
            assert manifest in manifests
            assert len(manifests[manifest]) > 100  # Should have substantial content
        
        print("  ✅ All required manifests generated")
        
        # Test deployment status
        status = deployment_manager.get_deployment_status()
        print(f"  ✅ Deployment status available: {len(status)} fields")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Deployment readiness test failed: {e}")
        return False


def test_monitoring_and_observability():
    """Test monitoring and observability."""
    print("📊 Testing Monitoring & Observability...")
    
    try:
        from protein_diffusion.robust_monitoring import (
            MonitoringManager, MetricsCollector, AlertManager, AlertRule, AlertSeverity
        )
        
        # Initialize monitoring
        monitoring_manager = MonitoringManager()
        monitoring_manager.start_monitoring()
        
        # Test metrics collection
        metrics_collector = monitoring_manager.metrics_collector
        
        # Record some test metrics
        for i in range(10):
            metrics_collector.increment_counter("test_requests", 1.0, {"endpoint": "generate"})
            metrics_collector.set_gauge("test_memory_usage", 1024 + i * 100)
            metrics_collector.record_histogram("test_latency", 0.1 + i * 0.01)
        
        # Get metrics
        latest_counter = metrics_collector.get_latest_value("test_requests")
        latest_gauge = metrics_collector.get_latest_value("test_memory_usage")
        
        print(f"  ✅ Metrics recorded: counter={latest_counter}, gauge={latest_gauge}")
        
        # Test alerting
        alert_manager = monitoring_manager.alert_manager
        
        # Add test alert rule
        test_rule = AlertRule(
            name="test_high_memory",
            metric_name="test_memory_usage",
            condition="> 1500",
            threshold=1500.0,
            severity=AlertSeverity.WARNING,
            description="Test memory usage alert"
        )
        alert_manager.add_alert_rule(test_rule)
        
        # Check for alerts
        active_alerts = alert_manager.get_active_alerts()
        print(f"  ✅ Alert system: {len(active_alerts)} active alerts")
        
        # Export metrics
        json_metrics = monitoring_manager.export_metrics("json")
        assert "timestamp" in json_metrics
        print("  ✅ Metrics export functional")
        
        monitoring_manager.stop_monitoring()
        return True
        
    except Exception as e:
        print(f"  ❌ Monitoring test failed: {e}")
        return False


def main():
    """Run all integration tests."""
    print("🧬 PROTEIN DIFFUSION DESIGN LAB - INTEGRATION TESTS")
    print("=" * 70)
    print("Testing autonomous SDLC implementation...")
    print()
    
    tests = [
        test_complete_workflow,
        test_quality_gates,
        test_error_recovery,
        test_caching_system,
        test_performance_optimization,
        test_deployment_readiness,
        test_monitoring_and_observability
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
                print("  ✅ PASSED\n")
            else:
                failed += 1
                print("  ❌ FAILED\n")
        except Exception as e:
            failed += 1
            print(f"  ❌ FAILED: {e}\n")
    
    print("=" * 70)
    print(f"🧪 INTEGRATION TEST RESULTS")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Success Rate: {passed/(passed+failed)*100:.1f}%")
    
    if failed == 0:
        print("\n🎉 ALL INTEGRATION TESTS PASSED!")
        print("🚀 Platform ready for production deployment!")
        
        # Generate test report
        report = {
            "timestamp": time.time(),
            "total_tests": len(tests),
            "passed": passed,
            "failed": failed,
            "success_rate": passed/(passed+failed),
            "status": "READY_FOR_PRODUCTION" if failed == 0 else "NEEDS_ATTENTION"
        }
        
        with open("integration_test_report.json", "w") as f:
            json.dump(report, f, indent=2)
        
        print("📄 Test report saved to integration_test_report.json")
    else:
        print(f"\n⚠️  {failed} tests failed - review and fix before deployment")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)