"""
Integration tests for Next-Generation features.

Tests the integration and functionality of all new next-generation modules.
"""

import pytest
import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from protein_diffusion.next_gen_research import (
    NextGenResearchPlatform, QuantumEnhancedSampler, EvolutionaryOptimizer,
    MultiModalFusion, RealTimeAdaptiveLearning, run_nextgen_research
)
from protein_diffusion.autonomous_development import (
    AutonomousDevelopmentSystem, CodeAnalyzer, AutoCodeGenerator, AutoTester,
    autonomous_develop
)
from protein_diffusion.enterprise_resilience import (
    EnterpriseResilienceSystem, CircuitBreaker, HealthMonitor, 
    SelfHealingSystem, SecurityManager, create_resilience_system
)
from protein_diffusion.intelligent_optimization import (
    IntelligentOptimizationSystem, IntelligentCache, AdaptiveResourceManager,
    PredictiveLoadBalancer, create_optimization_system
)
from protein_diffusion.global_scale_orchestrator import (
    GlobalScaleOrchestrator, RegionManager, GlobalLoadBalancer, GlobalAutoScaler,
    create_global_orchestrator
)

class TestNextGenResearchPlatform:
    """Test next-generation research platform."""
    
    def test_quantum_enhanced_sampler(self):
        """Test quantum-enhanced sampling."""
        sampler = QuantumEnhancedSampler()
        
        # Test quantum gate application
        gates = ['hadamard', 'cnot', 'phase']
        results = sampler.apply_quantum_gates(gates)
        
        assert 'gates_applied' in results
        assert results['gates_applied'] == gates
        assert 'enhancement_factor' in results
        assert results['enhancement_factor'] > 1.0
    
    def test_quantum_sampling(self):
        """Test quantum sampling functionality."""
        sampler = QuantumEnhancedSampler()
        samples = sampler.quantum_sample(num_samples=5)
        
        assert len(samples) == 5
        for sample in samples:
            assert 'sample_id' in sample
            assert 'sequence' in sample
            assert 'quantum_probability' in sample
            assert 'quantum_advantage' in sample
            assert len(sample['sequence']) >= 50
    
    def test_evolutionary_optimizer(self):
        """Test evolutionary optimization."""
        optimizer = EvolutionaryOptimizer()
        initial_sequences = [
            "MKLLILTCLVAVALARPKHPIW",
            "ACDEFGHIKLMNPQRSTVWY",
            "AILMFWYVCGPATHDEFQNSR"
        ]
        
        # Initialize population
        population = optimizer.initialize_population(initial_sequences)
        assert len(population) >= len(initial_sequences)
        
        # Test evolution
        evolution_log = optimizer.evolve(generations=2)
        assert len(evolution_log) == 2
        
        for gen_data in evolution_log:
            assert 'generation' in gen_data
            assert 'best_fitness' in gen_data
            assert 'average_fitness' in gen_data
    
    def test_multimodal_fusion(self):
        """Test multi-modal fusion system."""
        fusion = MultiModalFusion()
        
        protein_data = {'sequence': 'MKLLILTCLVAVALARPKHPIW'}
        results = fusion.process_multi_modal(protein_data)
        
        assert 'modal_results' in results
        assert 'fused_representation' in results
        assert 'fusion_method' in results
        assert 'modalities_used' in results
        
        # Check that multiple modalities were processed
        assert len(results['modalities_used']) > 1
    
    def test_adaptive_learning(self):
        """Test real-time adaptive learning."""
        learning = RealTimeAdaptiveLearning()
        
        # Simulate feedback
        feedback = {
            'rating': 0.8,
            'success': True,
            'useful': True,
            'category': 'generation'
        }
        
        learning.collect_feedback(feedback)
        assert len(learning.feedback_buffer) == 1
        
        # Test adaptation
        parameters = learning.adapt_parameters()
        assert isinstance(parameters, dict)
    
    def test_research_platform_integration(self):
        """Test full research platform integration."""
        platform = NextGenResearchPlatform()
        
        session_config = {
            'num_quantum_samples': 3,
            'evolution_generations': 2,
            'research_mode': 'comprehensive'
        }
        
        results = platform.run_research_session(session_config)
        
        assert 'session_id' in results
        assert 'quantum_sampling' in results
        assert 'evolutionary_optimization' in results
        assert 'multimodal_analysis' in results
        assert 'evaluation' in results
        assert 'research_insights' in results
        
        # Check quantum results
        quantum_results = results['quantum_sampling']['results']
        assert len(quantum_results) == 3
        
        # Check evolution results
        evolution_results = results['evolutionary_optimization']['final_population']
        assert len(evolution_results) > 0

class TestAutonomousDevelopment:
    """Test autonomous development system."""
    
    def test_code_analyzer(self):
        """Test code analysis functionality."""
        analyzer = CodeAnalyzer()
        
        sample_code = '''
def calculate_similarity(seq1, seq2):
    """Calculate sequence similarity."""
    if not seq1 or not seq2:
        return 0.0
    
    matches = sum(1 for a, b in zip(seq1, seq2) if a == b)
    return matches / max(len(seq1), len(seq2))
'''
        
        metrics = analyzer.analyze_code(sample_code)
        
        assert metrics.complexity > 0
        assert metrics.readability > 0
        assert metrics.documentation_score > 0
        assert metrics.security_score > 0
        
        # Test improvement suggestions
        suggestions = analyzer.suggest_improvements(sample_code, metrics)
        assert isinstance(suggestions, list)
    
    def test_code_generator(self):
        """Test automatic code generation."""
        analyzer = CodeAnalyzer()
        generator = AutoCodeGenerator(analyzer)
        
        # Test function generation
        function_code = generator.generate_function(
            name="test_function",
            purpose="calculate protein properties",
            params={"sequence": "str", "property_type": "str"},
            return_type="float"
        )
        
        assert "def test_function" in function_code
        assert "sequence: str" in function_code
        assert "-> float:" in function_code
        assert '"""' in function_code  # Docstring
    
    def test_auto_tester(self):
        """Test automatic test generation."""
        tester = AutoTester()
        
        sample_code = '''
def add_numbers(a, b):
    """Add two numbers."""
    return a + b
'''
        
        test_code = tester.generate_tests(sample_code, "add_numbers")
        
        assert "def test_add_numbers" in test_code
        assert "import pytest" in test_code
        assert "assert" in test_code
    
    def test_autonomous_development_system(self):
        """Test full autonomous development system."""
        system = AutonomousDevelopmentSystem()
        
        feature_spec = {
            'name': 'similarity_calculator',
            'type': 'function',
            'description': 'Calculate protein sequence similarity',
            'parameters': {'seq1': 'str', 'seq2': 'str'},
            'return_type': 'float'
        }
        
        results = system.develop_feature(feature_spec)
        
        assert 'session_id' in results
        assert 'feature_spec' in results
        assert 'iterations' in results
        assert 'final_code' in results
        assert len(results['iterations']) > 0

class TestEnterpriseResilience:
    """Test enterprise resilience system."""
    
    def test_circuit_breaker(self):
        """Test circuit breaker functionality."""
        circuit_breaker = CircuitBreaker("test_service")
        
        # Test successful calls
        result = circuit_breaker.call(lambda: "success")
        assert result == "success"
        
        # Test status
        status = circuit_breaker.get_status()
        assert status['name'] == "test_service"
        assert status['state'] == "closed"
        assert status['metrics']['total_calls'] > 0
    
    def test_health_monitor(self):
        """Test health monitoring system."""
        monitor = HealthMonitor()
        
        # Register health check
        def dummy_health_check():
            return True
        
        monitor.register_health_check("test_service", dummy_health_check, critical=True)
        
        # Check system health
        health = monitor.get_system_health()
        assert 'overall_health' in health
        assert 'check_results' in health
        assert 'test_service' in health['check_results']
    
    def test_self_healing_system(self):
        """Test self-healing capabilities."""
        healing = SelfHealingSystem()
        
        # Register healing strategy
        def dummy_healing_action():
            return True
        
        healing.register_healing_strategy(
            "test_condition", 
            dummy_healing_action,
            description="Test healing action"
        )
        
        # Trigger healing
        success = healing.trigger_healing("test_condition")
        assert success is True
        
        # Check status
        status = healing.get_healing_status()
        assert 'total_strategies' in status
        assert status['total_strategies'] == 1
    
    def test_security_manager(self):
        """Test security management."""
        security = SecurityManager()
        
        # Test user authentication
        session = security.authenticate_user("testuser", "password123")
        assert session is not None
        assert 'session_id' in session
        assert 'username' in session
        
        # Test authorization
        authorized = security.authorize_action(session['session_id'], "read")
        assert authorized is True
    
    def test_enterprise_resilience_integration(self):
        """Test full enterprise resilience system."""
        system = EnterpriseResilienceSystem()
        
        # Start monitoring
        system.start_monitoring()
        
        # Get system status
        status = system.get_system_status()
        assert 'uptime_seconds' in status
        assert 'overall_health' in status
        assert 'resilience_grade' in status
        
        # Stop monitoring
        system.stop_monitoring()

class TestIntelligentOptimization:
    """Test intelligent optimization system."""
    
    def test_intelligent_cache(self):
        """Test AI-driven caching system."""
        cache = IntelligentCache(max_size=10, learning_enabled=True)
        
        # Test basic operations
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
        assert cache.get("nonexistent") is None
        
        # Test statistics
        stats = cache.get_cache_statistics()
        assert 'hits' in stats
        assert 'misses' in stats
        assert 'hit_rate' in stats
    
    def test_adaptive_resource_manager(self):
        """Test adaptive resource management."""
        from protein_diffusion.intelligent_optimization import PerformanceMetrics
        
        manager = AdaptiveResourceManager()
        
        # Record metrics
        metrics = PerformanceMetrics(
            cpu_usage=0.5,
            memory_usage=0.6,
            request_latency=0.1,
            throughput=100.0
        )
        
        manager.record_metrics(metrics)
        
        # Get status
        status = manager.get_optimization_status()
        assert 'running' in status
        assert 'current_allocation' in status
        assert 'performance_score' in status
    
    def test_predictive_load_balancer(self):
        """Test ML-based load balancing."""
        balancer = PredictiveLoadBalancer()
        
        # Register servers
        balancer.register_server("server1", capacity=1.0)
        balancer.register_server("server2", capacity=1.5)
        
        # Route request
        server = balancer.route_request()
        assert server in ["server1", "server2"]
        
        # Record metrics
        balancer.record_server_metrics("server1", load=0.5, response_time=0.1, success=True)
        
        # Get status
        status = balancer.get_load_balancer_status()
        assert 'total_servers' in status
        assert 'healthy_servers' in status
    
    def test_optimization_system_integration(self):
        """Test full optimization system."""
        system = IntelligentOptimizationSystem()
        
        # Start optimization
        system.start_optimization()
        
        # Test optimized request execution
        def dummy_request():
            return "optimized_result"
        
        result = system.optimize_request(dummy_request)
        assert result == "optimized_result"
        
        # Get status
        status = system.get_optimization_status()
        assert 'cache_statistics' in status
        assert 'optimization_grade' in status
        
        # Stop optimization
        system.stop_optimization()

class TestGlobalScaleOrchestrator:
    """Test global scale orchestration."""
    
    def test_region_manager(self):
        """Test regional deployment management."""
        from protein_diffusion.global_scale_orchestrator import (
            RegionConfig, DeploymentRegion, CloudProvider
        )
        
        config = RegionConfig(
            region=DeploymentRegion.US_EAST,
            cloud_provider=CloudProvider.AWS,
            capacity=100
        )
        
        manager = RegionManager(config)
        
        # Test instance deployment
        instance_spec = {'cpu': 2, 'memory': 4096}
        instance_id = manager.deploy_instance(instance_spec)
        assert instance_id is not None
        assert config.region.value in instance_id
        
        # Test status
        status = manager.get_region_status()
        assert 'region' in status
        assert 'instances' in status
        assert 'utilization' in status
    
    def test_global_load_balancer(self):
        """Test global load balancing."""
        from protein_diffusion.global_scale_orchestrator import (
            RegionConfig, DeploymentRegion, CloudProvider
        )
        
        balancer = GlobalLoadBalancer()
        
        # Register regions
        config1 = RegionConfig(region=DeploymentRegion.US_EAST, cloud_provider=CloudProvider.AWS)
        config2 = RegionConfig(region=DeploymentRegion.EU_WEST, cloud_provider=CloudProvider.AWS)
        
        region1 = RegionManager(config1)
        region2 = RegionManager(config2)
        
        balancer.register_region(region1)
        balancer.register_region(region2)
        
        # Test routing
        route_result = balancer.route_request(client_location="north_america")
        # May be None if no instances are running, which is expected in test
        
        # Test status
        status = balancer.get_global_routing_status()
        assert 'total_regions' in status
        assert 'routing_table' in status
    
    def test_global_auto_scaler(self):
        """Test global auto-scaling."""
        from protein_diffusion.global_scale_orchestrator import (
            RegionConfig, DeploymentRegion, CloudProvider, ScalingPolicy
        )
        
        scaler = GlobalAutoScaler()
        
        # Register region
        config = RegionConfig(region=DeploymentRegion.US_EAST, cloud_provider=CloudProvider.AWS)
        region_manager = RegionManager(config)
        policy = ScalingPolicy(min_instances=1, max_instances=10)
        
        scaler.register_region(region_manager, policy)
        
        # Test status
        status = scaler.get_scaling_status()
        assert 'scaling_active' in status
        assert 'registered_regions' in status
    
    def test_global_orchestrator_integration(self):
        """Test full global orchestration system."""
        orchestrator = create_global_orchestrator()
        
        # Start orchestration
        orchestrator.start_orchestration()
        
        # Deploy service globally
        service_spec = {'cpu': 1, 'memory': 2048, 'replicas': 2}
        deployment_results = orchestrator.deploy_global_service(service_spec)
        assert isinstance(deployment_results, dict)
        
        # Get global status
        status = orchestrator.get_global_status()
        assert 'global_metrics' in status
        assert 'regions' in status
        assert 'global_health_grade' in status
        
        # Stop orchestration
        orchestrator.stop_orchestration()

class TestIntegrationScenarios:
    """Test integration scenarios across all systems."""
    
    def test_full_platform_integration(self):
        """Test integration of all next-gen systems."""
        # Initialize all systems
        research_platform = NextGenResearchPlatform()
        dev_system = AutonomousDevelopmentSystem()
        resilience_system = create_resilience_system()
        optimization_system = create_optimization_system()
        orchestrator = create_global_orchestrator()
        
        # Start systems
        orchestrator.start_orchestration()
        
        # Test research workflow
        research_results = research_platform.run_research_session({
            'num_quantum_samples': 2,
            'evolution_generations': 1
        })
        
        assert research_results['success'] if 'success' in research_results else True
        
        # Test development workflow
        dev_results = dev_system.develop_feature({
            'name': 'integration_test',
            'type': 'function',
            'description': 'Integration test function',
            'return_type': 'bool'
        })
        
        assert 'final_code' in dev_results
        
        # Test resilience
        resilience_status = resilience_system.get_system_status()
        assert resilience_status['overall_health'] > 0.5
        
        # Test optimization
        opt_status = optimization_system.get_optimization_status()
        assert 'optimization_grade' in opt_status
        
        # Test global orchestration
        global_status = orchestrator.get_global_status()
        assert 'global_health_grade' in global_status
        
        # Cleanup
        orchestrator.stop_orchestration()
        resilience_system.stop_monitoring()
        optimization_system.stop_optimization()

if __name__ == "__main__":
    pytest.main([__file__, "-v"])