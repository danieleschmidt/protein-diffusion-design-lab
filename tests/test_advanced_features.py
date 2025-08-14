"""
Comprehensive tests for advanced protein diffusion features.

This module tests all the new advanced features including:
- Advanced generation techniques
- Novel architectures
- Enhanced APIs
- Research innovations
- Robust validation
- Security framework
- Distributed scaling
"""

import pytest
import time
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

# Import modules to test
try:
    from src.protein_diffusion.advanced_generation import (
        AdvancedGenerationConfig, AdaptiveSampler, MultiObjectiveOptimizer,
        QualityAwareGenerator, DiversityBooster, EvolutionaryRefiner
    )
    from src.protein_diffusion.novel_architectures import (
        HierarchicalDiffusionConfig, FlowBasedConfig, MemoryAugmentedConfig
    )
    from src.protein_diffusion.api_enhancements import (
        GenerationRequest, GenerationResult, ProteinTemplate, 
        StreamingGenerator, BatchProcessor, ProteinDesignWorkflow
    )
    from src.protein_diffusion.research_innovations import (
        FlowBasedProteinGenerator, GraphDiffusionModel, PhysicsInformedDiffusion
    )
    from src.protein_diffusion.robust_validation import (
        ValidationLevel, ValidationResult, SequenceValidator, 
        ModelInputValidator, RobustValidationManager
    )
    from src.protein_diffusion.security_framework import (
        SecurityLevel, SecurityContext, AccessLevel, InputSanitizer,
        AuthenticationManager, SecurityManager
    )
    from src.protein_diffusion.distributed_scaling import (
        DistributedCache, CacheLevel, WorkerProcess, WorkerConfig,
        LoadBalancer, AutoScaler, DistributedProcessingManager
    )
    IMPORTS_AVAILABLE = True
except ImportError as e:
    IMPORTS_AVAILABLE = False
    print(f"Warning: Could not import modules for testing: {e}")


@pytest.fixture
def mock_torch():
    """Mock torch for testing without PyTorch dependency."""
    with patch('src.protein_diffusion.advanced_generation.torch') as mock:
        mock.cuda.is_available.return_value = False
        mock.randn.return_value = Mock()
        mock.zeros.return_value = Mock()
        mock.ones.return_value = Mock()
        yield mock


@pytest.fixture
def sample_protein_sequence():
    """Sample protein sequence for testing."""
    return "MKLLVLGLFTLVLLGLVGLALSTDQMAELNALKQVTGMSSDVSSMSLAVATFATLQPKQNVPQMPALPKV"


@pytest.fixture
def sample_generation_request():
    """Sample generation request."""
    if not IMPORTS_AVAILABLE:
        return {}
    return GenerationRequest(
        target_type="antibody",
        sequence_length=120,
        num_candidates=10,
        quality_filter=True,
        diversity_boost=True
    )


class TestAdvancedGeneration:
    """Test advanced generation techniques."""
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Required modules not available")
    def test_advanced_generation_config(self):
        """Test advanced generation configuration."""
        config = AdvancedGenerationConfig()
        
        assert config.adaptive_temperature_range == (0.5, 1.5)
        assert config.population_size == 100
        assert config.objectives is not None
        assert isinstance(config.objective_weights, dict)
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Required modules not available")
    def test_adaptive_sampler(self, mock_torch):
        """Test adaptive sampling functionality."""
        config = AdvancedGenerationConfig()
        sampler = AdaptiveSampler(config)
        
        # Mock sample data
        current_samples = Mock()
        quality_scores = [0.8, 0.7, 0.9, 0.6, 0.85]
        
        temp, guidance = sampler.adapt_parameters(current_samples, quality_scores, 10)
        
        assert isinstance(temp, float)
        assert isinstance(guidance, float)
        assert 0.5 <= temp <= 1.5
        assert 0.5 <= guidance <= 2.0
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Required modules not available")
    def test_multi_objective_optimizer(self, mock_torch):
        """Test multi-objective optimization."""
        config = AdvancedGenerationConfig()
        optimizer = MultiObjectiveOptimizer(config)
        
        # Mock samples and metadata
        samples = Mock()
        samples.size.return_value = (5, 100, 512)  # batch_size, seq_len, hidden_dim
        metadata = {'confidences': Mock()}
        
        objective_scores = optimizer.evaluate_objectives(samples, metadata)
        
        assert isinstance(objective_scores, dict)
        assert len(objective_scores) > 0
        
        # Test weight computation
        weights = optimizer.compute_pareto_weights(objective_scores)
        assert weights is not None
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Required modules not available")
    def test_quality_aware_generator(self, mock_torch):
        """Test quality-aware generation."""
        config = AdvancedGenerationConfig()
        generator = QualityAwareGenerator(config)
        
        # Mock samples
        samples = Mock()
        samples.size.return_value = (5, 100, 20000)  # batch, seq, vocab
        
        with patch.object(generator.quality_predictor, 'forward') as mock_predictor:
            mock_predictor.return_value = Mock()
            mock_predictor.return_value.tolist.return_value = [0.8, 0.7, 0.9, 0.6, 0.85]
            
            filtered_samples, quality_scores, quality_list = generator.filter_samples(samples)
            
            assert len(quality_list) == 5
            assert all(0 <= score <= 1 for score in quality_list)
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Required modules not available")
    def test_diversity_booster(self):
        """Test diversity boosting functionality."""
        config = AdvancedGenerationConfig()
        booster = DiversityBooster(config)
        
        # Add some sequences to memory
        sequences = [
            "MKLLVLGLFT",
            "ALSTDQMAEL",
            "QVTGMSSDVS"
        ]
        booster.add_sequences(sequences)
        
        # Test diversity penalty calculation
        test_sequences = ["MKLLVLGLFT", "XYZNEWSEQ"]  # One similar, one different
        penalty = booster.calculate_diversity_penalty(Mock(), test_sequences)
        
        assert len(penalty) == 2
        assert penalty[0] > penalty[1]  # First sequence should have higher penalty


class TestAPIEnhancements:
    """Test enhanced API functionality."""
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Required modules not available")
    def test_generation_request(self):
        """Test generation request structure."""
        request = GenerationRequest(
            target_type="enzyme",
            sequence_length=200,
            num_candidates=15
        )
        
        assert request.target_type == "enzyme"
        assert request.sequence_length == 200
        assert request.num_candidates == 15
        assert request.quality_filter is True  # Default
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Required modules not available")
    def test_generation_result(self):
        """Test generation result structure."""
        result = GenerationResult(
            sequence="MKLLVLGLFT",
            confidence=0.85,
            generation_time=2.3
        )
        
        assert result.sequence == "MKLLVLGLFT"
        assert result.confidence == 0.85
        assert result.generation_time == 2.3
        
        # Test serialization
        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert result_dict["sequence"] == "MKLLVLGLFT"
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Required modules not available")
    def test_protein_templates(self):
        """Test protein template functionality."""
        antibody_template = ProteinTemplate.get_antibody_template()
        enzyme_template = ProteinTemplate.get_enzyme_template()
        
        assert antibody_template.name == "antibody"
        assert enzyme_template.name == "enzyme"
        assert antibody_template.config["sequence_length"] == 120
        assert enzyme_template.config["sequence_length"] == 200
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Required modules not available")
    def test_streaming_generator(self):
        """Test streaming generation functionality."""
        mock_diffuser = Mock()
        generator = StreamingGenerator(mock_diffuser)
        
        # Add callbacks
        progress_callback = Mock()
        quality_callback = Mock()
        
        generator.add_progress_callback(progress_callback)
        generator.add_quality_callback(quality_callback)
        
        assert len(generator.progress_callbacks) == 1
        assert len(generator.quality_callbacks) == 1
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Required modules not available")
    def test_batch_processor(self):
        """Test batch processing functionality."""
        mock_diffuser = Mock()
        mock_ranker = Mock()
        processor = BatchProcessor(mock_diffuser, mock_ranker)
        
        # Mock generation results
        mock_diffuser.generate.return_value = [
            {"sequence": "MKLLVLGLFT", "confidence": 0.8},
            {"sequence": "ALSTDQMAEL", "confidence": 0.7}
        ]
        
        requests = [
            GenerationRequest(num_candidates=2),
            GenerationRequest(num_candidates=2)
        ]
        
        with patch.object(processor, '_process_request_batch') as mock_process:
            mock_process.return_value = [[Mock(), Mock()], [Mock(), Mock()]]
            results = processor.process_batch(requests, max_concurrent=2)
            
            assert len(results) == 2
            assert len(results[0]) == 2


class TestRobustValidation:
    """Test robust validation system."""
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Required modules not available")
    def test_sequence_validator(self, sample_protein_sequence):
        """Test protein sequence validation."""
        validator = SequenceValidator(ValidationLevel.NORMAL)
        
        # Test valid sequence
        result = validator.validate(sample_protein_sequence)
        assert result.is_valid
        assert len(result.errors) == 0
        
        # Test invalid sequence
        invalid_sequence = "INVALID@SEQUENCE#WITH$SYMBOLS"
        result = validator.validate(invalid_sequence)
        assert not result.is_valid
        assert len(result.errors) > 0
        
        # Test too short sequence
        short_sequence = "MKL"
        result = validator.validate(short_sequence)
        assert not result.is_valid
        assert any("too short" in error.lower() for error in result.errors)
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Required modules not available")
    def test_model_input_validator(self):
        """Test model input parameter validation."""
        validator = ModelInputValidator(ValidationLevel.NORMAL)
        
        # Test valid parameters
        valid_params = {
            "num_samples": 10,
            "temperature": 1.0,
            "guidance_scale": 1.5,
            "max_length": 200
        }
        result = validator.validate(valid_params)
        assert result.is_valid
        
        # Test invalid parameters
        invalid_params = {
            "num_samples": -5,  # Invalid: negative
            "temperature": 0,   # Invalid: zero
            "guidance_scale": -1  # Invalid: negative
        }
        result = validator.validate(invalid_params)
        assert not result.is_valid
        assert len(result.errors) > 0
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Required modules not available")
    def test_validation_manager(self, sample_protein_sequence):
        """Test comprehensive validation manager."""
        manager = RobustValidationManager(ValidationLevel.NORMAL)
        
        generation_params = {
            "num_samples": 5,
            "temperature": 0.8,
            "max_length": 100
        }
        
        result = manager.comprehensive_validation(
            sequence=sample_protein_sequence,
            generation_params=generation_params
        )
        
        assert isinstance(result, ValidationResult)
        assert result.validation_time > 0
        
        # Test statistics
        stats = manager.get_validation_statistics()
        assert "total_validations" in stats
        assert stats["total_validations"] >= 1


class TestSecurityFramework:
    """Test security framework functionality."""
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Required modules not available")
    def test_input_sanitizer(self, sample_protein_sequence):
        """Test input sanitization."""
        sanitizer = InputSanitizer(SecurityLevel.MEDIUM)
        
        # Test sequence sanitization
        clean_sequence = sanitizer.sanitize_sequence(sample_protein_sequence)
        assert isinstance(clean_sequence, str)
        assert len(clean_sequence) > 0
        
        # Test malicious sequence detection
        with pytest.raises(ValueError):
            sanitizer.sanitize_sequence("DROP TABLE proteins; --")
        
        # Test filename sanitization
        clean_filename = sanitizer.sanitize_filename("test_file.pdb")
        assert clean_filename == "test_file.pdb"
        
        with pytest.raises(ValueError):
            sanitizer.sanitize_filename("../../etc/passwd")
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Required modules not available")
    def test_authentication_manager(self):
        """Test authentication and authorization."""
        from src.protein_diffusion.security_framework import SecurityConfig
        
        config = SecurityConfig()
        auth_manager = AuthenticationManager(config)
        
        # Test API key generation
        api_key = auth_manager.generate_api_key("test_user", AccessLevel.USER)
        assert isinstance(api_key, str)
        assert len(api_key) > 0
        
        # Test API key validation
        context = auth_manager.validate_api_key(api_key)
        assert context is not None
        assert context.user_id == "test_user"
        assert context.access_level == AccessLevel.USER
        
        # Test invalid API key
        invalid_context = auth_manager.validate_api_key("invalid_key")
        assert invalid_context is None
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Required modules not available")
    def test_security_manager(self):
        """Test comprehensive security manager."""
        manager = SecurityManager()
        
        # Test request authentication
        api_key = manager.auth_manager.generate_api_key("test_user", AccessLevel.USER)
        context = manager.authenticate_request(api_key=api_key)
        
        assert context is not None
        assert context.user_id == "test_user"
        
        # Test authorization
        has_permission = manager.authorize_request(context, "generate_basic")
        assert has_permission
        
        has_admin_permission = manager.authorize_request(context, "admin_access")
        assert not has_admin_permission  # User level shouldn't have admin access


class TestDistributedScaling:
    """Test distributed scaling functionality."""
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Required modules not available")
    def test_distributed_cache(self):
        """Test hierarchical distributed cache."""
        config = {'l1_size': 1, 'l2_size': 10, 'l3_size': 100}
        cache = DistributedCache(config)
        
        # Test cache set/get
        test_key = "test_key"
        test_value = "test_value"
        
        cache.set(test_key, test_value, CacheLevel.L1_MEMORY)
        retrieved_value = cache.get(test_key)
        
        assert retrieved_value == test_value
        
        # Test cache miss
        missing_value = cache.get("nonexistent_key")
        assert missing_value is None
        
        # Test cache statistics
        stats = cache.get_stats()
        assert isinstance(stats, CacheStats)
        assert stats.hit_rate >= 0
        assert stats.miss_rate >= 0
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Required modules not available")
    def test_worker_process(self):
        """Test worker process functionality."""
        config = WorkerConfig(worker_id=1, cpu_cores=2, memory_limit_gb=4.0)
        worker = WorkerProcess(config)
        
        # Mock initialization to avoid actual model loading
        with patch.object(worker, 'initialize'):
            worker.initialize()
            worker.is_healthy = True
            worker.cache = Mock()
            worker.cache.get.return_value = None
            worker.cache.set = Mock()
            
            # Test request processing
            request = {
                "num_samples": 2,
                "max_length": 50,
                "temperature": 1.0
            }
            
            with patch.object(worker, '_generate_response') as mock_generate:
                mock_generate.return_value = {
                    "sequences": [{"sequence": "MKLL", "confidence": 0.8}],
                    "worker_id": 1
                }
                
                result = worker.process_request(request)
                assert result["worker_id"] == 1
                assert "sequences" in result
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Required modules not available")
    def test_load_balancer(self):
        """Test load balancing functionality."""
        # Create mock workers
        workers = []
        for i in range(3):
            worker = Mock()
            worker.config = WorkerConfig(worker_id=i)
            worker.is_healthy = True
            workers.append(worker)
        
        load_balancer = LoadBalancer(workers)
        
        # Test worker selection
        request = {"num_samples": 5}
        selected_worker = load_balancer.select_worker(request)
        
        assert selected_worker is not None
        assert selected_worker in workers
        
        # Test load distribution
        distribution = load_balancer.get_load_distribution()
        assert "total_load" in distribution
        assert "healthy_workers" in distribution
        assert distribution["healthy_workers"] == 3
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Required modules not available")
    def test_auto_scaler(self):
        """Test auto-scaling functionality."""
        from src.protein_diffusion.distributed_scaling import ScalingMetrics, ScalingStrategy
        
        scaler = AutoScaler(ScalingStrategy.REACTIVE)
        
        # Test scaling decision with high load
        high_load_metrics = ScalingMetrics(
            timestamp=time.time(),
            active_workers=2,
            queue_length=10,
            avg_response_time=15.0,
            cpu_utilization=0.9,
            memory_utilization=0.85,
            gpu_utilization=0.8,
            requests_per_second=50.0,
            error_rate=0.01,
            cost_per_hour=10.0
        )
        
        should_scale, scale_amount = scaler.should_scale(high_load_metrics)
        
        if should_scale:
            assert scale_amount > 0  # Should scale up
        
        # Test scaling decision with low load
        low_load_metrics = ScalingMetrics(
            timestamp=time.time(),
            active_workers=5,
            queue_length=1,
            avg_response_time=1.0,
            cpu_utilization=0.2,
            memory_utilization=0.25,
            gpu_utilization=0.1,
            requests_per_second=2.0,
            error_rate=0.001,
            cost_per_hour=25.0
        )
        
        # Need to wait for cooldown or modify last action time
        scaler.last_scaling_action = 0  # Reset cooldown
        should_scale, scale_amount = scaler.should_scale(low_load_metrics)
        
        if should_scale:
            assert scale_amount < 0  # Should scale down


class TestIntegration:
    """Integration tests for combined functionality."""
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Required modules not available")
    def test_workflow_integration(self, sample_generation_request):
        """Test integrated workflow from request to result."""
        # This would test the full pipeline but requires actual models
        # For now, test that components can be instantiated together
        
        try:
            config = AdvancedGenerationConfig()
            validator = SequenceValidator()
            sanitizer = InputSanitizer()
            
            # Test that components work together
            assert config is not None
            assert validator is not None
            assert sanitizer is not None
            
        except Exception as e:
            pytest.fail(f"Integration test failed: {e}")
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Required modules not available")
    def test_error_handling_integration(self):
        """Test that error handling works across components."""
        validator = SequenceValidator(ValidationLevel.STRICT)
        sanitizer = InputSanitizer(SecurityLevel.HIGH)
        
        # Test invalid input handling
        with pytest.raises(ValueError):
            sanitizer.sanitize_sequence("INVALID_SEQUENCE_WITH_NUMBERS123")
        
        # Test validation error handling
        result = validator.validate("TOO_SHORT")
        assert not result.is_valid
        assert len(result.errors) > 0


class TestPerformance:
    """Performance and benchmarking tests."""
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Required modules not available")
    def test_cache_performance(self):
        """Test cache performance under load."""
        config = {'l1_size': 1, 'l2_size': 10}
        cache = DistributedCache(config)
        
        # Measure cache operations
        start_time = time.time()
        
        for i in range(100):
            cache.set(f"key_{i}", f"value_{i}")
        
        set_time = time.time() - start_time
        
        start_time = time.time()
        
        for i in range(100):
            cache.get(f"key_{i}")
        
        get_time = time.time() - start_time
        
        # Basic performance assertions
        assert set_time < 1.0  # Should complete in under 1 second
        assert get_time < 0.5  # Gets should be faster than sets
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Required modules not available")
    def test_validation_performance(self):
        """Test validation performance."""
        validator = SequenceValidator()
        
        # Generate test sequences
        test_sequences = [
            "MKLLVLGLFTLVLLGLVGLALSTDQMAELNALKQVTGMSSDVSSMSLAVATFATLQPKQNVPQMPALPKV" * i
            for i in range(1, 11)
        ]
        
        start_time = time.time()
        
        for sequence in test_sequences:
            validator.validate(sequence)
        
        validation_time = time.time() - start_time
        
        # Should validate 10 sequences quickly
        assert validation_time < 0.5
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Required modules not available")
    def test_security_overhead(self):
        """Test security framework overhead."""
        manager = SecurityManager()
        
        # Measure authentication overhead
        api_key = manager.auth_manager.generate_api_key("test_user", AccessLevel.USER)
        
        start_time = time.time()
        
        for _ in range(100):
            context = manager.authenticate_request(api_key=api_key)
            manager.authorize_request(context, "generate_basic")
        
        auth_time = time.time() - start_time
        
        # Authentication should be fast
        assert auth_time < 0.1


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])