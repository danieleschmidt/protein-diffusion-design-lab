"""
Comprehensive test suite for TPU optimization components.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import json
import time

# Import components to test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from protein_diffusion.tpu_optimization import (
    TPUOptimizer, TPUConfig, TPUBackend, TPUVersion, create_tpu_optimizer
)
from protein_diffusion.zero_nas import (
    ZeroNAS, ArchitectureConfig, ProteinDiffusionArchitecture,
    ZeroCostEvaluator, SearchStrategy, create_protein_diffusion_search_space
)
from protein_diffusion.tpu_nas_integration import (
    TPUNeuralArchitectureSearch, TPUNASConfig, run_tpu_nas_search
)


class TestTPUOptimizer(unittest.TestCase):
    """Test TPU Optimizer functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = TPUConfig(
            backend=TPUBackend.JAX,
            version=TPUVersion.V6E,
            num_cores=8
        )
    
    def test_tpu_config_creation(self):
        """Test TPU configuration creation."""
        config = TPUConfig()
        self.assertIsInstance(config, TPUConfig)
        self.assertEqual(config.backend, TPUBackend.JAX)
        self.assertEqual(config.version, TPUVersion.V6E)
        self.assertEqual(config.num_cores, 8)
    
    def test_tpu_config_validation(self):
        """Test TPU configuration validation."""
        # Valid configuration should not raise
        config = TPUConfig(backend=TPUBackend.JAX, version=TPUVersion.V6E)
        self.assertIsInstance(config, TPUConfig)
        
        # Test enum values
        self.assertIn(config.backend, TPUBackend)
        self.assertIn(config.version, TPUVersion)
    
    def test_tpu_optimizer_initialization(self):
        """Test TPU optimizer initialization."""
        optimizer = TPUOptimizer(self.config)
        self.assertIsInstance(optimizer, TPUOptimizer)
        self.assertEqual(optimizer.config, self.config)
    
    def test_get_hardware_info(self):
        """Test hardware info retrieval."""
        optimizer = TPUOptimizer(self.config)
        hardware_info = optimizer.get_hardware_info()
        
        self.assertIsInstance(hardware_info, dict)
        self.assertIn('backend', hardware_info)
        self.assertIn('config', hardware_info)
        
        # Check config structure
        config_info = hardware_info['config']
        self.assertIn('version', config_info)
        self.assertIn('num_cores', config_info)
        self.assertIn('precision', config_info)
    
    def test_optimal_batch_size_calculation(self):
        """Test optimal batch size calculation."""
        optimizer = TPUOptimizer(self.config)
        
        # Test with different model sizes
        test_cases = [
            (10_000_000, 512),    # 10M params, 512 seq len
            (100_000_000, 512),   # 100M params, 512 seq len
            (1_000_000_000, 256), # 1B params, 256 seq len
        ]
        
        for model_params, seq_length in test_cases:
            batch_size = optimizer.get_optimal_batch_size(model_params, seq_length)
            
            # Batch size should be positive and reasonable
            self.assertGreater(batch_size, 0)
            self.assertLessEqual(batch_size, 1024)  # Reasonable upper bound
            
            # Should be divisible by number of cores for even distribution
            self.assertEqual(batch_size % self.config.num_cores, 0)
    
    def test_model_optimization(self):
        """Test model optimization for TPU."""
        optimizer = TPUOptimizer(self.config)
        
        # Create dummy model
        class DummyModel:
            def __init__(self):
                self.optimized = False
            
            def __call__(self, *args):
                return "output"
        
        model = DummyModel()
        input_shape = (32, 512)  # batch_size, seq_length
        
        # Optimize model
        optimized_model = optimizer.optimize_model_for_tpu(model, input_shape)
        
        # Should return some form of optimized model
        self.assertIsNotNone(optimized_model)
    
    def test_factory_function(self):
        """Test factory function for creating TPU optimizer."""
        optimizer = create_tpu_optimizer("jax", "v6e", 8)
        
        self.assertIsInstance(optimizer, TPUOptimizer)
        self.assertEqual(optimizer.config.backend, TPUBackend.JAX)
        self.assertEqual(optimizer.config.version, TPUVersion.V6E)
        self.assertEqual(optimizer.config.num_cores, 8)


class TestZeroNAS(unittest.TestCase):
    """Test ZeroNAS functionality."""
    
    def test_architecture_config_creation(self):
        """Test architecture configuration creation."""
        config = ArchitectureConfig(
            num_layers=12,
            hidden_size=768,
            num_attention_heads=12
        )
        
        self.assertEqual(config.num_layers, 12)
        self.assertEqual(config.hidden_size, 768)
        self.assertEqual(config.num_attention_heads, 12)
        
        # Test serialization
        config_dict = config.to_dict()
        self.assertIsInstance(config_dict, dict)
        self.assertEqual(config_dict['num_layers'], 12)
    
    def test_architecture_creation(self):
        """Test protein diffusion architecture creation."""
        config = ArchitectureConfig()
        architecture = ProteinDiffusionArchitecture(config)
        
        # Basic properties
        self.assertEqual(architecture.config, config)
        self.assertIsInstance(architecture.arch_id, str)
        self.assertGreater(len(architecture.arch_id), 0)
        
        # Parameter estimation
        self.assertGreater(architecture.estimated_params, 0)
        self.assertGreater(architecture.estimated_flops, 0)
        
        # Two architectures with same config should have same ID
        architecture2 = ProteinDiffusionArchitecture(config)
        self.assertEqual(architecture.arch_id, architecture2.arch_id)
    
    def test_parameter_estimation(self):
        """Test parameter and FLOP estimation."""
        # Small architecture
        small_config = ArchitectureConfig(
            num_layers=6,
            hidden_size=384,
            num_attention_heads=6
        )
        small_arch = ProteinDiffusionArchitecture(small_config)
        
        # Large architecture
        large_config = ArchitectureConfig(
            num_layers=24,
            hidden_size=1024,
            num_attention_heads=16
        )
        large_arch = ProteinDiffusionArchitecture(large_config)
        
        # Large architecture should have more parameters and FLOPs
        self.assertGreater(large_arch.estimated_params, small_arch.estimated_params)
        self.assertGreater(large_arch.estimated_flops, small_arch.estimated_flops)
    
    def test_zero_cost_evaluator(self):
        """Test zero-cost evaluator."""
        evaluator = ZeroCostEvaluator()
        
        config = ArchitectureConfig()
        architecture = ProteinDiffusionArchitecture(config)
        
        # Evaluate architecture
        metrics = evaluator.evaluate_architecture(architecture)
        
        # Check metrics structure
        self.assertIsNotNone(metrics)
        self.assertHasAttr(metrics, 'grad_norm')
        self.assertHasAttr(metrics, 'snip_score')
        self.assertHasAttr(metrics, 'overall_score')
        
        # Overall score should be calculated
        overall_score = metrics.overall_score()
        self.assertIsInstance(overall_score, float)
        self.assertGreaterEqual(overall_score, 0)
    
    def test_zero_nas_search(self):
        """Test ZeroNAS search functionality."""
        # Create small search space for fast testing
        search_space = {
            'num_layers': [6, 8],
            'hidden_size': [384, 512],
            'num_attention_heads': [6, 8]
        }
        
        nas = ZeroNAS(search_space, max_architectures=10)
        
        # Run small search
        architectures = nas.search(num_iterations=4)
        
        # Should return evaluated architectures
        self.assertGreater(len(architectures), 0)
        self.assertLessEqual(len(architectures), 4)
        
        # Architectures should be sorted by score
        scores = [arch.metrics.overall_score() for arch in architectures]
        self.assertEqual(scores, sorted(scores, reverse=True))
        
        # Best architecture should be set
        self.assertIsNotNone(nas.best_architecture)
        
        # Get top architectures
        top_3 = nas.get_top_architectures(3)
        self.assertLessEqual(len(top_3), 3)
    
    def test_search_space_creation(self):
        """Test search space creation helper."""
        search_space = create_protein_diffusion_search_space()
        
        self.assertIsInstance(search_space, dict)
        
        # Check required keys
        required_keys = ['num_layers', 'hidden_size', 'num_attention_heads']
        for key in required_keys:
            self.assertIn(key, search_space)
            self.assertIsInstance(search_space[key], list)
            self.assertGreater(len(search_space[key]), 0)
    
    def test_pareto_frontier(self):
        """Test Pareto frontier calculation."""
        search_space = {
            'num_layers': [6, 12],
            'hidden_size': [384, 768]
        }
        
        nas = ZeroNAS(search_space)
        architectures = nas.search(num_iterations=4)
        
        pareto_frontier = nas.get_pareto_frontier()
        
        # Pareto frontier should be subset of all architectures
        self.assertLessEqual(len(pareto_frontier), len(architectures))
        
        # All pareto architectures should be in original list
        for arch in pareto_frontier:
            self.assertIn(arch, architectures)


class TestTPUNASIntegration(unittest.TestCase):
    """Test TPU NAS Integration."""
    
    def test_tpu_nas_config_creation(self):
        """Test TPU NAS configuration."""
        config = TPUNASConfig(
            tpu_backend=TPUBackend.JAX,
            tpu_version=TPUVersion.V6E,
            num_iterations=10
        )
        
        self.assertEqual(config.tpu_backend, TPUBackend.JAX)
        self.assertEqual(config.tpu_version, TPUVersion.V6E)
        self.assertEqual(config.num_iterations, 10)
    
    def test_tpu_nas_initialization(self):
        """Test TPU NAS initialization."""
        config = TPUNASConfig(num_iterations=5)
        tpu_nas = TPUNeuralArchitectureSearch(config)
        
        self.assertIsInstance(tpu_nas, TPUNeuralArchitectureSearch)
        self.assertEqual(tpu_nas.config, config)
        
        # Should have TPU optimizer
        self.assertIsNotNone(tpu_nas.tpu_optimizer)
        
        # Should have evaluator
        self.assertIsNotNone(tpu_nas.evaluator)
    
    def test_tpu_optimized_search_space(self):
        """Test TPU-optimized search space creation."""
        config = TPUNASConfig()
        tpu_nas = TPUNeuralArchitectureSearch(config)
        
        search_space = tpu_nas._create_tpu_optimized_search_space()
        
        self.assertIsInstance(search_space, dict)
        
        # Check TPU-optimized values
        hidden_sizes = search_space['hidden_size']
        for size in hidden_sizes:
            # Should be aligned to 128 for TPU efficiency
            self.assertEqual(size % 128, 0, f"Hidden size {size} not aligned to 128")
    
    def test_architecture_constraints(self):
        """Test TPU architecture constraints."""
        config = TPUNASConfig()
        tpu_nas = TPUNeuralArchitectureSearch(config)
        
        # Test constraint application
        config_dict = {
            'num_layers': 12,
            'hidden_size': 777,  # Not TPU-aligned
            'num_attention_heads': 13  # Doesn't divide hidden_size evenly
        }
        
        constrained_config = tpu_nas._apply_tpu_constraints(config_dict)
        
        # Should fix divisibility issue
        self.assertEqual(
            constrained_config['hidden_size'] % constrained_config['num_attention_heads'], 
            0
        )
    
    def test_tpu_score_calculation(self):
        """Test TPU score calculation."""
        config = TPUNASConfig()
        tpu_nas = TPUNeuralArchitectureSearch(config)
        
        arch_config = ArchitectureConfig(
            hidden_size=768,  # TPU-aligned
            num_attention_heads=12
        )
        architecture = ProteinDiffusionArchitecture(arch_config)
        
        # Evaluate to get base metrics
        metrics = tpu_nas.evaluator.evaluate_architecture(architecture)
        
        # Calculate TPU score
        tpu_score = tpu_nas._compute_tpu_score(architecture)
        
        self.assertIsInstance(tpu_score, float)
        self.assertGreaterEqual(tpu_score, 0)
    
    def test_small_tpu_nas_search(self):
        """Test small TPU NAS search."""
        config = TPUNASConfig(
            num_iterations=3,
            max_concurrent_evaluations=1
        )
        tpu_nas = TPUNeuralArchitectureSearch(config)
        
        # Small search space
        search_space = {
            'num_layers': [6, 8],
            'hidden_size': [384, 512],
            'num_attention_heads': [6, 8]
        }
        
        # Run search
        architectures = tpu_nas.search(search_space)
        
        # Should return results
        self.assertGreater(len(architectures), 0)
        self.assertLessEqual(len(architectures), 3)
        
        # Should have best architecture
        self.assertIsNotNone(tpu_nas.best_architecture)
    
    def test_run_tpu_nas_search_function(self):
        """Test run_tpu_nas_search utility function."""
        # Test with minimal parameters
        tpu_nas = run_tpu_nas_search(
            num_iterations=2,
            search_space={
                'num_layers': [6],
                'hidden_size': [384],
                'num_attention_heads': [6]
            }
        )
        
        self.assertIsInstance(tpu_nas, TPUNeuralArchitectureSearch)
        self.assertGreater(len(tpu_nas.search_results), 0)


class TestIntegration(unittest.TestCase):
    """Test integration between components."""
    
    def test_end_to_end_pipeline(self):
        """Test end-to-end TPU NAS pipeline."""
        # Create TPU optimizer
        tpu_optimizer = create_tpu_optimizer("jax", "v6e", 8)
        
        # Create search space
        search_space = {
            'num_layers': [6, 8],
            'hidden_size': [384, 512],
            'num_attention_heads': [6, 8]
        }
        
        # Run NAS
        config = TPUNASConfig(num_iterations=2)
        tpu_nas = TPUNeuralArchitectureSearch(config)
        architectures = tpu_nas.search(search_space)
        
        # Should complete successfully
        self.assertGreater(len(architectures), 0)
        
        # Get optimal batch size for best architecture
        best_arch = tpu_nas.best_architecture
        optimal_batch = tpu_optimizer.get_optimal_batch_size(
            best_arch.estimated_params,
            best_arch.config.max_position_embeddings
        )
        
        self.assertGreater(optimal_batch, 0)
    
    def test_export_import_functionality(self):
        """Test export and import functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Run small NAS
            config = TPUNASConfig(num_iterations=2)
            tpu_nas = TPUNeuralArchitectureSearch(config)
            
            search_space = {'num_layers': [6], 'hidden_size': [384]}
            architectures = tpu_nas.search(search_space)
            
            # Export results
            export_path = Path(temp_dir) / "tpu_nas_results.json"
            tpu_nas.export_tpu_nas_results(str(export_path))
            
            # Check file was created
            self.assertTrue(export_path.exists())
            
            # Load and validate content
            with open(export_path) as f:
                results = json.load(f)
            
            self.assertIn('tpu_config', results)
            self.assertIn('search_config', results)
            self.assertIn('hardware_info', results)
            self.assertIn('best_architecture', results)


def assertHasAttr(test_case, obj, attr):
    """Helper function to check if object has attribute."""
    test_case.assertTrue(hasattr(obj, attr), f"Object {obj} missing attribute {attr}")


if __name__ == '__main__':
    # Run specific test classes
    unittest.main(verbosity=2)