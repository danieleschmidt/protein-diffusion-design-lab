"""
Basic functionality tests for TPU-optimized protein diffusion modules.
Tests core functionality without requiring external dependencies.
"""

import unittest
import tempfile
import json
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from protein_diffusion.zero_nas import (
    ArchitectureConfig, ProteinDiffusionArchitecture, 
    ZeroCostMetrics, create_protein_diffusion_search_space
)


class TestBasicFunctionality(unittest.TestCase):
    """Test basic functionality that doesn't require external dependencies."""
    
    def test_architecture_config_creation(self):
        """Test architecture configuration creation and serialization."""
        config = ArchitectureConfig(
            num_layers=12,
            hidden_size=768,
            num_attention_heads=12,
            max_position_embeddings=512
        )
        
        # Basic properties
        self.assertEqual(config.num_layers, 12)
        self.assertEqual(config.hidden_size, 768)
        self.assertEqual(config.num_attention_heads, 12)
        self.assertEqual(config.max_position_embeddings, 512)
        
        # Serialization
        config_dict = config.to_dict()
        self.assertIsInstance(config_dict, dict)
        self.assertEqual(config_dict['num_layers'], 12)
        self.assertEqual(config_dict['hidden_size'], 768)
        
        # All expected keys present
        expected_keys = [
            'num_layers', 'hidden_size', 'num_attention_heads',
            'intermediate_size', 'max_position_embeddings', 'conv_layers',
            'kernel_sizes', 'time_embedding_dim', 'condition_embedding_dim',
            'dropout_rate', 'layer_norm_eps', 'activation_function'
        ]
        for key in expected_keys:
            self.assertIn(key, config_dict)
    
    def test_architecture_creation_and_estimation(self):
        """Test protein diffusion architecture creation and parameter estimation."""
        config = ArchitectureConfig(
            num_layers=6,
            hidden_size=384,
            num_attention_heads=6
        )
        
        architecture = ProteinDiffusionArchitecture(config)
        
        # Basic properties
        self.assertEqual(architecture.config, config)
        self.assertIsInstance(architecture.arch_id, str)
        self.assertGreater(len(architecture.arch_id), 0)
        
        # Parameter estimation should be positive
        self.assertGreater(architecture.estimated_params, 0)
        self.assertGreater(architecture.estimated_flops, 0)
        
        # Architecture ID should be deterministic
        architecture2 = ProteinDiffusionArchitecture(config)
        self.assertEqual(architecture.arch_id, architecture2.arch_id)
    
    def test_parameter_scaling(self):
        """Test that parameter estimation scales correctly with architecture size."""
        # Small architecture
        small_config = ArchitectureConfig(
            num_layers=6,
            hidden_size=384,
            num_attention_heads=6,
            intermediate_size=1536
        )
        small_arch = ProteinDiffusionArchitecture(small_config)
        
        # Large architecture
        large_config = ArchitectureConfig(
            num_layers=24,
            hidden_size=1024,
            num_attention_heads=16,
            intermediate_size=4096
        )
        large_arch = ProteinDiffusionArchitecture(large_config)
        
        # Large architecture should have more parameters and FLOPs
        self.assertGreater(large_arch.estimated_params, small_arch.estimated_params)
        self.assertGreater(large_arch.estimated_flops, small_arch.estimated_flops)
        
        # Reasonable parameter counts (should be in millions/billions range)
        self.assertGreater(small_arch.estimated_params, 1_000_000)  # > 1M
        self.assertLess(small_arch.estimated_params, 1_000_000_000)  # < 1B
        
        self.assertGreater(large_arch.estimated_params, 100_000_000)  # > 100M
        self.assertLess(large_arch.estimated_params, 10_000_000_000)  # < 10B
    
    def test_zero_cost_metrics(self):
        """Test zero-cost metrics structure and calculation."""
        metrics = ZeroCostMetrics(
            grad_norm=1.5,
            snip_score=0.8,
            synflow_score=1000.0,
            sequence_complexity=0.7,
            structure_consistency=0.9,
            binding_potential=0.6
        )
        
        # Test individual metrics
        self.assertEqual(metrics.grad_norm, 1.5)
        self.assertEqual(metrics.snip_score, 0.8)
        self.assertEqual(metrics.synflow_score, 1000.0)
        
        # Test overall score calculation
        overall_score = metrics.overall_score()
        self.assertIsInstance(overall_score, float)
        self.assertGreater(overall_score, 0)
        
        # Overall score should be weighted combination
        # Manually calculate expected score
        expected_score = (
            1.5 * 0.15 +  # grad_norm
            0.8 * 0.15 +  # snip_score
            1000.0 * 0.1 +  # synflow_score
            0.7 * 0.05 +  # sequence_complexity
            0.9 * 0.03 +  # structure_consistency
            0.6 * 0.02   # binding_potential
        )
        
        self.assertAlmostEqual(overall_score, expected_score, places=4)
    
    def test_search_space_creation(self):
        """Test search space creation and validation."""
        search_space = create_protein_diffusion_search_space()
        
        self.assertIsInstance(search_space, dict)
        
        # Check all required parameters are present
        required_params = [
            'num_layers', 'hidden_size', 'num_attention_heads',
            'intermediate_size', 'max_position_embeddings',
            'conv_layers', 'kernel_sizes', 'time_embedding_dim',
            'condition_embedding_dim', 'dropout_rate', 'activation_function'
        ]
        
        for param in required_params:
            self.assertIn(param, search_space)
            self.assertIsInstance(search_space[param], list)
            self.assertGreater(len(search_space[param]), 0)
        
        # Validate parameter ranges
        self.assertGreater(min(search_space['num_layers']), 0)
        self.assertLess(max(search_space['num_layers']), 50)
        
        self.assertGreater(min(search_space['hidden_size']), 0)
        self.assertLess(max(search_space['hidden_size']), 5000)
        
        # Check dropout rates are reasonable
        for rate in search_space['dropout_rate']:
            self.assertGreaterEqual(rate, 0.0)
            self.assertLessEqual(rate, 1.0)
    
    def test_architecture_id_consistency(self):
        """Test that architecture IDs are consistent and unique."""
        config1 = ArchitectureConfig(num_layers=12, hidden_size=768)
        config2 = ArchitectureConfig(num_layers=12, hidden_size=768)  # Same
        config3 = ArchitectureConfig(num_layers=12, hidden_size=512)  # Different
        
        arch1 = ProteinDiffusionArchitecture(config1)
        arch2 = ProteinDiffusionArchitecture(config2)
        arch3 = ProteinDiffusionArchitecture(config3)
        
        # Same configs should have same ID
        self.assertEqual(arch1.arch_id, arch2.arch_id)
        
        # Different configs should have different IDs
        self.assertNotEqual(arch1.arch_id, arch3.arch_id)
        
        # IDs should be reasonable length (8 chars in our implementation)
        self.assertEqual(len(arch1.arch_id), 8)
        self.assertTrue(arch1.arch_id.isalnum())
    
    def test_conv_layer_configurations(self):
        """Test convolutional layer configurations."""
        config = ArchitectureConfig(
            conv_layers=[64, 128, 256],
            kernel_sizes=[3, 5, 7]
        )
        
        architecture = ProteinDiffusionArchitecture(config)
        
        # Should handle conv layers in parameter estimation
        self.assertGreater(architecture.estimated_params, 0)
        self.assertGreater(architecture.estimated_flops, 0)
        
        # Test with different conv configurations
        config2 = ArchitectureConfig(
            conv_layers=[32, 64],
            kernel_sizes=[3, 3]
        )
        
        architecture2 = ProteinDiffusionArchitecture(config2)
        
        # Different conv configs should give different parameter counts
        self.assertNotEqual(architecture.estimated_params, architecture2.estimated_params)
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Minimal architecture
        minimal_config = ArchitectureConfig(
            num_layers=1,
            hidden_size=128,
            num_attention_heads=1,
            intermediate_size=256
        )
        
        minimal_arch = ProteinDiffusionArchitecture(minimal_config)
        self.assertGreater(minimal_arch.estimated_params, 0)
        self.assertGreater(minimal_arch.estimated_flops, 0)
        
        # Large architecture
        large_config = ArchitectureConfig(
            num_layers=48,
            hidden_size=2048,
            num_attention_heads=32,
            intermediate_size=8192
        )
        
        large_arch = ProteinDiffusionArchitecture(large_config)
        self.assertGreater(large_arch.estimated_params, minimal_arch.estimated_params)
        
        # Zero dropout
        zero_dropout_config = ArchitectureConfig(dropout_rate=0.0)
        zero_dropout_arch = ProteinDiffusionArchitecture(zero_dropout_config)
        self.assertEqual(zero_dropout_arch.config.dropout_rate, 0.0)
        
        # Maximum dropout
        max_dropout_config = ArchitectureConfig(dropout_rate=0.5)
        max_dropout_arch = ProteinDiffusionArchitecture(max_dropout_config)
        self.assertEqual(max_dropout_arch.config.dropout_rate, 0.5)
    
    def test_json_serialization(self):
        """Test JSON serialization of architecture configs."""
        config = ArchitectureConfig(
            num_layers=12,
            hidden_size=768,
            num_attention_heads=12
        )
        
        # Convert to dict and serialize to JSON
        config_dict = config.to_dict()
        json_str = json.dumps(config_dict)
        
        # Deserialize and verify
        loaded_dict = json.loads(json_str)
        
        # All values should be preserved
        self.assertEqual(loaded_dict['num_layers'], 12)
        self.assertEqual(loaded_dict['hidden_size'], 768)
        self.assertEqual(loaded_dict['num_attention_heads'], 12)
        
        # Reconstruct config
        new_config = ArchitectureConfig(**loaded_dict)
        self.assertEqual(new_config.num_layers, config.num_layers)
        self.assertEqual(new_config.hidden_size, config.hidden_size)
    
    def test_metrics_properties(self):
        """Test properties of zero-cost metrics."""
        # Test default initialization
        default_metrics = ZeroCostMetrics()
        self.assertEqual(default_metrics.grad_norm, 0.0)
        self.assertEqual(default_metrics.snip_score, 0.0)
        self.assertEqual(default_metrics.overall_score(), 0.0)
        
        # Test with positive values
        positive_metrics = ZeroCostMetrics(
            grad_norm=2.0,
            fisher_information=1.5,
            snip_score=1.0,
            grasp_score=0.8,
            synflow_score=500.0
        )
        
        overall_score = positive_metrics.overall_score()
        self.assertGreater(overall_score, 0)
        
        # Score should increase with better individual metrics
        better_metrics = ZeroCostMetrics(
            grad_norm=3.0,  # Higher
            fisher_information=2.0,  # Higher
            snip_score=1.5,  # Higher
            grasp_score=1.0,  # Higher
            synflow_score=1000.0  # Higher
        )
        
        self.assertGreater(better_metrics.overall_score(), positive_metrics.overall_score())


class TestConfigurationValidation(unittest.TestCase):
    """Test configuration validation and constraints."""
    
    def test_attention_head_divisibility(self):
        """Test attention head and hidden size relationships."""
        # Valid configuration (768 divisible by 12)
        valid_config = ArchitectureConfig(
            hidden_size=768,
            num_attention_heads=12
        )
        
        architecture = ProteinDiffusionArchitecture(valid_config)
        self.assertEqual(768 % 12, 0)  # Should be divisible
        
        # Test head size calculation
        head_size = valid_config.hidden_size // valid_config.num_attention_heads
        self.assertEqual(head_size, 64)
    
    def test_intermediate_size_relationships(self):
        """Test intermediate size relationships with hidden size."""
        config = ArchitectureConfig(
            hidden_size=768,
            intermediate_size=3072  # Typically 4x hidden size
        )
        
        architecture = ProteinDiffusionArchitecture(config)
        
        # Intermediate size should be reasonable multiple of hidden size
        ratio = config.intermediate_size / config.hidden_size
        self.assertGreater(ratio, 2.0)  # At least 2x
        self.assertLess(ratio, 8.0)     # No more than 8x
    
    def test_position_embedding_constraints(self):
        """Test position embedding constraints."""
        config = ArchitectureConfig(
            max_position_embeddings=1024
        )
        
        architecture = ProteinDiffusionArchitecture(config)
        
        # Position embeddings should be power of 2 or common sizes
        common_sizes = [128, 256, 512, 768, 1024, 1536, 2048]
        self.assertIn(config.max_position_embeddings, common_sizes)


if __name__ == '__main__':
    unittest.main(verbosity=2)