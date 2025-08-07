"""
Unit tests for optimization module.
"""

import pytest
import threading
import time
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.protein_diffusion.optimization import (
    LRUCache,
    ModelCache,
    BatchProcessor,
    ModelOptimizer,
    MemoryOptimizer,
    AsyncProcessor,
    PerformanceManager,
    cache_sequence_generation,
    cache_structure_prediction,
    get_performance_manager
)


class TestLRUCache:
    """Test LRUCache implementation."""
    
    def test_initialization(self):
        """Test LRU cache initialization."""
        cache = LRUCache(max_size=10)
        assert cache.max_size == 10
        assert cache.size() == 0
        assert cache.hit_rate() == 0.0
    
    def test_put_and_get(self):
        """Test putting and getting values."""
        cache = LRUCache(max_size=10)
        
        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"
        assert cache.hits == 1
        assert cache.misses == 0
    
    def test_cache_miss(self):
        """Test cache miss."""
        cache = LRUCache(max_size=10)
        
        result = cache.get("nonexistent")
        assert result is None
        assert cache.hits == 0
        assert cache.misses == 1
    
    def test_lru_eviction(self):
        """Test LRU eviction."""
        cache = LRUCache(max_size=2)
        
        # Fill cache
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        
        # Access key1 to make it most recently used
        cache.get("key1")
        
        # Add new item, should evict key2
        cache.put("key3", "value3")
        
        assert cache.get("key1") == "value1"  # Should still be there
        assert cache.get("key2") is None       # Should be evicted
        assert cache.get("key3") == "value3"  # Should be there
    
    def test_update_existing_key(self):
        """Test updating existing key."""
        cache = LRUCache(max_size=10)
        
        cache.put("key1", "value1")
        cache.put("key1", "value2")  # Update
        
        assert cache.get("key1") == "value2"
        assert cache.size() == 1
    
    def test_clear(self):
        """Test cache clear."""
        cache = LRUCache(max_size=10)
        
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        
        cache.clear()
        
        assert cache.size() == 0
        assert cache.hits == 0
        assert cache.misses == 0
    
    def test_hit_rate(self):
        """Test hit rate calculation."""
        cache = LRUCache(max_size=10)
        
        cache.put("key1", "value1")
        
        # 2 hits, 1 miss
        cache.get("key1")  # Hit
        cache.get("key1")  # Hit
        cache.get("key2")  # Miss
        
        assert cache.hit_rate() == 2.0 / 3.0
    
    def test_stats(self):
        """Test cache statistics."""
        cache = LRUCache(max_size=10)
        
        cache.put("key1", "value1")
        cache.get("key1")  # Hit
        cache.get("key2")  # Miss
        
        stats = cache.stats()
        assert stats["size"] == 1
        assert stats["max_size"] == 10
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert abs(stats["hit_rate"] - 0.5) < 1e-6


class TestModelCache:
    """Test ModelCache."""
    
    @pytest.fixture
    def model_cache(self):
        """Create a model cache."""
        lru_cache = LRUCache(max_size=10)
        return ModelCache(lru_cache)
    
    def test_initialization(self, model_cache):
        """Test ModelCache initialization."""
        assert model_cache.enable_sequence_cache == True
        assert model_cache.enable_structure_cache == True
    
    def test_cache_sequence_generation_decorator(self, model_cache):
        """Test sequence generation caching decorator."""
        call_count = 0
        
        @model_cache.cache_sequence_generation
        def generate_sequence(motif, length):
            nonlocal call_count
            call_count += 1
            return f"sequence_for_{motif}_{length}"
        
        # First call should execute function
        result1 = generate_sequence("HELIX", 100)
        assert result1 == "sequence_for_HELIX_100"
        assert call_count == 1
        
        # Second call with same args should use cache
        result2 = generate_sequence("HELIX", 100)
        assert result2 == "sequence_for_HELIX_100"
        assert call_count == 1  # Function not called again
        
        # Call with different args should execute function
        result3 = generate_sequence("SHEET", 100)
        assert result3 == "sequence_for_SHEET_100"
        assert call_count == 2
    
    def test_cache_structure_prediction_decorator(self, model_cache):
        """Test structure prediction caching decorator."""
        call_count = 0
        
        @model_cache.cache_structure_prediction
        def predict_structure(sequence):
            nonlocal call_count
            call_count += 1
            return f"structure_for_{sequence}"
        
        # First call should execute function
        result1 = predict_structure("MKLL")
        assert result1 == "structure_for_MKLL"
        assert call_count == 1
        
        # Second call should use cache
        result2 = predict_structure("MKLL")
        assert result2 == "structure_for_MKLL"
        assert call_count == 1
    
    def test_disabled_cache(self):
        """Test cache with caching disabled."""
        lru_cache = LRUCache(max_size=10)
        model_cache = ModelCache(lru_cache, enable_sequence_cache=False)
        
        call_count = 0
        
        @model_cache.cache_sequence_generation
        def generate_sequence():
            nonlocal call_count
            call_count += 1
            return "result"
        
        # Both calls should execute function
        generate_sequence()
        generate_sequence()
        
        assert call_count == 2


class TestBatchProcessor:
    """Test BatchProcessor."""
    
    def test_initialization(self):
        """Test BatchProcessor initialization."""
        processor = BatchProcessor(max_batch_size=64, optimal_batch_size=32)
        assert processor.max_batch_size == 64
        assert processor.optimal_batch_size == 32
    
    def test_optimize_batches_default(self):
        """Test batch optimization with default size."""
        processor = BatchProcessor(optimal_batch_size=4)
        items = list(range(10))
        
        batches = processor.optimize_batches(items)
        
        # Should create 3 batches: [0,1,2,3], [4,5,6,7], [8,9]
        assert len(batches) == 3
        assert batches[0] == [0, 1, 2, 3]
        assert batches[1] == [4, 5, 6, 7]
        assert batches[2] == [8, 9]
    
    def test_optimize_batches_custom_size(self):
        """Test batch optimization with custom size."""
        processor = BatchProcessor()
        items = list(range(10))
        
        batches = processor.optimize_batches(items, batch_size=3)
        
        # Should create 4 batches: [0,1,2], [3,4,5], [6,7,8], [9]
        assert len(batches) == 4
        assert batches[-1] == [9]
    
    def test_dynamic_batch_size(self):
        """Test dynamic batch size calculation."""
        processor = BatchProcessor(max_batch_size=100)
        
        # Test with small items
        small_items = [50] * 10  # 10 items of size 50
        batch_size = processor.dynamic_batch_size(small_items, memory_limit=1.0)
        assert batch_size > 0
        assert batch_size <= processor.max_batch_size
        
        # Test with large items
        large_items = [1000] * 10  # 10 items of size 1000
        large_batch_size = processor.dynamic_batch_size(large_items, memory_limit=1.0)
        assert large_batch_size > 0
        assert large_batch_size <= batch_size  # Should be smaller than small items batch
    
    def test_dynamic_batch_size_empty_items(self):
        """Test dynamic batch size with empty items."""
        processor = BatchProcessor()
        batch_size = processor.dynamic_batch_size([], memory_limit=1.0)
        assert batch_size == processor.optimal_batch_size


class TestModelOptimizer:
    """Test ModelOptimizer."""
    
    @pytest.fixture
    def optimizer(self):
        """Create a model optimizer."""
        return ModelOptimizer()
    
    def test_initialization(self, optimizer):
        """Test ModelOptimizer initialization."""
        assert optimizer.optimizations_applied == []
    
    @pytest.mark.skipif(not hasattr(pytest, 'importorskip'), reason="Need PyTorch for this test")
    def test_optimize_for_inference_basic(self, optimizer):
        """Test basic model optimization."""
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            pytest.skip("PyTorch not available")
        
        # Create a simple mock model
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 1)
        )
        
        optimized = optimizer.optimize_for_inference(
            model, 
            use_jit=False,  # Skip JIT for this test
            use_half_precision=False,
            optimize_for_mobile=False
        )
        
        assert optimized is not None
        assert "eval_mode" in optimizer.optimizations_applied
    
    def test_optimize_without_torch(self, optimizer):
        """Test optimization when PyTorch is not available."""
        mock_model = Mock()
        
        with patch('src.protein_diffusion.optimization.TORCH_AVAILABLE', False):
            result = optimizer.optimize_for_inference(mock_model)
            assert result == mock_model  # Should return original model


class TestMemoryOptimizer:
    """Test MemoryOptimizer."""
    
    def test_enable_memory_efficient_attention(self):
        """Test enabling memory efficient attention."""
        # This test just verifies the function doesn't crash
        result = MemoryOptimizer.enable_memory_efficient_attention()
        assert isinstance(result, bool)
    
    def test_optimize_cuda_settings(self):
        """Test CUDA settings optimization."""
        # This test just verifies the function doesn't crash
        result = MemoryOptimizer.optimize_cuda_settings()
        assert isinstance(result, bool)
    
    def test_get_memory_stats(self):
        """Test memory statistics retrieval."""
        stats = MemoryOptimizer.get_memory_stats()
        assert isinstance(stats, dict)


class TestAsyncProcessor:
    """Test AsyncProcessor."""
    
    def test_async_processor_context_manager(self):
        """Test AsyncProcessor as context manager."""
        with AsyncProcessor(max_workers=2) as processor:
            assert processor._executor is not None
        
        # Should be cleaned up after exiting context
        assert processor._executor._shutdown
    
    def test_submit_batch_processing(self):
        """Test batch processing with AsyncProcessor."""
        def simple_function(batch):
            return [x * 2 for x in batch]
        
        batches = [[1, 2], [3, 4], [5, 6]]
        
        with AsyncProcessor(max_workers=2) as processor:
            results = processor.submit_batch(simple_function, batches)
        
        expected = [[2, 4], [6, 8], [10, 12]]
        assert results == expected
    
    def test_submit_batch_with_error(self):
        """Test batch processing with errors."""
        def failing_function(batch):
            if 5 in batch:
                raise ValueError("Test error")
            return [x * 2 for x in batch]
        
        batches = [[1, 2], [5, 6], [7, 8]]  # Second batch will fail
        
        with AsyncProcessor(max_workers=2) as processor:
            results = processor.submit_batch(failing_function, batches)
        
        assert results[0] == [2, 4]  # First batch succeeds
        assert results[1] is None     # Second batch fails
        assert results[2] == [14, 16]  # Third batch succeeds
    
    def test_async_processor_without_context_manager(self):
        """Test AsyncProcessor usage without context manager."""
        processor = AsyncProcessor(max_workers=2)
        
        with pytest.raises(RuntimeError, match="not properly initialized"):
            processor.submit_batch(lambda x: x, [[1, 2]])


class TestPerformanceManager:
    """Test PerformanceManager."""
    
    def test_initialization_lru(self):
        """Test PerformanceManager initialization with LRU cache."""
        manager = PerformanceManager(cache_backend="lru", cache_size=100)
        
        assert hasattr(manager, 'cache')
        assert hasattr(manager, 'model_cache')
        assert hasattr(manager, 'batch_processor')
        assert hasattr(manager, 'model_optimizer')
    
    @patch('src.protein_diffusion.optimization.REDIS_AVAILABLE', False)
    def test_initialization_redis_unavailable(self):
        """Test PerformanceManager with Redis unavailable."""
        # Should fall back to LRU cache
        manager = PerformanceManager(cache_backend="redis")
        assert hasattr(manager, 'cache')
    
    def test_get_cache_decorator(self):
        """Test getting cache decorators."""
        manager = PerformanceManager()
        
        seq_decorator = manager.get_cache_decorator("sequence")
        assert callable(seq_decorator)
        
        struct_decorator = manager.get_cache_decorator("structure")
        assert callable(struct_decorator)
        
        with pytest.raises(ValueError):
            manager.get_cache_decorator("invalid")
    
    def test_get_performance_stats(self):
        """Test getting performance statistics."""
        manager = PerformanceManager()
        stats = manager.get_performance_stats()
        
        assert "cache_stats" in stats
        assert "memory_stats" in stats
        assert "optimizations_applied" in stats
        assert "system_info" in stats
    
    def test_clear_caches(self):
        """Test clearing caches."""
        manager = PerformanceManager()
        
        # Add something to cache
        manager.cache.put("test", "value")
        assert manager.cache.get("test") == "value"
        
        # Clear caches
        manager.clear_caches()
        assert manager.cache.get("test") is None


class TestConvenienceDecorators:
    """Test convenience decorator functions."""
    
    def test_cache_sequence_generation_decorator(self):
        """Test convenience sequence generation decorator."""
        call_count = 0
        
        @cache_sequence_generation
        def generate_sequence(motif):
            nonlocal call_count
            call_count += 1
            return f"sequence_{motif}"
        
        # First call
        result1 = generate_sequence("HELIX")
        assert result1 == "sequence_HELIX"
        assert call_count == 1
        
        # Second call should use cache
        result2 = generate_sequence("HELIX")
        assert result2 == "sequence_HELIX"
        assert call_count == 1
    
    def test_cache_structure_prediction_decorator(self):
        """Test convenience structure prediction decorator."""
        call_count = 0
        
        @cache_structure_prediction
        def predict_structure(sequence):
            nonlocal call_count
            call_count += 1
            return f"structure_{sequence}"
        
        # First call
        result1 = predict_structure("MKLL")
        assert result1 == "structure_MKLL"
        assert call_count == 1
        
        # Second call should use cache
        result2 = predict_structure("MKLL")
        assert result2 == "structure_MKLL"
        assert call_count == 1


class TestThreadSafety:
    """Test thread safety of optimization components."""
    
    def test_lru_cache_thread_safety(self):
        """Test LRU cache thread safety."""
        cache = LRUCache(max_size=100)
        errors = []
        
        def worker(worker_id):
            try:
                for i in range(50):
                    key = f"worker_{worker_id}_item_{i}"
                    value = f"value_{worker_id}_{i}"
                    cache.put(key, value)
                    
                    retrieved = cache.get(key)
                    if retrieved != value:
                        errors.append(f"Value mismatch: expected {value}, got {retrieved}")
                    
                    # Also try to get some random keys to create contention
                    cache.get(f"worker_{(worker_id + 1) % 3}_item_{i % 10}")
                    
            except Exception as e:
                errors.append(f"Worker {worker_id} error: {e}")
        
        # Start multiple threads
        threads = []
        for i in range(3):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()
        
        # Wait for completion
        for t in threads:
            t.join()
        
        # Check for errors
        if errors:
            pytest.fail(f"Thread safety errors: {errors}")
        
        # Verify cache is in a consistent state
        stats = cache.stats()
        assert stats["size"] <= cache.max_size


class TestGlobalInstances:
    """Test global instance management."""
    
    def test_get_performance_manager_singleton(self):
        """Test that get_performance_manager returns singleton."""
        # Reset global instance
        import src.protein_diffusion.optimization as opt_module
        opt_module._performance_manager = None
        
        manager1 = get_performance_manager()
        manager2 = get_performance_manager()
        
        # Should be the same instance
        assert manager1 is manager2
    
    def test_get_performance_manager_with_params(self):
        """Test get_performance_manager with custom parameters."""
        # Reset global instance
        import src.protein_diffusion.optimization as opt_module
        opt_module._performance_manager = None
        
        manager = get_performance_manager(cache_backend="lru", cache_size=123)
        assert manager.cache.max_size == 123


class TestIntegration:
    """Integration tests for optimization components."""
    
    def test_end_to_end_caching(self):
        """Test end-to-end caching workflow."""
        manager = PerformanceManager(cache_size=10)
        
        # Create a function that tracks calls
        call_count = 0
        
        @manager.get_cache_decorator("sequence")
        def expensive_function(param1, param2=None):
            nonlocal call_count
            call_count += 1
            time.sleep(0.001)  # Simulate work
            return f"result_{param1}_{param2}"
        
        # First call should execute function
        result1 = expensive_function("A", param2="B")
        assert result1 == "result_A_B"
        assert call_count == 1
        
        # Second call should use cache
        result2 = expensive_function("A", param2="B")
        assert result2 == "result_A_B"
        assert call_count == 1
        
        # Different parameters should execute function again
        result3 = expensive_function("C", param2="D")
        assert result3 == "result_C_D"
        assert call_count == 2
        
        # Verify cache stats
        stats = manager.get_performance_stats()
        assert stats["cache_stats"]["hits"] >= 1
    
    def test_batch_processing_with_caching(self):
        """Test combining batch processing with caching."""
        manager = PerformanceManager()
        processor = BatchProcessor(optimal_batch_size=2)
        
        call_count = 0
        
        @manager.get_cache_decorator("sequence")
        def process_item(item):
            nonlocal call_count
            call_count += 1
            return item * 2
        
        items = [1, 2, 3, 4, 1, 2]  # Some duplicates for cache hits
        
        # Process items in batches
        batches = processor.optimize_batches(items, batch_size=2)
        
        results = []
        for batch in batches:
            batch_results = [process_item(item) for item in batch]
            results.extend(batch_results)
        
        expected = [2, 4, 6, 8, 2, 4]  # Last two should come from cache
        assert results == expected
        assert call_count == 4  # Should only call function 4 times due to caching