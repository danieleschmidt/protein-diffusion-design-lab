"""
Advanced Multi-Level Caching System for Protein Diffusion Design Lab.

This module provides sophisticated caching capabilities including distributed caching,
intelligent cache management, and performance optimization for production environments.
"""

import logging
import time
import json
import hashlib
import threading
import pickle
import zlib
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from collections import OrderedDict, defaultdict
import queue
from pathlib import Path
import asyncio

logger = logging.getLogger(__name__)


class CacheLevel(Enum):
    """Cache level enumeration."""
    L1_MEMORY = "l1_memory"
    L2_SSD = "l2_ssd"
    L3_NETWORK = "l3_network"
    L4_COLD = "l4_cold"


class CacheStrategy(Enum):
    """Cache eviction strategy."""
    LRU = "lru"
    LFU = "lfu"
    FIFO = "fifo"
    TTL = "ttl"
    ADAPTIVE = "adaptive"


class CompressionType(Enum):
    """Compression algorithms."""
    NONE = "none"
    ZLIB = "zlib"
    GZIP = "gzip"
    BROTLI = "brotli"


@dataclass
class CacheConfig:
    """Cache configuration."""
    # Memory cache settings
    l1_max_size: int = 1000  # Maximum items in L1 cache
    l1_max_memory_mb: int = 512  # Maximum memory usage in MB
    
    # SSD cache settings
    l2_max_size: int = 10000  # Maximum items in L2 cache
    l2_max_disk_gb: float = 10.0  # Maximum disk usage in GB
    l2_cache_dir: str = "./cache/l2"
    
    # Network cache settings
    l3_enabled: bool = False
    l3_redis_url: str = "redis://localhost:6379"
    l3_max_size: int = 100000
    
    # Cold storage settings
    l4_enabled: bool = False
    l4_cache_dir: str = "./cache/l4"
    l4_max_size_gb: float = 100.0
    
    # Performance settings
    compression: CompressionType = CompressionType.ZLIB
    compression_threshold: int = 1024  # Bytes
    
    # Eviction settings
    eviction_strategy: CacheStrategy = CacheStrategy.ADAPTIVE
    default_ttl: float = 3600.0  # 1 hour
    max_ttl: float = 86400.0  # 24 hours
    
    # Async settings
    enable_async: bool = True
    background_cleanup: bool = True
    cleanup_interval: float = 300.0  # 5 minutes
    
    # Statistics
    enable_statistics: bool = True
    stats_interval: float = 60.0  # 1 minute


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    timestamp: float
    ttl: float
    access_count: int = 0
    last_access: float = 0.0
    size_bytes: int = 0
    compression_used: CompressionType = CompressionType.NONE
    
    def __post_init__(self):
        if self.last_access == 0.0:
            self.last_access = self.timestamp
    
    def is_expired(self) -> bool:
        """Check if entry is expired."""
        return time.time() > (self.timestamp + self.ttl)
    
    def access(self):
        """Record access to this entry."""
        self.access_count += 1
        self.last_access = time.time()


@dataclass
class CacheStats:
    """Cache statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size: int = 0
    memory_usage_mb: float = 0.0
    hit_rate: float = 0.0
    avg_access_time_ms: float = 0.0
    
    # Level-specific stats
    l1_hits: int = 0
    l2_hits: int = 0
    l3_hits: int = 0
    l4_hits: int = 0
    
    # Performance metrics
    compression_ratio: float = 0.0
    serialization_time_ms: float = 0.0
    
    def calculate_hit_rate(self):
        """Calculate hit rate."""
        total_requests = self.hits + self.misses
        self.hit_rate = self.hits / total_requests if total_requests > 0 else 0.0


class CompressionManager:
    """Manages data compression and decompression."""
    
    @staticmethod
    def compress(data: bytes, compression_type: CompressionType) -> bytes:
        """Compress data using specified algorithm."""
        if compression_type == CompressionType.NONE:
            return data
        elif compression_type == CompressionType.ZLIB:
            return zlib.compress(data)
        elif compression_type == CompressionType.GZIP:
            import gzip
            return gzip.compress(data)
        elif compression_type == CompressionType.BROTLI:
            try:
                import brotli
                return brotli.compress(data)
            except ImportError:
                logger.warning("Brotli not available, falling back to zlib")
                return zlib.compress(data)
        else:
            return data
    
    @staticmethod
    def decompress(data: bytes, compression_type: CompressionType) -> bytes:
        """Decompress data using specified algorithm."""
        if compression_type == CompressionType.NONE:
            return data
        elif compression_type == CompressionType.ZLIB:
            return zlib.decompress(data)
        elif compression_type == CompressionType.GZIP:
            import gzip
            return gzip.decompress(data)
        elif compression_type == CompressionType.BROTLI:
            try:
                import brotli
                return brotli.decompress(data)
            except ImportError:
                logger.warning("Brotli not available, using zlib")
                return zlib.decompress(data)
        else:
            return data
    
    @staticmethod
    def should_compress(data: bytes, threshold: int) -> bool:
        """Determine if data should be compressed."""
        return len(data) >= threshold


class L1MemoryCache:
    """High-speed in-memory cache (L1)."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.lock = threading.RLock()
        self.access_times: Dict[str, List[float]] = defaultdict(list)
        
        logger.info(f"L1 Memory Cache initialized (max_size: {config.l1_max_size})")
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from L1 cache."""
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                
                if entry.is_expired():
                    del self.cache[key]
                    return None
                
                entry.access()
                # Move to end for LRU
                self.cache.move_to_end(key)
                
                return entry.value
            
            return None
    
    def put(self, key: str, value: Any, ttl: float = None) -> bool:
        """Put item in L1 cache."""
        with self.lock:
            if ttl is None:
                ttl = self.config.default_ttl
            
            # Calculate size
            size_bytes = self._estimate_size(value)
            
            # Check memory limits
            if not self._can_fit(size_bytes):
                self._evict_to_fit(size_bytes)
            
            # Create entry
            entry = CacheEntry(
                key=key,
                value=value,
                timestamp=time.time(),
                ttl=ttl,
                size_bytes=size_bytes
            )
            
            self.cache[key] = entry
            
            # Enforce size limit
            while len(self.cache) > self.config.l1_max_size:
                self._evict_one()
            
            return True
    
    def delete(self, key: str) -> bool:
        """Delete item from L1 cache."""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                return True
            return False
    
    def clear(self):
        """Clear all items from L1 cache."""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get L1 cache statistics."""
        with self.lock:
            total_size = sum(entry.size_bytes for entry in self.cache.values())
            
            return {
                "size": len(self.cache),
                "memory_usage_mb": total_size / (1024 * 1024),
                "max_size": self.config.l1_max_size,
                "max_memory_mb": self.config.l1_max_memory_mb
            }
    
    def _estimate_size(self, value: Any) -> int:
        """Estimate memory size of value."""
        try:
            return len(pickle.dumps(value))
        except:
            return 1024  # Default estimate
    
    def _can_fit(self, size_bytes: int) -> bool:
        """Check if new item can fit in cache."""
        current_memory = sum(entry.size_bytes for entry in self.cache.values())
        max_memory = self.config.l1_max_memory_mb * 1024 * 1024
        
        return (current_memory + size_bytes) <= max_memory
    
    def _evict_to_fit(self, needed_bytes: int):
        """Evict items to make space."""
        current_memory = sum(entry.size_bytes for entry in self.cache.values())
        max_memory = self.config.l1_max_memory_mb * 1024 * 1024
        
        while (current_memory + needed_bytes) > max_memory and self.cache:
            self._evict_one()
            current_memory = sum(entry.size_bytes for entry in self.cache.values())
    
    def _evict_one(self):
        """Evict one item based on strategy."""
        if not self.cache:
            return
        
        if self.config.eviction_strategy == CacheStrategy.LRU:
            # Remove least recently used (first in OrderedDict)
            key, _ = self.cache.popitem(last=False)
        elif self.config.eviction_strategy == CacheStrategy.LFU:
            # Remove least frequently used
            key = min(self.cache.keys(), key=lambda k: self.cache[k].access_count)
            del self.cache[key]
        elif self.config.eviction_strategy == CacheStrategy.FIFO:
            # Remove oldest entry
            key, _ = self.cache.popitem(last=False)
        elif self.config.eviction_strategy == CacheStrategy.TTL:
            # Remove expired entries first
            expired_keys = [k for k, v in self.cache.items() if v.is_expired()]
            if expired_keys:
                del self.cache[expired_keys[0]]
            else:
                key, _ = self.cache.popitem(last=False)
        else:  # ADAPTIVE
            # Use adaptive strategy based on access patterns
            self._adaptive_evict()
    
    def _adaptive_evict(self):
        """Adaptive eviction based on access patterns."""
        if not self.cache:
            return
        
        # Score entries based on multiple factors
        scores = {}
        current_time = time.time()
        
        for key, entry in self.cache.items():
            # Factor in recency, frequency, and TTL
            recency_score = 1.0 / max(1, current_time - entry.last_access)
            frequency_score = entry.access_count
            ttl_score = 1.0 / max(1, (entry.timestamp + entry.ttl) - current_time)
            
            # Combined score (lower is better for eviction)
            scores[key] = recency_score * frequency_score * ttl_score
        
        # Remove entry with lowest score
        victim_key = min(scores.keys(), key=lambda k: scores[k])
        del self.cache[victim_key]


class L2SSDCache:
    """SSD-based persistent cache (L2)."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.cache_dir = Path(config.l2_cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.index: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.RLock()
        
        # Load existing index
        self._load_index()
        
        logger.info(f"L2 SSD Cache initialized (dir: {self.cache_dir})")
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from L2 cache."""
        with self.lock:
            if key not in self.index:
                return None
            
            entry_info = self.index[key]
            
            # Check TTL
            if time.time() > (entry_info['timestamp'] + entry_info['ttl']):
                self._remove_entry(key)
                return None
            
            # Load from disk
            try:
                file_path = self.cache_dir / f"{key}.cache"
                with open(file_path, 'rb') as f:
                    data = f.read()
                
                # Decompress if needed
                if entry_info['compressed']:
                    data = CompressionManager.decompress(
                        data, CompressionType(entry_info['compression_type'])
                    )
                
                # Deserialize
                value = pickle.loads(data)
                
                # Update access info
                entry_info['access_count'] += 1
                entry_info['last_access'] = time.time()
                
                return value
                
            except Exception as e:
                logger.error(f"Error loading from L2 cache: {e}")
                self._remove_entry(key)
                return None
    
    def put(self, key: str, value: Any, ttl: float = None) -> bool:
        """Put item in L2 cache."""
        with self.lock:
            if ttl is None:
                ttl = self.config.default_ttl
            
            try:
                # Serialize
                data = pickle.dumps(value)
                original_size = len(data)
                
                # Compress if beneficial
                compression_type = CompressionType.NONE
                if CompressionManager.should_compress(data, self.config.compression_threshold):
                    compressed_data = CompressionManager.compress(data, self.config.compression)
                    if len(compressed_data) < len(data):
                        data = compressed_data
                        compression_type = self.config.compression
                
                # Check disk space
                if not self._can_fit_on_disk(len(data)):
                    self._evict_to_fit_disk(len(data))
                
                # Write to disk
                file_path = self.cache_dir / f"{key}.cache"
                with open(file_path, 'wb') as f:
                    f.write(data)
                
                # Update index
                self.index[key] = {
                    'timestamp': time.time(),
                    'ttl': ttl,
                    'size_bytes': len(data),
                    'original_size': original_size,
                    'compressed': compression_type != CompressionType.NONE,
                    'compression_type': compression_type.value,
                    'access_count': 0,
                    'last_access': time.time()
                }
                
                self._save_index()
                return True
                
            except Exception as e:
                logger.error(f"Error saving to L2 cache: {e}")
                return False
    
    def delete(self, key: str) -> bool:
        """Delete item from L2 cache."""
        with self.lock:
            return self._remove_entry(key)
    
    def clear(self):
        """Clear all items from L2 cache."""
        with self.lock:
            for key in list(self.index.keys()):
                self._remove_entry(key)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get L2 cache statistics."""
        with self.lock:
            total_size = sum(info['size_bytes'] for info in self.index.values())
            total_original_size = sum(info['original_size'] for info in self.index.values())
            
            compression_ratio = 0.0
            if total_original_size > 0:
                compression_ratio = total_size / total_original_size
            
            return {
                "size": len(self.index),
                "disk_usage_gb": total_size / (1024 * 1024 * 1024),
                "max_size": self.config.l2_max_size,
                "max_disk_gb": self.config.l2_max_disk_gb,
                "compression_ratio": compression_ratio
            }
    
    def _load_index(self):
        """Load cache index from disk."""
        index_file = self.cache_dir / "index.json"
        if index_file.exists():
            try:
                with open(index_file, 'r') as f:
                    self.index = json.load(f)
                logger.info(f"Loaded L2 cache index with {len(self.index)} entries")
            except Exception as e:
                logger.error(f"Error loading L2 cache index: {e}")
                self.index = {}
    
    def _save_index(self):
        """Save cache index to disk."""
        index_file = self.cache_dir / "index.json"
        try:
            with open(index_file, 'w') as f:
                json.dump(self.index, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving L2 cache index: {e}")
    
    def _remove_entry(self, key: str) -> bool:
        """Remove entry from cache and disk."""
        if key not in self.index:
            return False
        
        try:
            # Remove file
            file_path = self.cache_dir / f"{key}.cache"
            if file_path.exists():
                file_path.unlink()
            
            # Remove from index
            del self.index[key]
            self._save_index()
            
            return True
            
        except Exception as e:
            logger.error(f"Error removing L2 cache entry: {e}")
            return False
    
    def _can_fit_on_disk(self, size_bytes: int) -> bool:
        """Check if new item can fit on disk."""
        current_size = sum(info['size_bytes'] for info in self.index.values())
        max_size = self.config.l2_max_disk_gb * 1024 * 1024 * 1024
        
        return (current_size + size_bytes) <= max_size
    
    def _evict_to_fit_disk(self, needed_bytes: int):
        """Evict items to make disk space."""
        current_size = sum(info['size_bytes'] for info in self.index.values())
        max_size = self.config.l2_max_disk_gb * 1024 * 1024 * 1024
        
        # Sort by last access time (LRU)
        sorted_keys = sorted(
            self.index.keys(),
            key=lambda k: self.index[k]['last_access']
        )
        
        for key in sorted_keys:
            if (current_size + needed_bytes) <= max_size:
                break
            
            entry_size = self.index[key]['size_bytes']
            if self._remove_entry(key):
                current_size -= entry_size


class HierarchicalCacheManager:
    """Main cache manager orchestrating all cache levels."""
    
    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
        
        # Initialize cache levels
        self.l1_cache = L1MemoryCache(self.config)
        self.l2_cache = L2SSDCache(self.config)
        self.l3_cache = None  # Redis cache (optional)
        self.l4_cache = None  # Cold storage (optional)
        
        # Statistics
        self.stats = CacheStats()
        self.stats_lock = threading.Lock()
        
        # Background tasks
        self.background_tasks_running = False
        self.cleanup_thread = None
        
        # Start background tasks if enabled
        if self.config.background_cleanup:
            self.start_background_tasks()
        
        logger.info("Hierarchical Cache Manager initialized")
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache hierarchy."""
        start_time = time.time()
        
        try:
            # Try L1 (memory) first
            value = self.l1_cache.get(key)
            if value is not None:
                self._record_hit(CacheLevel.L1_MEMORY, start_time)
                return value
            
            # Try L2 (SSD)
            value = self.l2_cache.get(key)
            if value is not None:
                # Promote to L1
                self.l1_cache.put(key, value)
                self._record_hit(CacheLevel.L2_SSD, start_time)
                return value
            
            # Try L3 (Network) if available
            if self.l3_cache:
                value = self.l3_cache.get(key)
                if value is not None:
                    # Promote to L2 and L1
                    self.l2_cache.put(key, value)
                    self.l1_cache.put(key, value)
                    self._record_hit(CacheLevel.L3_NETWORK, start_time)
                    return value
            
            # Try L4 (Cold storage) if available
            if self.l4_cache:
                value = self.l4_cache.get(key)
                if value is not None:
                    # Promote through all levels
                    self.l2_cache.put(key, value)
                    self.l1_cache.put(key, value)
                    self._record_hit(CacheLevel.L4_COLD, start_time)
                    return value
            
            # Cache miss
            self._record_miss(start_time)
            return None
            
        except Exception as e:
            logger.error(f"Error getting from cache: {e}")
            self._record_miss(start_time)
            return None
    
    def put(self, key: str, value: Any, ttl: float = None) -> bool:
        """Put item in cache hierarchy."""
        try:
            # Store in all available levels
            success = True
            
            # L1 (Memory)
            if not self.l1_cache.put(key, value, ttl):
                success = False
            
            # L2 (SSD)
            if not self.l2_cache.put(key, value, ttl):
                success = False
            
            # L3 (Network) if available
            if self.l3_cache:
                if not self.l3_cache.put(key, value, ttl):
                    success = False
            
            # L4 (Cold storage) if available
            if self.l4_cache:
                if not self.l4_cache.put(key, value, ttl):
                    success = False
            
            return success
            
        except Exception as e:
            logger.error(f"Error putting to cache: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete item from all cache levels."""
        success = True
        
        # Delete from all levels
        if not self.l1_cache.delete(key):
            success = False
        
        if not self.l2_cache.delete(key):
            success = False
        
        if self.l3_cache and not self.l3_cache.delete(key):
            success = False
        
        if self.l4_cache and not self.l4_cache.delete(key):
            success = False
        
        return success
    
    def clear(self):
        """Clear all cache levels."""
        self.l1_cache.clear()
        self.l2_cache.clear()
        
        if self.l3_cache:
            self.l3_cache.clear()
        
        if self.l4_cache:
            self.l4_cache.clear()
        
        # Reset statistics
        with self.stats_lock:
            self.stats = CacheStats()
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        with self.stats_lock:
            # Calculate hit rate
            self.stats.calculate_hit_rate()
            
            # Get level-specific stats
            l1_stats = self.l1_cache.get_stats()
            l2_stats = self.l2_cache.get_stats()
            
            return {
                "overall": asdict(self.stats),
                "l1_memory": l1_stats,
                "l2_ssd": l2_stats,
                "l3_network": self.l3_cache.get_stats() if self.l3_cache else None,
                "l4_cold": self.l4_cache.get_stats() if self.l4_cache else None,
                "config": asdict(self.config)
            }
    
    def start_background_tasks(self):
        """Start background maintenance tasks."""
        if self.background_tasks_running:
            return
        
        self.background_tasks_running = True
        self.cleanup_thread = threading.Thread(target=self._background_cleanup, daemon=True)
        self.cleanup_thread.start()
        
        logger.info("Cache background tasks started")
    
    def stop_background_tasks(self):
        """Stop background maintenance tasks."""
        self.background_tasks_running = False
        if self.cleanup_thread:
            self.cleanup_thread.join(timeout=5.0)
        
        logger.info("Cache background tasks stopped")
    
    def _record_hit(self, level: CacheLevel, start_time: float):
        """Record cache hit."""
        with self.stats_lock:
            self.stats.hits += 1
            
            if level == CacheLevel.L1_MEMORY:
                self.stats.l1_hits += 1
            elif level == CacheLevel.L2_SSD:
                self.stats.l2_hits += 1
            elif level == CacheLevel.L3_NETWORK:
                self.stats.l3_hits += 1
            elif level == CacheLevel.L4_COLD:
                self.stats.l4_hits += 1
            
            # Update average access time
            access_time = (time.time() - start_time) * 1000  # ms
            if self.stats.hits == 1:
                self.stats.avg_access_time_ms = access_time
            else:
                self.stats.avg_access_time_ms = (
                    (self.stats.avg_access_time_ms * (self.stats.hits - 1) + access_time) / 
                    self.stats.hits
                )
    
    def _record_miss(self, start_time: float):
        """Record cache miss."""
        with self.stats_lock:
            self.stats.misses += 1
            
            # Update average access time
            access_time = (time.time() - start_time) * 1000  # ms
            total_requests = self.stats.hits + self.stats.misses
            if total_requests == 1:
                self.stats.avg_access_time_ms = access_time
            else:
                self.stats.avg_access_time_ms = (
                    (self.stats.avg_access_time_ms * (total_requests - 1) + access_time) / 
                    total_requests
                )
    
    def _background_cleanup(self):
        """Background cleanup task."""
        while self.background_tasks_running:
            try:
                # Cleanup expired entries
                # Note: L1 and L2 caches handle their own cleanup
                
                # Update statistics
                if self.config.enable_statistics:
                    with self.stats_lock:
                        # Update memory usage
                        l1_stats = self.l1_cache.get_stats()
                        self.stats.memory_usage_mb = l1_stats.get('memory_usage_mb', 0)
                        
                        # Update size
                        self.stats.size = l1_stats.get('size', 0)
                
                time.sleep(self.config.cleanup_interval)
                
            except Exception as e:
                logger.error(f"Error in background cleanup: {e}")
                time.sleep(self.config.cleanup_interval)


# Utility functions and decorators

def generate_cache_key(*args, **kwargs) -> str:
    """Generate cache key from function arguments."""
    # Create deterministic key from arguments
    key_data = {
        'args': args,
        'kwargs': sorted(kwargs.items()) if kwargs else {}
    }
    
    key_str = json.dumps(key_data, sort_keys=True, default=str)
    return hashlib.md5(key_str.encode()).hexdigest()


def cached(cache_manager: Optional[HierarchicalCacheManager] = None, ttl: float = None):
    """Decorator to add caching to functions."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            nonlocal cache_manager
            
            # Use global cache manager if none provided
            if cache_manager is None:
                cache_manager = get_global_cache_manager()
            
            # Generate cache key
            cache_key = f"{func.__module__}.{func.__name__}:" + generate_cache_key(*args, **kwargs)
            
            # Try to get from cache
            cached_result = cache_manager.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Store in cache
            cache_manager.put(cache_key, result, ttl)
            
            return result
        
        return wrapper
    return decorator


# Global cache manager instance
_global_cache_manager = None

def get_global_cache_manager() -> HierarchicalCacheManager:
    """Get global cache manager instance."""
    global _global_cache_manager
    if _global_cache_manager is None:
        _global_cache_manager = HierarchicalCacheManager()
    return _global_cache_manager