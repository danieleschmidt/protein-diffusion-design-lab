"""
Advanced Multi-Level Caching System for TPU-Optimized Protein Diffusion

This module provides a sophisticated caching framework specifically designed
for protein diffusion models running on TPUs, with intelligent cache management,
hierarchical storage, and performance optimization.
"""

import logging
import time
import json
import hashlib
import pickle
import asyncio
import threading
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import tempfile
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import weakref
import gc

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    import memcache
    MEMCACHE_AVAILABLE = True
except ImportError:
    MEMCACHE_AVAILABLE = False

logger = logging.getLogger(__name__)

class CacheLevel(Enum):
    """Cache hierarchy levels."""
    L1_MEMORY = "l1_memory"         # In-memory cache (fastest)
    L2_SSD = "l2_ssd"               # Local SSD cache
    L3_NETWORK = "l3_network"       # Network distributed cache
    L4_COLD_STORAGE = "l4_cold"     # Cold storage (slowest)

class CacheStrategy(Enum):
    """Cache eviction strategies."""
    LRU = "lru"                     # Least Recently Used
    LFU = "lfu"                     # Least Frequently Used
    TTL = "ttl"                     # Time To Live
    ADAPTIVE = "adaptive"           # Adaptive based on access patterns
    PROTEIN_AWARE = "protein_aware" # Protein-specific caching logic

class CompressionType(Enum):
    """Compression types for cached data."""
    NONE = "none"
    GZIP = "gzip"
    LZMA = "lzma"
    ZSTD = "zstd"
    CUSTOM = "custom"

@dataclass
class CacheConfig:
    """Configuration for advanced caching system."""
    # Cache hierarchy
    l1_size_mb: int = 1024          # 1GB L1 cache
    l2_size_mb: int = 10240         # 10GB L2 cache
    l3_size_mb: int = 102400        # 100GB L3 cache
    l4_size_mb: int = 1048576       # 1TB L4 cache
    
    # Cache strategies
    l1_strategy: CacheStrategy = CacheStrategy.LRU
    l2_strategy: CacheStrategy = CacheStrategy.ADAPTIVE
    l3_strategy: CacheStrategy = CacheStrategy.LFU
    l4_strategy: CacheStrategy = CacheStrategy.TTL
    
    # Performance tuning
    compression: CompressionType = CompressionType.ZSTD
    enable_prefetching: bool = True
    prefetch_threads: int = 4
    enable_background_cleanup: bool = True
    cleanup_interval: float = 300.0  # 5 minutes
    
    # TPU-specific optimizations
    tpu_memory_threshold: float = 0.8  # 80% memory usage threshold
    batch_size_aware: bool = True
    sequence_length_aware: bool = True
    
    # Network cache settings
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    
    # Persistence
    enable_persistence: bool = True
    persistence_path: str = "/tmp/protein_diffusion_cache"

@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    data: Any
    size_bytes: int
    created_time: float
    last_accessed: float
    access_count: int = 0
    compression: CompressionType = CompressionType.NONE
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def update_access(self):
        """Update access statistics."""
        self.last_accessed = time.time()
        self.access_count += 1

class AdvancedCacheManager:
    """
    Advanced multi-level cache manager for protein diffusion models.
    
    Features:
    - 4-level cache hierarchy (Memory -> SSD -> Network -> Cold Storage)
    - Intelligent cache promotion/demotion
    - TPU-aware caching strategies
    - Asynchronous prefetching and cleanup
    - Compression and serialization
    - Performance monitoring and analytics
    """
    
    def __init__(self, config: CacheConfig):
        self.config = config
        
        # Cache storage
        self.l1_cache: Dict[str, CacheEntry] = {}  # Memory cache
        self.l2_path = Path(config.persistence_path) / "l2_cache"
        self.l3_client = None  # Redis/Memcache client
        self.l4_path = Path(config.persistence_path) / "l4_cache"
        
        # Cache statistics
        self.stats = {
            'hits': {'l1': 0, 'l2': 0, 'l3': 0, 'l4': 0},
            'misses': 0,
            'evictions': {'l1': 0, 'l2': 0, 'l3': 0, 'l4': 0},
            'promotions': 0,
            'demotions': 0
        }
        
        # Thread management
        self.cleanup_thread = None
        self.prefetch_executor = ThreadPoolExecutor(max_workers=config.prefetch_threads)
        self.running = False
        
        # Locks for thread safety
        self.l1_lock = threading.RLock()
        self.stats_lock = threading.Lock()
        
        self._initialize_cache_directories()
        self._initialize_network_cache()
        self._start_background_tasks()
        
        logger.info("Advanced cache manager initialized")
    
    def _initialize_cache_directories(self):
        """Initialize cache directories."""
        self.l2_path.mkdir(parents=True, exist_ok=True)
        self.l4_path.mkdir(parents=True, exist_ok=True)
        
        # Create metadata files
        (self.l2_path / "metadata.json").touch()
        (self.l4_path / "metadata.json").touch()
    
    def _initialize_network_cache(self):
        """Initialize network cache (Redis/Memcache)."""
        if REDIS_AVAILABLE:
            try:
                import redis
                self.l3_client = redis.Redis(
                    host=self.config.redis_host,
                    port=self.config.redis_port,
                    db=self.config.redis_db,
                    decode_responses=False
                )
                # Test connection
                self.l3_client.ping()
                logger.info("Redis L3 cache initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Redis: {e}")
                self.l3_client = None
        
        if self.l3_client is None and MEMCACHE_AVAILABLE:
            try:
                import memcache
                self.l3_client = memcache.Client([f"{self.config.redis_host}:11211"])
                logger.info("Memcache L3 cache initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Memcache: {e}")
                self.l3_client = None
    
    def _start_background_tasks(self):
        """Start background cleanup and monitoring tasks."""
        self.running = True
        
        if self.config.enable_background_cleanup:
            self.cleanup_thread = threading.Thread(
                target=self._background_cleanup_loop,
                daemon=True
            )
            self.cleanup_thread.start()
    
    def _background_cleanup_loop(self):
        """Background cleanup loop."""
        while self.running:
            try:
                self._cleanup_expired_entries()
                self._optimize_cache_distribution()
                time.sleep(self.config.cleanup_interval)
            except Exception as e:
                logger.error(f"Background cleanup error: {e}")
                time.sleep(60)  # Wait longer on error
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get item from cache with intelligent lookup across hierarchy.
        
        Args:
            key: Cache key
            default: Default value if not found
            
        Returns:
            Cached value or default
        """
        # Try L1 cache first (fastest)
        if entry := self._get_from_l1(key):
            entry.update_access()
            with self.stats_lock:
                self.stats['hits']['l1'] += 1
            return entry.data
        
        # Try L2 cache (SSD)
        if entry := self._get_from_l2(key):
            # Promote to L1 if space available
            self._promote_to_l1(entry)
            with self.stats_lock:
                self.stats['hits']['l2'] += 1
            return entry.data
        
        # Try L3 cache (Network)
        if entry := self._get_from_l3(key):
            # Promote to L2 and possibly L1
            self._promote_to_l2(entry)
            with self.stats_lock:
                self.stats['hits']['l3'] += 1
            return entry.data
        
        # Try L4 cache (Cold storage)
        if entry := self._get_from_l4(key):
            # Promote to L3
            self._promote_to_l3(entry)
            with self.stats_lock:
                self.stats['hits']['l4'] += 1
            return entry.data
        
        # Cache miss
        with self.stats_lock:
            self.stats['misses'] += 1
        
        return default
    
    def put(self, key: str, value: Any, metadata: Optional[Dict[str, Any]] = None,
            ttl: Optional[float] = None) -> bool:
        """
        Put item into cache with intelligent placement.
        
        Args:
            key: Cache key
            value: Value to cache
            metadata: Optional metadata
            ttl: Time to live in seconds
            
        Returns:
            True if successfully cached
        """
        # Calculate size and create entry
        size_bytes = self._calculate_size(value)
        
        entry = CacheEntry(
            key=key,
            data=value,
            size_bytes=size_bytes,
            created_time=time.time(),
            last_accessed=time.time(),
            metadata=metadata or {}
        )
        
        # Add TTL if specified
        if ttl:
            entry.metadata['ttl'] = ttl
            entry.metadata['expires_at'] = time.time() + ttl
        
        # Determine optimal cache level based on size and access patterns
        optimal_level = self._determine_optimal_cache_level(entry)
        
        # Store in optimal level
        if optimal_level == CacheLevel.L1_MEMORY:
            return self._put_in_l1(entry)
        elif optimal_level == CacheLevel.L2_SSD:
            return self._put_in_l2(entry)
        elif optimal_level == CacheLevel.L3_NETWORK:
            return self._put_in_l3(entry)
        else:
            return self._put_in_l4(entry)
    
    def _get_from_l1(self, key: str) -> Optional[CacheEntry]:
        """Get from L1 memory cache."""
        with self.l1_lock:
            return self.l1_cache.get(key)
    
    def _get_from_l2(self, key: str) -> Optional[CacheEntry]:
        """Get from L2 SSD cache."""
        cache_file = self.l2_path / f"{self._hash_key(key)}.cache"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    entry = pickle.load(f)
                    
                # Check TTL
                if self._is_expired(entry):
                    cache_file.unlink()
                    return None
                    
                return entry
            except Exception as e:
                logger.warning(f"Failed to load L2 cache entry {key}: {e}")
                # Clean up corrupted file
                cache_file.unlink(missing_ok=True)
        
        return None
    
    def _get_from_l3(self, key: str) -> Optional[CacheEntry]:
        """Get from L3 network cache."""
        if not self.l3_client:
            return None
        
        try:
            data = self.l3_client.get(key)
            if data:
                entry = pickle.loads(data)
                
                # Check TTL
                if self._is_expired(entry):
                    self.l3_client.delete(key)
                    return None
                    
                return entry
        except Exception as e:
            logger.warning(f"Failed to get from L3 cache: {e}")
        
        return None
    
    def _get_from_l4(self, key: str) -> Optional[CacheEntry]:
        """Get from L4 cold storage."""
        cache_file = self.l4_path / f"{self._hash_key(key)}.cache"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    entry = pickle.load(f)
                    
                # Check TTL
                if self._is_expired(entry):
                    cache_file.unlink()
                    return None
                    
                return entry
            except Exception as e:
                logger.warning(f"Failed to load L4 cache entry {key}: {e}")
                cache_file.unlink(missing_ok=True)
        
        return None
    
    def _put_in_l1(self, entry: CacheEntry) -> bool:
        """Put entry in L1 memory cache."""
        with self.l1_lock:
            # Check if we need to evict
            current_size = sum(e.size_bytes for e in self.l1_cache.values())
            max_size = self.config.l1_size_mb * 1024 * 1024
            
            if current_size + entry.size_bytes > max_size:
                self._evict_from_l1(entry.size_bytes)
            
            self.l1_cache[entry.key] = entry
            return True
    
    def _put_in_l2(self, entry: CacheEntry) -> bool:
        """Put entry in L2 SSD cache."""
        cache_file = self.l2_path / f"{self._hash_key(entry.key)}.cache"
        
        try:
            # Compress if configured
            if self.config.compression != CompressionType.NONE:
                entry = self._compress_entry(entry)
            
            with open(cache_file, 'wb') as f:
                pickle.dump(entry, f)
            
            return True
        except Exception as e:
            logger.error(f"Failed to write L2 cache entry: {e}")
            return False
    
    def _put_in_l3(self, entry: CacheEntry) -> bool:
        """Put entry in L3 network cache."""
        if not self.l3_client:
            return False
        
        try:
            # Compress for network transfer
            if self.config.compression != CompressionType.NONE:
                entry = self._compress_entry(entry)
            
            data = pickle.dumps(entry)
            
            # Set TTL if specified
            ttl = entry.metadata.get('ttl')
            if ttl:
                self.l3_client.setex(entry.key, int(ttl), data)
            else:
                self.l3_client.set(entry.key, data)
            
            return True
        except Exception as e:
            logger.error(f"Failed to write L3 cache entry: {e}")
            return False
    
    def _put_in_l4(self, entry: CacheEntry) -> bool:
        """Put entry in L4 cold storage."""
        cache_file = self.l4_path / f"{self._hash_key(entry.key)}.cache"
        
        try:
            # Always compress for cold storage
            entry = self._compress_entry(entry)
            
            with open(cache_file, 'wb') as f:
                pickle.dump(entry, f)
            
            return True
        except Exception as e:
            logger.error(f"Failed to write L4 cache entry: {e}")
            return False
    
    def _determine_optimal_cache_level(self, entry: CacheEntry) -> CacheLevel:
        """Determine optimal cache level for entry."""
        size_mb = entry.size_bytes / (1024 * 1024)
        
        # Small frequently accessed items go to L1
        if size_mb < 10 and self._predict_access_frequency(entry) > 0.8:
            return CacheLevel.L1_MEMORY
        
        # Medium-sized items with moderate access go to L2
        if size_mb < 100 and self._predict_access_frequency(entry) > 0.5:
            return CacheLevel.L2_SSD
        
        # Large items or items with low access go to L3/L4
        if size_mb < 1000:
            return CacheLevel.L3_NETWORK
        else:
            return CacheLevel.L4_COLD_STORAGE
    
    def _predict_access_frequency(self, entry: CacheEntry) -> float:
        """Predict access frequency for cache placement."""
        # Simple heuristic based on metadata
        metadata = entry.metadata
        
        # Protein sequence data is frequently accessed
        if metadata.get('type') == 'protein_sequence':
            return 0.9
        
        # Model weights are moderately accessed
        if metadata.get('type') == 'model_weights':
            return 0.7
        
        # Training data is less frequently accessed
        if metadata.get('type') == 'training_data':
            return 0.4
        
        # Default frequency
        return 0.5
    
    def _promote_to_l1(self, entry: CacheEntry):
        """Promote entry to L1 cache."""
        if self._can_fit_in_l1(entry):
            self._put_in_l1(entry)
            with self.stats_lock:
                self.stats['promotions'] += 1
    
    def _promote_to_l2(self, entry: CacheEntry):
        """Promote entry to L2 cache."""
        self._put_in_l2(entry)
        # Also try L1 if small enough
        if entry.size_bytes < 10 * 1024 * 1024:  # < 10MB
            self._promote_to_l1(entry)
        with self.stats_lock:
            self.stats['promotions'] += 1
    
    def _promote_to_l3(self, entry: CacheEntry):
        """Promote entry to L3 cache."""
        self._put_in_l3(entry)
        with self.stats_lock:
            self.stats['promotions'] += 1
    
    def _can_fit_in_l1(self, entry: CacheEntry) -> bool:
        """Check if entry can fit in L1 cache."""
        with self.l1_lock:
            current_size = sum(e.size_bytes for e in self.l1_cache.values())
            max_size = self.config.l1_size_mb * 1024 * 1024
            return current_size + entry.size_bytes <= max_size
    
    def _evict_from_l1(self, needed_space: int):
        """Evict entries from L1 cache."""
        with self.l1_lock:
            # Sort by eviction priority (LRU by default)
            entries = list(self.l1_cache.values())
            entries.sort(key=lambda e: e.last_accessed)
            
            freed_space = 0
            for entry in entries:
                if freed_space >= needed_space:
                    break
                
                # Demote to L2
                self._put_in_l2(entry)
                del self.l1_cache[entry.key]
                freed_space += entry.size_bytes
                
                with self.stats_lock:
                    self.stats['evictions']['l1'] += 1
                    self.stats['demotions'] += 1
    
    def _compress_entry(self, entry: CacheEntry) -> CacheEntry:
        """Compress cache entry data."""
        if entry.compression != CompressionType.NONE:
            return entry  # Already compressed
        
        compression_type = self.config.compression
        
        try:
            if compression_type == CompressionType.GZIP:
                import gzip
                compressed_data = gzip.compress(pickle.dumps(entry.data))
            elif compression_type == CompressionType.LZMA:
                import lzma
                compressed_data = lzma.compress(pickle.dumps(entry.data))
            elif compression_type == CompressionType.ZSTD:
                try:
                    import zstandard as zstd
                    cctx = zstd.ZstdCompressor()
                    compressed_data = cctx.compress(pickle.dumps(entry.data))
                except ImportError:
                    # Fallback to gzip
                    import gzip
                    compressed_data = gzip.compress(pickle.dumps(entry.data))
            else:
                return entry  # No compression
            
            # Create new entry with compressed data
            compressed_entry = CacheEntry(
                key=entry.key,
                data=compressed_data,
                size_bytes=len(compressed_data),
                created_time=entry.created_time,
                last_accessed=entry.last_accessed,
                access_count=entry.access_count,
                compression=compression_type,
                metadata=entry.metadata.copy()
            )
            
            # Store original size for decompression
            compressed_entry.metadata['original_size'] = entry.size_bytes
            
            return compressed_entry
            
        except Exception as e:
            logger.warning(f"Compression failed: {e}")
            return entry
    
    def _decompress_entry(self, entry: CacheEntry) -> CacheEntry:
        """Decompress cache entry data."""
        if entry.compression == CompressionType.NONE:
            return entry
        
        try:
            if entry.compression == CompressionType.GZIP:
                import gzip
                original_data = pickle.loads(gzip.decompress(entry.data))
            elif entry.compression == CompressionType.LZMA:
                import lzma
                original_data = pickle.loads(lzma.decompress(entry.data))
            elif entry.compression == CompressionType.ZSTD:
                try:
                    import zstandard as zstd
                    dctx = zstd.ZstdDecompressor()
                    original_data = pickle.loads(dctx.decompress(entry.data))
                except ImportError:
                    raise ValueError("ZSTD decompression not available")
            else:
                return entry
            
            # Create decompressed entry
            decompressed_entry = CacheEntry(
                key=entry.key,
                data=original_data,
                size_bytes=entry.metadata.get('original_size', len(pickle.dumps(original_data))),
                created_time=entry.created_time,
                last_accessed=entry.last_accessed,
                access_count=entry.access_count,
                compression=CompressionType.NONE,
                metadata=entry.metadata.copy()
            )
            
            return decompressed_entry
            
        except Exception as e:
            logger.error(f"Decompression failed: {e}")
            return entry
    
    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if cache entry is expired."""
        expires_at = entry.metadata.get('expires_at')
        if expires_at:
            return time.time() > expires_at
        return False
    
    def _calculate_size(self, obj: Any) -> int:
        """Calculate size of object in bytes."""
        if NUMPY_AVAILABLE and isinstance(obj, np.ndarray):
            return obj.nbytes
        else:
            try:
                return len(pickle.dumps(obj))
            except:
                return 1024  # Default size estimate
    
    def _hash_key(self, key: str) -> str:
        """Generate hash for cache key."""
        return hashlib.sha256(key.encode()).hexdigest()
    
    def _cleanup_expired_entries(self):
        """Clean up expired cache entries."""
        current_time = time.time()
        
        # Clean L1 cache
        with self.l1_lock:
            expired_keys = [
                key for key, entry in self.l1_cache.items()
                if self._is_expired(entry)
            ]
            for key in expired_keys:
                del self.l1_cache[key]
        
        # Clean L2 cache
        for cache_file in self.l2_path.glob("*.cache"):
            try:
                with open(cache_file, 'rb') as f:
                    entry = pickle.load(f)
                if self._is_expired(entry):
                    cache_file.unlink()
            except:
                # Remove corrupted files
                cache_file.unlink(missing_ok=True)
        
        # Clean L4 cache
        for cache_file in self.l4_path.glob("*.cache"):
            try:
                with open(cache_file, 'rb') as f:
                    entry = pickle.load(f)
                if self._is_expired(entry):
                    cache_file.unlink()
            except:
                cache_file.unlink(missing_ok=True)
    
    def _optimize_cache_distribution(self):
        """Optimize cache distribution across levels."""
        # This would implement sophisticated cache optimization logic
        pass
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        with self.stats_lock:
            stats = self.stats.copy()
        
        # Calculate hit rates
        total_requests = sum(stats['hits'].values()) + stats['misses']
        if total_requests > 0:
            stats['hit_rate'] = sum(stats['hits'].values()) / total_requests
            stats['l1_hit_rate'] = stats['hits']['l1'] / total_requests
        
        # Cache sizes
        with self.l1_lock:
            l1_size = sum(entry.size_bytes for entry in self.l1_cache.values())
            l1_count = len(self.l1_cache)
        
        stats['cache_sizes'] = {
            'l1_bytes': l1_size,
            'l1_count': l1_count,
            'l2_count': len(list(self.l2_path.glob("*.cache"))),
            'l4_count': len(list(self.l4_path.glob("*.cache")))
        }
        
        return stats
    
    def clear_cache(self, level: Optional[CacheLevel] = None):
        """Clear cache at specified level or all levels."""
        if level is None or level == CacheLevel.L1_MEMORY:
            with self.l1_lock:
                self.l1_cache.clear()
        
        if level is None or level == CacheLevel.L2_SSD:
            for cache_file in self.l2_path.glob("*.cache"):
                cache_file.unlink()
        
        if level is None or level == CacheLevel.L3_NETWORK:
            if self.l3_client:
                try:
                    self.l3_client.flushdb()
                except:
                    pass
        
        if level is None or level == CacheLevel.L4_COLD_STORAGE:
            for cache_file in self.l4_path.glob("*.cache"):
                cache_file.unlink()
        
        logger.info(f"Cache cleared: {level.value if level else 'all levels'}")
    
    def shutdown(self):
        """Shutdown cache manager."""
        self.running = False
        
        if self.cleanup_thread:
            self.cleanup_thread.join(timeout=5)
        
        self.prefetch_executor.shutdown(wait=True)
        
        logger.info("Cache manager shutdown complete")

def create_cache_manager(l1_size_mb: int = 1024,
                        l2_size_mb: int = 10240,
                        compression: CompressionType = CompressionType.ZSTD) -> AdvancedCacheManager:
    """
    Factory function to create cache manager with common configurations.
    
    Args:
        l1_size_mb: L1 cache size in MB
        l2_size_mb: L2 cache size in MB
        compression: Compression type
        
    Returns:
        Configured cache manager
    """
    config = CacheConfig(
        l1_size_mb=l1_size_mb,
        l2_size_mb=l2_size_mb,
        compression=compression
    )
    
    return AdvancedCacheManager(config)

# Export main classes and functions
__all__ = [
    'AdvancedCacheManager', 'CacheConfig', 'CacheEntry', 'CacheLevel',
    'CacheStrategy', 'CompressionType', 'create_cache_manager'
]