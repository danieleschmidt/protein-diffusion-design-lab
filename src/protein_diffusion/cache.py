"""
Advanced caching system for protein diffusion models.

This module provides high-performance caching for model predictions,
embeddings, structure predictions, and other expensive computations.
"""

import hashlib
import pickle
import json
import time
import threading
from typing import Dict, Any, Optional, List, Union, Callable
from pathlib import Path
from dataclasses import dataclass, field
from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)

@dataclass
class CacheConfig:
    """Configuration for caching system."""
    # Memory cache settings
    max_memory_size_mb: int = 1024  # 1GB default
    max_entries: int = 10000
    ttl_seconds: int = 3600  # 1 hour
    
    # Disk cache settings
    disk_cache_enabled: bool = True
    disk_cache_dir: str = "./cache"
    max_disk_size_mb: int = 10240  # 10GB default
    
    # Cache strategies
    eviction_policy: str = "lru"  # "lru", "lfu", "ttl"
    compression_enabled: bool = True
    
    # Performance settings
    background_cleanup: bool = True
    cleanup_interval_seconds: int = 300  # 5 minutes


class CacheEntry:
    """A single cache entry with metadata."""
    
    def __init__(self, key: str, value: Any, ttl_seconds: Optional[int] = None):
        self.key = key
        self.value = value
        self.created_at = time.time()
        self.last_accessed = self.created_at
        self.access_count = 1
        self.expires_at = self.created_at + ttl_seconds if ttl_seconds else None
        self.size_bytes = self._estimate_size(value)
    
    def _estimate_size(self, obj: Any) -> int:
        """Estimate memory size of object."""
        try:
            return len(pickle.dumps(obj))
        except:
            # Fallback estimation
            if isinstance(obj, str):
                return len(obj.encode('utf-8'))
            elif isinstance(obj, (list, tuple)):
                return sum(self._estimate_size(item) for item in obj[:100])  # Sample first 100
            elif isinstance(obj, dict):
                return sum(self._estimate_size(k) + self._estimate_size(v) 
                          for k, v in list(obj.items())[:100])
            else:
                return 1024  # Default 1KB estimate
    
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at
    
    def touch(self):
        """Update access metadata."""
        self.last_accessed = time.time()
        self.access_count += 1


class MemoryCache:
    """High-performance in-memory cache with multiple eviction policies."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._current_size_bytes = 0
        
        if config.background_cleanup:
            self._start_cleanup_thread()
    
    def _start_cleanup_thread(self):
        """Start background cleanup thread."""
        def cleanup_worker():
            while True:
                try:
                    time.sleep(self.config.cleanup_interval_seconds)
                    self._cleanup_expired()
                    self._enforce_size_limits()
                except Exception as e:
                    logger.error(f"Cache cleanup error: {e}")
        
        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.start()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            if key not in self.cache:
                return None
            
            entry = self.cache[key]
            
            # Check expiration
            if entry.is_expired():
                self._remove_entry(key)
                return None
            
            # Update access metadata
            entry.touch()
            
            # Move to end for LRU
            if self.config.eviction_policy == "lru":
                self.cache.move_to_end(key)
            
            return entry.value
    
    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> bool:
        """Set value in cache."""
        with self._lock:
            # Remove existing entry if present
            if key in self.cache:
                self._remove_entry(key)
            
            # Create new entry
            entry = CacheEntry(key, value, ttl_seconds or self.config.ttl_seconds)
            
            # Check if entry is too large
            max_entry_size = self.config.max_memory_size_mb * 1024 * 1024 // 10  # 10% of cache
            if entry.size_bytes > max_entry_size:
                logger.warning(f"Entry too large for cache: {entry.size_bytes} bytes")
                return False
            
            # Make space if needed
            self._make_space(entry.size_bytes)
            
            # Add entry
            self.cache[key] = entry
            self._current_size_bytes += entry.size_bytes
            
            return True
    
    def delete(self, key: str) -> bool:
        """Delete entry from cache."""
        with self._lock:
            if key in self.cache:
                self._remove_entry(key)
                return True
            return False
    
    def clear(self):
        """Clear all entries."""
        with self._lock:
            self.cache.clear()
            self._current_size_bytes = 0
    
    def _remove_entry(self, key: str):
        """Remove entry and update size."""
        if key in self.cache:
            entry = self.cache.pop(key)
            self._current_size_bytes -= entry.size_bytes
    
    def _make_space(self, needed_bytes: int):
        """Make space in cache using eviction policy."""
        max_size_bytes = self.config.max_memory_size_mb * 1024 * 1024
        
        while (self._current_size_bytes + needed_bytes > max_size_bytes or 
               len(self.cache) >= self.config.max_entries):
            
            if not self.cache:
                break
            
            # Choose victim based on eviction policy
            if self.config.eviction_policy == "lru":
                # Remove least recently used (first in OrderedDict)
                key = next(iter(self.cache))
            elif self.config.eviction_policy == "lfu":
                # Remove least frequently used
                key = min(self.cache.keys(), key=lambda k: self.cache[k].access_count)
            elif self.config.eviction_policy == "ttl":
                # Remove oldest entry
                key = min(self.cache.keys(), key=lambda k: self.cache[k].created_at)
            else:
                key = next(iter(self.cache))
            
            self._remove_entry(key)
    
    def _cleanup_expired(self):
        """Remove expired entries."""
        with self._lock:
            expired_keys = [key for key, entry in self.cache.items() if entry.is_expired()]
            for key in expired_keys:
                self._remove_entry(key)
    
    def _enforce_size_limits(self):
        """Enforce cache size limits."""
        max_size_bytes = self.config.max_memory_size_mb * 1024 * 1024
        
        with self._lock:
            while (self._current_size_bytes > max_size_bytes or 
                   len(self.cache) > self.config.max_entries):
                if not self.cache:
                    break
                self._make_space(0)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_accesses = sum(entry.access_count for entry in self.cache.values())
            
            return {
                "entries": len(self.cache),
                "size_bytes": self._current_size_bytes,
                "size_mb": self._current_size_bytes / (1024 * 1024),
                "utilization": self._current_size_bytes / (self.config.max_memory_size_mb * 1024 * 1024),
                "total_accesses": total_accesses,
                "average_access_count": total_accesses / len(self.cache) if self.cache else 0,
                "config": self.config.__dict__,
            }


class DiskCache:
    """Persistent disk-based cache with compression."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.cache_dir = Path(config.disk_cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for organization
        (self.cache_dir / "embeddings").mkdir(exist_ok=True)
        (self.cache_dir / "predictions").mkdir(exist_ok=True)
        (self.cache_dir / "structures").mkdir(exist_ok=True)
        (self.cache_dir / "sequences").mkdir(exist_ok=True)
        
        self._index_file = self.cache_dir / "index.json"
        self._load_index()
    
    def _load_index(self):
        """Load cache index from disk."""
        try:
            if self._index_file.exists():
                with open(self._index_file, 'r') as f:
                    self._index = json.load(f)
            else:
                self._index = {}
        except Exception as e:
            logger.error(f"Failed to load cache index: {e}")
            self._index = {}
    
    def _save_index(self):
        """Save cache index to disk."""
        try:
            with open(self._index_file, 'w') as f:
                json.dump(self._index, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save cache index: {e}")
    
    def _get_cache_path(self, key: str, category: str = "general") -> Path:
        """Get file path for cache entry."""
        # Use hash to avoid filesystem issues with long keys
        key_hash = hashlib.sha256(key.encode()).hexdigest()[:16]
        return self.cache_dir / category / f"{key_hash}.cache"
    
    def get(self, key: str, category: str = "general") -> Optional[Any]:
        """Get value from disk cache."""
        try:
            if not self.config.disk_cache_enabled:
                return None
            
            cache_path = self._get_cache_path(key, category)
            
            if not cache_path.exists():
                return None
            
            # Check expiration
            if key in self._index:
                entry_info = self._index[key]
                if entry_info.get("expires_at") and time.time() > entry_info["expires_at"]:
                    self.delete(key, category)
                    return None
            
            # Load and deserialize
            with open(cache_path, 'rb') as f:
                if self.config.compression_enabled:
                    import gzip
                    data = gzip.decompress(f.read())
                else:
                    data = f.read()
                
                return pickle.loads(data)
        
        except Exception as e:
            logger.error(f"Failed to load from disk cache: {e}")
            return None
    
    def set(self, key: str, value: Any, category: str = "general", 
            ttl_seconds: Optional[int] = None) -> bool:
        """Set value in disk cache."""
        try:
            if not self.config.disk_cache_enabled:
                return False
            
            cache_path = self._get_cache_path(key, category)
            
            # Serialize
            data = pickle.dumps(value)
            
            # Compress if enabled
            if self.config.compression_enabled:
                import gzip
                data = gzip.compress(data)
            
            # Check size limits
            data_size_mb = len(data) / (1024 * 1024)
            if data_size_mb > self.config.max_disk_size_mb:
                logger.warning(f"Data too large for disk cache: {data_size_mb:.2f}MB")
                return False
            
            # Write to disk
            with open(cache_path, 'wb') as f:
                f.write(data)
            
            # Update index
            self._index[key] = {
                "category": category,
                "created_at": time.time(),
                "expires_at": time.time() + (ttl_seconds or self.config.ttl_seconds),
                "size_bytes": len(data),
                "path": str(cache_path)
            }
            
            self._save_index()
            return True
        
        except Exception as e:
            logger.error(f"Failed to save to disk cache: {e}")
            return False
    
    def delete(self, key: str, category: str = "general") -> bool:
        """Delete entry from disk cache."""
        try:
            cache_path = self._get_cache_path(key, category)
            
            if cache_path.exists():
                cache_path.unlink()
            
            if key in self._index:
                del self._index[key]
                self._save_index()
            
            return True
        except Exception as e:
            logger.error(f"Failed to delete from disk cache: {e}")
            return False
    
    def cleanup(self):
        """Clean up expired entries and enforce size limits."""
        try:
            current_time = time.time()
            expired_keys = []
            total_size = 0
            
            # Find expired entries and calculate total size
            for key, info in self._index.items():
                if info.get("expires_at", float('inf')) < current_time:
                    expired_keys.append(key)
                else:
                    total_size += info.get("size_bytes", 0)
            
            # Remove expired entries
            for key in expired_keys:
                self.delete(key, self._index[key].get("category", "general"))
            
            # Enforce size limits (remove oldest entries)
            max_size_bytes = self.config.max_disk_size_mb * 1024 * 1024
            if total_size > max_size_bytes:
                # Sort by creation time (oldest first)
                sorted_entries = sorted(
                    self._index.items(), 
                    key=lambda x: x[1].get("created_at", 0)
                )
                
                for key, info in sorted_entries:
                    if total_size <= max_size_bytes:
                        break
                    self.delete(key, info.get("category", "general"))
                    total_size -= info.get("size_bytes", 0)
        
        except Exception as e:
            logger.error(f"Cache cleanup failed: {e}")


class CacheManager:
    """
    Unified cache manager combining memory and disk caching.
    
    Provides a high-level interface for caching protein-related data
    with automatic performance optimization and memory management.
    """
    
    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
        self.memory_cache = MemoryCache(self.config)
        self.disk_cache = DiskCache(self.config)
        
        # Cache hit/miss statistics
        self._hits = 0
        self._misses = 0
        self._sets = 0
    
    def get(self, key: str, category: str = "general") -> Optional[Any]:
        """Get value from cache (memory first, then disk)."""
        # Try memory cache first
        value = self.memory_cache.get(key)
        if value is not None:
            self._hits += 1
            return value
        
        # Try disk cache
        value = self.disk_cache.get(key, category)
        if value is not None:
            # Promote to memory cache
            self.memory_cache.set(key, value)
            self._hits += 1
            return value
        
        self._misses += 1
        return None
    
    def set(self, key: str, value: Any, category: str = "general",
            ttl_seconds: Optional[int] = None, memory_only: bool = False) -> bool:
        """Set value in cache."""
        self._sets += 1
        
        # Always set in memory cache
        memory_success = self.memory_cache.set(key, value, ttl_seconds)
        
        # Set in disk cache unless memory_only
        disk_success = True
        if not memory_only:
            disk_success = self.disk_cache.set(key, value, category, ttl_seconds)
        
        return memory_success or disk_success
    
    def delete(self, key: str, category: str = "general") -> bool:
        """Delete from both caches."""
        memory_deleted = self.memory_cache.delete(key)
        disk_deleted = self.disk_cache.delete(key, category)
        return memory_deleted or disk_deleted
    
    def clear(self, category: Optional[str] = None):
        """Clear caches."""
        self.memory_cache.clear()
        if category is None:
            # Clear all disk cache - would need implementation
            pass
        
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        memory_stats = self.memory_cache.get_stats()
        
        hit_rate = self._hits / (self._hits + self._misses) if (self._hits + self._misses) > 0 else 0
        
        return {
            "hit_rate": hit_rate,
            "hits": self._hits,
            "misses": self._misses,
            "sets": self._sets,
            "memory_cache": memory_stats,
            "disk_cache_enabled": self.config.disk_cache_enabled,
        }
    
    # Convenience methods for specific data types
    
    def get_embedding(self, sequence: str) -> Optional[Any]:
        """Get cached embedding for sequence."""
        key = f"embedding:{hashlib.sha256(sequence.encode()).hexdigest()[:16]}"
        return self.get(key, "embeddings")
    
    def set_embedding(self, sequence: str, embedding: Any, ttl_seconds: Optional[int] = None) -> bool:
        """Cache embedding for sequence."""
        key = f"embedding:{hashlib.sha256(sequence.encode()).hexdigest()[:16]}"
        return self.set(key, embedding, "embeddings", ttl_seconds)
    
    def get_structure_prediction(self, sequence: str) -> Optional[Any]:
        """Get cached structure prediction."""
        key = f"structure:{hashlib.sha256(sequence.encode()).hexdigest()[:16]}"
        return self.get(key, "structures")
    
    def set_structure_prediction(self, sequence: str, prediction: Any, ttl_seconds: Optional[int] = None) -> bool:
        """Cache structure prediction."""
        key = f"structure:{hashlib.sha256(sequence.encode()).hexdigest()[:16]}"
        return self.set(key, prediction, "structures", ttl_seconds)
    
    def get_diffusion_prediction(self, prompt: str, config_hash: str) -> Optional[Any]:
        """Get cached diffusion model prediction."""
        key = f"diffusion:{hashlib.sha256((prompt + config_hash).encode()).hexdigest()[:16]}"
        return self.get(key, "predictions")
    
    def set_diffusion_prediction(self, prompt: str, config_hash: str, prediction: Any, 
                               ttl_seconds: Optional[int] = None) -> bool:
        """Cache diffusion model prediction."""
        key = f"diffusion:{hashlib.sha256((prompt + config_hash).encode()).hexdigest()[:16]}"
        return self.set(key, prediction, "predictions", ttl_seconds)


def cached(category: str = "general", ttl_seconds: Optional[int] = None, 
           key_func: Optional[Callable] = None):
    """
    Decorator for caching function results.
    
    Args:
        category: Cache category
        ttl_seconds: Time to live in seconds
        key_func: Function to generate cache key from arguments
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Get or create cache manager
            cache_manager = getattr(wrapper, '_cache_manager', None)
            if cache_manager is None:
                cache_manager = CacheManager()
                wrapper._cache_manager = cache_manager
            
            # Generate cache key
            if key_func:
                key = key_func(*args, **kwargs)
            else:
                key = f"{func.__name__}:{hash((args, tuple(sorted(kwargs.items()))))}"
            
            # Try to get from cache
            result = cache_manager.get(key, category)
            if result is not None:
                return result
            
            # Compute and cache result
            result = func(*args, **kwargs)
            cache_manager.set(key, result, category, ttl_seconds)
            return result
        
        return wrapper
    return decorator


# Global cache manager instance
_global_cache_manager = None

def get_cache_manager(config: Optional[CacheConfig] = None) -> CacheManager:
    """Get the global cache manager instance."""
    global _global_cache_manager
    if _global_cache_manager is None:
        _global_cache_manager = CacheManager(config)
    return _global_cache_manager