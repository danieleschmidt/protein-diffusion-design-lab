"""
Cache management system for expensive operations.

This module provides a unified caching interface with support for 
multiple backends including Redis and in-memory caching.
"""

import os
import pickle
import hashlib
import json
import logging
from typing import Any, Optional, Dict, Union, Callable
from dataclasses import dataclass
from functools import wraps
from datetime import datetime, timedelta
import threading
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class CacheConfig:
    """Configuration for cache management."""
    # Cache backends
    enable_redis: bool = True
    enable_memory: bool = True
    
    # Redis settings
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_url: Optional[str] = None
    
    # Memory cache settings
    memory_max_size: int = 1000  # Max number of items
    memory_max_age: timedelta = timedelta(hours=1)
    
    # Default TTL values (in seconds)
    default_ttl: int = 3600  # 1 hour
    structure_prediction_ttl: int = 86400  # 24 hours
    binding_calculation_ttl: int = 43200  # 12 hours
    model_inference_ttl: int = 7200  # 2 hours
    
    # Key prefixes
    key_prefix: str = "protein_diffusion"
    structure_prefix: str = "structure"
    binding_prefix: str = "binding"
    inference_prefix: str = "inference"
    
    @classmethod
    def from_env(cls) -> 'CacheConfig':
        """Create config from environment variables."""
        return cls(
            enable_redis=os.getenv('ENABLE_REDIS_CACHE', 'true').lower() == 'true',
            enable_memory=os.getenv('ENABLE_MEMORY_CACHE', 'true').lower() == 'true',
            redis_host=os.getenv('REDIS_HOST', 'localhost'),
            redis_port=int(os.getenv('REDIS_PORT', '6379')),
            redis_db=int(os.getenv('REDIS_DB', '0')),
            redis_url=os.getenv('REDIS_URL'),
            memory_max_size=int(os.getenv('MEMORY_CACHE_SIZE', '1000')),
            default_ttl=int(os.getenv('CACHE_DEFAULT_TTL', '3600')),
            structure_prediction_ttl=int(os.getenv('STRUCTURE_CACHE_TTL', '86400')),
            binding_calculation_ttl=int(os.getenv('BINDING_CACHE_TTL', '43200')),
            model_inference_ttl=int(os.getenv('INFERENCE_CACHE_TTL', '7200')),
        )


class CacheBackend(ABC):
    """Abstract base class for cache backends."""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache with optional TTL."""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        pass
    
    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        pass
    
    @abstractmethod
    def clear(self) -> bool:
        """Clear all cached data."""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        pass


class MemoryCache(CacheBackend):
    """In-memory cache implementation."""
    
    def __init__(self, max_size: int = 1000, max_age: timedelta = timedelta(hours=1)):
        self.max_size = max_size
        self.max_age = max_age
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
            "evictions": 0,
        }
    
    def _is_expired(self, entry: Dict[str, Any]) -> bool:
        """Check if cache entry is expired."""
        if "expires_at" not in entry:
            return False
        return datetime.utcnow() > entry["expires_at"]
    
    def _evict_expired(self):
        """Remove expired entries."""
        with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items() 
                if self._is_expired(entry)
            ]
            for key in expired_keys:
                del self._cache[key]
                self._stats["evictions"] += 1
    
    def _evict_lru(self):
        """Evict least recently used entries to maintain size limit."""
        with self._lock:
            if len(self._cache) <= self.max_size:
                return
            
            # Sort by last access time
            sorted_items = sorted(
                self._cache.items(),
                key=lambda x: x[1].get("last_access", datetime.min)
            )
            
            # Remove oldest entries
            num_to_remove = len(self._cache) - self.max_size
            for key, _ in sorted_items[:num_to_remove]:
                del self._cache[key]
                self._stats["evictions"] += 1
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from memory cache."""
        with self._lock:
            if key not in self._cache:
                self._stats["misses"] += 1
                return None
            
            entry = self._cache[key]
            
            # Check if expired
            if self._is_expired(entry):
                del self._cache[key]
                self._stats["misses"] += 1
                self._stats["evictions"] += 1
                return None
            
            # Update access time
            entry["last_access"] = datetime.utcnow()
            self._stats["hits"] += 1
            
            return entry["value"]
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in memory cache."""
        with self._lock:
            try:
                # Calculate expiration time
                expires_at = None
                if ttl is not None:
                    expires_at = datetime.utcnow() + timedelta(seconds=ttl)
                elif self.max_age:
                    expires_at = datetime.utcnow() + self.max_age
                
                # Store entry
                self._cache[key] = {
                    "value": value,
                    "created_at": datetime.utcnow(),
                    "last_access": datetime.utcnow(),
                    "expires_at": expires_at,
                }
                
                self._stats["sets"] += 1
                
                # Cleanup if needed
                self._evict_expired()
                self._evict_lru()
                
                return True
                
            except Exception as e:
                logger.error(f"Failed to set cache key {key}: {e}")
                return False
    
    def delete(self, key: str) -> bool:
        """Delete key from memory cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._stats["deletes"] += 1
                return True
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists in memory cache."""
        with self._lock:
            if key not in self._cache:
                return False
            
            # Check if expired
            if self._is_expired(self._cache[key]):
                del self._cache[key]
                self._stats["evictions"] += 1
                return False
            
            return True
    
    def clear(self) -> bool:
        """Clear all cached data."""
        with self._lock:
            self._cache.clear()
            return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            hit_rate = 0.0
            total_requests = self._stats["hits"] + self._stats["misses"]
            if total_requests > 0:
                hit_rate = self._stats["hits"] / total_requests
            
            return {
                **self._stats,
                "size": len(self._cache),
                "max_size": self.max_size,
                "hit_rate": hit_rate,
                "total_requests": total_requests,
            }


class RedisCache(CacheBackend):
    """Redis cache implementation."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self._client = None
        self._stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
            "errors": 0,
        }
        
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Redis client."""
        try:
            import redis
            
            if self.config.redis_url:
                self._client = redis.from_url(self.config.redis_url)
            else:
                self._client = redis.Redis(
                    host=self.config.redis_host,
                    port=self.config.redis_port,
                    db=self.config.redis_db,
                    decode_responses=False,  # We handle our own serialization
                )
            
            # Test connection
            self._client.ping()
            logger.info("Redis cache initialized successfully")
            
        except ImportError:
            logger.warning("Redis not available - install with: pip install redis")
            self._client = None
        except Exception as e:
            logger.error(f"Failed to initialize Redis cache: {e}")
            self._client = None
    
    def _serialize_value(self, value: Any) -> bytes:
        """Serialize value for storage."""
        return pickle.dumps(value)
    
    def _deserialize_value(self, data: bytes) -> Any:
        """Deserialize value from storage."""
        return pickle.loads(data)
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache."""
        if not self._client:
            return None
        
        try:
            data = self._client.get(key)
            if data is None:
                self._stats["misses"] += 1
                return None
            
            value = self._deserialize_value(data)
            self._stats["hits"] += 1
            return value
            
        except Exception as e:
            logger.error(f"Redis GET error for key {key}: {e}")
            self._stats["errors"] += 1
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in Redis cache."""
        if not self._client:
            return False
        
        try:
            data = self._serialize_value(value)
            result = self._client.set(key, data, ex=ttl)
            
            if result:
                self._stats["sets"] += 1
            
            return bool(result)
            
        except Exception as e:
            logger.error(f"Redis SET error for key {key}: {e}")
            self._stats["errors"] += 1
            return False
    
    def delete(self, key: str) -> bool:
        """Delete key from Redis cache."""
        if not self._client:
            return False
        
        try:
            result = self._client.delete(key)
            if result:
                self._stats["deletes"] += 1
            return bool(result)
            
        except Exception as e:
            logger.error(f"Redis DELETE error for key {key}: {e}")
            self._stats["errors"] += 1
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists in Redis cache."""
        if not self._client:
            return False
        
        try:
            return bool(self._client.exists(key))
        except Exception as e:
            logger.error(f"Redis EXISTS error for key {key}: {e}")
            self._stats["errors"] += 1
            return False
    
    def clear(self) -> bool:
        """Clear all cached data."""
        if not self._client:
            return False
        
        try:
            # Only clear keys with our prefix to avoid affecting other data
            pattern = f"{self.config.key_prefix}:*"
            keys = self._client.keys(pattern)
            if keys:
                self._client.delete(*keys)
            return True
            
        except Exception as e:
            logger.error(f"Redis CLEAR error: {e}")
            self._stats["errors"] += 1
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        hit_rate = 0.0
        total_requests = self._stats["hits"] + self._stats["misses"]
        if total_requests > 0:
            hit_rate = self._stats["hits"] / total_requests
        
        redis_info = {}
        if self._client:
            try:
                redis_info = {
                    "connected": True,
                    "memory_usage": self._client.info().get("used_memory_human"),
                    "keyspace_hits": self._client.info().get("keyspace_hits"),
                    "keyspace_misses": self._client.info().get("keyspace_misses"),
                }
            except Exception as e:
                redis_info = {"connected": False, "error": str(e)}
        
        return {
            **self._stats,
            "hit_rate": hit_rate,
            "total_requests": total_requests,
            "redis_info": redis_info,
        }


class CacheManager:
    """
    Main cache manager that coordinates multiple cache backends.
    
    This class provides a unified interface for caching with support for
    multiple backends (Redis, memory) and intelligent cache strategies.
    """
    
    def __init__(self, config: Optional[CacheConfig] = None):
        if config is None:
            config = CacheConfig.from_env()
        
        self.config = config
        self.backends = []
        
        # Initialize backends
        if config.enable_memory:
            memory_cache = MemoryCache(
                max_size=config.memory_max_size,
                max_age=config.memory_max_age
            )
            self.backends.append(memory_cache)
            logger.info("Memory cache backend initialized")
        
        if config.enable_redis:
            redis_cache = RedisCache(config)
            if redis_cache._client:  # Only add if Redis is available
                self.backends.append(redis_cache)
                logger.info("Redis cache backend initialized")
        
        if not self.backends:
            logger.warning("No cache backends available")
    
    def _make_key(self, key: str, prefix: Optional[str] = None) -> str:
        """Create a cache key with appropriate prefix."""
        parts = [self.config.key_prefix]
        if prefix:
            parts.append(prefix)
        parts.append(key)
        return ":".join(parts)
    
    def get(self, key: str, prefix: Optional[str] = None) -> Optional[Any]:
        """Get value from cache, trying backends in order."""
        cache_key = self._make_key(key, prefix)
        
        for backend in self.backends:
            try:
                value = backend.get(cache_key)
                if value is not None:
                    # Populate earlier backends with the found value
                    for earlier_backend in self.backends:
                        if earlier_backend == backend:
                            break
                        earlier_backend.set(cache_key, value)
                    
                    return value
            except Exception as e:
                logger.error(f"Error getting from cache backend: {e}")
                continue
        
        return None
    
    def set(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[int] = None, 
        prefix: Optional[str] = None
    ) -> bool:
        """Set value in all available cache backends."""
        cache_key = self._make_key(key, prefix)
        
        if ttl is None:
            ttl = self.config.default_ttl
        
        success = False
        for backend in self.backends:
            try:
                if backend.set(cache_key, value, ttl):
                    success = True
            except Exception as e:
                logger.error(f"Error setting cache in backend: {e}")
                continue
        
        return success
    
    def delete(self, key: str, prefix: Optional[str] = None) -> bool:
        """Delete key from all cache backends."""
        cache_key = self._make_key(key, prefix)
        
        success = False
        for backend in self.backends:
            try:
                if backend.delete(cache_key):
                    success = True
            except Exception as e:
                logger.error(f"Error deleting from cache backend: {e}")
                continue
        
        return success
    
    def exists(self, key: str, prefix: Optional[str] = None) -> bool:
        """Check if key exists in any cache backend."""
        cache_key = self._make_key(key, prefix)
        
        for backend in self.backends:
            try:
                if backend.exists(cache_key):
                    return True
            except Exception as e:
                logger.error(f"Error checking cache backend: {e}")
                continue
        
        return False
    
    def clear(self, prefix: Optional[str] = None) -> bool:
        """Clear cache in all backends."""
        success = False
        for backend in self.backends:
            try:
                if backend.clear():
                    success = True
            except Exception as e:
                logger.error(f"Error clearing cache backend: {e}")
                continue
        
        return success
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics from all cache backends."""
        stats = {"backends": []}
        
        for i, backend in enumerate(self.backends):
            try:
                backend_stats = backend.get_stats()
                backend_stats["type"] = backend.__class__.__name__
                stats["backends"].append(backend_stats)
            except Exception as e:
                stats["backends"].append({
                    "type": backend.__class__.__name__,
                    "error": str(e)
                })
        
        return stats
    
    # Specialized caching methods for common operations
    
    def cache_structure_prediction(
        self, 
        sequence: str, 
        result: Any, 
        method: str = "esmfold"
    ) -> bool:
        """Cache structure prediction result."""
        key_data = f"{method}:{hashlib.sha256(sequence.encode()).hexdigest()}"
        return self.set(
            key_data,
            result,
            ttl=self.config.structure_prediction_ttl,
            prefix=self.config.structure_prefix
        )
    
    def get_cached_structure(self, sequence: str, method: str = "esmfold") -> Optional[Any]:
        """Get cached structure prediction."""
        key_data = f"{method}:{hashlib.sha256(sequence.encode()).hexdigest()}"
        return self.get(key_data, prefix=self.config.structure_prefix)
    
    def cache_binding_calculation(
        self, 
        protein_sequence: str, 
        target_info: str, 
        result: Any
    ) -> bool:
        """Cache binding affinity calculation."""
        combined = f"{protein_sequence}:{target_info}"
        key_data = hashlib.sha256(combined.encode()).hexdigest()
        return self.set(
            key_data,
            result,
            ttl=self.config.binding_calculation_ttl,
            prefix=self.config.binding_prefix
        )
    
    def get_cached_binding(self, protein_sequence: str, target_info: str) -> Optional[Any]:
        """Get cached binding calculation."""
        combined = f"{protein_sequence}:{target_info}"
        key_data = hashlib.sha256(combined.encode()).hexdigest()
        return self.get(key_data, prefix=self.config.binding_prefix)
    
    def cache_model_inference(
        self, 
        input_data: Any, 
        model_config: Dict[str, Any], 
        result: Any
    ) -> bool:
        """Cache model inference result."""
        # Create a hash of input data and model config
        input_str = json.dumps(input_data, sort_keys=True, default=str)
        config_str = json.dumps(model_config, sort_keys=True)
        combined = f"{input_str}:{config_str}"
        key_data = hashlib.sha256(combined.encode()).hexdigest()
        
        return self.set(
            key_data,
            result,
            ttl=self.config.model_inference_ttl,
            prefix=self.config.inference_prefix
        )
    
    def get_cached_inference(
        self, 
        input_data: Any, 
        model_config: Dict[str, Any]
    ) -> Optional[Any]:
        """Get cached model inference result."""
        input_str = json.dumps(input_data, sort_keys=True, default=str)
        config_str = json.dumps(model_config, sort_keys=True)
        combined = f"{input_str}:{config_str}"
        key_data = hashlib.sha256(combined.encode()).hexdigest()
        
        return self.get(key_data, prefix=self.config.inference_prefix)


# Decorator for automatic caching
def cached(
    ttl: Optional[int] = None,
    prefix: Optional[str] = None,
    key_func: Optional[Callable] = None,
    cache_manager: Optional[CacheManager] = None,
):
    """
    Decorator for automatic function result caching.
    
    Args:
        ttl: Time to live in seconds
        prefix: Cache key prefix
        key_func: Function to generate cache key from arguments
        cache_manager: Cache manager instance to use
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal cache_manager
            
            if cache_manager is None:
                cache_manager = CacheManager()
            
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Default key generation
                key_parts = [func.__name__]
                key_parts.extend([str(arg) for arg in args])
                key_parts.extend([f"{k}={v}" for k, v in sorted(kwargs.items())])
                cache_key = hashlib.sha256(":".join(key_parts).encode()).hexdigest()
            
            # Try to get from cache
            cached_result = cache_manager.get(cache_key, prefix=prefix)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache_manager.set(cache_key, result, ttl=ttl, prefix=prefix)
            
            return result
        
        return wrapper
    return decorator


# Global cache manager instance
_cache_manager: Optional[CacheManager] = None


def get_cache_manager() -> CacheManager:
    """Get global cache manager instance."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager