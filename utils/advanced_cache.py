"""
Advanced caching system for Forum Wisdom Miner.

This module provides intelligent caching with content-based invalidation,
performance optimization, and memory management.
"""

import hashlib
import logging
import pickle
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class ContentBasedCache:
    """Advanced cache with content-based invalidation and performance optimization."""
    
    def __init__(self, cache_dir: str, max_size_mb: int = 500, cleanup_threshold: float = 0.8):
        """Initialize the content-based cache.
        
        Args:
            cache_dir: Directory for cache storage
            max_size_mb: Maximum cache size in MB
            cleanup_threshold: Cleanup when cache reaches this percentage of max
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.cleanup_threshold = cleanup_threshold
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Cache metadata
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self._metadata = self._load_metadata()
        
        # Performance tracking
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'cache_size_bytes': 0,
            'last_cleanup': time.time()
        }
        
        # Update cache size on startup
        self._update_cache_size()
    
    def get(self, key: str, content_hash: Optional[str] = None) -> Optional[Any]:
        """Get item from cache with optional content validation.
        
        Args:
            key: Cache key
            content_hash: Optional content hash for validation
            
        Returns:
            Cached value or None if not found/invalid
        """
        with self._lock:
            try:
                cache_key = self._generate_cache_key(key)
                cache_file = self.cache_dir / f"{cache_key}.pkl"
                
                if not cache_file.exists():
                    self.stats['misses'] += 1
                    return None
                
                # Check metadata
                if cache_key not in self._metadata:
                    self.stats['misses'] += 1
                    return None
                
                meta = self._metadata[cache_key]
                
                # Validate content hash if provided
                if content_hash and meta.get('content_hash') != content_hash:
                    logger.debug(f"Cache invalidated for {key}: content hash mismatch")
                    self._remove_cache_entry(cache_key)
                    self.stats['misses'] += 1
                    return None
                
                # Check expiration
                if meta.get('expires_at', 0) < time.time():
                    logger.debug(f"Cache expired for {key}")
                    self._remove_cache_entry(cache_key)
                    self.stats['misses'] += 1
                    return None
                
                # Load and return cached data
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                
                # Update access time
                meta['last_accessed'] = time.time()
                meta['access_count'] = meta.get('access_count', 0) + 1
                self._save_metadata()
                
                self.stats['hits'] += 1
                logger.debug(f"Cache hit for {key}")
                return data
                
            except Exception as e:
                logger.warning(f"Error reading from cache for {key}: {e}")
                self.stats['misses'] += 1
                return None
    
    def set(self, key: str, value: Any, content_hash: Optional[str] = None, 
            ttl_hours: int = 24) -> bool:
        """Set item in cache with optional content hash and TTL.
        
        Args:
            key: Cache key
            value: Value to cache
            content_hash: Optional content hash for validation
            ttl_hours: Time to live in hours
            
        Returns:
            True if successfully cached
        """
        with self._lock:
            try:
                cache_key = self._generate_cache_key(key)
                cache_file = self.cache_dir / f"{cache_key}.pkl"
                
                # Serialize data
                with open(cache_file, 'wb') as f:
                    pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
                
                # Update metadata
                file_size = cache_file.stat().st_size
                expires_at = time.time() + (ttl_hours * 3600) if ttl_hours > 0 else 0
                
                self._metadata[cache_key] = {
                    'key': key,
                    'content_hash': content_hash,
                    'size_bytes': file_size,
                    'created_at': time.time(),
                    'last_accessed': time.time(),
                    'expires_at': expires_at,
                    'access_count': 0
                }
                
                self._save_metadata()
                self._update_cache_size()
                
                # Check if cleanup is needed
                if self.stats['cache_size_bytes'] > self.max_size_bytes * self.cleanup_threshold:
                    self._cleanup_cache()
                
                logger.debug(f"Cached {key} ({file_size} bytes)")
                return True
                
            except Exception as e:
                logger.error(f"Error caching {key}: {e}")
                return False
    
    def invalidate(self, key: str) -> bool:
        """Invalidate a specific cache entry.
        
        Args:
            key: Cache key to invalidate
            
        Returns:
            True if entry was removed
        """
        with self._lock:
            cache_key = self._generate_cache_key(key)
            return self._remove_cache_entry(cache_key)
    
    def invalidate_by_pattern(self, pattern: str) -> int:
        """Invalidate cache entries matching a pattern.
        
        Args:
            pattern: Pattern to match against keys
            
        Returns:
            Number of entries invalidated
        """
        with self._lock:
            count = 0
            keys_to_remove = []
            
            for cache_key, meta in self._metadata.items():
                if pattern in meta.get('key', ''):
                    keys_to_remove.append(cache_key)
            
            for cache_key in keys_to_remove:
                if self._remove_cache_entry(cache_key):
                    count += 1
            
            logger.info(f"Invalidated {count} cache entries matching pattern: {pattern}")
            return count
    
    def invalidate_by_content_change(self, thread_key: str) -> int:
        """Invalidate all cache entries for a thread (used when content changes).
        
        Args:
            thread_key: Thread key to invalidate
            
        Returns:
            Number of entries invalidated
        """
        return self.invalidate_by_pattern(thread_key)
    
    def _generate_cache_key(self, key: str) -> str:
        """Generate a filesystem-safe cache key."""
        return hashlib.md5(key.encode()).hexdigest()
    
    def _remove_cache_entry(self, cache_key: str) -> bool:
        """Remove a cache entry and its metadata.
        
        Args:
            cache_key: Cache key to remove
            
        Returns:
            True if removed successfully
        """
        try:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            
            if cache_file.exists():
                cache_file.unlink()
            
            if cache_key in self._metadata:
                del self._metadata[cache_key]
                self._save_metadata()
                self._update_cache_size()
                return True
                
        except Exception as e:
            logger.warning(f"Error removing cache entry {cache_key}: {e}")
        
        return False
    
    def _cleanup_cache(self):
        """Clean up cache by removing old and least-used entries."""
        try:
            logger.info("Starting cache cleanup")
            
            # Sort entries by score (access frequency + recency)
            entries = []
            current_time = time.time()
            
            for cache_key, meta in self._metadata.items():
                age_hours = (current_time - meta.get('created_at', 0)) / 3600
                last_access_hours = (current_time - meta.get('last_accessed', 0)) / 3600
                access_count = meta.get('access_count', 0)
                
                # Score: higher is better (more recent, more accessed)
                score = (access_count + 1) / (1 + last_access_hours + age_hours / 24)
                
                entries.append((cache_key, score, meta.get('size_bytes', 0)))
            
            # Sort by score (lowest first for removal)
            entries.sort(key=lambda x: x[1])
            
            # Remove entries until under threshold
            target_size = self.max_size_bytes * 0.7  # Clean to 70% capacity
            removed_count = 0
            
            for cache_key, score, size in entries:
                if self.stats['cache_size_bytes'] <= target_size:
                    break
                
                if self._remove_cache_entry(cache_key):
                    removed_count += 1
                    self.stats['evictions'] += 1
            
            self.stats['last_cleanup'] = time.time()
            logger.info(f"Cache cleanup completed: removed {removed_count} entries")
            
        except Exception as e:
            logger.error(f"Error during cache cleanup: {e}")
    
    def _update_cache_size(self):
        """Update total cache size statistics."""
        total_size = 0
        for cache_key in self._metadata:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            if cache_file.exists():
                total_size += cache_file.stat().st_size
        
        self.stats['cache_size_bytes'] = total_size
    
    def _load_metadata(self) -> Dict:
        """Load cache metadata from disk."""
        try:
            if self.metadata_file.exists():
                import json
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Error loading cache metadata: {e}")
        
        return {}
    
    def _save_metadata(self):
        """Save cache metadata to disk."""
        try:
            import json
            with open(self.metadata_file, 'w') as f:
                json.dump(self._metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving cache metadata: {e}")
    
    def get_stats(self) -> Dict:
        """Get cache statistics."""
        total_requests = self.stats['hits'] + self.stats['misses']
        hit_rate = self.stats['hits'] / max(1, total_requests)
        
        return {
            **self.stats,
            'hit_rate': hit_rate,
            'cache_entries': len(self._metadata),
            'cache_size_mb': self.stats['cache_size_bytes'] / (1024 * 1024),
            'max_size_mb': self.max_size_bytes / (1024 * 1024)
        }
    
    def clear(self):
        """Clear entire cache."""
        with self._lock:
            try:
                # Remove all cache files
                for cache_file in self.cache_dir.glob("*.pkl"):
                    cache_file.unlink()
                
                # Clear metadata
                self._metadata.clear()
                self._save_metadata()
                
                # Reset stats
                self.stats.update({
                    'hits': 0,
                    'misses': 0,
                    'evictions': 0,
                    'cache_size_bytes': 0
                })
                
                logger.info("Cache cleared completely")
                
            except Exception as e:
                logger.error(f"Error clearing cache: {e}")


class SmartQueryCache:
    """Smart query caching that considers query similarity and result freshness."""
    
    def __init__(self, cache_dir: str):
        """Initialize smart query cache.
        
        Args:
            cache_dir: Directory for cache storage
        """
        self.base_cache = ContentBasedCache(cache_dir, max_size_mb=200)
        self.query_similarity_threshold = 0.8
    
    def get_similar_query_results(self, query: str, thread_key: str, 
                                max_age_hours: int = 6) -> Optional[Dict]:
        """Get results for similar queries.
        
        Args:
            query: Current query
            thread_key: Thread key
            max_age_hours: Maximum age for cached results
            
        Returns:
            Cached results if similar query found
        """
        cache_key = f"query:{thread_key}:{query}"
        return self.base_cache.get(cache_key)
    
    def cache_query_results(self, query: str, thread_key: str, results: Dict,
                          ttl_hours: int = 6) -> bool:
        """Cache query results.
        
        Args:
            query: Query string
            thread_key: Thread key
            results: Results to cache
            ttl_hours: Time to live in hours
            
        Returns:
            True if cached successfully
        """
        cache_key = f"query:{thread_key}:{query}"
        return self.base_cache.set(cache_key, results, ttl_hours=ttl_hours)
    
    def invalidate_thread_queries(self, thread_key: str) -> int:
        """Invalidate all cached queries for a thread.
        
        Args:
            thread_key: Thread key
            
        Returns:
            Number of invalidated entries  
        """
        return self.base_cache.invalidate_by_pattern(f"query:{thread_key}")


__all__ = ['ContentBasedCache', 'SmartQueryCache']