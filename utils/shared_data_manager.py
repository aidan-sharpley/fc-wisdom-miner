"""
Centralized data management for thread-level operations.
Memory-efficient caching and lazy loading for M1 MacBook Air.
"""

import json
import os
import threading
import time
import weakref
from typing import Dict, List, Optional, Any

from utils.file_utils import safe_read_json


class ThreadDataManager:
    """Centralized, memory-efficient data manager for thread operations."""
    
    _instances = weakref.WeakValueDictionary()
    _lock = threading.RLock()
    
    def __new__(cls, thread_dir: str):
        with cls._lock:
            if thread_dir in cls._instances:
                return cls._instances[thread_dir]
            instance = super().__new__(cls)
            cls._instances[thread_dir] = instance
            return instance
    
    def __init__(self, thread_dir: str):
        if hasattr(self, '_initialized'):
            return
        
        self.thread_dir = thread_dir
        self._posts = None
        self._metadata = None
        self._analytics = None
        self._last_access = {}
        self._memory_threshold = 100 * 1024 * 1024  # 100MB
        self._lock = threading.RLock()
        self._initialized = True
    
    def get_posts(self, force_reload: bool = False) -> List[Dict]:
        """Get posts with lazy loading and memory management."""
        with self._lock:
            if self._posts is None or force_reload:
                posts_file = os.path.join(self.thread_dir, "posts.json")
                self._posts = safe_read_json(posts_file) or []
                self._last_access['posts'] = time.time()
            return self._posts
    
    def get_metadata(self, force_reload: bool = False) -> Dict:
        """Get metadata with lazy loading."""
        with self._lock:
            if self._metadata is None or force_reload:
                metadata_file = os.path.join(self.thread_dir, "metadata.json")
                self._metadata = safe_read_json(metadata_file) or {}
                self._last_access['metadata'] = time.time()
            return self._metadata
    
    def get_analytics(self, force_reload: bool = False) -> Dict:
        """Get analytics with lazy loading."""
        with self._lock:
            if self._analytics is None or force_reload:
                analytics_file = os.path.join(self.thread_dir, "thread_analytics.json")
                self._analytics = safe_read_json(analytics_file) or {}
                self._last_access['analytics'] = time.time()
            return self._analytics
    
    def get_summary(self) -> Optional[Dict]:
        """Get comprehensive summary if available."""
        summary_file = os.path.join(self.thread_dir, "thread_summary.json")
        return safe_read_json(summary_file)
    
    def estimate_memory_usage(self) -> int:
        """Estimate current memory usage in bytes."""
        total = 0
        if self._posts:
            total += len(json.dumps(self._posts).encode('utf-8'))
        if self._metadata:
            total += len(json.dumps(self._metadata).encode('utf-8'))
        if self._analytics:
            total += len(json.dumps(self._analytics).encode('utf-8'))
        return total
    
    def cleanup_if_needed(self):
        """Clean up cached data if memory threshold exceeded."""
        if self.estimate_memory_usage() > self._memory_threshold:
            current_time = time.time()
            # Clear least recently used data
            if self._last_access.get('analytics', 0) < current_time - 300:  # 5min
                self._analytics = None
            if self._last_access.get('metadata', 0) < current_time - 180:  # 3min
                self._metadata = None
    
    def clear_cache(self):
        """Clear all cached data."""
        with self._lock:
            self._posts = None
            self._metadata = None
            self._analytics = None
            self._last_access.clear()


def get_data_manager(thread_dir: str) -> ThreadDataManager:
    """Get or create a data manager for the specified thread."""
    return ThreadDataManager(thread_dir)


__all__ = ['ThreadDataManager', 'get_data_manager']