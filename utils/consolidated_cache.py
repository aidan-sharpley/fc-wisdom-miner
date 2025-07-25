"""
Consolidated caching system replacing thousands of individual pickle files.
Memory-efficient storage for embeddings and other cached data.
"""

import os
import sqlite3
import threading
import time
from typing import Any, Optional, Dict
import pickle
import numpy as np

from utils.memory_optimizer import MemoryMonitor


class ConsolidatedCache:
    """SQLite-based cache for embeddings and other data."""
    
    def __init__(self, cache_dir: str, max_size_mb: int = 150):
        self.cache_dir = cache_dir
        self.max_size_mb = max_size_mb
        self.db_path = os.path.join(cache_dir, "cache.db")
        self._lock = threading.RLock()
        
        os.makedirs(cache_dir, exist_ok=True)
        self._init_db()
        
        # Register with memory monitor
        monitor = MemoryMonitor()
        monitor.register_cleanup_callback(self._memory_cleanup)
    
    def _init_db(self):
        """Initialize SQLite database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_entries (
                    key TEXT PRIMARY KEY,
                    content_hash TEXT,
                    data BLOB,
                    created_at REAL,
                    accessed_at REAL,
                    size_bytes INTEGER
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_accessed_at ON cache_entries(accessed_at)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_content_hash ON cache_entries(content_hash)
            """)
    
    def get(self, key: str, content_hash: str = None) -> Optional[Any]:
        """Get cached item."""
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "SELECT data, content_hash FROM cache_entries WHERE key = ?",
                        (key,)
                    )
                    row = cursor.fetchone()
                    
                    if row is None:
                        return None
                    
                    data_blob, stored_hash = row
                    
                    # Validate content hash if provided
                    if content_hash and stored_hash != content_hash:
                        self._delete_key(key, conn)
                        return None
                    
                    # Update access time
                    cursor.execute(
                        "UPDATE cache_entries SET accessed_at = ? WHERE key = ?",
                        (time.time(), key)
                    )
                    
                    return pickle.loads(data_blob)
            except Exception:
                return None
    
    def set(self, key: str, value: Any, content_hash: str, ttl_hours: int = 168):
        """Set cached item."""
        with self._lock:
            try:
                data_blob = pickle.dumps(value)
                size_bytes = len(data_blob)
                current_time = time.time()
                
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        INSERT OR REPLACE INTO cache_entries 
                        (key, content_hash, data, created_at, accessed_at, size_bytes)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (key, content_hash, data_blob, current_time, current_time, size_bytes))
                
                self._cleanup_if_needed()
            except Exception:
                pass  # Silent failure for cache operations
    
    def _delete_key(self, key: str, conn: sqlite3.Connection):
        """Delete a cache entry."""
        conn.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
    
    def _cleanup_if_needed(self):
        """Clean up old entries if cache is too large."""
        try:
            cache_size_mb = os.path.getsize(self.db_path) / 1024 / 1024
            if cache_size_mb > self.max_size_mb:
                self._cleanup_old_entries()
        except Exception:
            pass
    
    def _cleanup_old_entries(self):
        """Remove oldest entries to free space."""
        with sqlite3.connect(self.db_path) as conn:
            # Delete oldest 20% of entries
            conn.execute("""
                DELETE FROM cache_entries 
                WHERE key IN (
                    SELECT key FROM cache_entries 
                    ORDER BY accessed_at ASC 
                    LIMIT (SELECT COUNT(*) * 0.2 FROM cache_entries)
                )
            """)
            conn.execute("VACUUM")
    
    def _memory_cleanup(self):
        """Emergency cleanup for memory pressure."""
        with sqlite3.connect(self.db_path) as conn:
            # Delete oldest 50% of entries
            conn.execute("""
                DELETE FROM cache_entries 
                WHERE key IN (
                    SELECT key FROM cache_entries 
                    ORDER BY accessed_at ASC 
                    LIMIT (SELECT COUNT(*) * 0.5 FROM cache_entries)
                )
            """)
    
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM cache_entries")
                conn.execute("VACUUM")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*), SUM(size_bytes) FROM cache_entries")
                count, total_size = cursor.fetchone()
                
                cache_file_size = os.path.getsize(self.db_path)
                
                return {
                    'entries': count or 0,
                    'total_size_bytes': total_size or 0,
                    'file_size_bytes': cache_file_size,
                    'file_size_mb': cache_file_size / 1024 / 1024
                }
        except Exception:
            return {'entries': 0, 'total_size_bytes': 0, 'file_size_bytes': 0, 'file_size_mb': 0}


__all__ = ['ConsolidatedCache']