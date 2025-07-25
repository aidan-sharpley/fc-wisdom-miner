"""
Memory optimization utilities for M1 MacBook Air with 8GB RAM.
"""

import gc
import os
import psutil
import threading
from typing import Callable, Any

# Memory thresholds for 8GB system
MEMORY_WARNING_THRESHOLD = 6.0 * 1024 * 1024 * 1024  # 6GB
MEMORY_CRITICAL_THRESHOLD = 7.0 * 1024 * 1024 * 1024  # 7GB


class MemoryMonitor:
    """Lightweight memory monitoring for resource-constrained systems."""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        self._process = None
        self._cleanup_callbacks = []
        self._initialized = True
    
    def _ensure_process(self):
        """Lazily initialize the process object."""
        if self._process is None:
            self._process = psutil.Process(os.getpid())
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in bytes."""
        self._ensure_process()
        return self._process.memory_info().rss
    
    def get_memory_percent(self) -> float:
        """Get memory usage as percentage of system memory."""
        self._ensure_process()
        return self._process.memory_percent()
    
    def is_memory_critical(self) -> bool:
        """Check if memory usage is critical."""
        return self.get_memory_usage() > MEMORY_CRITICAL_THRESHOLD
    
    def is_memory_warning(self) -> bool:
        """Check if memory usage is at warning level."""
        return self.get_memory_usage() > MEMORY_WARNING_THRESHOLD
    
    def register_cleanup_callback(self, callback: Callable[[], None]):
        """Register callback for memory pressure cleanup."""
        self._cleanup_callbacks.append(callback)
    
    def trigger_cleanup(self):
        """Trigger all registered cleanup callbacks."""
        for callback in self._cleanup_callbacks:
            try:
                callback()
            except Exception:
                pass  # Silent cleanup
        gc.collect()
    
    def check_and_cleanup(self):
        """Check memory and cleanup if needed."""
        if self.is_memory_critical():
            self.trigger_cleanup()
        elif self.is_memory_warning():
            gc.collect()


def memory_efficient(func: Callable) -> Callable:
    """Decorator for memory-efficient function execution."""
    def wrapper(*args, **kwargs):
        monitor = MemoryMonitor()
        monitor.check_and_cleanup()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            if monitor.is_memory_warning():
                gc.collect()
    return wrapper


def get_memory_status() -> dict:
    """Get current memory status for debugging."""
    monitor = MemoryMonitor()
    return {
        'usage_bytes': monitor.get_memory_usage(),
        'usage_percent': monitor.get_memory_percent(),
        'is_warning': monitor.is_memory_warning(),
        'is_critical': monitor.is_memory_critical()
    }


__all__ = ['MemoryMonitor', 'memory_efficient', 'get_memory_status']