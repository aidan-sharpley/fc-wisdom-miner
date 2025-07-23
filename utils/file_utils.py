"""
File and directory utilities for Forum Wisdom Miner.

This module handles file operations, directory management, and thread-safe
file handling with proper locking mechanisms.
"""

import json
import logging
import os
import shutil
import time
from typing import Any, Dict, List, Optional

from config.settings import BASE_TMP_DIR, _file_lock

logger = logging.getLogger(__name__)


def get_thread_dir(thread_key: str) -> str:
    """Create and return thread directory path with validation.
    
    Args:
        thread_key: Unique identifier for the thread
        
    Returns:
        Absolute path to the thread directory
        
    Raises:
        ValueError: If thread_key is invalid
        OSError: If directory creation fails
    """
    if not thread_key or not isinstance(thread_key, str):
        raise ValueError(f"Invalid thread key: {thread_key}")
    
    # Sanitize thread key to prevent directory traversal
    import re
    sanitized_key = re.sub(r'[^\w\-_.]', '_', thread_key.strip())
    if not sanitized_key or sanitized_key in ('..', '.'):
        raise ValueError(f"Invalid thread key after sanitization: {thread_key}")
    
    path = os.path.join(BASE_TMP_DIR, sanitized_key)
    
    try:
        os.makedirs(path, exist_ok=True)
        logger.debug(f"Thread directory ready: {path}")
        return path
    except OSError as e:
        logger.error(f"Failed to create thread directory {path}: {e}")
        raise


def list_threads() -> List[str]:
    """List all available thread directories with validation.
    
    Returns:
        Sorted list of thread directory names
    """
    if not os.path.isdir(BASE_TMP_DIR):
        logger.debug(f"Base directory {BASE_TMP_DIR} does not exist")
        return []
    
    try:
        directories = []
        for item in os.listdir(BASE_TMP_DIR):
            item_path = os.path.join(BASE_TMP_DIR, item)
            if (os.path.isdir(item_path) and 
                item not in ('__pycache__', '.git', '.pytest_cache') and
                not item.startswith('.')):
                directories.append(item)
        
        return sorted(directories)
    except OSError as e:
        logger.error(f"Failed to list threads in {BASE_TMP_DIR}: {e}")
        return []


def atomic_write_json(file_path: str, data: Any, backup: bool = True) -> bool:
    """Write JSON data to file atomically with optional backup.
    
    Args:
        file_path: Target file path
        data: Data to write (must be JSON serializable)
        backup: Whether to create backup of existing file
        
    Returns:
        True if write was successful, False otherwise
    """
    try:
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with _file_lock:
            # Create backup if requested and file exists
            if backup and os.path.exists(file_path):
                backup_path = f"{file_path}.bak"
                try:
                    shutil.copy2(file_path, backup_path)
                except OSError as e:
                    logger.warning(f"Failed to create backup {backup_path}: {e}")
            
            # Write to temporary file first
            temp_path = f"{file_path}.tmp"
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            # Atomic move
            shutil.move(temp_path, file_path)
            
            # Clean up backup if write was successful
            if backup and os.path.exists(f"{file_path}.bak"):
                try:
                    os.remove(f"{file_path}.bak")
                except OSError:
                    pass  # Backup cleanup failure is not critical
            
            logger.debug(f"Successfully wrote JSON to {file_path}")
            return True
            
    except Exception as e:
        logger.error(f"Failed to write JSON to {file_path}: {e}")
        # Clean up temp file if it exists
        temp_path = f"{file_path}.tmp"
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass
        return False


def safe_read_json(file_path: str, default: Any = None) -> Any:
    """Safely read JSON file with fallback to default value.
    
    Args:
        file_path: Path to JSON file
        default: Default value if file cannot be read
        
    Returns:
        Parsed JSON data or default value
    """
    if not os.path.exists(file_path):
        logger.debug(f"JSON file does not exist: {file_path}")
        return default
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.debug(f"Successfully read JSON from {file_path}")
        return data
    except (json.JSONDecodeError, OSError) as e:
        logger.error(f"Failed to read JSON from {file_path}: {e}")
        return default


def get_file_info(file_path: str) -> Optional[Dict[str, Any]]:
    """Get detailed file information.
    
    Args:
        file_path: Path to file
        
    Returns:
        Dictionary with file info or None if file doesn't exist
    """
    try:
        if not os.path.exists(file_path):
            return None
        
        stat = os.stat(file_path)
        return {
            'size': stat.st_size,
            'modified': stat.st_mtime,
            'created': stat.st_ctime,
            'is_file': os.path.isfile(file_path),
            'is_dir': os.path.isdir(file_path),
            'readable': os.access(file_path, os.R_OK),
            'writable': os.access(file_path, os.W_OK)
        }
    except OSError as e:
        logger.error(f"Failed to get file info for {file_path}: {e}")
        return None


def cleanup_old_files(directory: str, max_age_days: int = 30, pattern: str = "*.tmp") -> int:
    """Clean up old temporary files in directory.
    
    Args:
        directory: Directory to clean
        max_age_days: Maximum age in days for files to keep
        pattern: File pattern to match (glob style)
        
    Returns:
        Number of files cleaned up
    """
    if not os.path.isdir(directory):
        return 0
    
    import glob
    cleanup_count = 0
    cutoff_time = time.time() - (max_age_days * 24 * 3600)
    
    try:
        for file_path in glob.glob(os.path.join(directory, pattern)):
            if os.path.isfile(file_path):
                try:
                    if os.path.getmtime(file_path) < cutoff_time:
                        os.remove(file_path)
                        cleanup_count += 1
                        logger.debug(f"Cleaned up old file: {file_path}")
                except OSError as e:
                    logger.warning(f"Failed to clean up {file_path}: {e}")
        
        if cleanup_count > 0:
            logger.info(f"Cleaned up {cleanup_count} old files in {directory}")
        
        return cleanup_count
    except Exception as e:
        logger.error(f"Error during cleanup in {directory}: {e}")
        return 0


def ensure_directory_exists(directory: str) -> bool:
    """Ensure directory exists, creating it if necessary.
    
    Args:
        directory: Directory path to ensure exists
        
    Returns:
        True if directory exists/was created, False otherwise
    """
    try:
        os.makedirs(directory, exist_ok=True)
        return True
    except OSError as e:
        logger.error(f"Failed to create directory {directory}: {e}")
        return False


def get_directory_size(directory: str) -> int:
    """Calculate total size of directory and its contents.
    
    Args:
        directory: Directory path
        
    Returns:
        Total size in bytes, 0 if directory doesn't exist or error occurs
    """
    if not os.path.isdir(directory):
        return 0
    
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(directory):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                try:
                    if os.path.exists(filepath):
                        total_size += os.path.getsize(filepath)
                except OSError:
                    continue  # Skip files we can't access
        return total_size
    except OSError as e:
        logger.error(f"Failed to calculate directory size for {directory}: {e}")
        return 0


def move_file_safely(src: str, dst: str) -> bool:
    """Move file safely with error handling.
    
    Args:
        src: Source file path
        dst: Destination file path
        
    Returns:
        True if move was successful, False otherwise
    """
    try:
        if not os.path.exists(src):
            logger.error(f"Source file does not exist: {src}")
            return False
        
        # Ensure destination directory exists
        dst_dir = os.path.dirname(dst)
        if dst_dir and not ensure_directory_exists(dst_dir):
            return False
        
        shutil.move(src, dst)
        logger.debug(f"Successfully moved file from {src} to {dst}")
        return True
    except (OSError, shutil.Error) as e:
        logger.error(f"Failed to move file from {src} to {dst}: {e}")
        return False


__all__ = [
    'get_thread_dir',
    'list_threads',
    'atomic_write_json',
    'safe_read_json',
    'get_file_info',
    'cleanup_old_files',
    'ensure_directory_exists',
    'get_directory_size',
    'move_file_safely'
]