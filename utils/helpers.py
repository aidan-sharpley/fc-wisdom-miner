"""
General utility functions for Forum Wisdom Miner.

This module contains helper functions for hashing, URL handling,
and other general-purpose utilities.
"""

import hashlib
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


def post_hash(content: str) -> str:
    """Create a consistent hash for caching embeddings.
    
    Args:
        content: The post content to hash
        
    Returns:
        Hexadecimal hash string for cache key
        
    Raises:
        None (handles invalid input gracefully)
    """
    if not content or not isinstance(content, str):
        logger.warning(f"Invalid content for hashing: {type(content)}")
        return hashlib.sha256(b"").hexdigest()  # Return hash of empty string
    
    return hashlib.sha256(content.encode("utf-8", errors="ignore")).hexdigest()


def normalize_url(url: str) -> str:
    """Ensure URL has proper schema and basic validation.
    
    Args:
        url: Input URL string
        
    Returns:
        Normalized URL with proper schema
        
    Raises:
        ValueError: If URL is invalid after normalization
    """
    if not url or not isinstance(url, str):
        raise ValueError(f"Invalid URL type: {type(url)}")
    
    url = url.strip()
    if not url:
        raise ValueError("Empty URL provided")
    
    # Add schema if missing
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    
    # Basic validation
    if not re.match(r"^https?://[^\s/$.?#].[^\s]*$", url):
        raise ValueError(f"Malformed URL: {url}")
    
    return url


def sanitize_filename(filename: str, max_length: int = 100) -> str:
    """Sanitize a string to be safe for use as a filename.
    
    Args:
        filename: Input filename string
        max_length: Maximum length for the filename
        
    Returns:
        Sanitized filename safe for filesystem use
    """
    if not filename:
        return "unnamed"
    
    # Remove or replace unsafe characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    sanitized = re.sub(r'[^\w\-_.]', '_', sanitized)
    sanitized = re.sub(r'_+', '_', sanitized)  # Collapse multiple underscores
    sanitized = sanitized.strip('_')  # Remove leading/trailing underscores
    
    # Truncate if too long
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length].rstrip('_')
    
    return sanitized or "unnamed"


def extract_domain(url: str) -> Optional[str]:
    """Extract domain from URL for analytics and categorization.
    
    Args:
        url: Full URL string
        
    Returns:
        Domain string or None if extraction fails
    """
    try:
        # Simple regex to extract domain
        match = re.search(r'https?://([^/]+)', url)
        if match:
            domain = match.group(1)
            # Remove www. prefix if present
            return domain.replace('www.', '')
        return None
    except Exception:
        return None


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string (e.g., "1.5 MB")
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"


def calculate_similarity_score(distance: float) -> float:
    """Convert distance to similarity score.
    
    Args:
        distance: Distance value from vector search
        
    Returns:
        Similarity score between 0 and 1
    """
    # Convert cosine distance to similarity
    return max(0.0, min(1.0, 1.0 - distance))


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text to specified length with optional suffix.
    
    Args:
        text: Text to truncate
        max_length: Maximum length including suffix
        suffix: Suffix to add if text is truncated
        
    Returns:
        Truncated text with suffix if needed
    """
    if not text or len(text) <= max_length:
        return text
    
    if len(suffix) >= max_length:
        return text[:max_length]
    
    return text[:max_length - len(suffix)] + suffix


def safe_int(value: str, default: int = 0) -> int:
    """Safely convert string to integer with fallback.
    
    Args:
        value: String value to convert
        default: Default value if conversion fails
        
    Returns:
        Integer value or default
    """
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def safe_float(value: str, default: float = 0.0) -> float:
    """Safely convert string to float with fallback.
    
    Args:
        value: String value to convert
        default: Default value if conversion fails
        
    Returns:
        Float value or default
    """
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


__all__ = [
    'post_hash',
    'normalize_url', 
    'sanitize_filename',
    'extract_domain',
    'format_file_size',
    'calculate_similarity_score',
    'truncate_text',
    'safe_int',
    'safe_float'
]