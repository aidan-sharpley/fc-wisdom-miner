"""
Security utilities for Forum Wisdom Miner.

This module provides robust input validation and sanitization to prevent
Server-Side Request Forgery (SSRF) and path traversal attacks.
"""

import ipaddress
import re
import urllib.parse
from typing import Set, Optional
import logging

logger = logging.getLogger(__name__)

# Dangerous ports that should be blocked
BLOCKED_PORTS = {
    22,    # SSH
    23,    # Telnet
    25,    # SMTP
    53,    # DNS
    110,   # POP3
    143,   # IMAP
    993,   # IMAPS
    995,   # POP3S
    1433,  # SQL Server
    3306,  # MySQL
    5432,  # PostgreSQL
    6379,  # Redis
    11211, # Memcached
    27017, # MongoDB
}

# Private IP ranges that should be blocked (RFC 1918, RFC 3927, etc.)
PRIVATE_IP_RANGES = [
    ipaddress.ip_network('10.0.0.0/8'),
    ipaddress.ip_network('172.16.0.0/12'),
    ipaddress.ip_network('192.168.0.0/16'),
    ipaddress.ip_network('169.254.0.0/16'),  # Link-local
    ipaddress.ip_network('127.0.0.0/8'),     # Loopback
    ipaddress.ip_network('::1/128'),          # IPv6 loopback
    ipaddress.ip_network('fc00::/7'),         # IPv6 unique local
    ipaddress.ip_network('fe80::/10'),        # IPv6 link-local
]


def validate_url(url: str, allowed_schemes: Set[str] = {'http', 'https'}) -> tuple[bool, str]:
    """
    Validate URL for security vulnerabilities including SSRF prevention.
    
    Args:
        url: URL string to validate
        allowed_schemes: Set of allowed URL schemes
        
    Returns:
        Tuple of (is_valid, normalized_url_or_error_message)
    """
    if not url or not isinstance(url, str):
        return False, "Invalid URL type"
    
    url = url.strip()
    if not url:
        return False, "Empty URL provided"
    
    # Add https if no scheme provided
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    
    try:
        parsed = urllib.parse.urlparse(url)
    except Exception as e:
        return False, f"URL parsing failed: {str(e)}"
    
    # Validate scheme
    if parsed.scheme not in allowed_schemes:
        return False, f"Scheme '{parsed.scheme}' not allowed. Only {', '.join(allowed_schemes)} are permitted"
    
    # Validate hostname exists
    if not parsed.netloc:
        return False, "URL must contain a valid hostname"
    
    # Extract hostname and port
    hostname = parsed.hostname
    port = parsed.port
    
    if not hostname:
        return False, "Invalid hostname in URL"
    
    # Validate port if specified
    if port is not None:
        if port in BLOCKED_PORTS:
            return False, f"Port {port} is not allowed for security reasons"
        if port < 1 or port > 65535:
            return False, f"Invalid port number: {port}"
    
    # Check for IP addresses (both IPv4 and IPv6)
    try:
        ip = ipaddress.ip_address(hostname)
        
        # Block private/internal IP ranges
        for private_range in PRIVATE_IP_RANGES:
            if ip in private_range:
                return False, f"Access to private/internal IP addresses is not allowed: {hostname}"
        
        # Block multicast and reserved ranges
        if ip.is_multicast or ip.is_reserved:
            return False, f"Access to multicast/reserved IP addresses is not allowed: {hostname}"
            
    except ValueError:
        # Not an IP address, validate as hostname
        if not _is_valid_hostname(hostname):
            return False, f"Invalid hostname format: {hostname}"
    
    # Additional validation for suspicious patterns
    if _contains_suspicious_patterns(url):
        return False, "URL contains suspicious patterns"
    
    # Reconstruct URL to ensure it's properly formatted
    try:
        normalized_url = urllib.parse.urlunparse(parsed)
        return True, normalized_url
    except Exception as e:
        return False, f"URL normalization failed: {str(e)}"


def _is_valid_hostname(hostname: str) -> bool:
    """Validate hostname format according to RFC standards."""
    if len(hostname) > 253:
        return False
    
    # Check for valid characters and format
    if not re.match(r'^[a-zA-Z0-9]([a-zA-Z0-9\-]*[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9\-]*[a-zA-Z0-9])?)*$', hostname):
        return False
    
    # Check each label
    labels = hostname.split('.')
    for label in labels:
        if len(label) > 63 or len(label) == 0:
            return False
        if label.startswith('-') or label.endswith('-'):
            return False
    
    return True


def _contains_suspicious_patterns(url: str) -> bool:
    """Check for suspicious patterns that might indicate attacks."""
    suspicious_patterns = [
        r'@',                    # URLs with @ (potential credential injection)
        r'%[0-9a-fA-F]{2}%[0-9a-fA-F]{2}',  # Double URL encoding
        r'\\',                   # Backslashes in URLs
        r'\s',                   # Whitespace characters
        r'[<>"]',               # HTML/XML characters
        r'javascript:',         # JavaScript protocol
        r'data:',               # Data protocol
        r'file:',               # File protocol
        r'ftp:',                # FTP protocol
    ]
    
    url_lower = url.lower()
    for pattern in suspicious_patterns:
        if re.search(pattern, url_lower):
            return True
    
    return False


def validate_thread_key(thread_key: str) -> bool:
    """
    Enhanced validation for thread keys to prevent path traversal and injection attacks.
    
    Args:
        thread_key: Thread key string to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not thread_key or not isinstance(thread_key, str):
        return False
    
    # Length validation
    if len(thread_key) < 1 or len(thread_key) > 100:
        return False
    
    # Character allowlist: alphanumeric, hyphens, underscores only
    if not re.match(r'^[a-zA-Z0-9_-]+$', thread_key):
        return False
    
    # Prevent path traversal attempts
    if any(pattern in thread_key for pattern in ['..', '/', '\\', '\0', '\r', '\n']):
        return False
    
    # Prevent reserved names
    reserved_names = {
        'con', 'prn', 'aux', 'nul', 
        'com1', 'com2', 'com3', 'com4', 'com5', 'com6', 'com7', 'com8', 'com9',
        'lpt1', 'lpt2', 'lpt3', 'lpt4', 'lpt5', 'lpt6', 'lpt7', 'lpt8', 'lpt9'
    }
    if thread_key.lower() in reserved_names:
        return False
    
    # Prevent leading/trailing dots or spaces
    if thread_key.startswith('.') or thread_key.endswith('.'):
        return False
    
    if thread_key.startswith(' ') or thread_key.endswith(' '):
        return False
    
    return True


def sanitize_thread_key_component(component: str, max_length: int = 20) -> str:
    """
    Sanitize a component for use in thread key generation.
    
    Args:
        component: String component to sanitize
        max_length: Maximum length for the component
        
    Returns:
        Sanitized component safe for use in thread keys
    """
    if not component or not isinstance(component, str):
        return ""
    
    # Remove/replace unsafe characters - only keep alphanumeric, hyphens, underscores
    sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', component)
    
    # Collapse multiple underscores/hyphens
    sanitized = re.sub(r'[_-]+', '_', sanitized)
    
    # Remove leading/trailing underscores/hyphens
    sanitized = sanitized.strip('_-')
    
    # Truncate to max length
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length].rstrip('_-')
    
    # Ensure it's not empty and doesn't start with numbers only
    if not sanitized or sanitized.isdigit():
        return ""
    
    return sanitized


__all__ = [
    'validate_url',
    'validate_thread_key', 
    'sanitize_thread_key_component',
    'BLOCKED_PORTS',
    'PRIVATE_IP_RANGES'
]