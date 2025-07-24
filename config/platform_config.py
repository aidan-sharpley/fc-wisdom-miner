"""
Dynamic platform configuration system for Forum Wisdom Miner.

This module handles loading and managing forum platform-specific configurations
from YAML files, allowing the scraper to adapt to different forum types.
"""

import logging
import os
import re
from typing import Dict, List, Optional, Any
from urllib.parse import urlparse
import yaml

logger = logging.getLogger(__name__)

# Base directory for platform configurations
PLATFORM_CONFIG_DIR = os.path.join(os.path.dirname(__file__), '..', 'configs', 'platforms')

class PlatformConfigManager:
    """Manages loading and caching of platform-specific configurations."""
    
    def __init__(self):
        self._config_cache: Dict[str, Dict] = {}
        self._domain_mappings: Dict[str, str] = {}
        self._load_all_configs()
    
    def _load_all_configs(self):
        """Load all platform configurations and build domain mappings."""
        config_dir = os.path.abspath(PLATFORM_CONFIG_DIR)
        
        if not os.path.exists(config_dir):
            logger.warning(f"Platform config directory not found: {config_dir}")
            return
        
        logger.info(f"Loading platform configurations from: {config_dir}")
        
        for filename in os.listdir(config_dir):
            if filename.endswith('.yaml') or filename.endswith('.yml'):
                platform_name = filename.replace('.yaml', '').replace('.yml', '')
                config_path = os.path.join(config_dir, filename)
                
                try:
                    config = self._load_config_file(config_path)
                    self._config_cache[platform_name] = config
                    
                    # Build domain mappings
                    domains = config.get('platform', {}).get('domains', [])
                    for domain in domains:
                        self._domain_mappings[domain.lower()] = platform_name
                    
                    logger.info(f"Loaded configuration for platform: {platform_name}")
                    
                except Exception as e:
                    logger.error(f"Failed to load config {filename}: {e}")
        
        logger.info(f"Loaded {len(self._config_cache)} platform configurations")
    
    def _load_config_file(self, config_path: str) -> Dict:
        """Load a single YAML configuration file."""
        with open(config_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    
    def detect_platform(self, url: str) -> str:
        """Detect forum platform based on URL or content analysis."""
        parsed_url = urlparse(url.lower())
        domain = parsed_url.netloc.replace('www.', '')
        
        # Check for exact domain matches
        if domain in self._domain_mappings:
            return self._domain_mappings[domain]
        
        # Check for partial domain matches
        for known_domain, platform in self._domain_mappings.items():
            if known_domain in domain or domain in known_domain:
                return platform
        
        # URL pattern-based detection
        path = parsed_url.path.lower()
        
        # XenForo patterns
        if any(pattern in path for pattern in ['/threads/', '/posts/', '/forums/']):
            if any(pattern in url.lower() for pattern in ['xenforo', 'xf']):
                return 'xenforo'
        
        # vBulletin patterns  
        if any(pattern in path for pattern in ['/showthread.php', '/forumdisplay.php']):
            return 'vbulletin'
        
        # phpBB patterns
        if any(pattern in path for pattern in ['/viewtopic.php', '/viewforum.php']):
            return 'phpbb'
        
        # Default to generic configuration
        logger.info(f"No specific platform detected for {url}, using generic configuration")
        return 'generic'
    
    def get_config(self, platform_or_url: str) -> Dict:
        """Get configuration for a platform or URL."""
        # If it looks like a URL, detect the platform first
        if platform_or_url.startswith(('http://', 'https://')):
            platform = self.detect_platform(platform_or_url)
        else:
            platform = platform_or_url.lower()
        
        if platform in self._config_cache:
            return self._config_cache[platform].copy()
        
        # Fallback to generic if platform not found
        if 'generic' in self._config_cache:
            logger.warning(f"Platform '{platform}' not found, using generic configuration")
            return self._config_cache['generic'].copy()
        
        # Ultimate fallback - return minimal configuration
        logger.error(f"No configuration found for platform '{platform}' and no generic fallback")
        return self._get_minimal_config()
    
    def _get_minimal_config(self) -> Dict:
        """Return minimal configuration as absolute fallback."""
        return {
            'platform': {'name': 'Unknown', 'version': 'unknown'},
            'selectors': {
                'posts': ['.post', '.message', 'article'],
                'content': ['.content', '.text', 'p'],
                'author': ['.author', '.username'],
                'date': ['time', '.date', '.timestamp'],
                'votes': {
                    'upvotes': ['.upvote', '.like'],
                    'downvotes': ['.downvote', '.dislike'],
                    'likes': ['.likes'],
                    'reactions': ['.reactions']
                }
            },
            'scraping': {
                'delay_between_requests': 1.0,
                'max_retries': 2,
                'headers': {
                    'User-Agent': 'Mozilla/5.0 (compatible; ForumWisdomMiner/2.0)'
                }
            },
            'processing': {
                'min_post_length': 5,
                'min_content_ratio': 0.2
            }
        }
    
    def list_available_platforms(self) -> List[str]:
        """Get list of available platform configurations."""
        return list(self._config_cache.keys())
    
    def get_selectors(self, platform_or_url: str, selector_type: str) -> List[str]:
        """Get specific selectors for a platform."""
        config = self.get_config(platform_or_url)
        selectors = config.get('selectors', {})
        
        if selector_type in selectors:
            return selectors[selector_type]
        
        # For nested selectors like votes
        if '.' in selector_type:
            keys = selector_type.split('.')
            current = selectors
            for key in keys:
                if isinstance(current, dict) and key in current:
                    current = current[key]
                else:
                    return []
            return current if isinstance(current, list) else []
        
        return []
    
    def get_scraping_config(self, platform_or_url: str) -> Dict:
        """Get scraping configuration for a platform."""
        config = self.get_config(platform_or_url)
        return config.get('scraping', {})
    
    def get_processing_config(self, platform_or_url: str) -> Dict:
        """Get processing configuration for a platform."""
        config = self.get_config(platform_or_url)
        return config.get('processing', {})
    
    def reload_configs(self):
        """Reload all platform configurations."""
        self._config_cache.clear()
        self._domain_mappings.clear()
        self._load_all_configs()


# Global instance for easy access
_platform_manager = None

def get_platform_manager() -> PlatformConfigManager:
    """Get the global platform configuration manager instance."""
    global _platform_manager
    if _platform_manager is None:
        _platform_manager = PlatformConfigManager()
    return _platform_manager

def get_platform_config(platform_or_url: str) -> Dict:
    """Convenience function to get platform configuration."""
    return get_platform_manager().get_config(platform_or_url)

def detect_forum_platform(url: str) -> str:
    """Convenience function to detect forum platform from URL."""
    return get_platform_manager().detect_platform(url)

def get_selectors_for_platform(platform_or_url: str, selector_type: str) -> List[str]:
    """Convenience function to get specific selectors."""
    return get_platform_manager().get_selectors(platform_or_url, selector_type)


__all__ = [
    'PlatformConfigManager',
    'get_platform_manager',
    'get_platform_config',
    'detect_forum_platform',
    'get_selectors_for_platform',
    'PLATFORM_CONFIG_DIR'
]