"""
Topic Index Cache Management

Provides persistent storage and retrieval for topic indexes created during
thread parsing. Manages cache directories and file operations.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime

from config.settings import BASE_TMP_DIR

logger = logging.getLogger(__name__)


class TopicIndexCache:
    """
    Manages persistent storage of topic indexes for threads.
    
    Stores topic indexes in a structured cache directory with metadata
    for efficient retrieval and updates.
    """
    
    def __init__(self, base_dir: str = None):
        """Initialize topic cache with base directory."""
        if base_dir is None:
            base_dir = BASE_TMP_DIR
        
        self.cache_dir = Path(base_dir) / 'topic_indexes'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Metadata file for cache management
        self.metadata_file = self.cache_dir / 'cache_metadata.json'
        self.metadata = self._load_metadata()
        
        logger.info(f"Topic index cache initialized at {self.cache_dir}")
    
    def _load_metadata(self) -> Dict:
        """Load cache metadata from file."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Error loading cache metadata: {e}")
        
        return {
            'created_at': datetime.now().isoformat(),
            'version': '1.0',
            'threads': {}
        }
    
    def _save_metadata(self):
        """Save cache metadata to file."""
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2)
        except IOError as e:
            logger.error(f"Error saving cache metadata: {e}")
    
    def _get_cache_file_path(self, thread_key: str) -> Path:
        """Get the cache file path for a thread."""
        # Sanitize thread key for filename
        safe_key = thread_key.replace('/', '_').replace('\\', '_')
        return self.cache_dir / f"{safe_key}_topic_index.json"
    
    def store_topic_index(self, thread_key: str, topic_index: Dict, metadata: Dict = None) -> bool:
        """
        Store a topic index for a thread.
        
        Args:
            thread_key: Unique identifier for the thread
            topic_index: Complete topic index data
            metadata: Optional metadata about the indexing process
            
        Returns:
            True if successfully stored, False otherwise
        """
        try:
            cache_file = self._get_cache_file_path(thread_key)
            
            # Prepare cache entry
            cache_entry = {
                'thread_key': thread_key,
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat(),
                'metadata': metadata or {},
                'topic_index': topic_index
            }
            
            # Write to cache file
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_entry, f, indent=2)
            
            # Update metadata
            self.metadata['threads'][thread_key] = {
                'cache_file': cache_file.name,
                'created_at': cache_entry['created_at'],
                'updated_at': cache_entry['updated_at'],
                'topic_count': len(topic_index.get('topics', {})),
                'total_matches': topic_index.get('thread_stats', {}).get('total_topic_matches', 0)
            }
            self._save_metadata()
            
            logger.info(f"Topic index stored for thread {thread_key}: {len(topic_index.get('topics', {}))} topics")
            return True
            
        except (IOError, json.JSONEncodeError) as e:
            logger.error(f"Error storing topic index for {thread_key}: {e}")
            return False
    
    def load_topic_index(self, thread_key: str) -> Optional[Dict]:
        """
        Load a topic index for a thread.
        
        Args:
            thread_key: Unique identifier for the thread
            
        Returns:
            Topic index data if found, None otherwise
        """
        try:
            cache_file = self._get_cache_file_path(thread_key)
            
            if not cache_file.exists():
                return None
            
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_entry = json.load(f)
            
            return cache_entry.get('topic_index')
            
        except (IOError, json.JSONDecodeError) as e:
            logger.error(f"Error loading topic index for {thread_key}: {e}")
            return None
    
    def has_topic_index(self, thread_key: str) -> bool:
        """
        Check if a topic index exists for a thread.
        
        Args:
            thread_key: Unique identifier for the thread
            
        Returns:
            True if index exists, False otherwise
        """
        return self._get_cache_file_path(thread_key).exists()
    
    def delete_topic_index(self, thread_key: str) -> bool:
        """
        Delete a topic index for a thread.
        
        Args:
            thread_key: Unique identifier for the thread
            
        Returns:
            True if successfully deleted, False otherwise
        """
        try:
            cache_file = self._get_cache_file_path(thread_key)
            
            if cache_file.exists():
                cache_file.unlink()
            
            # Remove from metadata
            if thread_key in self.metadata['threads']:
                del self.metadata['threads'][thread_key]
                self._save_metadata()
            
            logger.info(f"Topic index deleted for thread {thread_key}")
            return True
            
        except OSError as e:
            logger.error(f"Error deleting topic index for {thread_key}: {e}")
            return False
    
    def get_topic_matches_for_thread(self, thread_key: str, topic_id: str = None) -> List[Dict]:
        """
        Get topic matches for a specific thread and optionally a specific topic.
        
        Args:
            thread_key: Unique identifier for the thread
            topic_id: Optional specific topic to filter by
            
        Returns:
            List of topic match dictionaries
        """
        topic_index = self.load_topic_index(thread_key)
        if not topic_index:
            return []
        
        matches = []
        topics_data = topic_index.get('topics', {})
        
        if topic_id:
            # Get matches for specific topic
            topic_data = topics_data.get(topic_id, {})
            matches = topic_data.get('matches', [])
        else:
            # Get all matches
            for topic_data in topics_data.values():
                matches.extend(topic_data.get('matches', []))
        
        return matches
    
    def get_thread_topic_summary(self, thread_key: str) -> Optional[Dict]:
        """
        Get a summary of topics found in a thread.
        
        Args:
            thread_key: Unique identifier for the thread
            
        Returns:
            Dictionary with topic summaries, or None if not found
        """
        topic_index = self.load_topic_index(thread_key)
        if not topic_index:
            return None
        
        summary = {
            'thread_key': thread_key,
            'stats': topic_index.get('thread_stats', {}),
            'topics': {}
        }
        
        topics_data = topic_index.get('topics', {})
        for topic_id, topic_data in topics_data.items():
            topic_summary = topic_data.get('summary', {})
            summary['topics'][topic_id] = {
                'display_name': topic_summary.get('display_name', topic_id),
                'post_count': topic_summary.get('post_count', 0),
                'page_range': topic_summary.get('page_range', [0, 0]),
                'avg_score': topic_summary.get('avg_score', 0),
                'top_contributors': topic_summary.get('top_contributors', [])[:3]  # Top 3
            }
        
        return summary
    
    def search_topics_across_threads(self, topic_id: str, limit: int = 50) -> List[Dict]:
        """
        Search for a specific topic across all cached threads.
        
        Args:
            topic_id: Topic to search for
            limit: Maximum number of matches to return
            
        Returns:
            List of matches from all threads
        """
        all_matches = []
        
        for thread_key in self.metadata.get('threads', {}).keys():
            matches = self.get_topic_matches_for_thread(thread_key, topic_id)
            for match in matches:
                match['thread_key'] = thread_key  # Add thread context
                all_matches.append(match)
        
        # Sort by relevance score
        all_matches.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        return all_matches[:limit]
    
    def get_cache_stats(self) -> Dict:
        """
        Get statistics about the topic index cache.
        
        Returns:
            Dictionary with cache statistics
        """
        stats = {
            'total_threads': len(self.metadata.get('threads', {})),
            'cache_size_bytes': 0,
            'total_topics': 0,
            'total_matches': 0,
            'threads': []
        }
        
        for thread_key, thread_metadata in self.metadata.get('threads', {}).items():
            cache_file = self.cache_dir / thread_metadata['cache_file']
            file_size = cache_file.stat().st_size if cache_file.exists() else 0
            
            stats['cache_size_bytes'] += file_size
            stats['total_topics'] += thread_metadata.get('topic_count', 0)
            stats['total_matches'] += thread_metadata.get('total_matches', 0)
            
            stats['threads'].append({
                'thread_key': thread_key,
                'topic_count': thread_metadata.get('topic_count', 0),
                'total_matches': thread_metadata.get('total_matches', 0),
                'updated_at': thread_metadata.get('updated_at', ''),
                'size_bytes': file_size
            })
        
        return stats
    
    def cleanup_orphaned_files(self) -> int:
        """
        Clean up cache files that are not referenced in metadata.
        
        Returns:
            Number of files cleaned up
        """
        referenced_files = set()
        for thread_metadata in self.metadata.get('threads', {}).values():
            referenced_files.add(thread_metadata['cache_file'])
        
        cleaned_count = 0
        for cache_file in self.cache_dir.glob('*_topic_index.json'):
            if cache_file.name not in referenced_files:
                try:
                    cache_file.unlink()
                    cleaned_count += 1
                    logger.info(f"Cleaned up orphaned cache file: {cache_file.name}")
                except OSError as e:
                    logger.warning(f"Error cleaning up {cache_file.name}: {e}")
        
        return cleaned_count