"""
Post processing functionality for Forum Wisdom Miner.

This module handles processing of individual forum posts including
deduplication, filtering, and enhancement.
"""

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from config.settings import MIN_POST_LENGTH, MAX_POST_LENGTH
from utils.helpers import post_hash
from utils.text_utils import clean_post_content, get_text_statistics

logger = logging.getLogger(__name__)


class PostProcessor:
    """Processes and filters forum posts."""
    
    def __init__(self):
        """Initialize the post processor."""
        self.stats = {
            'total_processed': 0,
            'duplicates_removed': 0,
            'filtered_out': 0,
            'enhanced_posts': 0
        }
    
    def process_posts(self, raw_posts: List[Dict]) -> Tuple[List[Dict], Dict]:
        """Process a list of raw forum posts.
        
        Args:
            raw_posts: List of raw post dictionaries
            
        Returns:
            Tuple of (processed_posts, processing_stats)
        """
        logger.info(f"Processing {len(raw_posts)} raw posts")
        
        # Reset stats
        self.stats = {
            'total_processed': 0,
            'duplicates_removed': 0,
            'filtered_out': 0,
            'enhanced_posts': 0
        }
        
        # Step 1: Clean and normalize posts
        cleaned_posts = self._clean_posts(raw_posts)
        
        # Step 2: Remove duplicates
        deduplicated_posts = self._remove_duplicates(cleaned_posts)
        
        # Step 3: Filter posts
        filtered_posts = self._filter_posts(deduplicated_posts)
        
        # Step 4: Enhance posts with additional metadata
        enhanced_posts = self._enhance_posts(filtered_posts)
        
        # Step 5: Sort by position
        sorted_posts = sorted(enhanced_posts, key=lambda p: p.get('global_position', 0))
        
        self.stats['total_processed'] = len(raw_posts)
        
        logger.info(f"Post processing complete: {len(sorted_posts)} posts remain after processing")
        
        return sorted_posts, self.stats.copy()
    
    def _clean_posts(self, posts: List[Dict]) -> List[Dict]:
        """Clean post content and normalize data."""
        cleaned_posts = []
        
        for post in posts:
            try:
                # Clean content
                raw_content = post.get('content', '')
                clean_content = clean_post_content(raw_content)
                
                if not clean_content or len(clean_content.strip()) < 5:
                    continue
                
                # Create cleaned post
                cleaned_post = {
                    'content': clean_content,
                    'author': str(post.get('author', 'unknown-author')).strip(),
                    'date': str(post.get('date', 'unknown-date')).strip(),
                    'page': int(post.get('page', 1)),
                    'position_on_page': int(post.get('position_on_page', 0)),
                    'global_position': int(post.get('global_position', 0)),
                    'url': str(post.get('url', '')).strip(),
                    'raw_content': raw_content,  # Keep original for reference
                }
                
                # Generate hash
                cleaned_post['hash'] = post_hash(
                    clean_content, 
                    cleaned_post['author'], 
                    cleaned_post['date']
                )
                
                cleaned_posts.append(cleaned_post)
                
            except Exception as e:
                logger.warning(f"Error cleaning post: {e}")
                continue
        
        logger.debug(f"Cleaned {len(cleaned_posts)} posts from {len(posts)} raw posts")
        return cleaned_posts
    
    def _remove_duplicates(self, posts: List[Dict]) -> List[Dict]:
        """Remove duplicate posts based on content similarity and hash."""
        seen_hashes = set()
        seen_content = set()
        deduplicated = []
        
        for post in posts:
            post_hash_val = post['hash']
            content = post['content']
            
            # Check exact hash match
            if post_hash_val in seen_hashes:
                self.stats['duplicates_removed'] += 1
                continue
            
            # Check content similarity (simple approach)
            content_normalized = self._normalize_for_dedup(content)
            if content_normalized in seen_content:
                self.stats['duplicates_removed'] += 1
                continue
            
            # This is a unique post
            seen_hashes.add(post_hash_val)
            seen_content.add(content_normalized)
            deduplicated.append(post)
        
        logger.debug(f"Removed {self.stats['duplicates_removed']} duplicate posts")
        return deduplicated
    
    def _normalize_for_dedup(self, content: str) -> str:
        """Normalize content for duplicate detection."""
        # Remove extra whitespace and convert to lowercase
        normalized = ' '.join(content.lower().split())
        
        # Remove common forum artifacts
        normalized = normalized.replace('click to expand', '')
        normalized = normalized.replace('...', '')
        
        return normalized
    
    def _filter_posts(self, posts: List[Dict]) -> List[Dict]:
        """Filter posts based on quality criteria."""
        filtered_posts = []
        
        for post in posts:
            if self._should_filter_post(post):
                self.stats['filtered_out'] += 1
                continue
            
            filtered_posts.append(post)
        
        logger.debug(f"Filtered out {self.stats['filtered_out']} low-quality posts")
        return filtered_posts
    
    def _should_filter_post(self, post: Dict) -> bool:
        """Determine if a post should be filtered out."""
        content = post['content']
        
        # Length checks
        if len(content) < MIN_POST_LENGTH:
            return True
        
        if len(content) > MAX_POST_LENGTH:
            return True
        
        # Content quality checks
        if self._is_low_quality_content(content):
            return True
        
        # Author checks
        author = post['author'].lower()
        if author in ['deleted', 'banned', 'guest', '[deleted]']:
            return True
        
        return False
    
    def _is_low_quality_content(self, content: str) -> bool:
        """Check if content is low quality."""
        content_lower = content.lower()
        
        # Check for common low-quality patterns
        low_quality_patterns = [
            'deleted by moderator',
            'this post has been removed',
            'user has been banned',
            '+1',
            'me too',
            'same here',
            'this',
            '^',
            'bump',
        ]
        
        for pattern in low_quality_patterns:
            if pattern in content_lower:
                return True
        
        # Check ratio of letters to total characters
        letter_count = sum(1 for c in content if c.isalpha())
        if len(content) > 0:
            letter_ratio = letter_count / len(content)
            if letter_ratio < 0.5:  # Less than 50% letters
                return True
        
        # Check for excessive repetition
        words = content.split()
        if len(words) > 0:
            unique_words = set(words)
            if len(unique_words) / len(words) < 0.3:  # Less than 30% unique words
                return True
        
        return False
    
    def _enhance_posts(self, posts: List[Dict]) -> List[Dict]:
        """Enhance posts with additional metadata."""
        enhanced_posts = []
        
        for post in posts:
            try:
                enhanced_post = post.copy()
                
                # Add text statistics
                text_stats = get_text_statistics(post['content'])
                enhanced_post['text_stats'] = text_stats
                
                # Add processing metadata
                enhanced_post['processed_at'] = 'current_timestamp'  # Would use actual timestamp
                enhanced_post['content_length'] = len(post['content'])
                enhanced_post['word_count'] = text_stats.get('words', 0)
                
                # Add content type hints
                enhanced_post['content_type'] = self._classify_content_type(post['content'])
                
                # Add author activity level (simplified)
                enhanced_post['author_activity'] = self._estimate_author_activity(post['author'], posts)
                
                enhanced_posts.append(enhanced_post)
                self.stats['enhanced_posts'] += 1
                
            except Exception as e:
                logger.warning(f"Error enhancing post: {e}")
                enhanced_posts.append(post)  # Fallback to unenhanced
        
        logger.debug(f"Enhanced {self.stats['enhanced_posts']} posts with metadata")
        return enhanced_posts
    
    def _classify_content_type(self, content: str) -> str:
        """Classify the type of content in a post."""
        content_lower = content.lower()
        
        # Question indicators
        if any(word in content_lower for word in ['?', 'how', 'what', 'why', 'when', 'where']):
            return 'question'
        
        # Answer/solution indicators
        if any(word in content_lower for word in ['solution', 'answer', 'solved', 'try this']):
            return 'solution'
        
        # Opinion indicators
        if any(word in content_lower for word in ['think', 'believe', 'opinion', 'feel', 'imho']):
            return 'opinion'
        
        # Information sharing
        if any(word in content_lower for word in ['fyi', 'info', 'according to', 'source']):
            return 'information'
        
        return 'discussion'
    
    def _estimate_author_activity(self, author: str, all_posts: List[Dict]) -> str:
        """Estimate author activity level based on post count."""
        author_posts = sum(1 for post in all_posts if post['author'] == author)
        
        if author_posts >= 10:
            return 'very_active'
        elif author_posts >= 5:
            return 'active'
        elif author_posts >= 2:
            return 'moderate'
        else:
            return 'casual'
    
    def validate_posts(self, posts: List[Dict]) -> Tuple[bool, List[str]]:
        """Validate processed posts for consistency.
        
        Args:
            posts: List of processed posts
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check for required fields
        required_fields = ['content', 'author', 'date', 'hash', 'global_position']
        for i, post in enumerate(posts):
            for field in required_fields:
                if field not in post:
                    errors.append(f"Post {i}: Missing required field '{field}'")
        
        # Check for duplicate hashes
        hashes = [post.get('hash') for post in posts if post.get('hash')]
        if len(hashes) != len(set(hashes)):
            errors.append("Duplicate hashes found in posts")
        
        # Check position ordering
        positions = [post.get('global_position', 0) for post in posts]
        if positions != sorted(positions):
            errors.append("Posts are not ordered by global_position")
        
        # Check content quality
        empty_content = sum(1 for post in posts if not post.get('content', '').strip())
        if empty_content > 0:
            errors.append(f"{empty_content} posts have empty content")
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    def get_processing_summary(self, original_count: int, final_count: int) -> Dict:
        """Get a summary of the processing operation."""
        return {
            'original_posts': original_count,
            'final_posts': final_count,
            'posts_removed': original_count - final_count,
            'removal_rate': (original_count - final_count) / max(1, original_count),
            'stats': self.stats.copy()
        }


__all__ = ['PostProcessor']