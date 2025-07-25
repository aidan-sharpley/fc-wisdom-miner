"""
Automatic analytics generation for thread selection.
Optimized for speed and memory efficiency on M1 MacBook Air.
"""

import json
import logging
import os
import time
from typing import Dict, List, Optional, Any

from analytics.thread_analyzer import ThreadAnalyzer
from analytics.thread_narrative import ThreadNarrative
from utils.shared_data_manager import get_data_manager
from utils.memory_optimizer import memory_efficient
from utils.file_utils import atomic_write_json

logger = logging.getLogger(__name__)


class AutoAnalyticsGenerator:
    """Generates analytics and summaries automatically on thread access."""
    
    def __init__(self):
        self._generation_cache = {}
        self._max_cache_age = 3600  # 1 hour
    
    @memory_efficient
    def generate_thread_analytics(self, thread_key: str, thread_dir: str, force_refresh: bool = False) -> Dict[str, Any]:
        """Generate comprehensive analytics for a thread."""
        cache_key = f"{thread_key}_analytics"
        
        # Check cache first
        if not force_refresh and cache_key in self._generation_cache:
            cached_data, timestamp = self._generation_cache[cache_key]
            if time.time() - timestamp < self._max_cache_age:
                return cached_data
        
        start_time = time.time()
        data_manager = get_data_manager(thread_dir)
        
        # Check existing files
        analytics_file = os.path.join(thread_dir, "thread_analytics.json")
        summary_file = os.path.join(thread_dir, "thread_summary.json")
        
        needs_analytics = force_refresh or not os.path.exists(analytics_file)
        needs_summary = force_refresh or not os.path.exists(summary_file)
        
        if not needs_analytics and not needs_summary:
            # Load existing data
            analytics = data_manager.get_analytics()
            summary = data_manager.get_summary()
            result = {
                'analytics': analytics,
                'summary': summary,
                'generated': False,
                'cached': True
            }
        else:
            # Generate missing data
            posts = data_manager.get_posts()
            if not posts:
                logger.warning(f"No posts found for thread {thread_key}")
                return {'error': 'No posts found'}
            
            result = {'generated': True, 'cached': False}
            
            # Generate analytics if needed
            if needs_analytics:
                logger.info(f"Generating analytics for thread {thread_key}")
                analyzer = ThreadAnalyzer(thread_dir)
                analytics = analyzer.analyze_thread(posts, force_refresh=True)
                result['analytics'] = analytics
            else:
                result['analytics'] = data_manager.get_analytics()
            
            # Generate summary if needed
            if needs_summary:
                logger.info(f"Generating summary for thread {thread_key}")
                narrative_gen = ThreadNarrative()
                summary_data = narrative_gen.generate_narrative_and_analytics(thread_dir, posts)
                result['summary'] = summary_data
            else:
                result['summary'] = data_manager.get_summary()
        
        # Cache result
        generation_time = time.time() - start_time
        result['generation_time'] = generation_time
        self._generation_cache[cache_key] = (result, time.time())
        
        logger.info(f"Analytics generation completed for {thread_key} in {generation_time:.2f}s")
        return result
    
    def get_visual_analytics_data(self, thread_key: str, thread_dir: str) -> Dict[str, Any]:
        """Generate data for visual analytics display."""
        data_manager = get_data_manager(thread_dir)
        analytics = data_manager.get_analytics()
        posts = data_manager.get_posts()
        
        if not analytics or not posts:
            return {'error': 'Analytics or posts not available'}
        
        # Extract data for visualizations
        visual_data = {
            'timeline': self._extract_timeline_data(posts, analytics),
            'activity': self._extract_activity_data(analytics),
            'engagement': self._extract_engagement_data(posts),
            'participants': self._extract_participant_data(analytics),
            'content_stats': self._extract_content_stats(analytics)
        }
        
        return visual_data
    
    def _extract_timeline_data(self, posts: List[Dict], analytics: Dict) -> List[Dict]:
        """Extract timeline data for visualization."""
        timeline_data = []
        
        # Group posts by date
        daily_counts = {}
        for post in posts:
            date_str = post.get('date', '').split(' ')[0]  # Get date part
            if date_str:
                daily_counts[date_str] = daily_counts.get(date_str, 0) + 1
        
        # Convert to timeline format
        for date, count in sorted(daily_counts.items()):
            timeline_data.append({
                'date': date,
                'posts': count
            })
        
        return timeline_data[-30:]  # Last 30 days
    
    def _extract_activity_data(self, analytics: Dict) -> Dict[str, Any]:
        """Extract activity data from analytics."""
        activity = analytics.get('activity', {})
        return {
            'most_active_users': activity.get('top_authors', [])[:10],
            'posting_patterns': activity.get('hourly_distribution', {}),
            'peak_activity': activity.get('peak_periods', [])
        }
    
    def _extract_engagement_data(self, posts: List[Dict]) -> Dict[str, Any]:
        """Extract engagement metrics."""
        total_votes = 0
        high_engagement_posts = []
        
        for post in posts:
            votes = post.get('votes', {})
            upvotes = votes.get('upvotes', 0)
            reactions = post.get('reactions', 0)
            
            total_engagement = upvotes + reactions
            total_votes += total_engagement
            
            if total_engagement > 5:  # High engagement threshold
                high_engagement_posts.append({
                    'id': post.get('id', ''),
                    'author': post.get('author', ''),
                    'engagement': total_engagement,
                    'preview': post.get('content', '')[:100]
                })
        
        return {
            'total_engagement': total_votes,
            'high_engagement_posts': sorted(high_engagement_posts, key=lambda x: x['engagement'], reverse=True)[:5],
            'engagement_distribution': self._calculate_engagement_distribution(posts)
        }
    
    def _extract_participant_data(self, analytics: Dict) -> Dict[str, Any]:
        """Extract participant statistics."""
        overview = analytics.get('overview', {})
        activity = analytics.get('activity', {})
        
        return {
            'total_participants': overview.get('participants', 0),
            'active_participants': len(activity.get('top_authors', [])),
            'thread_creator': overview.get('thread_creator', 'Unknown'),
            'participant_breakdown': activity.get('author_stats', {})
        }
    
    def _extract_content_stats(self, analytics: Dict) -> Dict[str, Any]:
        """Extract content statistics."""
        content = analytics.get('content', {})
        overview = analytics.get('overview', {})
        
        return {
            'total_posts': overview.get('total_posts', 0),
            'total_pages': overview.get('pages', 0),
            'avg_post_length': content.get('average_post_length', 0),
            'total_words': content.get('total_words', 0),
            'readability_score': content.get('average_readability', 0),
            'top_keywords': content.get('primary_keywords', [])[:10]
        }
    
    def _calculate_engagement_distribution(self, posts: List[Dict]) -> Dict[str, int]:
        """Calculate distribution of engagement levels."""
        distribution = {'none': 0, 'low': 0, 'medium': 0, 'high': 0}
        
        for post in posts:
            votes = post.get('votes', {})
            engagement = votes.get('upvotes', 0) + post.get('reactions', 0)
            
            if engagement == 0:
                distribution['none'] += 1
            elif engagement <= 2:
                distribution['low'] += 1
            elif engagement <= 5:
                distribution['medium'] += 1
            else:
                distribution['high'] += 1
        
        return distribution
    
    def cleanup_cache(self):
        """Clean up expired cache entries."""
        current_time = time.time()
        expired_keys = []
        
        for key, (data, timestamp) in self._generation_cache.items():
            if current_time - timestamp > self._max_cache_age:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self._generation_cache[key]


# Global instance
auto_analytics = AutoAnalyticsGenerator()

__all__ = ['AutoAnalyticsGenerator', 'auto_analytics']