"""
Analytical data processing for Forum Wisdom Miner.

This module handles queries that require data aggregation and statistical analysis
rather than semantic search, such as "who is the most active user".
"""

import logging
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Any
from datetime import datetime

from utils.file_utils import safe_read_json

logger = logging.getLogger(__name__)


class ForumDataAnalyzer:
    """Analyzes forum data to answer statistical and analytical queries."""
    
    def __init__(self, thread_dir: str):
        """Initialize the data analyzer.
        
        Args:
            thread_dir: Directory containing thread data
        """
        self.thread_dir = thread_dir
        self.posts_cache = None
        self.analytics_cache = None
    
    def _load_posts(self) -> List[Dict]:
        """Load all posts from the thread."""
        if self.posts_cache is None:
            posts_file = f"{self.thread_dir}/posts.json"
            self.posts_cache = safe_read_json(posts_file) or []
            logger.info(f"Loaded {len(self.posts_cache)} posts for analysis")
        return self.posts_cache
    
    def _load_analytics(self) -> Dict:
        """Load thread analytics if available."""
        if self.analytics_cache is None:
            analytics_file = f"{self.thread_dir}/thread_analytics.json"
            self.analytics_cache = safe_read_json(analytics_file) or {}
        return self.analytics_cache
    
    def analyze_participant_activity(self, query: str) -> Dict[str, Any]:
        """Analyze participant activity patterns.
        
        Args:
            query: The original query
            
        Returns:
            Analysis results with participant activity data
        """
        posts = self._load_posts()
        
        if not posts:
            return {
                'error': 'No posts available for analysis',
                'type': 'participant_analysis'
            }
        
        # Count posts by author
        author_counts = Counter()
        author_dates = defaultdict(list)
        author_scores = defaultdict(int)
        
        for post in posts:
            author = post.get('author', 'Unknown')
            if author and author.lower() not in ['unknown', 'deleted', 'guest']:
                author_counts[author] += 1
                
                # Track posting dates
                date_str = post.get('date', '')
                if date_str:
                    author_dates[author].append(date_str)
                
                # Track scores/reactions
                total_score = post.get('total_score', 0)
                upvotes = post.get('upvotes', 0)
                reactions = post.get('reactions', 0)
                likes = post.get('likes', 0)
                
                author_scores[author] += total_score + upvotes + reactions + likes
        
        # Find most active author
        if not author_counts:
            return {
                'error': 'No valid authors found in thread',
                'type': 'participant_analysis'
            }
        
        most_active_author = author_counts.most_common(1)[0]
        top_5_authors = author_counts.most_common(5)
        
        # Calculate additional metrics
        total_posts = len(posts)
        unique_authors = len(author_counts)
        avg_posts_per_author = total_posts / unique_authors if unique_authors > 0 else 0
        
        result = {
            'type': 'participant_analysis',
            'query': query,
            'most_active_author': {
                'name': most_active_author[0],
                'post_count': most_active_author[1],
                'percentage': (most_active_author[1] / total_posts) * 100,
                'total_score': author_scores.get(most_active_author[0], 0)
            },
            'top_authors': [
                {
                    'name': author,
                    'post_count': count,
                    'percentage': (count / total_posts) * 100,
                    'total_score': author_scores.get(author, 0)
                }
                for author, count in top_5_authors
            ],
            'thread_stats': {
                'total_posts': total_posts,
                'unique_authors': unique_authors,
                'average_posts_per_author': round(avg_posts_per_author, 1)
            },
            'confidence': 1.0  # High confidence for data-based analysis
        }
        
        return result
    
    def analyze_content_statistics(self, query: str) -> Dict[str, Any]:
        """Analyze content statistics like post counts, lengths, etc.
        
        Args:
            query: The original query
            
        Returns:
            Statistical analysis results
        """
        posts = self._load_posts()
        
        if not posts:
            return {
                'error': 'No posts available for analysis',
                'type': 'content_statistics'
            }
        
        # Calculate statistics
        post_lengths = [len(post.get('content', '')) for post in posts]
        word_counts = [post.get('word_count', 0) for post in posts if post.get('word_count')]
        
        # Page distribution
        page_counts = Counter(post.get('page', 1) for post in posts)
        
        # Date analysis (if available)
        dates_available = sum(1 for post in posts if post.get('parsed_date'))
        
        result = {
            'type': 'content_statistics',
            'query': query,
            'post_statistics': {
                'total_posts': len(posts),
                'average_length': round(sum(post_lengths) / len(post_lengths), 1) if post_lengths else 0,
                'shortest_post': min(post_lengths) if post_lengths else 0,
                'longest_post': max(post_lengths) if post_lengths else 0,
                'average_word_count': round(sum(word_counts) / len(word_counts), 1) if word_counts else 0
            },
            'page_distribution': {
                'total_pages': len(page_counts),
                'posts_per_page': dict(page_counts.most_common(10))
            },
            'temporal_coverage': {
                'posts_with_dates': dates_available,
                'date_coverage_percentage': (dates_available / len(posts)) * 100 if posts else 0
            },
            'confidence': 1.0
        }
        
        return result
    
    def analyze_temporal_patterns(self, query: str) -> Dict[str, Any]:
        """Analyze temporal patterns in the thread.
        
        Args:
            query: The original query
            
        Returns:
            Temporal analysis results
        """
        posts = self._load_posts()
        
        if not posts:
            return {
                'error': 'No posts available for analysis',
                'type': 'temporal_analysis'
            }
        
        # Extract dates and sort posts
        dated_posts = []
        for post in posts:
            parsed_date = post.get('parsed_date')
            if parsed_date and isinstance(parsed_date, datetime):
                dated_posts.append((parsed_date, post))
        
        if not dated_posts:
            return {
                'error': 'No posts with valid dates found for temporal analysis',
                'type': 'temporal_analysis'
            }
        
        dated_posts.sort(key=lambda x: x[0])
        
        # Calculate temporal metrics
        first_post_date = dated_posts[0][0]
        last_post_date = dated_posts[-1][0]
        thread_duration = (last_post_date - first_post_date).days
        
        # Monthly activity
        monthly_counts = defaultdict(int)
        for date, post in dated_posts:
            month_key = f"{date.year}-{date.month:02d}"
            monthly_counts[month_key] += 1
        
        result = {
            'type': 'temporal_analysis',
            'query': query,
            'thread_timeline': {
                'first_post': first_post_date.strftime('%Y-%m-%d'),
                'last_post': last_post_date.strftime('%Y-%m-%d'),
                'duration_days': thread_duration,
                'posts_with_dates': len(dated_posts),
                'coverage_percentage': (len(dated_posts) / len(posts)) * 100
            },
            'activity_pattern': {
                'posts_per_month': dict(sorted(monthly_counts.items())),
                'most_active_month': max(monthly_counts.items(), key=lambda x: x[1]) if monthly_counts else None,
                'average_posts_per_day': len(dated_posts) / max(thread_duration, 1)
            },
            'confidence': 0.9
        }
        
        return result
    
    def analyze_positional_queries(self, query: str) -> Dict[str, Any]:
        """Analyze positional queries like 'who was the second user to post'.
        
        Args:
            query: The original query
            
        Returns:
            Analysis results with positional data
        """
        posts = self._load_posts()
        
        if not posts:
            return {
                'error': 'No posts available for analysis',
                'type': 'positional_analysis'
            }
        
        # Sort posts by position (global_position or index)
        sorted_posts = sorted(posts, key=lambda x: x.get('global_position', 0))
        
        query_lower = query.lower()
        
        # Extract ordinal number from query
        ordinal_map = {
            'first': 1, 'second': 2, 'third': 3, 'fourth': 4, 'fifth': 5,
            '1st': 1, '2nd': 2, '3rd': 3, '4th': 4, '5th': 5,
            'earliest': 1, 'initial': 1
        }
        
        position = 1  # default to first
        for ordinal, num in ordinal_map.items():
            if ordinal in query_lower:
                position = num
                break
        
        # Get unique authors in posting order
        unique_authors = []
        seen_authors = set()
        
        for post in sorted_posts:
            author = post.get('author', 'Unknown')
            if author and author.lower() not in ['unknown', 'deleted', 'guest']:
                if author not in seen_authors:
                    unique_authors.append(author)
                    seen_authors.add(author)
        
        # Get the requested position
        if position <= len(unique_authors):
            target_author = unique_authors[position - 1]
            
            # Find their first post for additional info
            first_post = None
            for post in sorted_posts:
                if post.get('author') == target_author:
                    first_post = post
                    break
            
            result = {
                'type': 'positional_analysis',
                'query': query,
                'position': position,
                'author': target_author,
                'total_unique_authors': len(unique_authors),
                'confidence': 0.95
            }
            
            if first_post:
                result.update({
                    'first_post_date': first_post.get('date', 'Unknown'),
                    'first_post_content': first_post.get('content', '')[:200] + '...' if len(first_post.get('content', '')) > 200 else first_post.get('content', ''),
                    'post_position': first_post.get('global_position', 0),
                    'post_url': first_post.get('url', ''),
                    'post_id': first_post.get('post_id', ''),
                    'page_number': first_post.get('page', 0)
                })
            
            return result
        else:
            return {
                'type': 'positional_analysis',
                'query': query,
                'error': f'Only {len(unique_authors)} unique authors found in thread (requested position: {position})',
                'total_unique_authors': len(unique_authors),
                'confidence': 0.9
            }
    
    def analyze_engagement_queries(self, query: str) -> Dict[str, Any]:
        """Analyze engagement-based queries like 'highest rated post', 'most popular post', etc.
        
        Args:
            query: The original query
            
        Returns:
            Analysis results with engagement data
        """
        posts = self._load_posts()
        
        if not posts:
            return {
                'error': 'No posts available for analysis',
                'type': 'engagement_analysis'
            }
        
        query_lower = query.lower()
        
        # Determine what engagement metric to analyze
        if any(term in query_lower for term in ['highest rated', 'most rated', 'top rated', 'best rated']):
            metric_type = 'total_score'
            sort_desc = True
            metric_name = 'highest rated'
        elif any(term in query_lower for term in ['most upvoted', 'most upvotes', 'top upvoted']):
            metric_type = 'upvotes'
            sort_desc = True
            metric_name = 'most upvoted'
        elif any(term in query_lower for term in ['most liked', 'most likes', 'top liked']):
            metric_type = 'likes'
            sort_desc = True
            metric_name = 'most liked'
        elif any(term in query_lower for term in ['most reactions', 'most reacted', 'top reactions']):
            metric_type = 'reactions'
            sort_desc = True
            metric_name = 'most reactions'
        elif any(term in query_lower for term in ['most popular', 'most engaged', 'top engagement']):
            # Calculate combined engagement score
            metric_type = 'combined_engagement'
            sort_desc = True
            metric_name = 'most popular'
        elif any(term in query_lower for term in ['lowest rated', 'least rated', 'worst rated']):
            metric_type = 'total_score'
            sort_desc = False
            metric_name = 'lowest rated'
        else:
            # Default to highest rated for general queries
            metric_type = 'total_score'
            sort_desc = True
            metric_name = 'highest rated'
        
        # Calculate engagement scores for each post
        scored_posts = []
        for post in posts:
            if metric_type == 'combined_engagement':
                # Combined score: upvotes + likes + reactions + (total_score * 2)
                score = (
                    post.get('upvotes', 0) + 
                    post.get('likes', 0) + 
                    post.get('reactions', 0) + 
                    (post.get('total_score', 0) * 2)
                )
            else:
                score = post.get(metric_type, 0)
            
            if score > 0 or not sort_desc:  # Include posts with engagement or if looking for lowest
                scored_posts.append({
                    'post': post,
                    'score': score,
                    'author': post.get('author', 'Unknown'),
                    'content_preview': post.get('content', '')[:200] + '...' if len(post.get('content', '')) > 200 else post.get('content', ''),
                    'date': post.get('date', 'Unknown'),
                    'page': post.get('page', 0),
                    'global_position': post.get('global_position', 0),
                    'post_url': post.get('url', ''),
                    'post_id': post.get('post_id', ''),
                    'upvotes': post.get('upvotes', 0),
                    'downvotes': post.get('downvotes', 0),
                    'likes': post.get('likes', 0),
                    'reactions': post.get('reactions', 0),
                    'total_score': post.get('total_score', 0)
                })
        
        if not scored_posts:
            return {
                'type': 'engagement_analysis',
                'query': query,
                'error': f'No posts found with {metric_name} data',
                'metric_type': metric_type,
                'confidence': 0.8
            }
        
        # Sort by engagement score
        scored_posts.sort(key=lambda x: x['score'], reverse=sort_desc)
        
        # Get top result and additional context
        top_post = scored_posts[0]
        top_5_posts = scored_posts[:5]
        
        result = {
            'type': 'engagement_analysis',
            'query': query,
            'metric_type': metric_type,
            'metric_name': metric_name,
            'top_post': top_post,
            'top_5_posts': top_5_posts,
            'total_posts_with_engagement': len(scored_posts),
            'total_posts_analyzed': len(posts),
            'confidence': 0.95
        }
        
        return result
    
    def can_handle_query(self, query: str, analytical_intent: List[str]) -> bool:
        """Determine if this analyzer can handle the given query.
        
        Args:
            query: The user's query
            analytical_intent: Detected analytical intents
            
        Returns:
            True if this analyzer can handle the query
        """
        query_lower = query.lower()
        
        # Participant activity queries
        participant_indicators = [
            'most active', 'who posted', 'top contributor', 'most posts',
            'who is', 'active user', 'frequent poster', 'main contributor'
        ]
        
        # Statistical queries
        stats_indicators = [
            'how many', 'count', 'number of', 'total posts', 'statistics',
            'post count', 'how much', 'volume of'
        ]
        
        # Temporal queries
        temporal_indicators = [
            'when', 'timeline', 'over time', 'chronology', 'first post',
            'last post', 'duration', 'activity pattern'
        ]
        
        # Positional/ordinal queries
        positional_indicators = [
            'first user', 'second user', 'third user', 'first poster', 'second poster',
            'who was the first', 'who was the second', 'who posted first', 'who posted second',
            'earliest user', 'initial poster', 'second to post', 'third to post'
        ]
        
        # Engagement/rating queries (NEW - captures post engagement metrics)
        engagement_indicators = [
            'highest rated', 'most rated', 'top rated', 'best rated', 'most popular',
            'most upvoted', 'most upvotes', 'top upvoted', 'most liked', 'most likes',
            'most reactions', 'most reacted', 'top reactions', 'most engaged',
            'top engagement', 'best post', 'top post', 'popular post', 'highest scoring',
            'lowest rated', 'least rated', 'worst rated', 'least popular', 'least liked'
        ]
        
        # Check if we can handle this query
        if any(indicator in query_lower for indicator in participant_indicators):
            return True
        
        if any(indicator in query_lower for indicator in stats_indicators):
            return True
        
        if any(indicator in query_lower for indicator in temporal_indicators):
            return True
            
        if any(indicator in query_lower for indicator in positional_indicators):
            return True
            
        if any(indicator in query_lower for indicator in engagement_indicators):
            return True
        
        # Check analytical intents
        if 'participants' in analytical_intent:
            return True
        
        if 'statistics' in analytical_intent:
            return True
        
        if 'timeline' in analytical_intent:
            return True
        
        return False
    
    def analyze_query(self, query: str, analytical_intent: List[str]) -> Dict[str, Any]:
        """Analyze a query and return appropriate results.
        
        Args:
            query: The user's query
            analytical_intent: Detected analytical intents
            
        Returns:
            Analysis results
        """
        query_lower = query.lower()
        
        # Route to appropriate analysis method
        # Check for engagement queries first (most specific - NEW)
        if any(indicator in query_lower for indicator in [
            'highest rated', 'most rated', 'top rated', 'best rated', 'most popular',
            'most upvoted', 'most upvotes', 'top upvoted', 'most liked', 'most likes',
            'most reactions', 'most reacted', 'top reactions', 'most engaged',
            'top engagement', 'best post', 'top post', 'popular post', 'highest scoring',
            'lowest rated', 'least rated', 'worst rated', 'least popular', 'least liked'
        ]):
            return self.analyze_engagement_queries(query)
        
        # Check for positional queries second (specific)
        elif any(indicator in query_lower for indicator in [
            'first user', 'second user', 'third user', 'first poster', 'second poster',
            'who was the first', 'who was the second', 'who posted first', 'who posted second',
            'earliest user', 'initial poster', 'second to post', 'third to post'
        ]):
            return self.analyze_positional_queries(query)
        
        elif any(indicator in query_lower for indicator in [
            'most active', 'who posted', 'top contributor', 'most posts',
            'who is', 'active user', 'frequent poster'
        ]) or 'participants' in analytical_intent:
            return self.analyze_participant_activity(query)
        
        elif any(indicator in query_lower for indicator in [
            'how many', 'count', 'number of', 'total posts', 'statistics'
        ]) or 'statistics' in analytical_intent:
            return self.analyze_content_statistics(query)
        
        elif any(indicator in query_lower for indicator in [
            'when', 'timeline', 'over time', 'chronology'
        ]) or 'timeline' in analytical_intent:
            return self.analyze_temporal_patterns(query)
        
        else:
            # Fallback to participant analysis for other participant queries
            return self.analyze_participant_activity(query)


__all__ = ['ForumDataAnalyzer']