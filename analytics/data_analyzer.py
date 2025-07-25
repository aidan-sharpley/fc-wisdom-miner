"""
Analytical data processing for Forum Wisdom Miner.

This module handles queries that require data aggregation and statistical analysis
rather than semantic search, such as "who is the most active user".
"""

import logging
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Any
from datetime import datetime

from utils.shared_data_manager import get_data_manager
from utils.memory_optimizer import memory_efficient

logger = logging.getLogger(__name__)


class ForumDataAnalyzer:
    """Memory-efficient forum data analyzer using shared data management."""
    
    def __init__(self, thread_dir: str):
        self.thread_dir = thread_dir
        self._data_manager = get_data_manager(thread_dir)
    
    @memory_efficient
    def _load_posts(self) -> List[Dict]:
        """Load posts via shared data manager."""
        posts = self._data_manager.get_posts()
        if posts:
            logger.info(f"Loaded {len(posts)} posts for analysis")
        return posts
    
    def _load_analytics(self) -> Dict:
        """Load analytics via shared data manager."""
        return self._data_manager.get_analytics()
    
    def _load_metadata(self) -> Dict:
        """Load metadata via shared data manager."""
        return self._data_manager.get_metadata()
    
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
    
    def analyze_thread_authorship(self, query: str) -> Dict[str, Any]:
        """Analyze thread authorship with metadata priority.
        
        Args:
            query: The original query
            
        Returns:
            Analysis results with thread creator information
        """
        analytics = self._load_analytics()
        
        # 1. Check URL-based thread creator (highest priority)
        thread_creator = analytics.get('metadata', {}).get('thread_creator')
        if thread_creator:
            username = thread_creator.get('username')
            return {
                'type': 'thread_authorship',
                'query': query,
                'answer': f"The thread author is {username}",
                'author': username,
                'source': thread_creator.get('source', 'canonical_url'),
                'confidence': 'high',
                'evidence': f"Thread URL contains author identifier: {username}",
                'extracted_from': thread_creator.get('extracted_from', '')
            }
        
        # 2. Fallback to first post author (medium priority)
        first_post = analytics.get('metadata', {}).get('first_post', {})
        first_post_author = first_post.get('author')
        if first_post_author:
            return {
                'type': 'thread_authorship', 
                'query': query,
                'answer': f"The thread author is {first_post_author} (based on first post)",
                'author': first_post_author,
                'source': 'first_post',
                'confidence': 'medium',
                'evidence': f"First post author: {first_post_author}",
                'post_date': first_post.get('date', ''),
                'post_position': first_post.get('position', 0)
            }
        
        # 3. Load posts as final fallback
        posts = self._load_posts()
        if posts:
            sorted_posts = sorted(posts, key=lambda x: x.get('global_position', 0))
            first_author = sorted_posts[0].get('author', 'Unknown')
            if first_author and first_author.lower() not in ['unknown', 'deleted', 'guest']:
                return {
                    'type': 'thread_authorship',
                    'query': query, 
                    'answer': f"The thread author is {first_author} (inferred from first post)",
                    'author': first_author,
                    'source': 'first_post_fallback',
                    'confidence': 'low',
                    'evidence': f"First post author from posts data: {first_author}"
                }
        
        return {
            'type': 'thread_authorship',
            'query': query,
            'answer': 'Thread author could not be determined',
            'error': 'No thread creator information available',
            'confidence': 'none'
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
        
        # Determine what engagement metric to analyze with smart vague query handling
        if any(term in query_lower for term in ['highest rated', 'most rated', 'top rated', 'best rated']):
            metric_type = 'total_score'
            sort_desc = True
            metric_name = 'highest rated'
        elif any(term in query_lower for term in ['rated', 'rating', 'score', 'scoring']) and len(query_lower.split()) <= 3:
            # Auto-interpret vague rating queries as "highest rated"
            metric_type = 'total_score'
            sort_desc = True
            metric_name = 'highest rated'
            logger.info(f"Auto-interpreting vague query '{query}' as 'highest rated post'")
        elif any(term in query_lower for term in ['best', 'good', 'great']) and len(query_lower.split()) <= 3:
            # Auto-interpret vague "best" queries as "highest rated"
            metric_type = 'total_score'
            sort_desc = True
            metric_name = 'highest rated'
            logger.info(f"Auto-interpreting vague query '{query}' as 'highest rated post'")
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
        
        # Thread authorship queries - should use metadata lookup
        thread_author_indicators = [
            'thread author', 'thread creator', 'who created', 'who started',
            'original poster', 'op', 'thread starter', 'who made this thread',
            'author of thread', 'creator of thread', 'thread originator',
            'who created this', 'who started this', 'thread op'
        ]
        
        # Engagement/rating queries (Enhanced - captures both explicit and vague engagement queries)
        engagement_indicators = [
            'highest rated', 'most rated', 'top rated', 'best rated', 'most popular',
            'most upvoted', 'most upvotes', 'top upvoted', 'most liked', 'most likes',
            'most reactions', 'most reacted', 'top reactions', 'most engaged',
            'top engagement', 'best post', 'top post', 'popular post', 'highest scoring',
            'lowest rated', 'least rated', 'worst rated', 'least popular', 'least liked',
            # Vague engagement queries that should be auto-detected
            'rated', 'rating', 'score', 'scoring', 'best', 'good', 'great', 'popular',
            'liked', 'favorite', 'well received', 'community favorite', 'highly rated',
            'upvoted', 'reactions', 'engagement', 'votes', 'voting'
        ]
        
        # Technical specification/settings queries (should find actual user data)
        technical_spec_indicators = [
            'what wattage', 'wattage setting', 'wattage do', 'power setting', 'watts',
            'what temperature', 'temp setting', 'temperature do', 'degrees', 'celsius', 'fahrenheit',
            'what voltage', 'voltage setting', 'volts', 'what resistance', 'ohms', 'resistance',
            'what settings', 'settings do', 'configuration', 'setup', 'how do people set',
            'what do people use', 'what do users set', 'community settings', 'recommended settings',
            'typical settings', 'common settings', 'standard settings', 'preferred settings'
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
            
        if any(indicator in query_lower for indicator in thread_author_indicators):
            return True
            
        if any(indicator in query_lower for indicator in engagement_indicators):
            return True
            
        if any(indicator in query_lower for indicator in technical_spec_indicators):
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
        # Check for engagement queries first (prioritized - includes vague queries)
        engagement_indicators = [
            'highest rated', 'most rated', 'top rated', 'best rated', 'most popular',
            'most upvoted', 'most upvotes', 'top upvoted', 'most liked', 'most likes',
            'most reactions', 'most reacted', 'top reactions', 'most engaged',
            'top engagement', 'best post', 'top post', 'popular post', 'highest scoring',
            'lowest rated', 'least rated', 'worst rated', 'least popular', 'least liked',
            # Vague engagement queries that should be auto-detected
            'rated', 'rating', 'score', 'scoring', 'best', 'good', 'great', 'popular',
            'liked', 'favorite', 'well received', 'community favorite', 'highly rated',
            'upvoted', 'reactions', 'engagement', 'votes', 'voting'
        ]
        
        # Smart engagement detection - prioritize even for vague queries
        engagement_detected = False
        for indicator in engagement_indicators:
            if indicator in query_lower:
                # Additional context checks for potentially ambiguous terms
                if indicator in ['best', 'good', 'great', 'popular'] and len(query_lower.split()) <= 3:
                    # For short vague queries with these terms, assume engagement intent
                    engagement_detected = True
                    logger.info(f"Auto-routing vague query '{query}' to engagement analysis (detected: '{indicator}')")
                    break
                elif indicator not in ['best', 'good', 'great', 'popular']:
                    # For more specific terms, always route to engagement
                    engagement_detected = True
                    logger.info(f"Routing query '{query}' to engagement analysis (detected: '{indicator}')")
                    break
        
        if engagement_detected:
            return self.analyze_engagement_queries(query)
        
        # Check for positional queries second (specific)
        elif any(indicator in query_lower for indicator in [
            'first user', 'second user', 'third user', 'first poster', 'second poster',
            'who was the first', 'who was the second', 'who posted first', 'who posted second',
            'earliest user', 'initial poster', 'second to post', 'third to post'
        ]):
            return self.analyze_positional_queries(query)
        
        elif any(indicator in query_lower for indicator in [
            'thread author', 'thread creator', 'who created', 'who started',
            'original poster', 'op', 'thread starter', 'who made this thread',
            'author of thread', 'creator of thread', 'thread originator',
            'who created this', 'who started this', 'thread op'
        ]):
            return self.analyze_thread_authorship(query)
        
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
            
        elif any(indicator in query_lower for indicator in [
            'what wattage', 'wattage setting', 'wattage do', 'power setting', 'watts',
            'what temperature', 'temp setting', 'temperature do', 'degrees', 'celsius', 'fahrenheit',
            'what voltage', 'voltage setting', 'volts', 'what resistance', 'ohms', 'resistance',
            'what settings', 'settings do', 'configuration', 'setup', 'how do people set',
            'what do people use', 'what do users set', 'community settings', 'recommended settings',
            'typical settings', 'common settings', 'standard settings', 'preferred settings'
        ]):
            return self.analyze_technical_specifications(query)
        
        else:
            # Fallback to participant analysis for other participant queries
            return self.analyze_participant_activity(query)
    
    def analyze_technical_specifications(self, query: str) -> Dict[str, Any]:
        """Analyze technical specification queries by finding actual user settings.
        
        Args:
            query: The original query about technical settings
            
        Returns:
            Analysis results with actual user data
        """
        posts = self._load_posts()
        if not posts:
            return {
                'error': 'No posts available for analysis',
                'total_posts_analyzed': 0,
                'type': 'technical_specifications'
            }
        
        query_lower = query.lower()
        
        # Extract what technical aspect they're asking about
        if any(term in query_lower for term in ['wattage', 'watts', 'power']):
            spec_type = 'wattage'
            search_patterns = ['watt', 'w ', ' w)', 'power', 'voltage']
        elif any(term in query_lower for term in ['temperature', 'temp', 'celsius', 'fahrenheit', 'degrees']):
            spec_type = 'temperature'
            search_patterns = ['°', 'degree', 'celsius', 'fahrenheit', 'temp', 'heat']
        elif any(term in query_lower for term in ['voltage', 'volts', 'volt']):
            spec_type = 'voltage'
            search_patterns = ['volt', 'v ', ' v)', 'voltage']
        elif any(term in query_lower for term in ['resistance', 'ohm']):
            spec_type = 'resistance'
            search_patterns = ['ohm', 'ω', 'resistance']
        else:
            spec_type = 'settings'
            search_patterns = ['setting', 'config', 'setup', 'adjust', 'set to', 'use']
        
        # Find posts mentioning technical specifications
        relevant_posts = []
        settings_mentioned = []
        
        import re
        
        for post in posts:
            content = post.get('content', '').lower()
            
            # Look for numerical values with units
            if spec_type == 'wattage':
                # Look for wattage patterns like "20W", "20 watts", "20 watt"
                watt_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:w\b|watts?\b|wattage)', content, re.IGNORECASE)
                if watt_matches:
                    for match in watt_matches:
                        settings_mentioned.append(f"{match}W")
                    relevant_posts.append({
                        **post,
                        'spec_values': [f"{match}W" for match in watt_matches],
                        'relevance_reason': f'Mentions {len(watt_matches)} wattage setting(s)'
                    })
            
            elif spec_type == 'temperature':
                # Look for temperature patterns
                temp_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:°?[cf]\b|celsius|fahrenheit|degrees?)', content, re.IGNORECASE)
                if temp_matches:
                    for match in temp_matches:
                        settings_mentioned.append(f"{match}°")
                    relevant_posts.append({
                        **post,
                        'spec_values': [f"{match}°" for match in temp_matches],
                        'relevance_reason': f'Mentions {len(temp_matches)} temperature setting(s)'
                    })
            
            elif any(pattern in content for pattern in search_patterns):
                # General settings/configuration posts
                relevant_posts.append({
                    **post,
                    'relevance_reason': f'Discusses {spec_type} settings'
                })
        
        if not relevant_posts:
            return {
                'error': f'No posts found discussing {spec_type} settings',
                'spec_type': spec_type,
                'total_posts_analyzed': len(posts),
                'type': 'technical_specifications'
            }
        
        # Analyze the findings
        analysis = {
            'type': 'technical_specifications',
            'spec_type': spec_type,
            'query': query,
            'total_posts_analyzed': len(posts),
            'relevant_posts_count': len(relevant_posts),
            'settings_found': len(set(settings_mentioned)) if settings_mentioned else 0,
            'common_settings': [],
            'top_posts': []
        }
        
        # Count frequency of specific settings
        if settings_mentioned:
            from collections import Counter
            setting_counts = Counter(settings_mentioned)
            analysis['common_settings'] = [
                {'setting': setting, 'mentions': count} 
                for setting, count in setting_counts.most_common(10)
            ]
        
        # Get top 5 most relevant posts
        # Sort by upvotes/engagement if available, otherwise by content length
        sorted_posts = sorted(relevant_posts, key=lambda p: (
            p.get('upvotes', 0) + p.get('likes', 0) + p.get('reactions', 0),
            len(p.get('content', ''))
        ), reverse=True)
        
        analysis['top_posts'] = []
        for post in sorted_posts[:5]:
            post_summary = {
                'author': post.get('author', 'Unknown'),
                'date': post.get('date', 'Unknown'),
                'page': post.get('page', 1),
                'relevance_reason': post.get('relevance_reason', ''),
                'engagement': post.get('upvotes', 0) + post.get('likes', 0) + post.get('reactions', 0),
                'content_preview': post.get('content', '')[:200] + '...' if len(post.get('content', '')) > 200 else post.get('content', '')
            }
            
            # Add spec values if found
            if 'spec_values' in post:
                post_summary['spec_values'] = post['spec_values']
            
            # Add post link if available
            if post.get('post_url'):
                post_summary['post_url'] = post['post_url']
            elif post.get('post_id'):
                post_summary['post_id'] = post['post_id']
            
            analysis['top_posts'].append(post_summary)
        
        logger.info(f"Found {len(relevant_posts)} posts discussing {spec_type} settings")
        return analysis


__all__ = ['ForumDataAnalyzer']