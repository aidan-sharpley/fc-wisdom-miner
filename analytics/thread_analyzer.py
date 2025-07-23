"""
Thread-level analytics for Forum Wisdom Miner.

This module provides comprehensive analysis of forum threads including
participant analysis, topic modeling, sentiment trends, and temporal patterns.
"""

import json
import logging
import os
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from config.settings import THREAD_ANALYTICS_NAME
from utils.file_utils import atomic_write_json, safe_read_json
from utils.text_utils import (
    calculate_readability_score,
    detect_language,
    extract_keywords,
    get_text_statistics,
)

logger = logging.getLogger(__name__)


class ThreadAnalyzer:
    """Comprehensive thread analysis and analytics generation."""

    def __init__(self, thread_dir: str):
        """Initialize analyzer for a specific thread.

        Args:
            thread_dir: Path to thread directory
        """
        self.thread_dir = thread_dir
        self.analytics_path = os.path.join(thread_dir, THREAD_ANALYTICS_NAME)
        self._cached_analytics = None
        self._cache_time = 0

    def analyze_thread(
        self, posts: List[Dict], force_refresh: bool = False
    ) -> Dict[str, Any]:
        """Perform comprehensive thread analysis.

        Args:
            posts: List of post dictionaries
            force_refresh: Force refresh of cached analytics

        Returns:
            Dictionary containing comprehensive thread analytics
        """
        # Check if we have cached analytics
        if not force_refresh and self._is_cache_valid():
            return self._cached_analytics

        logger.info(f'Analyzing thread with {len(posts)} posts')
        start_time = time.time()

        # Show progress for large datasets
        if len(posts) > 1000:
            from tqdm import tqdm
            analysis_steps = [
                ('metadata', self._analyze_metadata),
                ('participants', self._analyze_participants), 
                ('content', self._analyze_content),
                ('temporal', self._analyze_temporal_patterns),
                ('topics', self._analyze_topics),
                ('interaction', self._analyze_interactions),
                ('statistics', self._calculate_statistics)
            ]
            
            analytics = {
                'generated_at': time.time(),
                'analysis_time': 0,  # Will be updated below
            }
            
            with tqdm(total=len(analysis_steps), desc="Generating thread analytics", unit="step") as pbar:
                for step_name, step_func in analysis_steps:
                    analytics[step_name] = step_func(posts)
                    pbar.set_postfix({"step": step_name})
                    pbar.update(1)
        else:
            analytics = {
                'metadata': self._analyze_metadata(posts),
                'participants': self._analyze_participants(posts),
                'content': self._analyze_content(posts),
                'temporal': self._analyze_temporal_patterns(posts),
                'topics': self._analyze_topics(posts),
                'interaction': self._analyze_interactions(posts),
                'statistics': self._calculate_statistics(posts),
                'generated_at': time.time(),
                'analysis_time': 0,  # Will be updated below
            }

        analytics['analysis_time'] = time.time() - start_time

        # Cache the results
        self._cached_analytics = analytics
        self._cache_time = time.time()

        # Save to disk
        if atomic_write_json(self.analytics_path, analytics):
            logger.info(f'Thread analytics saved in {analytics["analysis_time"]:.2f}s')

        return analytics

    def _is_cache_valid(self, max_age_minutes: int = 30) -> bool:
        """Check if cached analytics are still valid."""
        if not self._cached_analytics:
            # Try to load from disk
            disk_analytics = safe_read_json(self.analytics_path)
            if disk_analytics:
                self._cached_analytics = disk_analytics
                self._cache_time = disk_analytics.get('generated_at', 0)

        if not self._cached_analytics:
            return False

        age_minutes = (time.time() - self._cache_time) / 60
        return age_minutes < max_age_minutes

    def _analyze_metadata(self, posts: List[Dict]) -> Dict[str, Any]:
        """Analyze basic thread metadata."""
        if not posts:
            return {}

        pages = set()
        urls = set()

        for post in posts:
            if post.get('page'):
                pages.add(post['page'])
            if post.get('url'):
                urls.add(post['url'])

        first_post = min(posts, key=lambda p: p.get('global_position', 0))
        last_post = max(posts, key=lambda p: p.get('global_position', 0))

        return {
            'total_posts': len(posts),
            'total_pages': len(pages),
            'unique_urls': len(urls),
            'first_post': {
                'author': first_post.get('author'),
                'date': first_post.get('date'),
                'position': first_post.get('global_position'),
            },
            'last_post': {
                'author': last_post.get('author'),
                'date': last_post.get('date'),
                'position': last_post.get('global_position'),
            },
        }

    def _analyze_participants(self, posts: List[Dict]) -> Dict[str, Any]:
        """Analyze thread participants and their activity patterns."""
        authors = defaultdict(
            lambda: {
                'post_count': 0,
                'total_characters': 0,
                'avg_post_length': 0,
                'first_post_position': float('inf'),
                'last_post_position': 0,
                'pages_active': set(),
                'keywords': Counter(),
            }
        )

        for post in posts:
            author = post.get('author', 'unknown')
            content = post.get('content', '')
            position = post.get('global_position', 0)
            page = post.get('page', 1)

            author_data = authors[author]
            author_data['post_count'] += 1
            author_data['total_characters'] += len(content)
            author_data['first_post_position'] = min(
                author_data['first_post_position'], position
            )
            author_data['last_post_position'] = max(
                author_data['last_post_position'], position
            )
            author_data['pages_active'].add(page)

            # Extract keywords for this author
            keywords = extract_keywords(content, max_count=10)
            for keyword in keywords:
                author_data['keywords'][keyword] += 1

        # Process author data
        processed_authors = {}
        for author, data in authors.items():
            if data['post_count'] > 0:
                data['avg_post_length'] = data['total_characters'] / data['post_count']
                data['pages_active'] = len(data['pages_active'])
                data['top_keywords'] = dict(data['keywords'].most_common(5))
                del data['keywords']  # Remove raw counter
                processed_authors[author] = data

        # Calculate summary statistics
        post_counts = [data['post_count'] for data in processed_authors.values()]
        most_active = (
            max(processed_authors.items(), key=lambda x: x[1]['post_count'])
            if processed_authors
            else ('unknown', {})
        )

        return {
            'total_participants': len(processed_authors),
            'authors': processed_authors,
            'most_active_author': {
                'name': most_active[0],
                'post_count': most_active[1].get('post_count', 0),
            },
            'avg_posts_per_author': sum(post_counts) / len(post_counts)
            if post_counts
            else 0,
            'participation_distribution': {
                'very_active': len([c for c in post_counts if c >= 10]),
                'active': len([c for c in post_counts if 3 <= c < 10]),
                'casual': len([c for c in post_counts if c < 3]),
            },
        }

    def _analyze_content(self, posts: List[Dict]) -> Dict[str, Any]:
        """Analyze content characteristics across the thread."""
        all_text = ' '.join(post.get('content', '') for post in posts)

        # Get overall text statistics
        text_stats = get_text_statistics(all_text)

        # Analyze individual post characteristics
        post_lengths = []
        languages = Counter()
        readability_scores = []

        for post in posts:
            content = post.get('content', '')
            if content:
                post_lengths.append(len(content))

                # Detect language for posts with sufficient content
                if len(content) > 50:
                    lang = detect_language(content)
                    languages[lang] += 1

                # Calculate readability (for posts with sufficient content)
                if len(content) > 100:
                    readability_scores.append(calculate_readability_score(content))

        # Extract thread-wide keywords
        thread_keywords = extract_keywords(all_text, max_count=30)

        return {
            'overall_stats': text_stats,
            'post_length_stats': {
                'min': min(post_lengths) if post_lengths else 0,
                'max': max(post_lengths) if post_lengths else 0,
                'avg': sum(post_lengths) / len(post_lengths) if post_lengths else 0,
                'median': sorted(post_lengths)[len(post_lengths) // 2]
                if post_lengths
                else 0,
            },
            'languages': dict(languages.most_common()),
            'primary_language': languages.most_common(1)[0][0]
            if languages
            else 'unknown',
            'avg_readability': sum(readability_scores) / len(readability_scores)
            if readability_scores
            else 0,
            'thread_keywords': thread_keywords[:20],  # Top 20 keywords
            'content_diversity': len(set(thread_keywords)) / len(thread_keywords)
            if thread_keywords
            else 0,
        }

    def _analyze_temporal_patterns(self, posts: List[Dict]) -> Dict[str, Any]:
        """Analyze temporal patterns in the thread."""
        try:
            # Parse dates (this is simplified - real implementation would need robust date parsing)
            dates = []
            for post in posts:
                date_str = post.get('date', '')
                if date_str and date_str != 'unknown-date':
                    # Simple date parsing - would need more robust implementation
                    try:
                        # Try common date formats
                        for fmt in [
                            '%Y-%m-%d',
                            '%Y-%m-%d %H:%M:%S',
                            '%m/%d/%Y',
                            '%d/%m/%Y',
                        ]:
                            try:
                                parsed_date = datetime.strptime(date_str[:10], fmt)
                                dates.append(parsed_date)
                                break
                            except ValueError:
                                continue
                    except:
                        continue

            if not dates:
                return {'error': 'No parseable dates found'}

            # Calculate thread duration
            thread_start = min(dates)
            thread_end = max(dates)
            duration_days = (thread_end - thread_start).days

            # Analyze posting frequency
            daily_posts = defaultdict(int)
            for date in dates:
                daily_posts[date.date()] += 1

            # Find peak activity periods
            peak_day = (
                max(daily_posts.items(), key=lambda x: x[1])
                if daily_posts
                else (None, 0)
            )

            return {
                'thread_duration_days': duration_days,
                'start_date': thread_start.isoformat() if thread_start else None,
                'end_date': thread_end.isoformat() if thread_end else None,
                'avg_posts_per_day': len(posts) / max(1, duration_days),
                'peak_activity_day': peak_day[0].isoformat() if peak_day[0] else None,
                'peak_activity_posts': peak_day[1],
                'posting_frequency': {
                    'total_active_days': len(daily_posts),
                    'max_posts_per_day': max(daily_posts.values())
                    if daily_posts
                    else 0,
                    'avg_posts_per_active_day': sum(daily_posts.values())
                    / len(daily_posts)
                    if daily_posts
                    else 0,
                },
            }
        except Exception as e:
            logger.warning(f'Error analyzing temporal patterns: {e}')
            return {'error': f'Temporal analysis failed: {e}'}

    def _analyze_topics(self, posts: List[Dict]) -> Dict[str, Any]:
        """Analyze topics and themes in the thread."""
        # Combine all content for topic analysis
        all_content = ' '.join(post.get('content', '') for post in posts)

        # Extract keywords and potential topics
        keywords = extract_keywords(all_content, max_count=50)

        # Simple topic clustering based on keyword co-occurrence
        # This is a simplified approach - could be enhanced with proper topic modeling
        topic_words = {}

        # Group related keywords (this is very basic - could use word embeddings)
        for i, word1 in enumerate(keywords[:20]):  # Limit for performance
            related_words = []
            for j, word2 in enumerate(keywords[:20]):
                if i != j and (
                    word1 in all_content.lower() and word2 in all_content.lower()
                ):
                    # Simple co-occurrence check
                    if word1 in all_content.lower() and word2 in all_content.lower():
                        related_words.append(word2)

            if related_words:
                topic_words[word1] = related_words[:5]  # Top 5 related words

        return {
            'primary_keywords': keywords[:15],
            'topic_clusters': topic_words,
            'keyword_diversity': len(set(keywords)) / len(all_content.split())
            if all_content
            else 0,
            'estimated_topics': min(
                10, len(topic_words)
            ),  # Rough estimate of topic count
        }

    def _analyze_interactions(self, posts: List[Dict]) -> Dict[str, Any]:
        """Analyze interaction patterns between participants."""
        # This is simplified - could be enhanced with quote detection, mentions, etc.

        author_interactions = defaultdict(lambda: defaultdict(int))

        # Simple interaction detection based on sequential posts
        for i in range(1, len(posts)):
            current_author = posts[i].get('author', 'unknown')
            prev_author = posts[i - 1].get('author', 'unknown')

            if (
                current_author != prev_author
                and current_author != 'unknown'
                and prev_author != 'unknown'
            ):
                author_interactions[current_author][prev_author] += 1

        # Find most interactive pairs
        interaction_pairs = []
        for author1, interactions in author_interactions.items():
            for author2, count in interactions.items():
                interaction_pairs.append((count, author1, author2))

        interaction_pairs.sort(reverse=True)

        return {
            'interaction_matrix': {k: dict(v) for k, v in author_interactions.items()},
            'top_interactions': [
                {'count': count, 'from': author1, 'to': author2}
                for count, author1, author2 in interaction_pairs[:10]
            ],
            'total_interactions': sum(
                sum(interactions.values())
                for interactions in author_interactions.values()
            ),
        }

    def _calculate_statistics(self, posts: List[Dict]) -> Dict[str, Any]:
        """Calculate overall thread statistics."""
        if not posts:
            return {}

        # Basic counts
        total_posts = len(posts)
        total_characters = sum(len(post.get('content', '')) for post in posts)

        # Page distribution
        page_distribution = Counter(post.get('page', 1) for post in posts)

        # Position statistics
        positions = [
            post.get('global_position', 0)
            for post in posts
            if post.get('global_position')
        ]

        return {
            'total_posts': total_posts,
            'total_characters': total_characters,
            'avg_characters_per_post': total_characters / total_posts
            if total_posts
            else 0,
            'page_distribution': dict(page_distribution),
            'posts_per_page': {
                'min': min(page_distribution.values()) if page_distribution else 0,
                'max': max(page_distribution.values()) if page_distribution else 0,
                'avg': sum(page_distribution.values()) / len(page_distribution)
                if page_distribution
                else 0,
            },
            'thread_health': {
                'completeness': len(positions) / total_posts if total_posts else 0,
                'consistency': 1.0
                if positions == list(range(1, len(positions) + 1))
                else 0.8,
            },
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get a high-level summary of thread analytics.

        Returns:
            Dictionary with key thread insights for quick overview
        """
        if not self._cached_analytics:
            return {'error': 'No analytics available. Run analyze_thread() first.'}

        analytics = self._cached_analytics

        return {
            'overview': {
                'total_posts': analytics.get('metadata', {}).get('total_posts', 0),
                'participants': analytics.get('participants', {}).get(
                    'total_participants', 0
                ),
                'pages': analytics.get('metadata', {}).get('total_pages', 0),
                'primary_language': analytics.get('content', {}).get(
                    'primary_language', 'unknown'
                ),
            },
            'activity': {
                'most_active_author': analytics.get('participants', {}).get(
                    'most_active_author', {}
                ),
                'avg_posts_per_author': round(
                    analytics.get('participants', {}).get('avg_posts_per_author', 0), 1
                ),
                'thread_duration_days': analytics.get('temporal', {}).get(
                    'thread_duration_days', 0
                ),
            },
            'content_insights': {
                'primary_keywords': analytics.get('topics', {}).get(
                    'primary_keywords', []
                )[:5],
                'avg_post_length': round(
                    analytics.get('content', {})
                    .get('post_length_stats', {})
                    .get('avg', 0)
                ),
                'readability_score': round(
                    analytics.get('content', {}).get('avg_readability', 0), 1
                ),
            },
            'interaction_level': {
                'total_interactions': analytics.get('interaction', {}).get(
                    'total_interactions', 0
                ),
                'participation_diversity': analytics.get('participants', {}).get(
                    'participation_distribution', {}
                ),
            },
        }


__all__ = ['ThreadAnalyzer']
