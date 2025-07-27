"""
Topic Indexer Module

Provides domain-specific topic categorization and indexing for forum posts.
Creates structured topic indexes during thread parsing for enhanced querying.
"""

import json
import logging
import os
import re
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Set, Tuple
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)


@dataclass
class TopicMatch:
    """Represents a topic match found in a post."""
    topic_id: str
    post_id: str
    page: int
    author: str
    permalink: str
    excerpt: str
    score: float
    matched_keywords: List[str]
    position_in_post: int


@dataclass
class TopicSummary:
    """Summary statistics for a topic within a thread."""
    topic_id: str
    display_name: str
    post_count: int
    page_range: Tuple[int, int]
    top_contributors: List[Tuple[str, int]]  # (author, post_count)
    avg_score: float
    sample_excerpts: List[str]


class TopicIndexer:
    """
    Domain-specific topic indexer for forum posts.
    
    Analyzes post content against predefined topic schemas and creates
    structured indexes for enhanced query performance.
    """
    
    def __init__(self, topics_config_path: str = None):
        """Initialize the topic indexer with configuration."""
        if topics_config_path is None:
            topics_config_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), 
                'config', 
                'topics.json'
            )
        
        self.topics_config = self._load_topics_config(topics_config_path)
        self.topics = self.topics_config.get('topics', {})
        self.config = self.topics_config.get('configuration', {})
        
        # Compile regex patterns for efficiency
        self._keyword_patterns = self._compile_keyword_patterns()
        
        logger.info(f"Topic indexer initialized with {len(self.topics)} topics")
    
    def _load_topics_config(self, config_path: str) -> Dict:
        """Load topics configuration from JSON file."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Topics config file not found: {config_path}")
            return {"topics": {}, "configuration": {}}
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in topics config: {e}")
            return {"topics": {}, "configuration": {}}
    
    def _compile_keyword_patterns(self) -> Dict[str, Dict[str, re.Pattern]]:
        """Compile regex patterns for keyword matching."""
        patterns = {}
        
        for topic_id, topic_data in self.topics.items():
            topic_patterns = {}
            keywords = topic_data.get('keywords', [])
            
            for keyword in keywords:
                # Create word boundary pattern for exact matches
                # Handle multi-word keywords and special characters
                escaped_keyword = re.escape(keyword)
                # Allow for some flexibility with word boundaries
                pattern = rf'\b{escaped_keyword}\b'
                try:
                    topic_patterns[keyword] = re.compile(pattern, re.IGNORECASE)
                except re.error:
                    # Fallback to simple case-insensitive search
                    topic_patterns[keyword] = re.compile(
                        re.escape(keyword), re.IGNORECASE
                    )
            
            patterns[topic_id] = topic_patterns
        
        return patterns
    
    def analyze_post(self, post_data: Dict) -> List[TopicMatch]:
        """
        Analyze a single post for topic matches.
        
        Args:
            post_data: Dictionary containing post information
            
        Returns:
            List of TopicMatch objects for matched topics
        """
        content = post_data.get('content', '')
        post_id = str(post_data.get('id', ''))
        page = post_data.get('page', 1)
        author = post_data.get('author', 'Unknown')
        url = post_data.get('url', '')
        
        if not content:
            return []
        
        matches = []
        
        for topic_id, topic_data in self.topics.items():
            match = self._analyze_topic_match(
                topic_id, topic_data, content, post_id, page, author, url
            )
            if match:
                matches.append(match)
        
        return matches
    
    def _analyze_topic_match(
        self, 
        topic_id: str, 
        topic_data: Dict, 
        content: str, 
        post_id: str, 
        page: int, 
        author: str, 
        url: str
    ) -> Optional[TopicMatch]:
        """Analyze if content matches a specific topic."""
        keywords = topic_data.get('keywords', [])
        topic_patterns = self._keyword_patterns.get(topic_id, {})
        
        matched_keywords = []
        keyword_positions = []
        total_matches = 0
        
        # Find keyword matches
        for keyword in keywords:
            pattern = topic_patterns.get(keyword)
            if pattern:
                matches = list(pattern.finditer(content))
                if matches:
                    matched_keywords.append(keyword)
                    total_matches += len(matches)
                    # Store first occurrence position
                    keyword_positions.append(matches[0].start())
        
        # Check if minimum matches threshold is met
        min_matches = self.config.get('min_keyword_matches', 2)
        if len(matched_keywords) < min_matches:
            return None
        
        # Calculate relevance score
        score = self._calculate_relevance_score(
            matched_keywords, keywords, total_matches, len(content), topic_data
        )
        
        min_score = self.config.get('min_relevance_score', 0.3)
        if score < min_score:
            return None
        
        # Extract excerpt around first keyword match
        excerpt = self._extract_excerpt(content, min(keyword_positions) if keyword_positions else 0)
        
        return TopicMatch(
            topic_id=topic_id,
            post_id=post_id,
            page=page,
            author=author,
            permalink=url,
            excerpt=excerpt,
            score=score,
            matched_keywords=matched_keywords,
            position_in_post=min(keyword_positions) if keyword_positions else 0
        )
    
    def _calculate_relevance_score(
        self, 
        matched_keywords: List[str], 
        all_keywords: List[str], 
        total_matches: int, 
        content_length: int,
        topic_data: Dict
    ) -> float:
        """Calculate relevance score for a topic match."""
        # Base score: percentage of topic keywords matched
        keyword_coverage = len(matched_keywords) / len(all_keywords) if all_keywords else 0
        
        # Frequency score: how often keywords appear relative to content length
        frequency_score = min(total_matches / max(content_length / 100, 1), 1.0)
        
        # Topic weight from configuration
        topic_weight = topic_data.get('weight', 1.0)
        
        # Combined score
        base_score = (keyword_coverage * 0.7) + (frequency_score * 0.3)
        final_score = base_score * topic_weight
        
        return min(final_score, 1.0)
    
    def _extract_excerpt(self, content: str, position: int) -> str:
        """Extract a relevant excerpt around the keyword position."""
        max_length = self.config.get('max_excerpt_length', 200)
        
        # Find sentence boundaries around the position
        start = max(0, position - max_length // 2)
        end = min(len(content), position + max_length // 2)
        
        # Try to align with sentence boundaries
        excerpt = content[start:end]
        
        # Clean up excerpt
        excerpt = ' '.join(excerpt.split())  # Normalize whitespace
        
        # Add ellipsis if truncated
        if start > 0:
            excerpt = '...' + excerpt
        if end < len(content):
            excerpt = excerpt + '...'
        
        return excerpt
    
    def create_thread_topic_index(self, posts: List[Dict]) -> Dict:
        """
        Create a complete topic index for a thread.
        
        Args:
            posts: List of post dictionaries
            
        Returns:
            Dictionary containing structured topic index
        """
        topic_matches = defaultdict(list)
        
        # Analyze all posts
        for post in posts:
            matches = self.analyze_post(post)
            for match in matches:
                topic_matches[match.topic_id].append(match)
        
        # Create topic summaries
        topic_summaries = {}
        for topic_id, matches in topic_matches.items():
            if matches:  # Only include topics with matches
                summary = self._create_topic_summary(topic_id, matches)
                topic_summaries[topic_id] = summary
        
        # Build complete index
        index = {
            'schema_version': self.topics_config.get('schema_version', '1.0'),
            'thread_stats': {
                'total_posts': len(posts),
                'topics_found': len(topic_summaries),
                'total_topic_matches': sum(len(matches) for matches in topic_matches.values())
            },
            'topics': {}
        }
        
        # Add detailed topic data
        for topic_id, matches in topic_matches.items():
            if matches:
                index['topics'][topic_id] = {
                    'summary': asdict(topic_summaries[topic_id]),
                    'matches': [asdict(match) for match in matches]
                }
        
        return index
    
    def _create_topic_summary(self, topic_id: str, matches: List[TopicMatch]) -> TopicSummary:
        """Create a summary for a topic based on its matches."""
        topic_data = self.topics.get(topic_id, {})
        display_name = topic_data.get('display_name', topic_id)
        
        # Calculate statistics
        pages = [match.page for match in matches]
        page_range = (min(pages), max(pages)) if pages else (0, 0)
        
        # Count posts by author
        author_counts = defaultdict(int)
        for match in matches:
            author_counts[match.author] += 1
        
        top_contributors = sorted(
            author_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5]  # Top 5 contributors
        
        # Average score
        avg_score = sum(match.score for match in matches) / len(matches) if matches else 0
        
        # Sample excerpts (best scoring ones)
        sorted_matches = sorted(matches, key=lambda x: x.score, reverse=True)
        sample_excerpts = [match.excerpt for match in sorted_matches[:3]]
        
        return TopicSummary(
            topic_id=topic_id,
            display_name=display_name,
            post_count=len(matches),
            page_range=page_range,
            top_contributors=top_contributors,
            avg_score=avg_score,
            sample_excerpts=sample_excerpts
        )
    
    def get_topic_by_query(self, query: str) -> Optional[str]:
        """
        Find the most relevant topic for a given query.
        
        Args:
            query: User query string
            
        Returns:
            Topic ID of best matching topic, or None
        """
        query_lower = query.lower()
        best_topic = None
        best_score = 0
        
        for topic_id, topic_data in self.topics.items():
            score = 0
            keywords = topic_data.get('keywords', [])
            
            # Direct keyword matches in query
            for keyword in keywords:
                if keyword.lower() in query_lower:
                    score += 1
            
            # Fuzzy matching with topic display name and description
            display_name = topic_data.get('display_name', '').lower()
            description = topic_data.get('description', '').lower()
            
            # Check similarity with display name
            name_similarity = SequenceMatcher(None, query_lower, display_name).ratio()
            if name_similarity > 0.3:
                score += name_similarity * 2
            
            # Check if query words appear in description
            query_words = set(query_lower.split())
            desc_words = set(description.split())
            word_overlap = len(query_words.intersection(desc_words))
            if word_overlap > 0:
                score += word_overlap * 0.5
            
            # Apply topic weight
            topic_weight = topic_data.get('weight', 1.0)
            final_score = score * topic_weight
            
            if final_score > best_score:
                best_score = final_score
                best_topic = topic_id
        
        return best_topic if best_score > 0.5 else None
    
    def get_available_topics(self) -> Dict[str, Dict]:
        """Get all available topics with their metadata."""
        result = {}
        for topic_id, topic_data in self.topics.items():
            result[topic_id] = {
                'display_name': topic_data.get('display_name', topic_id),
                'description': topic_data.get('description', ''),
                'category': topic_data.get('category', 'general'),
                'keyword_count': len(topic_data.get('keywords', []))
            }
        return result