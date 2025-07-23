"""
Result ranking and scoring system for Forum Wisdom Miner.

This module provides advanced ranking algorithms that combine semantic similarity,
recency, vote counts, and other factors to provide the most relevant results.
"""

import logging
import math
from typing import Dict, List, Optional
from datetime import datetime, timezone

from utils.date_parser import get_recency_score

logger = logging.getLogger(__name__)


class PostRanker:
    """Advanced post ranking system with multiple scoring factors."""
    
    def __init__(self, thread_analytics: Optional[Dict] = None):
        """Initialize the ranker with optional thread analytics.
        
        Args:
            thread_analytics: Thread analytics for context-aware scoring
        """
        self.thread_analytics = thread_analytics or {}
        
        # Scoring weights (can be tuned based on performance)
        self.weights = {
            'semantic_similarity': 0.4,    # Base semantic relevance
            'recency': 0.25,               # How recent the post is
            'vote_score': 0.2,             # Upvotes/reactions
            'author_authority': 0.1,       # Author reputation in thread
            'content_quality': 0.05        # Content length/quality indicators
        }
        
        # Calculate thread-specific statistics for normalization
        self._calculate_thread_stats()
    
    def _calculate_thread_stats(self):
        """Calculate thread statistics for score normalization."""
        self.thread_stats = {
            'max_votes': 0,
            'avg_votes': 0,
            'date_range_days': 365,  # Default
            'active_authors': set(),
            'total_posts': 0
        }
        
        if self.thread_analytics:
            stats = self.thread_analytics.get('statistics', {})
            activity = self.thread_analytics.get('activity', {})
            
            # Get vote statistics
            vote_stats = stats.get('vote_distribution', {})
            self.thread_stats['max_votes'] = vote_stats.get('max_score', 0)
            self.thread_stats['avg_votes'] = vote_stats.get('average_score', 0)
            
            # Get date range
            duration = activity.get('thread_duration_days', 365)
            self.thread_stats['date_range_days'] = max(duration, 1)
            
            # Get author information
            self.thread_stats['total_posts'] = stats.get('total_posts', 0)
            
            # Active authors from analytics
            participants = stats.get('participants', {})
            self.thread_stats['active_authors'] = set(participants.keys())
    
    def rank_results(self, results: List[Dict], query: str = "") -> List[Dict]:
        """Rank search results using multiple scoring factors.
        
        Args:
            results: List of search result dictionaries
            query: Original search query for context
            
        Returns:
            Reranked list of results with scores
        """
        if not results:
            return results
        
        logger.info(f"Ranking {len(results)} search results")
        
        # Calculate scores for each result
        scored_results = []
        for result in results:
            score_breakdown = self._calculate_comprehensive_score(result, query)
            result_copy = result.copy()
            result_copy.update(score_breakdown)
            scored_results.append(result_copy)
        
        # Sort by final score
        ranked_results = sorted(scored_results, key=lambda x: x['final_score'], reverse=True)
        
        logger.info(f"Results ranked - top score: {ranked_results[0]['final_score']:.3f}")
        return ranked_results
    
    def _calculate_comprehensive_score(self, result: Dict, query: str) -> Dict:
        """Calculate comprehensive score for a single result.
        
        Args:
            result: Single search result
            query: Search query
            
        Returns:
            Dictionary with score breakdown
        """
        scores = {
            'semantic_score': self._get_semantic_score(result),
            'recency_score': self._get_recency_score(result),
            'vote_score': self._get_vote_score(result),
            'authority_score': self._get_authority_score(result),
            'quality_score': self._get_quality_score(result, query)
        }
        
        # Calculate weighted final score
        score_weight_mapping = {
            'semantic_score': 'semantic_similarity',
            'recency_score': 'recency',
            'vote_score': 'vote_score',
            'authority_score': 'author_authority',
            'quality_score': 'content_quality'
        }
        
        final_score = sum(
            scores[score_type] * self.weights[score_weight_mapping[score_type]]
            for score_type in scores
        )
        
        return {
            **scores,
            'final_score': final_score,
            'score_breakdown': scores
        }
    
    def _get_semantic_score(self, result: Dict) -> float:
        """Get semantic similarity score."""
        # Use existing similarity score from search
        base_score = result.get('similarity_score', 0.0)
        
        # Boost for exact keyword matches in title/content
        content = result.get('content', '').lower()
        
        # Additional semantic indicators
        semantic_boost = 0.0
        
        # Boost for posts with technical details (for vape/device forums)
        technical_indicators = ['temperature', 'watts', 'ohm', 'voltage', 'resistance', 'coil']
        technical_matches = sum(1 for indicator in technical_indicators if indicator in content)
        if technical_matches > 0:
            semantic_boost += min(0.1, technical_matches * 0.02)
        
        return min(1.0, base_score + semantic_boost)
    
    def _get_recency_score(self, result: Dict) -> float:
        """Calculate recency score based on post date."""
        # Use parsed date if available
        parsed_date = result.get('parsed_date')
        if parsed_date:
            return get_recency_score(parsed_date, self.thread_stats['date_range_days'])
        
        # Fallback to timestamp
        timestamp = result.get('timestamp', 0)
        if timestamp > 0:
            post_date = datetime.fromtimestamp(timestamp, tz=timezone.utc)
            return get_recency_score(post_date, self.thread_stats['date_range_days'])
        
        # Fallback to position-based recency (newer posts have higher positions)
        global_pos = result.get('global_position', 0)
        total_posts = self.thread_stats.get('total_posts', 1)
        if global_pos > 0 and total_posts > 0:
            # Posts later in thread get higher recency scores
            return global_pos / total_posts
        
        return 0.1  # Default minimal score
    
    def _get_vote_score(self, result: Dict) -> float:
        """Calculate vote-based score."""
        total_score = result.get('total_score', 0)
        upvotes = result.get('upvotes', 0)
        reactions = result.get('reactions', 0)
        likes = result.get('likes', 0)
        
        # Combine all positive indicators
        combined_positive = total_score + upvotes + reactions + likes
        
        if combined_positive <= 0:
            return 0.0
        
        # Normalize against thread maximum
        max_votes = max(self.thread_stats['max_votes'], 1)
        normalized_score = min(1.0, combined_positive / max_votes)
        
        # Apply logarithmic scaling for extreme values
        if combined_positive > 10:
            normalized_score = math.log10(combined_positive + 1) / math.log10(max_votes + 1)
        
        return min(1.0, normalized_score)
    
    def _get_authority_score(self, result: Dict) -> float:
        """Calculate author authority score."""
        author = result.get('author', 'unknown')
        
        if not self.thread_analytics:
            return 0.5  # Neutral score without analytics
        
        # Get author statistics from thread analytics
        participants = self.thread_analytics.get('statistics', {}).get('participants', {})
        author_stats = participants.get(author, {})
        
        if not author_stats:
            return 0.3  # Lower score for unknown authors
        
        # Factors for authority
        post_count = author_stats.get('post_count', 0)
        avg_score = author_stats.get('average_score', 0)
        
        # Authority based on participation and average scores
        participation_score = min(1.0, post_count / max(self.thread_stats['total_posts'] * 0.1, 1))
        avg_score_normalized = min(1.0, avg_score / max(self.thread_stats['avg_votes'], 1))
        
        # Boost for original poster or very active participants
        activity = self.thread_analytics.get('activity', {})
        most_active = activity.get('most_active_author', {})
        if author == most_active.get('name'):
            participation_score *= 1.2
        
        return min(1.0, (participation_score + avg_score_normalized) / 2)
    
    def _get_quality_score(self, result: Dict, query: str) -> float:
        """Calculate content quality score."""
        content = result.get('content', '')
        content_length = len(content)
        
        # Optimal length range (not too short, not too long)
        if 50 <= content_length <= 1000:
            length_score = 1.0
        elif content_length < 50:
            length_score = content_length / 50.0
        else:
            # Diminishing returns for very long posts
            length_score = max(0.5, 1000.0 / content_length)
        
        # Quality indicators
        quality_indicators = {
            'has_links': 0.1 if ('http' in content or 'www.' in content) else 0,
            'has_questions': 0.05 if '?' in content else 0,
            'has_numbers': 0.05 if any(c.isdigit() for c in content) else 0,
            'proper_sentences': 0.1 if '. ' in content and content.count('. ') >= 2 else 0,
            'varied_punctuation': 0.05 if len(set(content) & set('.,!?;:')) >= 3 else 0
        }
        
        quality_boost = sum(quality_indicators.values())
        
        # Query relevance boost
        query_words = query.lower().split()
        content_lower = content.lower()
        relevance_boost = sum(0.02 for word in query_words if word in content_lower and len(word) > 3)
        
        final_quality = min(1.0, length_score + quality_boost + relevance_boost)
        return final_quality
    
    def get_ranking_explanation(self, result: Dict) -> str:
        """Get human-readable explanation of ranking factors.
        
        Args:
            result: Ranked result with scores
            
        Returns:
            Explanation string
        """
        explanations = []
        
        semantic = result.get('semantic_score', 0)
        if semantic > 0.7:
            explanations.append("highly relevant content")
        elif semantic > 0.5:
            explanations.append("relevant content")
        
        recency = result.get('recency_score', 0)
        if recency > 0.8:
            explanations.append("very recent")
        elif recency > 0.5:
            explanations.append("relatively recent")
        
        votes = result.get('vote_score', 0)
        if votes > 0.7:
            explanations.append("highly upvoted")
        elif votes > 0.3:
            explanations.append("positively received")
        
        authority = result.get('authority_score', 0)
        if authority > 0.7:
            explanations.append("from active participant")
        
        if not explanations:
            return "standard relevance"
        
        return ", ".join(explanations)
    
    def adjust_weights(self, **new_weights):
        """Dynamically adjust scoring weights.
        
        Args:
            **new_weights: New weight values for scoring factors
        """
        for factor, weight in new_weights.items():
            if factor in self.weights:
                self.weights[factor] = weight
                logger.info(f"Updated {factor} weight to {weight}")
        
        # Ensure weights sum to 1.0
        total_weight = sum(self.weights.values())
        if total_weight != 1.0:
            for factor in self.weights:
                self.weights[factor] /= total_weight
            logger.info("Normalized weights to sum to 1.0")


class ContextualRanker(PostRanker):
    """Ranker that adapts scoring based on query context."""
    
    def __init__(self, thread_analytics: Optional[Dict] = None, query_analysis: Optional[Dict] = None):
        """Initialize contextual ranker.
        
        Args:
            thread_analytics: Thread analytics
            query_analysis: Analysis of the current query
        """
        super().__init__(thread_analytics)
        self.query_analysis = query_analysis or {}
        
        # Adjust weights based on query type
        self._adjust_weights_for_query()
    
    def _adjust_weights_for_query(self):
        """Adjust weights based on query characteristics."""
        if not self.query_analysis:
            return
        
        analytical_intent = self.query_analysis.get('analytical_intent', [])
        question_type = self.query_analysis.get('question_type')
        is_vague = self.query_analysis.get('is_vague', False)
        
        # For recent/trending questions, boost recency
        if 'trends' in analytical_intent or 'latest' in str(self.query_analysis.get('original_query', '')).lower():
            self.adjust_weights(recency=0.4, semantic_similarity=0.3)
        
        # For popularity questions, boost vote scores
        elif 'popular' in str(self.query_analysis.get('original_query', '')).lower():
            self.adjust_weights(vote_score=0.4, semantic_similarity=0.3)
        
        # For participant questions, boost authority
        elif 'participants' in analytical_intent or question_type == 'who':
            self.adjust_weights(authority_score=0.3, semantic_similarity=0.4)
        
        # For vague questions, balance all factors
        elif is_vague:
            self.adjust_weights(
                semantic_similarity=0.3, recency=0.2, vote_score=0.2, 
                authority_score=0.15, content_quality=0.15
            )


__all__ = ['PostRanker', 'ContextualRanker']