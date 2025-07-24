"""
Keyword-based search for exact matches and brand names.

This module provides fast exact keyword matching to complement semantic search,
ensuring brand names, model numbers, and specific terms are not missed.
"""

import logging
import re
from typing import Dict, List, Set, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


class KeywordSearchEngine:
    """Fast keyword search for exact matches and important terms."""
    
    def __init__(self, posts: List[Dict]):
        """Initialize keyword search with thread posts.
        
        Args:
            posts: List of post dictionaries
        """
        self.posts = posts
        self.stats = {
            'total_searches': 0,
            'total_matches': 0,
            'average_matches': 0
        }
        
        # Build keyword index for faster searching
        self._build_keyword_index()
    
    def _build_keyword_index(self):
        """Build inverted index for faster keyword lookups."""
        self.keyword_index = defaultdict(set)
        
        for i, post in enumerate(self.posts):
            content = post.get('content', '').lower()
            # Index by words for faster lookup
            words = re.findall(r'\b\w+\b', content)
            for word in words:
                self.keyword_index[word].add(i)
        
        logger.debug(f"Built keyword index with {len(self.keyword_index)} unique terms")
    
    def search(self, query: str, top_k: int = 25) -> List[Dict]:
        """Search for exact keyword matches.
        
        Args:
            query: Search query
            top_k: Maximum number of results to return
            
        Returns:
            List of matching posts with match scores
        """
        self.stats['total_searches'] += 1
        
        # Extract keywords from query
        keywords = self._extract_keywords(query)
        if not keywords:
            return []
        
        # Find posts containing keywords
        matching_posts = self._find_keyword_matches(keywords)
        
        # Score and rank results
        scored_results = self._score_keyword_matches(matching_posts, keywords)
        
        # Sort by score and limit results
        ranked_results = sorted(scored_results, key=lambda x: x['keyword_score'], reverse=True)
        final_results = ranked_results[:top_k]
        
        self.stats['total_matches'] += len(final_results)
        self.stats['average_matches'] = self.stats['total_matches'] / self.stats['total_searches']
        
        logger.debug(f"Keyword search found {len(final_results)} matches for: {keywords}")
        return final_results
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract meaningful keywords from query.
        
        Args:
            query: Search query string
            
        Returns:
            List of important keywords
        """
        # Remove common stop words but keep product-relevant terms
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'what', 'when', 'where', 'why', 'how', 'who', 'which', 'that', 'this',
            'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
        }
        
        # Extract words, keep important terms
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = []
        
        for word in words:
            # Keep if not a stop word, or if it looks like a brand/model
            if (word not in stop_words or 
                len(word) <= 4 or  # Short words might be brands (TAG, GVB)
                word.isupper() or  # Acronyms
                any(char.isdigit() for char in word)):  # Model numbers
                keywords.append(word)
        
        return keywords
    
    def _find_keyword_matches(self, keywords: List[str]) -> Dict[int, Dict]:
        """Find posts containing keywords.
        
        Args:
            keywords: List of keywords to search for
            
        Returns:
            Dictionary mapping post indices to match info
        """
        matches = defaultdict(lambda: {'matched_keywords': set(), 'match_count': 0})
        
        for keyword in keywords:
            # Find posts containing this keyword
            post_indices = self.keyword_index.get(keyword, set())
            
            for post_idx in post_indices:
                matches[post_idx]['matched_keywords'].add(keyword)
                matches[post_idx]['match_count'] += 1
        
        return matches
    
    def _score_keyword_matches(self, matches: Dict[int, Dict], keywords: List[str]) -> List[Dict]:
        """Score keyword matches based on relevance.
        
        Args:
            matches: Dictionary of post matches
            keywords: Original keywords searched
            
        Returns:
            List of scored post dictionaries
        """
        scored_results = []
        
        for post_idx, match_info in matches.items():
            if post_idx >= len(self.posts):
                continue
                
            post = self.posts[post_idx].copy()
            
            # Calculate keyword score
            keyword_score = self._calculate_keyword_score(
                match_info['match_count'], 
                len(match_info['matched_keywords']),
                len(keywords),
                post.get('content', '')
            )
            
            post['keyword_score'] = keyword_score
            post['matched_keywords'] = list(match_info['matched_keywords'])
            post['keyword_match_count'] = match_info['match_count']
            
            scored_results.append(post)
        
        return scored_results
    
    def _calculate_keyword_score(self, match_count: int, unique_matches: int, 
                                total_keywords: int, content: str) -> float:
        """Calculate relevance score for keyword matches.
        
        Args:
            match_count: Total number of keyword matches
            unique_matches: Number of unique keywords matched
            total_keywords: Total keywords in query
            content: Post content for additional scoring
            
        Returns:
            Keyword relevance score (0-1)
        """
        # Base score from keyword coverage
        coverage_score = unique_matches / max(1, total_keywords)
        
        # Frequency bonus (multiple mentions of same keyword)
        frequency_bonus = min(match_count / max(1, unique_matches) - 1, 0.5)
        
        # Content quality bonus (longer content often more informative)
        content_bonus = min(len(content) / 1000, 0.2)
        
        # Final score
        score = coverage_score + frequency_bonus + content_bonus
        return min(score, 1.0)
    
    def get_stats(self) -> Dict:
        """Get keyword search statistics."""
        return self.stats.copy()


def merge_search_results(semantic_results: List[Dict], keyword_results: List[Dict], 
                        max_results: int = 40) -> List[Dict]:
    """Merge and deduplicate semantic and keyword search results.
    
    Args:
        semantic_results: Results from semantic search
        keyword_results: Results from keyword search
        max_results: Maximum number of results to return
        
    Returns:
        Merged and deduplicated results list
    """
    seen_hashes = set()
    merged_results = []
    
    # Add semantic results first (usually higher quality)
    for result in semantic_results:
        post_hash = result.get('hash')
        if post_hash and post_hash not in seen_hashes:
            # Mark as semantic result
            result['search_type'] = 'semantic'
            merged_results.append(result)
            seen_hashes.add(post_hash)
    
    # Add keyword results that weren't already found
    for result in keyword_results:
        post_hash = result.get('hash')
        if post_hash and post_hash not in seen_hashes:
            # Mark as keyword result
            result['search_type'] = 'keyword'
            merged_results.append(result)
            seen_hashes.add(post_hash)
            
            # Stop if we have enough results
            if len(merged_results) >= max_results:
                break
    
    logger.info(f"Merged search: {len(semantic_results)} semantic + {len(keyword_results)} keyword → {len(merged_results)} unique results")
    return merged_results[:max_results]


__all__ = ['KeywordSearchEngine', 'merge_search_results']