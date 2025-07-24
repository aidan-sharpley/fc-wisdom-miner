"""
Semantic search functionality for Forum Wisdom Miner.

This module provides advanced semantic search capabilities including
HyDE, reranking, and context expansion.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from analytics.query_analytics import AnalyticalSearchStrategy, ConversationalQueryProcessor
from embedding.embedding_manager import EmbeddingManager
from embedding.hnsw_index import HNSWIndex
from search.result_ranker import ContextualRanker
from utils.file_utils import safe_read_json
from utils.monitoring import monitor_search_operation, get_query_analytics

logger = logging.getLogger(__name__)


class SemanticSearchEngine:
    """Advanced semantic search engine for forum posts."""
    
    def __init__(self, thread_dir: str):
        """Initialize search engine for a specific thread.
        
        Args:
            thread_dir: Directory containing thread data
        """
        self.thread_dir = thread_dir
        self.embedding_manager = EmbeddingManager()
        self.query_processor = ConversationalQueryProcessor()
        
        # Load thread data
        self.posts = self._load_posts()
        self.posts_by_hash = {post['hash']: post for post in self.posts}
        
        # Initialize search index
        self.search_index = HNSWIndex(thread_dir)
        
        # Load analytics if available
        self.thread_analytics = self._load_analytics()
        
        # Statistics
        self.stats = {
            'total_searches': 0,
            'total_search_time': 0,
            'average_search_time': 0,
            'hyde_searches': 0,
            'reranked_searches': 0
        }
    
    @monitor_search_operation
    def search(self, query: str, top_k: int = 7, use_hyde: bool = True, 
               rerank: bool = True) -> Tuple[List[Dict], Dict]:
        """Perform semantic search on the thread.
        
        Args:
            query: Search query
            top_k: Number of results to return
            use_hyde: Whether to use HyDE (Hypothetical Document Embeddings)
            rerank: Whether to rerank results using LLM
            
        Returns:
            Tuple of (search_results, search_metadata)
        """
        start_time = time.time()
        
        logger.info(f"Searching for: '{query[:50]}...' (top_k={top_k})")
        
        try:
            # Step 1: Analyze the query
            query_analysis = self.query_processor.analyze_conversational_query(
                query, self.thread_analytics
            )
            
            # Step 2: Determine search strategy
            search_strategy = AnalyticalSearchStrategy.get_search_strategy(query_analysis)
            
            # Apply strategy adjustments
            effective_top_k = search_strategy.get('top_k', top_k)
            effective_use_hyde = search_strategy.get('use_hyde', use_hyde)
            effective_rerank = search_strategy.get('rerank', rerank)
            
            # Step 3: Generate query embedding
            if effective_use_hyde:
                query_embedding = self._get_hyde_embedding(query, query_analysis)
                self.stats['hyde_searches'] += 1
            else:
                query_embedding = self.embedding_manager.get_embeddings(query)
            
            # Step 4: Search the index
            post_hashes, distances = self.search_index.search(query_embedding, effective_top_k)
            
            # Step 5: Retrieve full post data
            raw_results = self._retrieve_posts(post_hashes, distances)
            
            # Step 6: Advanced reranking with comprehensive scoring
            if effective_rerank and len(raw_results) > 1:
                ranker = ContextualRanker(self.thread_analytics, query_analysis)
                reranked_results = ranker.rank_results(raw_results, query)
                self.stats['reranked_searches'] += 1
                logger.info(f"Advanced reranking applied with contextual scoring")
            else:
                reranked_results = raw_results
            
            # Step 7: Limit to requested number
            final_results = reranked_results[:effective_top_k]
            
            # Step 8: Add search metadata
            search_time = time.time() - start_time
            self.stats['total_searches'] += 1
            self.stats['total_search_time'] += search_time
            self.stats['average_search_time'] = self.stats['total_search_time'] / self.stats['total_searches']
            
            search_metadata = {
                'query': query,
                'query_analysis': query_analysis,
                'search_strategy': search_strategy,
                'search_time': search_time,
                'total_candidates': len(raw_results),
                'final_results': len(final_results),
                'used_hyde': effective_use_hyde,
                'used_reranking': effective_rerank,
                'index_stats': self.search_index.get_stats()
            }
            
            logger.info(f"Search completed in {search_time:.3f}s, found {len(final_results)} results")
            
            return final_results, search_metadata
            
        except Exception as e:
            logger.error(f"Error during search: {e}")
            return [], {'error': str(e)}
    
    def _load_posts(self) -> List[Dict]:
        """Load posts from thread directory."""
        posts_file = f"{self.thread_dir}/posts.json"
        posts = safe_read_json(posts_file) or []
        logger.info(f"Loaded {len(posts)} posts for search")
        return posts
    
    def _load_analytics(self) -> Optional[Dict]:
        """Load thread analytics if available."""
        analytics_file = f"{self.thread_dir}/analytics.json"
        analytics = safe_read_json(analytics_file)
        if analytics:
            logger.debug("Loaded thread analytics for search context")
        return analytics
    
    def _get_hyde_embedding(self, query: str, query_analysis: Dict) -> np.ndarray:
        """Generate HyDE embedding for the query.
        
        Args:
            query: Original query
            query_analysis: Query analysis results
            
        Returns:
            HyDE embedding
        """
        # Create context from thread analytics
        context = ""
        if self.thread_analytics:
            summary = self.thread_analytics.get('summary', {})
            keywords = summary.get('content_insights', {}).get('primary_keywords', [])
            if keywords:
                context = f"Thread discusses: {', '.join(keywords[:5])}"
        
        # Generate HyDE embedding
        hyde_embedding = self.embedding_manager.generate_hyde_embedding(query, context)
        
        logger.debug("Generated HyDE embedding for enhanced search")
        return hyde_embedding
    
    def _retrieve_posts(self, post_hashes: List[str], distances: List[float]) -> List[Dict]:
        """Retrieve full post data for search results.
        
        Args:
            post_hashes: List of post hashes from search
            distances: Corresponding similarity distances
            
        Returns:
            List of post dictionaries with search metadata
        """
        results = []
        
        for hash_val, distance in zip(post_hashes, distances):
            post = self.posts_by_hash.get(hash_val)
            if post:
                result = post.copy()
                result['search_distance'] = distance
                result['similarity_score'] = 1.0 - distance  # Convert distance to similarity
                results.append(result)
            else:
                logger.warning(f"Post hash {hash_val} not found in loaded posts")
        
        return results
    
    def _rerank_results(self, query: str, results: List[Dict], 
                       query_analysis: Dict) -> List[Dict]:
        """Rerank search results using LLM-based relevance scoring.
        
        Args:
            query: Original search query
            results: Initial search results
            query_analysis: Query analysis results
            
        Returns:
            Reranked results
        """
        logger.debug(f"Reranking {len(results)} search results")
        
        # Simple reranking based on multiple criteria
        # In a full implementation, this would use an LLM for relevance scoring
        
        for result in results:
            score = result['similarity_score']  # Base semantic similarity
            
            # Boost based on content length (moderate length preferred)
            content_length = len(result['content'])
            if 100 <= content_length <= 1000:
                score += 0.1
            elif content_length > 2000:
                score -= 0.05
            
            # Boost based on author activity
            author_activity = result.get('author_activity', 'casual')
            if author_activity in ['active', 'very_active']:
                score += 0.05
            
            # Boost based on content type relevance
            content_type = result.get('content_type', 'discussion')
            if query_analysis.get('question_type') == 'what' and content_type == 'information':
                score += 0.1
            elif 'solution' in query.lower() and content_type == 'solution':
                score += 0.15
            
            # Boost based on analytical intent match
            analytical_intents = query_analysis.get('analytical_intent', [])
            if 'statistics' in analytical_intents and any(char.isdigit() for char in result['content']):
                score += 0.1
            
            result['rerank_score'] = score
        
        # Sort by rerank score
        reranked = sorted(results, key=lambda x: x['rerank_score'], reverse=True)
        
        logger.debug("Results reranked based on multi-criteria scoring")
        return reranked
    
    def get_context_for_query(self, query: str, max_posts: int = 10) -> str:
        """Get relevant context for a query to pass to LLM.
        
        Args:
            query: User query
            max_posts: Maximum number of posts to include in context
            
        Returns:
            Formatted context string
        """
        # Perform search to get relevant posts
        results, _ = self.search(query, top_k=max_posts, rerank=True)
        
        if not results:
            return "No relevant posts found in this thread."
        
        # Format posts for LLM context
        context_parts = []
        for i, post in enumerate(results, 1):
            author = post.get('author', 'Unknown')
            content = post.get('content', '')[:1000]  # Limit length
            similarity = post.get('similarity_score', 0)
            
            context_parts.append(
                f"Post {i} (Author: {author}, Relevance: {similarity:.2f}):\n{content}\n"
            )
        
        context = "\n" + "="*50 + "\n".join(context_parts)
        
        logger.info(f"Generated context with {len(results)} posts for query")
        return context
    
    def expand_search_context(self, initial_results: List[Dict]) -> List[Dict]:
        """Expand search context by including related posts.
        
        Args:
            initial_results: Initial search results
            
        Returns:
            Expanded results with additional context
        """
        if not initial_results:
            return initial_results
        
        expanded_results = initial_results.copy()
        
        # Add posts from same authors
        relevant_authors = {post['author'] for post in initial_results}
        author_posts = [post for post in self.posts 
                       if post['author'] in relevant_authors 
                       and post not in initial_results]
        
        # Add a few most relevant author posts
        if author_posts:
            expanded_results.extend(author_posts[:3])
        
        # Add posts from similar time periods
        if initial_results:
            # Simple temporal expansion (would be more sophisticated in practice)
            reference_position = initial_results[0].get('global_position', 0)
            nearby_posts = [
                post for post in self.posts
                if abs(post.get('global_position', 0) - reference_position) <= 5
                and post not in expanded_results
            ]
            expanded_results.extend(nearby_posts[:2])
        
        logger.debug(f"Expanded context from {len(initial_results)} to {len(expanded_results)} posts")
        return expanded_results
    
    def get_search_suggestions(self, partial_query: str) -> List[str]:
        """Get search suggestions based on thread content.
        
        Args:
            partial_query: Partial user query
            
        Returns:
            List of suggested queries
        """
        suggestions = []
        
        if self.thread_analytics:
            # Suggest based on popular keywords
            keywords = self.thread_analytics.get('topics', {}).get('primary_keywords', [])
            for keyword in keywords[:5]:
                if keyword.lower() not in partial_query.lower():
                    suggestions.append(f"{partial_query} {keyword}")
            
            # Suggest analytical queries
            if len(partial_query) > 3:
                analytical_suggestions = [
                    f"What do people think about {partial_query}?",
                    f"Summarize the discussion about {partial_query}",
                    f"Who are the main contributors to {partial_query}?",
                    f"What are the different opinions on {partial_query}?"
                ]
                suggestions.extend(analytical_suggestions[:2])
        
        return suggestions[:5]  # Limit to 5 suggestions
    
    def get_stats(self) -> Dict:
        """Get search engine statistics."""
        return {
            **self.stats,
            'posts_loaded': len(self.posts),
            'index_stats': self.search_index.get_stats(),
            'embedding_stats': self.embedding_manager.get_stats()
        }


__all__ = ['SemanticSearchEngine']