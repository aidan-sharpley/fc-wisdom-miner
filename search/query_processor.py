"""
Query processing and response generation for Forum Wisdom Miner.

This module handles the final stage of query processing, including
LLM interaction and response formatting.
"""

import logging
import time
from typing import Dict, List, Optional

import requests

from analytics.query_analytics import ConversationalQueryProcessor
from config.settings import OLLAMA_BASE_URL, OLLAMA_CHAT_MODEL
from search.semantic_search import SemanticSearchEngine

logger = logging.getLogger(__name__)


class QueryProcessor:
    """Processes user queries and generates responses using LLM."""
    
    def __init__(self, thread_dir: str):
        """Initialize query processor for a specific thread.
        
        Args:
            thread_dir: Directory containing thread data
        """
        self.thread_dir = thread_dir
        self.search_engine = SemanticSearchEngine(thread_dir)
        self.query_analyzer = ConversationalQueryProcessor()
        
        # LLM configuration
        self.chat_url = f"{OLLAMA_BASE_URL}/api/chat"
        self.model = OLLAMA_CHAT_MODEL
        
        # Statistics
        self.stats = {
            'total_queries': 0,
            'total_response_time': 0,
            'average_response_time': 0,
            'llm_calls': 0,
            'total_llm_time': 0
        }
    
    def process_query(self, query: str, stream: bool = True) -> Dict:
        """Process a user query and generate a response.
        
        Args:
            query: User query string
            stream: Whether to stream the response
            
        Returns:
            Processing results and response
        """
        start_time = time.time()
        
        logger.info(f"Processing query: '{query[:50]}...'")
        
        try:
            # Step 1: Analyze the query
            query_analysis = self.query_analyzer.analyze_conversational_query(query)
            
            # Step 2: Perform semantic search
            search_results, search_metadata = self.search_engine.search(
                query, 
                top_k=query_analysis.get('suggested_approach') == 'comprehensive' and 15 or 10
            )
            
            # Step 3: Build context
            context = self._build_context(search_results, query_analysis)
            
            # Step 4: Generate enhanced prompt
            enhanced_prompt = self.query_analyzer.generate_analytical_prompt(
                query, query_analysis, context
            )
            
            # Step 5: Get LLM response
            if stream:
                response_generator = self._get_streaming_response(enhanced_prompt)
                return {
                    'query': query,
                    'analysis': query_analysis,
                    'search_metadata': search_metadata,
                    'context_posts': len(search_results),
                    'response_stream': response_generator,
                    'processing_time': time.time() - start_time
                }
            else:
                response_text = self._get_complete_response(enhanced_prompt)
                processing_time = time.time() - start_time
                
                # Update statistics
                self.stats['total_queries'] += 1
                self.stats['total_response_time'] += processing_time
                self.stats['average_response_time'] = (
                    self.stats['total_response_time'] / self.stats['total_queries']
                )
                
                return {
                    'query': query,
                    'response': response_text,
                    'analysis': query_analysis,
                    'search_metadata': search_metadata,
                    'context_posts': len(search_results),
                    'processing_time': processing_time
                }
                
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                'query': query,
                'error': str(e),
                'processing_time': time.time() - start_time
            }
    
    def _build_context(self, search_results: List[Dict], query_analysis: Dict) -> str:
        """Build context from search results.
        
        Args:
            search_results: Results from semantic search
            query_analysis: Query analysis results
            
        Returns:
            Formatted context string
        """
        if not search_results:
            return "No relevant posts found in this thread."
        
        context_parts = []
        
        for i, post in enumerate(search_results, 1):
            author = post.get('author', 'Unknown')
            content = post.get('content', '')
            date = post.get('date', 'Unknown date')
            similarity = post.get('similarity_score', 0)
            page = post.get('page', 1)
            
            # Format post for context
            post_header = f"Post {i} | Author: {author} | Page: {page} | Date: {date} | Relevance: {similarity:.2f}"
            post_content = content[:1500]  # Limit content length
            
            if len(content) > 1500:
                post_content += "\n[... content truncated ...]"
            
            context_parts.append(f"{post_header}\n{'-' * 50}\n{post_content}\n")
        
        # Add thread-level context if available
        thread_context = ""
        if hasattr(self.search_engine, 'thread_analytics') and self.search_engine.thread_analytics:
            analytics = self.search_engine.thread_analytics.get('summary', {})
            overview = analytics.get('overview', {})
            
            if overview:
                thread_context = (
                    f"\nTHREAD OVERVIEW:\n"
                    f"Total Posts: {overview.get('total_posts', 'Unknown')}\n"
                    f"Participants: {overview.get('participants', 'Unknown')}\n"
                    f"Pages: {overview.get('pages', 'Unknown')}\n"
                )
                
                # Add key topics
                content_insights = analytics.get('content_insights', {})
                keywords = content_insights.get('primary_keywords', [])
                if keywords:
                    thread_context += f"Main Topics: {', '.join(keywords[:5])}\n"
        
        context = thread_context + "\nRELEVANT POSTS:\n" + "\n".join(context_parts)
        
        logger.debug(f"Built context with {len(search_results)} posts ({len(context)} characters)")
        return context
    
    def _get_streaming_response(self, prompt: str):
        """Get streaming response from LLM.
        
        Args:
            prompt: Enhanced prompt for LLM
            
        Yields:
            Response chunks
        """
        logger.info("Getting streaming LLM response")
        start_time = time.time()
        
        try:
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "stream": True
            }
            
            response = requests.post(
                self.chat_url,
                json=payload,
                stream=True,
                timeout=120
            )
            response.raise_for_status()
            
            self.stats['llm_calls'] += 1
            
            for line in response.iter_lines():
                if line:
                    try:
                        import json
                        data = json.loads(line.decode('utf-8'))
                        
                        if 'message' in data and 'content' in data['message']:
                            content = data['message']['content']
                            if content:
                                yield content
                        
                        # Check if done
                        if data.get('done', False):
                            break
                            
                    except json.JSONDecodeError:
                        continue
            
            llm_time = time.time() - start_time
            self.stats['total_llm_time'] += llm_time
            logger.info(f"Streaming response completed in {llm_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Error getting streaming response: {e}")
            yield f"Error generating response: {e}"
    
    def _get_complete_response(self, prompt: str) -> str:
        """Get complete response from LLM.
        
        Args:
            prompt: Enhanced prompt for LLM
            
        Returns:
            Complete response text
        """
        logger.info("Getting complete LLM response")
        start_time = time.time()
        
        try:
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "stream": False
            }
            
            response = requests.post(
                self.chat_url,
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            
            result = response.json()
            response_text = result.get('message', {}).get('content', 'No response generated')
            
            self.stats['llm_calls'] += 1
            llm_time = time.time() - start_time
            self.stats['total_llm_time'] += llm_time
            
            logger.info(f"Complete response generated in {llm_time:.2f}s")
            return response_text
            
        except Exception as e:
            logger.error(f"Error getting complete response: {e}")
            return f"Error generating response: {e}"
    
    def get_query_suggestions(self, partial_query: str) -> List[str]:
        """Get query suggestions based on thread content.
        
        Args:
            partial_query: Partial user input
            
        Returns:
            List of suggested complete queries
        """
        return self.search_engine.get_search_suggestions(partial_query)
    
    def get_thread_summary(self) -> str:
        """Get a brief summary of the thread for context.
        
        Returns:
            Thread summary text
        """
        if hasattr(self.search_engine, 'thread_analytics') and self.search_engine.thread_analytics:
            analytics = self.search_engine.thread_analytics.get('summary', {})
            overview = analytics.get('overview', {})
            activity = analytics.get('activity', {})
            content_insights = analytics.get('content_insights', {})
            
            summary_parts = []
            
            # Basic stats
            posts = overview.get('total_posts', 0)
            participants = overview.get('participants', 0)
            pages = overview.get('pages', 0)
            
            if posts:
                summary_parts.append(f"This thread contains {posts} posts from {participants} participants across {pages} pages.")
            
            # Most active author
            most_active = activity.get('most_active_author', {})
            if most_active.get('name'):
                summary_parts.append(f"Most active participant: {most_active['name']} ({most_active.get('post_count', 0)} posts).")
            
            # Main topics
            keywords = content_insights.get('primary_keywords', [])
            if keywords:
                summary_parts.append(f"Main discussion topics: {', '.join(keywords[:5])}.")
            
            # Thread duration
            duration = activity.get('thread_duration_days', 0)
            if duration > 0:
                summary_parts.append(f"Discussion span: {duration} days.")
            
            return ' '.join(summary_parts)
        
        return "Thread analytics not available."
    
    def get_stats(self) -> Dict:
        """Get query processor statistics."""
        return {
            **self.stats,
            'average_llm_time': (
                self.stats['total_llm_time'] / max(1, self.stats['llm_calls'])
            ),
            'search_engine_stats': self.search_engine.get_stats()
        }


__all__ = ['QueryProcessor']