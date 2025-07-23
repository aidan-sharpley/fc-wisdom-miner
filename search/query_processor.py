"""
Query processing and response generation for Forum Wisdom Miner.

This module handles the final stage of query processing, including
LLM interaction and response formatting.
"""

import logging
import time
from typing import Dict, List, Optional

import requests

from analytics.data_analyzer import ForumDataAnalyzer
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
        self.data_analyzer = ForumDataAnalyzer(thread_dir)
        
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
            # Step 1: Analyze the query with smart enhancement
            query_analysis = self.query_analyzer.analyze_conversational_query(query)
            analytical_intent = query_analysis.get('analytical_intent', [])
            
            # Check if query was auto-expanded and log it
            expanded_query = query_analysis.get('expanded_query', query)
            if expanded_query != query:
                logger.info(f"Query auto-enhanced from '{query}' to '{expanded_query}' for better results")
            
            # Step 2: Check if this can be handled with direct data analysis
            if self.data_analyzer.can_handle_query(query, analytical_intent):
                logger.info("Routing query to analytical data processor")
                analytical_result = self.data_analyzer.analyze_query(query, analytical_intent)
                
                if 'error' not in analytical_result:
                    # Generate response from analytical data
                    if stream:
                        response_generator = self._generate_analytical_response_stream(analytical_result)
                        return {
                            'query': query,
                            'analysis': query_analysis,
                            'analytical_result': analytical_result,
                            'context_posts': analytical_result.get('thread_stats', {}).get('total_posts', 0),
                            'response_stream': response_generator,
                            'processing_time': time.time() - start_time,
                            'query_type': 'analytical'
                        }
                    else:
                        response_text = self._generate_analytical_response_text(analytical_result)
                        processing_time = time.time() - start_time
                        
                        # Update statistics
                        self.stats['total_queries'] += 1
                        self.stats['total_response_time'] += processing_time
                        self.stats['average_response_time'] = (
                            self.stats['total_response_time'] / self.stats['total_queries']
                        )
                        
                        return {
                            'query': query,
                            'analysis': query_analysis,
                            'analytical_result': analytical_result,
                            'response': response_text,
                            'processing_time': processing_time,
                            'query_type': 'analytical'
                        }
            
            # Step 3: Fallback to semantic search for non-analytical queries
            logger.info("Routing query to semantic search engine")
            search_results, search_metadata = self.search_engine.search(
                query, 
                top_k=query_analysis.get('suggested_approach') == 'comprehensive' and 15 or 10
            )
            
            # Step 4: Build context
            context = self._build_context(search_results, query_analysis)
            
            # Step 5: Generate enhanced prompt
            enhanced_prompt = self.query_analyzer.generate_analytical_prompt(
                query, query_analysis, context
            )
            
            # Step 6: Get LLM response
            if stream:
                response_generator = self._get_streaming_response(enhanced_prompt)
                return {
                    'query': query,
                    'analysis': query_analysis,
                    'search_metadata': search_metadata,
                    'context_posts': len(search_results),
                    'response_stream': response_generator,
                    'processing_time': time.time() - start_time,
                    'query_type': 'semantic'
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
    
    def _generate_analytical_response_stream(self, analytical_result: Dict):
        """Generate streaming response from analytical results.
        
        Args:
            analytical_result: Results from analytical data processing
            
        Yields:
            Response chunks
        """
        response_text = self._generate_analytical_response_text(analytical_result)
        
        # Yield the response in chunks to simulate streaming
        words = response_text.split()
        chunk_size = 5  # Words per chunk
        
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i+chunk_size])
            if i + chunk_size < len(words):
                chunk += ' '
            yield chunk
            time.sleep(0.05)  # Small delay for streaming effect
    
    def _generate_analytical_response_text(self, analytical_result: Dict) -> str:
        """Generate response text from analytical results.
        
        Args:
            analytical_result: Results from analytical data processing
            
        Returns:
            Formatted response text
        """
        result_type = analytical_result.get('type', 'unknown')
        
        if result_type == 'participant_analysis':
            return self._format_participant_analysis(analytical_result)
        elif result_type == 'content_statistics':
            return self._format_content_statistics(analytical_result)
        elif result_type == 'temporal_analysis':
            return self._format_temporal_analysis(analytical_result)
        elif result_type == 'positional_analysis':
            return self._format_positional_analysis(analytical_result)
        elif result_type == 'engagement_analysis':
            return self._format_engagement_analysis(analytical_result)
        else:
            return f"Analysis complete. Result type: {result_type}"
    
    def _format_participant_analysis(self, result: Dict) -> str:
        """Format participant analysis results."""
        most_active = result.get('most_active_author', {})
        top_authors = result.get('top_authors', [])
        thread_stats = result.get('thread_stats', {})
        
        response_parts = []
        
        # Main answer
        if most_active.get('name'):
            response_parts.append(
                f"**{most_active['name']}** is the most active user in this thread.\n"
            )
            
            response_parts.append(
                f"• **Post count**: {most_active['post_count']} posts "
                f"({most_active.get('percentage', 0):.1f}% of all posts)\n"
            )
            
            if most_active.get('total_score', 0) > 0:
                response_parts.append(
                    f"• **Community engagement**: {most_active['total_score']} total upvotes/reactions\n"
                )
        
        # Thread context
        if thread_stats:
            response_parts.append(
                f"\n**Thread Overview:**\n"
                f"• Total posts: {thread_stats.get('total_posts', 0)}\n"
                f"• Unique participants: {thread_stats.get('unique_authors', 0)}\n"
                f"• Average posts per participant: {thread_stats.get('average_posts_per_author', 0)}\n"
            )
        
        # Top contributors
        if len(top_authors) > 1:
            response_parts.append("\n**Top 5 Contributors:**\n")
            for i, author in enumerate(top_authors[:5], 1):
                response_parts.append(
                    f"{i}. **{author['name']}**: {author['post_count']} posts "
                    f"({author.get('percentage', 0):.1f}%)\n"
                )
        
        response_parts.append(
            f"\n*This analysis is based on the complete thread data "
            f"({thread_stats.get('total_posts', 0)} posts analyzed).*"
        )
        
        return ''.join(response_parts)
    
    def _format_content_statistics(self, result: Dict) -> str:
        """Format content statistics results."""
        post_stats = result.get('post_statistics', {})
        page_dist = result.get('page_distribution', {})
        temporal = result.get('temporal_coverage', {})
        
        response_parts = []
        
        response_parts.append("**Thread Content Statistics:**\n\n")
        
        # Basic stats
        response_parts.append(
            f"• **Total posts**: {post_stats.get('total_posts', 0)}\n"
            f"• **Average post length**: {post_stats.get('average_length', 0)} characters\n"
            f"• **Average word count**: {post_stats.get('average_word_count', 0)} words\n"
        )
        
        # Page distribution
        if page_dist.get('total_pages'):
            response_parts.append(
                f"• **Thread spans**: {page_dist['total_pages']} pages\n"
            )
        
        # Temporal coverage
        if temporal.get('date_coverage_percentage'):
            response_parts.append(
                f"• **Posts with timestamps**: {temporal['posts_with_dates']} "
                f"({temporal.get('date_coverage_percentage', 0):.1f}%)\n"
            )
        
        return ''.join(response_parts)
    
    def _format_temporal_analysis(self, result: Dict) -> str:
        """Format temporal analysis results."""
        timeline = result.get('thread_timeline', {})
        activity = result.get('activity_pattern', {})
        
        response_parts = []
        
        response_parts.append("**Thread Timeline Analysis:**\n\n")
        
        if timeline:
            response_parts.append(
                f"• **First post**: {timeline.get('first_post', 'Unknown')}\n"
                f"• **Last post**: {timeline.get('last_post', 'Unknown')}\n"
                f"• **Thread duration**: {timeline.get('duration_days', 0)} days\n"
                f"• **Posts with dates**: {timeline.get('posts_with_dates', 0)}\n"
            )
        
        if activity:
            most_active_month = activity.get('most_active_month')
            if most_active_month:
                response_parts.append(
                    f"• **Most active month**: {most_active_month[0]} ({most_active_month[1]} posts)\n"
                )
            
            avg_per_day = activity.get('average_posts_per_day', 0)
            if avg_per_day > 0:
                response_parts.append(
                    f"• **Average activity**: {avg_per_day:.1f} posts per day\n"
                )
        
        return ''.join(response_parts)
    
    def _format_engagement_analysis(self, result: Dict) -> str:
        """Format engagement analysis results with smart query interpretation."""
        if 'error' in result:
            return f"**Analysis Error:**\n\n{result['error']}\n\nTotal posts analyzed: {result.get('total_posts_analyzed', 0)}"
        
        metric_name = result.get('metric_name', 'engagement')
        top_post = result.get('top_post', {})
        top_5_posts = result.get('top_5_posts', [])
        original_query = result.get('query', '')
        
        response_parts = []
        
        # Add smart interpretation note for vague queries
        if len(original_query.split()) <= 3 and any(term in original_query.lower() for term in ['rated', 'best', 'good', 'popular']):
            response_parts.append(f"💡 **Smart Interpretation:** I understood '{original_query}' as a request for the {metric_name} post based on community engagement.\n\n")
        
        response_parts.append(f"**{metric_name.title()} Post Analysis:**\n\n")
        
        # Top post details
        if top_post:
            response_parts.append(f"🏆 **Top Result:**\n")
            response_parts.append(f"• **Author**: {top_post.get('author', 'Unknown')}\n")
            response_parts.append(f"• **Score**: {top_post.get('score', 0)}\n")
            response_parts.append(f"• **Date**: {top_post.get('date', 'Unknown')}\n")
            response_parts.append(f"• **Post position**: #{top_post.get('global_position', 0)}\n")
            response_parts.append(f"• **Page**: {top_post.get('page', 0)}\n")
            
            # Engagement breakdown
            response_parts.append(f"\n📊 **Engagement Breakdown:**\n")
            if top_post.get('upvotes', 0) > 0:
                response_parts.append(f"• **Upvotes**: {top_post.get('upvotes', 0)}\n")
            if top_post.get('downvotes', 0) > 0:
                response_parts.append(f"• **Downvotes**: {top_post.get('downvotes', 0)}\n")
            if top_post.get('likes', 0) > 0:
                response_parts.append(f"• **Likes**: {top_post.get('likes', 0)}\n")
            if top_post.get('reactions', 0) > 0:
                response_parts.append(f"• **Reactions**: {top_post.get('reactions', 0)}\n")
            if top_post.get('total_score', 0) > 0:
                response_parts.append(f"• **Total Score**: {top_post.get('total_score', 0)}\n")
            
            # Post link if available
            post_url = top_post.get('post_url')
            post_id = top_post.get('post_id')
            if post_url:
                response_parts.append(f"• **Direct link**: {post_url}\n")
            elif post_id:
                response_parts.append(f"• **Post ID**: {post_id}\n")
            
            # Content preview
            content_preview = top_post.get('content_preview')
            if content_preview:
                response_parts.append(f"\n**Content Preview:**\n> {content_preview}\n")
        
        # Top 5 summary if multiple results
        if len(top_5_posts) > 1:
            response_parts.append(f"\n📋 **Top 5 {metric_name.title()} Posts:**\n")
            for i, post in enumerate(top_5_posts[:5], 1):
                response_parts.append(
                    f"{i}. **{post.get('author', 'Unknown')}** (Score: {post.get('score', 0)}) - "
                    f"Post #{post.get('global_position', 0)}\n"
                )
        
        # Summary stats
        response_parts.append(f"\n📈 **Summary:**\n")
        response_parts.append(f"• **Posts with engagement**: {result.get('total_posts_with_engagement', 0)}\n")
        response_parts.append(f"• **Total posts analyzed**: {result.get('total_posts_analyzed', 0)}\n")
        
        return ''.join(response_parts)
    
    def _format_positional_analysis(self, result: Dict) -> str:
        """Format positional analysis results."""
        if 'error' in result:
            return f"**Analysis Error:**\n\n{result['error']}\n\nTotal unique authors found: {result.get('total_unique_authors', 0)}"
        
        position = result.get('position', 1)
        author = result.get('author', 'Unknown')
        total_authors = result.get('total_unique_authors', 0)
        
        ordinal_map = {1: 'first', 2: 'second', 3: 'third', 4: 'fourth', 5: 'fifth'}
        position_text = ordinal_map.get(position, f"{position}th")
        
        response_parts = []
        response_parts.append(f"**{position_text.title()} User to Post:**\n\n")
        response_parts.append(f"• **Author**: {author}\n")
        response_parts.append(f"• **Position**: {position} out of {total_authors} unique authors\n")
        
        # Add first post details if available
        first_post_date = result.get('first_post_date')
        post_position = result.get('post_position')
        first_post_content = result.get('first_post_content')
        post_url = result.get('post_url')
        post_id = result.get('post_id')
        page_number = result.get('page_number')
        
        if first_post_date:
            response_parts.append(f"• **First post date**: {first_post_date}\n")
        
        if post_position:
            response_parts.append(f"• **Post position in thread**: #{post_position}\n")
        
        if page_number:
            response_parts.append(f"• **Page**: {page_number}\n")
        
        # Add post link if available
        if post_url:
            response_parts.append(f"• **Direct link**: {post_url}\n")
        elif post_id:
            response_parts.append(f"• **Post ID**: {post_id}\n")
        
        if first_post_content:
            response_parts.append(f"\n**First post preview:**\n> {first_post_content}\n")
        
        return ''.join(response_parts)


__all__ = ['QueryProcessor']