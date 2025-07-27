"""
Query processing and response generation for Forum Wisdom Miner.

This module handles the final stage of query processing, including
LLM interaction and response formatting.
"""

import json
import logging
import time
from typing import Dict, List, Optional

import requests

from analytics.data_analyzer import ForumDataAnalyzer
from analytics.query_analytics import ConversationalQueryProcessor
from analytics.llm_query_router import LLMQueryRouter
from analytics.topic_indexer import TopicIndexer
from config.settings import OLLAMA_BASE_URL, OLLAMA_CHAT_MODEL
from search.semantic_search import SemanticSearchEngine
from search.response_refiner import ResponseRefiner
from search.keyword_search import KeywordSearchEngine, merge_search_results
# Lazy import for verifiable response system to speed up startup
from utils.file_utils import safe_read_json
from utils.shared_data_manager import get_data_manager
from utils.memory_optimizer import memory_efficient
from utils.topic_cache import TopicIndexCache

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
        self.response_refiner = ResponseRefiner()
        self.llm_router = LLMQueryRouter()
        
        # Initialize topic indexing components
        self.topic_indexer = TopicIndexer()
        self.topic_cache = TopicIndexCache()
        
        # Initialize keyword search with posts from semantic search engine
        self.keyword_search = KeywordSearchEngine(self.search_engine.posts)
        
        # Initialize verifiable response system lazily
        self.verifiable_response = None
        
        # Extract thread key from directory path
        import os
        self.thread_key = os.path.basename(thread_dir)
        
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
            # Step 1: Check for topic index matches first
            topic_search_result = self._search_topic_index(query)
            
            # Step 2: Use LLM to intelligently route the query
            thread_metadata = self._get_thread_metadata()
            routing_decision = self.llm_router.route_query(query, thread_metadata)
            
            logger.info(f"LLM routing decision: {routing_decision.get('method')} - {routing_decision.get('reasoning')}")
            
            # Step 3: Apply query enhancement for semantic queries
            if routing_decision.get('method') == 'semantic':
                query_analysis = self.query_analyzer.analyze_conversational_query(query)
                expanded_query = query_analysis.get('expanded_query', query)
                if expanded_query != query:
                    logger.info(f"Query auto-enhanced from '{query}' to '{expanded_query}' for better results")
            else:
                query_analysis = {'expanded_query': query}
                expanded_query = query
            
            # Step 4: Check if topic index found strong matches
            if topic_search_result.get('total_matches', 0) > 0:
                logger.info(f"Found {topic_search_result['total_matches']} topic matches for '{topic_search_result.get('primary_topic')}', prioritizing topic-based response")
                
                # Use topic matches as primary context for semantic processing
                topic_matches = topic_search_result.get('matches', [])
                if topic_matches and routing_decision.get('method') == 'semantic':
                    # Convert topic matches to search results format
                    topic_based_results = self._convert_topic_matches_to_search_results(topic_matches)
                    
                    # Generate response using topic context
                    if stream:
                        response_generator = self._generate_topic_based_response_stream(
                            query, topic_search_result, topic_based_results
                        )
                        return {
                            'query': query,
                            'analysis': query_analysis,
                            'topic_context': topic_search_result,
                            'search_results': topic_based_results,
                            'context_posts': len(topic_matches),
                            'response_stream': response_generator,
                            'processing_time': time.time() - start_time,
                            'query_type': 'topic_semantic',
                            'search_method': 'topic_index'
                        }
            
            # Step 5: Route to appropriate processor with topic context
            if routing_decision.get('method') == 'analytical':
                logger.info("Routing query to analytical data processor")
                # Map the LLM's query_type to our analytical capabilities
                analytical_intent = [routing_decision.get('query_type', 'general')]
                analytical_result = self.data_analyzer.analyze_query(query, analytical_intent)
                
                # Check if analytical processing found meaningful results
                if 'error' not in analytical_result and self._has_meaningful_analytical_result(analytical_result):
                    # Generate verifiable response with evidence
                    verification_report = self._get_verification_report({
                        'analytical_result': analytical_result,
                        'query_type': 'analytical',
                        'processing_time': time.time() - start_time
                    })
                    
                    if stream:
                        response_generator = self._generate_analytical_response_stream(analytical_result, verification_report)
                        return {
                            'query': query,
                            'analysis': query_analysis,
                            'analytical_result': analytical_result,
                            'verification_report': verification_report,
                            'context_posts': analytical_result.get('thread_stats', {}).get('total_posts', 0),
                            'response_stream': response_generator,
                            'processing_time': time.time() - start_time,
                            'query_type': 'analytical',
                            'fact_checked': True
                        }
                    else:
                        response_text = self._generate_analytical_response_text(analytical_result, verification_report)
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
                            'verification_report': verification_report,
                            'response': response_text,
                            'processing_time': processing_time,
                            'query_type': 'analytical',
                            'fact_checked': True
                        }
                else:
                    # Analytical processing failed or found no relevant content
                    # Fall back to semantic search for content-based queries
                    logger.info("Analytical processing failed or found no content, falling back to semantic search")
                    routing_decision = {'method': 'semantic', 'search_depth': 15}
            
            else:
                # Step 4: Semantic search processing
                logger.info("Routing query to semantic search engine")
                
                # Use LLM-recommended search depth
                search_depth = routing_decision.get('search_depth', 10)
                logger.info(f"Using LLM-recommended search depth: {search_depth}")
                
                # Additional depth adjustments based on query analysis
                if query_analysis.get('suggested_approach') == 'comprehensive':
                    search_depth = max(search_depth, 15)
                
                # Multi-stage search pipeline: keyword + semantic
                search_query = expanded_query
                
                # Stage 1: Keyword search for exact matches
                keyword_results = self.keyword_search.search(query, top_k=max(25, search_depth // 2))
                
                # Stage 2: Enhanced semantic search with topic awareness
                enhanced_query = self._enhance_query_with_topics(search_query, query_analysis)
                semantic_results, search_metadata = self.search_engine.search(enhanced_query, top_k=search_depth)
                
                # Stage 3: Merge and deduplicate results with topic boosting
                search_results = merge_search_results(semantic_results, keyword_results, max_results=min(search_depth + 15, 50))
                search_results = self._boost_topic_relevant_results(search_results, query_analysis)
                
                logger.info(f"Multi-stage search: {len(keyword_results)} keyword + {len(semantic_results)} semantic → {len(search_results)} merged results")
                
                # Step 5: Build context
                context = self._build_context(search_results, query_analysis)
                
                # Step 6: Generate enhanced prompt
                enhanced_prompt = self.query_analyzer.generate_analytical_prompt(
                    query, query_analysis, context
                )
                
                # Step 7: Get LLM response
                if stream:
                    raw_response_generator = self._get_streaming_response(enhanced_prompt)
                    refined_response_generator = self.response_refiner.refine_response_stream(
                        raw_response_generator, query, 'semantic'
                    )
                    return {
                        'query': query,
                        'analysis': query_analysis,
                        'search_metadata': search_metadata,
                        'context_posts': len(search_results),
                        'response_stream': refined_response_generator,
                        'processing_time': time.time() - start_time,
                        'query_type': 'semantic',
                        'routing_decision': routing_decision
                    }
                else:
                    raw_response_text = self._get_complete_response(enhanced_prompt)
                    response_text = self.response_refiner.refine_response(raw_response_text, query, 'semantic')
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
                        'processing_time': processing_time,
                        'routing_decision': routing_decision
                    }
                
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                'query': query,
                'error': str(e),
                'processing_time': time.time() - start_time
            }
    
    def _has_meaningful_analytical_result(self, analytical_result: Dict) -> bool:
        """Check if analytical result contains meaningful content (not just metadata).
        
        Args:
            analytical_result: Result from analytical processing
            
        Returns:
            True if result contains meaningful content, False otherwise
        """
        result_type = analytical_result.get('type', '')
        
        # These are meaningful analytical results
        if result_type in ['participant_analysis', 'positional_analysis', 'engagement_analysis']:
            return True
        
        # Technical specifications should have actual content
        if result_type == 'technical_specifications':
            # Check if we found actual posts with technical content
            relevant_posts = analytical_result.get('relevant_posts', [])
            return len(relevant_posts) > 0
        
        # Content statistics are meaningful 
        if result_type == 'content_statistics':
            return True
            
        # Default to false for unknown or empty types
        return False
    
    def _build_context(self, search_results: List[Dict], query_analysis: Dict) -> str:
        """Build enhanced context from search results with topic-aware narratives.
        
        Args:
            search_results: Results from semantic search
            query_analysis: Query analysis results
            
        Returns:
            Enhanced context string with topic narratives and metadata
        """
        if not search_results:
            return "No relevant posts found in this thread."
        
        context_parts = []
        matched_topics = set()
        
        # Load thread narratives for topic context
        narrative_data = self._load_thread_narratives()
        
        for i, post in enumerate(search_results, 1):
            author = post.get('author', 'Unknown')
            content = post.get('content', '')
            date = post.get('date', 'Unknown date')
            similarity = post.get('similarity_score', 0)
            page = post.get('page', 1)
            post_position = post.get('global_position', 0)
            url = post.get('url', '')
            
            # Find which topic this post belongs to
            matching_topic = self._find_matching_topic(post_position, narrative_data)
            if matching_topic:
                matched_topics.add(matching_topic['topic'])
                topic_info = f" | Topic: {matching_topic['topic']}"
            else:
                topic_info = ""
            
            # Format post for context with topic information
            post_header = f"Post {i} | Author: {author} | Page: {page} | Date: {date} | Relevance: {similarity:.2f}{topic_info}"
            if url:
                post_header += f" | [View Post]({url})"
            
            post_content = content[:1500]  # Limit content length
            
            if len(content) > 1500:
                post_content += "\n[... content truncated ...]"
            
            context_parts.append(f"{post_header}\n{'-' * 60}\n{post_content}\n")
        
        # Add relevant topic narratives at the top for context
        topic_context = self._build_topic_context(matched_topics, narrative_data, query_analysis)
        
        # Add enhanced thread-level context if available
        thread_context = ""
        if hasattr(self.search_engine, 'thread_analytics') and self.search_engine.thread_analytics:
            analytics = self.search_engine.thread_analytics
            metadata = analytics.get('metadata', {})
            participants = analytics.get('participants', {})
            topics = analytics.get('topics', {})
            
            if metadata:
                thread_context = (
                    f"\nTHREAD OVERVIEW:\n"
                    f"Total Posts: {metadata.get('total_posts', 'Unknown')}\n"
                    f"Participants: {participants.get('total_participants', 'Unknown')}\n"
                    f"Pages: {metadata.get('total_pages', 'Unknown')}\n"
                )
                
                # Add thread creator if available (highest priority for authorship questions)
                thread_creator = metadata.get('thread_creator')
                if thread_creator:
                    creator_name = thread_creator.get('username', 'Unknown')
                    thread_context += f"Thread Creator: {creator_name} (extracted from URL)\n"
                
                # Add most active contributor from participants data
                authors = participants.get('authors', {})
                if authors:
                    # Find most active author by post count
                    most_active_author = max(authors.items(), key=lambda x: x[1].get('post_count', 0))
                    author_name, author_data = most_active_author
                    post_count = author_data.get('post_count', 0)
                    if post_count > 0:
                        thread_context += f"Most Active: {author_name} ({post_count} posts)\n"
                
                # Add key topics  
                keywords = topics.get('primary_keywords', [])
                if keywords:
                    thread_context += f"Main Topics: {', '.join(keywords[:5])}\n"
                
                # Add search coverage info
                total_posts = metadata.get('total_posts', 0)
                if total_posts > 0:
                    coverage_pct = (len(search_results) / total_posts) * 100
                    thread_context += f"Search Coverage: {len(search_results)}/{total_posts} posts ({coverage_pct:.1f}%)\n"
        
        context = thread_context + topic_context + "\nRELEVANT POSTS:\n" + "\n".join(context_parts)
        
        # Add metadata priority guidance for authorship queries
        query = query_analysis.get('original_query', '').lower()
        if any(indicator in query for indicator in [
            'thread author', 'thread creator', 'who created', 'who started',
            'original poster', 'op', 'thread starter', 'who made this thread'
        ]):
            if hasattr(self.search_engine, 'thread_analytics') and self.search_engine.thread_analytics:
                metadata = self.search_engine.thread_analytics.get('metadata', {})
                thread_creator = metadata.get('thread_creator')
                if thread_creator:
                    creator_name = thread_creator.get('username', 'Unknown')
                    context = (
                        f"\n🎯 METADATA PRIORITY: For thread authorship questions, use the Thread Creator from metadata: {creator_name}\n"
                        f"This information is extracted from the canonical URL and has highest confidence.\n"
                        f"Do not infer authorship from post frequency or content analysis.\n\n"
                    ) + context
        
        logger.debug(f"Built enhanced context with {len(search_results)} posts, {len(matched_topics)} topics ({len(context)} characters)")
        return context
    
    def _load_thread_narratives(self) -> Dict:
        """Load thread narratives from thread summary file."""
        try:
            summary_file = f"{self.thread_dir}/thread_summary.json"
            summary_data = safe_read_json(summary_file)
            if summary_data and 'narrative' in summary_data:
                narrative = summary_data['narrative']
                if 'narrative_sections' in narrative:
                    return {'narrative_sections': narrative['narrative_sections']}
            return {'narrative_sections': []}
        except Exception as e:
            logger.debug(f"Could not load thread narratives: {e}")
            return {'narrative_sections': []}
    
    def _find_matching_topic(self, post_position: int, narrative_data: Dict) -> Optional[Dict]:
        """Find which topic a post belongs to based on its position."""
        if not narrative_data or not narrative_data.get('narrative_sections'):
            return None
        
        for section in narrative_data['narrative_sections']:
            phase_summary = section.get('phase_summary', {})
            post_range = phase_summary.get('post_range', '')
            
            # Parse post range like "posts 1-24" 
            if 'posts' in post_range:
                try:
                    range_part = post_range.split('posts ')[1]
                    if '-' in range_part:
                        start_pos, end_pos = map(int, range_part.split('-'))
                        if start_pos <= post_position <= end_pos:
                            return {
                                'topic': phase_summary.get('topic', 'General'),
                                'title': section.get('topic_title', 'Discussion'),
                                'narrative': section.get('narrative_text', ''),
                                'first_post_url': section.get('first_post_url', ''),
                                'keywords': section.get('topic_keywords', [])
                            }
                except (ValueError, IndexError):
                    continue
        
        return None
    
    def _build_topic_context(self, matched_topics: set, narrative_data: Dict, query_analysis: Dict) -> str:
        """Build topic context from matched topics and their narratives."""
        if not matched_topics or not narrative_data.get('narrative_sections'):
            return ""
        
        topic_context = "\n📋 RELEVANT TOPIC NARRATIVES:\n"
        topic_context += "=" * 50 + "\n"
        
        # Find narratives for matched topics
        for section in narrative_data['narrative_sections']:
            phase_summary = section.get('phase_summary', {})
            topic = phase_summary.get('topic', 'General')
            
            if topic in matched_topics:
                title = section.get('topic_title', f"{topic} Discussion")
                narrative = section.get('narrative_text', '')
                post_range = phase_summary.get('post_range', '')
                page_range = phase_summary.get('page_range', '')
                first_post_url = section.get('first_post_url', '')
                
                topic_context += f"\n🔍 **{title}** ({post_range}, {page_range})\n"
                if first_post_url:
                    topic_context += f"📎 [Jump to topic start]({first_post_url})\n"
                
                if narrative:
                    topic_context += f"📖 {narrative}\n"
                
                # Add topic keywords for search context
                keywords = section.get('topic_keywords', [])
                if keywords:
                    topic_context += f"🏷️ Key terms: {', '.join(keywords[:5])}\n"
                
                topic_context += "-" * 40 + "\n"
        
        topic_context += "\n"
        return topic_context
    
    def _enhance_query_with_topics(self, original_query: str, query_analysis: Dict) -> str:
        """Enhance search query with relevant topic keywords from narratives."""
        try:
            narrative_data = self._load_thread_narratives()
            if not narrative_data.get('narrative_sections'):
                return original_query
            
            # Extract all topic keywords from narratives
            all_topic_keywords = set()
            for section in narrative_data['narrative_sections']:
                keywords = section.get('topic_keywords', [])
                all_topic_keywords.update(keywords)
            
            # Find keywords that match or relate to the query
            query_lower = original_query.lower()
            matching_keywords = []
            
            for keyword in all_topic_keywords:
                if (len(keyword) > 3 and 
                    (keyword in query_lower or 
                     any(word in keyword for word in query_lower.split() if len(word) > 3))):
                    matching_keywords.append(keyword)
            
            # Enhance query with top 3 matching keywords
            if matching_keywords:
                enhanced_query = f"{original_query} {' '.join(matching_keywords[:3])}"
                logger.info(f"Enhanced query with topic keywords: '{original_query}' → '{enhanced_query}'")
                return enhanced_query
            
            return original_query
            
        except Exception as e:
            logger.debug(f"Query enhancement failed: {e}")
            return original_query
    
    def _boost_topic_relevant_results(self, search_results: List[Dict], query_analysis: Dict) -> List[Dict]:
        """Boost search results that are from topics relevant to the query."""
        try:
            narrative_data = self._load_thread_narratives()
            if not narrative_data.get('narrative_sections'):
                return search_results
            
            query_lower = query_analysis.get('original_query', '').lower()
            
            # Score boost for posts from relevant topics
            for post in search_results:
                post_position = post.get('global_position', 0)
                matching_topic = self._find_matching_topic(post_position, narrative_data)
                
                if matching_topic:
                    # Check if topic keywords match the query
                    topic_keywords = matching_topic.get('keywords', [])
                    topic_name = matching_topic.get('topic', '').lower()
                    
                    boost_score = 0
                    
                    # Boost if topic name appears in query
                    if topic_name in query_lower:
                        boost_score += 0.1
                    
                    # Boost if topic keywords match query terms
                    query_words = set(query_lower.split())
                    keyword_matches = len(set(topic_keywords) & query_words)
                    if keyword_matches > 0:
                        boost_score += keyword_matches * 0.05
                    
                    # Apply boost to similarity score
                    if boost_score > 0:
                        current_score = post.get('similarity_score', 0)
                        post['similarity_score'] = min(1.0, current_score + boost_score)
                        post['topic_boosted'] = True
                        logger.debug(f"Boosted post {post_position} from topic '{topic_name}' by {boost_score:.3f}")
            
            # Re-sort by boosted scores
            search_results.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
            return search_results
            
        except Exception as e:
            logger.debug(f"Topic boosting failed: {e}")
            return search_results
    
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
            'search_engine_stats': self.search_engine.get_stats(),
            'keyword_search_stats': self.keyword_search.get_stats(),
            'response_refiner_stats': self.response_refiner.get_stats(),
            'llm_router_stats': self.llm_router.get_stats()
        }
    
    def _get_thread_metadata(self) -> Dict:
        """Get thread metadata for LLM routing context."""
        try:
            # Get basic thread stats
            posts_count = len(self.search_engine.posts) if hasattr(self.search_engine, 'posts') else 0
            
            # Try to get analytics data if available
            analytics_file = f"{self.thread_dir}/thread_analytics.json"
            analytics = safe_read_json(analytics_file)
            
            if analytics and 'metadata' in analytics:
                metadata = analytics['metadata']
                participants = analytics.get('participants', {})
                return {
                    'total_posts': metadata.get('total_posts', posts_count),
                    'participants': participants.get('total_participants', 'unknown'),
                    'pages': metadata.get('total_pages', 'unknown')
                }
            else:
                return {
                    'total_posts': posts_count,
                    'participants': 'unknown',
                    'pages': 'unknown'
                }
        except Exception as e:
            logger.warning(f"Could not load thread metadata: {e}")
            return {
                'total_posts': 'unknown',
                'participants': 'unknown', 
                'pages': 'unknown'
            }
    
    def _generate_analytical_response_stream(self, analytical_result: Dict, verification_report: Dict = None):
        """Generate streaming response from analytical results with verification.
        
        Args:
            analytical_result: Results from analytical data processing
            verification_report: Verification report with evidence
            
        Yields:
            Response chunks
        """
        response_text = self._generate_analytical_response_text(analytical_result, verification_report)
        
        # For analytical responses, yield the complete formatted text at once
        # to avoid breaking markdown formatting during streaming
        yield response_text
    
    def _generate_analytical_response_text(self, analytical_result: Dict, verification_report: Dict = None) -> str:
        """Generate response text from analytical results with verification.
        
        Args:
            analytical_result: Results from analytical data processing
            verification_report: Verification report with evidence
            
        Returns:
            Formatted response text with citations
        """
        result_type = analytical_result.get('type', 'unknown')
        
        if result_type == 'participant_analysis':
            return self._format_participant_analysis(analytical_result, verification_report)
        elif result_type == 'content_statistics':
            return self._format_content_statistics(analytical_result, verification_report)
        elif result_type == 'temporal_analysis':
            return self._format_temporal_analysis(analytical_result, verification_report)
        elif result_type == 'positional_analysis':
            return self._format_positional_analysis(analytical_result, verification_report)
        elif result_type == 'engagement_analysis':
            return self._format_engagement_analysis(analytical_result, verification_report)
        elif result_type == 'technical_specifications':
            return self._format_technical_specifications(analytical_result, verification_report)
        else:
            return f"Analysis complete. Result type: {result_type}"
    
    def _format_participant_analysis(self, result: Dict, verification_report: Dict = None) -> str:
        """Format participant analysis results with verification."""
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
        
        # Add verification information if available
        if verification_report:
            response_parts.append(self._format_verification_section(verification_report))
        
        return ''.join(response_parts)
    
    def _format_verification_section(self, verification_report: Dict) -> str:
        """Format verification section for responses."""
        if not verification_report or not verification_report.get('verifiable_claims'):
            return ""
        
        verification_parts = ["\n\n## 📋 Verification & Evidence\n"]
        
        confidence = verification_report.get('confidence_assessment', 'medium')
        confidence_emoji = {'high': '🟢', 'medium': '🟡', 'low': '🔴'}.get(confidence, '🟡')
        
        verification_parts.append(f"**Confidence Level**: {confidence_emoji} {confidence.title()}\n\n")
        
        # Add verifiable claims with evidence
        for i, claim in enumerate(verification_report.get('verifiable_claims', []), 1):
            evidence_list = claim.get('evidence', [])
            if evidence_list:
                verification_parts.append(f"**Evidence {i}**: {claim.get('verification_summary', '')}\n")
                
                # Show top 2 pieces of evidence
                for evidence in evidence_list[:2]:
                    citation = evidence.get('citation', '')
                    quote = evidence.get('quote', '')
                    
                    if citation and quote:
                        verification_parts.append(f"- {citation}: \"{quote[:100]}{'...' if len(quote) > 100 else ''}\"\n")
                    elif citation:
                        verification_parts.append(f"- {citation}\n")
                
                verification_parts.append("\n")
        
        return ''.join(verification_parts)
    
    def _format_content_statistics(self, result: Dict, verification_report: Dict = None) -> str:
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
        
        # Add verification information if available
        if verification_report:
            response_parts.append(self._format_verification_section(verification_report))
        
        return ''.join(response_parts)
    
    def _format_temporal_analysis(self, result: Dict, verification_report: Dict = None) -> str:
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
        
        # Add verification information if available
        if verification_report:
            response_parts.append(self._format_verification_section(verification_report))
        
        return ''.join(response_parts)
    
    def _format_engagement_analysis(self, result: Dict, verification_report: Dict = None) -> str:
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
        
        # Add verification information if available
        if verification_report:
            response_parts.append(self._format_verification_section(verification_report))
        
        return ''.join(response_parts)
    
    def _format_positional_analysis(self, result: Dict, verification_report: Dict = None) -> str:
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
        
        # Add verification information if available
        if verification_report:
            response_parts.append(self._format_verification_section(verification_report))
        
        return ''.join(response_parts)
    
    def _format_technical_specifications(self, result: Dict, verification_report: Dict = None) -> str:
        """Format technical specifications analysis results with clean structure."""
        if 'error' in result:
            return f"**Analysis Error:**\n\n{result['error']}\n\nTotal posts analyzed: {result.get('total_posts_analyzed', 0)}"
        
        spec_type = result.get('spec_type', 'settings')
        common_settings = result.get('common_settings', [])
        top_posts = result.get('top_posts', [])
        
        response_parts = []
        
        # Main answer first - most common settings
        if common_settings:
            response_parts.append(f"## Most Common {spec_type.title()} Settings:\n\n")
            for i, setting_data in enumerate(common_settings[:5], 1):
                setting = setting_data['setting']
                mentions = setting_data['mentions']
                response_parts.append(f"**{i}. {setting}** ({mentions} mentions)\n")
            response_parts.append("\n")
        
        # Collapsible detailed analysis section
        response_parts.append("<details>\n")
        response_parts.append("<summary><strong>📋 View Detailed Post Analysis</strong></summary>\n\n")
        
        # Show top relevant posts in collapsible section
        if top_posts:
            response_parts.append(f"### Top Community Posts About {spec_type.title()}:\n\n")
            for i, post in enumerate(top_posts[:3], 1):
                response_parts.append(f"**{i}. {post.get('author', 'Unknown')}** (Page {post.get('page', 1)})")
                if post.get('engagement', 0) > 0:
                    response_parts.append(f" - {post['engagement']} ⬆️")
                response_parts.append("\n")
                
                # Show specific values found  
                if 'spec_values' in post and post['spec_values']:
                    unique_values = list(set(post['spec_values']))  # Remove duplicates
                    response_parts.append(f"• **Settings**: {', '.join(unique_values[:5])}")  # Limit to 5
                    if len(post['spec_values']) > 5:
                        response_parts.append(f" (+{len(post['spec_values'])-5} more)")
                    response_parts.append("\n")
                
                # Show content preview
                preview = post.get('content_preview', '')
                if preview:
                    response_parts.append(f"• **Quote**: \"{preview[:150]}{'...' if len(preview) > 150 else ''}\"\n")
                
                # Add post link if available
                if post.get('post_url'):
                    response_parts.append(f"• **Link**: {post['post_url']}\n")
                elif post.get('post_id'):
                    response_parts.append(f"• **Post ID**: {post['post_id']}\n")
                
                response_parts.append("\n")
        
        # Summary stats in collapsible section
        response_parts.append("### Analysis Summary:\n")
        response_parts.append(f"- Found **{result.get('relevant_posts_count', 0)} relevant posts** discussing {spec_type}\n")
        response_parts.append(f"- Identified **{result.get('settings_found', 0)} different {spec_type} values**\n")
        response_parts.append(f"- Analyzed **{result.get('total_posts_analyzed', 0)} total posts**\n")
        
        response_parts.append("\n</details>")
        
        # Add verification information if available
        if verification_report:
            response_parts.append(self._format_verification_section(verification_report))
        
        return ''.join(response_parts)
    
    def _get_verification_report(self, response_data: Dict) -> Dict:
        """Get verification report with lazy loading."""
        if self.verifiable_response is None:
            try:
                from search.verifiable_response_system import VerifiableResponseSystem
                self.verifiable_response = VerifiableResponseSystem(self.search_engine.posts)
            except ImportError:
                logger.warning("Verifiable response system not available")
                return {'error': 'Verification system not available'}
            except Exception as e:
                logger.warning(f"Failed to initialize verification system: {e}")
                return {'error': f'Verification system error: {e}'}
        
        return self.verifiable_response.generate_fact_check_report(response_data)
    
    def _search_topic_index(self, query: str) -> Dict:
        """Search topic index for relevant matches.
        
        Args:
            query: User query string
            
        Returns:
            Dictionary with topic search results
        """
        try:
            # Check if topic index exists for this thread
            if not self.topic_cache.has_topic_index(self.thread_key):
                logger.debug(f"No topic index found for thread {self.thread_key}")
                return {'topics_found': [], 'total_matches': 0, 'search_method': 'topic_index'}
            
            # Find the most relevant topic for the query
            relevant_topic = self.topic_indexer.get_topic_by_query(query)
            
            if not relevant_topic:
                logger.debug(f"No relevant topic found for query: {query}")
                return {'topics_found': [], 'total_matches': 0, 'search_method': 'topic_index'}
            
            # Get topic matches for the thread
            topic_matches = self.topic_cache.get_topic_matches_for_thread(self.thread_key, relevant_topic)
            
            if not topic_matches:
                logger.debug(f"No matches found for topic {relevant_topic} in thread {self.thread_key}")
                return {'topics_found': [], 'total_matches': 0, 'search_method': 'topic_index'}
            
            # Sort matches by relevance score
            sorted_matches = sorted(topic_matches, key=lambda x: x.get('score', 0), reverse=True)
            
            logger.info(f"Topic index search found {len(sorted_matches)} matches for topic '{relevant_topic}'")
            
            return {
                'topics_found': [relevant_topic],
                'primary_topic': relevant_topic,
                'matches': sorted_matches[:10],  # Top 10 matches
                'total_matches': len(sorted_matches),
                'search_method': 'topic_index'
            }
            
        except Exception as e:
            logger.error(f"Error searching topic index: {e}")
            return {'topics_found': [], 'total_matches': 0, 'search_method': 'topic_index', 'error': str(e)}
    
    def _convert_topic_matches_to_search_results(self, topic_matches: List[Dict]) -> List[Dict]:
        """Convert topic matches to standard search results format."""
        results = []
        for match in topic_matches:
            result = {
                'post_id': match.get('post_id'),
                'page': match.get('page'),
                'author': match.get('author'),
                'content': match.get('excerpt'),
                'url': match.get('permalink'),
                'relevance_score': match.get('score', 0),
                'matched_keywords': match.get('matched_keywords', []),
                'topic_id': match.get('topic_id'),
                'source': 'topic_index'
            }
            results.append(result)
        return results
    
    def _generate_topic_based_response_stream(self, query: str, topic_context: Dict, search_results: List[Dict]):
        """Generate streaming response using topic context."""
        topic_name = topic_context.get('primary_topic', 'Unknown')
        total_matches = topic_context.get('total_matches', 0)
        
        # Build context for LLM
        context_posts = []
        for result in search_results[:5]:  # Top 5 for context
            post_text = f"**{result['author']} (Page {result['page']})**: {result['content']}"
            if result.get('url'):
                post_text += f" [Link]({result['url']})"
            context_posts.append(post_text)
        
        # Create prompt with topic context
        prompt = f"""Based on the following posts about '{topic_name}' from a forum discussion, please answer this question: "{query}"

**Relevant Posts ({total_matches} total matches found):**

{chr(10).join(context_posts)}

Please provide a comprehensive answer based on these posts. Include specific details and mention the sources when relevant."""

        # Stream response from LLM
        try:
            response = requests.post(
                self.chat_url,
                json={
                    'model': self.model,
                    'messages': [{'role': 'user', 'content': prompt}],
                    'stream': True,
                    'options': {'temperature': 0.7}
                },
                stream=True,
                timeout=60
            )
            
            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            if 'message' in data and 'content' in data['message']:
                                yield data['message']['content']
                        except json.JSONDecodeError:
                            continue
            else:
                yield f"Error: Failed to get response from LLM (status {response.status_code})"
                
        except Exception as e:
            yield f"Error generating topic-based response: {str(e)}"

# Old query detection methods removed - now using LLM-based routing


__all__ = ['QueryProcessor']