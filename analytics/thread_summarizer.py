"""
Advanced LLM-powered summarization for Forum Wisdom Miner.

This module provides intelligent thread summarization by identifying key posts
using the multi-factor ranking system and generating comprehensive summaries
using the OLLAMA_CHAT_MODEL.
"""

import logging
import json
import time
from typing import Dict, List, Optional, Tuple
import requests

from config.settings import OLLAMA_BASE_URL, OLLAMA_CHAT_MODEL
from search.result_ranker import PostRanker
from utils.file_utils import safe_read_json

logger = logging.getLogger(__name__)


class ThreadSummarizer:
    """Advanced thread summarization using multi-factor ranking and LLM analysis."""
    
    def __init__(self):
        """Initialize the thread summarizer."""
        self.chat_url = f"{OLLAMA_BASE_URL}/api/chat"
        self.model = OLLAMA_CHAT_MODEL
        
        # Statistics
        self.stats = {
            'total_summaries': 0,
            'successful_summaries': 0,
            'failed_summaries': 0,
            'avg_processing_time': 0,
            'total_posts_analyzed': 0
        }
    
    def generate_summary(self, thread_dir: str, max_posts: int = 15) -> Dict:
        """Generate a comprehensive thread summary using top-ranked posts.
        
        Args:
            thread_dir: Directory containing thread data
            max_posts: Maximum number of posts to analyze (10-15 recommended)
            
        Returns:
            Dictionary containing summary and metadata
        """
        start_time = time.time()
        self.stats['total_summaries'] += 1
        
        try:
            # Step 1: Load thread data
            posts_data = self._load_thread_posts(thread_dir)
            analytics_data = self._load_thread_analytics(thread_dir)
            
            if not posts_data:
                raise ValueError("No posts found in thread")
            
            # Step 2: Identify top posts using multi-factor ranking
            logger.info(f"Identifying top {max_posts} posts from {len(posts_data)} total posts")
            top_posts = self._get_top_posts(posts_data, analytics_data, max_posts)
            
            if not top_posts:
                raise ValueError("No suitable posts found for summarization")
            
            # Step 3: Generate LLM summary
            logger.info(f"Generating LLM summary from {len(top_posts)} key posts")
            summary_data = self._generate_llm_summary(top_posts, analytics_data)
            
            # Step 4: Add metadata
            processing_time = time.time() - start_time
            summary_data.update({
                'metadata': {
                    'generated_at': time.time(),
                    'processing_time': processing_time,
                    'posts_analyzed': len(top_posts),
                    'total_posts_available': len(posts_data),
                    'thread_directory': thread_dir
                },
                'ranking_details': {
                    'top_posts_count': len(top_posts),
                    'selection_criteria': 'Multi-factor ranking (semantic similarity, recency, votes, authority)',
                    'posts_metadata': [
                        {
                            'author': post.get('author', 'Unknown'),
                            'position': post.get('global_position', 0),
                            'votes': post.get('upvotes', 0) + post.get('likes', 0),
                            'length': len(post.get('content', '')),
                            'ranking_score': post.get('_ranking_score', 0)
                        }
                        for post in top_posts[:5]  # Include top 5 post details
                    ]
                }
            })
            
            # Update statistics
            self.stats['successful_summaries'] += 1
            self.stats['total_posts_analyzed'] += len(top_posts)
            self.stats['avg_processing_time'] = (
                (self.stats['avg_processing_time'] * (self.stats['successful_summaries'] - 1) + processing_time) /
                self.stats['successful_summaries']
            )
            
            logger.info(f"Summary generated successfully in {processing_time:.2f}s")
            return summary_data
            
        except Exception as e:
            self.stats['failed_summaries'] += 1
            logger.error(f"Failed to generate summary: {e}")
            raise
    
    def _load_thread_posts(self, thread_dir: str) -> List[Dict]:
        """Load posts from thread directory."""
        posts_file = f"{thread_dir}/posts.json"
        posts_data = safe_read_json(posts_file)
        
        if not posts_data:
            raise ValueError(f"Could not load posts from {posts_file}")
        
        return posts_data
    
    def _load_thread_analytics(self, thread_dir: str) -> Optional[Dict]:
        """Load analytics from thread directory."""
        analytics_file = f"{thread_dir}/thread_analytics.json"
        return safe_read_json(analytics_file)
    
    def _get_top_posts(self, posts: List[Dict], analytics: Optional[Dict], max_posts: int) -> List[Dict]:
        """Use multi-factor ranking to identify the most important posts."""
        # Initialize the ranker with thread analytics
        ranker = PostRanker(analytics)
        
        # Create pseudo-results for ranking (simulate search results)
        pseudo_results = []
        for i, post in enumerate(posts):
            pseudo_result = {
                **post,
                'similarity_score': 0.8,  # High base similarity for top post selection
                'metadata': {
                    'post_index': i,
                    'original_post': post
                }
            }
            pseudo_results.append(pseudo_result)
        
        # Rank all posts using the comprehensive ranking system
        ranked_results = ranker.rank_results(pseudo_results, query="comprehensive thread summary")
        
        # Select top posts, ensuring diversity
        top_posts = self._ensure_post_diversity(ranked_results[:max_posts * 2], max_posts)
        
        # Add ranking scores for metadata
        for i, post in enumerate(top_posts):
            post['_ranking_score'] = ranked_results[i].get('final_score', 0)
        
        return top_posts
    
    def _ensure_post_diversity(self, ranked_posts: List[Dict], target_count: int) -> List[Dict]:
        """Ensure diversity in selected posts by author and position."""
        selected_posts = []
        used_authors = set()
        position_ranges = set()
        
        for post in ranked_posts:
            if len(selected_posts) >= target_count:
                break
            
            author = post.get('author', 'Unknown')
            position = post.get('global_position', 0)
            position_range = position // 100  # Group by hundreds
            
            # Prefer posts from different authors and different parts of the thread
            author_diversity = author not in used_authors or len(used_authors) < 3
            position_diversity = position_range not in position_ranges or len(position_ranges) < 5
            
            if author_diversity or position_diversity or len(selected_posts) < target_count // 2:
                selected_posts.append(post)
                used_authors.add(author)
                position_ranges.add(position_range)
        
        # Fill remaining slots if needed
        for post in ranked_posts:
            if len(selected_posts) >= target_count:
                break
            if post not in selected_posts:
                selected_posts.append(post)
        
        return selected_posts[:target_count]
    
    def _generate_llm_summary(self, top_posts: List[Dict], analytics: Optional[Dict]) -> Dict:
        """Generate comprehensive summary using LLM."""
        # Prepare context information
        thread_context = self._build_thread_context(analytics)
        posts_content = self._format_posts_for_llm(top_posts)
        
        # Build the summarization prompt
        summary_prompt = f"""
You are an expert forum analysis assistant. Analyze these key posts from a forum thread and provide a comprehensive summary.

THREAD CONTEXT:
{thread_context}

KEY POSTS TO ANALYZE:
{posts_content}

TASK: Create a comprehensive thread summary that covers:

1. **MAIN ARGUMENTS & POSITIONS**: What are the primary viewpoints, debates, or positions discussed?

2. **KEY QUESTIONS & PROBLEMS**: What main questions were asked? What problems were participants trying to solve?

3. **SOLUTIONS & RECOMMENDATIONS**: What solutions, advice, or recommendations emerged from the discussion?

4. **IMPORTANT INSIGHTS**: What valuable insights, tips, or discoveries were shared?

5. **COMMUNITY CONSENSUS**: Where did participants agree? What became accepted wisdom?

6. **ONGOING DEBATES**: What questions remain unresolved or controversial?

FORMATTING REQUIREMENTS:
- Use clear headers for each section
- Include specific examples and quotes when relevant
- Mention key contributors by name when appropriate
- Focus on content substance, not technical forum details
- Keep each section concise but informative (2-4 sentences)

CRITICAL: Focus on the actual discussion content and valuable information shared, not on forum mechanics or post formatting.

Generate the summary:
"""
        
        try:
            # Get LLM response
            llm_response = self._get_llm_response(summary_prompt)
            
            # Parse and structure the response
            summary_sections = self._parse_summary_response(llm_response)
            
            return {
                'summary': {
                    'full_text': llm_response,
                    'sections': summary_sections,
                    'word_count': len(llm_response.split()),
                    'key_topics': self._extract_key_topics(llm_response)
                },
                'source_posts': {
                    'count': len(top_posts),
                    'authors': list(set(post.get('author', 'Unknown') for post in top_posts)),
                    'date_range': self._get_posts_date_range(top_posts),
                    'total_characters': sum(len(post.get('content', '')) for post in top_posts)
                }
            }
            
        except Exception as e:
            logger.error(f"Error in LLM summary generation: {e}")
            raise
    
    def _build_thread_context(self, analytics: Optional[Dict]) -> str:
        """Build context information about the thread."""
        if not analytics:
            return "Thread analytics not available."
        
        metadata = analytics.get('metadata', {})
        participants = analytics.get('participants', {})
        
        context_parts = []
        
        # Basic thread info
        total_posts = metadata.get('total_posts', 'Unknown')
        total_pages = metadata.get('total_pages', 'Unknown')
        context_parts.append(f"Thread size: {total_posts} posts across {total_pages} pages")
        
        # Participant info
        total_participants = participants.get('total_participants', 'Unknown')
        context_parts.append(f"Participants: {total_participants} unique contributors")
        
        # Most active participants
        authors = participants.get('authors', {})
        if authors:
            top_authors = sorted(authors.items(), key=lambda x: x[1].get('post_count', 0), reverse=True)[:3]
            author_names = [name for name, _ in top_authors]
            context_parts.append(f"Most active contributors: {', '.join(author_names)}")
        
        # Date range
        first_post = metadata.get('first_post', {})
        last_post = metadata.get('last_post', {})
        if first_post and last_post:
            first_date = first_post.get('date', 'Unknown')
            last_date = last_post.get('date', 'Unknown')
            context_parts.append(f"Discussion period: {first_date} to {last_date}")
        
        return "\n".join(context_parts)
    
    def _format_posts_for_llm(self, posts: List[Dict]) -> str:
        """Format posts for LLM analysis."""
        formatted_posts = []
        
        for i, post in enumerate(posts, 1):
            author = post.get('author', 'Unknown')
            content = post.get('content', '').strip()
            position = post.get('global_position', 0)
            votes = post.get('upvotes', 0) + post.get('likes', 0)
            
            # Truncate very long posts to keep within token limits
            if len(content) > 800:
                content = content[:800] + "... [truncated]"
            
            post_header = f"POST #{i} (Author: {author}, Position: {position}, Votes: {votes})"
            formatted_posts.append(f"{post_header}\n{content}\n")
        
        return "\n" + "="*80 + "\n".join(formatted_posts) + "="*80
    
    def _get_llm_response(self, prompt: str) -> str:
        """Get response from LLM for summarization."""
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
                timeout=120  # Longer timeout for summarization
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get('message', {}).get('content', '')
            
        except Exception as e:
            logger.error(f"Error getting LLM summarization response: {e}")
            raise
    
    def _parse_summary_response(self, response: str) -> Dict:
        """Parse the LLM response into structured sections."""
        sections = {}
        current_section = None
        current_content = []
        
        lines = response.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if this is a section header
            if any(keyword in line.upper() for keyword in ['MAIN ARGUMENTS', 'KEY QUESTIONS', 'SOLUTIONS', 'INSIGHTS', 'CONSENSUS', 'DEBATES']):
                # Save previous section
                if current_section and current_content:
                    sections[current_section] = '\n'.join(current_content).strip()
                
                # Start new section
                current_section = line.replace('*', '').replace('#', '').strip()
                current_content = []
            else:
                if current_section:
                    current_content.append(line)
        
        # Save final section
        if current_section and current_content:
            sections[current_section] = '\n'.join(current_content).strip()
        
        return sections
    
    def _extract_key_topics(self, summary_text: str) -> List[str]:
        """Extract key topics from the summary text."""
        # Simple keyword extraction (could be enhanced with NLP)
        import re
        
        # Common forum discussion topics (can be expanded)
        topic_patterns = [
            r'\b(?:temperature|temp|wattage|voltage|setting|config)\w*\b',
            r'\b(?:technique|method|approach|way|process)\w*\b',
            r'\b(?:problem|issue|question|concern|challenge)\w*\b',
            r'\b(?:solution|fix|answer|recommendation|advice)\w*\b',
            r'\b(?:opinion|view|thought|experience|feedback)\w*\b'
        ]
        
        topics = set()
        for pattern in topic_patterns:
            matches = re.findall(pattern, summary_text, re.IGNORECASE)
            topics.update(match.lower() for match in matches)
        
        return sorted(list(topics))[:10]  # Return top 10 topics
    
    def _get_posts_date_range(self, posts: List[Dict]) -> Dict:
        """Get the date range of the analyzed posts."""
        dates = []
        for post in posts:
            if post.get('parsed_date'):
                dates.append(post['parsed_date'])
        
        if dates:
            return {
                'earliest': min(dates),
                'latest': max(dates),
                'span_days': (max(dates) - min(dates)).days if len(dates) > 1 else 0
            }
        
        return {'earliest': None, 'latest': None, 'span_days': 0}
    
    def get_stats(self) -> Dict:
        """Get summarization statistics."""
        return self.stats.copy()


__all__ = ['ThreadSummarizer']