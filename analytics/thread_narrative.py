"""
Advanced chronological thread summarization with topic detection and analytics.

This module creates running narratives of forum threads with topic shifts,
reaction-based highlighting, and comprehensive analytics for single-run processing.
"""

import logging
import json
import time
import re
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Tuple, Set
import requests

from config.settings import OLLAMA_BASE_URL, OLLAMA_CHAT_MODEL
from analytics.thread_analyzer import ThreadAnalyzer
from utils.file_utils import atomic_write_json

logger = logging.getLogger(__name__)


class ThreadNarrative:
    """Creates comprehensive thread narratives with topic detection and analytics."""
    
    def __init__(self):
        """Initialize the narrative generator."""
        self.chat_url = f"{OLLAMA_BASE_URL}/api/chat"
        self.model = OLLAMA_CHAT_MODEL
        
    def generate_narrative_and_analytics(self, thread_dir: str, posts: List[Dict]) -> Dict:
        """Generate both narrative summary and analytics in a single run."""
        start_time = time.time()
        
        # Sort posts chronologically
        sorted_posts = sorted(posts, key=lambda x: x.get('global_position', 0))
        
        # Generate analytics using existing analyzer
        analyzer = ThreadAnalyzer(thread_dir)
        analytics = analyzer.analyze_thread(sorted_posts, force_refresh=True)
        
        # Generate narrative summary
        narrative_data = self._generate_chronological_narrative(sorted_posts, analytics)
        
        # Combine results
        combined_result = {
            'narrative': narrative_data,
            'analytics': analytics,
            'generation_metadata': {
                'generated_at': time.time(),
                'processing_time': time.time() - start_time,
                'total_posts': len(sorted_posts),
                'method': 'single_run_comprehensive'
            }
        }
        
        # Save to thread directory
        output_file = f"{thread_dir}/thread_summary.json"
        atomic_write_json(output_file, combined_result)
        
        logger.info(f"Generated narrative and analytics in {time.time() - start_time:.2f}s")
        return combined_result
    
    def _generate_chronological_narrative(self, posts: List[Dict], analytics: Dict) -> Dict:
        """Generate chronological narrative with topic detection."""
        logger.info("Detecting conversation phases...")
        # Detect conversation phases
        phases = self._detect_conversation_phases(posts)
        logger.info(f"Detected {len(phases)} conversation phases")
        
        logger.info("Identifying high-reaction posts...")
        # Identify high-reaction posts
        reaction_posts = self._identify_reaction_posts(posts)
        logger.info(f"Found {len(reaction_posts)} high-reaction posts")
        
        logger.info("Generating narrative sections with LLM...")
        # Generate narrative sections
        narrative_sections = []
        from tqdm import tqdm
        with tqdm(total=len(phases), desc="Generating narratives", unit="phase") as pbar:
            for i, phase in enumerate(phases, 1):
                topic = phase.get('topic', 'Unknown')
                pbar.set_description(f"Generating narrative: {topic}")
                logger.info(f"Generating narrative for phase {i}/{len(phases)}: {topic}")
                section = self._generate_phase_narrative(phase, posts, reaction_posts)
                narrative_sections.append(section)
                pbar.update(1)
        
        logger.info("Creating overall thread summary with LLM...")
        # Create overall thread summary
        thread_summary = self._generate_thread_overview(posts, phases, analytics)
        
        return {
            'thread_overview': thread_summary,
            'conversation_phases': phases,
            'narrative_sections': narrative_sections,
            'high_reaction_posts': reaction_posts,
            'topic_evolution': self._analyze_topic_evolution(phases),
            'key_contributors': self._identify_key_contributors(posts, phases)
        }
    
    def _detect_conversation_phases(self, posts: List[Dict]) -> List[Dict]:
        """Detect major conversation phases using content analysis."""
        phases = []
        current_phase = None
        phase_posts = []
        
        # Keywords that indicate topic shifts
        topic_indicators = {
            'design': ['design', 'prototype', 'concept', 'blueprint', 'model', 'version'],
            'shipping': ['ship', 'delivery', 'order', 'purchase', 'buy', 'price', 'cost'],
            'reviews': ['review', 'test', 'experience', 'opinion', 'feedback', 'impression'],
            'technical': ['specs', 'specification', 'technical', 'measurement', 'dimension'],
            'troubleshooting': ['problem', 'issue', 'fix', 'broken', 'error', 'help'],
            'community': ['meet', 'group', 'community', 'together', 'social']
        }
        
        for i, post in enumerate(posts):
            content = post.get('content', '').lower()
            
            # Analyze content for topic indicators
            topic_scores = {}
            for topic, keywords in topic_indicators.items():
                score = sum(1 for keyword in keywords if keyword in content)
                if score > 0:
                    topic_scores[topic] = score
            
            # Determine dominant topic
            dominant_topic = max(topic_scores.keys(), key=lambda k: topic_scores[k]) if topic_scores else 'general'
            
            # Check for phase change
            if current_phase != dominant_topic and len(phase_posts) >= 5:
                # End current phase
                if phase_posts:
                    phases.append(self._create_phase_summary(current_phase, phase_posts, posts))
                
                # Start new phase
                current_phase = dominant_topic
                phase_posts = [post]
            else:
                if current_phase is None:
                    current_phase = dominant_topic
                phase_posts.append(post)
        
        # Add final phase
        if phase_posts:
            phases.append(self._create_phase_summary(current_phase, phase_posts, posts))
        
        return phases
    
    def _create_phase_summary(self, topic: str, phase_posts: List[Dict], all_posts: List[Dict]) -> Dict:
        """Create summary for a conversation phase."""
        start_pos = phase_posts[0].get('global_position', 0)
        end_pos = phase_posts[-1].get('global_position', 0)
        
        # Calculate page range
        start_page = phase_posts[0].get('page', 1)
        end_page = phase_posts[-1].get('page', 1)
        
        # Get key participants
        authors = Counter(post.get('author', 'Unknown') for post in phase_posts)
        
        # Calculate engagement
        total_reactions = sum(
            post.get('upvotes', 0) + post.get('likes', 0) + post.get('reactions', 0)
            for post in phase_posts
        )
        
        return {
            'topic': topic.title(),
            'post_range': f"posts {start_pos}-{end_pos}",
            'page_range': f"pages {start_page}-{end_page}" if start_page != end_page else f"page {start_page}",
            'post_count': len(phase_posts),
            'key_participants': dict(authors.most_common(3)),
            'total_engagement': total_reactions,
            'start_date': phase_posts[0].get('date', ''),
            'end_date': phase_posts[-1].get('date', ''),
            'posts': phase_posts[:5]  # Store sample posts for narrative generation
        }
    
    def _identify_reaction_posts(self, posts: List[Dict]) -> List[Dict]:
        """Identify posts with high community engagement."""
        scored_posts = []
        
        for post in posts:
            # Calculate engagement score
            upvotes = post.get('upvotes', 0)
            likes = post.get('likes', 0)
            reactions = post.get('reactions', 0)
            replies = post.get('reply_count', 0)
            content_length = len(post.get('content', ''))
            
            # Weighted scoring
            engagement_score = (upvotes * 3) + (likes * 2) + reactions + (replies * 2)
            
            # Bonus for substantial content
            if content_length > 200:
                engagement_score += 2
            
            if engagement_score > 0:
                post_copy = post.copy()
                post_copy['engagement_score'] = engagement_score
                scored_posts.append(post_copy)
        
        # Return top 10 most engaging posts
        return sorted(scored_posts, key=lambda x: x['engagement_score'], reverse=True)[:10]
    
    def _generate_phase_narrative(self, phase: Dict, all_posts: List[Dict], reaction_posts: List[Dict]) -> Dict:
        """Generate narrative text for a conversation phase."""
        topic = phase['topic']
        sample_posts = phase.get('posts', [])
        
        # Build context for LLM
        context_parts = [
            f"Topic: {topic}",
            f"Time period: {phase.get('start_date', '')} to {phase.get('end_date', '')}",
            f"Page range: {phase['page_range']}",
            f"Key participants: {', '.join(phase['key_participants'].keys())}"
        ]
        
        # Format sample posts
        post_samples = []
        for post in sample_posts[:3]:
            author = post.get('author', 'Unknown')
            content = post.get('content', '')[:300]
            post_samples.append(f"{author}: {content}...")
        
        # Generate narrative using LLM
        prompt = f"""
Summarize this conversation phase in 2-3 sentences focusing on the main developments:

{' | '.join(context_parts)}

Sample posts:
{chr(10).join(post_samples)}

Create a concise narrative about what happened in this phase."""
        
        try:
            narrative_text = self._get_llm_response(prompt)
        except Exception as e:
            logger.warning(f"Failed to generate LLM narrative for {topic}: {e}")
            narrative_text = f"Discussion about {topic.lower()} with {phase['post_count']} posts from key participants."
        
        return {
            'phase_summary': phase,
            'narrative_text': narrative_text,
            'highlights': [post for post in reaction_posts if any(
                p.get('global_position') == post.get('global_position') 
                for p in sample_posts
            )]
        }
    
    def _generate_thread_overview(self, posts: List[Dict], phases: List[Dict], analytics: Dict) -> str:
        """Generate high-level thread overview."""
        total_posts = len(posts)
        total_participants = len(set(post.get('author', '') for post in posts))
        
        # Get date range
        first_date = posts[0].get('date', '') if posts else ''
        last_date = posts[-1].get('date', '') if posts else ''
        
        # Summarize phases
        phase_summary = []
        for phase in phases:
            phase_summary.append(f"{phase['topic']} ({phase['page_range']})")
        
        overview_prompt = f"""
Create a 3-4 sentence overview of this forum thread:

Thread stats: {total_posts} posts from {total_participants} participants
Date range: {first_date} to {last_date}
Main topics: {', '.join(phase_summary)}

Focus on what the thread accomplished and key outcomes."""
        
        try:
            return self._get_llm_response(overview_prompt)
        except Exception as e:
            logger.warning(f"Failed to generate thread overview: {e}")
            return f"Forum discussion with {total_posts} posts covering topics: {', '.join(p['topic'] for p in phases)}"
    
    def _analyze_topic_evolution(self, phases: List[Dict]) -> List[Dict]:
        """Analyze how topics evolved throughout the thread."""
        evolution = []
        
        for i, phase in enumerate(phases):
            evolution_entry = {
                'sequence': i + 1,
                'topic': phase['topic'],
                'page_range': phase['page_range'],
                'participant_count': len(phase['key_participants']),
                'engagement_level': phase['total_engagement']
            }
            
            if i > 0:
                prev_phase = phases[i-1]
                evolution_entry['transition_from'] = prev_phase['topic']
                evolution_entry['engagement_change'] = phase['total_engagement'] - prev_phase['total_engagement']
            
            evolution.append(evolution_entry)
        
        return evolution
    
    def _identify_key_contributors(self, posts: List[Dict], phases: List[Dict]) -> List[Dict]:
        """Identify key contributors across different phases."""
        contributor_stats = defaultdict(lambda: {
            'post_count': 0,
            'total_engagement': 0,
            'phases_active': set(),
            'first_post': None,
            'last_post': None
        })
        
        for post in posts:
            author = post.get('author', 'Unknown')
            if author == 'Unknown':
                continue
                
            stats = contributor_stats[author]
            stats['post_count'] += 1
            stats['total_engagement'] += (
                post.get('upvotes', 0) + 
                post.get('likes', 0) + 
                post.get('reactions', 0)
            )
            
            if stats['first_post'] is None:
                stats['first_post'] = post.get('global_position', 0)
            stats['last_post'] = post.get('global_position', 0)
        
        # Add phase activity
        for phase in phases:
            for author in phase['key_participants']:
                if author in contributor_stats:
                    contributor_stats[author]['phases_active'].add(phase['topic'])
        
        # Convert to list and sort by influence
        contributors = []
        for author, stats in contributor_stats.items():
            influence_score = (
                stats['post_count'] * 2 + 
                stats['total_engagement'] + 
                len(stats['phases_active']) * 3
            )
            
            contributors.append({
                'author': author,
                'post_count': stats['post_count'],
                'total_engagement': stats['total_engagement'],
                'phases_active': list(stats['phases_active']),
                'influence_score': influence_score,
                'post_range': f"{stats['first_post']}-{stats['last_post']}"
            })
        
        return sorted(contributors, key=lambda x: x['influence_score'], reverse=True)[:10]
    
    def _get_llm_response(self, prompt: str) -> str:
        """Get response from LLM."""
        try:
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False
            }
            
            response = requests.post(self.chat_url, json=payload, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            return result.get('message', {}).get('content', '').strip()
            
        except Exception as e:
            logger.error(f"Error getting LLM response: {e}")
            raise


__all__ = ['ThreadNarrative']