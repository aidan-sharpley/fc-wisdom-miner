"""
Advanced chronological thread summarization with topic detection and analytics.
Optimized for M1 MacBook Air with 8GB RAM using batched LLM calls and caching.
"""

import logging
import json
import time
import os
import asyncio
import concurrent.futures
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Tuple, Set
import requests
from tqdm import tqdm

from config.settings import OLLAMA_BASE_URL, OLLAMA_CHAT_MODEL
from analytics.thread_analyzer import ThreadAnalyzer
from utils.file_utils import atomic_write_json

logger = logging.getLogger(__name__)


class ThreadNarrative:
    """Creates comprehensive thread narratives with batched LLM processing."""
    
    def __init__(self):
        self.chat_url = f"{OLLAMA_BASE_URL}/api/chat"
        self.model = OLLAMA_CHAT_MODEL
        self.max_workers = 3
        
    def generate_narrative_and_analytics(self, thread_dir: str, posts: List[Dict]) -> Dict:
        """Generate both narrative summary and analytics in a single run."""
        start_time = time.time()
        
        # Check cache first
        cache_file = f"{thread_dir}/thread_summary.json"
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                    if cached_data.get('generation_metadata', {}).get('total_posts') == len(posts):
                        logger.info(f"Using cached narrative data for {len(posts)} posts")
                        return cached_data
            except Exception:
                pass
        
        sorted_posts = sorted(posts, key=lambda x: x.get('global_position', 0))
        
        analyzer = ThreadAnalyzer(thread_dir)
        analytics = analyzer.analyze_thread(sorted_posts, force_refresh=True)
        
        narrative_data = self._generate_optimized_narrative(sorted_posts, analytics)
        
        combined_result = {
            'narrative': narrative_data,
            'analytics': analytics,
            'generation_metadata': {
                'generated_at': time.time(),
                'processing_time': time.time() - start_time,
                'total_posts': len(sorted_posts),
                'method': 'optimized_batched'
            }
        }
        
        atomic_write_json(cache_file, combined_result)
        logger.info(f"Generated narrative and analytics in {time.time() - start_time:.2f}s")
        return combined_result
    
    def _generate_optimized_narrative(self, posts: List[Dict], analytics: Dict) -> Dict:
        """Generate narrative using optimized batching and clustering."""
        phases = self._detect_optimized_phases(posts)
        reaction_posts = self._identify_reaction_posts(posts)
        
        # Group phases for batch processing
        phase_groups = self._group_phases_for_batching(phases)
        
        narrative_sections = []
        with tqdm(total=len(phase_groups), desc="Generating narratives", unit="batch") as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit batched narrative generation tasks
                future_to_group = {
                    executor.submit(self._generate_batch_narrative, group, posts, reaction_posts): group
                    for group in phase_groups
                }
                
                for future in concurrent.futures.as_completed(future_to_group):
                    batch_sections = future.result()
                    narrative_sections.extend(batch_sections)
                    pbar.update(1)
        
        # Sort sections by original phase order
        narrative_sections.sort(key=lambda x: x['phase_summary']['sequence'])
        
        thread_summary = self._generate_thread_overview(posts, phases, analytics)
        
        return {
            'thread_overview': thread_summary,
            'conversation_phases': phases,
            'narrative_sections': narrative_sections,
            'high_reaction_posts': reaction_posts,
            'topic_evolution': self._analyze_topic_evolution(phases),
            'key_contributors': self._identify_key_contributors(posts, phases)
        }
    
    def _detect_optimized_phases(self, posts: List[Dict]) -> List[Dict]:
        """Detect conversation phases using semantic clustering and page boundaries."""
        if len(posts) < 20:
            return [self._create_single_phase(posts)]
        
        phases = []
        current_phase_posts = []
        current_topic = None
        page_break_threshold = 25
        
        topic_indicators = {
            'design': ['design', 'prototype', 'concept', 'blueprint', 'model', 'version', 'build'],
            'shipping': ['ship', 'delivery', 'order', 'purchase', 'buy', 'price', 'cost', 'payment'],
            'reviews': ['review', 'test', 'experience', 'opinion', 'feedback', 'impression', 'rating'],
            'technical': ['specs', 'specification', 'technical', 'measurement', 'dimension', 'size'],
            'troubleshooting': ['problem', 'issue', 'fix', 'broken', 'error', 'help', 'repair'],
            'community': ['meet', 'group', 'community', 'together', 'social', 'gathering']
        }
        
        for i, post in enumerate(posts):
            content = post.get('content', '').lower()
            
            # Calculate topic scores
            topic_scores = {}
            for topic, keywords in topic_indicators.items():
                score = sum(2 if keyword in content else 0 for keyword in keywords)
                if score > 0:
                    topic_scores[topic] = score
            
            dominant_topic = max(topic_scores.keys(), key=lambda k: topic_scores[k]) if topic_scores else 'general'
            
            # Check for phase transition
            should_transition = (
                (current_topic and current_topic != dominant_topic and len(current_phase_posts) >= 10) or
                len(current_phase_posts) >= page_break_threshold or
                (i > 0 and posts[i-1].get('page', 1) != post.get('page', 1) and len(current_phase_posts) >= 8)
            )
            
            if should_transition:
                if current_phase_posts:
                    phases.append(self._create_optimized_phase_summary(current_topic or 'general', current_phase_posts, len(phases)))
                current_phase_posts = [post]
                current_topic = dominant_topic
            else:
                current_phase_posts.append(post)
                if current_topic is None:
                    current_topic = dominant_topic
        
        if current_phase_posts:
            phases.append(self._create_optimized_phase_summary(current_topic or 'general', current_phase_posts, len(phases)))
        
        return phases
    
    def _create_single_phase(self, posts: List[Dict]) -> Dict:
        """Create a single phase for small threads."""
        return self._create_optimized_phase_summary('general', posts, 0)
    
    def _create_optimized_phase_summary(self, topic: str, phase_posts: List[Dict], sequence: int) -> Dict:
        """Create optimized phase summary with essential data."""
        start_pos = phase_posts[0].get('global_position', 0)
        end_pos = phase_posts[-1].get('global_position', 0)
        start_page = phase_posts[0].get('page', 1)
        end_page = phase_posts[-1].get('page', 1)
        
        authors = Counter(post.get('author', 'Unknown') for post in phase_posts)
        total_reactions = sum(
            post.get('upvotes', 0) + post.get('likes', 0) + post.get('reactions', 0)
            for post in phase_posts
        )
        
        # Sample representative posts
        sample_size = min(3, len(phase_posts))
        if sample_size == len(phase_posts):
            sample_posts = phase_posts
        else:
            # Take first, middle, and last posts
            indices = [0, len(phase_posts) // 2, -1] if len(phase_posts) > 2 else [0, -1]
            sample_posts = [phase_posts[i] for i in indices[:sample_size]]
        
        return {
            'sequence': sequence,
            'topic': topic.title(),
            'post_range': f"posts {start_pos}-{end_pos}",
            'page_range': f"pages {start_page}-{end_page}" if start_page != end_page else f"page {start_page}",
            'post_count': len(phase_posts),
            'key_participants': dict(authors.most_common(3)),
            'total_engagement': total_reactions,
            'start_date': phase_posts[0].get('date', ''),
            'end_date': phase_posts[-1].get('date', ''),
            'sample_posts': sample_posts
        }
    
    def _group_phases_for_batching(self, phases: List[Dict]) -> List[List[Dict]]:
        """Group phases into batches for efficient LLM processing."""
        if len(phases) <= 4:
            return [phases]
        
        # Group similar topics together and batch by size
        topic_groups = defaultdict(list)
        for phase in phases:
            topic_groups[phase['topic']].append(phase)
        
        batches = []
        current_batch = []
        
        for topic, topic_phases in topic_groups.items():
            for phase in topic_phases:
                current_batch.append(phase)
                if len(current_batch) >= 4:  # Batch size of 4
                    batches.append(current_batch)
                    current_batch = []
        
        if current_batch:
            batches.append(current_batch)
        
        return batches
    
    def _generate_batch_narrative(self, phase_batch: List[Dict], all_posts: List[Dict], reaction_posts: List[Dict]) -> List[Dict]:
        """Generate narratives for a batch of phases in a single LLM call."""
        if len(phase_batch) == 1:
            return [self._generate_single_phase_narrative(phase_batch[0], all_posts, reaction_posts)]
        
        # Construct batch prompt
        batch_context = []
        for i, phase in enumerate(phase_batch, 1):
            sample_posts = phase.get('sample_posts', [])
            post_samples = []
            for post in sample_posts[:2]:  # Reduced sample size
                author = post.get('author', 'Unknown')
                content = post.get('content', '')[:200]  # Reduced content length
                post_samples.append(f"{author}: {content}...")
            
            phase_text = f"""
Phase {i}: {phase['topic']} ({phase['page_range']})
- {phase['post_count']} posts from: {', '.join(list(phase['key_participants'].keys())[:3])}
- Sample posts: {' | '.join(post_samples)}
"""
            batch_context.append(phase_text.strip())
        
        prompt = f"""
Summarize these {len(phase_batch)} conversation phases. For each phase, write 2-3 sentences focusing on key developments:

{chr(10).join(batch_context)}

Format: 
Phase 1: [summary]
Phase 2: [summary]
etc.
"""
        
        try:
            batch_response = self._get_llm_response(prompt)
            return self._parse_batch_response(batch_response, phase_batch, reaction_posts)
        except Exception as e:
            logger.warning(f"Batch narrative generation failed: {e}")
            return [self._generate_fallback_narrative(phase) for phase in phase_batch]
    
    def _generate_single_phase_narrative(self, phase: Dict, all_posts: List[Dict], reaction_posts: List[Dict]) -> Dict:
        """Generate narrative for a single phase."""
        sample_posts = phase.get('sample_posts', [])
        post_samples = []
        for post in sample_posts[:3]:
            author = post.get('author', 'Unknown')
            content = post.get('content', '')[:300]
            post_samples.append(f"{author}: {content}...")
        
        prompt = f"""
Summarize this conversation phase in 2-3 sentences focusing on key developments:

Topic: {phase['topic']} ({phase['page_range']})
Participants: {', '.join(list(phase['key_participants'].keys()))}
Sample posts: {chr(10).join(post_samples)}

Focus on what happened and key outcomes."""
        
        try:
            narrative_text = self._get_llm_response(prompt)
        except Exception as e:
            logger.warning(f"Single phase narrative failed: {e}")
            narrative_text = self._generate_fallback_narrative(phase)['narrative_text']
        
        return {
            'phase_summary': phase,
            'narrative_text': narrative_text,
            'highlights': self._find_phase_highlights(phase, reaction_posts)
        }
    
    def _parse_batch_response(self, response: str, phases: List[Dict], reaction_posts: List[Dict]) -> List[Dict]:
        """Parse batched LLM response into individual phase narratives."""
        sections = []
        lines = response.strip().split('\n')
        current_phase_idx = 0
        current_text = []
        
        for line in lines:
            line = line.strip()
            if line.startswith(f'Phase {current_phase_idx + 1}:'):
                if current_text and current_phase_idx < len(phases):
                    # Save previous phase
                    narrative_text = ' '.join(current_text).strip()
                    sections.append({
                        'phase_summary': phases[current_phase_idx],
                        'narrative_text': narrative_text,
                        'highlights': self._find_phase_highlights(phases[current_phase_idx], reaction_posts)
                    })
                
                # Start new phase
                current_text = [line.replace(f'Phase {current_phase_idx + 1}:', '').strip()]
                current_phase_idx += 1
            elif line and current_phase_idx <= len(phases):
                current_text.append(line)
        
        # Handle last phase
        if current_text and current_phase_idx <= len(phases):
            narrative_text = ' '.join(current_text).strip()
            sections.append({
                'phase_summary': phases[current_phase_idx - 1],
                'narrative_text': narrative_text,
                'highlights': self._find_phase_highlights(phases[current_phase_idx - 1], reaction_posts)
            })
        
        # Fill any missing phases with fallbacks
        while len(sections) < len(phases):
            sections.append(self._generate_fallback_narrative(phases[len(sections)]))
        
        return sections[:len(phases)]
    
    def _generate_fallback_narrative(self, phase: Dict) -> Dict:
        """Generate fallback narrative when LLM fails."""
        participant_count = len(phase['key_participants'])
        top_participant = list(phase['key_participants'].keys())[0] if phase['key_participants'] else 'participants'
        
        narrative_text = f"Discussion about {phase['topic'].lower()} with {phase['post_count']} posts across {phase['page_range']}. Key contributor was {top_participant} with {participant_count} total participants involved."
        
        return {
            'phase_summary': phase,
            'narrative_text': narrative_text,
            'highlights': []
        }
    
    def _find_phase_highlights(self, phase: Dict, reaction_posts: List[Dict]) -> List[Dict]:
        """Find highlighted posts within a phase."""
        phase_positions = set()
        for post in phase.get('sample_posts', []):
            phase_positions.add(post.get('global_position'))
        
        return [
            post for post in reaction_posts[:5]
            if post.get('global_position') in phase_positions
        ]
    
    def _identify_reaction_posts(self, posts: List[Dict]) -> List[Dict]:
        """Identify posts with high community engagement."""
        scored_posts = []
        
        for post in posts:
            upvotes = post.get('upvotes', 0)
            likes = post.get('likes', 0)
            reactions = post.get('reactions', 0)
            content_length = len(post.get('content', ''))
            
            engagement_score = (upvotes * 3) + (likes * 2) + reactions
            if content_length > 200:
                engagement_score += 2
            
            if engagement_score > 0:
                post_copy = post.copy()
                post_copy['engagement_score'] = engagement_score
                scored_posts.append(post_copy)
        
        return sorted(scored_posts, key=lambda x: x['engagement_score'], reverse=True)[:10]
    
    def _generate_thread_overview(self, posts: List[Dict], phases: List[Dict], analytics: Dict) -> str:
        """Generate high-level thread overview."""
        total_posts = len(posts)
        total_participants = len(set(post.get('author', '') for post in posts))
        
        first_date = posts[0].get('date', '') if posts else ''
        last_date = posts[-1].get('date', '') if posts else ''
        
        phase_topics = [phase['topic'] for phase in phases[:5]]  # Top 5 topics
        
        overview_prompt = f"""
Create a 3-4 sentence overview of this forum thread:

Stats: {total_posts} posts from {total_participants} participants ({first_date} to {last_date})
Main topics: {', '.join(phase_topics)}

Focus on what the thread accomplished and key outcomes."""
        
        try:
            return self._get_llm_response(overview_prompt)
        except Exception as e:
            logger.warning(f"Thread overview generation failed: {e}")
            return f"Forum discussion with {total_posts} posts covering topics: {', '.join(phase_topics)}"
    
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
        
        for phase in phases:
            for author in phase['key_participants']:
                if author in contributor_stats:
                    contributor_stats[author]['phases_active'].add(phase['topic'])
        
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
        """Get response from LLM with timeout and error handling."""
        try:
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False
            }
            
            response = requests.post(self.chat_url, json=payload, timeout=25)
            response.raise_for_status()
            
            result = response.json()
            return result.get('message', {}).get('content', '').strip()
            
        except Exception as e:
            logger.error(f"Error getting LLM response: {e}")
            raise


__all__ = ['ThreadNarrative']