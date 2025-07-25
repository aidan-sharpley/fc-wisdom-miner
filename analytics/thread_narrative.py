"""
Advanced chronological thread summarization with multi-model optimization.
Optimized for M1 MacBook Air with 8GB RAM using progressive fallback strategy.
"""

import logging
import json
import time
import os
import concurrent.futures
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

from config.settings import (
    NARRATIVE_BATCH_SIZE, NARRATIVE_MAX_WORKERS
)
from analytics.thread_analyzer import ThreadAnalyzer
from utils.file_utils import atomic_write_json
from utils.llm_manager import llm_manager, TaskType

logger = logging.getLogger(__name__)


class ThreadNarrative:
    """Creates comprehensive thread narratives with multi-model optimization."""
    
    def __init__(self):
        self.max_workers = NARRATIVE_MAX_WORKERS
        self.batch_size = NARRATIVE_BATCH_SIZE
        
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
                'method': 'multi_model_optimized',
                'llm_stats': llm_manager.get_stats()
            }
        }
        
        atomic_write_json(cache_file, combined_result)
        logger.info(f"Generated narrative and analytics in {time.time() - start_time:.2f}s")
        return combined_result
    
    def _generate_optimized_narrative(self, posts: List[Dict], analytics: Dict) -> Dict:
        """Generate narrative using multi-model optimization and batching."""
        phases = self._detect_optimized_phases(posts)
        reaction_posts = self._identify_reaction_posts(posts)
        
        logger.info(f"Detected {len(phases)} phases, creating {len(self._group_phases_for_batching(phases))} batches")
        
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
                    try:
                        batch_sections = future.result()
                        narrative_sections.extend(batch_sections)
                        pbar.set_postfix(sections=len(narrative_sections))
                    except Exception as e:
                        logger.error(f"Batch narrative generation failed: {e}")
                        # Generate fallback sections for failed batch
                        group = future_to_group[future]
                        fallback_sections = [self._generate_fallback_narrative(phase) for phase in group]
                        narrative_sections.extend(fallback_sections)
                    finally:
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
        page_break_threshold = 30  # Increased for fewer, larger phases
        
        topic_indicators = {
            'design': ['design', 'prototype', 'concept', 'blueprint', 'model', 'version', 'build'],
            'shipping': ['ship', 'delivery', 'order', 'purchase', 'buy', 'price', 'cost', 'payment'],
            'reviews': ['review', 'test', 'experience', 'opinion', 'feedback', 'impression', 'rating'],
            'technical': ['specs', 'specification', 'technical', 'measurement', 'dimension', 'size'],
            'troubleshooting': ['problem', 'issue', 'fix', 'broken', 'error', 'help', 'repair', 'solve'],
            'community': ['meet', 'group', 'community', 'together', 'social', 'gathering', 'event']
        }
        
        for i, post in enumerate(posts):
            content = post.get('content', '').lower()
            
            # Calculate topic scores with better weighting
            topic_scores = {}
            for topic, keywords in topic_indicators.items():
                score = sum(3 if keyword in content else 0 for keyword in keywords)
                if score > 0:
                    topic_scores[topic] = score
            
            dominant_topic = max(topic_scores.keys(), key=lambda k: topic_scores[k]) if topic_scores else 'general'
            
            # More intelligent phase transition logic
            should_transition = (
                (current_topic and current_topic != dominant_topic and len(current_phase_posts) >= 15) or
                len(current_phase_posts) >= page_break_threshold or
                (i > 0 and posts[i-1].get('page', 1) != post.get('page', 1) and len(current_phase_posts) >= 12)
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
        
        # Smarter representative post sampling
        sample_size = min(4, len(phase_posts))  # Increased sample size
        if sample_size == len(phase_posts):
            sample_posts = phase_posts
        else:
            # Take posts with highest engagement + beginning/end
            scored_posts = []
            for post in phase_posts:
                engagement = post.get('upvotes', 0) + post.get('likes', 0) + post.get('reactions', 0)
                content_length = len(post.get('content', ''))
                score = engagement * 3 + (1 if content_length > 200 else 0)
                scored_posts.append((score, post))
            
            # Sort by engagement and take top posts
            scored_posts.sort(key=lambda x: x[0], reverse=True)
            sample_posts = [post for _, post in scored_posts[:sample_size]]
        
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
        """Group phases into optimal batches for LLM processing."""
        if len(phases) <= self.batch_size:
            return [phases]
        
        # Group similar topics together for better context
        topic_groups = defaultdict(list)
        for phase in phases:
            topic_groups[phase['topic']].append(phase)
        
        batches = []
        current_batch = []
        
        # Process topic groups, trying to keep related topics together
        for topic, topic_phases in topic_groups.items():
            for phase in topic_phases:
                current_batch.append(phase)
                if len(current_batch) >= self.batch_size:
                    batches.append(current_batch)
                    current_batch = []
        
        if current_batch:
            batches.append(current_batch)
        
        return batches
    
    def _generate_batch_narrative(self, phase_batch: List[Dict], all_posts: List[Dict], reaction_posts: List[Dict]) -> List[Dict]:
        """Generate narratives for a batch of phases using optimized LLM."""
        if len(phase_batch) == 1:
            return [self._generate_single_phase_narrative(phase_batch[0], all_posts, reaction_posts)]
        
        # Construct optimized batch prompt
        batch_context = []
        for i, phase in enumerate(phase_batch, 1):
            sample_posts = phase.get('sample_posts', [])
            post_samples = []
            
            # Use best sample posts with more context
            for post in sample_posts[:2]:
                author = post.get('author', 'Unknown')
                content = post.get('content', '')[:250]  # Slightly more content
                engagement = post.get('upvotes', 0) + post.get('likes', 0)
                engagement_note = f" (+{engagement} votes)" if engagement > 0 else ""
                post_samples.append(f"{author}{engagement_note}: {content}...")
            
            phase_text = f"""Phase {i}: {phase['topic']} ({phase['page_range']})
- {phase['post_count']} posts from: {', '.join(list(phase['key_participants'].keys())[:3])}
- Key discussions: {' | '.join(post_samples)}"""
            
            batch_context.append(phase_text.strip())
        
        system_prompt = "You are a forum discussion analyst. Create concise, informative summaries focusing on key developments and outcomes."
        
        prompt = f"""Analyze these {len(phase_batch)} conversation phases from a forum thread. For each phase, write 2-3 sentences highlighting the main developments, decisions, or insights:

{chr(10).join(batch_context)}

Format your response as:
Phase 1: [2-3 sentence summary focusing on key developments]
Phase 2: [2-3 sentence summary focusing on key developments]
etc.

Focus on concrete outcomes, decisions made, problems solved, or insights gained."""
        
        try:
            batch_response, model_used = llm_manager.get_narrative_response(prompt, system_prompt)
            logger.debug(f"Batch narrative generated using {model_used}")
            return self._parse_batch_response(batch_response, phase_batch, reaction_posts)
        except Exception as e:
            logger.warning(f"Batch narrative generation failed: {e}")
            return [self._generate_fallback_narrative(phase) for phase in phase_batch]
    
    def _generate_single_phase_narrative(self, phase: Dict, all_posts: List[Dict], reaction_posts: List[Dict]) -> Dict:
        """Generate narrative for a single phase using optimized model."""
        sample_posts = phase.get('sample_posts', [])
        post_samples = []
        
        for post in sample_posts[:3]:
            author = post.get('author', 'Unknown')
            content = post.get('content', '')[:300]
            post_samples.append(f"{author}: {content}...")
        
        system_prompt = "You are analyzing a forum discussion. Focus on concrete developments and outcomes."
        
        prompt = f"""Summarize this conversation phase in 2-3 sentences focusing on key developments:

Topic: {phase['topic']} ({phase['page_range']})
Participants: {', '.join(list(phase['key_participants'].keys()))}
Sample discussions:
{chr(10).join(post_samples)}

Focus on what was accomplished, decided, or learned in this phase."""
        
        try:
            narrative_text, model_used = llm_manager.get_narrative_response(prompt, system_prompt)
            logger.debug(f"Single phase narrative generated using {model_used}")
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
        """Generate high-quality fallback narrative when LLM fails."""
        participant_count = len(phase['key_participants'])
        top_participants = list(phase['key_participants'].keys())[:2]
        
        if len(top_participants) == 1:
            participant_text = f"{top_participants[0]} led the discussion"
        elif len(top_participants) == 2:
            participant_text = f"{top_participants[0]} and {top_participants[1]} were key contributors"
        else:
            participant_text = f"{participant_count} participants contributed"
        
        engagement_text = ""
        if phase['total_engagement'] > 0:
            engagement_text = f" with {phase['total_engagement']} community reactions"
        
        narrative_text = f"Discussion about {phase['topic'].lower()} spanning {phase['post_count']} posts across {phase['page_range']}. {participant_text.capitalize()}{engagement_text}, covering key aspects of the topic."
        
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
            
            # Enhanced engagement scoring
            engagement_score = (upvotes * 3) + (likes * 2) + reactions
            if content_length > 200:
                engagement_score += 2
            if content_length > 500:
                engagement_score += 1
            
            if engagement_score > 0:
                post_copy = post.copy()
                post_copy['engagement_score'] = engagement_score
                scored_posts.append(post_copy)
        
        return sorted(scored_posts, key=lambda x: x['engagement_score'], reverse=True)[:10]
    
    def _generate_thread_overview(self, posts: List[Dict], phases: List[Dict], analytics: Dict) -> str:
        """Generate high-level thread overview using optimized model."""
        total_posts = len(posts)
        total_participants = len(set(post.get('author', '') for post in posts))
        
        first_date = posts[0].get('date', '') if posts else ''
        last_date = posts[-1].get('date', '') if posts else ''
        
        phase_topics = [phase['topic'] for phase in phases[:6]]  # Top 6 topics
        
        system_prompt = "Create a concise overview of this forum thread focusing on outcomes and key developments."
        
        overview_prompt = f"""Create a 3-4 sentence overview of this forum discussion thread:

Thread Statistics:
- {total_posts} posts from {total_participants} participants
- Time span: {first_date} to {last_date}
- Main discussion topics: {', '.join(phase_topics)}

Focus on what the thread accomplished, key decisions made, and overall outcomes for the community."""
        
        try:
            overview, model_used = llm_manager.get_narrative_response(overview_prompt, system_prompt)
            logger.debug(f"Thread overview generated using {model_used}")
            return overview
        except Exception as e:
            logger.warning(f"Thread overview generation failed: {e}")
            return f"Forum discussion with {total_posts} posts from {total_participants} participants covering topics: {', '.join(phase_topics)}. The discussion spanned from {first_date} to {last_date}."
    
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


__all__ = ['ThreadNarrative']