"""
Optimized thread narrative generation for M1 MacBook Air with 8GB RAM.
Focus on performance, memory efficiency, and accuracy.
"""

import logging
import json
import time
import os
import hashlib
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Tuple
import gc

from config.settings import BASE_TMP_DIR
from analytics.thread_analyzer import ThreadAnalyzer
from utils.file_utils import atomic_write_json

logger = logging.getLogger(__name__)


def _get_llm_manager():
    """Get LLM manager instance."""
    from utils.llm_manager import llm_manager
    return llm_manager


class ThreadNarrative:
    """Creates comprehensive thread narratives optimized for performance."""
    
    def __init__(self):
        self._prompt_cache = {}
        self._cache_file = os.path.join(BASE_TMP_DIR, 'narrative_cache.json')
        self._load_cache()
    
    def _load_cache(self):
        """Load prompt cache from disk."""
        try:
            if os.path.exists(self._cache_file):
                with open(self._cache_file, 'r') as f:
                    self._prompt_cache = json.load(f)
        except Exception:
            self._prompt_cache = {}
    
    def _save_cache(self):
        """Save prompt cache to disk."""
        try:
            with open(self._cache_file, 'w') as f:
                json.dump(self._prompt_cache, f)
        except Exception:
            pass
    
    def _get_prompt_hash(self, prompt: str) -> str:
        """Generate hash for prompt caching."""
        return hashlib.md5(prompt.encode()).hexdigest()[:16]
        
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
                        logger.info(f"Using cached narrative for {len(posts)} posts")
                        return cached_data
            except Exception:
                pass
        
        sorted_posts = sorted(posts, key=lambda x: x.get('global_position', 0))
        
        # Memory-efficient loading strategy
        use_streaming = len(sorted_posts) > 1000
        logger.info(f"Processing {len(sorted_posts)} posts using {'streaming' if use_streaming else 'batch'} strategy")
        
        analyzer = ThreadAnalyzer(thread_dir)
        analytics = analyzer.analyze_thread(sorted_posts, force_refresh=True)
        
        narrative_data = self._generate_optimized_narrative(sorted_posts, analytics, use_streaming)
        
        combined_result = {
            'narrative': narrative_data,
            'analytics': analytics,
            'generation_metadata': {
                'generated_at': time.time(),
                'processing_time': time.time() - start_time,
                'total_posts': len(sorted_posts),
                'method': 'performance_optimized',
                'streaming_used': use_streaming
            }
        }
        
        atomic_write_json(cache_file, combined_result)
        self._save_cache()
        processing_time = time.time() - start_time
        logger.info(f"Generated narrative and analytics in {processing_time:.2f}s")
        return combined_result
    
    def _generate_optimized_narrative(self, posts: List[Dict], analytics: Dict, use_streaming: bool = False) -> Dict:
        """Generate narrative using performance-optimized single-pass approach."""
        phases = self._detect_intelligent_phases(posts)
        reaction_posts = self._identify_reaction_posts(posts)
        
        logger.info(f"Detected {len(phases)} phases for narrative generation")
        
        # Single-pass narrative generation
        narrative_sections = self._generate_all_narratives(phases, posts, reaction_posts)
        thread_summary = self._generate_thread_overview(posts, phases, analytics)
        
        return {
            'thread_overview': thread_summary,
            'conversation_phases': phases,
            'narrative_sections': narrative_sections,
            'high_reaction_posts': reaction_posts[:5],  # Top 5 only
            'topic_evolution': self._analyze_topic_evolution(phases),
            'key_contributors': self._identify_key_contributors(posts, phases)[:8]  # Top 8 only
        }
    
    def _detect_intelligent_phases(self, posts: List[Dict]) -> List[Dict]:
        """Detect conversation phases with aggressive grouping for performance."""
        if len(posts) < 50:
            return [self._create_single_phase(posts)]
        
        phases = []
        current_phase_posts = []
        current_topic = None
        phase_min_size = max(50, len(posts) // 15)  # Aim for max 15 phases total
        
        topic_indicators = {
            'technical': ['specs', 'specification', 'technical', 'measurement', 'performance', 'test'],
            'troubleshooting': ['problem', 'issue', 'fix', 'broken', 'error', 'help', 'solution'],
            'community': ['meet', 'group', 'community', 'social', 'discussion', 'opinion'],
            'general': ['question', 'info', 'information', 'about', 'general', 'new']
        }
        
        topic_votes = Counter()
        for i, post in enumerate(posts):
            content = post.get('content', '').lower()
            
            # Simple topic detection
            for topic, keywords in topic_indicators.items():
                if any(keyword in content for keyword in keywords):
                    topic_votes[topic] += 1
                    break
            else:
                topic_votes['general'] += 1
            
            current_phase_posts.append(post)
            
            # Phase transition: only on significant size or topic shift
            if len(current_phase_posts) >= phase_min_size:
                # Determine dominant topic for this phase
                phase_topics = Counter()
                for p in current_phase_posts[-phase_min_size:]:
                    p_content = p.get('content', '').lower()
                    for topic, keywords in topic_indicators.items():
                        if any(keyword in p_content for keyword in keywords):
                            phase_topics[topic] += 1
                            break
                    else:
                        phase_topics['general'] += 1
                
                dominant_topic = phase_topics.most_common(1)[0][0] if phase_topics else 'general'
                
                if current_topic is None:
                    current_topic = dominant_topic
                elif current_topic != dominant_topic and len(current_phase_posts) >= phase_min_size:
                    phases.append(self._create_phase_summary(current_topic, current_phase_posts, len(phases)))
                    current_phase_posts = []
                    current_topic = dominant_topic
        
        if current_phase_posts:
            phases.append(self._create_phase_summary(current_topic or 'general', current_phase_posts, len(phases)))
        
        return phases
    
    def _extract_topic_keywords(self, phase_posts: List[Dict], topic: str) -> List[str]:
        """Extract key keywords from phase posts for search enhancement."""
        all_content = ' '.join([post.get('content', '').lower() for post in phase_posts])
        
        # Common forum discussion words to filter out
        stop_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those',
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
            'post', 'thread', 'forum', 'user', 'member', 'page', 'reply', 'quote', 'edit'
        }
        
        # Extract words that appear frequently in this phase
        words = [word.strip('.,!?();:[]{}') for word in all_content.split() if len(word) > 3]
        word_counts = Counter([word for word in words if word not in stop_words])
        
        # Get top 8 most frequent words as keywords
        top_keywords = [word for word, count in word_counts.most_common(8) if count > 1]
        
        # Add topic-specific terms
        if topic.lower() != 'general':
            top_keywords.insert(0, topic.lower())
        
        return top_keywords[:8]
    
    def _create_single_phase(self, posts: List[Dict]) -> Dict:
        """Create a single phase for small threads."""
        return self._create_phase_summary('general', posts, 0)
    
    def _create_phase_summary(self, topic: str, phase_posts: List[Dict], sequence: int) -> Dict:
        """Create enhanced phase summary with rich topic narrative data."""
        start_pos = phase_posts[0].get('global_position', 0)
        end_pos = phase_posts[-1].get('global_position', 0)
        start_page = phase_posts[0].get('page', 1)
        end_page = phase_posts[-1].get('page', 1)
        
        # Get first post for URL generation
        first_post = phase_posts[0]
        first_post_url = first_post.get('url', '')
        
        authors = Counter(post.get('author', 'Unknown') for post in phase_posts)
        total_reactions = sum(
            post.get('upvotes', 0) + post.get('likes', 0) + post.get('reactions', 0)
            for post in phase_posts
        )
        
        # Top 3 posts by engagement
        scored_posts = []
        for post in phase_posts:
            engagement = post.get('upvotes', 0) + post.get('likes', 0) + post.get('reactions', 0)
            content_length = len(post.get('content', ''))
            score = engagement * 3 + (1 if content_length > 150 else 0)
            scored_posts.append((score, post))
        
        scored_posts.sort(key=lambda x: x[0], reverse=True)
        sample_posts = [post for _, post in scored_posts[:3]]
        
        # Generate topic keywords for narrative context
        topic_keywords = self._extract_topic_keywords(phase_posts, topic)
        
        return {
            'sequence': sequence,
            'topic': topic.title(),
            'title': f"{topic.title()} Discussion",  # Enhanced title
            'post_range': f"posts {start_pos}-{end_pos}",
            'page_range': f"pages {start_page}-{end_page}" if start_page != end_page else f"page {start_page}",
            'post_count': len(phase_posts),
            'key_participants': dict(authors.most_common(2)),
            'total_engagement': total_reactions,
            'sample_posts': sample_posts,
            'first_post_url': first_post_url,  # Clickable URL to topic start
            'topic_keywords': topic_keywords,  # Keywords for search enhancement
            'start_date': first_post.get('date', ''),
            'end_date': phase_posts[-1].get('date', ''),
            'narrative_ready': True  # Flag for narrative generation
        }
    
    def _generate_all_narratives(self, phases: List[Dict], all_posts: List[Dict], reaction_posts: List[Dict]) -> List[Dict]:
        """Generate all phase narratives in a single optimized LLM call."""
        if len(phases) <= 3:
            return self._generate_single_batch_narrative(phases, all_posts, reaction_posts)
        
        # For larger threads, process in groups of 5
        narrative_sections = []
        for i in range(0, len(phases), 5):
            batch = phases[i:i+5]
            batch_sections = self._generate_single_batch_narrative(batch, all_posts, reaction_posts)
            narrative_sections.extend(batch_sections)
        
        return narrative_sections
    
    def _generate_single_batch_narrative(self, phase_batch: List[Dict], all_posts: List[Dict], reaction_posts: List[Dict]) -> List[Dict]:
        """Generate narratives for a batch of phases using single LLM call."""
        # For single phase, handle directly
        if len(phase_batch) == 1:
            return [self._generate_single_phase_narrative_safe(phase_batch[0], all_posts, reaction_posts)]
        
        # Create compact context for all phases
        phase_contexts = []
        for i, phase in enumerate(phase_batch, 1):
            sample_posts = phase.get('sample_posts', [])[:2]  # Only top 2 posts
            
            post_samples = []
            for post in sample_posts:
                author = post.get('author', 'Unknown')
                content = post.get('content', '').strip()[:200]  # Truncate content
                engagement = post.get('upvotes', 0) + post.get('likes', 0)
                if engagement > 0:
                    post_samples.append(f"{author} (+{engagement}): {content}")
                else:
                    post_samples.append(f"{author}: {content}")
            
            phase_text = f"Phase {i} - {phase['topic']} ({phase['page_range']}, {phase['post_count']} posts):\n" + "\n".join(post_samples)
            phase_contexts.append(phase_text)
        
        # Check cache first
        context_text = "\n\n".join(phase_contexts)
        prompt_hash = self._get_prompt_hash(context_text)
        
        if prompt_hash in self._prompt_cache:
            logger.info(f"Using cached narrative for {len(phase_batch)} phases")
            cached_response = self._prompt_cache[prompt_hash]
            return self._parse_batch_response(cached_response, phase_batch, reaction_posts)
        
        system_prompt = "Create concise 2-3 sentence summaries focusing on key developments and outcomes."
        
        prompt = f"""Analyze these {len(phase_batch)} forum discussion phases. For each phase, write exactly 2-3 sentences highlighting main developments, decisions, or insights:

{context_text}

Format as:
Phase 1: [2-3 sentences about key developments]
Phase 2: [2-3 sentences about key developments]
etc.

Focus on concrete outcomes and what was accomplished."""
        
        try:
            batch_response, model_used = _get_llm_manager().get_narrative_response(prompt, system_prompt)
            logger.info(f"Generated narrative for {len(phase_batch)} phases using {model_used}")
            
            # Cache the response
            self._prompt_cache[prompt_hash] = batch_response
            
            return self._parse_batch_response(batch_response, phase_batch, reaction_posts)
            
        except Exception as e:
            logger.warning(f"Batch narrative generation failed: {e}")
            return [self._generate_fallback_narrative(phase) for phase in phase_batch]
    
    def _generate_single_phase_narrative_safe(self, phase: Dict, all_posts: List[Dict], reaction_posts: List[Dict]) -> Dict:
        """Generate enhanced narrative for a single phase with topic-aware content."""
        sample_posts = phase.get('sample_posts', [])
        post_samples = []
        
        for post in sample_posts[:3]:  # Increased to 3 posts for richer context
            author = post.get('author', 'Unknown')
            content = post.get('content', '').strip()[:300]  # Increased content length
            engagement = post.get('upvotes', 0) + post.get('likes', 0)
            if engagement > 0:
                post_samples.append(f"{author} (+{engagement}): {content}")
            else:
                post_samples.append(f"{author}: {content}")
        
        # Enhanced cache key with topic keywords
        topic_keywords = ', '.join(phase.get('topic_keywords', [])[:3])
        cache_key = f"enhanced_{phase['topic']}_{phase['post_count']}_{topic_keywords}"
        
        if cache_key in self._prompt_cache:
            logger.info("Using cached enhanced phase narrative")
            narrative_text = self._prompt_cache[cache_key]
        else:
            system_prompt = """Create a rich 3-4 sentence topic narrative that captures what this discussion section covers. 
Include what participants discussed, key points raised, and any conclusions or developments. 
Write in an engaging, informative tone suitable for forum users browsing topics."""
            
            # Enhanced prompt with more context
            participants_list = ', '.join(list(phase['key_participants'].keys()))
            keywords_context = f"Key discussion terms: {', '.join(phase.get('topic_keywords', [])[:5])}" if phase.get('topic_keywords') else ""
            
            prompt = f"""Generate a comprehensive topic narrative for this forum discussion section:

**Topic**: {phase['topic']} Discussion ({phase['page_range']}, {phase['post_count']} posts)
**Timeframe**: {phase.get('start_date', 'Unknown')} to {phase.get('end_date', 'Unknown')}
**Key Participants**: {participants_list}
**Engagement Level**: {phase.get('total_engagement', 0)} reactions
{keywords_context}

**Representative Posts**:
{chr(10).join(post_samples)}

Create a 3-4 sentence narrative that explains:
1. What this topic section discusses and its main focus
2. Key points, questions, or developments covered
3. Level of community engagement and participant contributions
4. Any notable outcomes, solutions, or conclusions reached

Write as if describing this topic to someone browsing the thread overview."""
            
            try:
                narrative_text, model_used = _get_llm_manager().get_narrative_response(prompt, system_prompt)
                logger.info(f"Generated enhanced phase narrative using {model_used}")
                # Cache the response
                self._prompt_cache[cache_key] = narrative_text
            except Exception as e:
                logger.warning(f"Enhanced phase narrative failed: {e}")
                narrative_text = self._generate_enhanced_fallback_narrative(phase)
        
        return {
            'phase_summary': phase,
            'narrative_text': narrative_text,
            'topic_title': phase.get('title', f"{phase['topic']} Discussion"),
            'first_post_url': phase.get('first_post_url', ''),
            'topic_keywords': phase.get('topic_keywords', []),
            'highlights': self._find_phase_highlights(phase, reaction_posts),
            'engagement_summary': self._create_engagement_summary(phase)
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
            post for post in reaction_posts[:3]  # Reduced to top 3
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
        
        return sorted(scored_posts, key=lambda x: x['engagement_score'], reverse=True)[:8]  # Top 8 only
    
    def _generate_thread_overview(self, posts: List[Dict], phases: List[Dict], analytics: Dict) -> str:
        """Generate high-level thread overview using optimized model."""
        total_posts = len(posts)
        total_participants = len(set(post.get('author', '') for post in posts))
        
        first_date = posts[0].get('date', '') if posts else ''
        last_date = posts[-1].get('date', '') if posts else ''
        
        phase_topics = [phase['topic'] for phase in phases[:5]]  # Top 5 topics only
        
        # Check cache first
        overview_key = f"overview_{total_posts}_{total_participants}_{len(phases)}"
        if overview_key in self._prompt_cache:
            logger.info("Using cached thread overview")
            return self._prompt_cache[overview_key]
        
        system_prompt = "Create a concise overview focusing on outcomes and key developments."
        
        overview_prompt = f"""Create a 3-sentence overview of this forum discussion:

{total_posts} posts from {total_participants} participants across {len(phases)} discussion phases.
Time span: {first_date} to {last_date}
Main topics: {', '.join(phase_topics)}

Focus on what was accomplished and key outcomes."""
        
        try:
            overview, model_used = _get_llm_manager().get_narrative_response(overview_prompt, system_prompt)
            logger.info(f"Generated thread overview using {model_used}")
            
            # Cache the overview
            self._prompt_cache[overview_key] = overview
            
            return overview
        except Exception as e:
            logger.warning(f"Thread overview generation failed: {e}")
            fallback = f"Forum discussion with {total_posts} posts from {total_participants} participants covering {', '.join(phase_topics)}. Discussion spanned from {first_date} to {last_date} across {len(phases)} main phases."
            self._prompt_cache[overview_key] = fallback
            return fallback
    
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
                if prev_phase['topic'] != phase['topic']:
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
        
        return sorted(contributors, key=lambda x: x['influence_score'], reverse=True)
    
    def _generate_enhanced_fallback_narrative(self, phase: Dict) -> str:
        """Generate high-quality enhanced fallback narrative when LLM fails."""
        participant_count = len(phase['key_participants'])
        top_participants = list(phase['key_participants'].keys())[:2]
        
        if len(top_participants) == 1:
            participant_text = f"{top_participants[0]} led the discussion"
        elif len(top_participants) == 2:
            participant_text = f"{top_participants[0]} and {top_participants[1]} were key contributors"
        else:
            participant_text = f"{participant_count} participants contributed"
        
        engagement_text = ""
        if phase.get('total_engagement', 0) > 0:
            engagement_text = f" with {phase['total_engagement']} community reactions"
        
        # Enhanced fallback with topic keywords
        keywords = phase.get('topic_keywords', [])
        keywords_text = f" focusing on {', '.join(keywords[:3])}" if keywords else ""
        
        topic_name = phase.get('topic', 'General').lower()
        
        fallback_narrative = (
            f"This {topic_name} discussion section spans {phase['post_count']} posts across {phase['page_range']}"
            f"{keywords_text}. {participant_text.capitalize()}{engagement_text}, "
            f"covering key aspects of the topic. The conversation took place from {phase.get('start_date', 'unknown date')} "
            f"to {phase.get('end_date', 'unknown date')}, representing a focused segment of the broader thread discussion."
        )
        
        return fallback_narrative
    
    def _create_engagement_summary(self, phase: Dict) -> Dict:
        """Create engagement summary for the phase."""
        total_engagement = phase.get('total_engagement', 0)
        participant_count = len(phase.get('key_participants', {}))
        post_count = phase.get('post_count', 0)
        
        # Calculate engagement metrics
        avg_engagement = total_engagement / post_count if post_count > 0 else 0
        engagement_level = 'high' if avg_engagement > 2 else 'medium' if avg_engagement > 0.5 else 'low'
        
        return {
            'total_reactions': total_engagement,
            'participant_count': participant_count,
            'average_engagement': round(avg_engagement, 2),
            'engagement_level': engagement_level,
            'posts_per_participant': round(post_count / participant_count, 1) if participant_count > 0 else 0
        }


__all__ = ['ThreadNarrative']