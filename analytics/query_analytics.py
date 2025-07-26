"""
Query analytics and enhanced conversational understanding for Forum Wisdom Miner.

This module provides advanced query analysis capabilities for better handling
of vague and conversational queries, with context-aware understanding.
"""

import logging
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from config.settings import ANALYTICAL_QUERY_PATTERNS

logger = logging.getLogger(__name__)


class ConversationalQueryProcessor:
    """Enhanced query processor for vague and conversational queries."""
    
    def __init__(self):
        """Initialize the conversational query processor."""
        self.analytical_patterns = ANALYTICAL_QUERY_PATTERNS
        self.context_memory = {}  # For maintaining conversation context
        
        # Enhanced patterns for analytical queries
        self.enhanced_patterns = {
            **self.analytical_patterns,
            'insights': ['insights', 'interesting', 'notable', 'remarkable', 'key findings'],
            'problems': ['problem', 'issue', 'challenge', 'difficulty', 'concern'],
            'solutions': ['solution', 'fix', 'answer', 'resolve', 'workaround'],
            'recommendations': ['recommend', 'suggest', 'advice', 'should', 'best'],
            'experience': ['experience', 'story', 'happened', 'encounter', 'went through'],
            'outcome': ['outcome', 'result', 'what happened', 'end result', 'final'],
            'learning': ['learn', 'lesson', 'takeaway', 'discovery', 'realize'],
            'engagement': ['highest rated', 'most rated', 'top rated', 'best rated', 'most popular', 
                          'most upvoted', 'highest scoring', 'best post', 'top post', 'popular post',
                          'most liked', 'most reactions', 'most engaged', 'top engagement', 
                          'well received', 'community favorite', 'highly rated']
        }
        
        # Smart query expansion patterns for vague queries
        self.expansion_patterns = {
            'rated': 'highest rated most popular best post community engagement upvotes reactions',
            'popular': 'most popular highest rated best post community favorite well received',
            'best': 'best post highest rated most popular top engagement community favorite',
            'top': 'top post highest rated most popular best engagement community favorite',
            'good': 'good post well received highly rated popular community favorite',
            'post': 'specific post content author engagement rating community response',
            
            # Product recommendation expansions (generic for any product type)
            'recommend': 'recommend suggest advice best choice favorite buy purchase love prefer',
            'which': 'which what recommend suggest best choice favorite popular commonly used most often',
            'buy': 'buy purchase recommend suggest best choice favorite where to buy link store',
            'popular': 'popular recommend best choice favorite commonly used most often highly rated',
            'favorite': 'favorite recommend best choice popular love prefer top choice',
            'often': 'often recommend commonly used popular frequently mentioned best choice',
            
            # Technical specification expansions
            'materials': 'materials material made construction build composition parts components',
            'material': 'material materials made construction build composition quality type',
            'settings': 'settings configuration setup parameters values numbers specs',
            'setting': 'setting settings configuration setup parameter value spec',
            'specifications': 'specifications specs details parameters configuration technical',
            'specs': 'specs specifications details parameters technical configuration',
            'what are': 'what are details specifications information about parameters'
        }
        
        # Vague query indicators
        self.vague_indicators = [
            'what about', 'tell me about', 'anything about', 'general', 'overall',
            'what do you think', 'what happened', 'how did', 'what are',
            'explain', 'describe', 'give me info', 'what can you tell me'
        ]
        
        # Question type patterns
        self.question_types = {
            'what': r'\bwhat\b',
            'how': r'\bhow\b',
            'why': r'\bwhy\b',
            'when': r'\bwhen\b',
            'where': r'\bwhere\b',
            'who': r'\bwho\b',
            'which': r'\bwhich\b'
        }
    
    def analyze_conversational_query(self, query: str, thread_analytics: Optional[Dict] = None) -> Dict[str, Any]:
        """Analyze a conversational query and determine the best response strategy.
        
        Args:
            query: User's query string
            thread_analytics: Optional thread analytics for context
            
        Returns:
            Analysis results with suggested approach
        """
        query_lower = query.lower().strip()
        
        analysis = {
            'original_query': query,
            'is_vague': self._is_vague_query(query_lower),
            'analytical_intent': self._detect_analytical_intent(query_lower),
            'question_type': self._detect_question_type(query_lower),
            'focus_areas': self._extract_focus_areas(query_lower),
            'suggested_approach': 'semantic_search',  # Default
            'context_hints': [],
            'expanded_query': query,
            'confidence': 0.5
        }
        
        # Enhance analysis based on detected patterns
        if analysis['is_vague']:
            analysis = self._handle_vague_query(analysis, thread_analytics)
        
        if analysis['analytical_intent']:
            analysis = self._handle_analytical_query(analysis, thread_analytics)
        
        # Generate expanded query for better search
        analysis['expanded_query'] = self._expand_query(query, analysis, thread_analytics)
        
        return analysis
    
    def _is_vague_query(self, query: str) -> bool:
        """Determine if a query is vague and needs expansion."""
        # Check for vague indicators
        for indicator in self.vague_indicators:
            if indicator in query:
                return True
        
        # Check for very short queries
        if len(query.split()) <= 3:
            return True
        
        # Check for question words without specific topics
        question_words = ['what', 'how', 'why', 'tell me']
        has_question_word = any(word in query for word in question_words)
        has_specific_terms = len([word for word in query.split() if len(word) > 4]) >= 2
        
        return has_question_word and not has_specific_terms
    
    def _detect_analytical_intent(self, query: str) -> List[str]:
        """Detect analytical intent types in the query."""
        detected_intents = []
        
        for intent_type, patterns in self.enhanced_patterns.items():
            for pattern in patterns:
                if pattern in query:
                    detected_intents.append(intent_type)
                    break
        
        return detected_intents
    
    def _detect_question_type(self, query: str) -> Optional[str]:
        """Detect the primary question type (what, how, why, etc.)."""
        for q_type, pattern in self.question_types.items():
            if re.search(pattern, query, re.IGNORECASE):
                return q_type
        return None
    
    def _extract_focus_areas(self, query: str) -> List[str]:
        """Extract potential focus areas from the query."""
        # Common focus areas in forum discussions
        focus_patterns = {
            'technical': ['technical', 'specs', 'specification', 'performance', 'features'],
            'usability': ['usability', 'user', 'interface', 'experience', 'easy', 'difficult'],
            'comparison': ['compare', 'vs', 'versus', 'better', 'worse', 'difference'],
            'cost': ['cost', 'price', 'expensive', 'cheap', 'budget', 'value'],
            'reliability': ['reliable', 'quality', 'durable', 'last', 'break'],
            'community': ['community', 'people', 'users', 'opinions', 'thoughts']
        }
        
        detected_areas = []
        for area, patterns in focus_patterns.items():
            if any(pattern in query for pattern in patterns):
                detected_areas.append(area)
        
        return detected_areas
    
    def _handle_vague_query(self, analysis: Dict, thread_analytics: Optional[Dict]) -> Dict:
        """Handle vague queries by suggesting broader search approaches."""
        analysis['suggested_approach'] = 'broad_search'
        analysis['confidence'] = 0.3
        
        # Add context hints based on thread analytics
        if thread_analytics:
            analytics_summary = thread_analytics.get('summary', {})
            
            # Suggest focusing on popular topics
            keywords = analytics_summary.get('content_insights', {}).get('primary_keywords', [])
            if keywords:
                analysis['context_hints'].append(f"Popular topics in this thread: {', '.join(keywords[:3])}")
            
            # Suggest focusing on active participants
            most_active = analytics_summary.get('activity', {}).get('most_active_author', {})
            if most_active.get('name'):
                analysis['context_hints'].append(f"Most active participant: {most_active['name']}")
            
            # Suggest temporal focus if thread spans time
            duration = analytics_summary.get('activity', {}).get('thread_duration_days', 0)
            if duration > 7:
                analysis['context_hints'].append(f"Thread spans {duration} days - consider temporal aspects")
        
        return analysis
    
    def _handle_analytical_query(self, analysis: Dict, thread_analytics: Optional[Dict]) -> Dict:
        """Handle analytical queries with specific intent."""
        intents = analysis['analytical_intent']
        analysis['suggested_approach'] = 'analytical_search'
        analysis['confidence'] = 0.8
        
        # Provide specific guidance based on analytical intent
        guidance = []
        
        if 'summary' in intents:
            guidance.append("Focus on comprehensive overview and key points")
            analysis['suggested_approach'] = 'summary_generation'
        
        if 'sentiment' in intents:
            guidance.append("Analyze opinions and emotional tone of posts")
        
        if 'statistics' in intents:
            guidance.append("Provide quantitative data and counts")
            if thread_analytics:
                stats = thread_analytics.get('statistics', {})
                guidance.append(f"Thread has {stats.get('total_posts', 0)} posts from {stats.get('participants', {}).get('total_participants', 0)} participants")
        
        if 'timeline' in intents:
            guidance.append("Focus on chronological progression and development")
        
        if 'comparison' in intents:
            guidance.append("Identify and contrast different viewpoints or options")
        
        if 'participants' in intents:
            guidance.append("Focus on who is involved and their contributions")
        
        analysis['context_hints'] = guidance
        return analysis
    
    def _expand_query(self, original_query: str, analysis: Dict, thread_analytics: Optional[Dict]) -> str:
        """Expand the original query for better search results with smart enhancement."""
        query_lower = original_query.lower()
        expanded_parts = [original_query]
        
        # Smart expansion for recommendation and engagement queries
        expansion_applied = []
        for key, expansion in self.expansion_patterns.items():
            if key in query_lower:
                logger.info(f"Auto-expanding query '{original_query}' with '{key}' terms")
                # Use the expansion as-is (no special cases for specific products)
                expanded_parts.append(expansion)
                expansion_applied.append(key)
                # Limit to 2 expansions to avoid over-specificity
                if len(expansion_applied) >= 2:
                    break
        
        # Detect and enhance vague queries automatically
        if analysis['is_vague'] and not expansion_applied:
            # If it's a very short query, try to make it more specific
            if len(query_lower.split()) <= 2:
                if any(term in query_lower for term in ['rated', 'rating', 'score']):
                    expanded_parts.append('highest rated most popular best post community engagement')
                elif any(term in query_lower for term in ['best', 'good', 'great']):
                    expanded_parts.append('best post highest rated most popular community favorite')
                elif any(term in query_lower for term in ['popular', 'liked']):
                    expanded_parts.append('most popular highest rated best post well received')
                elif any(term in query_lower for term in ['post', 'comment']):
                    expanded_parts.append('specific post content author engagement rating community response')
        
        # Add analytical intent terms
        if analysis['analytical_intent']:
            for intent in analysis['analytical_intent']:
                if intent in self.enhanced_patterns:
                    expanded_parts.extend(self.enhanced_patterns[intent][:2])  # Add top 2 related terms
        
        # Add focus area terms
        if analysis['focus_areas']:
            expanded_parts.extend(analysis['focus_areas'])
        
        # Add thread-specific context if available
        if thread_analytics:
            keywords = thread_analytics.get('summary', {}).get('content_insights', {}).get('primary_keywords', [])
            if keywords and analysis['is_vague']:
                expanded_parts.extend(keywords[:3])  # Add top 3 thread keywords for vague queries
        
        # Remove duplicates while preserving order
        seen = set()
        unique_parts = []
        for part in expanded_parts:
            if part.lower() not in seen:
                unique_parts.append(part)
                seen.add(part.lower())
        
        expanded_query = ' '.join(unique_parts)
        
        # Log the expansion for debugging
        if expanded_query != original_query:
            logger.info(f"Query expanded from '{original_query}' to '{expanded_query}'")
        
        return expanded_query
    
    def generate_analytical_prompt(self, query: str, analysis: Dict, context: str) -> str:
        """Generate an enhanced prompt for analytical queries.
        
        Args:
            query: Original user query
            analysis: Query analysis results
            context: Retrieved post context
            
        Returns:
            Enhanced prompt for the LLM
        """
        base_prompt = "You are an expert forum analyst. "
        
        # Detect technical specification queries
        query_lower = query.lower()
        is_technical_query = any(term in query_lower for term in [
            'materials', 'material', 'settings', 'setting', 'specifications', 'specs', 
            'what are', 'parameters', 'configuration', 'components', 'parts'
        ])
        
        # Customize prompt based on query type and intent
        if is_technical_query:
            base_prompt += ("The user is asking for specific technical information. Focus on extracting concrete details, "
                          "specifications, materials, settings, and configuration information from the posts. "
                          "Look for specific values, measurements, part names, materials mentioned, and user experiences with different settings. "
                          "If no specific technical information is found, clearly state that no posts discuss these details. ")
        elif analysis.get('is_vague'):
            base_prompt += ("The user has asked a broad question. Provide a comprehensive overview "
                          "covering the main themes, key participants, and important insights from the discussion. ")
        
        if analysis.get('analytical_intent'):
            intents = analysis['analytical_intent']
            
            if 'summary' in intents:
                base_prompt += "Provide a structured summary with key points, main conclusions, and takeaways. "
            
            if 'sentiment' in intents:
                base_prompt += "Analyze the overall sentiment and opinions expressed in the discussion. "
            
            if 'statistics' in intents:
                base_prompt += "Include relevant numbers, counts, and quantitative information. "
            
            if 'timeline' in intents:
                base_prompt += "Present information in chronological order showing how the discussion evolved. "
            
            if 'comparison' in intents:
                base_prompt += "Compare and contrast different viewpoints or options discussed. "
            
            if 'participants' in intents:
                base_prompt += "Focus on who said what and the different perspectives of participants. "
        
        # Add context hints
        if analysis.get('context_hints'):
            base_prompt += f"Keep in mind: {'; '.join(analysis['context_hints'])}. "
        
        base_prompt += ("Use ONLY the information from the provided posts. Be specific and cite relevant details. "
                       "If the information isn't available in the posts, clearly state that.\n\n")
        
        base_prompt += f"---FORUM CONTEXT---\n{context}\n---END CONTEXT---\n\n"
        base_prompt += f"User Question: {query}\n\nProvide a detailed answer:"
        
        return base_prompt


class AnalyticalSearchStrategy:
    """Strategy for handling different types of analytical searches."""
    
    @staticmethod
    def get_search_strategy(analysis: Dict) -> Dict[str, Any]:
        """Determine the best search strategy based on query analysis.
        
        Args:
            analysis: Query analysis results
            
        Returns:
            Search strategy configuration
        """
        strategy = {
            'approach': 'semantic',
            'top_k': 7,
            'use_hyde': True,
            'rerank': True,
            'expand_context': False
        }
        
        # Adjust strategy based on query characteristics
        if analysis.get('is_vague'):
            # For vague queries, cast a wider net
            strategy.update({
                'top_k': 10,
                'expand_context': True,
                'approach': 'broad_semantic'
            })
        
        if 'summary' in analysis.get('analytical_intent', []):
            # For summary requests, get more comprehensive context
            strategy.update({
                'top_k': 15,
                'expand_context': True,
                'approach': 'comprehensive'
            })
        
        if 'recommendations' in analysis.get('analytical_intent', []):
            # For recommendation queries, cast wider net to find all product mentions
            strategy.update({
                'top_k': 25,
                'expand_context': True,
                'approach': 'product_recommendation'
            })
        
        if 'statistics' in analysis.get('analytical_intent', []):
            # For statistical queries, focus on finding all relevant data
            strategy.update({
                'top_k': 20,
                'rerank': False,  # Don't rerank, we want comprehensive coverage
                'approach': 'exhaustive'
            })
        
        if analysis.get('question_type') == 'who':
            # For "who" questions, focus on participant information
            strategy.update({
                'approach': 'participant_focused',
                'top_k': 8
            })
        
        return strategy


__all__ = ['ConversationalQueryProcessor', 'AnalyticalSearchStrategy']