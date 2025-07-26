"""
LLM-based intelligent query routing for Forum Wisdom Miner.

This module uses an LLM to intelligently determine the best approach
for answering user queries based on available system capabilities.
"""

import logging
import json
from typing import Dict, List, Optional
import requests

from config.settings import OLLAMA_BASE_URL, OLLAMA_CHAT_MODEL

logger = logging.getLogger(__name__)


class LLMQueryRouter:
    """Uses LLM to intelligently route queries to the best available method."""
    
    def __init__(self):
        """Initialize the LLM query router."""
        self.chat_url = f"{OLLAMA_BASE_URL}/api/chat"
        self.model = OLLAMA_CHAT_MODEL
        
        # Define our system capabilities
        self.system_capabilities = self._build_capabilities_description()
        
        # Statistics
        self.stats = {
            'total_routings': 0,
            'analytical_routings': 0,
            'semantic_routings': 0,
            'routing_errors': 0
        }
    
    def _build_capabilities_description(self) -> str:
        """Build a description of our system's analytical and search capabilities."""
        return """
# Forum Analysis System Capabilities

## ANALYTICAL DATA PROCESSING (Fast, Factual)
Can directly analyze thread data to answer:

**Participant Analysis:**
- Who is most active? Who posted most?
- User activity patterns and participation levels
- Author rankings and contribution statistics

**Positional/Chronological Analysis:**
- Who was the first/second/third user to post?
- Timeline analysis and posting order
- Chronological patterns and thread evolution

**Thread Authorship Analysis:**
- Who is the thread author/creator?
- Who started this thread?
- Original poster (OP) identification using metadata priority

**Technical Specifications Analysis:**
- What wattage/temperature/voltage settings do people use?
- Materials, configurations, and component specifications
- Technical specifications and user preferences  
- Equipment settings and recommendations
- Specific parameter values mentioned in discussions

**Engagement Analysis:**
- Highest/lowest rated posts (by upvotes, likes, reactions)
- Most popular or well-received content
- Community favorites and trending posts

**Statistical Analysis:**
- How many posts/participants/pages?
- Thread metrics and quantitative data
- Content volume and distribution stats

## SEMANTIC SEARCH (Comprehensive, Contextual)
Best for understanding content and concepts:

**Content Understanding:**
- Explaining techniques, methods, processes
- Summarizing discussions and opinions
- Finding conceptual information and advice

**Opinion Mining:**
- What do people think about X?
- Pros/cons discussions and comparisons
- Subjective experiences and recommendations

**Instructional Content:**
- How-to guides and tutorials
- Step-by-step processes and methods
- Troubleshooting and problem-solving

**General Knowledge:**
- Broad topic exploration
- Conceptual questions about the subject matter
- Educational content and explanations

## ROUTING DECISION FACTORS

**Choose ANALYTICAL when:**
- Query asks "who", "how many", "which user", "most active"
- Looking for user activity, participation statistics, or rankings
- Needs quantitative analysis of thread metadata (post counts, participant counts)
- Seeking thread statistics, user rankings, or activity patterns

**Choose SEMANTIC when:**
- Query asks about specific content, technical details, or product information
- Looking for explanations, recommendations, or detailed information from posts
- Needs to find and synthesize information from actual post content
- Seeking technical specifications, product details, or user experiences
- Query asks "what are", "what materials", "what settings", "how to", "explain"
"""
    
    def route_query(self, query: str, thread_metadata: Optional[Dict] = None) -> Dict:
        """Route a query to the best processing method using LLM intelligence.
        
        Args:
            query: User's query
            thread_metadata: Optional metadata about the thread
            
        Returns:
            Routing decision with reasoning
        """
        self.stats['total_routings'] += 1
        
        try:
            # Build context about the thread if available
            thread_context = ""
            if thread_metadata:
                posts_count = thread_metadata.get('total_posts', 'unknown')
                participants = thread_metadata.get('participants', 'unknown') 
                thread_context = f"\nThread Context: {posts_count} posts, {participants} participants"
            
            # Build the routing prompt
            routing_prompt = f"""
{self.system_capabilities}

{thread_context}

USER QUERY: "{query}"

CRITICAL: Respond with ONLY valid JSON. Do not include any thinking, explanations, or additional text.

Return this exact JSON format:
{{
    "method": "analytical",
    "confidence": 0.8,
    "reasoning": "Query asks for technical specifications",
    "query_type": "technical_specs",
    "search_depth": 10
}}

Rules:
- method: "analytical" ONLY for who/how many/most active user queries, "semantic" for what are/what materials/what settings/technical content queries
- IMPORTANT: Queries about specific content, materials, settings, or technical details should ALWAYS use "semantic"
- IMPORTANT: Only use "analytical" for user activity statistics and thread metadata
- confidence: 0.1 to 1.0
- reasoning: One sentence explanation
- query_type: content_search, participant_analysis, engagement, etc
- search_depth: 10-50 (only used for semantic queries)

JSON ONLY - NO OTHER TEXT:
"""
            
            # Get LLM routing decision
            response = self._get_llm_response(routing_prompt)
            
            # Parse the JSON response
            try:
                # Clean the response to remove any thinking tags or extra text
                clean_response = self._clean_llm_response(response)
                routing_decision = json.loads(clean_response)
                
                # Validate the response format
                if not all(key in routing_decision for key in ['method', 'confidence', 'reasoning']):
                    raise ValueError("Missing required keys in routing response")
                
                # Update statistics
                if routing_decision['method'] == 'analytical':
                    self.stats['analytical_routings'] += 1
                else:
                    self.stats['semantic_routings'] += 1
                
                logger.info(f"LLM routing: {routing_decision['method']} ({routing_decision.get('confidence', 0):.2f}) - {routing_decision.get('reasoning', '')}")
                
                return routing_decision
                
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"Failed to parse LLM routing response: {e}")
                logger.error(f"Raw response: {response}")
                
                # Fallback to simple heuristics
                return self._fallback_routing(query)
        
        except Exception as e:
            logger.error(f"Error in LLM query routing: {e}")
            self.stats['routing_errors'] += 1
            
            # Fallback to simple heuristics
            return self._fallback_routing(query)
    
    def _get_llm_response(self, prompt: str) -> str:
        """Get response from LLM for routing decision."""
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
                timeout=30  # Quick timeout for routing
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get('message', {}).get('content', '')
            
        except Exception as e:
            logger.error(f"Error getting LLM routing response: {e}")
            raise
    
    def _clean_llm_response(self, response: str) -> str:
        """Clean LLM response to extract only the JSON part."""
        import re
        
        # Remove <think> tags and their content
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL | re.IGNORECASE)
        
        # Find JSON object in the response
        # Look for the first { and last } to extract JSON
        start_idx = response.find('{')
        end_idx = response.rfind('}')
        
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_part = response[start_idx:end_idx+1]
            return json_part.strip()
        
        # If no JSON found, return the cleaned response
        return response.strip()
    
    def _fallback_routing(self, query: str) -> Dict:
        """Fallback routing using simple heuristics when LLM fails."""
        query_lower = query.lower()
        
        # Simple analytical indicators
        analytical_indicators = [
            'who is', 'who was', 'who posted', 'most active', 'how many',
            'what wattage', 'what temperature', 'what settings', 'highest rated',
            'first user', 'second user', 'count', 'number of',
            'thread author', 'thread creator', 'who created', 'who started',
            'original poster', 'op', 'thread starter'
        ]
        
        if any(indicator in query_lower for indicator in analytical_indicators):
            return {
                'method': 'analytical',
                'confidence': 0.7,
                'reasoning': 'Fallback: Query contains analytical indicators',
                'query_type': 'fallback_analytical',
                'search_depth': 10
            }
        else:
            return {
                'method': 'semantic',
                'confidence': 0.6,
                'reasoning': 'Fallback: Default to semantic search',
                'query_type': 'fallback_semantic',
                'search_depth': 15
            }
    
    def get_stats(self) -> Dict:
        """Get routing statistics."""
        return self.stats.copy()


__all__ = ['LLMQueryRouter']