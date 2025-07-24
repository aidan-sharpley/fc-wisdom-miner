"""
Response refinement for Forum Wisdom Miner.

This module takes raw LLM responses and refines them into clean,
user-friendly answers by removing stream-of-consciousness elements
and focusing on actionable information.
"""

import logging
import re
from typing import Dict, List, Optional

import requests

from config.settings import OLLAMA_BASE_URL, OLLAMA_CHAT_MODEL

logger = logging.getLogger(__name__)


class ResponseRefiner:
    """Refines raw LLM responses into polished, user-friendly answers."""
    
    def __init__(self):
        """Initialize response refiner."""
        self.chat_url = f"{OLLAMA_BASE_URL}/api/chat"
        self.model = OLLAMA_CHAT_MODEL
        
        # Statistics
        self.stats = {
            'responses_refined': 0,
            'total_refinement_time': 0,
            'average_refinement_time': 0
        }
    
    def refine_response(self, raw_response: str, query: str, query_type: str = 'semantic') -> str:
        """Refine a raw LLM response into a clean, user-friendly answer.
        
        Args:
            raw_response: The original LLM response
            query: The original user query
            query_type: Type of query (semantic, analytical, etc.)
            
        Returns:
            Refined response text
        """
        import time
        start_time = time.time()
        
        logger.info(f"Refining {query_type} response for query: '{query[:50]}...'")
        
        try:
            # Quick heuristic check - if response is already clean, don't refine
            if self._is_response_already_clean(raw_response):
                logger.info("Response already clean, skipping refinement")
                return raw_response
            
            # Build refinement prompt
            refinement_prompt = self._build_refinement_prompt(raw_response, query, query_type)
            
            # Get refined response from LLM
            refined_response = self._get_refined_response(refinement_prompt)
            
            # Update statistics
            refinement_time = time.time() - start_time
            self.stats['responses_refined'] += 1
            self.stats['total_refinement_time'] += refinement_time
            self.stats['average_refinement_time'] = (
                self.stats['total_refinement_time'] / self.stats['responses_refined']
            )
            
            logger.info(f"Response refined in {refinement_time:.2f}s")
            return refined_response
            
        except Exception as e:
            logger.error(f"Error refining response: {e}")
            # Return original response if refinement fails
            return raw_response
    
    def _is_response_already_clean(self, response: str) -> bool:
        """Check if response is already clean and doesn't need refinement.
        
        Args:
            response: Response text to check
            
        Returns:
            True if response is already clean
        """
        # Check for stream-of-consciousness indicators
        thinking_indicators = [
            "let me think",
            "i'm thinking", 
            "let's see",
            "first, looking at",
            "okay, so",
            "let me analyze",
            "i need to figure out",
            "looking at this",
            "so i'm looking at",
            "this might be",
            "looking at the posts",
            "post 1 from",
            "post 2 from", 
            "post 3 from",
            "putting all this together",
            "so the answer would",
            "here's a cleaner",
            "there are several issues",
            "needs improvement"
        ]
        
        response_lower = response.lower()
        
        # If response contains thinking indicators, it needs refinement
        if any(indicator in response_lower for indicator in thinking_indicators):
            return False
        
        # If response is very long (>2000 chars), it might be rambling
        if len(response) > 2000:
            return False
        
        # If response has proper structure (headers, bullets), it's likely clean
        if ('**' in response or '##' in response or 
            response.count('\n•') > 2 or response.count('\n-') > 2):
            return True
        
        return True  # Default to clean if unsure
    
    def _build_refinement_prompt(self, raw_response: str, query: str, query_type: str) -> str:
        """Build prompt for refining the response.
        
        Args:
            raw_response: Original response to refine
            query: Original user query
            query_type: Type of query
            
        Returns:
            Refinement prompt
        """
        refinement_prompt = f"""Please refine this response to make it clear, concise, and user-friendly.

ORIGINAL QUERY: "{query}"

RAW RESPONSE TO REFINE:
{raw_response}

REFINEMENT INSTRUCTIONS:
1. Remove any "thinking out loud" or stream-of-consciousness elements
2. Focus on the key information that directly answers the user's question
3. Use clear formatting with headers (##) and bullet points (•) where appropriate
4. Keep technical information but make it accessible
5. Provide specific, actionable information
6. If the response mentions temperature ranges, product names, or specific recommendations, keep those details
7. Remove unnecessary explanations about your reasoning process
8. Start directly with the answer, not preambles like "Based on the information..."

REFINED RESPONSE:"""

        return refinement_prompt
    
    def _get_refined_response(self, prompt: str) -> str:
        """Get refined response from LLM.
        
        Args:
            prompt: Refinement prompt
            
        Returns:
            Refined response text
        """
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
                timeout=60  # Shorter timeout for refinement
            )
            response.raise_for_status()
            
            result = response.json()
            refined_text = result.get('message', {}).get('content', 'Refinement failed')
            
            return refined_text
            
        except Exception as e:
            logger.error(f"Error getting refined response: {e}")
            raise
    
    def refine_response_stream(self, raw_response_stream, query: str, query_type: str = 'semantic'):
        """Refine a streaming response.
        
        Args:
            raw_response_stream: Generator yielding response chunks
            query: Original user query
            query_type: Type of query
            
        Yields:
            Refined response chunks
        """
        # Collect the full response first
        full_response = ""
        for chunk in raw_response_stream:
            full_response += chunk
        
        # Refine the complete response
        refined_response = self.refine_response(full_response, query, query_type)
        
        # Yield refined response in chunks
        words = refined_response.split()
        chunk_size = 5  # Words per chunk
        
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i+chunk_size])
            if i + chunk_size < len(words):
                chunk += ' '
            yield chunk
    
    def get_stats(self) -> Dict:
        """Get refinement statistics."""
        return self.stats.copy()


__all__ = ['ResponseRefiner']