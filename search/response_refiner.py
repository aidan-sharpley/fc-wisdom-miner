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
            
            # Step 1: Clean obvious problems with text processing
            pre_cleaned = self._pre_clean_response(raw_response)
            
            # Step 2: Use LLM refinement only if still needed
            if self._is_response_already_clean(pre_cleaned):
                logger.info("Response cleaned with text processing only")
                refined_response = pre_cleaned
            else:
                # Build refinement prompt
                refinement_prompt = self._build_refinement_prompt(pre_cleaned, query, query_type)
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
            "needs improvement",
            "i'm looking at this query",
            "alright, i",
            "first, i need",
            "my goal is to",
            "<think>",
            "</think>"
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
    
    def _pre_clean_response(self, raw_response: str) -> str:
        """Pre-clean response using text processing to remove obvious issues.
        
        Args:
            raw_response: Raw response to clean
            
        Returns:
            Pre-cleaned response
        """
        import re
        
        # Remove <think> tags and their content completely
        cleaned = re.sub(r'<think>.*?</think>', '', raw_response, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove common thinking phrases
        thinking_patterns = [
            r'let me think.*?\.', 
            r'i\'m looking at this.*?\.', 
            r'alright,? i.*?\.', 
            r'first,? looking at.*?\.', 
            r'okay,? so.*?\.', 
            r'putting all this together.*?\.', 
            r'so the answer would.*?\.', 
            r'let me analyze.*?\.', 
            r'i need to figure out.*?\.',
            r'my goal is to.*?\.',
            r'post \d+ from.*?\.',
            r'looking at the posts.*?\.'
        ]
        
        for pattern in thinking_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
        
        # Clean up extra whitespace and line breaks
        cleaned = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned)  # Remove triple+ line breaks
        cleaned = re.sub(r'^\s+', '', cleaned, flags=re.MULTILINE)  # Remove leading whitespace
        cleaned = cleaned.strip()
        
        # If we removed everything, return a basic fallback
        if not cleaned or len(cleaned) < 20:
            return "Information about this topic is available in the forum posts."
        
        return cleaned
    
    def _build_refinement_prompt(self, raw_response: str, query: str, query_type: str) -> str:
        """Build prompt for refining the response.
        
        Args:
            raw_response: Original response to refine
            query: Original user query
            query_type: Type of query
            
        Returns:
            Refinement prompt
        """
        refinement_prompt = f"""Turn this into a well-formatted answer. Focus on structure and clarity. DO NOT change any factual information.

USER ASKED: "{query}"

TEXT TO IMPROVE:
{raw_response}

Make it:
- Well-structured with ## headers and • bullet points
- Direct and to-the-point
- Remove any references to "posts" or "looking at" 
- Start with the main answer
- Use professional formatting
- IMPORTANT: Keep all factual information exactly as stated (temperatures, numbers, etc.)

IMPROVED ANSWER:"""

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