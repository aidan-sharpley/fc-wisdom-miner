"""
Embedding management for Forum Wisdom Miner.

This module handles generation and management of text embeddings for semantic search.
"""

import logging
import time
from typing import Dict, List, Optional, Union

import numpy as np
import requests
from tqdm import tqdm

from config.settings import (
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_CACHE_SIZE,
    EMBEDDING_MAX_RETRIES,
    OLLAMA_BASE_URL,
    OLLAMA_EMBED_MODEL,
    BASE_TMP_DIR,
)
from utils.advanced_cache import ContentBasedCache
from utils.helpers import hash_text
from utils.monitoring import monitor_embedding_operation

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Manages text embeddings with caching and batch processing."""

    def __init__(self, model: str = None, base_url: str = None):
        """Initialize the embedding manager.

        Args:
            model: Ollama embedding model name
            base_url: Ollama base URL
        """
        self.model = model or OLLAMA_EMBED_MODEL
        self.base_url = base_url or OLLAMA_BASE_URL
        self.embed_url = f"{self.base_url}/api/embeddings"

        # Advanced cache for embeddings (M1 MacBook Air 8GB optimized)
        cache_dir = f"{BASE_TMP_DIR}/embeddings_cache"
        self.cache = ContentBasedCache(cache_dir, max_size_mb=150)  # Reduced from 200MB for 8GB systems
        
        # Legacy cache tracking for compatibility
        self._cache_hits = 0
        self._cache_misses = 0

        # Statistics
        self.stats = {
            "total_embeddings_generated": 0,
            "total_api_calls": 0,
            "total_processing_time": 0,
            "average_embedding_time": 0,
            "cache_hit_rate": 0,
        }

    @monitor_embedding_operation
    def get_embeddings(
        self, texts: Union[str, List[str]], use_cache: bool = True, 
        preprocess: bool = True, progress_callback=None
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """Get embeddings for one or more texts with optional preprocessing.

        Args:
            texts: Single text string or list of text strings
            use_cache: Whether to use cached embeddings
            preprocess: Whether to apply domain-specific preprocessing
            progress_callback: Optional callback for progress updates

        Returns:
            Single embedding array or list of embedding arrays
        """
        is_single = isinstance(texts, str)
        if is_single:
            texts = [texts]

        # Apply domain-specific preprocessing
        if preprocess:
            texts = [self._preprocess_for_embedding(text) for text in texts]

        embeddings = []
        uncached_texts = []
        uncached_indices = []

        # Check cache first
        for i, text in enumerate(texts):
            if use_cache:
                # Generate content hash for cache validation
                content_hash = hash_text(text)
                cache_key = f"embedding:{self.model}:{content_hash}"
                
                cached_embedding = self.cache.get(cache_key, content_hash)
                if cached_embedding is not None:
                    embeddings.append(cached_embedding)
                    self._cache_hits += 1
                    continue

            embeddings.append(None)  # Placeholder
            uncached_texts.append(text)
            uncached_indices.append(i)
            self._cache_misses += 1

        # Generate embeddings for uncached texts
        if uncached_texts:
            start_time = time.time()
            new_embeddings = self._generate_embeddings_batch(uncached_texts, progress_callback)
            processing_time = time.time() - start_time

            self.stats["total_processing_time"] += processing_time
            self.stats["total_embeddings_generated"] += len(new_embeddings)

            # Insert new embeddings and cache them
            for idx, embedding in zip(uncached_indices, new_embeddings):
                embeddings[idx] = embedding
                if use_cache:
                    # Cache with advanced cache system
                    text = uncached_texts[uncached_indices.index(idx)]
                    content_hash = hash_text(text)
                    cache_key = f"embedding:{self.model}:{content_hash}"
                    self.cache.set(cache_key, embedding, content_hash, ttl_hours=168)  # 1 week

        # Update statistics
        self._update_stats()

        return embeddings[0] if is_single else embeddings

    def _generate_embeddings_batch(self, texts: List[str], progress_callback=None) -> List[np.ndarray]:
        """Generate embeddings for a batch of texts.

        Args:
            texts: List of texts to embed
            progress_callback: Optional callback for progress updates

        Returns:
            List of embedding arrays
        """
        embeddings = []
        total_batches = (len(texts) + EMBEDDING_BATCH_SIZE - 1) // EMBEDDING_BATCH_SIZE

        # Process in batches with progress updates
        if progress_callback and len(texts) > 50:  # Only send detailed updates for large batches
            for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
                batch = texts[i : i + EMBEDDING_BATCH_SIZE]
                batch_embeddings = self._generate_single_batch(batch)
                embeddings.extend(batch_embeddings)
                
                # Send progress update
                completed = len(embeddings)
                progress_percent = (completed / len(texts)) * 100
                progress_callback(f"Generating embeddings: {completed}/{len(texts)} ({progress_percent:.1f}%)")
        else:
            # Use tqdm for console progress when no callback
            with tqdm(total=total_batches, desc="Generating embeddings", unit="batch") as pbar:
                for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
                    batch = texts[i : i + EMBEDDING_BATCH_SIZE]
                    batch_embeddings = self._generate_single_batch(batch)
                    embeddings.extend(batch_embeddings)
                    pbar.update(1)
                    pbar.set_postfix({"embeddings": len(embeddings), "total": len(texts)})

        return embeddings

    def _generate_single_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for a single batch using Ollama API.

        Args:
            texts: Batch of texts to embed

        Returns:
            List of embedding arrays
        """
        embeddings = []

        for text in texts:
            for attempt in range(EMBEDDING_MAX_RETRIES):
                try:
                    response = requests.post(
                        self.embed_url,
                        json={"model": self.model, "prompt": text},
                        timeout=60,
                    )
                    response.raise_for_status()

                    result = response.json()
                    embedding = np.array(result["embedding"], dtype=np.float32)
                    embeddings.append(embedding)
                    self.stats["total_api_calls"] += 1
                    break

                except requests.exceptions.RequestException as e:
                    logger.warning(f"Embedding API error (attempt {attempt + 1}): {e}")
                    if attempt < EMBEDDING_MAX_RETRIES - 1:
                        time.sleep(2**attempt)  # Exponential backoff
                    else:
                        # Fallback: return zero vector
                        logger.error(
                            f"Failed to generate embedding after {EMBEDDING_MAX_RETRIES} attempts"
                        )
                        embeddings.append(
                            np.zeros(384, dtype=np.float32)
                        )  # Default dimension

                except Exception as e:
                    logger.error(f"Unexpected error generating embedding: {e}")
                    embeddings.append(np.zeros(384, dtype=np.float32))
                    break

        return embeddings

    def invalidate_cache_for_content(self, content_hash: str):
        """Invalidate cache entries for specific content.
        
        Args:
            content_hash: Content hash to invalidate
        """
        cache_key = f"embedding:{self.model}:{content_hash}"
        self.cache.invalidate(cache_key)

    def _update_stats(self):
        """Update internal statistics."""
        total_requests = self._cache_hits + self._cache_misses
        if total_requests > 0:
            self.stats["cache_hit_rate"] = self._cache_hits / total_requests

        if self.stats["total_embeddings_generated"] > 0:
            self.stats["average_embedding_time"] = (
                self.stats["total_processing_time"]
                / self.stats["total_embeddings_generated"]
            )

    def generate_hyde_embedding(self, query: str, context: str = "") -> np.ndarray:
        """Generate a HyDE (Hypothetical Document Embedding) for better search.

        HyDE creates a hypothetical document that would answer the query,
        then embeds that document for more effective semantic search.

        Args:
            query: User query
            context: Optional context about the thread

        Returns:
            HyDE embedding array
        """
        # Create hypothetical document
        hyde_prompt = self._create_hyde_prompt(query, context)

        # Generate embedding for the hypothetical document
        hyde_embedding = self.get_embeddings(hyde_prompt)

        logger.debug(f"Generated HyDE embedding for query: {query[:50]}...")
        return hyde_embedding

    def _create_hyde_prompt(self, query: str, context: str = "") -> str:
        """Create an enhanced hypothetical document prompt for HyDE optimized for vape/device forums.

        Args:
            query: User query
            context: Optional context

        Returns:
            Hypothetical document text optimized for device discussions
        """
        # Enhanced HyDE template for device/vape forums
        query_lower = query.lower()
        
        # Device-specific enhancements
        if any(term in query_lower for term in ['temperature', 'temp', 'heat', 'celsius', 'fahrenheit']):
            tech_context = "The discussion would include specific temperature settings, heating techniques, and temperature-related performance characteristics. Users would share their preferred temperature ranges and explain the effects of different heat settings."
        elif any(term in query_lower for term in ['battery', 'charge', 'power', 'voltage', 'watt']):
            tech_context = "The conversation would cover battery life, charging methods, power settings, and electrical specifications. Participants would discuss wattage recommendations and battery performance optimization."
        elif any(term in query_lower for term in ['coil', 'resistance', 'ohm', 'mesh', 'ceramic']):
            tech_context = "The discussion would focus on coil types, resistance values, material properties, and performance characteristics. Users would share experiences with different coil configurations and materials."
        elif any(term in query_lower for term in ['latest', 'new', 'recent', 'popular', 'best']):
            tech_context = "The thread would include recent developments, newest products, popular techniques, and current community preferences. Users would share up-to-date information and trending approaches."
        else:
            tech_context = "The discussion would include technical specifications, user experiences, troubleshooting tips, and practical advice from experienced community members."
        
        # Create enhanced hypothetical document
        hyde_template = f"""
        Forum Discussion Topic: {query}
        
        {tech_context}
        
        Community members would share detailed experiences including:
        - Specific settings and configurations that work well
        - Comparisons between different options or approaches  
        - Technical explanations of why certain methods are effective
        - Personal recommendations based on actual usage
        - Troubleshooting common issues and solutions
        - Links to relevant resources or additional information
        
        {context}
        
        The most upvoted and recent posts would provide the most valuable insights, with experienced users sharing their proven techniques and newer community members asking clarifying questions. The discussion would evolve to cover both basic concepts and advanced techniques.
        """

        return hyde_template.strip()

    def _preprocess_for_embedding(self, text: str) -> str:
        """Apply domain-specific preprocessing to optimize embeddings for vape/device content.
        
        Args:
            text: Original text
            
        Returns:
            Preprocessed text optimized for embedding
        """
        if not text:
            return text
            
        # Normalize technical terms and abbreviations
        text = self._normalize_technical_terms(text)
        
        # Enhance context for technical specifications
        text = self._enhance_technical_context(text)
        
        # Normalize units and measurements
        text = self._normalize_units(text)
        
        return text
    
    def _normalize_technical_terms(self, text: str) -> str:
        """Normalize technical terms and brand names for better semantic matching.
        
        Args:
            text: Input text
            
        Returns:
            Text with normalized technical terms
        """
        import re
        
        # Common technical term normalizations for vape/device forums
        normalizations = {
            # Temperature variants
            r'\b(?:temp|temps)\b': 'temperature',
            r'\b(?:°c|celsius)\b': 'degrees celsius',
            r'\b(?:°f|fahrenheit)\b': 'degrees fahrenheit',
            
            # Power/electrical terms
            r'\b(?:watts?|w)\b': 'wattage',
            r'\b(?:volts?|v)\b': 'voltage',
            r'\b(?:ohms?|ω)\b': 'resistance ohms',
            r'\b(?:amps?|a)\b': 'amperage',
            r'\bmah\b': 'milliamp hours',
            
            # Device components
            r'\b(?:atomizer|atty)\b': 'atomizer tank',
            r'\b(?:mod|device)\b': 'vaporizer device',
            r'\b(?:coils?)\b': 'heating coil',
            r'\b(?:tanks?)\b': 'liquid tank',
            
            # Materials
            r'\b(?:ss|stainless)\b': 'stainless steel',
            r'\b(?:kanthal|ni80)\b': 'heating wire',
            r'\bmesh\b': 'mesh coil',
            
            # Vaping specific
            r'\b(?:vg|vegetable glycerin)\b': 'vegetable glycerin',
            r'\b(?:pg|propylene glycol)\b': 'propylene glycol',
            r'\b(?:nic|nicotine)\b': 'nicotine',
            
            # Device types
            r'\b(?:dry herb|flower)\b': 'dry herb vaporizer',
            r'\b(?:concentrate|dab|wax)\b': 'concentrate vaporizer',
            r'\b(?:convection|conduction)\b': 'heating method',
        }
        
        for pattern, replacement in normalizations.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text
    
    def _enhance_technical_context(self, text: str) -> str:
        """Add contextual information to technical specifications for better embedding.
        
        Args:
            text: Input text
            
        Returns:
            Text with enhanced technical context
        """
        import re
        
        # Add context to isolated numbers/specs
        enhancements = []
        
        # Temperature specifications
        temp_matches = re.findall(r'(\d+)\s*(?:°?[cf]|celsius|fahrenheit)', text, re.IGNORECASE)
        if temp_matches:
            enhancements.append("temperature settings heating specifications")
        
        # Wattage specifications  
        watt_matches = re.findall(r'(\d+)\s*(?:w|watts?|wattage)', text, re.IGNORECASE)
        if watt_matches:
            enhancements.append("power consumption wattage settings")
        
        # Resistance specifications
        ohm_matches = re.findall(r'(\d+\.?\d*)\s*(?:ohms?|ω|resistance)', text, re.IGNORECASE)
        if ohm_matches:
            enhancements.append("coil resistance electrical specifications")
        
        # Battery specifications
        if re.search(r'(\d+)\s*mah', text, re.IGNORECASE):
            enhancements.append("battery capacity power specifications")
        
        # Volume specifications
        if re.search(r'(\d+)\s*ml', text, re.IGNORECASE):
            enhancements.append("liquid capacity tank volume")
        
        # Add enhancements to text
        if enhancements:
            text += " " + " ".join(enhancements)
        
        return text
    
    def _normalize_units(self, text: str) -> str:
        """Normalize unit representations for consistent embedding.
        
        Args:
            text: Input text
            
        Returns:
            Text with normalized units
        """
        import re
        
        # Normalize common unit representations
        unit_normalizations = {
            r'(\d+)\s*°?c\b': r'\1 degrees celsius',
            r'(\d+)\s*°?f\b': r'\1 degrees fahrenheit', 
            r'(\d+)\s*w\b': r'\1 watts',
            r'(\d+\.?\d*)\s*ω\b': r'\1 ohms',
            r'(\d+)\s*v\b': r'\1 volts',
            r'(\d+)\s*a\b': r'\1 amps',
            r'(\d+)\s*ml\b': r'\1 milliliters',
            r'(\d+)\s*g\b': r'\1 grams',
            r'(\d+)\s*mg\b': r'\1 milligrams',
        }
        
        for pattern, replacement in unit_normalizations.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text

    def clear_cache(self):
        """Clear the embedding cache."""
        self.cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        logger.info("Embedding cache cleared")

    def get_stats(self) -> Dict:
        """Get embedding manager statistics."""
        cache_stats = self.cache.get_stats()
        return {
            **self.stats,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "advanced_cache_stats": cache_stats,
        }

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings from this model.

        Returns:
            Embedding dimension
        """
        # Test with a small string to get dimension
        try:
            test_embedding = self.get_embeddings("test", use_cache=False)
            return len(test_embedding)
        except Exception:
            logger.warning("Could not determine embedding dimension, using default")
            return 384  # Default for many models


__all__ = ["EmbeddingManager"]
