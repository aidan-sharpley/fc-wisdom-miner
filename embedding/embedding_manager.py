"""
Embedding management for Forum Wisdom Miner.

This module handles generation and management of text embeddings for semantic search.
"""

import logging
import time
from typing import Dict, List, Optional, Union

import numpy as np
import requests

from config.settings import (
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_CACHE_SIZE,
    EMBEDDING_MAX_RETRIES,
    OLLAMA_BASE_URL,
    OLLAMA_EMBED_MODEL,
)
from utils.helpers import hash_text

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

        # Cache for embeddings
        self._cache = {}
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

    def get_embeddings(
        self, texts: Union[str, List[str]], use_cache: bool = True
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """Get embeddings for one or more texts.

        Args:
            texts: Single text string or list of text strings
            use_cache: Whether to use cached embeddings

        Returns:
            Single embedding array or list of embedding arrays
        """
        is_single = isinstance(texts, str)
        if is_single:
            texts = [texts]

        embeddings = []
        uncached_texts = []
        uncached_indices = []

        # Check cache first
        for i, text in enumerate(texts):
            if use_cache:
                cached_embedding = self._get_cached_embedding(text)
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
            new_embeddings = self._generate_embeddings_batch(uncached_texts)
            processing_time = time.time() - start_time

            self.stats["total_processing_time"] += processing_time
            self.stats["total_embeddings_generated"] += len(new_embeddings)

            # Insert new embeddings and cache them
            for idx, embedding in zip(uncached_indices, new_embeddings):
                embeddings[idx] = embedding
                if use_cache:
                    self._cache_embedding(
                        uncached_texts[uncached_indices.index(idx)], embedding
                    )

        # Update statistics
        self._update_stats()

        return embeddings[0] if is_single else embeddings

    def _generate_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for a batch of texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding arrays
        """
        embeddings = []

        # Process in batches
        for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
            batch = texts[i : i + EMBEDDING_BATCH_SIZE]
            batch_embeddings = self._generate_single_batch(batch)
            embeddings.extend(batch_embeddings)

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

    def _get_cached_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding from cache if available.

        Args:
            text: Text to look up

        Returns:
            Cached embedding or None
        """
        text_hash = hash_text(text)
        return self._cache.get(text_hash)

    def _cache_embedding(self, text: str, embedding: np.ndarray):
        """Cache an embedding.

        Args:
            text: Original text
            embedding: Embedding array to cache
        """
        text_hash = hash_text(text)

        # Simple LRU-like cache management
        if len(self._cache) >= EMBEDDING_CACHE_SIZE:
            # Remove oldest entry (this is simplified - could use proper LRU)
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]

        self._cache[text_hash] = embedding

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
        """Create a hypothetical document prompt for HyDE.

        Args:
            query: User query
            context: Optional context

        Returns:
            Hypothetical document text
        """
        # Create a hypothetical answer to the query
        hyde_template = f"""
        Question: {query}
        
        A comprehensive answer to this question would include:
        
        The main points discussed in the forum thread would cover various perspectives and opinions from different participants. The discussion would likely include specific examples, detailed explanations, and personal experiences shared by community members.
        
        Key insights would emerge from the collective knowledge and diverse viewpoints presented throughout the conversation. The most relevant information would address the core aspects of the question while providing practical context and actionable insights.
        
        {context}
        
        The discussion would conclude with a synthesis of the main themes and takeaways that directly address the original question.
        """

        return hyde_template.strip()

    def clear_cache(self):
        """Clear the embedding cache."""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        logger.info("Embedding cache cleared")

    def get_stats(self) -> Dict:
        """Get embedding manager statistics."""
        return {
            **self.stats,
            "cache_size": len(self._cache),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
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
