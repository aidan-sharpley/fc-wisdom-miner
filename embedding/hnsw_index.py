"""
HNSW (Hierarchical Navigable Small World) index for efficient similarity search.

This module provides a fast approximate nearest neighbor search for embeddings.
"""

import logging
import os
import pickle
import time
from typing import Dict, List, Optional, Tuple

import hnswlib
import numpy as np

from config.settings import (
    HNSW_EF_CONSTRUCTION,
    HNSW_EF_SEARCH,
    HNSW_INDEX_NAME,
    HNSW_M,
    HNSW_MAX_ELEMENTS,
)
from utils.file_utils import atomic_write_json, safe_read_json

logger = logging.getLogger(__name__)


class HNSWIndex:
    """HNSW index wrapper for efficient similarity search."""

    def __init__(self, thread_dir: str, dimension: int = 384):
        """Initialize HNSW index for a thread.

        Args:
            thread_dir: Directory for this thread
            dimension: Embedding dimension
        """
        self.thread_dir = thread_dir
        self.dimension = dimension
        self.index_path = os.path.join(thread_dir, HNSW_INDEX_NAME)
        self.metadata_path = os.path.join(
            thread_dir, f'{HNSW_INDEX_NAME}.metadata.json'
        )

        # HNSW index
        self.index = None
        self.metadata = {
            'dimension': dimension,
            'num_elements': 0,
            'post_ids': [],  # Maps index position to post hash
            'created_at': time.time(),
            'last_updated': time.time(),
        }

        # Statistics
        self.stats = {
            'total_searches': 0,
            'total_search_time': 0,
            'average_search_time': 0,
            'last_search_results': 0,
        }

        self._load_or_create_index()

    def _load_or_create_index(self):
        """Load existing index or create a new one."""
        try:
            if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
                self._load_index()
                logger.info(
                    f'Loaded HNSW index with {self.metadata["num_elements"]} elements'
                )
            else:
                self._create_new_index()
                logger.info(f'Created new HNSW index with dimension {self.dimension}')
        except Exception as e:
            logger.error(f'Error loading/creating HNSW index: {e}')
            self._create_new_index()

    def _load_index(self):
        """Load existing HNSW index from disk."""
        # Load metadata
        metadata = safe_read_json(self.metadata_path)
        if metadata:
            self.metadata.update(metadata)
            self.dimension = self.metadata['dimension']

        # Load HNSW index
        self.index = hnswlib.Index(space='cosine', dim=self.dimension)
        self.index.load_index(self.index_path, max_elements=HNSW_MAX_ELEMENTS)

        # Set search parameters
        self.index.set_ef(HNSW_EF_SEARCH)

    def _create_new_index(self):
        """Create a new HNSW index."""
        self.index = hnswlib.Index(space='cosine', dim=self.dimension)
        self.index.init_index(
            max_elements=HNSW_MAX_ELEMENTS,
            ef_construction=HNSW_EF_CONSTRUCTION,
            M=HNSW_M,
        )
        self.index.set_ef(HNSW_EF_SEARCH)

        # Reset metadata
        self.metadata = {
            'dimension': self.dimension,
            'num_elements': 0,
            'post_ids': [],
            'created_at': time.time(),
            'last_updated': time.time(),
        }

    def add_embeddings(self, embeddings: List[np.ndarray], post_hashes: List[str]):
        """Add embeddings to the index.

        Args:
            embeddings: List of embedding arrays
            post_hashes: List of corresponding post hashes
        """
        if len(embeddings) != len(post_hashes):
            raise ValueError('Number of embeddings must match number of post hashes')

        if not embeddings:
            return

        # Convert to numpy array
        embedding_matrix = np.array(embeddings, dtype=np.float32)

        # Add to index
        start_idx = self.metadata['num_elements']
        indices = list(range(start_idx, start_idx + len(embeddings)))

        self.index.add_items(embedding_matrix, indices)

        # Update metadata
        self.metadata['post_ids'].extend(post_hashes)
        self.metadata['num_elements'] += len(embeddings)
        self.metadata['last_updated'] = time.time()

        logger.info(f'Added {len(embeddings)} embeddings to HNSW index')

    def search(
        self, query_embedding: np.ndarray, k: int = 10
    ) -> Tuple[List[str], List[float]]:
        """Search for similar embeddings.

        Args:
            query_embedding: Query embedding array
            k: Number of results to return

        Returns:
            Tuple of (post_hashes, distances)
        """
        if self.metadata['num_elements'] == 0:
            return [], []

        start_time = time.time()

        try:
            # Ensure query embedding is 2D
            if query_embedding.ndim == 1:
                query_embedding = query_embedding.reshape(1, -1)

            # Search
            indices, distances = self.index.knn_query(
                query_embedding, k=min(k, self.metadata['num_elements'])
            )

            # Convert to lists if single result
            if isinstance(indices, int):
                indices = [indices]
                distances = [distances]
            elif len(indices.shape) > 1:
                indices = indices[0]
                distances = distances[0]

            # Map indices to post hashes
            post_hashes = []
            for idx in indices:
                if 0 <= idx < len(self.metadata['post_ids']):
                    post_hashes.append(self.metadata['post_ids'][idx])
                else:
                    logger.warning(f'Invalid index {idx} in search results')

            # Update statistics
            search_time = time.time() - start_time
            self.stats['total_searches'] += 1
            self.stats['total_search_time'] += search_time
            self.stats['average_search_time'] = (
                self.stats['total_search_time'] / self.stats['total_searches']
            )
            self.stats['last_search_results'] = len(post_hashes)

            logger.debug(
                f'HNSW search completed in {search_time:.3f}s, found {len(post_hashes)} results'
            )

            return post_hashes, distances.tolist()

        except Exception as e:
            logger.error(f'Error during HNSW search: {e}')
            return [], []

    def update_embedding(self, post_hash: str, new_embedding: np.ndarray):
        """Update an existing embedding in the index.

        Args:
            post_hash: Hash of the post to update
            new_embedding: New embedding array
        """
        try:
            # Find the index of the post
            if post_hash in self.metadata['post_ids']:
                idx = self.metadata['post_ids'].index(post_hash)
                self.index.add_items(new_embedding.reshape(1, -1), [idx])
                self.metadata['last_updated'] = time.time()
                logger.debug(f'Updated embedding for post {post_hash}')
            else:
                logger.warning(f'Post hash {post_hash} not found in index')
        except Exception as e:
            logger.error(f'Error updating embedding: {e}')

    def remove_embedding(self, post_hash: str):
        """Remove an embedding from the index.

        Note: HNSW doesn't support efficient deletion, so this marks the item as deleted
        but doesn't actually remove it from the index.

        Args:
            post_hash: Hash of the post to remove
        """
        try:
            if post_hash in self.metadata['post_ids']:
                idx = self.metadata['post_ids'].index(post_hash)
                self.index.mark_deleted(idx)
                self.metadata['last_updated'] = time.time()
                logger.debug(f'Marked embedding for post {post_hash} as deleted')
            else:
                logger.warning(f'Post hash {post_hash} not found in index')
        except Exception as e:
            logger.error(f'Error removing embedding: {e}')

    def rebuild_index(self, embeddings: List[np.ndarray], post_hashes: List[str]):
        """Rebuild the entire index from scratch.

        Args:
            embeddings: All embeddings to include
            post_hashes: Corresponding post hashes
        """
        logger.info('Rebuilding HNSW index from scratch')

        # Create new index
        self._create_new_index()

        # Add all embeddings
        if embeddings:
            self.add_embeddings(embeddings, post_hashes)

        # Save immediately
        self.save()

        logger.info(f'Index rebuilt with {len(embeddings)} embeddings')

    def save(self):
        """Save the index and metadata to disk."""
        try:
            # Save HNSW index
            os.makedirs(self.thread_dir, exist_ok=True)
            self.index.save_index(self.index_path)

            # Save metadata
            atomic_write_json(self.metadata_path, self.metadata)

            logger.debug('HNSW index saved successfully')

        except Exception as e:
            logger.error(f'Error saving HNSW index: {e}')

    def get_stats(self) -> Dict:
        """Get index statistics."""
        return {
            **self.stats,
            'num_elements': self.metadata['num_elements'],
            'dimension': self.dimension,
            'index_size_mb': self._get_index_size_mb(),
            'last_updated': self.metadata['last_updated'],
        }

    def _get_index_size_mb(self) -> float:
        """Get index file size in MB."""
        try:
            if os.path.exists(self.index_path):
                size_bytes = os.path.getsize(self.index_path)
                return size_bytes / (1024 * 1024)
        except Exception:
            pass
        return 0.0

    def get_all_post_hashes(self) -> List[str]:
        """Get all post hashes in the index."""
        return self.metadata['post_ids'].copy()

    def contains_post(self, post_hash: str) -> bool:
        """Check if a post is in the index."""
        return post_hash in self.metadata['post_ids']

    def get_size(self) -> int:
        """Get the number of elements in the index."""
        return self.metadata['num_elements']

    def clear(self):
        """Clear the entire index."""
        logger.info('Clearing HNSW index')
        self._create_new_index()
        self.save()


__all__ = ['HNSWIndex']
