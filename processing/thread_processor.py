"""
Thread-level processing functionality for Forum Wisdom Miner.

This module handles high-level thread processing including coordination
of scraping, processing, embedding, and indexing operations.
"""

import logging
import os
import time
from typing import Dict, List, Optional, Tuple

from analytics.thread_analyzer import ThreadAnalyzer
from config.settings import THREADS_DIR
from embedding.embedding_manager import EmbeddingManager
from embedding.hnsw_index import HNSWIndex
from processing.post_processor import PostProcessor
from scraping.forum_scraper import ForumScraper
from utils.file_utils import atomic_write_json, get_thread_dir, safe_read_json
from utils.helpers import normalize_url

logger = logging.getLogger(__name__)


class ThreadProcessor:
    """Coordinates processing of entire forum threads."""

    def __init__(self):
        """Initialize the thread processor."""
        self.scraper = ForumScraper()
        self.post_processor = PostProcessor()
        self.embedding_manager = EmbeddingManager()

        # Statistics
        self.stats = {
            "threads_processed": 0,
            "total_posts_processed": 0,
            "total_processing_time": 0,
            "average_processing_time": 0,
        }

    def process_thread(self, url: str, force_refresh: bool = False, progress_callback=None) -> Tuple[str, Dict]:
        """Process a complete forum thread.

        Args:
            url: Thread URL to process
            force_refresh: Force re-scraping even if thread exists
            progress_callback: Optional callback function for progress updates

        Returns:
            Tuple of (thread_key, processing_results)
        """
        start_time = time.time()
        normalized_url = normalize_url(url)
        thread_key = self._generate_thread_key(normalized_url)
        thread_dir = get_thread_dir(thread_key)

        logger.info(f"Processing thread: {thread_key}")

        try:
            # Step 1: Check if thread already exists and is up to date
            if not force_refresh and self._is_thread_current(thread_dir):
                logger.info(f"Thread {thread_key} is current, skipping processing")
                return thread_key, self._load_existing_results(thread_dir)

            # Step 2: Scrape the thread
            logger.info(f"Scraping thread from {normalized_url}")
            raw_posts, scrape_metadata = self.scraper.scrape_thread(
                normalized_url, save_html=True, thread_dir=thread_dir
            )

            if not raw_posts:
                raise ValueError("No posts found in thread")

            # Step 3: Process posts
            logger.info(f"Processing {len(raw_posts)} raw posts")
            processed_posts, processing_stats = self.post_processor.process_posts(
                raw_posts
            )

            if not processed_posts:
                raise ValueError("No valid posts after processing")

            # Step 4: Generate embeddings
            logger.info(f"Generating embeddings for {len(processed_posts)} posts")
            if progress_callback:
                progress_callback(f"Generating embeddings for {len(processed_posts)} posts...")
                logger.info("DEBUG: Progress callback is available and called for embeddings")
            else:
                logger.warning("DEBUG: No progress callback provided for embeddings")
            embeddings = self._generate_embeddings(processed_posts, progress_callback)

            # Step 5: Build/update search index
            logger.info("Building search index")
            if progress_callback:
                progress_callback("Building search index...")
            search_index = self._build_search_index(
                thread_dir, processed_posts, embeddings, progress_callback
            )

            # Step 6: Generate analytics
            logger.info("Generating thread analytics")
            analyzer = ThreadAnalyzer(thread_dir)
            analytics = analyzer.analyze_thread(processed_posts, force_refresh=True)

            # Step 7: Save processed data
            logger.info("Saving processed thread data")
            processing_results = self._save_thread_data(
                thread_dir,
                processed_posts,
                scrape_metadata,
                processing_stats,
                analytics,
            )

            # Update statistics
            processing_time = time.time() - start_time
            self.stats["threads_processed"] += 1
            self.stats["total_posts_processed"] += len(processed_posts)
            self.stats["total_processing_time"] += processing_time
            self.stats["average_processing_time"] = (
                self.stats["total_processing_time"] / self.stats["threads_processed"]
            )

            processing_results["processing_time"] = processing_time
            processing_results["thread_key"] = thread_key

            logger.info(
                f"Thread processing complete in {processing_time:.2f}s: {len(processed_posts)} posts"
            )
            return thread_key, processing_results

        except Exception as e:
            logger.error(f"Error processing thread {thread_key}: {e}")
            raise

    def reprocess_existing_thread(self, thread_key: str, progress_callback=None) -> Tuple[str, Dict]:
        """Reprocess an existing thread by re-parsing saved HTML files with new optimizations.

        Args:
            thread_key: Thread identifier
            progress_callback: Optional callback function for progress updates

        Returns:
            Tuple of (thread_key, processing_results)
        """
        start_time = time.time()
        thread_dir = get_thread_dir(thread_key)

        logger.info(f"Reprocessing existing thread from HTML files: {thread_key}")

        if not os.path.exists(thread_dir):
            raise ValueError(f"Thread {thread_key} not found")

        # Check for HTML files
        html_dir = os.path.join(thread_dir, "html_pages")
        if not os.path.exists(html_dir):
            # Fallback to old method if no HTML files exist
            logger.warning(
                f"No HTML files found for {thread_key}, using existing posts data"
            )
            return self._reprocess_from_posts_json(thread_key, progress_callback)

        try:
            # Step 1: Re-parse HTML files to extract raw posts
            logger.info("Re-parsing saved HTML files")
            raw_posts = self._reprocess_html_files(html_dir)

            if not raw_posts:
                raise ValueError("No posts found in HTML files during reprocessing")

            logger.info(f"Extracted {len(raw_posts)} raw posts from HTML files")

            # Step 2: Process posts with current optimizations
            logger.info("Processing posts with current optimizations")
            processed_posts, processing_stats = self.post_processor.process_posts(
                raw_posts
            )

            if not processed_posts:
                raise ValueError("No valid posts after processing HTML content")

            # Step 3: Generate embeddings with current embedding strategy
            logger.info(f"Generating embeddings for {len(processed_posts)} posts")
            if progress_callback:
                progress_callback(f"Regenerating embeddings for {len(processed_posts)} posts...")
            embeddings = self._generate_embeddings(processed_posts, progress_callback)

            # Step 4: Build/update search index
            logger.info("Rebuilding search index")
            if progress_callback:
                progress_callback("Rebuilding search index...")
            search_index = self._build_search_index(
                thread_dir, processed_posts, embeddings, progress_callback
            )

            # Step 5: Generate analytics with current analytics system
            logger.info("Regenerating thread analytics")
            analyzer = ThreadAnalyzer(thread_dir)
            analytics = analyzer.analyze_thread(processed_posts, force_refresh=True)

            # Step 6: Save reprocessed data
            logger.info("Saving reprocessed thread data")

            # Load existing metadata to preserve scrape info
            metadata_file = f"{thread_dir}/metadata.json"
            existing_metadata = safe_read_json(metadata_file) or {}

            # Update processing stats and reprocessing info
            existing_metadata.update(
                {
                    "processing_stats": processing_stats,
                    "posts_count": len(processed_posts),
                    "last_updated": time.time(),
                    "last_reprocessed": time.time(),
                    "reprocessing_count": existing_metadata.get("reprocessing_count", 0)
                    + 1,
                    "analytics_generated": time.time(),
                }
            )

            # Save updated posts and metadata
            posts_file = f"{thread_dir}/posts.json"
            atomic_write_json(posts_file, processed_posts)
            atomic_write_json(metadata_file, existing_metadata)

            # Update statistics
            processing_time = time.time() - start_time

            processing_results = {
                "posts_count": len(processed_posts),
                "metadata": existing_metadata,
                "analytics_summary": analytics.get("summary", {}),
                "from_cache": False,
                "reprocessed": True,
                "processing_time": processing_time,
                "reprocessing_method": "html_files",
            }

            logger.info(
                f"Thread reprocessing complete in {processing_time:.2f}s: {len(processed_posts)} posts"
            )
            return thread_key, processing_results

        except Exception as e:
            logger.error(f"Error reprocessing thread {thread_key}: {e}")
            raise

    def _reprocess_html_files(self, html_dir: str) -> List[Dict]:
        """Re-parse HTML files to extract raw posts.

        Args:
            html_dir: Directory containing saved HTML files

        Returns:
            List of raw post dictionaries
        """
        import glob

        from bs4 import BeautifulSoup

        all_raw_posts = []

        # Get all HTML files in order
        html_files = sorted(glob.glob(os.path.join(html_dir, "page_*.html")))

        if not html_files:
            logger.warning(f"No HTML files found in {html_dir}")
            return []

        logger.info(f"Found {len(html_files)} HTML files to reprocess")
        global_post_position = 1

        for i, html_file in enumerate(html_files):
            try:
                page_num = i + 1
                logger.debug(f"Reprocessing HTML file: {html_file} (page {page_num})")

                with open(html_file, "r", encoding="utf-8") as f:
                    html_content = f.read()

                soup = BeautifulSoup(html_content, "html.parser")
                posts = self.scraper._extract_posts(
                    soup, page_num, global_post_position
                )

                if posts:
                    all_raw_posts.extend(posts)
                    global_post_position += len(posts)
                    logger.debug(f"Extracted {len(posts)} posts from page {page_num}")

            except Exception as e:
                logger.error(f"Error reprocessing HTML file {html_file}: {e}")
                continue

        logger.info(f"Reprocessed HTML files: extracted {len(all_raw_posts)} raw posts")
        return all_raw_posts

    def _reprocess_from_posts_json(self, thread_key: str, progress_callback=None) -> Tuple[str, Dict]:
        """Fallback method: reprocess from existing posts.json for older threads.

        Args:
            thread_key: Thread identifier

        Returns:
            Tuple of (thread_key, processing_results)
        """
        start_time = time.time()
        thread_dir = get_thread_dir(thread_key)

        logger.info(f"Fallback reprocessing from posts.json: {thread_key}")

        try:
            # Load existing posts data
            posts_file = f"{thread_dir}/posts.json"
            if not os.path.exists(posts_file):
                raise ValueError(f"No posts data found for thread {thread_key}")

            posts = safe_read_json(posts_file)
            if not posts:
                raise ValueError(f"No valid posts data found for thread {thread_key}")

            logger.info(f"Reprocessing {len(posts)} existing posts")

            # Step 1: Generate fresh embeddings
            logger.info(f"Regenerating embeddings for {len(posts)} posts")
            if progress_callback:
                progress_callback(f"Regenerating embeddings for {len(posts)} posts...")
            embeddings = self._generate_embeddings(posts, progress_callback)

            # Step 2: Rebuild search index
            logger.info("Rebuilding search index")
            if progress_callback:
                progress_callback("Rebuilding search index...")
            search_index = self._build_search_index(thread_dir, posts, embeddings, progress_callback)

            # Step 3: Regenerate analytics
            logger.info("Regenerating thread analytics")
            analyzer = ThreadAnalyzer(thread_dir)
            analytics = analyzer.analyze_thread(posts, force_refresh=True)

            # Step 4: Update metadata
            logger.info("Updating thread metadata")
            metadata_file = f"{thread_dir}/metadata.json"
            existing_metadata = safe_read_json(metadata_file) or {}

            # Update with reprocessing info
            existing_metadata.update(
                {
                    "last_reprocessed": time.time(),
                    "reprocessing_count": existing_metadata.get("reprocessing_count", 0)
                    + 1,
                }
            )

            atomic_write_json(metadata_file, existing_metadata)

            # Update statistics
            processing_time = time.time() - start_time

            processing_results = {
                "posts_count": len(posts),
                "metadata": existing_metadata,
                "analytics_summary": analytics.get("summary", {}),
                "from_cache": False,
                "reprocessed": True,
                "processing_time": processing_time,
                "reprocessing_method": "posts_json_fallback",
            }

            logger.info(
                f"Fallback reprocessing complete in {processing_time:.2f}s: {len(posts)} posts"
            )
            return thread_key, processing_results

        except Exception as e:
            logger.error(f"Error in fallback reprocessing for thread {thread_key}: {e}")
            raise

    def _generate_thread_key(self, url: str) -> str:
        """Generate a secure unique key for a thread."""
        import hashlib
        from urllib.parse import urlparse
        from utils.security import sanitize_thread_key_component, validate_thread_key

        parsed = urlparse(url)
        # Use domain + path + query for uniqueness with SHA-256 for better security
        key_base = f"{parsed.netloc}{parsed.path}{parsed.query}"
        key_hash = hashlib.sha256(key_base.encode('utf-8')).hexdigest()[:12]

        # Try to extract a readable part and sanitize it
        path_parts = parsed.path.strip("/").split("/")
        readable_part = None

        for part in reversed(path_parts):
            if part and len(part) > 2:
                sanitized_part = sanitize_thread_key_component(part, max_length=20)
                if sanitized_part:  # Only use if sanitization produced valid result
                    readable_part = sanitized_part
                    break

        # Generate the thread key
        if readable_part:
            thread_key = f"{readable_part}_{key_hash}"
        else:
            thread_key = f"thread_{key_hash}"
        
        # Validate the generated key for security
        if not validate_thread_key(thread_key):
            # Fallback to hash-only if validation fails
            thread_key = f"thread_{key_hash}"
            
            # If even the fallback fails, use a simple secure format
            if not validate_thread_key(thread_key):
                thread_key = f"thread_{hashlib.sha256(url.encode('utf-8')).hexdigest()[:16]}"
        
        return thread_key

    def _is_thread_current(self, thread_dir: str, max_age_hours: int = 24) -> bool:
        """Check if a thread is current and doesn't need re-processing."""
        posts_file = f"{thread_dir}/posts.json"
        metadata_file = f"{thread_dir}/metadata.json"

        try:
            metadata = safe_read_json(metadata_file)
            if not metadata:
                return False

            # Check if files exist
            import os

            if not os.path.exists(posts_file):
                return False

            # Check age
            last_updated = metadata.get("last_updated", 0)
            age_hours = (time.time() - last_updated) / 3600

            return age_hours < max_age_hours

        except Exception:
            return False

    def _load_existing_results(self, thread_dir: str) -> Dict:
        """Load existing processing results."""
        try:
            metadata = safe_read_json(f"{thread_dir}/metadata.json") or {}
            posts = safe_read_json(f"{thread_dir}/posts.json") or []
            analytics = safe_read_json(f"{thread_dir}/analytics.json") or {}

            return {
                "posts_count": len(posts),
                "metadata": metadata,
                "analytics_summary": analytics.get("summary", {}),
                "from_cache": True,
            }
        except Exception as e:
            logger.error(f"Error loading existing results: {e}")
            return {}

    def _generate_embeddings(self, posts: List[Dict], progress_callback=None) -> List:
        """Generate embeddings for all posts."""
        texts = [post["content"] for post in posts]
        embeddings = self.embedding_manager.get_embeddings(texts, progress_callback=progress_callback)

        logger.info(f"Generated {len(embeddings)} embeddings")
        return embeddings

    def _build_search_index(
        self, thread_dir: str, posts: List[Dict], embeddings: List, progress_callback=None
    ) -> HNSWIndex:
        """Build or update the search index."""
        # Get embedding dimension
        dimension = len(embeddings[0]) if embeddings else 384

        # Create index
        index = HNSWIndex(thread_dir, dimension)

        # Clear and rebuild
        post_hashes = [post["hash"] for post in posts]
        index.rebuild_index(embeddings, post_hashes, progress_callback=progress_callback)

        # Save index
        index.save()

        logger.info(f"Built search index with {len(embeddings)} embeddings")
        return index

    def _save_thread_data(
        self,
        thread_dir: str,
        posts: List[Dict],
        scrape_metadata: Dict,
        processing_stats: Dict,
        analytics: Dict,
    ) -> Dict:
        """Save all thread data to disk."""
        import os

        os.makedirs(thread_dir, exist_ok=True)

        # Save posts
        posts_file = f"{thread_dir}/posts.json"
        atomic_write_json(posts_file, posts)

        # Save metadata
        metadata = {
            "scrape_metadata": scrape_metadata,
            "processing_stats": processing_stats,
            "posts_count": len(posts),
            "last_updated": time.time(),
            "analytics_generated": time.time(),
        }
        metadata_file = f"{thread_dir}/metadata.json"
        atomic_write_json(metadata_file, metadata)

        # Analytics are saved by ThreadAnalyzer

        return {
            "posts_count": len(posts),
            "metadata": metadata,
            "analytics_summary": analytics.get("summary", {}),
            "from_cache": False,
        }

    def get_thread_summary(self, thread_key: str) -> Optional[Dict]:
        """Get a summary of a processed thread.

        Args:
            thread_key: Thread identifier

        Returns:
            Thread summary or None if not found
        """
        try:
            thread_dir = get_thread_dir(thread_key)

            # Load metadata
            metadata = safe_read_json(f"{thread_dir}/metadata.json")
            if not metadata:
                return None

            # Load analytics summary
            analyzer = ThreadAnalyzer(thread_dir)
            analytics_summary = analyzer.get_summary()

            return {
                "thread_key": thread_key,
                "posts_count": metadata.get("posts_count", 0),
                "last_updated": metadata.get("last_updated", 0),
                "scrape_metadata": metadata.get("scrape_metadata", {}),
                "analytics": analytics_summary,
            }

        except Exception as e:
            logger.error(f"Error getting thread summary for {thread_key}: {e}")
            return None

    def list_processed_threads(self) -> List[Dict]:
        """List all processed threads with their summaries."""
        threads = []

        try:
            if not os.path.exists(THREADS_DIR):
                return threads

            for thread_key in os.listdir(THREADS_DIR):
                thread_path = os.path.join(THREADS_DIR, thread_key)
                if os.path.isdir(thread_path):
                    summary = self.get_thread_summary(thread_key)
                    if summary:
                        threads.append(summary)

            # Sort by last updated
            threads.sort(key=lambda t: t.get("last_updated", 0), reverse=True)

        except Exception as e:
            logger.error(f"Error listing processed threads: {e}")

        return threads

    def delete_thread(self, thread_key: str) -> bool:
        """Delete a processed thread and all its data.

        Args:
            thread_key: Thread identifier

        Returns:
            True if deleted successfully
        """
        try:
            thread_dir = get_thread_dir(thread_key)

            import os
            import shutil

            if os.path.exists(thread_dir):
                shutil.rmtree(thread_dir)
                logger.info(f"Deleted thread {thread_key}")
                return True

            return False

        except Exception as e:
            logger.error(f"Error deleting thread {thread_key}: {e}")
            return False

    def get_stats(self) -> Dict:
        """Get processing statistics."""
        return {
            **self.stats,
            "scraper_stats": self.scraper.get_stats(),
            "embedding_stats": self.embedding_manager.get_stats(),
        }


__all__ = ["ThreadProcessor"]
