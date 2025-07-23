"""
Configuration and settings for Forum Wisdom Miner.

This module centralizes all configuration parameters, environment variables,
and constants used throughout the application.
"""

import os
import threading

# ==================== Core Application Settings ====================

# Base directory for data storage
BASE_TMP_DIR = os.environ.get("BASE_TMP_DIR", "tmp")
THREADS_DIR = os.path.join(BASE_TMP_DIR, "threads")

# API endpoints for Ollama services
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_API_URL = os.environ.get("OLLAMA_API_URL", "http://localhost:11434/api/generate")
OLLAMA_EMBED_API_URL = os.environ.get(
    "OLLAMA_EMBED_API_URL", "http://localhost:11434/api/embeddings"
)

# Model configurations
OLLAMA_CHAT_MODEL = os.environ.get("OLLAMA_CHAT_MODEL", "deepseek-r1:1.5b")
OLLAMA_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text:v1.5")

# ==================== File Naming Constants ====================

# Standard file names for thread data
INDEX_META_NAME = "index_meta.pkl"
HNSW_INDEX_NAME = "index_hnsw.bin"
METADATA_INDEX_NAME = "metadata_index.json"
POST_MAPPING_NAME = "post_mapping.json"
THREAD_ANALYTICS_NAME = "thread_analytics.json"  # New: Thread-level analytics
CACHE_PATH = os.path.join(BASE_TMP_DIR, "embeddings_cache.pkl")

# ==================== Performance Tuning Parameters ====================

# HTTP and API settings
API_TIMEOUT = 15
MAX_RETRIES = 3
RETRY_BACKOFF = 2

# Text processing settings
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
MIN_POST_LENGTH = 10  # Minimum characters for a valid post
MAX_POST_LENGTH = 10000  # Maximum characters for a valid post

# Search and retrieval settings
QUERY_RERANK_SIZE = 20
BATCH_RERANK_TIMEOUT = 45
FINAL_TOP_K = 7
MAX_WORKERS = 4  # Maximum concurrent threads for embedding

# Embedding settings
EMBEDDING_BATCH_SIZE = 10
EMBEDDING_MAX_RETRIES = 3
EMBEDDING_CACHE_SIZE = 1000
DELAY_BETWEEN_REQUESTS = 0.1

# ==================== HNSW Index Parameters ====================

# HNSW algorithm parameters optimized for forum content
HNSW_M = 32  # Lower M for faster search
HNSW_EF_CONSTRUCTION = 200  # Lower for faster index building
HNSW_EF_SEARCH = 100  # Search parameter

# Maximum number of elements for HNSW index
HNSW_MAX_ELEMENTS = 12000

# Expected embedding dimension (adjust based on model)
EXPECTED_EMBED_DIM = 768

# ==================== Thread Safety ====================

# Global locks for thread-safe operations
_cache_lock = threading.RLock()  # Reentrant lock for embedding cache operations
_file_lock = threading.Lock()  # Lock for file operations

# ==================== Analytics and UI Settings ====================

# Analytics collection settings
ENABLE_DETAILED_ANALYTICS = True
ANALYTICS_RETENTION_DAYS = 30

# UI progress and feedback settings
PROGRESS_UPDATE_INTERVAL = 0.1  # Seconds between progress updates
MAX_PROMPT_LENGTH = 1000  # Maximum characters in user prompts
UI_TIMEOUT_SECONDS = 120  # UI timeout for long operations

# ==================== Forum Scraping Settings ====================

# Web scraping parameters
USER_AGENT = "Mozilla/5.0 (compatible; ForumWisdomMiner/1.0)"
REQUEST_DELAY = 0.5  # Seconds between requests to be respectful
MAX_PAGES_PER_THREAD = 1000  # Safety limit

# CSS selectors for different forum types (in priority order)
POST_SELECTORS = [
    "article.message",  # XenForo
    ".post",  # Generic
    ".postbit_legacy",  # vBulletin
    "[data-post-id]",  # Data attribute fallback
    ".message",  # Alternative
    ".postbody",  # phpBB
]

# Content extraction selectors
CONTENT_SELECTORS = [
    "div.message-userContent .bbWrapper",  # XenForo
    ".message-body .bbWrapper",  # XenForo alt
    ".postbody .content",  # phpBB
    ".post_message",  # vBulletin
    ".message-content",  # Generic
]

# Author extraction selectors
AUTHOR_SELECTORS = [
    ".message-name .username",  # XenForo
    ".author .username",  # Generic
    ".postbit_legacy .bigusername",  # vBulletin
    ".postauthor .username",  # phpBB
]

# Date extraction selectors
DATE_SELECTORS = ["time[datetime]", ".message-date time", ".postDate", "[data-time]"]

# ==================== Enhanced Query Processing ====================

# Query analysis confidence thresholds
MIN_INTENT_CONFIDENCE = 0.3
HIGH_CONFIDENCE_THRESHOLD = 0.7
VERY_HIGH_CONFIDENCE_THRESHOLD = 0.9

# Query types and their weights for analytical queries
ANALYTICAL_QUERY_PATTERNS = {
    "summary": ["summary", "summarize", "overview", "tldr", "main points"],
    "sentiment": ["sentiment", "opinion", "feeling", "tone", "mood"],
    "statistics": ["how many", "count", "number of", "statistics", "stats"],
    "timeline": ["timeline", "chronology", "sequence", "progression", "evolution"],
    "comparison": ["compare", "difference", "versus", "vs", "contrast"],
    "trends": ["trend", "pattern", "development", "change over time"],
    "key_topics": ["topics", "themes", "subjects", "main discussion"],
    "participants": ["who", "participants", "contributors", "active users"],
    "conclusion": ["conclusion", "result", "outcome", "resolution"],
    "controversy": ["controversy", "debate", "disagreement", "argument"],
}

# ==================== Cache Management ====================

# Cache size limits and management
MAX_CACHE_SIZE_MB = 500  # Maximum cache size in MB
CACHE_CLEANUP_THRESHOLD = 0.8  # Cleanup when cache reaches 80% of max
CACHE_VALIDATION_INTERVAL = 3600  # Seconds between cache validations

# ==================== Logging Configuration ====================

# Log levels for different components
LOG_LEVELS = {
    "scraping": "INFO",
    "embedding": "INFO",
    "search": "INFO",
    "analytics": "DEBUG",
    "performance": "INFO",
}

# Performance monitoring settings
ENABLE_PERFORMANCE_MONITORING = True
PERFORMANCE_LOG_THRESHOLD = 1.0  # Log operations taking longer than 1 second

# ==================== Error Handling ====================

# Retry configurations for different operations
RETRY_CONFIG = {
    "embedding": {"max_retries": 3, "backoff": 2.0},
    "search": {"max_retries": 2, "backoff": 1.5},
    "scraping": {"max_retries": 3, "backoff": 1.0},
    "file_ops": {"max_retries": 2, "backoff": 0.5},
}

# Error message templates
ERROR_MESSAGES = {
    "no_posts_found": "No relevant posts found for your query. The thread may be empty or your question may not be covered.",
    "processing_failed": "Thread processing failed. Please check if the URL is valid and accessible.",
    "llm_unavailable": "AI service is currently unavailable. Please check if Ollama is running.",
    "invalid_url": "The provided URL appears to be invalid or inaccessible.",
    "timeout": "The operation timed out. Please try again with a simpler query.",
}

# ==================== Feature Flags ====================

# Feature toggles for experimental or optional features
FEATURES = {
    "enable_hyde": True,  # HyDE (Hypothetical Document Embeddings)
    "enable_reranking": True,  # LLM-based reranking
    "enable_analytics": True,  # Thread analytics collection
    "enable_caching": True,  # Embedding caching
    "enable_compression": False,  # Index compression (experimental)
    "enable_auto_cleanup": True,  # Automatic cache cleanup
    "debug_mode": False,  # Enhanced debugging information
}

# ==================== Export All Settings ====================

__all__ = [
    # Core settings
    "BASE_TMP_DIR",
    "THREADS_DIR",
    "OLLAMA_BASE_URL",
    "OLLAMA_API_URL",
    "OLLAMA_EMBED_API_URL",
    "OLLAMA_CHAT_MODEL",
    "OLLAMA_EMBED_MODEL",
    # File names
    "INDEX_META_NAME",
    "HNSW_INDEX_NAME",
    "METADATA_INDEX_NAME",
    "POST_MAPPING_NAME",
    "THREAD_ANALYTICS_NAME",
    "CACHE_PATH",
    # Performance parameters
    "API_TIMEOUT",
    "MAX_RETRIES",
    "RETRY_BACKOFF",
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
    "QUERY_RERANK_SIZE",
    "FINAL_TOP_K",
    "MAX_WORKERS",
    # HNSW parameters
    "HNSW_M",
    "HNSW_EF_CONSTRUCTION",
    "HNSW_EF_SEARCH",
    "EXPECTED_EMBED_DIM",
    # Thread safety
    "_cache_lock",
    "_file_lock",
    # UI and analytics
    "ENABLE_DETAILED_ANALYTICS",
    "MAX_PROMPT_LENGTH",
    "UI_TIMEOUT_SECONDS",
    # Forum scraping
    "POST_SELECTORS",
    "CONTENT_SELECTORS",
    "AUTHOR_SELECTORS",
    "DATE_SELECTORS",
    # Query processing
    "ANALYTICAL_QUERY_PATTERNS",
    "MIN_INTENT_CONFIDENCE",
    # Error handling
    "RETRY_CONFIG",
    "ERROR_MESSAGES",
    # Features
    "FEATURES",
]
