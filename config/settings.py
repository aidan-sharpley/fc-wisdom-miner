"""
Configuration and settings for Forum Wisdom Miner.

This module centralizes all configuration parameters, environment variables,
and constants used throughout the application.
"""

import os
import threading

# ==================== Core Application Settings ====================

# Base directory for data storage
BASE_TMP_DIR = os.environ.get('BASE_TMP_DIR', 'tmp')
THREADS_DIR = os.path.join(BASE_TMP_DIR, 'threads')

# API endpoints for Ollama services
OLLAMA_BASE_URL = os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434')
OLLAMA_API_URL = os.environ.get('OLLAMA_API_URL', 'http://localhost:11434/api/generate')
OLLAMA_EMBED_API_URL = os.environ.get(
    'OLLAMA_EMBED_API_URL', 'http://localhost:11434/api/embeddings'
)

# Model configurations - Multi-model strategy for M1 8GB optimization
OLLAMA_CHAT_MODEL = os.environ.get('OLLAMA_CHAT_MODEL', 'deepseek-r1:1.5b')
OLLAMA_EMBED_MODEL = os.environ.get('OLLAMA_EMBED_MODEL', 'nomic-embed-text:v1.5')

# Specialized models for different tasks
OLLAMA_ANALYTICS_MODEL = os.environ.get('OLLAMA_ANALYTICS_MODEL', 'qwen2.5:0.5b')  # Ultra-fast for structured data
OLLAMA_NARRATIVE_MODEL = os.environ.get('OLLAMA_NARRATIVE_MODEL', 'qwen2.5:1.5b')  # Fast for creative tasks
OLLAMA_FALLBACK_MODEL = os.environ.get('OLLAMA_FALLBACK_MODEL', 'qwen2.5:0.5b')   # Emergency fallback

# ==================== File Naming Constants ====================

# Standard file names for thread data
INDEX_META_NAME = 'index_meta.pkl'
HNSW_INDEX_NAME = 'index_hnsw.bin'
METADATA_INDEX_NAME = 'metadata_index.json'
POST_MAPPING_NAME = 'post_mapping.json'
THREAD_ANALYTICS_NAME = 'thread_analytics.json'  # New: Thread-level analytics
CACHE_PATH = os.path.join(BASE_TMP_DIR, 'embeddings_cache.pkl')

# ==================== Performance Tuning Parameters ====================

# HTTP and API settings
API_TIMEOUT = 15
MAX_RETRIES = 3
RETRY_BACKOFF = 2

# Text processing settings
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
MIN_POST_LENGTH = 5  # Minimum characters for a valid post
MAX_POST_LENGTH = 10000  # Maximum characters for a valid post

# Search and retrieval settings optimized for M1 MacBook Air (8GB RAM)
QUERY_RERANK_SIZE = 15  # Reduced from 20 to save memory
BATCH_RERANK_TIMEOUT = 45
FINAL_TOP_K = 7
MAX_WORKERS = 2  # Further reduced for narrative generation stability

# LLM timeout configurations
LLM_TIMEOUT_FAST = 30     # For analytics and structured tasks
LLM_TIMEOUT_NARRATIVE = 45 # For narrative generation
LLM_TIMEOUT_FALLBACK = 60  # Emergency timeout

# Batch processing settings
NARRATIVE_BATCH_SIZE = 2   # Reduced from 4 for M1 stability
NARRATIVE_MAX_WORKERS = 2  # Concurrent narrative workers

# Embedding settings optimized for M1 performance
EMBEDDING_BATCH_SIZE = 8  # Reduced from 10 to prevent memory spikes on 8GB systems
EMBEDDING_MAX_RETRIES = 3
EMBEDDING_CACHE_SIZE = 800  # Reduced from 1000 for memory management
DELAY_BETWEEN_REQUESTS = 0.3  # Fast delay for local embedding operations

# ==================== HNSW Index Parameters ====================

# HNSW algorithm parameters optimized for M1 MacBook Air performance
HNSW_M = 12  # Reduced from 16 for M1 memory efficiency
HNSW_EF_CONSTRUCTION = 80  # Reduced from 100 for faster building on M1
HNSW_EF_SEARCH = 40  # Reduced from 50 for faster search on M1

# Maximum number of elements for HNSW index (8GB RAM optimized)
HNSW_MAX_ELEMENTS = 10000  # Reduced from 12000 for memory management

# Expected embedding dimension (nomic-embed-text:v1.5 = 768d)
EXPECTED_EMBED_DIM = 768  # Confirmed for nomic-embed-text model

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
MAX_LOGGED_PROMPT_LENGTH = 50  # Maximum characters to log from prompts
UI_TIMEOUT_SECONDS = 120  # UI timeout for long operations

# Cache management settings
QUERY_PROCESSOR_CACHE_SIZE = 50  # Maximum number of query processors to cache
EMBEDDING_CACHE_SIZE_MB = 150  # Maximum embedding cache size in MB

# ==================== Forum Scraping Settings ====================

# Web scraping parameters - respectful scraping settings
USER_AGENT = 'Mozilla/5.0 (compatible; ForumWisdomMiner/1.0)'
BASE_REQUEST_DELAY = 1.5  # Base delay between requests (seconds)
JITTER_RANGE = (0.5, 2.0)  # Random jitter range to add to base delay
MAX_PAGES_PER_THREAD = 1000  # Safety limit
RESPECTFUL_SCRAPING = True  # Enable respectful scraping features

# User-Agent rotation for more natural browsing patterns
USER_AGENTS = [
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0',
    'Mozilla/5.0 (compatible; ForumWisdomMiner/2.0; +https://github.com/user/fc-wisdom-miner)'
]

# CSS selectors for different forum types (in priority order)
POST_SELECTORS = [
    'article.message',  # XenForo
    '.post',  # Generic
    '.postbit_legacy',  # vBulletin
    '[data-post-id]',  # Data attribute fallback
    '.message',  # Alternative
    '.postbody',  # phpBB
]

# Content extraction selectors
CONTENT_SELECTORS = [
    'div.message-userContent .bbWrapper',  # XenForo
    '.message-body .bbWrapper',  # XenForo alt
    '.postbody .content',  # phpBB
    '.post_message',  # vBulletin
    '.message-content',  # Generic
]

# Author extraction selectors
AUTHOR_SELECTORS = [
    '.message-name .username',  # XenForo
    '.author .username',  # Generic
    '.postbit_legacy .bigusername',  # vBulletin
    '.postauthor .username',  # phpBB
]

# Date extraction selectors
DATE_SELECTORS = ['time[datetime]', '.message-date time', '.postDate', '[data-time]']

# Vote/reaction extraction selectors for different forum platforms
VOTE_SELECTORS = {
    'upvotes': [
        '.message-reaction-score[data-reaction="1"]',  # XenForo likes
        '.like-count',
        '.upvote-count',
        '.positive-count',
        '.reaction-score .positive',
        '[data-score]',
        '.vote-up .count',
        '.thumbs-up .count',
    ],
    'downvotes': [
        '.message-reaction-score[data-reaction="-1"]',  # XenForo dislikes
        '.dislike-count',
        '.downvote-count',
        '.negative-count',
        '.reaction-score .negative',
        '.vote-down .count',
        '.thumbs-down .count',
    ],
    'likes': [
        '.like-button .count',
        '.likes .count',
        '.heart .count',
        '.message-reaction[data-reaction="Like"] .count',
        '.likes-received',
        '[data-likes]',
    ],
    'reactions': [
        '.message-reactionSummary .count',
        '.reaction-count',
        '.total-reactions',
        '.emoji-count',
        '.reaction-bar .count',
    ],
}

# ==================== Enhanced Query Processing ====================

# Query analysis confidence thresholds
MIN_INTENT_CONFIDENCE = 0.3
HIGH_CONFIDENCE_THRESHOLD = 0.7
VERY_HIGH_CONFIDENCE_THRESHOLD = 0.9

# Query types and their weights for analytical queries
ANALYTICAL_QUERY_PATTERNS = {
    'summary': ['summary', 'summarize', 'overview', 'tldr', 'main points'],
    'sentiment': ['sentiment', 'opinion', 'feeling', 'tone', 'mood'],
    'statistics': ['how many', 'count', 'number of', 'statistics', 'stats'],
    'timeline': ['timeline', 'chronology', 'sequence', 'progression', 'evolution'],
    'comparison': ['compare', 'difference', 'versus', 'vs', 'contrast'],
    'trends': ['trend', 'pattern', 'development', 'change over time'],
    'key_topics': ['topics', 'themes', 'subjects', 'main discussion'],
    'participants': ['who', 'participants', 'contributors', 'active users'],
    'conclusion': ['conclusion', 'result', 'outcome', 'resolution'],
    'controversy': ['controversy', 'debate', 'disagreement', 'argument'],
}

# ==================== Cache Management ====================

# Cache size limits and management
MAX_CACHE_SIZE_MB = 500  # Maximum cache size in MB
CACHE_CLEANUP_THRESHOLD = 0.8  # Cleanup when cache reaches 80% of max
CACHE_VALIDATION_INTERVAL = 3600  # Seconds between cache validations

# ==================== Logging Configuration ====================

# Log levels for different components
LOG_LEVELS = {
    'scraping': 'INFO',
    'embedding': 'INFO',
    'search': 'INFO',
    'analytics': 'DEBUG',
    'performance': 'INFO',
}

# Performance monitoring settings
ENABLE_PERFORMANCE_MONITORING = True
PERFORMANCE_LOG_THRESHOLD = 1.0  # Log operations taking longer than 1 second

# ==================== Error Handling ====================

# Retry configurations for different operations
RETRY_CONFIG = {
    'embedding': {'max_retries': 3, 'backoff': 2.0},
    'search': {'max_retries': 2, 'backoff': 1.5},
    'scraping': {'max_retries': 3, 'backoff': 2.0, 'base_delay': 5.0},  # Respectful backoff
    'file_ops': {'max_retries': 2, 'backoff': 0.5},
}

# Error message templates
ERROR_MESSAGES = {
    'no_posts_found': 'No relevant posts found for your query. The thread may be empty or your question may not be covered.',
    'processing_failed': 'Thread processing failed. Please check if the URL is valid and accessible.',
    'llm_unavailable': 'AI service is currently unavailable. Please check if Ollama is running.',
    'invalid_url': 'The provided URL appears to be invalid or inaccessible.',
    'timeout': 'The operation timed out. Please try again with a simpler query.',
}

# ==================== Feature Flags ====================

# Feature toggles for experimental or optional features
FEATURES = {
    'enable_hyde': True,  # HyDE (Hypothetical Document Embeddings)
    'enable_reranking': True,  # LLM-based reranking
    'enable_analytics': True,  # Thread analytics collection
    'enable_caching': True,  # Embedding caching
    'enable_compression': False,  # Index compression (experimental)
    'enable_auto_cleanup': True,  # Automatic cache cleanup
    'debug_mode': False,  # Enhanced debugging information
}

# ==================== Export All Settings ====================

__all__ = [
    # Core settings
    'BASE_TMP_DIR',
    'THREADS_DIR',
    'OLLAMA_BASE_URL',
    'OLLAMA_API_URL',
    'OLLAMA_EMBED_API_URL',
    'OLLAMA_CHAT_MODEL',
    'OLLAMA_EMBED_MODEL',
    'OLLAMA_ANALYTICS_MODEL',
    'OLLAMA_NARRATIVE_MODEL', 
    'OLLAMA_FALLBACK_MODEL',
    # File names
    'INDEX_META_NAME',
    'HNSW_INDEX_NAME',
    'METADATA_INDEX_NAME',
    'POST_MAPPING_NAME',
    'THREAD_ANALYTICS_NAME',
    'CACHE_PATH',
    # Performance parameters
    'API_TIMEOUT',
    'MAX_RETRIES',
    'RETRY_BACKOFF',
    'CHUNK_SIZE',
    'CHUNK_OVERLAP',
    'QUERY_RERANK_SIZE',
    'FINAL_TOP_K',
    'MAX_WORKERS',
    'LLM_TIMEOUT_FAST',
    'LLM_TIMEOUT_NARRATIVE',
    'LLM_TIMEOUT_FALLBACK',
    'NARRATIVE_BATCH_SIZE',
    'NARRATIVE_MAX_WORKERS',
    # HNSW parameters
    'HNSW_M',
    'HNSW_EF_CONSTRUCTION',
    'HNSW_EF_SEARCH',
    'EXPECTED_EMBED_DIM',
    # Thread safety
    '_cache_lock',
    '_file_lock',
    # UI and analytics
    'ENABLE_DETAILED_ANALYTICS',
    'MAX_PROMPT_LENGTH',
    'UI_TIMEOUT_SECONDS',
    # Forum scraping
    'POST_SELECTORS',
    'CONTENT_SELECTORS',
    'AUTHOR_SELECTORS',
    'DATE_SELECTORS',
    'VOTE_SELECTORS',
    # Query processing
    'ANALYTICAL_QUERY_PATTERNS',
    'MIN_INTENT_CONFIDENCE',
    # Error handling
    'RETRY_CONFIG',
    'ERROR_MESSAGES',
    # Features
    'FEATURES',
]
