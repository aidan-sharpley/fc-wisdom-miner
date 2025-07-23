# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

This is a professional Python Flask application that requires Ollama running locally. The main application is in `app.py`.

### Running the Application
```bash
python app.py
```
The app will start on port 8080 at http://localhost:8080

### Prerequisites
- **Ollama must be running locally** with the required models:
  - `deepseek-r1:1.5b` (for text generation)
  - `nomic-embed-text:v1.5` (for embeddings)
- Python 3.9+ with required dependencies

### Dependencies
Install these Python packages (no requirements.txt found):
- Flask
- requests
- beautifulsoup4
- hnswlib
- numpy
- tqdm
- threading (built-in)

### Environment Variables
- `BASE_TMP_DIR`: Directory for data storage (default: "tmp")
- `OLLAMA_API_URL`: Ollama API endpoint (default: "http://localhost:11434/api/generate")
- `OLLAMA_EMBED_API_URL`: Ollama embeddings endpoint (default: "http://localhost:11434/api/embeddings")
- `OLLAMA_MODEL`: Model for text generation (default: "deepseek-r1:1.5b")
- `OLLAMA_EMBED_MODEL`: Model for embeddings (default: "nomic-embed-text:v1.5")

## Architecture Overview

This is a professional forum thread analysis application that scrapes, processes, and enables intelligent Q&A on multi-page forum discussions. The code has been optimized for performance, robustness, and maintainability.

### Core Components

1. **Enhanced Web Scraping Module** (`app.py:227-296`)
   - `detect_last_page()`: Robust pagination detection for various forum formats
   - `fetch_forum_pages()`: Downloads all pages with proper error handling
   - Multiple CSS selectors with fallback mechanisms
   - Proper URL validation and normalization

2. **Advanced Data Processing Pipeline** (`app.py:626-1133`)
   - `preprocess_thread()`: Completely rewritten for robustness and position tracking
   - **CRITICAL FIX**: Now properly tracks post positions to enable "second post on first page" queries
   - Enhanced metadata with `page_position`, `global_position`, and `file_index`
   - Thread-safe operations with locks
   - Comprehensive error handling and recovery
   - Atomic file operations for data consistency

3. **Professional Vector Search System** (`app.py:320-451, 1965-2124`)
   - Thread-safe embedding cache with atomic writes and validation
   - Enhanced HNSW index with proper index-to-post mapping
   - Improved HyDE (Hypothetical Document Embeddings) implementation
   - LLM reranking with batch processing and timeout handling
   - Performance monitoring and detailed logging

4. **Intelligent Query Processing** (`app.py:1515-1962`)
   - Enhanced `IntelligentQueryProcessor` with better pattern matching
   - **FIXED**: Specific post retrieval now works correctly with position metadata
   - Supports complex queries like "second post on first page"
   - Author-based searches with fuzzy matching
   - Temporal searches with flexible date matching
   - Metadata searches with range queries

5. **Professional Flask Web Interface** (`app.py:2135-2542`)
   - Enhanced error handling with detailed status reporting
   - Performance monitoring and timing statistics
   - Input validation and sanitization
   - Graceful degradation and user-friendly error messages
   - Streaming responses with proper error recovery

### Enhanced Data Storage Structure

```
tmp/
├── embeddings_cache.pkl           # Thread-safe global embedding cache
├── app.log                        # Comprehensive application logs
└── [thread-key]/                  # Per-thread directory
    ├── page1.html, page2.html...  # Downloaded forum pages (sorted)
    ├── posts/                     # Individual post JSON files with enhanced metadata
    │   ├── 0.json, 1.json...      # Files use file_index for consistent mapping
    ├── index_hnsw.bin             # HNSW vector index
    ├── index_meta.pkl             # Enhanced index metadata with mapping
    ├── metadata_index.json        # Post metadata with position tracking
    └── post_mapping.json          # NEW: Maps global positions to file indices
```

### Key Features and Improvements

- **FIXED: Position-Based Queries**: Now correctly handles "second post", "first post on page 2", etc.
- **Thread Safety**: All file operations and caching use proper locking mechanisms
- **Professional Error Handling**: Comprehensive error recovery and user feedback
- **Performance Monitoring**: Detailed timing and statistics for all operations
- **Enhanced Metadata**: Rich post metadata enables complex query types
- **Robust Caching**: Thread-safe embedding cache with validation and cleanup
- **Professional Logging**: Structured logging with appropriate levels and context

### Critical Bug Fixes

1. **Post Position Tracking**: Fixed the core issue where "second post on first page" queries failed
   - Added `page_position`, `global_position`, and `file_index` to all posts
   - Created `post_mapping.json` for O(1) position lookups
   - Fixed index-to-file mapping inconsistencies

2. **Thread Safety**: Fixed race conditions in embedding cache operations
   - Added `_cache_lock` and `_file_lock` for synchronized operations
   - Atomic file writes with temporary files and moves
   - Proper cache validation and cleanup

3. **Data Consistency**: Fixed metadata/post index misalignment
   - Ensured all data structures maintain consistent indexing
   - Added validation checks throughout the pipeline
   - Proper error handling when posts are skipped

### Performance Optimizations

- **Enhanced Concurrency**: Thread-safe embedding generation with controlled workers
- **Improved Caching**: Validated cache entries with cleanup of invalid data
- **Optimized Search**: Enhanced HNSW parameters and proper index mapping
- **Memory Management**: Explicit cleanup of large arrays and proper garbage collection
- **Batch Operations**: Efficient batch reranking and atomic file operations

### Development Notes

- The code now includes comprehensive professional comments
- All functions have detailed docstrings with Args, Returns, and Raises
- Error handling follows best practices with specific exception types
- Performance monitoring is built into all major operations
- The codebase is maintainable and follows professional Python standards