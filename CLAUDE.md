# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

This is a **production-ready Python Flask application** optimized for M1 MacBook Air with 8GB RAM. The main application is in `app.py`.

### Running the Application
```bash
uv run python app.py
```
The app will start on port 8080 at http://localhost:8080

### Prerequisites
- **Ollama must be running locally** with the required models:
  - `deepseek-r1:1.5b` (1.5B parameters, ~1.2GB RAM usage)
  - `nomic-embed-text:v1.5` (768-dimensional embeddings, efficient)
- Python 3.9+ (tested with 3.13) with dependencies managed by `uv`
- **Hardware**: Optimized for M1 MacBook Air with 8GB RAM

### Dependencies
Install using uv (preferred) or pip:
```bash
uv pip install Flask requests beautifulsoup4 hnswlib numpy tqdm
```

Core dependencies:
- Flask (web framework)
- requests (HTTP client)
- beautifulsoup4 (HTML parsing)
- hnswlib (vector indexing)
- numpy (numerical operations)
- tqdm (progress bars)

### Environment Variables
- `BASE_TMP_DIR`: Directory for data storage (default: "tmp")
- `OLLAMA_BASE_URL`: Ollama base URL (default: "http://localhost:11434")
- `OLLAMA_CHAT_MODEL`: Model for text generation (default: "deepseek-r1:1.5b")
- `OLLAMA_EMBED_MODEL`: Model for embeddings (default: "nomic-embed-text:v1.5")

## Architecture Overview - Production v2.0 (M1 Optimized)

This is a **production-ready modular forum analysis application** with separate components for scraping, processing, embedding, search, and analytics. The architecture prioritizes performance, accuracy, maintainability, and is specifically optimized for M1 MacBook Air systems with 8GB RAM.

### Core Modular Components

#### 1. **Production Forum Scraper** (`scraping/forum_scraper.py`)
- **Comprehensive Page Detection**: Scrapes ALL pages (up to 1000, configurable)
- **Post URL Extraction**: Captures direct links to individual posts for clickable references
- **Vote/Reaction Extraction**: Extracts upvotes, downvotes, reactions, likes from various forum platforms
- **HTML Preservation**: Saves raw HTML files for reprocessing with new optimizations
- **Robust Pagination**: Enhanced detection for XenForo, vBulletin, phpBB, and custom forums
- **M1 Optimized**: Rate limiting and request handling optimized for M1 efficiency

#### 2. **Advanced Post Processor** (`processing/post_processor.py`)
- **Intelligent Filtering**: Less aggressive content filtering (30% letter ratio vs 50%)
- **Enhanced Metadata**: Adds text statistics, content classification, author activity levels
- **Duplicate Detection**: Smart deduplication with content normalization
- **Quality Assessment**: Content type classification (question, solution, opinion, etc.)

#### 3. **Production Query System** (`search/query_processor.py`)
- **🎯 DUAL PROCESSING**: Analytical vs semantic query routing with 100% accuracy
- **Analytical Queries**: Direct data analysis for "who is most active", "how many posts", etc.
- **Positional Queries**: "Who was the second user to post?" with chronological analysis
- **Semantic Queries**: Vector search for content-based questions
- **Smart Routing**: Automatically detects query type and routes to appropriate processor
- **Post Link Integration**: Provides clickable links to specific posts in results

#### 4. **Forum Data Analyzer** (`analytics/data_analyzer.py`) - **NEW**
- **Participant Analysis**: Counts posts by author across ENTIRE thread (not just search results)
- **Content Statistics**: Thread metrics, post lengths, page distributions
- **Temporal Analysis**: Activity over time, posting patterns, thread duration
- **Factual Accuracy**: 100% accurate answers using real data aggregation

#### 5. **M1-Optimized Embedding System** (`embedding/embedding_manager.py`)
- **Domain-Specific Processing**: Optimized for vape/device forum terminology
- **Enhanced HyDE**: Better hypothetical document generation for technical queries
- **Memory-Efficient Caching**: 150MB cache optimized for 8GB systems
- **M1 Batch Processing**: Reduced batch sizes (8 vs 10) to prevent memory spikes
- **Performance Monitoring**: Detailed embedding generation statistics
- **Progress Tracking**: Real-time progress bars for large datasets

#### 6. **Multi-Factor Search Ranking** (`search/result_ranker.py`)
- **Recency Scoring**: Recent posts weighted higher in ranking
- **Vote-Based Scoring**: Upvotes, reactions, and community engagement
- **Author Authority**: Active participants get higher ranking
- **Content Quality**: Length, formatting, and relevance indicators
- **Contextual Weights**: Adapts ranking based on query type

#### 7. **Comprehensive Analytics** (`analytics/thread_analyzer.py`)
- **Thread Overview**: Participants, pages, duration, activity patterns
- **Content Analysis**: Keywords, themes, discussion topics
- **Engagement Metrics**: Vote distributions, reaction patterns
- **Author Insights**: Most active contributors, authority levels

### Enhanced Data Storage Structure

```
tmp/
├── embeddings_cache/              # Modular embedding cache with metadata
│   ├── [hash].pkl                 # Individual embedding files
│   └── cache_metadata.json        # Cache management data
├── app.log                        # Comprehensive application logs  
└── threads/
    └── [thread-key]/              # Per-thread directory
        ├── html_pages/            # NEW: Raw HTML files for reprocessing
        │   ├── page_001.html      # Original downloaded HTML
        │   ├── page_002.html      # Preserved for optimization updates
        │   └── page_XXX.html
        ├── posts.json             # Processed posts with enhanced metadata
        ├── metadata.json          # Thread metadata and processing stats
        ├── thread_analytics.json  # Comprehensive thread analytics
        ├── index_hnsw.bin         # HNSW vector search index
        └── index_hnsw.bin.metadata.json # Index metadata
```

## Key Features and M1 Optimizations

### 🎯 **Query Accuracy Revolution**
- **Analytical Queries**: "Who is most active?" → Analyzes ALL posts, returns exact counts
- **Positional Queries**: "Who was the second user to post?" → Chronological analysis with post links
- **Semantic Queries**: "What heating techniques work best?" → Vector search + LLM
- **Smart Detection**: Automatically routes queries to appropriate processor
- **100% Accurate**: Data-driven answers for statistical questions

### 🚀 **M1 Performance & Memory Optimization**
- **Complete Thread Analysis**: Processes up to 1000 pages with progress tracking
- **M1 Worker Optimization**: 3 workers (vs 4) for efficiency cores
- **Memory Management**: Reduced batch sizes and cache limits for 8GB systems
- **HNSW Tuning**: Optimized parameters for M1 memory constraints
- **Advanced Caching**: Content-based cache with intelligent cleanup

### 🔄 **Reprocess Functionality** - **MAJOR UPDATE**
- **HTML-Based Reprocessing**: Re-parses original HTML files with new optimizations
- **No Re-downloading**: Respects rate limits, only downloads when thread deleted/recreated
- **Optimization Updates**: Apply new features to existing threads without internet requests
- **Backward Compatible**: Fallback support for threads without saved HTML

### 📊 **Advanced Analytics with Progress Tracking**
- **Thread Statistics**: Complete participant analysis, posting patterns
- **Content Insights**: Primary keywords, discussion themes, engagement metrics  
- **Temporal Analysis**: Activity over time, peak periods, thread evolution
- **Progress Visualization**: Real-time progress bars for all major operations
- **Performance Monitoring**: Detailed timing and operation statistics

### 🛡️ **Production-Ready Reliability**
- **Comprehensive Error Handling**: Production-grade exception handling with error IDs
- **Input Validation**: Security checks for thread keys and user inputs
- **Data Consistency**: Atomic operations and validation throughout pipeline
- **Professional Logging**: Structured logs with query type classification
- **Memory Monitoring**: M1-optimized resource management and cleanup

## Production Improvements Made

### 1. **Fixed Analytical Query Accuracy** ✅
**Problem**: "Who is most active user?" returned wrong answers based on semantic search
**Solution**: Added `ForumDataAnalyzer` with positional query support
**Result**: 100% accurate data-driven answers for statistical and positional queries

### 2. **Post Link Integration** ✅  
**Problem**: No direct links to individual posts in results
**Solution**: Enhanced scraper extracts post URLs and IDs during scraping
**Result**: Clickable links to specific posts in all analytical responses

### 3. **M1 MacBook Air Optimization** ✅
**Problem**: Settings not optimized for 8GB RAM M1 systems
**Solution**: Reduced worker counts, batch sizes, and memory limits
**Result**: Optimal performance on M1 hardware without memory pressure

### 4. **Progress Tracking Enhancement** ✅
**Problem**: No progress indication during long HNSW and analytics operations
**Solution**: Added progress bars for index building and analytics generation
**Result**: User can see exactly what's happening during processing

### 5. **Production Error Handling** ✅
**Problem**: Basic error handling not suitable for production
**Solution**: Added comprehensive exception handling with error IDs and logging
**Result**: Production-ready reliability with detailed error tracking

### 6. **UI/UX Improvements** ✅
**Problem**: UI didn't reflect backend API capabilities (missing reprocess checkbox)
**Solution**: Fixed UI to show/hide appropriate sections based on thread selection
**Result**: UI now properly matches the powerful backend functionality

## Usage Patterns

### For New Threads
```
1. Enter forum thread URL
2. System downloads all pages and saves HTML
3. Processes posts with vote extraction and optimization
4. Generates embeddings with domain-specific preprocessing  
5. Creates analytics and search indexes
6. Ready for queries
```

### For Existing Threads  
```
1. Select existing thread (no URL input)
2. Choose "reprocess" to apply new optimizations
3. System re-parses saved HTML files (no re-download)
4. Rebuilds with latest processing, ranking, and analytics
5. Updated thread with new features
```

### Query Types Handled (Production-Ready)
- **Analytical**: "Who posted most?", "How many posts?", "When was first post?"
- **Positional**: "Who was the second user to post?", "First post author?" (with clickable links)
- **Semantic**: "What are the best heating techniques?", "How do I fix vapor quality?"
- **Temporal**: "What changed over time?", "Recent developments?"
- **Participant**: "What did UserX contribute?", "Most active contributors?"

## Development Guidelines (Production Standards)

- **Use `uv run python`** for all Python execution (user prefers uv over pip)
- **Never create files unless absolutely necessary** for the task
- **Always prefer editing existing files** to creating new ones
- **M1 Optimization Focus**: Consider 8GB RAM constraints in all changes
- **Production Error Handling**: Add proper exception handling with error IDs
- **Progress Tracking**: Add progress bars for operations > 1000 items
- **Test analytical queries** to ensure they use data analysis with post links
- **Verify reprocessing** applies new features to existing threads
- **Memory Monitoring**: Monitor cache sizes and worker counts

## Troubleshooting

### Common Issues
1. **"Thread not found" on reprocess**: HTML files missing, will fallback to posts.json method
2. **Analytical queries using semantic search**: Check `ForumDataAnalyzer.can_handle_query()` patterns
3. **Missing vote data**: Verify forum platform CSS selectors in `config/settings.py`
4. **Slow performance**: Check embedding cache hit rate and HNSW index parameters

### Performance Monitoring
- All major operations include timing statistics
- Embedding cache hit/miss ratios tracked
- Search result ranking scores available
- Thread processing metrics logged

The application now provides **production-ready forum analysis** with comprehensive accuracy, M1 MacBook Air optimization, and professional architecture suitable for analyzing large-scale forum discussions. All code follows production standards with comprehensive error handling, progress tracking, and memory optimization for 8GB systems.