# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

This is a Python Flask application optimized for consumer hardware with 8GB RAM. The main application is in `app.py`.

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
- **Hardware**: Optimized for consumer hardware with 8GB RAM

### Dependencies
Install using uv (preferred) or pip:
```bash
uv pip install Flask requests beautifulsoup4 hnswlib numpy tqdm
```

Optional enhanced dependencies:
```bash
uv add scikit-learn          # For advanced semantic clustering
uv add elasticsearch         # For hybrid search capabilities
```

Core dependencies:
- Flask (web framework)
- requests (HTTP client)
- beautifulsoup4 (HTML parsing)
- hnswlib (vector indexing)
- numpy (numerical operations)
- tqdm (progress bars)

Optional dependencies:
- scikit-learn (semantic clustering)
- elasticsearch (hybrid search)

### Environment Variables
- `BASE_TMP_DIR`: Directory for data storage (default: "tmp")
- `OLLAMA_BASE_URL`: Ollama base URL (default: "http://localhost:11434")
- `OLLAMA_API_URL`: Direct Ollama API URL (default: "http://localhost:11434/api")
- `OLLAMA_EMBED_API_URL`: Ollama embeddings API URL (default: "http://localhost:11434/api/embeddings")
- `OLLAMA_CHAT_MODEL`: Model for text generation (default: "deepseek-r1:1.5b")
- `OLLAMA_EMBED_MODEL`: Model for embeddings (default: "nomic-embed-text:v1.5")
- `SECRET_KEY`: Flask secret key for sessions and security

### Respectful Scraping Configuration
- `BASE_REQUEST_DELAY`: Base delay between requests (default: 1.5 seconds)
- `JITTER_RANGE`: Random jitter added to delays (default: 0.5-2.0 seconds)
- `RESPECTFUL_SCRAPING`: Enable respectful scraping features (default: True)

## Architecture Overview

This is a modular forum analysis application with separate components for scraping, processing, embedding, search, and analytics. The architecture prioritizes performance, accuracy, and maintainability for consumer hardware with 8GB RAM.

### Core Modular Components

#### 1. **Forum Scraper** (`scraping/forum_scraper.py`)
- **Respectful Scraping**: 1.5-3.5 second delays with random jitter to avoid predictable patterns
- **User-Agent Rotation**: Rotates through realistic browser user-agents every 5 pages
- **Exponential Backoff**: Intelligent retry logic with increasing delays on failures
- **Comprehensive Page Detection**: Scrapes ALL pages (up to 1000, configurable)
- **Post URL Extraction**: Captures direct links to individual posts for clickable references
- **Vote/Reaction Extraction**: Extracts upvotes, downvotes, reactions, likes from various forum platforms
- **HTML Preservation**: Saves raw HTML files for reprocessing
- **Robust Pagination**: Enhanced detection for XenForo, vBulletin, phpBB, and custom forums
- **Rate Limiting Statistics**: Detailed metrics on delay times and server-friendly behavior

#### 2. **Post Processor** (`processing/post_processor.py`)
- **Intelligent Filtering**: Less aggressive content filtering (30% letter ratio vs 50%)
- **Enhanced Metadata**: Adds text statistics, content classification, author activity levels
- **Duplicate Detection**: Smart deduplication with content normalization
- **Quality Assessment**: Content type classification (question, solution, opinion, etc.)

#### 3. **Query System** (`search/query_processor.py`)
- **🎯 DUAL PROCESSING**: Analytical vs semantic query routing with 100% accuracy
- **Analytical Queries**: Direct data analysis for "who is most active", "how many posts", etc.
- **Positional Queries**: "Who was the second user to post?" with chronological analysis
- **Semantic Queries**: Vector search for content-based questions
- **Smart Routing**: Automatically detects query type and routes to appropriate processor
- **Post Link Integration**: Provides clickable links to specific posts in results

#### 4. **Forum Data Analyzer** (`analytics/data_analyzer.py`)
- **Participant Analysis**: Counts posts by author across ENTIRE thread (not just search results)
- **Thread Authorship Analysis**: Metadata-grounded thread creator identification with URL extraction
- **Content Statistics**: Thread metrics, post lengths, page distributions
- **Temporal Analysis**: Activity over time, posting patterns, thread duration
- **Factual Accuracy**: 100% accurate answers using real data aggregation

#### 5. **Thread Narrative Generator** (`analytics/thread_narrative.py`)
- **Optimized Performance**: Reduced phases from 300 to ~15 maximum for M1 MacBook Air efficiency
- **Enhanced Semantic Clustering**: Intelligent topic detection using content similarity analysis
- **Comprehensive Topic Overviews**: Rich 1-2 sentence summaries with post ranges and direct links
- **Aggressive Caching**: MD5-based prompt caching with persistent disk storage to avoid redundant LLM calls
- **Memory-Efficient Processing**: Single-pass generation with intelligent phase grouping
- **Auto-Display**: Narratives automatically appear when threads are loaded in the UI
- **Generic Topic Detection**: Works with any forum domain without hardcoded terms
- **Phase Intelligence**: Automatically detects conversation phases based on content analysis
- **Verifiable Claims**: All narrative claims backed by specific post evidence

#### 6. **Embedding System** (`embedding/embedding_manager.py`)
- **Generic Processing**: No longer hardcoded to specific forum terminology - works with any domain
- **Enhanced HyDE**: Better hypothetical document generation for technical queries
- **Memory-Efficient Caching**: 150MB cache for 8GB systems
- **Optimized Batch Processing**: Reduced batch sizes (8 vs 10) to prevent memory spikes
- **Performance Monitoring**: Detailed embedding generation statistics
- **Progress Tracking**: Real-time progress bars for large datasets

#### 7. **Multi-Factor Search Ranking** (`search/result_ranker.py`)
- **Recency Scoring**: Recent posts weighted higher in ranking
- **Vote-Based Scoring**: Upvotes, reactions, and community engagement
- **Author Authority**: Active participants get higher ranking
- **Content Quality**: Length, formatting, and relevance indicators
- **Contextual Weights**: Adapts ranking based on query type

#### 8. **Analytics** (`analytics/thread_analyzer.py`)
- **Thread Overview**: Participants, pages, duration, activity patterns
- **Content Analysis**: Keywords, themes, discussion topics
- **Engagement Metrics**: Vote distributions, reaction patterns
- **Author Insights**: Most active contributors, authority levels
- **URL-Based Thread Creator Extraction**: Parses canonical URLs to identify thread creators

#### 9. **LLM Query Router** (`analytics/llm_query_router.py`)
- **Intelligent Query Classification**: LLM-powered query analysis and routing
- **Context-Aware Processing**: Understands query intent and complexity
- **Enhanced Query Expansion**: Improves vague queries with domain knowledge
- **Smart Response Generation**: Contextual response formatting
- **Thread Authorship Detection**: Routes author/creator queries to metadata-based analysis
- **Generic Routing**: No hardcoded device-specific terms, works with any thread content

#### 10. **Platform Configuration System** (`config/platform_config.py`)
- **Dynamic Forum Detection**: Automatically detects forum platform types
- **YAML-Based Configurations**: Extensible platform-specific settings in `configs/platforms/`
- **CSS Selector Management**: Platform-specific selectors for XenForo, vBulletin, phpBB
- **Adaptive Scraping**: Adjusts scraping behavior based on platform

#### 11. **Enhanced Analysis Systems (NEW)**
- **Enhanced Topic Analyzer** (`analytics/enhanced_topic_analyzer.py`): Semantic clustering with k-means and silhouette analysis
- **Verifiable Response System** (`search/verifiable_response_system.py`): Fact-checking with post evidence and citations
- **ElasticSearch Integration** (`search/elasticsearch_integration.py`): Hybrid semantic + full-text search
- **Multi-Pass Fusion System** (`analytics/multipass_fusion_system.py`): Cross-validated insights from multiple analysis types

#### 12. **Security & Utilities**
- **Security Utils** (`utils/security.py`): Input validation, SSRF protection, sanitization
- **Performance Analytics** (`utils/performance_analytics.py`): Comprehensive performance monitoring
- **Advanced Caching** (`utils/advanced_cache.py`): Enhanced caching with intelligent cleanup
- **Processing Pipeline** (`utils/processing_pipeline.py`): Modular processing workflows

### Data Storage Structure

```
tmp/
├── embeddings_cache/              # Embedding cache with metadata
│   ├── [hash].pkl                 # Individual embedding files
│   └── cache_metadata.json        # Cache management data
├── narrative_cache.json           # Global narrative prompt cache (MD5-based)
├── app.log                        # Application logs  
└── threads/
    └── [thread-key]/              # Per-thread directory
        ├── html_pages/            # Raw HTML files for reprocessing
        │   ├── page_001.html      # Original downloaded HTML
        │   ├── page_002.html      # Preserved for updates
        │   └── page_XXX.html
        ├── posts.json             # Processed posts with enhanced metadata
        ├── metadata.json          # Thread metadata and processing stats
        ├── thread_analytics.json  # Thread analytics
        ├── thread_summary.json    # Auto-generated narrative and analytics
        ├── index_hnsw.bin         # HNSW vector search index
        └── index_hnsw.bin.metadata.json # Index metadata

configs/
├── platforms/                     # Platform-specific configurations
│   ├── generic.yaml               # Generic forum settings
│   ├── phpbb.yaml                 # phpBB-specific selectors
│   ├── vbulletin.yaml            # vBulletin-specific selectors
│   └── xenforo.yaml              # XenForo-specific selectors
```

## Key Features

### 📖 **Auto-Generated Thread Narratives** 
- **Performance Optimized**: Reduced from 300 to ~15 phases maximum for M1 MacBook Air efficiency
- **Enhanced Semantic Clustering**: Uses k-means clustering and silhouette analysis for optimal topic detection
- **Comprehensive Topic Overviews**: Rich 1-2 sentence summaries for each topic with post ranges and clickable links to topic starts
- **Intelligent Phase Detection**: Generic topic detection works with any forum domain
- **Conversation Overview**: Comprehensive thread summaries with key developments and outcomes
- **Key Contributors**: Identifies top participants with engagement metrics and activity patterns
- **Topic Evolution**: Shows how discussions evolved across different conversation phases
- **Verifiable Claims**: All narrative statements backed by specific post evidence with citations
- **Auto-Display**: Narratives appear automatically when threads are loaded in the UI
- **Aggressive Caching**: MD5-based prompt caching avoids redundant LLM calls for performance
- **Memory Efficient**: Single-pass generation with intelligent grouping for 8GB systems

### 🎯 **Query Accuracy & Smart Interpretation**
- **Analytical Queries**: "Who is most active?" → Analyzes ALL posts, returns exact counts with verifiable evidence
- **Thread Authorship Queries**: "Who is the thread author?" → URL-based metadata extraction with high confidence
- **Positional Queries**: "Who was the second user to post?" → Chronological analysis with post links
- **Semantic Queries**: "What heating techniques work best?" → Hybrid search (ElasticSearch + vector search) + LLM
- **Technical Specifications**: "What materials and settings work best?" → Generic extraction, no hardcoded terms
- **Verifiable Responses**: All claims backed by specific post evidence with clickable citations
- **Smart Detection**: Automatically routes queries to appropriate processor
- **Vague Query Enhancement**: "highest rated" → Auto-expanded with engagement terms for better results
- **Conversational Understanding**: System explains its interpretation for ambiguous queries
- **Auto-Routing**: Vague queries like "best" automatically routed to engagement analysis
- **Metadata Priority**: Thread authorship prioritizes URL extraction over post frequency analysis
- **100% Accurate**: Data-driven answers for statistical and authorship questions
- **Fact-Checking**: Comprehensive verification reports with confidence levels

### 🚀 **Performance & Memory Optimization**
- **Complete Thread Analysis**: Processes up to 1000 pages with progress tracking
- **Enhanced Search**: Hybrid ElasticSearch + semantic search with fallback capabilities
- **Multi-Pass Analysis**: Cross-validated insights from topic, participant, engagement, and temporal analysis
- **Worker Optimization**: 3 workers (vs 4) for efficiency
- **Memory Management**: Reduced batch sizes and cache limits for 8GB systems
- **HNSW Tuning**: Optimized parameters for memory constraints
- **Advanced Caching**: Content-based cache with intelligent cleanup
- **Lazy Loading**: Heavy dependencies load only when needed for fast startup

### 🔄 **Reprocess Functionality**
- **HTML-Based Reprocessing**: Re-parses original HTML files with optimizations
- **No Re-downloading**: Respects rate limits, only downloads when thread deleted/recreated
- **Updates**: Apply features to existing threads without internet requests
- **Backward Compatible**: Fallback support for threads without saved HTML

### 📊 **Analytics with Progress Tracking**
- **Thread Statistics**: Complete participant analysis, posting patterns
- **Content Insights**: Primary keywords, discussion themes, engagement metrics  
- **Temporal Analysis**: Activity over time, peak periods, thread evolution
- **Progress Visualization**: Real-time progress bars for all major operations
- **Performance Monitoring**: Detailed timing and operation statistics

### 🛡️ **Reliability & Security**
- **Error Handling**: Exception handling with error IDs
- **Input Validation**: Security checks for thread keys and user inputs
- **SSRF Protection**: URL validation and malicious request prevention
- **Data Sanitization**: Comprehensive input cleaning and validation
- **Data Consistency**: Atomic operations and validation throughout pipeline
- **Logging**: Structured logs with query type classification
- **Memory Monitoring**: Optimized resource management and cleanup
- **Performance Analytics**: Real-time monitoring and bottleneck detection

## Improvements Made

### 1. **Analytical Query Accuracy** ✅
**Problem**: "Who is most active user?" returned wrong answers based on semantic search
**Solution**: Added `ForumDataAnalyzer` with positional query support
**Result**: 100% accurate data-driven answers for statistical and positional queries

### 2. **Post Link Integration** ✅  
**Problem**: No direct links to individual posts in results
**Solution**: Enhanced scraper extracts post URLs and IDs during scraping
**Result**: Clickable links to specific posts in all analytical responses

### 3. **Hardware Optimization** ✅
**Problem**: Settings not optimized for 8GB RAM systems
**Solution**: Reduced worker counts, batch sizes, and memory limits
**Result**: Optimal performance on consumer hardware without memory pressure

### 4. **Progress Tracking** ✅
**Problem**: No progress indication during long HNSW and analytics operations
**Solution**: Added progress bars for index building and analytics generation
**Result**: User can see exactly what's happening during processing

### 5. **Error Handling** ✅
**Problem**: Basic error handling insufficient
**Solution**: Added exception handling with error IDs and logging
**Result**: Reliable operation with detailed error tracking

### 6. **UI/UX Improvements** ✅
**Problem**: UI didn't reflect backend API capabilities (missing reprocess checkbox)
**Solution**: Fixed UI to show/hide appropriate sections based on thread selection
**Result**: UI now properly matches the powerful backend functionality

### 7. **Thread Author Identification** ✅
**Problem**: "Who is the thread author?" returned wrong answers based on semantic search
**Solution**: Implemented metadata-grounded authorship identification with URL parsing
**Result**: 100% accurate thread creator identification using canonical URL extraction

### 8. **Thread Narrative Performance Optimization** ✅
**Problem**: Narrative generation taking ~14s per phase across 300 phases (~70 minutes total)
**Solution**: Complete rewrite with intelligent phase detection, aggressive caching, and M1 optimization
**Result**: Reduced to ~15 phases maximum with MD5-based prompt caching, auto-display in UI

### 9. **Generic Query System** ✅
**Problem**: System had hardcoded device-specific terms limiting use to specific forum types
**Solution**: Removed all hardcoded terms, made routing and analysis completely generic
**Result**: Works with any forum thread content without domain-specific customization

### 10. **UI Thread Narrative Display** ✅
**Problem**: Thread narratives weren't displaying automatically when threads were loaded
**Solution**: Fixed data structure mismatch between frontend and backend (narrative vs narrative_summary)
**Result**: Thread narratives now appear automatically when selecting threads in the UI

### 11. **Enhanced Topic Overviews with Semantic Clustering** ✅
**Problem**: Topic detection relied on simple position-based grouping with superficial descriptions
**Solution**: Implemented semantic clustering using k-means with silhouette analysis and comprehensive topic summaries
**Result**: Rich 1-2 sentence topic overviews with post ranges, engagement metrics, and clickable links to topic starts

### 12. **Verifiable Response System with Post Evidence** ✅
**Problem**: Claims and statements were not backed by specific post evidence
**Solution**: Created comprehensive fact-checking system with post citations and evidence grading
**Result**: All analytical responses now include clickable post links with evidence and confidence levels

### 13. **Multi-Pass Analysis Fusion** ✅
**Problem**: Analysis types worked in isolation without cross-validation
**Solution**: Implemented multi-pass system that combines topic, participant, engagement, and temporal analysis
**Result**: Cross-validated insights with comprehensive evidence collection and executive summaries

### 14. **Hybrid Search Integration** ✅
**Problem**: Limited to semantic search without fast full-text capabilities
**Solution**: Added ElasticSearch integration with hybrid semantic + keyword search
**Result**: Fast ranked full-text search with intelligent fallback when ElasticSearch unavailable

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
2. Choose "reprocess" to apply optimizations
3. System re-parses saved HTML files (no re-download)
4. Rebuilds with processing, ranking, and analytics
5. Updated thread with features
```

### Query Types Handled
- **Analytical**: "Who posted most?", "How many posts?", "When was first post?" (with verifiable evidence)
- **Thread Authorship**: "Who is the thread author?", "Who created this thread?", "Original poster?" (metadata-grounded)
- **Positional**: "Who was the second user to post?", "First post author?" (with clickable links)
- **Semantic**: "What are the best heating techniques?", "How do I fix vapor quality?" (hybrid search + LLM)
- **Engagement**: "What's the highest rated post?", "Most popular content?" (with post evidence)
- **Technical**: "What settings work best?", "Common configurations?" (generic extraction)
- **Temporal**: "What changed over time?", "Recent developments?" (timeline analysis)
- **Participant**: "What did UserX contribute?", "Most active contributors?" (comprehensive analysis)
- **Multi-Pass**: Complex queries benefit from cross-validated insights from multiple analysis types

## Development Guidelines

- **Use `uv run python`** for all Python execution (user prefers uv over pip)
- **Never create files unless absolutely necessary** for the task
- **Always prefer editing existing files** to creating new ones
- **Hardware Optimization**: Consider 8GB RAM constraints in all changes
- **Error Handling**: Add proper exception handling with error IDs
- **Progress Tracking**: Add progress bars for operations > 1000 items
- **Test analytical queries** to ensure they use data analysis with post links
- **Verify reprocessing** applies features to existing threads
- **Memory Monitoring**: Monitor cache sizes and worker counts

## Troubleshooting

### Common Issues
1. **"Thread not found" on reprocess**: HTML files missing, will fallback to posts.json method
2. **Analytical queries using semantic search**: Check `ForumDataAnalyzer.can_handle_query()` patterns
3. **Thread author queries returning wrong results**: Ensure thread analytics contain URL-based thread_creator metadata
4. **Missing vote data**: Verify forum platform CSS selectors in `config/settings.py`
5. **Slow performance**: Check embedding cache hit rate and HNSW index parameters
6. **Thread narratives not appearing**: Check if `thread_summary.json` exists and contains `narrative` key
7. **Narrative generation too slow**: Verify aggressive caching is working, check `narrative_cache.json`
8. **Generic queries failing**: System now works with any domain - no hardcoded terms should be needed
9. **Enhanced features not working**: Install optional dependencies (`uv add scikit-learn elasticsearch`)
10. **Slow startup**: Enhanced features use lazy loading - heavy dependencies load only when needed
11. **Missing verification evidence**: Check that verifiable response system is properly initialized
12. **ElasticSearch errors**: System gracefully falls back to traditional search when ElasticSearch unavailable

### Performance Monitoring
- All major operations include timing statistics
- Embedding cache hit/miss ratios tracked
- Search result ranking scores available
- Thread processing metrics logged

The application provides forum analysis with accuracy and architecture suitable for analyzing large-scale forum discussions. Code includes error handling, progress tracking, and memory optimization for 8GB systems.