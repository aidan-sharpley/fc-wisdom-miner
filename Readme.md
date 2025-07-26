# Intelligent Forum Analysis & Search Engine

This is a high-performance forum analysis application that uses local LLMs to provide accurate, data-driven insights and powerful semantic search for any forum thread.

Designed to run efficiently on consumer hardware (8GB RAM), this Flask-based tool scrapes entire forum threads, processes and enriches the content, and builds a sophisticated local search engine with **auto-generated thread narratives**. It features a unique dual-engine query system that can distinguish between:

- **Analytical questions** (e.g., "Who is the most active user?", "Who created this thread?") and
- **Semantic questions** (e.g., "What are the best heating techniques?").

The system provides data-driven answers for analytical queries and context-aware responses for semantic queries. **Thread narratives are automatically generated and displayed when threads are loaded**, providing immediate comprehensive insights. Results are generated automatically and users should verify important information.

## ✨ Key Features

- 📖 **Auto-Generated Thread Narratives**: Comprehensive thread summaries with conversation phases, key contributors, and topic evolution - displayed automatically when threads are loaded.
- 🎯 **Dual-Engine Query System**: Provides analytical (data-driven) and semantic (LLM-based) query processing with automatic routing.
- 👤 **Thread Author Identification**: Metadata-grounded thread creator detection using URL parsing with high accuracy.
- 🔗 **Clickable Post Links**: Provides direct links to the specific source posts for all analytical results, ensuring full traceability.
- 🧠 **Local LLM Powered**: Uses Ollama with deepseek-r1:1.5b and nomic-embed-text models for privacy and performance.
- 🔄 **Smart Reprocessing**: Re-analyze existing threads without re-downloading using saved HTML.
- 📊 **Advanced Analytics**: Thread summaries, participant analysis, and engagement metrics.
- ⚡ **M1 Optimized**: Performance-optimized for M1 MacBook Air with 8GB RAM, aggressive caching, and memory-efficient processing.
- 🛡️ **Security First**: Input validation, SSRF protection, and data sanitization.
- 🌐 **Multi-Platform**: Supports XenForo, vBulletin, phpBB, and generic forums.
- 🤝 **Respectful Scraping**: 1.5-3.5s delays with jitter, User-Agent rotation, exponential backoff.
- 🚀 **Performance Monitoring**: Real-time analytics and bottleneck detection.

## 🏗️ Architecture

### Core Components

- **Thread Narrative Generator**: Optimized narrative generation with intelligent phase detection and aggressive caching
- **Forum Scraper**: Respectful scraping with jitter, comprehensive page detection and HTML preservation
- **Query Router**: LLM-powered intelligent query classification and routing (fully generic, no hardcoded terms)
- **Data Analyzer**: Statistical analysis with post links and thread authorship detection
- **Embedding System**: Domain-optimized vector search with HyDE enhancement
- **Platform Manager**: Dynamic configuration for different forum platforms
- **Security Layer**: Input validation and SSRF protection

## 🚀 Quick Start

### Prerequisites

1. **Ollama** running locally with required models:
   ```bash
   ollama pull deepseek-r1:1.5b
   ollama pull nomic-embed-text:v1.5
   ```

2. **Python 3.9+** with uv (recommended):
   ```bash
   pip install uv
   uv pip install Flask requests beautifulsoup4 hnswlib numpy tqdm
   ```

### Running the Application

```bash
uv run python app.py
```

Visit http://localhost:8080 to access the web interface.

## 💡 Usage Examples

### New Thread Analysis
1. Enter a forum thread URL
2. System scrapes all pages and builds search index
3. Query with natural language questions

### Thread Narratives
- **Auto-Generated**: Thread narratives appear automatically when selecting threads
- **Conversation Phases**: Intelligent grouping of discussion phases with topic detection
- **Key Contributors**: Top participants with engagement metrics and activity patterns
- **Topic Evolution**: How discussions evolved across different phases

### Query Examples
- **Analytical**: "Who is the most active user?" → Data-driven response with post counts
- **Thread Authorship**: "Who created this thread?" → Metadata-based creator identification
- **Positional**: "Who was the second user to post?" → Chronological analysis with links
- **Semantic**: "What are the best heating techniques?" → Vector search + LLM analysis
- **Technical**: "What materials and settings work best?" → Generic technical specification extraction

## 🔧 Configuration

### Environment Variables
```bash
BASE_TMP_DIR=tmp                           # Data storage directory
OLLAMA_BASE_URL=http://localhost:11434     # Ollama server URL
OLLAMA_CHAT_MODEL=deepseek-r1:1.5b        # Text generation model
OLLAMA_EMBED_MODEL=nomic-embed-text:v1.5  # Embedding model
SECRET_KEY=your-secret-key                 # Flask security key
```

### Platform Support
Supports major forum platforms with YAML-based configuration:
- XenForo (most common)
- vBulletin 
- phpBB
- Generic/custom forums

## 📈 Performance

**M1 MacBook Air Optimized** for 8GB RAM systems:
- **Memory**: <2GB typical usage with aggressive garbage collection
- **Narrative Generation**: ~15 phases (reduced from 300) with prompt caching
- **Processing**: Sequential processing with memory monitoring
- **Storage**: Intelligent caching with content-based hashing
- **Threading**: 3-worker optimization for memory efficiency
- **Progress Tracking**: Real-time progress bars for all major operations

## 🛡️ Security Features

- Input validation and sanitization
- SSRF protection for URL requests
- Secure thread key generation
- Rate limiting and request throttling
- Comprehensive error handling with logging

## ⚠️ Disclaimer

This tool generates automated responses based on forum content analysis. Users should verify important information independently. Results are for informational purposes and should not be considered authoritative without proper verification.
