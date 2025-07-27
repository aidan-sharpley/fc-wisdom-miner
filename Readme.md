# Intelligent Forum Analysis & Search Engine

This is a forum analysis application that uses local LLMs to provide "accurate", data-driven insights and semantic search for any forum thread.

Designed to run efficiently on consumer hardware (8GB RAM), this Flask-based tool scrapes entire forum threads, processes and enriches the content, and builds a sophisticated local search engine with **auto-generated thread narratives**. It features a unique dual-engine query system that can distinguish between:

- **Analytical questions** (e.g., "Who is the most active user?", "Who created this thread?") and
- **Semantic questions** (e.g., "What are the best heating techniques?").

The system provides **verifiable, fact-checked answers** with clickable post evidence for analytical queries and context-aware responses for semantic queries. **Thread narratives with enhanced topic overviews are automatically generated** using semantic clustering, providing immediate comprehensive insights with supporting evidence. All claims are backed by specific post links for full traceability.

## Foreward by human engineer
I was inspired to start this project when reading a public forum that had pretty minimal and tedious search capabilities. I wanted to explore what large language models can actually do—separate from the large-scale cloud services and interfaces built around tools like ChatGPT. As a test case, I focused on a real-world challenge: **how difficult it is to find specific, valuable info buried in long forum threads**.

I used chatgpt, claude, gemini, deepseek, and qwen models to engineer this mvp. This was a lot of fun and I'd say the experiment did return an answer. Check out the screenshots below to see the app in action! Any advice given by the app is purely for educaton / entertainment.

## ✨ Key Features

- 📖 **Enhanced Thread Narratives**: Semantic clustering with comprehensive 1-2 sentence topic overviews, post ranges, and clickable links to topic starts
- 🎯 **Dual-Engine Query System**: Provides analytical (data-driven) and semantic (LLM-based) query processing with automatic routing
- ✅ **Verifiable Responses**: All claims backed by specific post evidence with clickable citations and confidence levels
- 🔍 **Hybrid Search**: ElasticSearch integration with semantic search fallback for fast, ranked full-text search
- 🔄 **Multi-Pass Analysis**: Cross-validated insights from topic, participant, engagement, and temporal analysis
- 👤 **Thread Author Identification**: Metadata-grounded thread creator detection using URL parsing with high accuracy
- 🔗 **Clickable Post Links**: Provides direct links to the specific source posts for all analytical results, ensuring full traceability
- 🧠 **Local LLM Powered**: Uses Ollama with deepseek-r1:1.5b and nomic-embed-text models for privacy and performance
- 🔄 **Smart Reprocessing**: Re-analyze existing threads without re-downloading using saved HTML
- 📊 **Advanced Analytics**: Thread summaries, participant analysis, and engagement metrics
- ⚡ **M1 Optimized**: Performance-optimized for M1 MacBook Air with 8GB RAM, aggressive caching, and memory-efficient processing
- 🛡️ **Security First**: Input validation, SSRF protection, and data sanitization
- 🌐 **Multi-Platform**: Supports XenForo, vBulletin, phpBB, and generic forums
- 🤝 **Respectful Scraping**: 1.5-3.5s delays with jitter, User-Agent rotation, exponential backoff
- 🚀 **Performance Monitoring**: Real-time analytics and bottleneck detection

## 🏗️ Architecture

### Core Components

- **Enhanced Thread Narrative Generator**: Semantic clustering with comprehensive topic overviews and verifiable claims
- **Verifiable Response System**: Fact-checking with post evidence, citations, and confidence scoring
- **Hybrid Search Engine**: ElasticSearch + semantic search with intelligent fallback
- **Multi-Pass Fusion System**: Cross-validated insights from multiple analysis types
- **Forum Scraper**: Respectful scraping with jitter, comprehensive page detection and HTML preservation
- **Query Router**: LLM-powered intelligent query classification and routing (fully generic, no hardcoded terms)
- **Data Analyzer**: Statistical analysis with post links and thread authorship detection
- **Embedding System**: Domain-optimized vector search with HyDE enhancement
- **Platform Manager**: Dynamic configuration for different forum platforms
- **Security Layer**: Input validation and SSRF protection

## Screens!
Interactive Chat
<img width="969" height="605" alt="image" src="https://github.com/user-attachments/assets/509b2ad3-78c0-44ef-b0d1-b9b0fdd17d14" />

Thread Narratives
<img width="1044" height="489" alt="image" src="https://github.com/user-attachments/assets/002346b9-2427-4045-9ec6-9f433ae6cafd" />

Visuals
<img width="963" height="408" alt="image" src="https://github.com/user-attachments/assets/6651b82e-a144-4e2b-b025-e334e2caf521" />

API
<img width="1155" height="577" alt="image" src="https://github.com/user-attachments/assets/ec6579af-a66a-488a-b3f4-5f723953c2bb" />

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

3. **Optional Enhanced Dependencies** (for advanced features):
   ```bash
   uv add scikit-learn          # For semantic clustering
   uv add elasticsearch         # For hybrid search
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
