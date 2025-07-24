# Intelligent Forum Analysis & Search Engine

This is a high-performance forum analysis application that uses local LLMs to provide accurate, data-driven insights and powerful semantic search for any forum thread.

Designed to run efficiently on consumer hardware (8GB RAM), this Flask-based tool scrapes entire forum threads, processes and enriches the content, and builds a sophisticated local search engine. It features a unique dual-engine query system that can distinguish between:

- **Analytical questions** (e.g., "Who is the most active user?") and
- **Semantic questions** (e.g., "What are the best heating techniques?").

The system provides 100% accurate, data-driven answers for analytical queries and nuanced, context-aware responses for semantic queries.

## ✨ Key Features

- 🎯 **Dual-Engine Query System**: Accurately answers both analytical (data-driven) and semantic (LLM-based) questions. No more wrong answers for statistical queries.
- 🔗 **Clickable Post Links**: Provides direct links to the specific source posts for all analytical results, ensuring full traceability.
- 🧠 **Local LLM Powered**: Uses Ollama with deepseek-r1:1.5b and nomic-embed-text models for privacy and performance.
- 🔄 **Smart Reprocessing**: Re-analyze existing threads without re-downloading using saved HTML.
- 📊 **Advanced Analytics**: Thread summaries, participant analysis, and engagement metrics.
- 🛡️ **Security First**: Input validation, SSRF protection, and data sanitization.
- ⚡ **Hardware Optimized**: Runs efficiently on 8GB RAM systems with progress tracking.
- 🌐 **Multi-Platform**: Supports XenForo, vBulletin, phpBB, and generic forums.
- 🚀 **Performance Monitoring**: Real-time analytics and bottleneck detection.

## 🏗️ Architecture

### Core Components

- **Forum Scraper**: Comprehensive page detection and HTML preservation
- **Query Router**: LLM-powered intelligent query classification and routing
- **Data Analyzer**: 100% accurate statistical analysis with post links
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

### Existing Thread Queries
- **Analytical**: "Who is the most active user?" → Data-driven answer with post counts
- **Positional**: "Who was the second user to post?" → Chronological analysis with links
- **Semantic**: "What are the best heating techniques?" → Vector search + LLM analysis
- **Temporal**: "How did opinions change over time?" → Timeline analysis

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

Optimized for consumer hardware:
- **Memory**: 8GB RAM recommended, <2GB typical usage
- **Processing**: Batch operations with progress tracking
- **Storage**: Efficient caching with automatic cleanup
- **Threading**: 3-worker optimization for memory efficiency

## 🛡️ Security Features

- Input validation and sanitization
- SSRF protection for URL requests
- Secure thread key generation
- Rate limiting and request throttling
- Comprehensive error handling with logging

## 📝 Development

See [CLAUDE.md](CLAUDE.md) for detailed development guidelines, architecture documentation, and troubleshooting information.
