# Forum Wisdom Miner v2.0 - M1 Optimized

A professional Flask application for comprehensive forum thread analysis using advanced semantic search, analytics, and AI-powered insights. Optimized for M1 MacBook Air with 8GB RAM.

---

## 🚀 Features

### **Advanced Forum Analysis**
- **Complete Thread Scraping**: Downloads all forum pages (up to 1000 pages)
- **Multi-Platform Support**: XenForo, vBulletin, phpBB, and custom forums
- **Vote/Reaction Extraction**: Captures upvotes, reactions, and community engagement
- **Smart Post Processing**: Content filtering, deduplication, and quality assessment
- **HTML Preservation**: Saves raw HTML for reprocessing with new optimizations

### **Dual Query System**
- **Analytical Queries**: Direct data analysis for "who is most active", "how many posts", etc.
- **Semantic Queries**: Vector search for content-based questions using embeddings
- **Positional Queries**: "Who was the second user to post?" with chronological analysis
- **Smart Routing**: Automatically detects query type and uses appropriate processor

### **Semantic Search & AI**
- **HNSW Vector Index**: Fast approximate nearest neighbor search (M1 optimized)
- **HyDE Enhancement**: Hypothetical Document Embeddings for better search accuracy
- **Domain-Specific Processing**: Optimized for technical/device forum terminology
- **Multi-Factor Ranking**: Recency, votes, author authority, and content quality

### **Thread Analytics**
- **Participant Analysis**: Complete user activity across entire thread
- **Content Statistics**: Post metrics, page distributions, engagement patterns
- **Temporal Analysis**: Activity over time, posting patterns, thread evolution
- **Topic Extraction**: Primary keywords, themes, and discussion insights

### **Production Features**
- **Advanced Caching**: Content-based embedding cache with intelligent cleanup
- **Progress Tracking**: Real-time progress bars for all major operations
- **Reprocess Capability**: Re-parse HTML with new optimizations without re-downloading
- **Error Handling**: Production-ready error handling with detailed logging
- **Memory Optimization**: Tuned for M1 MacBook Air 8GB RAM constraints

---

## 🛠 Requirements

### **System Requirements**
- **Hardware**: M1 MacBook Air (8GB RAM optimized) or compatible
- **OS**: macOS (optimized) or Linux/Windows
- **Python**: 3.9+ (tested with 3.13)

### **Dependencies**
- **Flask**: Web framework for the application interface
- **Requests**: HTTP client for forum scraping
- **BeautifulSoup4**: HTML parsing and content extraction
- **hnswlib**: High-performance vector similarity search
- **numpy**: Numerical operations for embeddings
- **tqdm**: Progress bars for long operations

### **AI Models (Ollama)**
- **Chat Model**: `deepseek-r1:1.5b` (1.5B parameters, ~1.2GB RAM)
- **Embedding Model**: `nomic-embed-text:v1.5` (768 dimensions, efficient)

---

## 📦 Installation

### **1. Install Ollama and Models**
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull required models
ollama pull deepseek-r1:1.5b
ollama pull nomic-embed-text:v1.5
```

### **2. Setup Application**
```bash
# Clone repository
git clone <repository-url>
cd fc-wisdom-miner

# Setup with uv (recommended)
uv venv --python 3.13
uv pip install Flask requests beautifulsoup4 hnswlib numpy tqdm

# Or with pip
pip install Flask requests beautifulsoup4 hnswlib numpy tqdm
```

### **3. Run Application**
```bash
# Using uv (recommended)
uv run python app.py

# Or with activated virtual environment
source .venv/bin/activate
python app.py
```

---

## 🌐 Usage

### **Access the Application**
- **Local**: http://127.0.0.1:8080
- **Network**: http://192.168.1.209:8080 (for other devices)

### **Processing New Threads**
1. **Enter Forum URL**: Paste the thread URL in the input field
2. **Ask Question**: Enter your question about the thread
3. **Click "Analyze Thread"**: System will scrape, process, and answer

### **Using Existing Threads**
1. **Select Thread**: Choose from dropdown of processed threads
2. **Optional Reprocess**: Check "Reprocess thread" for latest optimizations
3. **Ask Question**: Enter your question
4. **Get Results**: Receive AI-powered analysis with post links

### **Query Types**

#### **Analytical Queries** (Data Analysis)
- "Who is the most active user?"
- "How many posts are in this thread?"
- "Who was the second user to post?"
- "What are the thread statistics?"

#### **Semantic Queries** (Content Search)
- "What are the best heating techniques?"
- "How do I fix vapor quality issues?"
- "What do people think about temperature control?"
- "Summarize the main discussion points"

#### **Positional Queries** (Chronological)
- "Who posted first?" / "Who was the first user?"
- "Who was the second/third user to post?"
- "What did the initial post say?"

---

## ⚙️ Configuration

### **M1 MacBook Air Optimizations**
The application is pre-configured for optimal performance on M1 MacBook Air with 8GB RAM:

- **MAX_WORKERS**: 3 (optimized for M1 efficiency cores)
- **EMBEDDING_BATCH_SIZE**: 8 (prevents memory spikes)
- **HNSW_M**: 12 (memory-efficient indexing)
- **CACHE_SIZE**: 150MB (fits comfortably in 8GB)

### **Environment Variables**
```bash
export BASE_TMP_DIR="tmp"                           # Data storage directory
export OLLAMA_BASE_URL="http://localhost:11434"    # Ollama API URL
export OLLAMA_CHAT_MODEL="deepseek-r1:1.5b"       # Chat model
export OLLAMA_EMBED_MODEL="nomic-embed-text:v1.5"  # Embedding model
```

---

## 📊 Data Storage

```
tmp/
├── embeddings_cache/              # Vector embeddings cache
│   ├── *.pkl                     # Individual embedding files
│   └── cache_metadata.json       # Cache management
├── app.log                       # Application logs
└── threads/
    └── [thread-key]/             # Per-thread directories
        ├── html_pages/           # Raw HTML files
        ├── posts.json            # Processed posts
        ├── metadata.json         # Thread metadata
        ├── analytics.json        # Thread analytics
        └── index_hnsw.bin        # Vector search index
```

---

## 🔧 Troubleshooting

### **Common Issues**

#### **"Thread not found" on reprocess**
- HTML files missing, will fallback to posts.json method
- Thread may need to be recreated from URL

#### **Slow performance**
- Check embedding cache hit rate
- Consider reducing batch sizes for 8GB systems
- Monitor system memory usage

#### **Missing post links**
- Reprocess existing threads to extract URLs
- Check forum platform CSS selectors

### **Performance Monitoring**
- All operations include timing statistics
- Embedding cache hit/miss ratios tracked
- Search result ranking scores available
- Memory usage optimized for M1 systems

---

## 🏗 Architecture

### **Modular Design**
- **Forum Scraper**: Multi-platform thread downloading
- **Post Processor**: Content cleaning and enhancement
- **Embedding Manager**: Vector generation and caching
- **Search Engine**: Semantic and analytical query processing
- **Analytics Engine**: Thread statistics and insights
- **Query Processor**: Smart routing and response generation

### **M1 Optimizations**
- Reduced worker threads for efficiency cores
- Memory-conscious batch processing
- Optimized vector index parameters
- Intelligent caching strategies

---

## 🤝 Contributing

This is a professional-grade application optimized for M1 MacBook Air systems. All code follows production best practices with comprehensive error handling and logging.

---

## 📄 License

Professional forum analysis application for educational and research purposes.