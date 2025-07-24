# Docker Setup for Forum Wisdom Miner

This document provides instructions for running the Forum Wisdom Miner in Docker containers for both development and production environments.

## Quick Start

### Prerequisites
- Docker installed and running
- At least 8GB RAM available
- Ollama running locally with required models:
  - `deepseek-r1:1.5b`
  - `nomic-embed-text:v1.5`

### Development Mode
```bash
# Start development container with live code editing
./docker-dev.sh
```

### Production Mode
```bash
# Start production container with optimizations
./docker-prod.sh
```

### Interactive Shell
```bash
# Access container shell for debugging
./docker-shell.sh
```

## Detailed Usage

### Development Mode Features
- **Live Code Editing**: Source code mounted for real-time changes
- **Hot Reload**: Flask automatically restarts on code changes
- **Data Persistence**: `tmp/` directory preserved between runs
- **Debug Mode**: Full Flask debugging enabled
- **Port**: Accessible at http://localhost:5000

**Usage:**
```bash
./docker-dev.sh
```

This is equivalent to:
```bash
docker run -it --rm \
  --name forum-analyzer-dev \
  -p 5000:5000 \
  -v "$(pwd)":/app \
  -v "$(pwd)/tmp":/app/tmp \
  -e FLASK_ENV=development \
  -e BASE_TMP_DIR=/app/tmp \
  forum-analyzer
```

### Production Mode Features
- **Security**: No source code mounts
- **Memory Optimized**: 6GB limit for 8GB systems
- **Auto-restart**: Container restarts on failure
- **Data Only**: Only `tmp/` directory mounted
- **Daemon Mode**: Runs in background

**Usage:**
```bash
./docker-prod.sh
```

**Management Commands:**
```bash
# View logs
docker logs -f forum-analyzer-prod

# Stop container
docker stop forum-analyzer-prod

# Remove container
docker rm forum-analyzer-prod

# Access shell
docker exec -it forum-analyzer-prod /bin/bash
```

### Manual Docker Commands

**Build Image:**
```bash
docker build -t forum-analyzer .
```

**Development Run:**
```bash
docker run -it --rm \
  -p 5000:5000 \
  -v "$(pwd)":/app \
  -v "$(pwd)/tmp":/app/tmp \
  -e FLASK_ENV=development \
  -e BASE_TMP_DIR=/app/tmp \
  forum-analyzer
```

**Production Run:**
```bash
docker run -d \
  --name forum-analyzer-prod \
  -p 5000:5000 \
  -v "$(pwd)/tmp":/app/tmp \
  -e BASE_TMP_DIR=/app/tmp \
  --memory=6g \
  --restart unless-stopped \
  forum-analyzer
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FLASK_ENV` | `production` | Set to `development` for debug mode |
| `PORT` | `5000` | Flask server port |
| `BASE_TMP_DIR` | `/app/tmp` | Directory for data storage |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API endpoint |
| `OLLAMA_CHAT_MODEL` | `deepseek-r1:1.5b` | LLM model for chat |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text:v1.5` | Embedding model |

## Data Persistence

The `tmp/` directory contains:
- **Embeddings Cache**: Pre-computed embeddings (`embeddings_cache/`)
- **Thread Data**: Processed forum threads (`threads/`)
- **HTML Files**: Saved forum pages for reprocessing
- **Indexes**: HNSW search indexes
- **Analytics**: Generated thread analytics
- **Logs**: Application logs

**Important**: Always mount `tmp/` to preserve your processed data and cache between container runs.

## Networking

### Ollama Connection
The container needs to connect to Ollama running on the host:

**On macOS/Windows:**
```bash
# Ollama accessible via host.docker.internal
-e OLLAMA_BASE_URL=http://host.docker.internal:11434
```

**On Linux:**
```bash
# Use host networking or docker bridge
--network host
# OR
-e OLLAMA_BASE_URL=http://172.17.0.1:11434
```

### Custom Network
```bash
# Create custom network
docker network create forum-network

# Run with custom network
docker run --network forum-network ...
```

## Troubleshooting

### Common Issues

**1. Cannot connect to Ollama:**
```bash
# Check Ollama is running
ollama list

# Test connection from container
docker exec -it forum-analyzer-dev curl http://host.docker.internal:11434/api/tags
```

**2. Permission errors on tmp/:**
```bash
# Fix permissions
chmod -R 777 tmp/

# Or run with user mapping
docker run --user $(id -u):$(id -g) ...
```

**3. Memory issues:**
```bash
# Increase Docker memory limit
# Docker Desktop: Settings > Resources > Memory

# Monitor container memory
docker stats forum-analyzer-prod
```

**4. Port already in use:**
```bash
# Use different port
docker run -p 5001:5000 ...

# Check what's using port 5000
lsof -i :5000
```

### Performance Tuning

**For 8GB Systems:**
```bash
# Limit container memory
docker run --memory=6g --memory-swap=6g ...

# Monitor resource usage
docker stats
```

**For Development:**
```bash
# Faster rebuilds with cache
docker build --cache-from forum-analyzer .

# Skip unnecessary files
# (already configured in .dockerignore)
```

## Security Considerations

### Production Security
- Source code not mounted in production
- Limited memory allocation
- Non-root user execution (if needed)
- Network isolation options

### Development Security
- Source code mounted read-only if needed:
  ```bash
  -v "$(pwd)":/app:ro
  ```

## Integration Examples

### Docker Compose (Alternative)
```yaml
version: '3.8'
services:
  forum-analyzer:
    build: .
    ports:
      - "5000:5000"
    volumes:
      - ./tmp:/app/tmp
    environment:
      - BASE_TMP_DIR=/app/tmp
      - OLLAMA_BASE_URL=http://host.docker.internal:11434
    restart: unless-stopped
    mem_limit: 6g
```

### Kubernetes Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: forum-analyzer
spec:
  replicas: 1
  selector:
    matchLabels:
      app: forum-analyzer
  template:
    metadata:
      labels:
        app: forum-analyzer
    spec:
      containers:
      - name: forum-analyzer
        image: forum-analyzer:latest
        ports:
        - containerPort: 5000
        env:
        - name: BASE_TMP_DIR
          value: /app/tmp
        resources:
          limits:
            memory: "6Gi"
        volumeMounts:
        - name: data-volume
          mountPath: /app/tmp
      volumes:
      - name: data-volume
        persistentVolumeClaim:
          claimName: forum-data-pvc
```

This Docker setup provides a robust, production-ready containerization solution for the Forum Wisdom Miner while maintaining excellent developer ergonomics.