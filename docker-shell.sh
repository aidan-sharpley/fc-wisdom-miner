#!/bin/bash

# Interactive Shell Access for Forum Wisdom Miner Docker Container

set -e

IMAGE_NAME="forum-analyzer"

echo "🐚 Forum Wisdom Miner - Interactive Shell"
echo "=========================================="

# Build image if it doesn't exist
if [[ "$(docker images -q $IMAGE_NAME 2> /dev/null)" == "" ]]; then
    echo "📦 Building Docker image..."
    docker build -t $IMAGE_NAME .
fi

echo "🚀 Starting interactive shell with:"
echo "   - Source code mounted for debugging"
echo "   - Persistent data (tmp/ mounted)"
echo "   - Full bash access"
echo ""

# Run interactive shell
docker run -it --rm \
  -v "$(pwd)":/app \
  -v "$(pwd)/tmp":/app/tmp \
  -e BASE_TMP_DIR=/app/tmp \
  -e PYTHONUNBUFFERED=1 \
  --entrypoint /bin/bash \
  $IMAGE_NAME

echo ""
echo "✅ Shell session ended"