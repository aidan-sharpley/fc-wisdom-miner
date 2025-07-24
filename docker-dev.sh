#!/bin/bash

# Development Mode Docker Runner for Forum Wisdom Miner
# Provides live code editing and data persistence

set -e

IMAGE_NAME="forum-analyzer"
CONTAINER_NAME="forum-analyzer-dev"

echo "🔧 Forum Wisdom Miner - Development Mode"
echo "========================================="

# Build image if it doesn't exist
if [[ "$(docker images -q $IMAGE_NAME 2> /dev/null)" == "" ]]; then
    echo "📦 Building Docker image..."
    docker build -t $IMAGE_NAME .
fi

# Stop existing container if running
if [ $(docker ps -aq -f name=^${CONTAINER_NAME}$) ]; then
    echo "🛑 Stopping existing development container..."
    docker stop $CONTAINER_NAME >/dev/null 2>&1 || true
    docker rm $CONTAINER_NAME >/dev/null 2>&1 || true
fi

echo "🚀 Starting development container with:"
echo "   - Live code editing (source mounted)"
echo "   - Persistent data (tmp/ mounted)"
echo "   - Hot reload enabled"
echo "   - Port 5000 exposed"
echo ""

# Run in development mode
docker run -it --rm \
  --name $CONTAINER_NAME \
  -p 5000:5000 \
  -v "$(pwd)":/app \
  -v "$(pwd)/tmp":/app/tmp \
  -e FLASK_ENV=development \
  -e BASE_TMP_DIR=/app/tmp \
  -e PYTHONUNBUFFERED=1 \
  $IMAGE_NAME

echo ""
echo "✅ Development container stopped"