#!/bin/bash

# Production Mode Docker Runner for Forum Wisdom Miner
# Optimized for deployment with data persistence only

set -e

IMAGE_NAME="forum-analyzer"
CONTAINER_NAME="forum-analyzer-prod"

echo "🚀 Forum Wisdom Miner - Production Mode"
echo "========================================"

# Build image
echo "📦 Building Docker image..."
docker build -t $IMAGE_NAME .

# Stop existing container if running
if [ $(docker ps -aq -f name=^${CONTAINER_NAME}$) ]; then
    echo "🛑 Stopping existing production container..."
    docker stop $CONTAINER_NAME >/dev/null 2>&1 || true
    docker rm $CONTAINER_NAME >/dev/null 2>&1 || true
fi

echo "🚀 Starting production container with:"
echo "   - No source code mounts (security)"
echo "   - Persistent data only (tmp/ mounted)"
echo "   - Memory limit: 6GB (for 8GB systems)"
echo "   - Auto-restart policy"
echo "   - Port 5000 exposed"
echo ""

# Run in production mode
docker run -d \
  --name $CONTAINER_NAME \
  -p 5000:5000 \
  -v "$(pwd)/tmp":/app/tmp \
  -e BASE_TMP_DIR=/app/tmp \
  -e PYTHONUNBUFFERED=1 \
  --memory=6g \
  --restart unless-stopped \
  $IMAGE_NAME

echo "✅ Production container started: $CONTAINER_NAME"
echo ""
echo "📊 Container Status:"
docker ps --filter name=$CONTAINER_NAME --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
echo ""
echo "📋 Useful commands:"
echo "   View logs:    docker logs -f $CONTAINER_NAME"
echo "   Stop:         docker stop $CONTAINER_NAME"
echo "   Remove:       docker rm $CONTAINER_NAME"
echo "   Shell:        docker exec -it $CONTAINER_NAME /bin/bash"