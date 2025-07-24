FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies
RUN uv pip install --system -r requirements.txt

# Copy source code (only used in production)
COPY . .

# Create tmp directory and set permissions
RUN mkdir -p /app/tmp && chmod 777 /app/tmp

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV BASE_TMP_DIR=/app/tmp
ENV FLASK_APP=app.py

# Expose port
EXPOSE 5000

# Default command
CMD ["uv", "run", "python", "app.py"]