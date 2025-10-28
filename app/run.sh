#!/bin/bash
set -e

# Absolute path for data folder
DATA_DIR=$(pwd)/data
mkdir -p "$DATA_DIR"

# Ask for GROQ API key securely
read -sp "Enter GROQ API Key: " GROQ_API_KEY
echo ""

# Build Docker image
docker build -t chatbot-groq ./app

# Remove old container if exists
if [ "$(docker ps -aq -f name=chatbot-groq-container)" ]; then
    docker rm -f chatbot-groq-container
fi

# Run Docker container
docker run -d \
  -p 8000:8000 \
  -v "$DATA_DIR":/data \
  -e GROQ_API_KEY="$GROQ_API_KEY" \
  --name chatbot-groq-container \
  chatbot-groq

echo "✅ Container running. API available at http://localhost:8000"
