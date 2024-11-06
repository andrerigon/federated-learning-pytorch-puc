#!/bin/bash

# Define default output directory
OUTPUT_DIR="${OUTPUT_DIR:-$(pwd)/output}"

# Check if Dockerfile exists in the current directory
if [ ! -f Dockerfile ]; then
    echo "Dockerfile not found! Please ensure you're running this script in the directory with the Dockerfile."
    exit 1
fi

# Define default output directory
OUTPUT_DIR="${PWD}/output"

# Build the Docker image
echo "Building Docker image 'federated-sim'..."

# Build Docker image with a unique tag to avoid rebuilds if unchanged
docker build -t federated-sim:latest .

# Check if the image build was successful
if [ $? -ne 0 ]; then
    echo "Failed to build Docker image."
    exit 1
fi

# Run the Docker container with output directory and passed parameters
echo "Running Docker container with output directory mounted at ${OUTPUT_DIR}..."

docker run --rm \
  -v "${OUTPUT_DIR}:/app/output:cached" \
  --privileged \
  --cpuset-cpus="0-9" \
  --shm-size=10g \
  --cpus="10" federated-sim "$@" 
  
  
  