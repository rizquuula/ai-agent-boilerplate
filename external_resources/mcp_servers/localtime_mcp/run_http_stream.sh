#!/bin/bash
# Run LocalTime MCP server with HTTP stream transport in Docker on port 3001

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER_NAME="localtime-mcp-http"
IMAGE_NAME="localtime-mcp:latest"

# Check if container is already running
if docker ps --format "{{.Names}}" | grep -q "^${CONTAINER_NAME}$"; then
    echo "LocalTime MCP HTTP container is already running"
    echo "  Container: ${CONTAINER_NAME}"
    echo "  Port: 3001"
    exit 1
fi

# Check if container exists but is stopped
if docker ps -a --format "{{.Names}}" | grep -q "^${CONTAINER_NAME}$"; then
    echo "Removing stopped container: ${CONTAINER_NAME}"
    docker rm "${CONTAINER_NAME}" > /dev/null 2>&1
fi

# Build the Docker image if needed
echo "Building Docker image ${IMAGE_NAME}..."
cd "${SCRIPT_DIR}"

docker build -t "${IMAGE_NAME}" --load .
BUILD_EXIT_CODE=$?

if [ $BUILD_EXIT_CODE -ne 0 ]; then
    echo "ERROR: Failed to build Docker image (exit code: $BUILD_EXIT_CODE)"
    exit 1
fi

# Verify image was created
echo ""
echo "Verifying image was created..."
docker image ls | grep "localtime-mcp" || echo "WARNING: Image 'localtime-mcp' not found in 'docker image ls'"

echo ""
echo "Docker image built successfully: ${IMAGE_NAME}"
echo "Starting LocalTime MCP HTTP container on port 3001..."

# Run the container in detached mode
docker run -d \
    --name "${CONTAINER_NAME}" \
    -e MCP_TRANSPORT=http \
    -e MCP_PORT=3001 \
    -p 3001:3001 \
    "${IMAGE_NAME}"

if [ $? -eq 0 ]; then
    echo ""
    echo "LocalTime MCP HTTP container started successfully!"
    echo "  Container: ${CONTAINER_NAME}"
    echo "  Port: 3001"
    echo ""
    echo "Stop with: ./stop_http_stream.sh"
else
    echo "Failed to start LocalTime MCP HTTP container"
    exit 1
fi
