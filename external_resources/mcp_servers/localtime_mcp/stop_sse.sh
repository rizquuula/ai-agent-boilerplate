#!/bin/bash
# Stop and remove the LocalTime MCP SSE Docker container

CONTAINER_NAME="localtime-mcp-sse"

# Check if container exists
if ! docker ps -a --format "{{.Names}}" | grep -q "^${CONTAINER_NAME}$"; then
    echo "Container '${CONTAINER_NAME}' does not exist"
    exit 1
fi

echo "Stopping LocalTime MCP SSE container: ${CONTAINER_NAME}..."
docker stop "${CONTAINER_NAME}" > /dev/null 2>&1

if [ $? -eq 0 ]; then
    echo "Container stopped successfully"
else
    echo "Failed to stop container (may already be stopped)"
fi

echo "Removing container: ${CONTAINER_NAME}..."
docker rm "${CONTAINER_NAME}" > /dev/null 2>&1

if [ $? -eq 0 ]; then
    echo "Container removed successfully"
else
    echo "Failed to remove container"
    exit 1
fi

echo "LocalTime MCP SSE container stopped and removed"
