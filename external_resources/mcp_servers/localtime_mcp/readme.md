# LocalTime MCP

A simple MCP server that returns the current local time with timezone information.

## Functions

### get_current_time()

Returns:
  - ISO datetime string
  - UNIX timestamp
  - Timezone

## Transport Modes

This MCP server supports three transport modes:

### 1. stdio (default)

Standard input/output transport for MCP clients.

```bash
# Run with stdio (default)
uvx external_resources/mcp_servers/localtime_mcp

# Or with environment variable
MCP_TRANSPORT=stdio uvx external_resources/mcp_servers/localtime_mcp
```

### 2. SSE (Server-Sent Events)

HTTP-based transport using Server-Sent Events. Runs in Docker container.

```bash
cd external_resources/mcp_servers/localtime_mcp

# Start SSE server on port 3000
./run_sse.sh

# Stop SSE server
./stop_sse.sh
```

The SSE server will be available at `http://localhost:3000`.

### 3. HTTP Stream

HTTP-based transport using streamable HTTP. Runs in Docker container.

```bash
cd external_resources/mcp_servers/localtime_mcp

# Start HTTP server on port 3001
./run_http_stream.sh

# Stop HTTP server
./stop_http_stream.sh
```

The HTTP server will be available at `http://localhost:3001`.

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MCP_TRANSPORT` | Transport type: `stdio`, `sse`, or `http` | `stdio` |
| `MCP_PORT` | Port for HTTP-based transports | `3000` |

## Docker Usage

The SSE and HTTP stream transports run inside Docker containers.

### Prerequisites

- Docker must be installed and running

### How it works

The shell scripts handle Docker operations:

- `run_sse.sh`: Builds the image (if needed) and runs `docker run -d`
- `stop_sse.sh`: Runs `docker stop` then `docker rm`
- `run_http_stream.sh`: Builds the image (if needed) and runs `docker run -d`
- `stop_http_stream.sh`: Runs `docker stop` then `docker rm`

### Manual Docker commands

```bash
cd external_resources/mcp_servers/localtime_mcp

# Build the image
docker build -t localtime-mcp .

# Run with SSE
docker run -d \
    --name localtime-mcp-sse \
    -e MCP_TRANSPORT=sse \
    -e MCP_PORT=3000 \
    -p 3000:3000 \
    localtime-mcp

# Stop SSE container
docker stop localtime-mcp-sse
docker rm localtime-mcp-sse

# Run with HTTP
docker run -d \
    --name localtime-mcp-http \
    -e MCP_TRANSPORT=http \
    -e MCP_PORT=3001 \
    -p 3001:3001 \
    localtime-mcp

# Stop HTTP container
docker stop localtime-mcp-http
docker rm localtime-mcp-http
```

## Requirements

- Python >= 3.11
- Docker (for SSE and HTTP transports)
- See [pyproject.toml](pyproject.toml) for dependencies

## Installation

```bash
cd external_resources/mcp_servers/localtime_mcp
uv sync
```
