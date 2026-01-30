FROM python:3.11-slim

WORKDIR /app

# Install uv
RUN pip install uv

# Copy project files
COPY pyproject.toml readme.md ./
COPY localtime_mcp/ ./localtime_mcp/

# Install dependencies
RUN uv sync

# Environment variables with defaults
ENV MCP_TRANSPORT=stdio
ENV MCP_PORT=3000

# Expose the port for HTTP-based transports
EXPOSE 3000
EXPOSE 3001

# Run the MCP server
CMD ["uv", "run", "python", "-m", "localtime_mcp.server"]
