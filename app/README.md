# Agentic SQL App

Autonomous Business Intelligence System with MCP Integration

## Quick Start

```bash
# Install dependencies
uv sync

# Run the server
uv run python main.py

# Or use uvicorn directly
uv run uvicorn app.main:app --reload
```

## Features

- FastAPI backend with MCP server/client integration
- Qdrant vector database for semantic search
- Multi-database support via MCP protocol
- WebSocket support for real-time investigations