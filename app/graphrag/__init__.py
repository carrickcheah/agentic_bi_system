"""
GraphRAG Module

Microsoft GraphRAG integration as MCP service for complex business intelligence.
Provides relationship intelligence and cross-domain analysis for comprehensive investigations.

Key Components:
- graphrag_server.py: Stateful GraphRAG server with hybrid architecture
- mcp_server.py: MCP interface for protocol compliance
- config.py: GraphRAG-specific configuration settings
- models.py: Pydantic models for GraphRAG operations
"""

from .config import GraphRAGConfig
from .models import (
    GraphRAGEntitySearchRequest,
    GraphRAGEntitySearchResponse,
    GraphRAGGlobalSearchRequest,
    GraphRAGGlobalSearchResponse
)

__all__ = [
    "GraphRAGConfig",
    "GraphRAGEntitySearchRequest",
    "GraphRAGEntitySearchResponse", 
    "GraphRAGGlobalSearchRequest",
    "GraphRAGGlobalSearchResponse"
]