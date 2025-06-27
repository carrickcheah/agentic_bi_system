"""
GraphRAG MCP Server

MCP protocol interface for GraphRAG server.
Provides standardized tools for business relationship intelligence.

Key Features:
- MCP protocol compliance
- GraphRAG server integration  
- Error handling and fallback strategies
- Performance monitoring and cost controls
"""

import asyncio
import os
import sys
from typing import Dict, Any, List
from datetime import datetime

# MCP imports
from mcp.server.models import (
    GetPromptResult,
    InitializeResult,
    ListPromptsResult,
    ListToolsResult,
    GetToolResult,
    CallToolResult,
    Tool,
    ToolCall,
)
from mcp.server.stdio import create_stdio_server
from mcp.types import JSONRPCRequest, JSONRPCResponse

from .config import GraphRAGConfig
from .graphrag_server import GraphRAGServer
from .models import (
    GraphRAGEntitySearchRequest,
    GraphRAGGlobalSearchRequest,
    GraphRAGErrorResponse
)


class GraphRAGMCPServer:
    """
    MCP server interface for GraphRAG operations.
    
    Exposes GraphRAG capabilities as standard MCP tools for business intelligence.
    """
    
    def __init__(self):
        self.config = GraphRAGConfig()
        self.graphrag_server: GraphRAGServer = None
        self.server = None
        self.request_counter = 0
        self.startup_time = datetime.utcnow()
    
    async def initialize(self):
        """Initialize GraphRAG MCP server."""
        try:
            print("🚀 Initializing GraphRAG MCP Server...", file=sys.stderr)
            
            # Create and start GraphRAG server
            self.graphrag_server = GraphRAGServer(self.config)
            success = await self.graphrag_server.start_server()
            
            if not success:
                raise RuntimeError("GraphRAG server failed to start")
            
            # Create MCP server
            self.server = create_stdio_server("graphrag")
            
            # Register MCP handlers
            self._register_handlers()
            
            print("✅ GraphRAG MCP Server initialized successfully", file=sys.stderr)
            
        except Exception as e:
            print(f"❌ GraphRAG MCP Server initialization failed: {e}", file=sys.stderr)
            raise
    
    def _register_handlers(self):
        """Register MCP protocol handlers."""
        
        @self.server.list_tools()
        async def list_tools() -> ListToolsResult:
            """List available GraphRAG tools."""
            return ListToolsResult(
                tools=[
                    Tool(
                        name="search_business_entities",
                        description="Search for business entities in the knowledge graph using natural language. "
                                  "Finds people, organizations, concepts, and relationships relevant to business questions.",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "Natural language search query for entities"
                                },
                                "limit": {
                                    "type": "integer", 
                                    "description": "Maximum number of entities to return (1-50)",
                                    "minimum": 1,
                                    "maximum": 50,
                                    "default": 10
                                }
                            },
                            "required": ["query"]
                        }
                    ),
                    Tool(
                        name="global_business_analysis",
                        description="Perform comprehensive cross-domain business analysis using GraphRAG. "
                                  "Analyzes relationships across multiple business domains to generate strategic insights.",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "Business question for comprehensive analysis"
                                },
                                "max_communities": {
                                    "type": "integer",
                                    "description": "Maximum business domains to analyze (1-10)",
                                    "minimum": 1,
                                    "maximum": 10,
                                    "default": 3
                                }
                            },
                            "required": ["query"]
                        }
                    ),
                    Tool(
                        name="get_graphrag_health",
                        description="Get GraphRAG server health status and performance metrics.",
                        inputSchema={
                            "type": "object",
                            "properties": {},
                            "additionalProperties": False
                        }
                    )
                ]
            )
        
        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> CallToolResult:
            """Handle tool calls."""
            self.request_counter += 1
            request_id = f"req_{self.request_counter}_{int(datetime.utcnow().timestamp())}"
            
            try:
                if name == "search_business_entities":
                    return await self._handle_entity_search(arguments, request_id)
                elif name == "global_business_analysis":
                    return await self._handle_global_analysis(arguments, request_id)
                elif name == "get_graphrag_health":
                    return await self._handle_health_check(arguments, request_id)
                else:
                    return CallToolResult(
                        content=[{
                            "type": "text",
                            "text": f"Unknown tool: {name}"
                        }],
                        isError=True
                    )
            
            except Exception as e:
                error_response = GraphRAGErrorResponse(
                    error_type="tool_execution_error",
                    error_message=str(e),
                    request_id=request_id,
                    fallback_available=name == "global_business_analysis"  # Can fallback to vector search
                )
                
                return CallToolResult(
                    content=[{
                        "type": "text", 
                        "text": error_response.model_dump_json(indent=2)
                    }],
                    isError=True
                )
    
    async def _handle_entity_search(self, arguments: Dict[str, Any], request_id: str) -> CallToolResult:
        """Handle entity search requests."""
        try:
            # Validate and parse request
            request = GraphRAGEntitySearchRequest(**arguments)
            
            # Execute search with timeout
            response = await self.graphrag_server.search_entities(
                query=request.query,
                limit=request.limit,
                timeout=self.config.entity_search_timeout
            )
            
            # Check for success
            if response.status == "success":
                return CallToolResult(
                    content=[{
                        "type": "text",
                        "text": response.model_dump_json(indent=2)
                    }]
                )
            else:
                return CallToolResult(
                    content=[{
                        "type": "text",
                        "text": f"Entity search failed: {response.status}"
                    }],
                    isError=True
                )
                
        except Exception as e:
            error_response = GraphRAGErrorResponse(
                error_type="entity_search_error",
                error_message=f"Entity search failed: {str(e)}",
                request_id=request_id,
                fallback_available=True,
                retry_after_seconds=5.0
            )
            
            return CallToolResult(
                content=[{
                    "type": "text",
                    "text": error_response.model_dump_json(indent=2)
                }],
                isError=True
            )
    
    async def _handle_global_analysis(self, arguments: Dict[str, Any], request_id: str) -> CallToolResult:
        """Handle global business analysis requests."""
        try:
            # Validate and parse request
            request = GraphRAGGlobalSearchRequest(**arguments)
            
            # Check daily budget
            metrics = await self.graphrag_server.get_performance_metrics()
            if metrics.total_cost_today >= self.config.daily_budget_limit:
                error_response = GraphRAGErrorResponse(
                    error_type="budget_limit_exceeded",
                    error_message=f"Daily budget limit of ${self.config.daily_budget_limit} exceeded",
                    request_id=request_id,
                    fallback_available=True,
                    retry_after_seconds=3600.0  # Retry after 1 hour
                )
                
                return CallToolResult(
                    content=[{
                        "type": "text",
                        "text": error_response.model_dump_json(indent=2)
                    }],
                    isError=True
                )
            
            # Execute global search with strict limits for MCP tool usage
            response = await self.graphrag_server.global_search(
                query=request.query,
                max_communities=min(request.max_communities, 3),  # Hard limit for MCP
                timeout=self.config.global_search_timeout,
                max_cost=request.max_cost or self.config.cost_limit_per_query
            )
            
            # Check for success
            if response.status == "success":
                return CallToolResult(
                    content=[{
                        "type": "text",
                        "text": response.model_dump_json(indent=2)
                    }]
                )
            else:
                return CallToolResult(
                    content=[{
                        "type": "text",
                        "text": f"Global analysis failed: {response.status}"
                    }],
                    isError=True
                )
                
        except Exception as e:
            error_response = GraphRAGErrorResponse(
                error_type="global_analysis_error",
                error_message=f"Global analysis failed: {str(e)}",
                request_id=request_id,
                fallback_available=True,
                retry_after_seconds=10.0
            )
            
            return CallToolResult(
                content=[{
                    "type": "text",
                    "text": error_response.model_dump_json(indent=2)
                }],
                isError=True
            )
    
    async def _handle_health_check(self, arguments: Dict[str, Any], request_id: str) -> CallToolResult:
        """Handle health check requests."""
        try:
            health = await self.graphrag_server.get_health_status()
            metrics = await self.graphrag_server.get_performance_metrics()
            
            health_data = {
                "server_health": health.model_dump(),
                "performance_metrics": metrics.model_dump(),
                "mcp_server_info": {
                    "startup_time": self.startup_time.isoformat(),
                    "total_requests": self.request_counter,
                    "server_version": "1.0.0"
                }
            }
            
            return CallToolResult(
                content=[{
                    "type": "text",
                    "text": f"```json\n{health_data}\n```"
                }]
            )
            
        except Exception as e:
            return CallToolResult(
                content=[{
                    "type": "text",
                    "text": f"Health check failed: {str(e)}"
                }],
                isError=True
            )
    
    async def run(self):
        """Run the GraphRAG MCP server."""
        if not self.server:
            await self.initialize()
        
        try:
            print("🟢 GraphRAG MCP Server running...", file=sys.stderr)
            async with self.server:
                await self.server.serve()
        except Exception as e:
            print(f"❌ GraphRAG MCP Server error: {e}", file=sys.stderr)
            raise
        finally:
            if self.graphrag_server:
                await self.graphrag_server.shutdown()
            print("🛑 GraphRAG MCP Server stopped", file=sys.stderr)


async def main():
    """Main entry point for GraphRAG MCP server."""
    try:
        server = GraphRAGMCPServer()
        await server.run()
    except KeyboardInterrupt:
        print("🛑 GraphRAG MCP Server interrupted", file=sys.stderr)
    except Exception as e:
        print(f"❌ GraphRAG MCP Server fatal error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())