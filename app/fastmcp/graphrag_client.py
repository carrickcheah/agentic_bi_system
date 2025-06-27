"""
GraphRAG MCP Client

Provides GraphRAG operations through MCP protocol.
Follows the established MCP client pattern for consistency.
"""

from typing import Dict, Any, List, Optional
from mcp.client.session import ClientSession

try:
    from ..utils.logging import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class GraphRAGClient:
    """GraphRAG operations through MCP."""
    
    def __init__(self, session: ClientSession):
        self.session = session
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call an MCP tool directly."""
        try:
            result = await self.session.call_tool(tool_name, arguments)
            return result
        except Exception as e:
            logger.error(f"GraphRAG MCP tool call failed: {tool_name} - {e}")
            raise
    
    async def search_business_entities(
        self,
        query: str,
        limit: int = 10
    ) -> Dict[str, Any]:
        """
        Search for business entities in the knowledge graph.
        
        Args:
            query: Natural language search query for entities
            limit: Maximum number of entities to return (1-50)
            
        Returns:
            Dict containing entity search results with metadata
        """
        try:
            result = await self.session.call_tool(
                "search_business_entities",
                {
                    "query": query,
                    "limit": limit
                }
            )
            
            return result
            
        except Exception as e:
            logger.error(f"GraphRAG entity search failed: {e}")
            raise
    
    async def global_business_analysis(
        self,
        query: str,
        max_communities: int = 3
    ) -> Dict[str, Any]:
        """
        Perform comprehensive cross-domain business analysis.
        
        Args:
            query: Business question for comprehensive analysis
            max_communities: Maximum business domains to analyze (1-10)
            
        Returns:
            Dict containing cross-domain insights and synthesized analysis
        """
        try:
            result = await self.session.call_tool(
                "global_business_analysis",
                {
                    "query": query,
                    "max_communities": max_communities
                }
            )
            
            return result
            
        except Exception as e:
            logger.error(f"GraphRAG global analysis failed: {e}")
            raise
    
    async def get_health_status(self) -> Dict[str, Any]:
        """
        Get GraphRAG server health status and performance metrics.
        
        Returns:
            Dict containing server health and performance data
        """
        try:
            result = await self.session.call_tool(
                "get_graphrag_health",
                {}
            )
            
            return result
            
        except Exception as e:
            logger.error(f"GraphRAG health check failed: {e}")
            raise
    
    async def test_connection(self) -> bool:
        """
        Test GraphRAG MCP connection.
        
        Returns:
            True if connection is healthy, False otherwise
        """
        try:
            health_result = await self.get_health_status()
            
            # Check if we got a valid response
            if isinstance(health_result, dict) and "server_health" in health_result:
                server_health = health_result["server_health"]
                return server_health.get("status") in ["healthy", "initializing"]
            
            return False
            
        except Exception as e:
            logger.error(f"GraphRAG connection test failed: {e}")
            return False
    
    # Business intelligence helper methods
    
    async def analyze_business_relationships(
        self,
        entity_name: str,
        relationship_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Analyze business relationships for a specific entity.
        
        Args:
            entity_name: Name of the business entity to analyze
            relationship_types: Optional list of relationship types to focus on
            
        Returns:
            Dict containing relationship analysis results
        """
        try:
            # First, search for the entity
            entity_search = await self.search_business_entities(
                query=entity_name,
                limit=5
            )
            
            if not entity_search.get("entities"):
                return {
                    "status": "entity_not_found",
                    "message": f"No entities found matching '{entity_name}'"
                }
            
            # Then perform global analysis focusing on relationships
            relationship_query = f"What are the key business relationships and connections involving {entity_name}?"
            if relationship_types:
                relationship_query += f" Focus on {', '.join(relationship_types)} relationships."
            
            global_analysis = await self.global_business_analysis(
                query=relationship_query,
                max_communities=3
            )
            
            return {
                "status": "success",
                "entity_search": entity_search,
                "relationship_analysis": global_analysis
            }
            
        except Exception as e:
            logger.error(f"GraphRAG relationship analysis failed: {e}")
            raise
    
    async def cross_domain_investigation(
        self,
        business_question: str,
        domains: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Perform cross-domain business investigation.
        
        Args:
            business_question: The business question to investigate
            domains: Optional list of specific domains to focus on
            
        Returns:
            Dict containing cross-domain investigation results
        """
        try:
            # Enhance query with domain specification if provided
            enhanced_query = business_question
            if domains:
                enhanced_query += f" Consider impacts across these business domains: {', '.join(domains)}."
            
            # Perform global analysis with focus on cross-domain insights
            result = await self.global_business_analysis(
                query=enhanced_query,
                max_communities=min(len(domains) if domains else 3, 5)  # Limit communities
            )
            
            # Add metadata about the investigation approach
            if isinstance(result, dict):
                result["investigation_metadata"] = {
                    "original_question": business_question,
                    "specified_domains": domains,
                    "investigation_type": "cross_domain_analysis"
                }
            
            return result
            
        except Exception as e:
            logger.error(f"GraphRAG cross-domain investigation failed: {e}")
            raise
    
    async def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get a summary of GraphRAG performance metrics.
        
        Returns:
            Dict containing simplified performance summary
        """
        try:
            health_data = await self.get_health_status()
            
            if not isinstance(health_data, dict):
                return {"status": "error", "message": "Invalid health data format"}
            
            server_health = health_data.get("server_health", {})
            performance_metrics = health_data.get("performance_metrics", {})
            
            return {
                "status": server_health.get("status", "unknown"),
                "uptime_hours": round(server_health.get("uptime_seconds", 0) / 3600, 2),
                "cache_hit_rate": round(server_health.get("cache_hit_rate", 0), 1),
                "total_requests": server_health.get("total_requests", 0),
                "avg_entity_search_time": round(performance_metrics.get("entity_search_avg_time", 0), 2),
                "avg_global_search_time": round(performance_metrics.get("global_search_avg_time", 0), 2),
                "total_cost_today": round(performance_metrics.get("total_cost_today", 0), 3),
                "error_rate": round(performance_metrics.get("error_rate_percentage", 0), 1)
            }
            
        except Exception as e:
            logger.error(f"GraphRAG performance summary failed: {e}")
            return {
                "status": "error", 
                "message": f"Failed to get performance summary: {str(e)}"
            }