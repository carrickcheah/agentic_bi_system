"""
Qdrant MCP Client

Provides Qdrant vector operations through MCP protocol.
"""

from typing import Dict, Any, List, Optional, Union
from mcp.client.session import ClientSession

try:
    from ..utils.logging import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class QdrantClient:
    """Qdrant vector operations through MCP."""
    
    def __init__(self, session: ClientSession):
        self.session = session
    
    async def store_text(
        self,
        text: str,
        metadata: Dict[str, Any] = None,
        collection: str = None
    ) -> Dict[str, Any]:
        """Store text with automatic embedding through MCP."""
        try:
            result = await self.session.call_tool(
                "qdrant-store",
                {
                    "content": text,
                    "metadata": metadata or {},
                    "collection": collection or "sql_knowledge"
                }
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Qdrant MCP store failed: {e}")
            raise
    
    async def search_similar(
        self,
        query: str,
        limit: int = 10,
        threshold: float = 0.7,
        collection: str = None
    ) -> List[Dict[str, Any]]:
        """Search for similar content through MCP."""
        try:
            result = await self.session.call_tool(
                "qdrant-find",
                {
                    "query": query,
                    "limit": limit,
                    "threshold": threshold,
                    "collection": collection or "sql_knowledge"
                }
            )
            
            return result.get("results", [])
            
        except Exception as e:
            logger.error(f"Qdrant MCP search failed: {e}")
            raise
    
    async def store_sql_query(
        self,
        sql_query: str,
        description: str,
        result_summary: str = None,
        execution_time: float = None,
        success: bool = True
    ) -> Dict[str, Any]:
        """Store SQL query with semantic information."""
        try:
            metadata = {
                "type": "sql_query",
                "sql": sql_query,
                "description": description,
                "success": success
            }
            
            if result_summary:
                metadata["result_summary"] = result_summary
            if execution_time:
                metadata["execution_time"] = execution_time
            
            content = f"SQL Query: {description}\n\nQuery: {sql_query}"
            if result_summary:
                content += f"\n\nResult: {result_summary}"
            
            return await self.store_text(content, metadata)
            
        except Exception as e:
            logger.error(f"Failed to store SQL query: {e}")
            raise
    
    async def find_similar_queries(
        self,
        description: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Find similar SQL queries based on description."""
        try:
            results = await self.search_similar(
                query=f"SQL Query: {description}",
                limit=limit,
                threshold=0.6
            )
            
            # Filter for SQL query type
            sql_results = [
                r for r in results 
                if r.get("metadata", {}).get("type") == "sql_query"
            ]
            
            return sql_results
            
        except Exception as e:
            logger.error(f"Failed to find similar queries: {e}")
            raise
    
    async def store_error_pattern(
        self,
        error_message: str,
        solution: str,
        sql_query: str = None,
        error_type: str = None
    ) -> Dict[str, Any]:
        """Store error pattern for future reference."""
        try:
            metadata = {
                "type": "error_pattern",
                "error_message": error_message,
                "solution": solution,
                "error_type": error_type or "unknown"
            }
            
            if sql_query:
                metadata["sql_query"] = sql_query
            
            content = f"Error: {error_message}\n\nSolution: {solution}"
            if sql_query:
                content += f"\n\nQuery: {sql_query}"
            
            return await self.store_text(content, metadata)
            
        except Exception as e:
            logger.error(f"Failed to store error pattern: {e}")
            raise
    
    async def find_error_solutions(
        self,
        error_message: str,
        limit: int = 3
    ) -> List[Dict[str, Any]]:
        """Find solutions for similar errors."""
        try:
            results = await self.search_similar(
                query=f"Error: {error_message}",
                limit=limit,
                threshold=0.5
            )
            
            # Filter for error pattern type
            error_results = [
                r for r in results 
                if r.get("metadata", {}).get("type") == "error_pattern"
            ]
            
            return error_results
            
        except Exception as e:
            logger.error(f"Failed to find error solutions: {e}")
            raise
    
    async def store_schema_info(
        self,
        table_name: str,
        schema_description: str,
        columns: List[Dict[str, Any]],
        database: str = "mariadb"
    ) -> Dict[str, Any]:
        """Store database schema information."""
        try:
            metadata = {
                "type": "schema_info",
                "table_name": table_name,
                "database": database,
                "columns": columns
            }
            
            content = f"Table: {table_name} ({database})\n\n{schema_description}"
            content += f"\n\nColumns: {', '.join([col['name'] for col in columns])}"
            
            return await self.store_text(content, metadata)
            
        except Exception as e:
            logger.error(f"Failed to store schema info: {e}")
            raise
    
    async def find_relevant_tables(
        self,
        description: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Find relevant tables based on description."""
        try:
            results = await self.search_similar(
                query=description,
                limit=limit,
                threshold=0.6
            )
            
            # Filter for schema info type
            schema_results = [
                r for r in results 
                if r.get("metadata", {}).get("type") == "schema_info"
            ]
            
            return schema_results
            
        except Exception as e:
            logger.error(f"Failed to find relevant tables: {e}")
            raise
    
    async def get_collection_info(self, collection: str = None) -> Dict[str, Any]:
        """Get collection information."""
        try:
            result = await self.session.call_tool(
                "get_collection_info",
                {"collection": collection or "sql_knowledge"}
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            raise
    
    async def test_connection(self) -> bool:
        """Test Qdrant MCP connection."""
        try:
            await self.get_collection_info()
            return True
        except Exception as e:
            logger.error(f"Qdrant MCP connection test failed: {e}")
            return False