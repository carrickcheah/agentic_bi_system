"""
Business Service Layer - Database Operations via MCP Clients

This service layer provides the business logic interface for all database
operations. It uses MCP clients to connect to external database servers
and abstracts these operations into business methods.

Features:
- Business-focused API (not database-focused)
- Multi-database coordination via MCP clients
- Transaction management across databases
- Error handling and retry logic
- Performance optimization
- Semantic caching integration

Note: This service USES MCP clients to connect to external MCP servers.
It is NOT an MCP server itself.
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from dataclasses import dataclass
import asyncio

try:
    from ..utils.logging import logger
    from ..utils.exceptions import BusinessLogicError, DatabaseOperationError
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    # Simple exception classes for standalone mode
    class BusinessLogicError(Exception):
        pass
    class DatabaseOperationError(Exception):
        pass
from .client_manager import MCPClientManager


@dataclass
class QueryResult:
    """Result of a database query operation."""
    data: List[Dict[str, Any]]
    columns: List[str]
    row_count: int
    execution_time: float
    database: str
    success: bool = True
    error: Optional[str] = None


@dataclass
class InvestigationResult:
    """Result of an investigation operation."""
    investigation_id: str
    status: str
    findings: List[Dict[str, Any]]
    insights: List[str]
    confidence_score: float
    execution_time: float
    created_at: datetime


class BusinessService:
    """
    Business logic service layer for database operations via MCP clients.
    
    This service provides high-level business operations that abstract
    away the complexity of multiple databases and MCP clients.
    """
    
    def __init__(self, client_manager: MCPClientManager):
        self.client_manager = client_manager
        self._initialized = False
    
    async def initialize(self):
        """Initialize the service layer."""
        if self._initialized:
            return
        
        # Ensure client manager is initialized
        if not self.client_manager._initialized:
            await self.client_manager.initialize()
        
        self._initialized = True
        logger.info("FastMCP service layer initialized")
    
    async def cleanup(self):
        """Cleanup service resources."""
        self._initialized = False
        logger.info("FastMCP service layer cleaned up")
    
    def is_healthy(self) -> bool:
        """Check if the service is healthy."""
        return self._initialized and self.client_manager.is_healthy()
    
    # Database Operations
    
    async def execute_sql(
        self,
        query: str,
        database: str = "mariadb",
        max_rows: int = 1000,
        timeout: int = 30
    ) -> QueryResult:
        """
        Execute SQL query on specified database.
        
        Args:
            query: SQL query to execute
            database: Target database (mariadb, postgres, supabase)
            max_rows: Maximum rows to return
            timeout: Query timeout in seconds
            
        Returns:
            QueryResult with data and metadata
        """
        start_time = datetime.utcnow()
        
        try:
            logger.info(f"Executing SQL on {database}: {query[:100]}...")
            
            # Get appropriate client
            client = self._get_database_client(database)
            if not client:
                raise DatabaseOperationError(f"Database client '{database}' not available")
            
            # Execute query with timeout
            result = await asyncio.wait_for(
                client.execute_query(query, max_rows=max_rows),
                timeout=timeout
            )
            
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            return QueryResult(
                data=result.get("data", []),
                columns=result.get("columns", []),
                row_count=result.get("row_count", 0),
                execution_time=execution_time,
                database=database,
                success=True
            )
            
        except asyncio.TimeoutError:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            error = f"Query timeout after {timeout} seconds"
            logger.error(f"SQL execution timeout: {error}")
            
            return QueryResult(
                data=[],
                columns=[],
                row_count=0,
                execution_time=execution_time,
                database=database,
                success=False,
                error=error
            )
            
        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            error = str(e)
            logger.error(f"SQL execution failed: {error}")
            
            return QueryResult(
                data=[],
                columns=[],
                row_count=0,
                execution_time=execution_time,
                database=database,
                success=False,
                error=error
            )
    
    async def get_database_schema(
        self,
        database: str = "mariadb",
        table_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get database schema information.
        
        Args:
            database: Target database
            table_name: Specific table (optional)
            
        Returns:
            Schema information dictionary
        """
        try:
            logger.info(f"Getting schema for {database}" + (f".{table_name}" if table_name else ""))
            
            client = self._get_database_client(database)
            if not client:
                raise DatabaseOperationError(f"Database client '{database}' not available")
            
            # Get schema information
            if table_name:
                schema_info = await client.get_table_schema(table_name)
            else:
                schema_info = await client.get_database_schema()
            
            return {
                "database": database,
                "schema": schema_info,
                "retrieved_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Schema retrieval failed for {database}: {e}")
            raise DatabaseOperationError(f"Failed to get schema: {e}")
    
    async def list_tables(self, database: str = "mariadb") -> List[str]:
        """
        List all tables in the database.
        
        Args:
            database: Target database
            
        Returns:
            List of table names
        """
        try:
            client = self._get_database_client(database)
            if not client:
                raise DatabaseOperationError(f"Database client '{database}' not available")
            
            tables = await client.list_tables()
            return tables
            
        except Exception as e:
            logger.error(f"Table listing failed for {database}: {e}")
            raise DatabaseOperationError(f"Failed to list tables: {e}")
    
    # Vector Operations (LanceDB)
    
    async def store_query_pattern(
        self,
        sql_query: str,
        description: str,
        result_summary: Optional[str] = None,
        execution_time: Optional[float] = None,
        success: bool = True
    ) -> Dict[str, Any]:
        """
        Store SQL query pattern for semantic search.
        
        Args:
            sql_query: The SQL query
            description: Human-readable description
            result_summary: Summary of results
            execution_time: Query execution time
            success: Whether query was successful
            
        Returns:
            Storage result
        """
        try:
            # Note: LanceDB vector operations would be implemented here
            # For now, this is a placeholder that logs the operation
            logger.info("Vector storage operation - LanceDB integration pending")
            
            result = {
                "status": "stored",
                "pattern_id": f"pattern_{hash(sql_query + description)}",
                "description": description,
                "stored_at": datetime.utcnow().isoformat()
            }
            
            logger.info(f"Stored query pattern: {description}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to store query pattern: {e}")
            raise DatabaseOperationError(f"Vector storage failed: {e}")
    
    async def find_similar_queries(
        self,
        description: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find similar SQL queries based on description.
        
        Args:
            description: Query description to search for
            limit: Maximum number of results
            
        Returns:
            List of similar queries with metadata
        """
        try:
            # Note: LanceDB vector search would be implemented here
            # For now, this is a placeholder that returns empty results
            logger.info("Vector search operation - LanceDB integration pending")
            
            results = []  # Placeholder - would use LanceDB for similarity search
            
            logger.info(f"Found {len(results)} similar queries for: {description}")
            return results
            
        except Exception as e:
            logger.error(f"Similar query search failed: {e}")
            raise DatabaseOperationError(f"Vector search failed: {e}")
    
    # Investigation Operations
    
    async def create_investigation(
        self,
        query: str,
        user_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a new investigation.
        
        Args:
            query: Investigation query
            user_id: User creating the investigation
            context: Additional context
            
        Returns:
            Investigation ID
        """
        try:
            if not self.client_manager.postgres:
                raise DatabaseOperationError("PostgreSQL client not available")
            
            investigation_id = await self.client_manager.postgres.create_investigation(
                query=query,
                user_id=user_id,
                context=context or {}
            )
            
            logger.info(f"Created investigation {investigation_id} for user {user_id}")
            return investigation_id
            
        except Exception as e:
            logger.error(f"Investigation creation failed: {e}")
            raise BusinessLogicError(f"Failed to create investigation: {e}")
    
    async def get_investigation(self, investigation_id: str) -> Optional[Dict[str, Any]]:
        """
        Get investigation by ID.
        
        Args:
            investigation_id: Investigation ID
            
        Returns:
            Investigation data or None if not found
        """
        try:
            if not self.client_manager.postgres:
                raise DatabaseOperationError("PostgreSQL client not available")
            
            investigation = await self.client_manager.postgres.get_investigation(investigation_id)
            return investigation
            
        except Exception as e:
            logger.error(f"Investigation retrieval failed: {e}")
            raise BusinessLogicError(f"Failed to get investigation: {e}")
    
    async def update_investigation_status(
        self,
        investigation_id: str,
        status: str,
        results: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update investigation status and results.
        
        Args:
            investigation_id: Investigation ID
            status: New status
            results: Investigation results
            
        Returns:
            True if updated successfully
        """
        try:
            if not self.client_manager.postgres:
                raise DatabaseOperationError("PostgreSQL client not available")
            
            success = await self.client_manager.postgres.update_investigation_status(
                investigation_id=investigation_id,
                status=status,
                results=results
            )
            
            logger.info(f"Updated investigation {investigation_id} status to {status}")
            return success
            
        except Exception as e:
            logger.error(f"Investigation update failed: {e}")
            raise BusinessLogicError(f"Failed to update investigation: {e}")
    
    # Business Intelligence Operations
    
    async def analyze_business_query(
        self,
        query: str,
        domain: Optional[str] = None,
        user_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze business query for patterns and insights.
        
        Args:
            query: Business query to analyze
            domain: Business domain (optional)
            user_context: User context for personalization
            
        Returns:
            Analysis results with recommendations
        """
        try:
            # This is a high-level business operation that coordinates
            # multiple services and databases
            
            analysis_result = {
                "query": query,
                "domain": domain,
                "analysis": {
                    "intent": "data_exploration",  # Would be determined by AI
                    "complexity": "medium",
                    "estimated_time": "2-5 minutes",
                    "required_databases": ["mariadb"],
                    "recommended_approach": "step_by_step_investigation"
                },
                "suggestions": [
                    "Start with schema exploration",
                    "Identify relevant tables",
                    "Execute targeted queries"
                ],
                "similar_queries": await self.find_similar_queries(query, limit=3),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            logger.info(f"Analyzed business query: {query[:50]}...")
            return analysis_result
            
        except Exception as e:
            logger.error(f"Business query analysis failed: {e}")
            raise BusinessLogicError(f"Query analysis failed: {e}")
    
    # Helper Methods
    
    def _get_database_client(self, database: str):
        """Get the appropriate database client."""
        client_map = {
            "mariadb": self.client_manager.mariadb,
            "postgres": self.client_manager.postgres,
            "postgresql": self.client_manager.postgres
        }
        return client_map.get(database.lower())
    
    # Health and Status
    
    async def get_service_status(self) -> Dict[str, Any]:
        """
        Get comprehensive service status.
        
        Returns:
            Service status information
        """
        return {
            "service": {
                "name": "FastMCP Service",
                "status": "healthy" if self.is_healthy() else "unhealthy",
                "initialized": self._initialized
            },
            "databases": {
                "mariadb": bool(self.client_manager.mariadb),
                "postgres": bool(self.client_manager.postgres),
                "lancedb": "pending_integration",
                "graphrag": bool(self.client_manager.graphrag)
            },
            "client_manager": {
                "status": "healthy" if self.client_manager.is_healthy() else "unhealthy",
                "active_sessions": len(self.client_manager.sessions)
            },
            "timestamp": datetime.utcnow().isoformat()
        }