"""
PostgreSQL MCP Client

Provides PostgreSQL operations through MCP protocol.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from mcp.client.session import ClientSession

try:
    from ..database.models import QueryResult, TableSchema, ColumnInfo
    from ..utils.logging import logger
except ImportError:
    # Simple standalone versions
    from dataclasses import dataclass
    from typing import Any
    import logging
    logger = logging.getLogger(__name__)
    
    @dataclass
    class QueryResult:
        data: List[Dict[str, Any]]
        columns: List[str]
        row_count: int
        execution_time: float
        database: str
        success: bool = True
        error: Optional[str] = None
    
    @dataclass
    class ColumnInfo:
        name: str
        type: str
        nullable: bool
        default: Any = None
    
    @dataclass  
    class TableSchema:
        table_name: str
        columns: List[ColumnInfo]


class PostgreSQLClient:
    """PostgreSQL operations through MCP for agent memory."""
    
    def __init__(self, session: ClientSession):
        self.session = session
    
    async def execute_query(
        self,
        query: str,
        parameters: Dict[str, Any] = None,
        max_rows: int = 1000,
        timeout: int = 30
    ) -> QueryResult:
        """Execute SQL query on PostgreSQL through MCP."""
        try:
            # Call MCP tool for query execution
            result = await self.session.call_tool(
                "execute_sql",
                {
                    "query": query,
                    "parameters": parameters or {},
                    "max_rows": max_rows,
                    "timeout": timeout
                }
            )
            
            # Convert MCP result to QueryResult model
            return QueryResult(
                columns=result.get("columns", []),
                rows=result.get("rows", []),
                row_count=result.get("row_count", 0),
                execution_time=result.get("execution_time", 0.0),
                query=query
            )
            
        except Exception as e:
            logger.error(f"PostgreSQL MCP query failed: {e}")
            raise
    
    # Session Management
    async def create_session(self, session_data: Dict[str, Any]):
        """Create a new user session."""
        try:
            query = """
            INSERT INTO sessions (session_id, user_id, context, created_at, updated_at)
            VALUES ($1, $2, $3, $4, $5)
            """
            
            now = datetime.utcnow()
            await self.execute_query(query, {
                "session_id": session_data["session_id"],
                "user_id": session_data["user_id"],
                "context": session_data.get("context", {}),
                "created_at": now,
                "updated_at": now
            })
            
            logger.info(f"Session created: {session_data['session_id']}")
            
        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            raise
    
    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session by ID."""
        try:
            query = "SELECT * FROM sessions WHERE session_id = $1"
            result = await self.execute_query(query, {"session_id": session_id})
            
            return result.rows[0] if result.rows else None
            
        except Exception as e:
            logger.error(f"Failed to get session: {e}")
            raise
    
    async def update_session(self, session_id: str, context: Dict[str, Any]):
        """Update session context."""
        try:
            query = """
            UPDATE sessions 
            SET context = $1, updated_at = $2 
            WHERE session_id = $3
            """
            
            await self.execute_query(query, {
                "context": context,
                "updated_at": datetime.utcnow(),
                "session_id": session_id
            })
            
        except Exception as e:
            logger.error(f"Failed to update session: {e}")
            raise
    
    # Memory Management
    async def store_short_term_memory(
        self,
        session_id: str,
        content: str,
        memory_type: str = "query_result"
    ):
        """Store short-term memory."""
        try:
            query = """
            INSERT INTO short_term_memory (session_id, content, memory_type, created_at)
            VALUES ($1, $2, $3, $4)
            """
            
            await self.execute_query(query, {
                "session_id": session_id,
                "content": content,
                "memory_type": memory_type,
                "created_at": datetime.utcnow()
            })
            
        except Exception as e:
            logger.error(f"Failed to store short-term memory: {e}")
            raise
    
    async def store_long_term_memory(
        self,
        agent_id: str,
        content: str,
        embedding_id: Optional[str] = None,
        importance_score: float = 0.5
    ):
        """Store long-term memory."""
        try:
            query = """
            INSERT INTO long_term_memory (agent_id, content, embedding_id, importance_score, created_at)
            VALUES ($1, $2, $3, $4, $5)
            """
            
            await self.execute_query(query, {
                "agent_id": agent_id,
                "content": content,
                "embedding_id": embedding_id,
                "importance_score": importance_score,
                "created_at": datetime.utcnow()
            })
            
        except Exception as e:
            logger.error(f"Failed to store long-term memory: {e}")
            raise
    
    async def get_recent_memories(
        self,
        session_id: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get recent memories for a session."""
        try:
            query = """
            SELECT * FROM short_term_memory 
            WHERE session_id = $1 
            ORDER BY created_at DESC 
            LIMIT $2
            """
            
            result = await self.execute_query(query, {
                "session_id": session_id,
                "limit": limit
            })
            
            return result.rows
            
        except Exception as e:
            logger.error(f"Failed to get recent memories: {e}")
            raise
    
    # Investigation Management
    async def save_investigation(self, investigation):
        """Save investigation to database."""
        # Implementation will be added when we have the schema
        logger.info("Investigation save not yet implemented")
        pass
    
    async def get_investigation(self, investigation_id: str):
        """Get investigation by ID."""
        # Implementation will be added when we have the schema
        logger.info("Investigation get not yet implemented")
        pass
    
    async def list_investigations(self, user_id: str = None, limit: int = 20, offset: int = 0):
        """List investigations."""
        # Implementation will be added when we have the schema
        logger.info("Investigation list not yet implemented")
        pass
    
    async def cancel_investigation(self, investigation_id: str):
        """Cancel investigation."""
        # Implementation will be added when we have the schema
        logger.info("Investigation cancel not yet implemented")
        pass
    
    async def update_investigation_error(self, investigation_id: str, error: str):
        """Update investigation with error."""
        # Implementation will be added when we have the schema
        logger.info("Investigation error update not yet implemented")
        pass
    
    async def test_connection(self) -> bool:
        """Test PostgreSQL MCP connection."""
        try:
            await self.execute_query("SELECT 1 as test")
            return True
        except Exception as e:
            logger.error(f"PostgreSQL MCP connection test failed: {e}")
            return False