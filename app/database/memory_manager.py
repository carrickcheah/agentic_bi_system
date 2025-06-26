"""
Memory Management

Handles session storage, context management, and result caching using PostgreSQL MCP.
"""

import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from ..utils.logging import logger
from ..fastmcp.postgres_client import PostgreSQLClient


class MemoryManager:
    """Manages session memory and caching using PostgreSQL MCP."""
    
    def __init__(self, postgres_client: PostgreSQLClient):
        self.postgres_client = postgres_client
    
    async def create_session(self, session_data: Dict[str, Any]):
        """Create a new user session."""
        try:
            await self.postgres_client.create_session(session_data)
            logger.info(f"Session created: {session_data['session_id']}")
            
        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            raise
    
    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session by ID."""
        try:
            return await self.postgres_client.get_session(session_id)
                
        except Exception as e:
            logger.error(f"Failed to get session {session_id}: {e}")
            return None
    
    async def update_session_context(
        self, 
        session_id: str, 
        context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Update session context."""
        try:
            await self.postgres_client.update_session(session_id, context)
            return await self.get_session(session_id)
                
        except Exception as e:
            logger.error(f"Failed to update session {session_id}: {e}")
            return None
    
    async def store_context(
        self,
        session_id: str,
        context_type: str,
        content: str,
        metadata: Dict[str, Any] = None
    ):
        """Store context information in short-term memory."""
        try:
            await self.postgres_client.store_short_term_memory(
                session_id=session_id,
                content=content,
                memory_type=context_type
            )
            
            logger.debug(f"Context stored for session {session_id}")
            
        except Exception as e:
            logger.error(f"Failed to store context: {e}")
            raise
    
    async def get_recent_context(
        self,
        session_id: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get recent context for a session."""
        try:
            return await self.postgres_client.get_recent_memories(
                session_id=session_id,
                limit=limit
            )
            
        except Exception as e:
            logger.error(f"Failed to get recent context: {e}")
            return []
    
    async def store_long_term_knowledge(
        self,
        agent_id: str,
        content: str,
        importance_score: float = 0.5,
        embedding_id: Optional[str] = None
    ):
        """Store important information in long-term memory."""
        try:
            await self.postgres_client.store_long_term_memory(
                agent_id=agent_id,
                content=content,
                embedding_id=embedding_id,
                importance_score=importance_score
            )
            
            logger.debug(f"Long-term knowledge stored for agent {agent_id}")
            
        except Exception as e:
            logger.error(f"Failed to store long-term knowledge: {e}")
            raise
    
    async def cleanup_expired_sessions(self, hours: int = 24):
        """Clean up expired sessions."""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            
            result = await self.postgres_client.execute_query(
                "DELETE FROM sessions WHERE updated_at < $1",
                {"cutoff_time": cutoff_time}
            )
            
            logger.info(f"Cleaned up expired sessions: {result.row_count} removed")
            
        except Exception as e:
            logger.error(f"Failed to cleanup expired sessions: {e}")
    
    async def get_session_stats(self) -> Dict[str, Any]:
        """Get session statistics."""
        try:
            total_result = await self.postgres_client.execute_query(
                "SELECT COUNT(*) as total FROM sessions"
            )
            
            active_result = await self.postgres_client.execute_query(
                "SELECT COUNT(*) as active FROM sessions WHERE updated_at > $1",
                {"cutoff": datetime.utcnow() - timedelta(hours=1)}
            )
            
            return {
                "total_sessions": total_result.rows[0]["total"] if total_result.rows else 0,
                "active_sessions": active_result.rows[0]["active"] if active_result.rows else 0,
                "timestamp": datetime.utcnow()
            }
            
        except Exception as e:
            logger.error(f"Failed to get session stats: {e}")
            return {"total_sessions": 0, "active_sessions": 0, "timestamp": datetime.utcnow()}