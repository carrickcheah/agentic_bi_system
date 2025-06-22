"""
Memory Management

Handles session storage, context management, and result caching using PostgreSQL.
"""

import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from ..utils.logging import logger
from .connections import DatabaseManager


class MemoryManager:
    """Manages session memory and caching using PostgreSQL."""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
    
    async def create_session(self, session_data: Dict[str, Any]):
        """Create a new user session."""
        try:
            query = """
            INSERT INTO sessions (session_id, user_id, context, created_at, updated_at)
            VALUES ($1, $2, $3, $4, $5)
            """
            
            async with self.db_manager.get_postgres_connection() as conn:
                await conn.execute(
                    query,
                    session_data["session_id"],
                    session_data["user_id"],
                    json.dumps(session_data["context"]),
                    session_data["created_at"],
                    session_data["updated_at"]
                )
            
            logger.info(f"Session created: {session_data['session_id']}")
            
        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            raise
    
    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session by ID."""
        try:
            query = """
            SELECT session_id, user_id, context, created_at, updated_at
            FROM sessions
            WHERE session_id = $1
            """
            
            async with self.db_manager.get_postgres_connection() as conn:
                result = await conn.execute(query, session_id)
                row = result.fetchone()
                
                if not row:
                    return None
                
                return {
                    "session_id": row[0],
                    "user_id": row[1],
                    "context": json.loads(row[2]) if row[2] else {},
                    "created_at": row[3],
                    "updated_at": row[4]
                }
                
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
            query = """
            UPDATE sessions
            SET context = $1, updated_at = $2
            WHERE session_id = $3
            RETURNING session_id, user_id, context, created_at, updated_at
            """
            
            async with self.db_manager.get_postgres_connection() as conn:
                result = await conn.execute(
                    query,
                    json.dumps(context),
                    datetime.utcnow(),
                    session_id
                )
                row = result.fetchone()
                
                if not row:
                    return None
                
                return {
                    "session_id": row[0],
                    "user_id": row[1],
                    "context": json.loads(row[2]) if row[2] else {},
                    "created_at": row[3],
                    "updated_at": row[4]
                }
                
        except Exception as e:
            logger.error(f"Failed to update session context {session_id}: {e}")
            return None
    
    async def get_session_history(
        self, 
        session_id: str, 
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get investigation history for a session."""
        try:
            query = """
            SELECT investigation_id, query, status, created_at, results
            FROM investigations
            WHERE session_id = $1
            ORDER BY created_at DESC
            LIMIT $2
            """
            
            async with self.db_manager.get_postgres_connection() as conn:
                result = await conn.execute(query, session_id, limit)
                
                history = []
                for row in result.fetchall():
                    history.append({
                        "investigation_id": row[0],
                        "query": row[1],
                        "status": row[2],
                        "created_at": row[3],
                        "results": json.loads(row[4]) if row[4] else None
                    })
                
                return history
                
        except Exception as e:
            logger.error(f"Failed to get session history {session_id}: {e}")
            return []
    
    async def cache_result(
        self,
        session_id: str,
        key: str,
        data: Dict[str, Any],
        ttl: int = 1800  # 30 minutes default
    ):
        """Cache investigation result."""
        try:
            expires_at = datetime.utcnow() + timedelta(seconds=ttl)
            
            query = """
            INSERT INTO cached_results (session_id, cache_key, data, created_at, expires_at)
            VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT (session_id, cache_key)
            DO UPDATE SET data = $3, created_at = $4, expires_at = $5
            """
            
            async with self.db_manager.get_postgres_connection() as conn:
                await conn.execute(
                    query,
                    session_id,
                    key,
                    json.dumps(data),
                    datetime.utcnow(),
                    expires_at
                )
            
            logger.info(f"Result cached: {session_id}/{key}")
            
        except Exception as e:
            logger.error(f"Failed to cache result {session_id}/{key}: {e}")
            raise
    
    async def get_cached_result(
        self,
        session_id: str,
        key: str
    ) -> Optional[Dict[str, Any]]:
        """Get cached result if not expired."""
        try:
            query = """
            SELECT data
            FROM cached_results
            WHERE session_id = $1 AND cache_key = $2 AND expires_at > $3
            """
            
            async with self.db_manager.get_postgres_connection() as conn:
                result = await conn.execute(
                    query, 
                    session_id, 
                    key, 
                    datetime.utcnow()
                )
                row = result.fetchone()
                
                if not row:
                    return None
                
                return json.loads(row[0])
                
        except Exception as e:
            logger.error(f"Failed to get cached result {session_id}/{key}: {e}")
            return None
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete session and all associated data."""
        try:
            async with self.db_manager.get_postgres_connection() as conn:
                # Delete cached results
                await conn.execute(
                    "DELETE FROM cached_results WHERE session_id = $1",
                    session_id
                )
                
                # Delete session
                result = await conn.execute(
                    "DELETE FROM sessions WHERE session_id = $1",
                    session_id
                )
                
                return result.rowcount > 0
                
        except Exception as e:
            logger.error(f"Failed to delete session {session_id}: {e}")
            return False
    
    async def cleanup_expired_cache(self):
        """Clean up expired cache entries."""
        try:
            query = "DELETE FROM cached_results WHERE expires_at < $1"
            
            async with self.db_manager.get_postgres_connection() as conn:
                result = await conn.execute(query, datetime.utcnow())
                
                if result.rowcount > 0:
                    logger.info(f"Cleaned up {result.rowcount} expired cache entries")
                
        except Exception as e:
            logger.error(f"Failed to cleanup expired cache: {e}")