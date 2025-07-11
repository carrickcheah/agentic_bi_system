"""
Direct Database Manager - Alternative to MCP for programmatic access.
"""

import asyncio
import aiomysql
import asyncpg
from typing import Dict, Any, Optional, List
import logging

try:
    from ..config import settings
except ImportError:
    from config import settings

logger = logging.getLogger(__name__)


class DirectDatabaseManager:
    """Direct database connections without MCP overhead."""
    
    def __init__(self):
        self.connections: Dict[str, Any] = {}
        self._initialized = False
        
        # Database client wrappers
        self.mariadb = None
        self.postgres = None
    
    async def initialize(self, services: Optional[List[str]] = None):
        """Initialize direct database connections."""
        if self._initialized and services is None:
            return
        
        try:
            if services is None or "mariadb" in services:
                await self._init_mariadb()
                
            if services is None or "postgres" in services:
                await self._init_postgres()
            
            self._initialized = True
            logger.info(f"Database manager initialized with services: {list(self.connections.keys())}")
            
        except Exception as e:
            logger.error(f"Failed to initialize database connections: {e}")
            # Continue without database - don't block the application
            self._initialized = True
    
    async def _init_mariadb(self):
        """Initialize MariaDB connection pool."""
        try:
            self.connections["mariadb"] = await aiomysql.create_pool(
                host=settings.mariadb_host,
                port=settings.mariadb_port,
                user=settings.mariadb_user,
                password=settings.mariadb_password,
                db=settings.mariadb_database,
                minsize=1,
                maxsize=10,
                echo=False,
                autocommit=True
            )
            
            # Create wrapper
            self.mariadb = MariaDBDirectClient(self.connections["mariadb"])
            logger.info("MariaDB connection pool created")
            
        except Exception as e:
            logger.error(f"Failed to connect to MariaDB: {e}")
    
    async def _init_postgres(self):
        """Initialize PostgreSQL connection pool."""
        try:
            self.connections["postgres"] = await asyncpg.create_pool(
                settings.postgres_url,
                min_size=1,
                max_size=10
            )
            
            # Create wrapper
            self.postgres = PostgreSQLDirectClient(self.connections["postgres"])
            logger.info("PostgreSQL connection pool created")
            
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
    
    async def close(self):
        """Close all database connections."""
        for name, pool in self.connections.items():
            try:
                if hasattr(pool, 'close'):
                    pool.close()
                    await pool.wait_closed()
                elif hasattr(pool, 'terminate'):
                    await pool.terminate()
                logger.info(f"Closed {name} connection pool")
            except Exception as e:
                logger.error(f"Error closing {name} pool: {e}")
        
        self.connections.clear()
        self._initialized = False


class MariaDBDirectClient:
    """Direct MariaDB client wrapper."""
    
    def __init__(self, pool):
        self.pool = pool
    
    async def execute_query(self, query: str) -> Dict[str, Any]:
        """Execute a query and return results."""
        async with self.pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cursor:
                await cursor.execute(query)
                rows = await cursor.fetchall()
                return {
                    "success": True,
                    "data": rows,
                    "row_count": cursor.rowcount
                }
    
    async def get_schema(self) -> Dict[str, Any]:
        """Get database schema information."""
        query = """
        SELECT 
            TABLE_NAME,
            COLUMN_NAME,
            DATA_TYPE,
            IS_NULLABLE,
            COLUMN_KEY
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = DATABASE()
        ORDER BY TABLE_NAME, ORDINAL_POSITION
        """
        return await self.execute_query(query)


class PostgreSQLDirectClient:
    """Direct PostgreSQL client wrapper."""
    
    def __init__(self, pool):
        self.pool = pool
    
    async def execute_query(self, query: str) -> Dict[str, Any]:
        """Execute a query and return results."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query)
            return {
                "success": True,
                "data": [dict(row) for row in rows],
                "row_count": len(rows)
            }
    
    async def get_schema(self) -> Dict[str, Any]:
        """Get database schema information."""
        query = """
        SELECT 
            table_name,
            column_name,
            data_type,
            is_nullable,
            column_default
        FROM information_schema.columns
        WHERE table_schema = 'public'
        ORDER BY table_name, ordinal_position
        """
        return await self.execute_query(query)


# Global instance
db_manager = DirectDatabaseManager()