"""
Database Connection Management

Handles connection pools for MariaDB and PostgreSQL databases.
"""

import asyncio
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager

import asyncpg
import aiomysql
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import NullPool

from ..config import settings
from ..utils.logging import logger
from ..utils.monitoring import PerformanceTimer


class DatabaseManager:
    """Manages database connections and provides connection pooling."""
    
    def __init__(self):
        self.mariadb_engine = None
        self.postgres_engine = None
        self.postgres_session_factory = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize all database connections."""
        if self._initialized:
            return
        
        try:
            # Initialize MariaDB connection
            if hasattr(settings, 'mariadb_url') and settings.mariadb_url:
                await self._init_mariadb()
            
            # Initialize PostgreSQL connection
            if hasattr(settings, 'postgres_url') and settings.postgres_url:
                await self._init_postgres()
            
            self._initialized = True
            logger.info("Database connections initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database connections: {e}")
            raise
    
    async def _init_mariadb(self):
        """Initialize MariaDB connection pool."""
        try:
            # Convert to async MySQL URL if needed
            mariadb_url = settings.mariadb_url
            if mariadb_url.startswith('mysql://'):
                mariadb_url = mariadb_url.replace('mysql://', 'mysql+aiomysql://', 1)
            elif not mariadb_url.startswith('mysql+aiomysql://'):
                mariadb_url = f"mysql+aiomysql://{mariadb_url}"
            
            self.mariadb_engine = create_async_engine(
                mariadb_url,
                poolclass=NullPool,  # Use NullPool for better connection management
                echo=settings.debug
            )
            
            # Test connection
            async with self.mariadb_engine.begin() as conn:
                await conn.execute("SELECT 1")
            
            logger.info("MariaDB connection pool initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize MariaDB connection: {e}")
            raise
    
    async def _init_postgres(self):
        """Initialize PostgreSQL connection pool."""
        try:
            # Ensure asyncpg URL format
            postgres_url = settings.postgres_url
            if postgres_url.startswith('postgresql://'):
                postgres_url = postgres_url.replace('postgresql://', 'postgresql+asyncpg://', 1)
            elif not postgres_url.startswith('postgresql+asyncpg://'):
                postgres_url = f"postgresql+asyncpg://{postgres_url}"
            
            self.postgres_engine = create_async_engine(
                postgres_url,
                poolclass=NullPool,
                echo=settings.debug
            )
            
            # Create session factory
            self.postgres_session_factory = async_sessionmaker(
                self.postgres_engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            # Test connection
            async with self.postgres_engine.begin() as conn:
                await conn.execute("SELECT 1")
            
            logger.info("PostgreSQL connection pool initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL connection: {e}")
            raise
    
    @asynccontextmanager
    async def get_mariadb_connection(self):
        """Get MariaDB database connection."""
        if not self.mariadb_engine:
            raise RuntimeError("MariaDB engine not initialized")
        
        async with self.mariadb_engine.begin() as conn:
            yield conn
    
    @asynccontextmanager
    async def get_postgres_connection(self):
        """Get PostgreSQL database connection."""
        if not self.postgres_engine:
            raise RuntimeError("PostgreSQL engine not initialized")
        
        async with self.postgres_engine.begin() as conn:
            yield conn
    
    @asynccontextmanager
    async def get_postgres_session(self):
        """Get PostgreSQL session for ORM operations."""
        if not self.postgres_session_factory:
            raise RuntimeError("PostgreSQL session factory not initialized")
        
        async with self.postgres_session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
    
    async def execute_mariadb_query(
        self, 
        query: str, 
        parameters: Dict[str, Any] = None,
        timeout: int = 30
    ):
        """Execute query on MariaDB with timeout."""
        with PerformanceTimer(f"MariaDB query execution"):
            async with self.get_mariadb_connection() as conn:
                try:
                    result = await asyncio.wait_for(
                        conn.execute(query, parameters or {}),
                        timeout=timeout
                    )
                    return result
                except asyncio.TimeoutError:
                    raise Exception(f"Query timeout after {timeout} seconds")
    
    async def execute_postgres_query(
        self, 
        query: str, 
        parameters: Dict[str, Any] = None,
        timeout: int = 30
    ):
        """Execute query on PostgreSQL with timeout."""
        with PerformanceTimer(f"PostgreSQL query execution"):
            async with self.get_postgres_connection() as conn:
                try:
                    result = await asyncio.wait_for(
                        conn.execute(query, parameters or {}),
                        timeout=timeout
                    )
                    return result
                except asyncio.TimeoutError:
                    raise Exception(f"Query timeout after {timeout} seconds")
    
    def is_healthy(self) -> bool:
        """Check if database connections are healthy."""
        return self._initialized and (
            self.mariadb_engine is not None or 
            self.postgres_engine is not None
        )
    
    async def close(self):
        """Close all database connections."""
        try:
            if self.mariadb_engine:
                await self.mariadb_engine.dispose()
                logger.info("MariaDB connection pool closed")
            
            if self.postgres_engine:
                await self.postgres_engine.dispose()
                logger.info("PostgreSQL connection pool closed")
            
            self._initialized = False
            
        except Exception as e:
            logger.error(f"Error closing database connections: {e}")


    # Investigation management methods
    async def save_investigation(self, investigation):
        """Save investigation to database."""
        # Implementation will be added when we have the schema
        pass
    
    async def get_investigation(self, investigation_id: str):
        """Get investigation by ID."""
        # Implementation will be added when we have the schema
        pass
    
    async def list_investigations(self, user_id: str = None, limit: int = 20, offset: int = 0):
        """List investigations."""
        # Implementation will be added when we have the schema
        pass
    
    async def cancel_investigation(self, investigation_id: str):
        """Cancel investigation."""
        # Implementation will be added when we have the schema
        pass
    
    async def update_investigation_error(self, investigation_id: str, error: str):
        """Update investigation with error."""
        # Implementation will be added when we have the schema
        pass