"""
LanceDB Service Wrapper
Provides a service interface for LanceDB integration in the Agentic SQL system.
"""

import asyncio
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging

from .runner import SQLEmbeddingService
from .config import settings

logger = logging.getLogger(__name__)


class LanceDBService:
    """
    Service wrapper for LanceDB functionality.
    Provides SQL query caching and business pattern search.
    """
    
    def __init__(self):
        self.embedding_service = None
        self._initialized = False
        
    async def initialize(self):
        """Initialize the LanceDB service."""
        if self._initialized:
            return
            
        try:
            self.embedding_service = SQLEmbeddingService()
            await self.embedding_service.initialize()
            self._initialized = True
            logger.info("LanceDB service initialized successfully")
            
            # Ingest business patterns if available
            try:
                stats = await self.embedding_service.ingest_business_patterns()
                logger.info(f"Ingested business patterns: {stats}")
            except Exception as e:
                logger.warning(f"Failed to ingest business patterns: {e}")
                
        except Exception as e:
            logger.error(f"Failed to initialize LanceDB service: {e}")
            # Don't raise - allow system to continue without vector search
            self._initialized = False
    
    async def search_similar_queries(
        self, 
        query: str, 
        limit: int = 10,
        threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Search for similar SQL queries."""
        if not self._initialized:
            logger.warning("LanceDB service not initialized, returning empty results")
            return []
            
        try:
            results = await self.embedding_service.find_similar_queries(
                query=query,
                threshold=threshold,
                limit=limit
            )
            return results
        except Exception as e:
            logger.error(f"Error searching similar queries: {e}")
            return []
    
    async def store_query(
        self,
        query_id: str,
        sql_query: str,
        business_question: str,
        database: str = "mariadb",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Store a new SQL query with embeddings."""
        if not self._initialized:
            logger.warning("LanceDB service not initialized, cannot store query")
            return None
            
        try:
            query_data = {
                "query_id": query_id,
                "sql_query": sql_query,
                "original_question": business_question,
                "database": database,
                "metadata": metadata or {}
            }
            
            stored_id = await self.embedding_service.store_sql_query(query_data)
            return stored_id
        except Exception as e:
            logger.error(f"Error storing query: {e}")
            return None
    
    async def search_business_patterns(
        self,
        query: str,
        search_type: str = "semantic",
        domain_filter: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search for business patterns."""
        if not self._initialized:
            logger.warning("LanceDB service not initialized, returning empty results")
            return []
            
        try:
            results = await self.embedding_service.search_business_patterns(
                query=query,
                search_type=search_type,
                domain_filter=domain_filter,
                limit=limit
            )
            return results
        except Exception as e:
            logger.error(f"Error searching business patterns: {e}")
            return []
    
    async def get_recommended_patterns(
        self,
        user_role: str,
        complexity_preference: str = "moderate",
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get recommended patterns for a user role."""
        if not self._initialized:
            logger.warning("LanceDB service not initialized, returning empty results")
            return []
            
        try:
            results = await self.embedding_service.get_recommended_patterns(
                user_role=user_role,
                complexity_preference=complexity_preference,
                limit=limit
            )
            return results
        except Exception as e:
            logger.error(f"Error getting recommended patterns: {e}")
            return []
    
    async def health_check(self) -> Dict[str, Any]:
        """Check LanceDB service health."""
        if not self._initialized:
            return {
                "status": "unavailable",
                "initialized": False,
                "message": "LanceDB service not initialized"
            }
            
        try:
            health = await self.embedding_service.health_check()
            return {
                "status": "healthy",
                "initialized": True,
                **health
            }
        except Exception as e:
            return {
                "status": "error",
                "initialized": self._initialized,
                "error": str(e)
            }
    
    async def cleanup(self):
        """Clean up resources."""
        if self.embedding_service:
            await self.embedding_service.cleanup()
        self._initialized = False


# Global service instance
_lancedb_service = None


async def get_lancedb_service() -> LanceDBService:
    """Get or create the global LanceDB service instance."""
    global _lancedb_service
    
    if _lancedb_service is None:
        _lancedb_service = LanceDBService()
        await _lancedb_service.initialize()
    
    return _lancedb_service