"""
PostgreSQL Cache Client - Tier 1b Cache

Manages PostgreSQL-based hybrid caching system with 100ms target response time.

Two-level cache:
1. Personal Cache: User-specific insights with permissions
2. Organizational Cache: Team-shared business intelligence

Features:
- Intelligent TTL based on data volatility
- Permission-aware caching
- Cross-user knowledge sharing
- Personal vs organizational cache separation
"""

import asyncio
import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

from ..utils.logging import logger
from ..mcp.postgres_client import PostgreSQLClient


class PostgreSQLCacheClient:
    """
    PostgreSQL-based caching client for hybrid personal + organizational caching.
    
    Provides fast access to business intelligence insights while respecting
    user permissions and enabling organizational knowledge sharing.
    """
    
    def __init__(self):
        self.postgres_client: Optional[PostgreSQLClient] = None
        self.default_ttl = 3600  # 1 hour
        self.similarity_threshold = 0.8
        
    async def initialize(self):
        """Initialize PostgreSQL cache client."""
        try:
            # Initialize PostgreSQL client connection
            # This will be injected from the MCP client manager
            logger.info("PostgreSQL cache client initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL cache: {e}")
            raise
    
    def set_postgres_client(self, client: PostgreSQLClient):
        """Set the PostgreSQL client instance."""
        self.postgres_client = client
    
    async def get_personal_insights(
        self,
        user_id: str,
        semantic_hash: str,
        business_domain: str
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve user's personal cached insights.
        
        Args:
            user_id: User identifier
            semantic_hash: Semantic hash of the business question
            business_domain: Business domain
            
        Returns:
            Personal cached insights or None if not found
        """
        try:
            if not self.postgres_client:
                return None
            
            # Query personal cache table
            query = """
            SELECT insights, similarity_score, cached_at, access_level, metadata
            FROM personal_cache
            WHERE user_id = %s 
            AND semantic_hash = %s 
            AND business_domain = %s
            AND expires_at > NOW()
            ORDER BY cached_at DESC
            LIMIT 1
            """
            
            result = await self.postgres_client.execute_query(
                query, 
                params=[user_id, semantic_hash, business_domain]
            )
            
            if result.rows:
                row = result.rows[0]
                return {
                    "insights": row["insights"],
                    "similarity_score": row["similarity_score"],
                    "cached_at": row["cached_at"],
                    "access_level": row["access_level"],
                    "metadata": row.get("metadata", {})
                }
            
            # If no exact match, look for similar semantic hashes
            return await self._search_similar_personal_insights(
                user_id, semantic_hash, business_domain
            )
            
        except Exception as e:
            logger.warning(f"Failed to get personal insights: {e}")
            return None
    
    async def get_organizational_insights(
        self,
        organization_id: str,
        semantic_hash: str,
        business_domain: str,
        user_permissions: List[str]
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve organizational cached insights that user has access to.
        
        Args:
            organization_id: Organization identifier
            semantic_hash: Semantic hash of the business question
            business_domain: Business domain
            user_permissions: User's permission list
            
        Returns:
            Organizational cached insights or None if not found
        """
        try:
            if not self.postgres_client:
                return None
            
            # Build permission filter
            permission_filter = "required_permissions = '[]'::jsonb"
            if user_permissions:
                permission_conditions = []
                for perm in user_permissions:
                    permission_conditions.append(f"required_permissions ? '{perm}'")
                if permission_conditions:
                    permission_filter += f" OR ({' OR '.join(permission_conditions)})"
            
            # Query organizational cache table
            query = f"""
            SELECT insights, similarity_score, cached_at, required_permissions, 
                   original_analyst, metadata
            FROM organizational_cache
            WHERE organization_id = %s 
            AND semantic_hash = %s 
            AND business_domain = %s
            AND expires_at > NOW()
            AND ({permission_filter})
            ORDER BY cached_at DESC
            LIMIT 1
            """
            
            result = await self.postgres_client.execute_query(
                query,
                params=[organization_id, semantic_hash, business_domain]
            )
            
            if result.rows:
                row = result.rows[0]
                return {
                    "insights": row["insights"],
                    "similarity_score": row["similarity_score"],
                    "cached_at": row["cached_at"],
                    "required_permissions": row["required_permissions"],
                    "original_analyst": row.get("original_analyst"),
                    "metadata": row.get("metadata", {})
                }
            
            # If no exact match, look for similar insights
            return await self._search_similar_organizational_insights(
                organization_id, semantic_hash, business_domain, user_permissions
            )
            
        except Exception as e:
            logger.warning(f"Failed to get organizational insights: {e}")
            return None
    
    async def store_personal_insights(
        self,
        user_id: str,
        semantic_hash: str,
        business_domain: str,
        insights: Dict[str, Any],
        access_level: str,
        ttl_seconds: int
    ):
        """Store insights in user's personal cache."""
        try:
            if not self.postgres_client:
                return
            
            expires_at = datetime.utcnow() + timedelta(seconds=ttl_seconds)
            
            # Insert or update personal cache entry
            query = """
            INSERT INTO personal_cache 
            (user_id, semantic_hash, business_domain, insights, access_level, 
             cached_at, expires_at, similarity_score, metadata)
            VALUES (%s, %s, %s, %s, %s, NOW(), %s, %s, %s)
            ON CONFLICT (user_id, semantic_hash, business_domain)
            DO UPDATE SET
                insights = EXCLUDED.insights,
                access_level = EXCLUDED.access_level,
                cached_at = NOW(),
                expires_at = EXCLUDED.expires_at,
                metadata = EXCLUDED.metadata
            """
            
            metadata = {
                "cache_type": "personal",
                "ttl_seconds": ttl_seconds,
                "business_domain": business_domain
            }
            
            await self.postgres_client.execute_query(
                query,
                params=[
                    user_id, semantic_hash, business_domain,
                    json.dumps(insights), access_level, expires_at,
                    1.0, json.dumps(metadata)
                ]
            )
            
            logger.debug(f"Stored personal insights for user {user_id}")
            
        except Exception as e:
            logger.error(f"Failed to store personal insights: {e}")
    
    async def store_organizational_insights(
        self,
        organization_id: str,
        semantic_hash: str,
        business_domain: str,
        insights: Dict[str, Any],
        required_permissions: List[str],
        original_analyst: str,
        ttl_seconds: int
    ):
        """Store insights in organizational cache."""
        try:
            if not self.postgres_client:
                return
            
            expires_at = datetime.utcnow() + timedelta(seconds=ttl_seconds)
            
            # Insert or update organizational cache entry
            query = """
            INSERT INTO organizational_cache 
            (organization_id, semantic_hash, business_domain, insights, required_permissions,
             original_analyst, cached_at, expires_at, similarity_score, metadata)
            VALUES (%s, %s, %s, %s, %s, %s, NOW(), %s, %s, %s)
            ON CONFLICT (organization_id, semantic_hash, business_domain)
            DO UPDATE SET
                insights = EXCLUDED.insights,
                required_permissions = EXCLUDED.required_permissions,
                original_analyst = EXCLUDED.original_analyst,
                cached_at = NOW(),
                expires_at = EXCLUDED.expires_at,
                metadata = EXCLUDED.metadata
            """
            
            metadata = {
                "cache_type": "organizational",
                "ttl_seconds": ttl_seconds,
                "business_domain": business_domain,
                "shared_knowledge": True
            }
            
            await self.postgres_client.execute_query(
                query,
                params=[
                    organization_id, semantic_hash, business_domain,
                    json.dumps(insights), json.dumps(required_permissions),
                    original_analyst, expires_at, 1.0, json.dumps(metadata)
                ]
            )
            
            logger.debug(f"Stored organizational insights for org {organization_id}")
            
        except Exception as e:
            logger.error(f"Failed to store organizational insights: {e}")
    
    async def _search_similar_personal_insights(
        self,
        user_id: str,
        semantic_hash: str,
        business_domain: str
    ) -> Optional[Dict[str, Any]]:
        """Search for semantically similar personal insights."""
        try:
            # Use approximate string matching for semantic similarity
            # In a production system, this would use vector similarity
            query = """
            SELECT insights, similarity(semantic_hash, %s) as similarity_score,
                   cached_at, access_level, metadata
            FROM personal_cache
            WHERE user_id = %s 
            AND business_domain = %s
            AND expires_at > NOW()
            AND similarity(semantic_hash, %s) > %s
            ORDER BY similarity_score DESC
            LIMIT 1
            """
            
            result = await self.postgres_client.execute_query(
                query,
                params=[
                    semantic_hash, user_id, business_domain, 
                    semantic_hash, self.similarity_threshold
                ]
            )
            
            if result.rows:
                row = result.rows[0]
                if row["similarity_score"] >= self.similarity_threshold:
                    return {
                        "insights": row["insights"],
                        "similarity_score": row["similarity_score"],
                        "cached_at": row["cached_at"],
                        "access_level": row["access_level"],
                        "metadata": row.get("metadata", {})
                    }
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to search similar personal insights: {e}")
            return None
    
    async def _search_similar_organizational_insights(
        self,
        organization_id: str,
        semantic_hash: str,
        business_domain: str,
        user_permissions: List[str]
    ) -> Optional[Dict[str, Any]]:
        """Search for semantically similar organizational insights."""
        try:
            # Build permission filter
            permission_filter = "required_permissions = '[]'::jsonb"
            if user_permissions:
                permission_conditions = []
                for perm in user_permissions:
                    permission_conditions.append(f"required_permissions ? '{perm}'")
                if permission_conditions:
                    permission_filter += f" OR ({' OR '.join(permission_conditions)})"
            
            query = f"""
            SELECT insights, similarity(semantic_hash, %s) as similarity_score,
                   cached_at, required_permissions, original_analyst, metadata
            FROM organizational_cache
            WHERE organization_id = %s 
            AND business_domain = %s
            AND expires_at > NOW()
            AND similarity(semantic_hash, %s) > %s
            AND ({permission_filter})
            ORDER BY similarity_score DESC
            LIMIT 1
            """
            
            result = await self.postgres_client.execute_query(
                query,
                params=[
                    semantic_hash, organization_id, business_domain,
                    semantic_hash, self.similarity_threshold
                ]
            )
            
            if result.rows:
                row = result.rows[0]
                if row["similarity_score"] >= self.similarity_threshold:
                    return {
                        "insights": row["insights"],
                        "similarity_score": row["similarity_score"],
                        "cached_at": row["cached_at"],
                        "required_permissions": row["required_permissions"],
                        "original_analyst": row.get("original_analyst"),
                        "metadata": row.get("metadata", {})
                    }
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to search similar organizational insights: {e}")
            return None
    
    async def invalidate_user_cache(self, user_id: str, business_domain: Optional[str] = None):
        """Invalidate user's personal cache entries."""
        try:
            if not self.postgres_client:
                return
            
            if business_domain:
                query = "DELETE FROM personal_cache WHERE user_id = %s AND business_domain = %s"
                params = [user_id, business_domain]
            else:
                query = "DELETE FROM personal_cache WHERE user_id = %s"
                params = [user_id]
            
            await self.postgres_client.execute_query(query, params=params)
            logger.info(f"Invalidated personal cache for user {user_id}")
            
        except Exception as e:
            logger.warning(f"Failed to invalidate user cache: {e}")
    
    async def invalidate_organizational_cache(
        self,
        organization_id: str,
        business_domain: Optional[str] = None
    ):
        """Invalidate organizational cache entries."""
        try:
            if not self.postgres_client:
                return
            
            if business_domain:
                query = "DELETE FROM organizational_cache WHERE organization_id = %s AND business_domain = %s"
                params = [organization_id, business_domain]
            else:
                query = "DELETE FROM organizational_cache WHERE organization_id = %s"
                params = [organization_id]
            
            await self.postgres_client.execute_query(query, params=params)
            logger.info(f"Invalidated organizational cache for org {organization_id}")
            
        except Exception as e:
            logger.warning(f"Failed to invalidate organizational cache: {e}")
    
    async def cleanup_expired_cache(self):
        """Clean up expired cache entries."""
        try:
            if not self.postgres_client:
                return
            
            # Clean up personal cache
            personal_cleanup = await self.postgres_client.execute_query(
                "DELETE FROM personal_cache WHERE expires_at < NOW()"
            )
            
            # Clean up organizational cache
            org_cleanup = await self.postgres_client.execute_query(
                "DELETE FROM organizational_cache WHERE expires_at < NOW()"
            )
            
            logger.info(f"Cache cleanup completed: {personal_cleanup.row_count} personal, "
                       f"{org_cleanup.row_count} organizational entries removed")
            
        except Exception as e:
            logger.warning(f"Failed to cleanup expired cache: {e}")
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get PostgreSQL cache performance statistics."""
        try:
            if not self.postgres_client:
                return {}
            
            # Get cache statistics
            stats_query = """
            SELECT 
                'personal' as cache_type,
                COUNT(*) as total_entries,
                COUNT(*) FILTER (WHERE expires_at > NOW()) as active_entries,
                AVG(EXTRACT(EPOCH FROM (expires_at - cached_at))) as avg_ttl_seconds
            FROM personal_cache
            UNION ALL
            SELECT 
                'organizational' as cache_type,
                COUNT(*) as total_entries,
                COUNT(*) FILTER (WHERE expires_at > NOW()) as active_entries,
                AVG(EXTRACT(EPOCH FROM (expires_at - cached_at))) as avg_ttl_seconds
            FROM organizational_cache
            """
            
            result = await self.postgres_client.execute_query(stats_query)
            
            stats = {
                "cache_type": "postgresql_hybrid",
                "target_response_time_ms": 100,
                "personal_cache": {},
                "organizational_cache": {}
            }
            
            for row in result.rows:
                cache_type = row["cache_type"]
                stats[f"{cache_type}_cache"] = {
                    "total_entries": row["total_entries"],
                    "active_entries": row["active_entries"],
                    "avg_ttl_seconds": row["avg_ttl_seconds"]
                }
            
            return stats
            
        except Exception as e:
            logger.warning(f"Failed to get PostgreSQL cache stats: {e}")
            return {}