"""
Semantic Cache Client - Pattern-Based Intelligence Cache

Manages PostgreSQL database for semantic similarity matching and pattern recognition.

Features:
- Semantic similarity matching for business questions
- Cross-domain pattern recognition
- Organizational knowledge accumulation
- FAQ-style instant responses
- Business intelligence pattern learning
"""

import asyncio
import hashlib
import json
from typing import Dict, List, Optional, Any, Tuple, TYPE_CHECKING
from datetime import datetime

from .cache_logging import logger
from .ttl_manager import TTLManager

if TYPE_CHECKING:
    from ..fastmcp.postgres_client import PostgreSQLClient


class SemanticCacheClient:
    """
    Client for PostgreSQL-based semantic caching and pattern recognition.
    
    Stores business intelligence patterns and provides semantic similarity
    matching for instant responses to similar questions.
    """
    
    def __init__(self):
        self.postgres_client: Optional['PostgreSQLClient'] = None
        self.ttl_manager = TTLManager()
        self.table_name = "semantic_cache"
        self.similarity_threshold = 0.75
        self.max_results = 10
        
    async def initialize(self):
        """Initialize PostgreSQL semantic cache client."""
        try:
            # Initialize PostgreSQL client connection
            # This will be injected from the MCP client manager
            logger.info("Semantic cache client initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize semantic cache: {e}")
            raise
    
    def set_postgres_client(self, client: 'PostgreSQLClient'):
        """Set the PostgreSQL client instance."""
        self.postgres_client = client
        logger.info("PostgreSQL client set for semantic cache")
    
    async def find_similar_insights(
        self,
        semantic_intent: Dict[str, Any],
        organization_id: str,
        user_permissions: List[str],
        max_results: int = 5
    ) -> Optional[Dict[str, Any]]:
        """
        Find semantically similar insights using PostgreSQL database functions.
        
        Args:
            semantic_intent: Processed business question intent
            organization_id: Organization identifier
            user_permissions: User's permissions for filtering
            max_results: Maximum number of results to return
            
        Returns:
            Similar insights if found, None otherwise
        """
        try:
            if not self.postgres_client:
                return None
            
            # Use the PostgreSQL function we created
            query = """
            SELECT * FROM find_similar_semantic_patterns(
                $1, $2, $3, $4, $5, $6
            )
            """
            
            business_domain = semantic_intent.get("business_domain")
            
            result = await self.postgres_client.execute_query(
                query,
                params=[
                    organization_id,
                    json.dumps(semantic_intent),
                    business_domain,
                    json.dumps(user_permissions),
                    self.similarity_threshold,
                    max_results
                ]
            )
            
            if not result.rows:
                return None
            
            # Get the best match (first result, already sorted by similarity)
            best_match = result.rows[0]
            
            # Update usage count for the matched pattern
            await self.update_pattern_usage(best_match["pattern_id"])
            
            # Return the best matching insights
            return {
                "insights": best_match["insights"],
                "business_domain": best_match["business_domain"],
                "similarity_score": best_match["similarity_score"],
                "pattern_id": best_match["pattern_id"],
                "semantic_hash": best_match["semantic_hash"],
                "usage_count": best_match["usage_count"],
                "last_used": best_match["last_used"],
                "cache_tier": "semantic"
            }
            
        except Exception as e:
            logger.warning(f"Semantic cache search failed: {e}")
            return None
    
    async def store_semantic_pattern(
        self,
        semantic_intent: Dict[str, Any],
        business_domain: str,
        organization_id: str,
        insights: Dict[str, Any],
        user_permissions: List[str],
        original_question: Optional[str] = None
    ):
        """
        Store business intelligence pattern for semantic matching.
        
        Args:
            semantic_intent: Processed business question intent
            business_domain: Business domain classification
            organization_id: Organization identifier
            insights: Generated business insights
            user_permissions: Required permissions to access this pattern
            original_question: Original natural language question
        """
        try:
            if not self.postgres_client:
                return
            
            # Generate pattern ID and semantic hash
            pattern_id = self._generate_pattern_id(semantic_intent, business_domain, organization_id)
            semantic_hash = self._generate_semantic_hash(semantic_intent)
            
            # Extract metadata from semantic intent
            business_intent = semantic_intent.get("business_intent", {})
            question_type = business_intent.get("question_type", "descriptive")
            analysis_type = semantic_intent.get("analysis_type", "general")
            
            # Determine if pattern is public
            public_pattern = len(user_permissions) == 0 or "public" in user_permissions
            
            # Store pattern in PostgreSQL
            query = """
            INSERT INTO semantic_cache 
            (pattern_id, semantic_hash, organization_id, business_domain, 
             semantic_intent, insights, original_question, question_type, 
             analysis_type, required_permissions, public_pattern, metadata)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
            ON CONFLICT (pattern_id)
            DO UPDATE SET
                semantic_intent = EXCLUDED.semantic_intent,
                insights = EXCLUDED.insights,
                last_used = NOW(),
                metadata = EXCLUDED.metadata
            """
            
            metadata = {
                "question_type": question_type,
                "time_period": business_intent.get("time_period"),
                "analysis_type": analysis_type,
                "complexity_score": semantic_intent.get("complexity_score", 0),
                "stored_at": datetime.utcnow().isoformat()
            }
            
            await self.postgres_client.execute_query(
                query,
                params=[
                    pattern_id, semantic_hash, organization_id, business_domain,
                    json.dumps(semantic_intent), json.dumps(insights),
                    original_question or "", question_type, analysis_type,
                    json.dumps(user_permissions), public_pattern, json.dumps(metadata)
                ]
            )
            
            logger.debug(f"Stored semantic pattern: {pattern_id} for {business_domain}")
            
        except Exception as e:
            logger.error(f"Failed to store semantic pattern: {e}")
    
    async def update_pattern_usage(self, pattern_id: str):
        """Update usage count for a semantic pattern."""
        try:
            if not self.postgres_client:
                return
            
            # Update usage count and last_used - the trigger will handle incrementing
            query = """
            UPDATE semantic_cache 
            SET last_used = NOW()
            WHERE pattern_id = $1
            """
            
            await self.postgres_client.execute_query(query, params=[pattern_id])
            
        except Exception as e:
            logger.warning(f"Failed to update pattern usage: {e}")
    
    async def get_popular_patterns(
        self,
        organization_id: str,
        business_domain: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get most popular semantic patterns for an organization.
        
        Args:
            organization_id: Organization identifier
            business_domain: Optional business domain filter
            limit: Maximum number of patterns to return
            
        Returns:
            List of popular patterns
        """
        try:
            if not self.postgres_client:
                return []
            
            # Use the PostgreSQL function we created
            query = """
            SELECT * FROM get_popular_semantic_patterns($1, $2, $3)
            """
            
            result = await self.postgres_client.execute_query(
                query,
                params=[organization_id, business_domain, limit]
            )
            
            if not result.rows:
                return []
            
            return [
                {
                    "pattern_id": row["pattern_id"],
                    "business_domain": row["business_domain"],
                    "question_type": row["question_type"],
                    "usage_count": row["usage_count"],
                    "last_used": row["last_used"],
                    "original_question": row["original_question"],
                    "insights_summary": row["insights_summary"]
                }
                for row in result.rows
            ]
            
        except Exception as e:
            logger.error(f"Failed to get popular patterns: {e}")
            return []
    
    async def invalidate_patterns(
        self,
        organization_id: str,
        business_domain: Optional[str] = None,
        pattern_ids: Optional[List[str]] = None
    ):
        """
        Invalidate semantic patterns.
        
        Args:
            organization_id: Organization identifier
            business_domain: Optional business domain filter
            pattern_ids: Optional specific pattern IDs to invalidate
        """
        try:
            if not self.postgres_client:
                return
            
            if pattern_ids:
                # Invalidate specific patterns
                for pattern_id in pattern_ids:
                    await self._deactivate_pattern(pattern_id)
                logger.info(f"Invalidated {len(pattern_ids)} specific patterns")
            else:
                # Invalidate by organization/domain
                if business_domain:
                    query = """
                    UPDATE semantic_cache 
                    SET active = false, deactivated_at = NOW()
                    WHERE organization_id = $1 AND business_domain = $2
                    """
                    params = [organization_id, business_domain]
                else:
                    query = """
                    UPDATE semantic_cache 
                    SET active = false, deactivated_at = NOW()
                    WHERE organization_id = $1
                    """
                    params = [organization_id]
                
                result = await self.postgres_client.execute_query(query, params=params)
                logger.info(f"Invalidated {result.row_count} patterns for {organization_id}")
            
        except Exception as e:
            logger.error(f"Failed to invalidate patterns: {e}")
    
    def _generate_semantic_hash(self, semantic_intent: Dict[str, Any]) -> str:
        """Generate semantic hash for the intent."""
        # Create a consistent hash from the semantic intent
        intent_str = json.dumps(semantic_intent, sort_keys=True)
        return hashlib.sha256(intent_str.encode()).hexdigest()[:16]
    
    def _generate_pattern_id(self, semantic_intent: Dict[str, Any], business_domain: str, organization_id: str) -> str:
        """Generate unique pattern ID."""
        # Create hash from key components
        hash_components = [
            semantic_intent.get("business_domain", ""),
            business_domain,
            organization_id,
            str(semantic_intent.get("business_intent", {})),
            str(datetime.utcnow().date())  # Include date for daily uniqueness
        ]
        
        hash_string = "|".join(hash_components)
        return hashlib.sha256(hash_string.encode()).hexdigest()[:16]
    
    async def _deactivate_pattern(self, pattern_id: str):
        """Deactivate a semantic pattern."""
        try:
            if not self.postgres_client:
                return
            
            query = """
            UPDATE semantic_cache 
            SET active = false, deactivated_at = NOW()
            WHERE pattern_id = $1
            """
            
            await self.postgres_client.execute_query(query, params=[pattern_id])
            
        except Exception as e:
            logger.warning(f"Failed to deactivate pattern {pattern_id}: {e}")
    
    async def get_semantic_statistics(self, organization_id: Optional[str] = None) -> Dict[str, Any]:
        """Get semantic cache statistics."""
        try:
            if not self.postgres_client:
                return {
                    "table_name": self.table_name,
                    "similarity_threshold": self.similarity_threshold,
                    "max_results": self.max_results,
                    "status": "inactive"
                }
            
            # Use the PostgreSQL function for statistics
            query = """
            SELECT * FROM get_semantic_cache_stats($1)
            """
            
            result = await self.postgres_client.execute_query(query, params=[organization_id])
            
            if result.rows:
                stats = result.rows[0]
                return {
                    "table_name": self.table_name,
                    "similarity_threshold": self.similarity_threshold,
                    "max_results": self.max_results,
                    "status": "active",
                    "total_patterns": stats["total_patterns"],
                    "active_patterns": stats["active_patterns"],
                    "total_usage": stats["total_usage"],
                    "avg_usage_per_pattern": float(stats["avg_usage_per_pattern"]),
                    "top_business_domain": stats["top_business_domain"],
                    "most_popular_pattern": stats["most_popular_pattern"],
                    "cache_efficiency": float(stats["cache_efficiency"])
                }
            else:
                return {
                    "table_name": self.table_name,
                    "similarity_threshold": self.similarity_threshold,
                    "max_results": self.max_results,
                    "status": "active",
                    "total_patterns": 0
                }
                
        except Exception as e:
            logger.error(f"Failed to get semantic statistics: {e}")
            return {
                "table_name": self.table_name,
                "error": str(e),
                "status": "error"
            }