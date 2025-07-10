"""
Cache Manager - Anthropic Cache Only

Manages Anthropic cache for cost-efficient conversation caching.
PostgreSQL is NOT used for caching - it's only for chat history storage.

Features:
- Anthropic Cache: 50ms response time, 90% cost savings
- Organization-wide cache sharing
- Semantic similarity matching
"""

import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime

from .cache_logging import logger
from .ttl_manager import TTLManager
from .anthropic_cache import AnthropicCacheClient


class CacheManager:
    """
    Cache manager for Anthropic-only caching.
    
    Simplified architecture:
    - Only uses Anthropic cache for cost savings
    - No PostgreSQL caching (PostgreSQL is for chat history only)
    - Maintains semantic matching capabilities
    """
    
    def __init__(self):
        """Initialize cache manager with Anthropic cache only."""
        self.ttl_manager = TTLManager()
        self.anthropic_cache = AnthropicCacheClient()
        
        # Performance tracking
        self.cache_stats = {
            "anthropic_hits": 0,
            "cache_misses": 0,
            "total_requests": 0,
            "cost_savings": 0.0
        }
    
    async def initialize(self):
        """Initialize Anthropic cache client."""
        try:
            await self.anthropic_cache.initialize()
            logger.info("Cache manager initialized with Anthropic cache")
            
        except Exception as e:
            logger.error(f"Failed to initialize cache manager: {e}")
            raise
    
    async def get_cached_insights(
        self,
        semantic_hash: str,
        business_domain: str,
        semantic_intent: Dict[str, Any],
        user_context: Dict[str, Any],
        organization_context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve insights from Anthropic cache.
        
        Args:
            semantic_hash: Semantic hash of the business question
            business_domain: Business domain classification
            semantic_intent: Processed business question intent
            user_context: User information and permissions
            organization_context: Organizational context
            
        Returns:
            Cached insights if found, None otherwise
        """
        try:
            self.cache_stats["total_requests"] += 1
            
            organization_id = organization_context.get("organization_id", "default")
            
            # Check Anthropic cache
            result = await self.anthropic_cache.get_similar_conversation(
                semantic_hash=semantic_hash,
                business_domain=business_domain,
                organization_id=organization_id,
                similarity_threshold=0.85
            )
            
            if result:
                # Update statistics
                self.cache_stats["anthropic_hits"] += 1
                self.cache_stats["cost_savings"] += 0.90  # 90% cost savings
                
                logger.debug(f"Anthropic cache hit for {business_domain}")
                result["cache_tier"] = "anthropic"
                return result
            
            # Cache miss
            self.cache_stats["cache_misses"] += 1
            logger.debug(f"Cache miss for {business_domain}.{semantic_hash}")
            return None
            
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {e}")
            return None
    
    async def store_insights(
        self,
        semantic_hash: str,
        business_domain: str,
        semantic_intent: Dict[str, Any],
        user_context: Dict[str, Any],
        organization_context: Dict[str, Any],
        insights: Dict[str, Any],
        conversation_context: Optional[Dict[str, Any]] = None
    ):
        """
        Store insights in Anthropic cache.
        
        Args:
            semantic_hash: Semantic hash of the business question
            business_domain: Business domain classification
            semantic_intent: Processed business question intent
            user_context: User information and permissions
            organization_context: Organizational context
            insights: Generated business insights
            conversation_context: Complete conversation context for Anthropic cache
        """
        try:
            # Get intelligent TTL
            ttl_seconds = self.ttl_manager.get_ttl_for_question(
                semantic_intent=semantic_intent,
                user_context=user_context,
                organization_context=organization_context
            )
            
            organization_id = organization_context.get("organization_id", "default")
            
            # Only store in Anthropic cache if we have conversation context
            if conversation_context:
                await self.anthropic_cache.store_conversation(
                    semantic_hash=semantic_hash,
                    business_domain=business_domain,
                    organization_id=organization_id,
                    conversation_context=conversation_context,
                    insights=insights,
                    ttl_seconds=ttl_seconds
                )
                logger.debug(f"Stored insights in Anthropic cache with TTL {ttl_seconds}s")
            else:
                logger.warning("No conversation context provided, skipping cache storage")
            
        except Exception as e:
            logger.error(f"Failed to store insights: {e}")
    
    async def invalidate_cache(
        self,
        target: str,
        identifier: str,
        business_domain: Optional[str] = None
    ):
        """
        Invalidate cache entries.
        
        Note: Anthropic cache is TTL-based and cannot be directly invalidated.
        This method is kept for interface compatibility.
        
        Args:
            target: Invalidation target ('organization', 'domain')
            identifier: Target identifier (org_id, domain_name)
            business_domain: Optional business domain filter
        """
        try:
            if target == "organization":
                # Anthropic cache invalidation (if supported)
                await self.anthropic_cache.invalidate_conversation(
                    semantic_hash="*",  # Wildcard
                    business_domain=business_domain or "*",
                    organization_id=identifier
                )
            
            logger.info(f"Cache invalidation requested for {target}: {identifier}")
            
        except Exception as e:
            logger.warning(f"Cache invalidation not fully supported: {e}")
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self.cache_stats["total_requests"]
        if total_requests == 0:
            return {"message": "No cache requests yet"}
        
        hit_rate = self.cache_stats["anthropic_hits"] / total_requests
        miss_rate = self.cache_stats["cache_misses"] / total_requests
        
        return {
            "total_requests": total_requests,
            "total_hits": self.cache_stats["anthropic_hits"],
            "total_misses": self.cache_stats["cache_misses"],
            "hit_rate": round(hit_rate * 100, 2),
            "miss_rate": round(miss_rate * 100, 2),
            "cache_type": "anthropic_only",
            "cost_savings": {
                "total_saved": round(self.cache_stats["cost_savings"], 2),
                "avg_per_request": round(self.cache_stats["cost_savings"] / total_requests, 4)
            },
            "ttl_statistics": self.ttl_manager.get_ttl_statistics()
        }
    
    async def warm_cache(
        self,
        organization_id: str,
        common_queries: List[Dict[str, Any]]
    ):
        """
        Warm Anthropic cache with common queries.
        
        Args:
            organization_id: Organization identifier
            common_queries: List of common query patterns with insights
        """
        try:
            await self.anthropic_cache.warm_cache_for_organization(
                organization_id=organization_id,
                common_queries=common_queries
            )
            
            logger.info(f"Cache warming completed for organization {organization_id}")
            
        except Exception as e:
            logger.error(f"Cache warming failed: {e}")