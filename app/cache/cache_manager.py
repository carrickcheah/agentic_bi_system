"""
Cache Manager - Multi-Tier Cache Orchestrator

Coordinates the 3-tier cache architecture for optimal performance and cost efficiency:
- Tier 1a: Anthropic Cache (50ms, organization-wide, 90% cost savings)
- Tier 1b: PostgreSQL Cache (100ms, permission-aware, organizational learning)
- Tier 2: Full Investigation (15s+, complete analysis)

Features:
- Intelligent cache tier selection based on TTL
- Multi-level cache sharing with permission controls
- Organizational knowledge accumulation
- Performance optimization and cost management
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from enum import Enum

from .cache_logging import logger
from .ttl_manager import TTLManager, DataVolatility, CachePriority
from .anthropic_cache import AnthropicCacheClient
from .postgresql_cache import PostgreSQLCacheClient
from .semantic_cache import SemanticCacheClient


class CacheStrategy(Enum):
    """Cache strategy selection based on TTL and context."""
    ANTHROPIC_ONLY = "anthropic_only"         # Long TTL, stable data
    POSTGRESQL_ONLY = "postgresql_only"       # Short TTL, dynamic data
    HYBRID = "hybrid"                         # Medium TTL, use both tiers
    SEMANTIC_PRIORITY = "semantic_priority"   # Semantic matching first


class CacheManager:
    """
    Multi-tier cache orchestrator for intelligent cache management.
    
    Coordinates cache operations across multiple tiers to optimize for:
    - Response time (50ms � 100ms � 15s+)
    - Cost efficiency (90% savings on hits)
    - Data freshness (TTL-based invalidation)
    - Permission awareness (role-based access)
    - Organizational learning (shared knowledge)
    """
    
    def __init__(self):
        """Initialize cache manager with all cache clients."""
        self.ttl_manager = TTLManager()
        self.anthropic_cache = AnthropicCacheClient()
        self.postgresql_cache = PostgreSQLCacheClient()
        self.semantic_cache = SemanticCacheClient()
        
        # Cache tier selection thresholds (seconds)
        self.anthropic_threshold = 7200    # >= 2 hours: Use Anthropic
        self.postgresql_threshold = 300    # >= 5 minutes: Use PostgreSQL
        self.hybrid_threshold = 86400      # >= 24 hours: Use both tiers
        
        # Performance tracking
        self.cache_stats = {
            "anthropic_hits": 0,
            "postgresql_hits": 0,
            "semantic_hits": 0,
            "cache_misses": 0,
            "total_requests": 0,
            "cost_savings": 0.0
        }
    
    async def initialize(self):
        """Initialize all cache clients."""
        try:
            await asyncio.gather(
                self.anthropic_cache.initialize(),
                self.postgresql_cache.initialize(),
                self.semantic_cache.initialize()
            )
            logger.info("Cache manager initialized with all tiers")
            
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
        Retrieve insights from multi-tier cache with intelligent fallback.
        
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
            
            # Determine cache strategy
            strategy = self._select_cache_strategy(
                semantic_intent, user_context, organization_context
            )
            
            # Execute cache retrieval strategy
            result = await self._execute_cache_strategy(
                strategy=strategy,
                semantic_hash=semantic_hash,
                business_domain=business_domain,
                semantic_intent=semantic_intent,
                user_context=user_context,
                organization_context=organization_context
            )
            
            if result:
                # Update statistics
                cache_tier = result.get("cache_tier", "unknown")
                self.cache_stats[f"{cache_tier}_hits"] += 1
                
                # Calculate cost savings (90% savings on Anthropic hits)
                if cache_tier == "anthropic":
                    self.cache_stats["cost_savings"] += 0.90
                elif cache_tier == "postgresql":
                    self.cache_stats["cost_savings"] += 0.75
                
                logger.debug(f"Cache hit: {cache_tier} tier for {business_domain}")
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
        Store insights across appropriate cache tiers with intelligent TTL.
        
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
            
            # Determine storage strategy
            storage_tiers = self._select_storage_tiers(ttl_seconds, user_context, organization_context)
            
            # Store across selected tiers
            await self._store_across_tiers(
                tiers=storage_tiers,
                semantic_hash=semantic_hash,
                business_domain=business_domain,
                semantic_intent=semantic_intent,
                user_context=user_context,
                organization_context=organization_context,
                insights=insights,
                conversation_context=conversation_context,
                ttl_seconds=ttl_seconds
            )
            
            logger.debug(f"Stored insights across {len(storage_tiers)} tiers with TTL {ttl_seconds}s")
            
        except Exception as e:
            logger.error(f"Failed to store insights: {e}")
    
    async def invalidate_cache(
        self,
        target: str,
        identifier: str,
        business_domain: Optional[str] = None
    ):
        """
        Invalidate cache entries across all tiers.
        
        Args:
            target: Invalidation target ('user', 'organization', 'domain')
            identifier: Target identifier (user_id, org_id, domain_name)
            business_domain: Optional business domain filter
        """
        try:
            if target == "user":
                await self.postgresql_cache.invalidate_user_cache(identifier, business_domain)
            elif target == "organization":
                await self.postgresql_cache.invalidate_organizational_cache(identifier, business_domain)
            elif target == "domain":
                # Invalidate all caches for a business domain
                await asyncio.gather(
                    self.postgresql_cache.invalidate_user_cache(identifier, business_domain),
                    self.postgresql_cache.invalidate_organizational_cache(identifier, business_domain),
                    # Note: Anthropic cache TTL-based, semantic cache pattern-based
                )
            
            logger.info(f"Invalidated {target} cache: {identifier}")
            
        except Exception as e:
            logger.error(f"Cache invalidation failed: {e}")
    
    def _select_cache_strategy(
        self,
        semantic_intent: Dict[str, Any],
        user_context: Dict[str, Any],
        organization_context: Dict[str, Any]
    ) -> CacheStrategy:
        """Select optimal cache retrieval strategy."""
        
        # Get TTL to understand data volatility
        ttl_seconds = self.ttl_manager.get_ttl_for_question(
            semantic_intent, user_context, organization_context
        )
        
        # High volatility data � PostgreSQL only
        if ttl_seconds <= self.postgresql_threshold:
            return CacheStrategy.POSTGRESQL_ONLY
        
        # Long-term stable data � Anthropic priority
        if ttl_seconds >= self.anthropic_threshold:
            return CacheStrategy.ANTHROPIC_ONLY
        
        # Medium volatility � Hybrid approach
        if ttl_seconds >= self.hybrid_threshold:
            return CacheStrategy.HYBRID
        
        # Default: Semantic matching first
        return CacheStrategy.SEMANTIC_PRIORITY
    
    async def _execute_cache_strategy(
        self,
        strategy: CacheStrategy,
        semantic_hash: str,
        business_domain: str,
        semantic_intent: Dict[str, Any],
        user_context: Dict[str, Any],
        organization_context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Execute the selected cache retrieval strategy."""
        
        user_id = user_context.get("user_id")
        organization_id = organization_context.get("organization_id")
        user_permissions = user_context.get("permissions", [])
        
        if strategy == CacheStrategy.ANTHROPIC_ONLY:
            # Try Anthropic cache first
            result = await self.anthropic_cache.get_similar_conversation(
                semantic_hash, business_domain, organization_id
            )
            if result:
                result["cache_tier"] = "anthropic"
                return result
        
        elif strategy == CacheStrategy.POSTGRESQL_ONLY:
            # Try PostgreSQL cache tiers
            # 1. Personal cache
            result = await self.postgresql_cache.get_personal_insights(
                user_id, semantic_hash, business_domain
            )
            if result:
                result["cache_tier"] = "postgresql"
                return result
            
            # 2. Organizational cache
            result = await self.postgresql_cache.get_organizational_insights(
                organization_id, semantic_hash, business_domain, user_permissions
            )
            if result:
                result["cache_tier"] = "postgresql"
                return result
        
        elif strategy == CacheStrategy.HYBRID:
            # Try all tiers in order of speed
            # 1. Anthropic (fastest, organization-wide)
            result = await self.anthropic_cache.get_similar_conversation(
                semantic_hash, business_domain, organization_id
            )
            if result:
                result["cache_tier"] = "anthropic"
                return result
            
            # 2. PostgreSQL personal
            result = await self.postgresql_cache.get_personal_insights(
                user_id, semantic_hash, business_domain
            )
            if result:
                result["cache_tier"] = "postgresql"
                return result
            
            # 3. PostgreSQL organizational
            result = await self.postgresql_cache.get_organizational_insights(
                organization_id, semantic_hash, business_domain, user_permissions
            )
            if result:
                result["cache_tier"] = "postgresql"
                return result
        
        elif strategy == CacheStrategy.SEMANTIC_PRIORITY:
            # Try semantic matching first
            result = await self.semantic_cache.find_similar_insights(
                semantic_intent, organization_id, user_permissions
            )
            if result:
                result["cache_tier"] = "semantic"
                return result
            
            # Fallback to PostgreSQL
            result = await self.postgresql_cache.get_organizational_insights(
                organization_id, semantic_hash, business_domain, user_permissions
            )
            if result:
                result["cache_tier"] = "postgresql"
                return result
        
        return None
    
    def _select_storage_tiers(
        self,
        ttl_seconds: int,
        user_context: Dict[str, Any],
        organization_context: Dict[str, Any]
    ) -> List[str]:
        """Select which cache tiers to store data in based on TTL and context."""
        tiers = []
        
        # Always store in PostgreSQL for fast access
        tiers.append("postgresql")
        
        # Store in Anthropic cache for longer TTL (cost efficiency)
        if ttl_seconds >= self.anthropic_threshold:
            tiers.append("anthropic")
        
        # Store in semantic cache for pattern learning
        if organization_context.get("enable_semantic_learning", True):
            tiers.append("semantic")
        
        return tiers
    
    async def _store_across_tiers(
        self,
        tiers: List[str],
        semantic_hash: str,
        business_domain: str,
        semantic_intent: Dict[str, Any],
        user_context: Dict[str, Any],
        organization_context: Dict[str, Any],
        insights: Dict[str, Any],
        conversation_context: Optional[Dict[str, Any]],
        ttl_seconds: int
    ):
        """Store insights across multiple cache tiers."""
        
        user_id = user_context.get("user_id")
        organization_id = organization_context.get("organization_id")
        user_permissions = user_context.get("permissions", [])
        access_level = user_context.get("role", "analyst")
        
        storage_tasks = []
        
        if "postgresql" in tiers:
            # Store in both personal and organizational PostgreSQL cache
            storage_tasks.extend([
                self.postgresql_cache.store_personal_insights(
                    user_id=user_id,
                    semantic_hash=semantic_hash,
                    business_domain=business_domain,
                    insights=insights,
                    access_level=access_level,
                    ttl_seconds=ttl_seconds
                ),
                self.postgresql_cache.store_organizational_insights(
                    organization_id=organization_id,
                    semantic_hash=semantic_hash,
                    business_domain=business_domain,
                    insights=insights,
                    required_permissions=user_permissions,
                    original_analyst=user_id,
                    ttl_seconds=ttl_seconds
                )
            ])
        
        if "anthropic" in tiers and conversation_context:
            # Store in Anthropic cache
            storage_tasks.append(
                self.anthropic_cache.store_conversation(
                    semantic_hash=semantic_hash,
                    business_domain=business_domain,
                    organization_id=organization_id,
                    conversation_context=conversation_context,
                    insights=insights,
                    ttl_seconds=ttl_seconds
                )
            )
        
        if "semantic" in tiers:
            # Store in semantic cache
            storage_tasks.append(
                self.semantic_cache.store_semantic_pattern(
                    semantic_intent=semantic_intent,
                    business_domain=business_domain,
                    organization_id=organization_id,
                    insights=insights,
                    user_permissions=user_permissions,
                    original_question=conversation_context.get("original_question") if conversation_context else None
                )
            )
        
        # Execute all storage operations concurrently
        await asyncio.gather(*storage_tasks, return_exceptions=True)
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache performance statistics."""
        total_requests = self.cache_stats["total_requests"]
        if total_requests == 0:
            return {"message": "No cache requests yet"}
        
        total_hits = (
            self.cache_stats["anthropic_hits"] + 
            self.cache_stats["postgresql_hits"] + 
            self.cache_stats["semantic_hits"]
        )
        
        hit_rate = total_hits / total_requests
        miss_rate = self.cache_stats["cache_misses"] / total_requests
        
        return {
            "total_requests": total_requests,
            "total_hits": total_hits,
            "total_misses": self.cache_stats["cache_misses"],
            "hit_rate": round(hit_rate * 100, 2),
            "miss_rate": round(miss_rate * 100, 2),
            "tier_performance": {
                "anthropic_hits": self.cache_stats["anthropic_hits"],
                "postgresql_hits": self.cache_stats["postgresql_hits"],
                "semantic_hits": self.cache_stats["semantic_hits"],
            },
            "cost_savings": {
                "total_saved": round(self.cache_stats["cost_savings"], 2),
                "avg_per_request": round(self.cache_stats["cost_savings"] / total_requests, 4)
            },
            "ttl_statistics": self.ttl_manager.get_ttl_statistics()
        }
    
    async def warm_cache(
        self,
        business_domains: List[str],
        organization_id: str,
        priority_questions: Optional[List[Dict[str, Any]]] = None
    ):
        """Warm cache with common business intelligence patterns."""
        try:
            # Delegate to cache warming engine
            from .cache_warming import CacheWarmingEngine
            
            warming_engine = CacheWarmingEngine(self)
            await warming_engine.initialize()
            
            await warming_engine.warm_organizational_cache(
                business_domains=business_domains,
                organization_id=organization_id,
                priority_questions=priority_questions
            )
            
            logger.info(f"Cache warming completed for {len(business_domains)} domains")
            
        except Exception as e:
            logger.error(f"Cache warming failed: {e}")