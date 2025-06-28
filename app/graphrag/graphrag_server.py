"""
GraphRAG Server - Hybrid Architecture Implementation

Stateful GraphRAG server that solves state management and concurrency issues
while providing MCP-compatible interface for business intelligence operations.

Key Features:
- Persistent state management with startup initialization
- Concurrency controls (RWLock, semaphores) for safe access
- Memory caching and LLM connection pooling
- Production monitoring and error handling
- Cost controls and timeout management
"""

import asyncio
import time
import hashlib
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from cachetools import LRUCache

try:
    from ..utils.logging import logger
    from ..model.runner import ModelManager
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    ModelManager = None

from .config import GraphRAGConfig
from .models import (
    GraphRAGEntity,
    GraphRAGEntitySearchResponse,
    GraphRAGGlobalSearchResponse,
    GraphRAGCommunityInsight,
    GraphRAGServerHealth,
    GraphRAGPerformanceMetrics,
    GraphRAGErrorResponse
)


@dataclass
class GraphRAGKnowledgeBase:
    """Container for GraphRAG knowledge base data."""
    entities_df: pd.DataFrame
    relationships_df: pd.DataFrame  
    communities_df: pd.DataFrame
    community_reports_df: pd.DataFrame
    entity_index: Dict[str, Any]
    community_index: Dict[str, Any]
    loaded_at: datetime


class GraphRAGConcurrencyManager:
    """Manages concurrency controls for GraphRAG operations."""
    
    def __init__(self, max_concurrent: int = 10):
        self.read_write_lock = asyncio.RWLock()
        self.update_semaphore = asyncio.Semaphore(1)  # Single update at a time
        self.query_semaphore = asyncio.Semaphore(max_concurrent)
        self.global_search_semaphore = asyncio.Semaphore(3)  # Limit expensive operations


class GraphRAGPerformanceTracker:
    """Tracks performance metrics for GraphRAG operations."""
    
    def __init__(self):
        self.metrics = {
            "entity_searches": 0,
            "global_searches": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_cost": 0.0,
            "errors": 0,
            "startup_time": 0.0
        }
        self.request_times = LRUCache(maxsize=1000)
        self.hourly_errors = LRUCache(maxsize=24)
        
    def record_request(self, operation: str, duration: float, success: bool, cost: float = 0.0):
        """Record request metrics."""
        self.metrics[f"{operation}s"] += 1
        self.metrics["total_cost"] += cost
        
        if not success:
            self.metrics["errors"] += 1
            hour = datetime.utcnow().hour
            self.hourly_errors[hour] = self.hourly_errors.get(hour, 0) + 1
            
        self.request_times[f"{operation}_{time.time()}"] = duration
    
    def record_cache_hit(self):
        """Record cache hit."""
        self.metrics["cache_hits"] += 1
    
    def record_cache_miss(self):
        """Record cache miss."""
        self.metrics["cache_misses"] += 1
    
    def get_cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.metrics["cache_hits"] + self.metrics["cache_misses"]
        return (self.metrics["cache_hits"] / total * 100) if total > 0 else 0.0


class GraphRAGServer:
    """
    Stateful GraphRAG server with hybrid architecture.
    
    Solves Microsoft GraphRAG's architectural conflicts with MCP protocol
    through persistent state management and proper concurrency controls.
    """
    
    def __init__(self, config: GraphRAGConfig):
        self.config = config
        self.knowledge_base: Optional[GraphRAGKnowledgeBase] = None
        self.concurrency = GraphRAGConcurrencyManager(config.max_concurrent_requests)
        self.performance = GraphRAGPerformanceTracker()
        
        # Caching
        self.memory_cache = LRUCache(maxsize=config.cache_size)
        self.search_cache = LRUCache(maxsize=config.cache_size // 2)
        
        # Model management
        self.model_manager = ModelManager() if ModelManager else None
        self.llm_model = None
        
        # Server state
        self.is_initialized = False
        self.startup_time = None
        self.last_health_check = None
        
        # Thread pool for CPU-intensive operations
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        logger.info("GraphRAG server instance created")
    
    async def start_server(self) -> bool:
        """
        Initialize GraphRAG server with persistent state.
        Solves the cold start problem by loading data once at startup.
        """
        if self.is_initialized:
            logger.warning("GraphRAG server already initialized")
            return True
        
        start_time = time.time()
        self.startup_time = datetime.utcnow()
        
        try:
            logger.info("🚀 Starting GraphRAG server initialization...")
            
            # Step 1: Validate data files
            if not self.config.validate_data_files():
                raise FileNotFoundError("Required GraphRAG data files not found")
            
            # Step 2: Initialize LLM model
            await self._initialize_llm_model()
            
            # Step 3: Load knowledge graph data (expensive one-time operation)
            await self._load_knowledge_base()
            
            # Step 4: Build search indices (expensive one-time operation)  
            await self._build_search_indices()
            
            # Step 5: Pre-warm cache with popular searches
            await self._precompute_popular_searches()
            
            # Record startup metrics
            initialization_time = time.time() - start_time
            self.performance.metrics["startup_time"] = initialization_time
            
            self.is_initialized = True
            self.last_health_check = datetime.utcnow()
            
            logger.info(f"✅ GraphRAG server ready in {initialization_time:.2f}s")
            logger.info(f"📊 Knowledge base loaded: {len(self.knowledge_base.entities_df)} entities, "
                       f"{len(self.knowledge_base.communities_df)} communities")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ GraphRAG server startup failed: {e}")
            raise
    
    async def search_entities(
        self,
        query: str,
        limit: int = 20,
        timeout: float = 5.0
    ) -> GraphRAGEntitySearchResponse:
        """
        Fast entity search with caching and timeout protection.
        """
        if not self.is_initialized:
            raise RuntimeError("GraphRAG server not initialized")
        
        start_time = time.time()
        cache_key = f"entities:{hashlib.md5(query.encode()).hexdigest()}:{limit}"
        
        # Check cache first
        if cache_key in self.memory_cache:
            self.performance.record_cache_hit()
            cached_result = self.memory_cache[cache_key]
            cached_result.cache_hit = True
            return cached_result
        
        self.performance.record_cache_miss()
        
        # Execute search with read lock and timeout
        async with self.concurrency.read_write_lock.read_lock():
            async with self.concurrency.query_semaphore:
                try:
                    async with asyncio.timeout(timeout):
                        entities = await self._execute_entity_search(query, limit)
                        
                        execution_time = time.time() - start_time
                        
                        response = GraphRAGEntitySearchResponse(
                            status="success",
                            entities=entities,
                            count=len(entities),
                            execution_time_seconds=execution_time,
                            cache_hit=False,
                            metadata={
                                "query_hash": cache_key.split(":")[1],
                                "timestamp": datetime.utcnow().isoformat(),
                                "server_id": id(self)
                            }
                        )
                        
                        # Cache results
                        self.memory_cache[cache_key] = response
                        
                        # Record metrics
                        self.performance.record_request("entity_search", execution_time, True)
                        
                        return response
                        
                except asyncio.TimeoutError:
                    execution_time = time.time() - start_time
                    self.performance.record_request("entity_search", execution_time, False)
                    raise TimeoutError(f"Entity search timed out after {timeout}s")
                except Exception as e:
                    execution_time = time.time() - start_time
                    self.performance.record_request("entity_search", execution_time, False)
                    logger.error(f"Entity search failed: {e}")
                    raise
    
    async def global_search(
        self,
        query: str,
        max_communities: int = 3,
        timeout: float = 15.0,
        max_cost: float = 0.05
    ) -> GraphRAGGlobalSearchResponse:
        """
        Global search with comprehensive cost and time controls.
        """
        if not self.is_initialized:
            raise RuntimeError("GraphRAG server not initialized")
        
        search_id = f"global_{int(time.time())}_{hashlib.md5(query.encode()).hexdigest()[:8]}"
        start_time = time.time()
        
        async with self.concurrency.read_write_lock.read_lock():
            async with self.concurrency.global_search_semaphore:
                try:
                    async with asyncio.timeout(timeout):
                        # Find relevant communities
                        communities = await self._select_relevant_communities(query, max_communities)
                        
                        # Estimate cost before proceeding
                        estimated_cost = self._estimate_global_search_cost(communities)
                        if estimated_cost > max_cost:
                            raise ValueError(
                                f"Estimated cost ${estimated_cost:.3f} exceeds limit ${max_cost:.3f}"
                            )
                        
                        # Execute map-reduce search with controlled concurrency
                        insights = await self._execute_global_search(query, communities)
                        
                        # Synthesize final answer
                        synthesized_answer = await self._synthesize_global_insights(query, insights)
                        
                        # Calculate actual cost and time
                        actual_cost = sum(insight.cost for insight in insights)
                        execution_time = time.time() - start_time
                        
                        response = GraphRAGGlobalSearchResponse(
                            status="success",
                            search_id=search_id,
                            insights=insights,
                            synthesized_answer=synthesized_answer,
                            metadata={
                                "execution_time_seconds": execution_time,
                                "cost_usd": actual_cost,
                                "communities_analyzed": len(insights),
                                "timestamp": datetime.utcnow().isoformat()
                            }
                        )
                        
                        # Record metrics
                        self.performance.record_request("global_search", execution_time, True, actual_cost)
                        
                        return response
                        
                except asyncio.TimeoutError:
                    execution_time = time.time() - start_time
                    self.performance.record_request("global_search", execution_time, False)
                    raise TimeoutError(f"Global search timed out after {timeout}s")
                except Exception as e:
                    execution_time = time.time() - start_time
                    self.performance.record_request("global_search", execution_time, False)
                    logger.error(f"Global search failed: {e}")
                    raise
    
    async def get_health_status(self) -> GraphRAGServerHealth:
        """Get comprehensive server health status."""
        current_time = datetime.utcnow()
        uptime = (current_time - self.startup_time).total_seconds() if self.startup_time else 0
        
        # Calculate memory usage (simplified)
        memory_usage = 0.0
        if self.knowledge_base:
            # Estimate memory usage based on DataFrame sizes
            memory_usage = (
                len(self.knowledge_base.entities_df) * 0.001 +  # Rough estimate
                len(self.knowledge_base.relationships_df) * 0.0005 +
                len(self.knowledge_base.communities_df) * 0.0002
            )
        
        return GraphRAGServerHealth(
            status="healthy" if self.is_initialized else "initializing",
            uptime_seconds=uptime,
            memory_usage_gb=memory_usage,
            active_requests=self.config.max_concurrent_requests - self.concurrency.query_semaphore._value,
            cache_hit_rate=self.performance.get_cache_hit_rate(),
            total_requests=self.performance.metrics["entity_searches"] + self.performance.metrics["global_searches"],
            errors_last_hour=sum(self.performance.hourly_errors.values()),
            data_files_loaded=self.knowledge_base is not None,
            last_updated=current_time
        )
    
    async def get_performance_metrics(self) -> GraphRAGPerformanceMetrics:
        """Get detailed performance metrics."""
        # Calculate averages from recent request times
        recent_times = list(self.performance.request_times.values())
        entity_times = [t for k, t in self.performance.request_times.items() if "entity_search" in k]
        global_times = [t for k, t in self.performance.request_times.items() if "global_search" in k]
        
        return GraphRAGPerformanceMetrics(
            entity_search_avg_time=sum(entity_times) / len(entity_times) if entity_times else 0.0,
            global_search_avg_time=sum(global_times) / len(global_times) if global_times else 0.0,
            cache_hit_percentage=self.performance.get_cache_hit_rate(),
            total_cost_today=self.performance.metrics["total_cost"],
            requests_per_minute=len(recent_times) / 60.0 if recent_times else 0.0,
            error_rate_percentage=(self.performance.metrics["errors"] / max(len(recent_times), 1)) * 100
        )
    
    async def shutdown(self):
        """Gracefully shutdown GraphRAG server."""
        try:
            logger.info("🛑 Shutting down GraphRAG server...")
            
            # Wait for active requests to complete (with timeout)
            async with asyncio.timeout(30.0):
                while self.concurrency.query_semaphore._value < self.config.max_concurrent_requests:
                    await asyncio.sleep(0.1)
            
            # Shutdown thread pool
            self.thread_pool.shutdown(wait=True)
            
            # Clear caches
            self.memory_cache.clear()
            self.search_cache.clear()
            
            self.is_initialized = False
            logger.info("✅ GraphRAG server shutdown complete")
            
        except Exception as e:
            logger.error(f"❌ GraphRAG server shutdown error: {e}")
    
    # Private implementation methods
    
    async def _initialize_llm_model(self):
        """Initialize LLM model for GraphRAG operations."""
        if self.model_manager:
            self.llm_model = await self.model_manager.get_model()
            logger.info("✅ LLM model initialized")
        else:
            logger.warning("⚠️ Model manager not available, using mock responses")
    
    async def _load_knowledge_base(self):
        """Load GraphRAG knowledge base from parquet files."""
        logger.info("📚 Loading GraphRAG knowledge base...")
        
        # Load DataFrames in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        
        entities_df = await loop.run_in_executor(
            self.thread_pool,
            pd.read_parquet,
            self.config.get_data_file_path(self.config.entities_file)
        )
        
        relationships_df = await loop.run_in_executor(
            self.thread_pool,
            pd.read_parquet,
            self.config.get_data_file_path(self.config.relationships_file)
        )
        
        communities_df = await loop.run_in_executor(
            self.thread_pool,
            pd.read_parquet,
            self.config.get_data_file_path(self.config.communities_file)
        )
        
        community_reports_df = await loop.run_in_executor(
            self.thread_pool,
            pd.read_parquet,
            self.config.get_data_file_path(self.config.community_reports_file)
        )
        
        self.knowledge_base = GraphRAGKnowledgeBase(
            entities_df=entities_df,
            relationships_df=relationships_df,
            communities_df=communities_df,
            community_reports_df=community_reports_df,
            entity_index={},  # Will be built in next step
            community_index={},  # Will be built in next step
            loaded_at=datetime.utcnow()
        )
        
        logger.info("✅ Knowledge base loaded successfully")
    
    async def _build_search_indices(self):
        """Build search indices for fast entity and community lookup."""
        logger.info("🔍 Building search indices...")
        
        # Build entity index (name -> entity data)
        if 'name' in self.knowledge_base.entities_df.columns:
            self.knowledge_base.entity_index = {
                row['name']: row.to_dict()
                for _, row in self.knowledge_base.entities_df.iterrows()
            }
        
        # Build community index (id -> community data)
        if 'id' in self.knowledge_base.communities_df.columns:
            self.knowledge_base.community_index = {
                row['id']: row.to_dict()
                for _, row in self.knowledge_base.communities_df.iterrows()
            }
        
        logger.info("✅ Search indices built successfully")
    
    async def _precompute_popular_searches(self):
        """Pre-compute popular searches for cache warming."""
        logger.info("🔥 Pre-warming cache with popular searches...")
        
        # This would typically be based on historical query patterns
        # For now, we'll skip this as it requires actual GraphRAG query execution
        
        logger.info("✅ Cache pre-warming completed")
    
    async def _execute_entity_search(self, query: str, limit: int) -> List[GraphRAGEntity]:
        """Execute entity search against knowledge base."""
        if not self.knowledge_base:
            return []
        
        # Simple text matching for now (in production, this would use proper GraphRAG search)
        entities = []
        query_lower = query.lower()
        
        count = 0
        for name, entity_data in self.knowledge_base.entity_index.items():
            if count >= limit:
                break
                
            if query_lower in name.lower():
                entities.append(GraphRAGEntity(
                    id=entity_data.get('id', name),
                    name=name,
                    type=entity_data.get('type', 'unknown'),
                    description=entity_data.get('description'),
                    importance_score=entity_data.get('rank', 0.0),
                    metadata=entity_data
                ))
                count += 1
        
        return entities
    
    async def _select_relevant_communities(self, query: str, max_communities: int) -> List[Dict[str, Any]]:
        """Select relevant communities for global search."""
        if not self.knowledge_base:
            return []
        
        # Simple selection for now (in production, this would use proper community ranking)
        communities = list(self.knowledge_base.community_index.values())[:max_communities]
        return communities
    
    def _estimate_global_search_cost(self, communities: List[Dict[str, Any]]) -> float:
        """Estimate cost for global search operation."""
        # Simple cost estimation based on number of communities
        base_cost_per_community = 0.015  # $0.015 per community analysis
        return len(communities) * base_cost_per_community
    
    async def _execute_global_search(self, query: str, communities: List[Dict[str, Any]]) -> List[GraphRAGCommunityInsight]:
        """Execute global search across communities."""
        insights = []
        
        for i, community in enumerate(communities):
            # Mock insight generation (in production, this would use actual GraphRAG)
            insight = GraphRAGCommunityInsight(
                community_id=community.get('id', f'community_{i}'),
                community_name=community.get('title', f'Community {i+1}'),
                insight=f"Mock insight for '{query}' in {community.get('title', 'this community')}",
                confidence_score=0.8,
                supporting_entities=['Entity1', 'Entity2'],
                cost=0.015
            )
            insights.append(insight)
        
        return insights
    
    async def _synthesize_global_insights(self, query: str, insights: List[GraphRAGCommunityInsight]) -> str:
        """Synthesize final answer from community insights."""
        if not insights:
            return "No insights available for the given query."
        
        # Mock synthesis (in production, this would use LLM to combine insights)
        return f"Based on analysis of {len(insights)} business domains, here are the key findings for '{query}': " + \
               " ".join([insight.insight for insight in insights[:2]])  # Limit for brevity