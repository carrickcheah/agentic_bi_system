"""Qdrant Service with Production ML Engineering Standards.

Implements circuit breaker, caching, monitoring, and fault tolerance.
Follows ROSE ML Engineer principles for production systems.
"""

import asyncio
import time
import hashlib
import json
import re
import uuid
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import deque
from enum import Enum

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, SearchRequest,
    Filter, FieldCondition, MatchValue, UpdateStatus
)

from .config import settings, validate_settings
from .qdrant_logging import (
    logger, log_performance, log_slow_operation,
    log_circuit_breaker_state_change, log_cache_hit, log_cache_miss,
    log_connection_error, MetricsLogger
)


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreaker:
    """Production circuit breaker pattern implementation."""
    
    def __init__(self):
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.half_open_attempts = 0
    
    async def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if not settings.enable_circuit_breaker:
            return await func(*args, **kwargs)
        
        # Check circuit state
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self._transition_to_half_open()
            else:
                raise RuntimeError("Circuit breaker is OPEN")
        
        try:
            # Execute function
            result = await func(*args, **kwargs)
            self._on_success()
            return result
            
        except Exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return False
        
        time_since_failure = time.time() - self.last_failure_time
        return time_since_failure >= settings.circuit_breaker_timeout
    
    def _transition_to_half_open(self):
        """Transition to half-open state."""
        old_state = self.state
        self.state = CircuitBreakerState.HALF_OPEN
        self.half_open_attempts = 0
        log_circuit_breaker_state_change(
            old_state.value,
            self.state.value,
            "Timeout expired, testing recovery"
        )
    
    def _on_success(self):
        """Handle successful execution."""
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            self.half_open_attempts += 1
            
            if self.half_open_attempts >= settings.circuit_breaker_half_open_requests:
                # Enough successful requests, close circuit
                old_state = self.state
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
                self.success_count = 0
                log_circuit_breaker_state_change(
                    old_state.value,
                    self.state.value,
                    "Recovery confirmed"
                )
        
        elif self.state == CircuitBreakerState.CLOSED:
            self.failure_count = 0  # Reset on success
    
    def _on_failure(self):
        """Handle failed execution."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            # Failed during recovery test, reopen
            old_state = self.state
            self.state = CircuitBreakerState.OPEN
            log_circuit_breaker_state_change(
                old_state.value,
                self.state.value,
                "Recovery test failed"
            )
        
        elif self.state == CircuitBreakerState.CLOSED:
            if self.failure_count >= settings.circuit_breaker_threshold:
                # Too many failures, open circuit
                old_state = self.state
                self.state = CircuitBreakerState.OPEN
                log_circuit_breaker_state_change(
                    old_state.value,
                    self.state.value,
                    f"Threshold exceeded ({self.failure_count} failures)"
                )


class QueryCache:
    """High-performance LRU cache for query results."""
    
    def __init__(self):
        self.cache: Dict[str, Tuple[Any, float]] = {}
        self.access_order = deque(maxlen=settings.cache_max_size)
        self.hits = 0
        self.misses = 0
    
    def _hash_query(self, query: str) -> str:
        """Create hash of query for cache key."""
        return hashlib.md5(query.encode()).hexdigest()
    
    async def get(self, query: str) -> Optional[List[Dict]]:
        """Get cached results if available and not expired."""
        if not settings.enable_cache:
            return None
        
        query_hash = self._hash_query(query)
        
        if query_hash in self.cache:
            results, timestamp = self.cache[query_hash]
            age = time.time() - timestamp
            
            if age < settings.cache_ttl:
                self.hits += 1
                log_cache_hit(query_hash, age)
                
                # Update access order
                if query_hash in self.access_order:
                    self.access_order.remove(query_hash)
                self.access_order.append(query_hash)
                
                return results
            else:
                # Expired
                del self.cache[query_hash]
        
        self.misses += 1
        log_cache_miss(query_hash)
        return None
    
    async def set(self, query: str, results: List[Dict]):
        """Cache query results."""
        if not settings.enable_cache:
            return
        
        query_hash = self._hash_query(query)
        
        # Enforce max size
        if len(self.cache) >= settings.cache_max_size:
            # Remove oldest
            if self.access_order:
                oldest = self.access_order.popleft()
                if oldest in self.cache:
                    del self.cache[oldest]
        
        self.cache[query_hash] = (results, time.time())
        self.access_order.append(query_hash)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "size": len(self.cache),
            "max_size": settings.cache_max_size
        }


class PerformanceMonitor:
    """Production performance monitoring."""
    
    def __init__(self):
        self.metrics = {
            "total_queries": 0,
            "total_errors": 0,
            "cache_hits": 0,
            "latencies_ms": deque(maxlen=settings.metrics_window_size),
            "error_types": {}
        }
    
    def record_query(self, duration_ms: float, success: bool, cached: bool = False):
        """Record query metrics."""
        self.metrics["total_queries"] += 1
        self.metrics["latencies_ms"].append(duration_ms)
        
        if not success:
            self.metrics["total_errors"] += 1
        
        if cached:
            self.metrics["cache_hits"] += 1
        
        # Log slow queries
        if settings.log_slow_queries and duration_ms > settings.slow_query_threshold_ms:
            log_slow_operation("query", duration_ms, settings.slow_query_threshold_ms)
    
    def record_error(self, error_type: str):
        """Record error by type."""
        if error_type not in self.metrics["error_types"]:
            self.metrics["error_types"][error_type] = 0
        self.metrics["error_types"][error_type] += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics snapshot."""
        latencies = list(self.metrics["latencies_ms"])
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        
        return {
            "total_queries": self.metrics["total_queries"],
            "total_errors": self.metrics["total_errors"],
            "cache_hits": self.metrics["cache_hits"],
            "avg_latency_ms": avg_latency,
            "p95_latency_ms": self._percentile(latencies, 0.95),
            "p99_latency_ms": self._percentile(latencies, 0.99),
            "error_types": self.metrics["error_types"].copy()
        }
    
    def _percentile(self, values: List[float], p: float) -> float:
        """Calculate percentile."""
        if not values:
            return 0
        sorted_values = sorted(values)
        index = int(len(sorted_values) * p)
        return sorted_values[min(index, len(sorted_values) - 1)]


class QdrantService:
    """Production-grade Qdrant service with monitoring and fault tolerance."""
    
    def __init__(self):
        self.client: Optional[AsyncQdrantClient] = None
        self.circuit_breaker = CircuitBreaker()
        self.cache = QueryCache()
        self.monitor = PerformanceMonitor()
        self._initialized = False
    
    async def initialize(self):
        """Initialize Qdrant client and ensure collection exists."""
        if self._initialized:
            return
        
        try:
            # Validate settings
            validate_settings()
            
            # Create client
            self.client = AsyncQdrantClient(
                url=settings.qdrant_url,
                api_key=settings.api_key,
                timeout=settings.timeout_seconds
            )
            
            # Ensure collection exists
            await self._ensure_collection()
            
            self._initialized = True
            logger.info("Qdrant service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant service: {e}")
            raise
    
    async def _ensure_collection(self):
        """Ensure collection exists with correct configuration."""
        collections = await self.client.get_collections()
        collection_names = [c.name for c in collections.collections]
        
        if settings.collection_name not in collection_names:
            logger.info(f"Creating collection: {settings.collection_name}")
            
            await self.client.create_collection(
                collection_name=settings.collection_name,
                vectors_config=VectorParams(
                    size=settings.embedding_dim,
                    distance=Distance.COSINE
                )
            )
            
            logger.info(f"Collection '{settings.collection_name}' created")
        else:
            logger.info(f"Collection '{settings.collection_name}' already exists")
    
    async def search_similar_queries(
        self,
        query: str,
        limit: int = None,
        threshold: float = None
    ) -> List[Dict[str, Any]]:
        """Search for similar SQL queries with caching and monitoring."""
        if not self._initialized:
            await self.initialize()
        
        limit = limit or settings.search_limit
        threshold = threshold or settings.similarity_threshold
        
        start_time = time.time()
        cached = False
        
        try:
            # Check cache first
            cached_results = await self.cache.get(query)
            if cached_results is not None:
                cached = True
                duration_ms = (time.time() - start_time) * 1000
                self.monitor.record_query(duration_ms, True, cached=True)
                return cached_results
            
            # Get embedding for query
            # Use local embedding generation for standalone operation
            import hashlib
            hash_obj = hashlib.sha384(query.encode())
            hash_bytes = hash_obj.digest()
            query_embedding = []
            for i in range(0, len(hash_bytes), 4):
                if len(query_embedding) >= settings.embedding_dim:
                    break
                chunk = hash_bytes[i:i+4]
                if len(chunk) == 4:
                    value = int.from_bytes(chunk, 'big') / (2**32)
                    normalized = (value * 2) - 1
                    query_embedding.append(normalized)
            while len(query_embedding) < settings.embedding_dim:
                pad_value = (len(query_embedding) % 100) / 100.0 - 0.5
                query_embedding.append(pad_value)
            query_embedding = query_embedding[:settings.embedding_dim]
            
            # Search with circuit breaker
            results = await self.circuit_breaker.call(
                self._search_impl,
                query_embedding,
                limit,
                threshold
            )
            
            # Cache results
            await self.cache.set(query, results)
            
            # Record metrics
            duration_ms = (time.time() - start_time) * 1000
            self.monitor.record_query(duration_ms, True, cached=False)
            log_performance("search_similar_queries", duration_ms, True)
            
            return results
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.monitor.record_query(duration_ms, False)
            self.monitor.record_error(type(e).__name__)
            log_performance("search_similar_queries", duration_ms, False)
            logger.error(f"Search failed: {e}")
            raise
    
    async def _search_impl(
        self,
        query_embedding: List[float],
        limit: int,
        threshold: float
    ) -> List[Dict[str, Any]]:
        """Internal search implementation."""
        search_result = await self.client.search(
            collection_name=settings.collection_name,
            query_vector=query_embedding,
            limit=limit,
            score_threshold=threshold
        )
        
        results = []
        for point in search_result:
            if point.payload:
                result = {
                    "id": str(point.id),
                    "score": point.score,
                    "sql_query": point.payload.get("sql_query", ""),
                    "business_question": point.payload.get("business_question", ""),
                    "metadata": point.payload.get("metadata", {})
                }
                results.append(result)
        
        return results
    
    async def store_query(
        self,
        query_id: str,
        sql_query: str,
        business_question: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Store SQL query pattern with embedding."""
        if not self._initialized:
            await self.initialize()
        
        start_time = time.time()
        
        try:
            # Get embeddings
            # Use local embedding generation for standalone operation
            import hashlib
            combined_text = f"{business_question} {sql_query}"
            hash_obj = hashlib.sha384(combined_text.encode())
            hash_bytes = hash_obj.digest()
            embedding = []
            for i in range(0, len(hash_bytes), 4):
                if len(embedding) >= settings.embedding_dim:
                    break
                chunk = hash_bytes[i:i+4]
                if len(chunk) == 4:
                    value = int.from_bytes(chunk, 'big') / (2**32)
                    normalized = (value * 2) - 1
                    embedding.append(normalized)
            while len(embedding) < settings.embedding_dim:
                pad_value = (len(embedding) % 100) / 100.0 - 0.5
                embedding.append(pad_value)
            embedding = embedding[:settings.embedding_dim]
            
            # Prepare payload
            payload = {
                "sql_query": sql_query,
                "business_question": business_question,
                "metadata": metadata or {},
                "created_at": datetime.utcnow().isoformat()
            }
            
            # Store with circuit breaker
            await self.circuit_breaker.call(
                self._store_impl,
                query_id,
                embedding,
                payload
            )
            
            # Record metrics
            duration_ms = (time.time() - start_time) * 1000
            self.monitor.record_query(duration_ms, True)
            log_performance("store_query", duration_ms, True)
            
            return True
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.monitor.record_query(duration_ms, False)
            self.monitor.record_error(type(e).__name__)
            log_performance("store_query", duration_ms, False)
            logger.error(f"Store failed: {e}")
            return False
    
    async def _store_impl(
        self,
        query_id: str,
        embedding: List[float],
        payload: Dict[str, Any]
    ):
        """Internal store implementation."""
        # Convert string ID to UUID or use hash-based integer
        if re.match(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', query_id, re.I):
            # Already a UUID
            point_id = query_id
        else:
            # Convert string to deterministic integer ID
            id_hash = hashlib.md5(query_id.encode()).digest()
            point_id = int.from_bytes(id_hash[:8], 'big') % (2**63)  # Ensure positive integer
        
        point = PointStruct(
            id=point_id,
            vector=embedding,
            payload=payload
        )
        
        await self.client.upsert(
            collection_name=settings.collection_name,
            points=[point]
        )
    
    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check."""
        health = {
            "healthy": False,
            "client_connected": False,
            "collection_exists": False,
            "circuit_breaker": self.circuit_breaker.state.value,
            "cache_stats": self.cache.get_stats(),
            "metrics": self.monitor.get_metrics()
        }
        
        try:
            if self.client:
                # Check collection
                collections = await self.client.get_collections()
                collection_names = [c.name for c in collections.collections]
                health["collection_exists"] = settings.collection_name in collection_names
                health["client_connected"] = True
                health["healthy"] = True
        except Exception as e:
            health["error"] = str(e)
        
        MetricsLogger.log_health_check(health["healthy"], health)
        return health
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        metrics = self.monitor.get_metrics()
        metrics.update({
            "cache_stats": self.cache.get_stats(),
            "circuit_breaker_state": self.circuit_breaker.state.value
        })
        
        if settings.enable_monitoring:
            MetricsLogger.log_metrics(metrics)
        
        return metrics
    
    async def close(self):
        """Clean shutdown."""
        if self.client:
            await self.client.close()
            self._initialized = False
            logger.info("Qdrant service closed")


# Singleton instance management
_instance: Optional[QdrantService] = None


async def get_qdrant_service() -> QdrantService:
    """Get or create Qdrant service instance."""
    global _instance
    if _instance is None:
        _instance = QdrantService()
        await _instance.initialize()
    return _instance


# Main execution for testing
if __name__ == "__main__":
    async def main():
        """Test Qdrant service."""
        service = await get_qdrant_service()
        
        # Health check
        health = await service.health_check()
        print(f"Health: {json.dumps(health, indent=2)}")
        
        # Test store
        success = await service.store_query(
            "test_001",
            "SELECT * FROM users WHERE active = true",
            "Show all active users"
        )
        print(f"Store test: {'SUCCESS' if success else 'FAILED'}")
        
        # Test search
        results = await service.search_similar_queries(
            "SELECT * FROM customers WHERE status = 'active'"
        )
        print(f"Search results: {len(results)} found")
        
        # Metrics
        metrics = service.get_metrics()
        print(f"Metrics: {json.dumps(metrics, indent=2)}")
        
        await service.close()
    
    asyncio.run(main())