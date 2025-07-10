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
            
            # Import OptimizersConfigDiff for indexing configuration
            from qdrant_client.models import OptimizersConfigDiff
            
            await self.client.create_collection(
                collection_name=settings.collection_name,
                vectors_config=VectorParams(
                    size=settings.embedding_dim,
                    distance=Distance.COSINE
                ),
                # Force indexing even with small datasets
                optimizers_config=OptimizersConfigDiff(
                    indexing_threshold=0  # Index immediately, don't wait for 20k points
                )
            )
            
            logger.info(f"Collection '{settings.collection_name}' created with immediate indexing")
        else:
            logger.info(f"Collection '{settings.collection_name}' already exists")
            
            # Update existing collection to enable immediate indexing
            try:
                from qdrant_client.models import OptimizersConfigDiff
                await self.client.update_collection(
                    collection_name=settings.collection_name,
                    optimizers_config=OptimizersConfigDiff(
                        indexing_threshold=0  # Force immediate indexing
                    )
                )
                logger.info(f"Updated collection '{settings.collection_name}' to enable immediate indexing")
            except Exception as e:
                logger.warning(f"Could not update collection indexing config: {e}")
    
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
        
        logger.info(f"Qdrant search started - Query: '{query[:50]}...', Limit: {limit}, Threshold: {threshold}")
        
        start_time = time.time()
        cached = False
        
        try:
            # Check cache first
            cached_results = await self.cache.get(query)
            if cached_results is not None:
                cached = True
                duration_ms = (time.time() - start_time) * 1000
                self.monitor.record_query(duration_ms, True, cached=True)
                logger.info(f"Qdrant cache hit - Returning {len(cached_results)} cached results")
                return cached_results
            
            # Get embedding for query using OpenAI embeddings
            try:
                from model import create_embedding_model
                embedding_model = create_embedding_model()
                query_embedding = await embedding_model.embed_text_async(query)
                logger.debug(f"Generated OpenAI embedding for query, dim: {len(query_embedding)}")
            except Exception as e:
                logger.error(f"Failed to generate OpenAI embeddings: {e}")
                raise RuntimeError(f"Embedding generation failed: {e}")
            
            # Search with circuit breaker
            results = await self.circuit_breaker.call(
                self._search_impl,
                query_embedding,
                limit,
                threshold
            )
            
            # Cache results
            await self.cache.set(query, results)
            
            # Log results summary
            logger.info(f"Qdrant search completed - Found {len(results)} matches")
            if results:
                best_score = results[0].get('score', 0)
                logger.info(f"Best match score: {best_score:.3f}")
            
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
            # Get embeddings using OpenAI embeddings
            # IMPORTANT: Only embed the business question for better matching
            # Do NOT combine with SQL query - that causes search mismatches
            try:
                from model import create_embedding_model
                embedding_model = create_embedding_model()
                embedding = await embedding_model.embed_text_async(business_question)
                logger.debug(f"Generated OpenAI embedding for ingestion, dim: {len(embedding)}")
            except Exception as e:
                logger.error(f"Failed to generate OpenAI embeddings: {e}")
                raise RuntimeError(f"Embedding generation failed: {e}")
            
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
            
            # Force index update for small collections (development mode)
            if settings.enable_dev_mode:
                try:
                    # Get collection info to check point count
                    info = await self.client.get_collection(settings.collection_name)
                    if info.points_count < 100:  # Only for small dev collections
                        logger.info("Forcing index update for development collection")
                        # Trigger optimization to build index
                        await self.client.update_collection(
                            collection_name=settings.collection_name,
                            optimizers_config={"indexing_threshold": 0}
                        )
                except Exception as e:
                    logger.debug(f"Could not force index update: {e}")
            
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
    
    async def ingest_from_directory(self, directory_path: Optional[str] = None) -> Dict[str, int]:
        """Ingest all JSON pattern files from directory with auto-enrichment.
        
        Args:
            directory_path: Path to directory with JSON files. Uses settings.file_path if not provided.
            
        Returns:
            Dictionary with ingestion statistics
        """
        import json
        from pathlib import Path
        from datetime import datetime
        from .auto_enrich import PatternEnricher
        
        path = Path(directory_path or settings.file_path)
        if not path.exists():
            raise ValueError(f"Directory not found: {path}")
        
        stats = {
            "total_files": 0,
            "total_entries": 0,
            "success": 0,
            "failed": 0,
            "files_processed": []
        }
        
        # Initialize enricher
        enricher = PatternEnricher()
        
        # Find all JSON files
        json_files = list(path.glob("*.json"))
        stats["total_files"] = len(json_files)
        
        logger.info(f"Found {len(json_files)} JSON files in {path}")
        
        for json_file in json_files:
            try:
                logger.info(f"Processing: {json_file.name}")
                
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Handle single object or array
                entries = [data] if isinstance(data, dict) else data
                
                for entry in entries:
                    stats["total_entries"] += 1
                    
                    # AUTO-ENRICH the pattern data
                    enriched_entry = enricher.enrich_pattern(entry, json_file.name)
                    
                    # Extract query components from enriched data
                    query_content = enriched_entry.get("query_content", {})
                    sql_query = query_content.get("sql_query")
                    business_question = query_content.get("business_question")
                    
                    if not sql_query or not business_question:
                        logger.warning(f"Skipping entry - missing required fields")
                        stats["failed"] += 1
                        continue
                    
                    # Use enriched ID
                    query_id = enriched_entry.get("_id")
                    
                    # Extract metadata from enriched entry
                    metadata = self._extract_metadata(enriched_entry)
                    metadata["source_file"] = json_file.name
                    metadata["ingested_at"] = datetime.utcnow().isoformat()
                    
                    # Store query with enriched data
                    success = await self.store_query(
                        query_id=str(query_id),
                        sql_query=sql_query,
                        business_question=business_question,
                        metadata=metadata
                    )
                    
                    if success:
                        stats["success"] += 1
                        logger.info(f"Successfully ingested enriched pattern: {query_id}")
                    else:
                        stats["failed"] += 1
                
                stats["files_processed"].append(json_file.name)
                
            except Exception as e:
                logger.error(f"Error processing {json_file.name}: {e}")
                stats["failed"] += 1
        
        logger.info(f"Ingestion complete: {stats['success']}/{stats['total_entries']} successful")
        return stats
    
    def _extract_metadata(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """Extract ALL metadata from pattern entry - including nulls and zeros."""
        metadata = {}
        
        # Copy ALL fields except the ones we handle separately
        exclude_keys = {"_id", "query_content", "sql_query", "business_question"}
        
        for key, value in entry.items():
            if key not in exclude_keys:
                metadata[key] = value
        
        # Add query_intent and query_type from query_content
        query_content = entry.get("query_content", {})
        if query_content:
            metadata["query_intent"] = query_content.get("query_intent")
            metadata["query_type"] = query_content.get("query_type")
        
        # Return ALL values including None and 0
        return metadata
    
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


# Main execution for batch ingestion
if __name__ == "__main__":
    async def main():
        """Perform batch ingestion from configured directory."""
        service = await get_qdrant_service()
        
        print(f"\nStarting batch ingestion from: {settings.file_path}")
        print("=" * 60)
        
        try:
            stats = await service.ingest_from_directory()
            
            print("\nIngestion Results:")
            print(f"  Total files: {stats['total_files']}")
            print(f"  Total entries: {stats['total_entries']}")
            print(f"  Successful: {stats['success']}")
            print(f"  Failed: {stats['failed']}")
            if stats['total_entries'] > 0:
                print(f"  Success rate: {stats['success']/stats['total_entries']*100:.1f}%")
            print(f"\nFiles processed: {', '.join(stats['files_processed'])}")
            
            return 0 if stats['failed'] == 0 else 1
            
        except Exception as e:
            logger.error(f"Ingestion failed: {e}")
            return 1
        finally:
            await service.close()
    
    asyncio.run(main())