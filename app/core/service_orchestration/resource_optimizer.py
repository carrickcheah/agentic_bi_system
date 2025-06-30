"""
Resource Optimizer - Optimizes database connections and resource allocation.
Self-contained resource optimization for service orchestration.
Zero external dependencies beyond module boundary.
"""

from typing import Dict, List, Optional, Any
import asyncio
from dataclasses import dataclass

try:
    from .config import settings
    from .orchestration_logging import setup_logger, performance_monitor, log_service_metrics
except ImportError:
    from config import settings
    from orchestration_logging import setup_logger, performance_monitor, log_service_metrics


@dataclass
class ResourceAllocation:
    """Resource allocation configuration for a service."""
    service_name: str
    cpu_allocation: float  # 0.0 to 1.0
    memory_allocation_mb: int
    connection_pool_size: int
    query_timeout_seconds: int
    priority_level: int  # 1 = highest priority


@dataclass
class OptimizationProfile:
    """Optimization profile based on complexity level."""
    complexity_level: str
    resource_multiplier: float
    parallel_factor: float
    cache_strategy: str
    timeout_multiplier: float


class ResourceOptimizer:
    """
    Optimizes database connections and resource allocation for service orchestration.
    Implements intelligent resource management based on investigation complexity.
    """
    
    def __init__(self):
        self.logger = setup_logger("resource_optimizer")
        self._optimization_profiles = {}
        self._resource_allocations = {}
        self._active_optimizations = {}
        self._initialized = False
        
        self.logger.info("Resource Optimizer initialized")
    
    async def initialize(self) -> None:
        """Initialize resource optimizer with complexity-based profiles."""
        if self._initialized:
            return
        
        try:
            # Define optimization profiles for each complexity level
            self._optimization_profiles = {
                "simple": OptimizationProfile(
                    complexity_level="simple",
                    resource_multiplier=1.0,
                    parallel_factor=1.0,
                    cache_strategy="basic",
                    timeout_multiplier=1.0
                ),
                "analytical": OptimizationProfile(
                    complexity_level="analytical",
                    resource_multiplier=1.5,
                    parallel_factor=1.3,
                    cache_strategy="enhanced",
                    timeout_multiplier=1.5
                ),
                "computational": OptimizationProfile(
                    complexity_level="computational",
                    resource_multiplier=2.0,
                    parallel_factor=1.7,
                    cache_strategy="aggressive",
                    timeout_multiplier=2.0
                ),
                "investigative": OptimizationProfile(
                    complexity_level="investigative",
                    resource_multiplier=3.0,
                    parallel_factor=2.0,
                    cache_strategy="maximum",
                    timeout_multiplier=3.0
                )
            }
            
            self._initialized = True
            self.logger.info("Resource Optimizer fully initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Resource Optimizer: {e}")
            raise
    
    @performance_monitor("connection_optimization")
    async def optimize_connections(
        self, 
        connections: Dict[str, Any], 
        complexity_score: Any
    ) -> Dict[str, Any]:
        """
        Optimize database connections based on complexity level.
        
        Args:
            connections: Dictionary of service connections to optimize
            complexity_score: ComplexityScore from Intelligence Module
            
        Returns:
            Optimized connection configurations
        """
        if not self._initialized:
            await self.initialize()
        
        complexity_level = complexity_score.level.value
        optimization_profile = self._optimization_profiles.get(complexity_level)
        
        if not optimization_profile:
            self.logger.warning(f"No optimization profile for complexity: {complexity_level}")
            return connections
        
        self.logger.info(f"Optimizing connections for {complexity_level} complexity")
        
        optimized_connections = {}
        
        for service_name, connection_config in connections.items():
            try:
                optimized_config = await self._optimize_service_connection(
                    service_name, connection_config, optimization_profile, complexity_score
                )
                optimized_connections[service_name] = optimized_config
                
                # Log optimization metrics
                log_service_metrics(service_name, {
                    "optimization_level": complexity_level,
                    "resource_multiplier": optimization_profile.resource_multiplier,
                    "cache_strategy": optimization_profile.cache_strategy
                })
                
            except Exception as e:
                self.logger.error(f"Failed to optimize {service_name}: {e}")
                optimized_connections[service_name] = connection_config  # Use original config
        
        return optimized_connections
    
    async def _optimize_service_connection(
        self,
        service_name: str,
        connection_config: Dict[str, Any],
        optimization_profile: OptimizationProfile,
        complexity_score: Any
    ) -> Dict[str, Any]:
        """Optimize a specific service connection."""
        
        optimized_config = connection_config.copy()
        
        # Service-specific optimizations
        if service_name == "mariadb":
            optimized_config = await self._optimize_mariadb_connection(
                optimized_config, optimization_profile, complexity_score
            )
        elif service_name == "postgresql":
            optimized_config = await self._optimize_postgresql_connection(
                optimized_config, optimization_profile, complexity_score
            )
        elif service_name == "lancedb":
            optimized_config = await self._optimize_lancedb_connection(
                optimized_config, optimization_profile, complexity_score
            )
        elif service_name == "graphrag":
            optimized_config = await self._optimize_graphrag_connection(
                optimized_config, optimization_profile, complexity_score
            )
        
        # Apply general optimizations
        optimized_config = self._apply_general_optimizations(
            optimized_config, optimization_profile
        )
        
        return optimized_config
    
    async def _optimize_mariadb_connection(
        self,
        config: Dict[str, Any],
        profile: OptimizationProfile,
        complexity_score: Any
    ) -> Dict[str, Any]:
        """Optimize MariaDB connection for business data operations."""
        
        # Adjust connection pool size based on complexity
        base_pool_size = config.get("connection_pool_size", 5)
        optimized_pool_size = int(base_pool_size * profile.resource_multiplier)
        config["connection_pool_size"] = min(optimized_pool_size, 20)  # Cap at 20
        
        # Adjust query timeout
        base_timeout = config.get("query_timeout", 30)
        config["query_timeout"] = int(base_timeout * profile.timeout_multiplier)
        
        # Business data specific optimizations
        config["business_data_optimizations"].update({
            "enable_query_planning": profile.cache_strategy in ["enhanced", "aggressive", "maximum"],
            "index_optimization": profile.resource_multiplier >= 1.5,
            "table_scan_optimization": profile.resource_multiplier >= 2.0,
            "query_result_caching": profile.cache_strategy != "basic"
        })
        
        # Estimated query volume optimization
        if complexity_score.estimated_queries > 10:
            config["optimization_features"]["batch_processing"] = True
            config["optimization_features"]["connection_reuse"] = True
        
        return config
    
    async def _optimize_postgresql_connection(
        self,
        config: Dict[str, Any],
        profile: OptimizationProfile,
        complexity_score: Any
    ) -> Dict[str, Any]:
        """Optimize PostgreSQL connection for memory and caching."""
        
        # Memory optimization based on complexity
        if profile.cache_strategy == "maximum":
            config["cache_optimizations"]["cache_ttl_minutes"] = 120
            config["cache_optimizations"]["preload_cache"] = True
        elif profile.cache_strategy == "aggressive":
            config["cache_optimizations"]["cache_ttl_minutes"] = 90
            config["cache_optimizations"]["preload_cache"] = True
        elif profile.cache_strategy == "enhanced":
            config["cache_optimizations"]["cache_ttl_minutes"] = 60
        
        # Session management optimization
        config["session_management"]["session_timeout_minutes"] = int(
            30 * profile.timeout_multiplier
        )
        
        # Investigation storage optimization
        if complexity_score.estimated_duration_minutes > 30:
            config["session_management"]["persistent_investigation_storage"] = True
            config["session_management"]["investigation_backup_enabled"] = True
        
        return config
    
    async def _optimize_lancedb_connection(
        self,
        config: Dict[str, Any],
        profile: OptimizationProfile,
        complexity_score: Any
    ) -> Dict[str, Any]:
        """Optimize LanceDB connection for vector search operations."""
        
        # Vector search optimization based on complexity
        if profile.resource_multiplier >= 2.0:
            config["vector_search_config"]["max_results"] = 200
            config["vector_search_config"]["enable_parallel_search"] = True
        elif profile.resource_multiplier >= 1.5:
            config["vector_search_config"]["max_results"] = 150
        
        # Embedding optimization
        config["embedding_optimizations"].update({
            "preload_embeddings": profile.cache_strategy in ["aggressive", "maximum"],
            "similarity_cache": profile.cache_strategy != "basic",
            "pattern_matching_cache": profile.resource_multiplier >= 1.5,
            "embedding_batch_size": int(100 * profile.parallel_factor)
        })
        
        # Similarity threshold adjustment
        if complexity_score.confidence < 0.7:
            config["similarity_threshold"] = 0.7  # Lower threshold for uncertain queries
        
        return config
    
    async def _optimize_graphrag_connection(
        self,
        config: Dict[str, Any],
        profile: OptimizationProfile,
        complexity_score: Any
    ) -> Dict[str, Any]:
        """Optimize GraphRAG connection for knowledge graph operations."""
        
        # Knowledge graph depth optimization
        base_depth = config.get("knowledge_graph_depth", 3)
        config["knowledge_graph_depth"] = min(
            int(base_depth * profile.resource_multiplier), 7
        )  # Cap at depth 7
        
        # Analysis configuration optimization
        config["analysis_config"].update({
            "max_traversal_depth": min(int(5 * profile.resource_multiplier), 10),
            "enable_inference": profile.resource_multiplier >= 1.5,
            "relationship_weighting": profile.cache_strategy in ["enhanced", "aggressive", "maximum"],
            "entity_scoring": profile.resource_multiplier >= 2.0,
            "parallel_analysis": profile.parallel_factor >= 1.5
        })
        
        # Graph optimization based on estimated duration
        if complexity_score.estimated_duration_minutes > 60:
            config["graph_optimizations"]["enable_graph_cache"] = True
            config["graph_optimizations"]["persistent_graph_storage"] = True
        
        return config
    
    def _apply_general_optimizations(
        self,
        config: Dict[str, Any],
        profile: OptimizationProfile
    ) -> Dict[str, Any]:
        """Apply general optimizations across all service types."""
        
        # Add performance monitoring
        config["performance_monitoring"] = {
            "enable_metrics_collection": settings.enable_service_metrics,
            "enable_performance_logging": settings.enable_performance_logging,
            "optimization_level": profile.complexity_level,
            "resource_multiplier": profile.resource_multiplier
        }
        
        # Add parallel execution configuration
        config["parallel_execution"] = {
            "enable_parallel_queries": profile.parallel_factor > 1.0,
            "parallel_factor": profile.parallel_factor,
            "max_concurrent_operations": int(5 * profile.parallel_factor)
        }
        
        # Add timeout configuration
        config["timeout_configuration"] = {
            "operation_timeout_seconds": int(settings.connection_timeout_seconds * profile.timeout_multiplier),
            "health_check_timeout_seconds": settings.health_check_timeout_seconds,
            "coordination_timeout_seconds": settings.service_coordination_timeout_seconds
        }
        
        return config
    
    async def calculate_resource_allocation(
        self,
        service_types: List[str],
        complexity_score: Any
    ) -> Dict[str, ResourceAllocation]:
        """Calculate optimal resource allocation for services."""
        
        complexity_level = complexity_score.level.value
        optimization_profile = self._optimization_profiles.get(complexity_level)
        
        if not optimization_profile:
            raise ValueError(f"No optimization profile for complexity: {complexity_level}")
        
        allocations = {}
        total_services = len(service_types)
        
        for i, service_type in enumerate(service_types):
            # Priority based on service order (MariaDB highest, GraphRAG lowest)
            priority = i + 1
            
            # CPU allocation based on service type and complexity
            base_cpu = 0.5 if service_type == "mariadb" else 0.3
            cpu_allocation = min(base_cpu * optimization_profile.resource_multiplier, 1.0)
            
            # Memory allocation
            base_memory = {
                "mariadb": 512,
                "postgresql": 256,
                "lancedb": 1024,
                "graphrag": 2048
            }.get(service_type, 256)
            
            memory_allocation = int(base_memory * optimization_profile.resource_multiplier)
            
            # Connection pool size
            base_pool = {
                "mariadb": 10,
                "postgresql": 5,
                "lancedb": 3,
                "graphrag": 2
            }.get(service_type, 5)
            
            pool_size = min(int(base_pool * optimization_profile.resource_multiplier), 20)
            
            # Query timeout
            base_timeout = {
                "mariadb": 30,
                "postgresql": 45,
                "lancedb": 60,
                "graphrag": 120
            }.get(service_type, 30)
            
            query_timeout = int(base_timeout * optimization_profile.timeout_multiplier)
            
            allocations[service_type] = ResourceAllocation(
                service_name=service_type,
                cpu_allocation=cpu_allocation,
                memory_allocation_mb=memory_allocation,
                connection_pool_size=pool_size,
                query_timeout_seconds=query_timeout,
                priority_level=priority
            )
            
            self.logger.debug(f"Resource allocation for {service_type}: CPU={cpu_allocation:.2f}, Memory={memory_allocation}MB")
        
        return allocations
    
    async def health_check(self) -> bool:
        """Check health of resource optimizer."""
        if not self._initialized:
            return False
        
        # Check if optimization profiles are loaded
        required_profiles = ["simple", "analytical", "computational", "investigative"]
        
        for profile_name in required_profiles:
            if profile_name not in self._optimization_profiles:
                self.logger.error(f"Missing optimization profile: {profile_name}")
                return False
        
        return True
    
    async def cleanup(self) -> None:
        """Cleanup resource optimizer."""
        try:
            self._active_optimizations.clear()
            self._resource_allocations.clear()
            self._initialized = False
            
            self.logger.info("Resource Optimizer cleaned up")
            
        except Exception as e:
            self.logger.error(f"Error during resource optimizer cleanup: {e}")
