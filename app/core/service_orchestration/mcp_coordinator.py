"""
MCP Service Coordinator - Manages MCP client connections and service activation.
Coordinates with existing FastMCP client manager for database operations.
Zero external dependencies beyond module boundary.
"""

from typing import Dict, List, Optional, Any
import asyncio
import json
from pathlib import Path

try:
    from .config import settings
    from .orchestration_logging import setup_logger, performance_monitor, log_service_metrics
except ImportError:
    from config import settings
    from orchestration_logging import setup_logger, performance_monitor, log_service_metrics


class MCPServiceCoordinator:
    """
    Coordinates MCP database services for Phase 3 orchestration.
    Interfaces with existing FastMCP client manager.
    """
    
    def __init__(self):
        self.logger = setup_logger("mcp_coordinator")
        self._service_connections = {}
        self._service_health = {}
        self._initialized = False
        
        # Service configuration mapping
        self._service_config = {
            "mariadb": {
                "service_name": settings.mariadb_service_name,
                "capabilities": ["business_data", "sql_queries", "transaction_support"],
                "optimization_priority": 1
            },
            "postgresql": {
                "service_name": settings.postgres_service_name,
                "capabilities": ["memory_cache", "investigation_storage", "session_management"],
                "optimization_priority": 2
            },
            "qdrant": {
                "service_name": settings.qdrant_service_name,
                "capabilities": ["vector_search", "embeddings", "similarity_queries"],
                "optimization_priority": 3
            },
            "graphrag": {
                "service_name": settings.graphrag_service_name,
                "capabilities": ["knowledge_graph", "entity_resolution", "relationship_analysis"],
                "optimization_priority": 4
            }
        }
        
        self.logger.info("MCP Service Coordinator initialized")
    
    async def initialize(self) -> None:
        """Initialize MCP service coordinator."""
        if self._initialized:
            return
        
        try:
            # Load MCP configuration
            mcp_config = await self._load_mcp_configuration()
            
            # Validate service availability
            await self._validate_service_availability()
            
            self._initialized = True
            self.logger.info("MCP Service Coordinator fully initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize MCP Service Coordinator: {e}")
            raise
    
    async def _load_mcp_configuration(self) -> Dict[str, Any]:
        """Load MCP configuration from file."""
        try:
            config_path = Path(settings.mcp_config_path)
            if not config_path.exists():
                raise FileNotFoundError(f"MCP config not found: {config_path}")
            
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            self.logger.info(f"Loaded MCP configuration from {config_path}")
            return config
            
        except Exception as e:
            self.logger.error(f"Failed to load MCP configuration: {e}")
            raise
    
    async def _validate_service_availability(self) -> None:
        """Validate that required MCP services are available."""
        required_services = [
            settings.mariadb_service_name,
            settings.postgres_service_name,
            settings.qdrant_service_name,
            settings.graphrag_service_name
        ]
        
        # In a real implementation, this would check actual MCP service availability
        # For now, we'll simulate service validation
        for service_name in required_services:
            self._service_health[service_name] = True
            self.logger.debug(f"Validated service availability: {service_name}")
    
    @performance_monitor("service_preparation")
    async def prepare_service(
        self, 
        service_type: str, 
        optimization_settings: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Prepare a specific service for investigation execution.
        
        Args:
            service_type: Type of service to prepare (mariadb, postgresql, Qdrant, graphrag)
            optimization_settings: Service-specific optimization parameters
            
        Returns:
            Service connection and optimization context
        """
        if service_type not in self._service_config:
            raise ValueError(f"Unknown service type: {service_type}")
        
        service_config = self._service_config[service_type]
        service_name = service_config["service_name"]
        
        self.logger.info(f"Preparing service: {service_type} ({service_name})")
        
        try:
            # Service-specific preparation
            if service_type == "mariadb":
                connection_context = await self._prepare_mariadb_service(optimization_settings)
            elif service_type == "postgresql":
                connection_context = await self._prepare_postgresql_service(optimization_settings)
            elif service_type == "qdrant":
                connection_context = await self._prepare_qdrant_service(optimization_settings)
            elif service_type == "graphrag":
                connection_context = await self._prepare_graphrag_service(optimization_settings)
            else:
                raise ValueError(f"Unsupported service type: {service_type}")
            
            # Store connection context
            self._service_connections[service_type] = connection_context
            
            # Log service metrics
            log_service_metrics(service_type, {
                "status": "prepared",
                "optimization_level": len(optimization_settings),
                "capabilities": len(service_config["capabilities"])
            })
            
            self.logger.info(f"Successfully prepared {service_type} service")
            return connection_context
            
        except Exception as e:
            self.logger.error(f"Failed to prepare {service_type} service: {e}")
            raise
    
    async def _prepare_mariadb_service(self, optimization_settings: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare MariaDB service for business data operations."""
        
        connection_context = {
            "service_type": "mariadb",
            "connection_pool_size": optimization_settings.get("connection_pool_size", 5),
            "query_timeout": optimization_settings.get("query_timeout", 30),
            "transaction_isolation": "READ_COMMITTED",
            "optimization_features": {
                "query_cache": True,
                "connection_pooling": True,
                "prepared_statements": True
            },
            "business_data_optimizations": {
                "enable_query_planning": True,
                "index_optimization": True,
                "table_scan_optimization": True
            }
        }
        
        # Simulate connection preparation
        await asyncio.sleep(0.1)  # Simulate connection setup time
        
        return connection_context
    
    async def _prepare_postgresql_service(self, optimization_settings: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare PostgreSQL service for memory and caching operations."""
        
        connection_context = {
            "service_type": "postgresql",
            "enable_caching": optimization_settings.get("enable_caching", True),
            "memory_optimization": optimization_settings.get("memory_optimization", True),
            "session_management": {
                "enable_session_cache": True,
                "session_timeout_minutes": 30,
                "investigation_storage": True
            },
            "cache_optimizations": {
                "query_result_cache": True,
                "investigation_cache": True,
                "user_context_cache": True,
                "cache_ttl_minutes": 60
            }
        }
        
        # Simulate connection preparation
        await asyncio.sleep(0.1)
        
        return connection_context
    
    async def _prepare_qdrant_service(self, optimization_settings: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare Qdrant service for vector search operations."""
        
        connection_context = {
            "service_type": "qdrant",
            "vector_index_optimization": optimization_settings.get("vector_index_optimization", True),
            "similarity_threshold": optimization_settings.get("similarity_threshold", 0.8),
            "embedding_optimizations": {
                "preload_embeddings": True,
                "similarity_cache": True,
                "pattern_matching_cache": True
            },
            "vector_search_config": {
                "search_algorithm": "approximate_nearest_neighbor",
                "index_type": "hnsw",
                "max_results": 100,
                "enable_reranking": True
            }
        }
        
        # Simulate vector index preparation
        await asyncio.sleep(0.2)  # Vector operations typically take longer
        
        return connection_context
    
    async def _prepare_graphrag_service(self, optimization_settings: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare GraphRAG service for knowledge graph operations."""
        
        connection_context = {
            "service_type": "graphrag",
            "knowledge_graph_depth": optimization_settings.get("knowledge_graph_depth", 3),
            "entity_resolution": optimization_settings.get("entity_resolution", True),
            "graph_optimizations": {
                "enable_graph_cache": True,
                "relationship_indexing": True,
                "entity_disambiguation": True
            },
            "analysis_config": {
                "max_traversal_depth": 5,
                "enable_inference": True,
                "relationship_weighting": True,
                "entity_scoring": True
            }
        }
        
        # Simulate knowledge graph preparation
        await asyncio.sleep(0.3)  # Graph operations are most complex
        
        return connection_context
    
    async def activate_services(self, service_types: List[str]) -> Dict[str, bool]:
        """
        Activate multiple services in parallel.
        
        Args:
            service_types: List of service types to activate
            
        Returns:
            Dictionary of service activation status
        """
        if not settings.enable_parallel_service_activation:
            # Sequential activation
            activation_status = {}
            for service_type in service_types:
                try:
                    await self._activate_single_service(service_type)
                    activation_status[service_type] = True
                except Exception as e:
                    self.logger.error(f"Failed to activate {service_type}: {e}")
                    activation_status[service_type] = False
            return activation_status
        
        # Parallel activation
        activation_tasks = []
        for service_type in service_types:
            task = self._activate_single_service(service_type)
            activation_tasks.append(task)
        
        results = await asyncio.gather(*activation_tasks, return_exceptions=True)
        
        activation_status = {}
        for i, result in enumerate(results):
            service_type = service_types[i]
            if isinstance(result, Exception):
                self.logger.error(f"Failed to activate {service_type}: {result}")
                activation_status[service_type] = False
            else:
                activation_status[service_type] = True
        
        return activation_status
    
    async def _activate_single_service(self, service_type: str) -> None:
        """Activate a single service."""
        if service_type not in self._service_config:
            raise ValueError(f"Unknown service type: {service_type}")
        
        # Simulate service activation
        await asyncio.sleep(0.1)
        
        self.logger.debug(f"Activated service: {service_type}")
    
    async def get_service_connection(self, service_type: str) -> Optional[Dict[str, Any]]:
        """Get connection context for a specific service."""
        return self._service_connections.get(service_type)
    
    async def check_service_health(self, service_type: str) -> bool:
        """Check health of a specific service."""
        if service_type not in self._service_config:
            return False
        
        service_name = self._service_config[service_type]["service_name"]
        
        # In a real implementation, this would perform actual health checks
        # For now, simulate health check
        await asyncio.sleep(0.05)
        
        is_healthy = self._service_health.get(service_name, False)
        
        log_service_metrics(service_type, {
            "health_status": "healthy" if is_healthy else "unhealthy",
            "last_check": "now"
        })
        
        return is_healthy
    
    async def health_check(self) -> bool:
        """Check overall health of MCP coordinator."""
        if not self._initialized:
            return False
        
        # Check if core services are available
        core_services = ["mariadb", "postgresql"]
        health_checks = []
        
        for service_type in core_services:
            health_checks.append(self.check_service_health(service_type))
        
        results = await asyncio.gather(*health_checks, return_exceptions=True)
        
        # Consider healthy if at least core services are working
        healthy_core_services = sum(1 for result in results if result is True)
        
        return healthy_core_services >= len(core_services) // 2
    
    async def cleanup(self) -> None:
        """Cleanup MCP service connections."""
        try:
            for service_type in list(self._service_connections.keys()):
                self._service_connections.pop(service_type, None)
                self.logger.debug(f"Cleaned up {service_type} connection")
            
            self._initialized = False
            self.logger.info("MCP Service Coordinator cleaned up")
            
        except Exception as e:
            self.logger.error(f"Error during MCP coordinator cleanup: {e}")