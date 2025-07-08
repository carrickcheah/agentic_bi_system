"""
Service Orchestrator - Main Phase 3 orchestration class.
Coordinates database services based on Intelligence Module outputs.
Zero external dependencies beyond module boundary.
"""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import asyncio

try:
    from .config import settings
    from .orchestration_logging import setup_logger, performance_monitor, log_orchestration_event
    from .mcp_coordinator import MCPServiceCoordinator
    from .resource_optimizer import ResourceOptimizer
    from .health_monitor import HealthMonitor
except ImportError:
    from config import settings
    from orchestration_logging import setup_logger, performance_monitor, log_orchestration_event
    from mcp_coordinator import MCPServiceCoordinator
    from resource_optimizer import ResourceOptimizer
    from health_monitor import HealthMonitor


class ComplexityLevel(Enum):
    """Investigation complexity levels aligned with Intelligence Module."""
    SIMPLE = "simple"              # 2-5 minutes: Single source, descriptive
    ANALYTICAL = "analytical"      # 5-15 minutes: Multiple sources, comparative
    COMPUTATIONAL = "computational"  # 15-45 minutes: Advanced analytics, modeling
    INVESTIGATIVE = "investigative"  # 30-120 minutes: Multi-domain, predictive


class ServiceType(Enum):
    """Database service types for orchestration."""
    MARIADB = "mariadb"           # Business data operations
    POSTGRESQL = "postgresql"     # Memory and caching
    QDRANT = "qdrant"            # Vector search and embeddings
    GRAPHRAG = "graphrag"        # Knowledge graph analysis


@dataclass
class ComplexityScore:
    """Input from Intelligence Module Phase 2."""
    level: ComplexityLevel
    score: float  # 0.0 to 1.0
    estimated_duration_minutes: int
    estimated_queries: int
    estimated_services: int
    confidence: float
    methodology: str


@dataclass
class ContextualStrategy:
    """Input from Intelligence Module Phase 2."""
    adapted_methodology: str
    estimated_timeline: Dict[str, int]  # phase -> minutes
    communication_style: str
    deliverable_format: str
    user_preferences: Dict[str, Union[str, float]]
    organizational_constraints: Dict[str, str]


@dataclass
class ServiceConfiguration:
    """Configuration for a specific database service."""
    service_type: ServiceType
    enabled: bool
    priority: int  # 1 = highest priority
    estimated_load: float  # 0.0 to 1.0
    optimization_settings: Dict[str, Any]


@dataclass
class OrchestrationResult:
    """Output from Phase 3 to Phase 4."""
    coordinated_services: List[ServiceConfiguration]
    optimized_connections: Dict[str, Any]
    execution_context: Dict[str, Any]
    health_status: Dict[str, bool]
    estimated_performance: Dict[str, float]
    fallback_strategy: Optional[str]


class ServiceOrchestrator:
    """
    Phase 3: Service Orchestration
    
    Coordinates database services based on Intelligence Module outputs.
    Implements intelligent service selection and resource optimization.
    """
    
    def __init__(self):
        self.logger = setup_logger("service_orchestrator")
        self.mcp_coordinator = MCPServiceCoordinator()
        self.resource_optimizer = ResourceOptimizer()
        self.health_monitor = HealthMonitor()
        self._initialized = False
        
        self.logger.info("Service Orchestrator initialized")
    
    async def initialize(self) -> None:
        """Initialize orchestrator components."""
        if self._initialized:
            return
        
        try:
            # Initialize components in parallel
            await asyncio.gather(
                self.mcp_coordinator.initialize(),
                self.resource_optimizer.initialize(),
                self.health_monitor.initialize()
            )
            
            self._initialized = True
            self.logger.info("Service Orchestrator fully initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Service Orchestrator: {e}")
            raise
    
    @performance_monitor("service_orchestration")
    async def orchestrate_services(
        self,
        complexity_score: ComplexityScore,
        contextual_strategy: ContextualStrategy,
        investigation_context: Optional[Dict[str, Any]] = None
    ) -> OrchestrationResult:
        """
        Main orchestration method - Phase 3 core function.
        
        Args:
            complexity_score: Output from Intelligence Module complexity analysis
            contextual_strategy: Output from Intelligence Module context analysis
            investigation_context: Additional context for orchestration
            
        Returns:
            OrchestrationResult ready for Phase 4 execution
        """
        log_orchestration_event("orchestration_start", {
            "complexity_level": complexity_score.level.value,
            "complexity_score": complexity_score.score,
            "methodology": complexity_score.methodology,
            "estimated_duration": complexity_score.estimated_duration_minutes
        })
        
        try:
            # Step 1: Service Selection based on complexity
            service_selection = await self._select_services(complexity_score, contextual_strategy)
            
            # Step 2: Service Preparation and Optimization
            optimized_connections = await self._prepare_services(
                service_selection, complexity_score, contextual_strategy
            )
            
            # Step 3: Health Validation
            health_status = await self._validate_service_health(service_selection)
            
            # Step 4: Performance Estimation
            performance_estimates = await self._estimate_performance(
                service_selection, complexity_score
            )
            
            # Step 5: Execution Context Preparation
            execution_context = await self._prepare_execution_context(
                complexity_score, contextual_strategy, investigation_context
            )
            
            # Step 6: Fallback Strategy Preparation
            fallback_strategy = await self._prepare_fallback_strategy(
                service_selection, complexity_score
            )
            
            result = OrchestrationResult(
                coordinated_services=service_selection,
                optimized_connections=optimized_connections,
                execution_context=execution_context,
                health_status=health_status,
                estimated_performance=performance_estimates,
                fallback_strategy=fallback_strategy
            )
            
            log_orchestration_event("orchestration_complete", {
                "services_activated": len([s for s in service_selection if s.enabled]),
                "health_status": "healthy" if all(health_status.values()) else "degraded",
                "fallback_available": fallback_strategy is not None
            })
            
            return result
            
        except Exception as e:
            log_orchestration_event("orchestration_failed", {
                "error": str(e),
                "complexity_level": complexity_score.level.value
            })
            raise
    
    async def _select_services(
        self, 
        complexity_score: ComplexityScore,
        contextual_strategy: ContextualStrategy
    ) -> List[ServiceConfiguration]:
        """Select appropriate services based on complexity level."""
        
        services = []
        
        # MariaDB - Always required for business data
        services.append(ServiceConfiguration(
            service_type=ServiceType.MARIADB,
            enabled=True,
            priority=1,
            estimated_load=0.7,  # Base business data load
            optimization_settings={
                "connection_pool_size": settings.connection_pool_size,
                "query_timeout": settings.connection_timeout_seconds
            }
        ))
        
        # PostgreSQL - Required for ANALYTICAL and above
        if complexity_score.score >= settings.analytical_complexity_threshold:
            services.append(ServiceConfiguration(
                service_type=ServiceType.POSTGRESQL,
                enabled=True,
                priority=2,
                estimated_load=0.5,
                optimization_settings={
                    "enable_caching": settings.cache_warmup_enabled,
                    "memory_optimization": True
                }
            ))
        
        # Qdrant - Required for COMPUTATIONAL and above
        if complexity_score.score >= settings.computational_complexity_threshold:
            services.append(ServiceConfiguration(
                service_type=ServiceType.QDRANT,
                enabled=True,
                priority=3,
                estimated_load=0.6,
                optimization_settings={
                    "vector_index_optimization": True,
                    "similarity_threshold": 0.8
                }
            ))
        
        # GraphRAG - Required for INVESTIGATIVE level
        if complexity_score.score >= settings.investigative_complexity_threshold:
            services.append(ServiceConfiguration(
                service_type=ServiceType.GRAPHRAG,
                enabled=True,
                priority=4,
                estimated_load=0.8,
                optimization_settings={
                    "knowledge_graph_depth": 3,
                    "entity_resolution": True
                }
            ))
        
        # Adjust based on user preferences
        if contextual_strategy.user_preferences.get("speed_preference", 0.5) > 0.8:
            # High speed preference - reduce service complexity
            services = [s for s in services if s.service_type in [ServiceType.MARIADB, ServiceType.POSTGRESQL]]
        
        self.logger.info(f"Selected {len(services)} services for {complexity_score.level.value} complexity")
        
        return services
    
    async def _prepare_services(
        self,
        service_selection: List[ServiceConfiguration],
        complexity_score: ComplexityScore,
        contextual_strategy: ContextualStrategy
    ) -> Dict[str, Any]:
        """Prepare and optimize selected services."""
        
        optimized_connections = {}
        
        # Prepare services in parallel if enabled
        if settings.enable_parallel_service_activation:
            preparation_tasks = []
            for service_config in service_selection:
                if service_config.enabled:
                    task = self.mcp_coordinator.prepare_service(
                        service_config.service_type.value,
                        service_config.optimization_settings
                    )
                    preparation_tasks.append(task)
            
            results = await asyncio.gather(*preparation_tasks, return_exceptions=True)
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.error(f"Service preparation failed: {result}")
                else:
                    service_name = service_selection[i].service_type.value
                    optimized_connections[service_name] = result
        else:
            # Sequential preparation
            for service_config in service_selection:
                if service_config.enabled:
                    try:
                        connection = await self.mcp_coordinator.prepare_service(
                            service_config.service_type.value,
                            service_config.optimization_settings
                        )
                        optimized_connections[service_config.service_type.value] = connection
                    except Exception as e:
                        self.logger.error(f"Failed to prepare {service_config.service_type.value}: {e}")
        
        # Apply resource optimization
        if settings.query_preparation_enabled:
            await self.resource_optimizer.optimize_connections(
                optimized_connections, complexity_score
            )
        
        return optimized_connections
    
    async def _validate_service_health(
        self, service_selection: List[ServiceConfiguration]
    ) -> Dict[str, bool]:
        """Validate health of all selected services."""
        
        health_status = {}
        
        for service_config in service_selection:
            if service_config.enabled:
                try:
                    is_healthy = await self.health_monitor.check_service_health(
                        service_config.service_type.value
                    )
                    health_status[service_config.service_type.value] = is_healthy
                except Exception as e:
                    self.logger.error(f"Health check failed for {service_config.service_type.value}: {e}")
                    health_status[service_config.service_type.value] = False
        
        healthy_services = sum(1 for status in health_status.values() if status)
        total_services = len(health_status)
        
        self.logger.info(f"Service health: {healthy_services}/{total_services} services healthy")
        
        return health_status
    
    async def _estimate_performance(
        self,
        service_selection: List[ServiceConfiguration],
        complexity_score: ComplexityScore
    ) -> Dict[str, float]:
        """Estimate performance characteristics for the service configuration."""
        
        performance_estimates = {
            "expected_response_time_seconds": 0.0,
            "estimated_throughput_qps": 0.0,
            "resource_utilization": 0.0,
            "parallel_execution_factor": 1.0
        }
        
        # Base performance calculation
        base_response_time = complexity_score.estimated_duration_minutes * 60
        
        # Adjust based on service complexity
        service_count = len([s for s in service_selection if s.enabled])
        
        if service_count == 1:  # Simple - MariaDB only
            performance_estimates["expected_response_time_seconds"] = base_response_time * 0.3
            performance_estimates["estimated_throughput_qps"] = 10.0
        elif service_count == 2:  # Analytical - MariaDB + PostgreSQL
            performance_estimates["expected_response_time_seconds"] = base_response_time * 0.5
            performance_estimates["estimated_throughput_qps"] = 7.0
        elif service_count == 3:  # Computational - + Qdrant
            performance_estimates["expected_response_time_seconds"] = base_response_time * 0.7
            performance_estimates["estimated_throughput_qps"] = 4.0
        else:  # Investigative - All services
            performance_estimates["expected_response_time_seconds"] = base_response_time * 1.0
            performance_estimates["estimated_throughput_qps"] = 2.0
        
        # Parallel execution factor
        if settings.enable_parallel_service_activation and service_count > 1:
            performance_estimates["parallel_execution_factor"] = min(service_count * 0.7, 3.0)
            performance_estimates["expected_response_time_seconds"] /= performance_estimates["parallel_execution_factor"]
        
        performance_estimates["resource_utilization"] = min(service_count * 0.25, 1.0)
        
        return performance_estimates
    
    async def _prepare_execution_context(
        self,
        complexity_score: ComplexityScore,
        contextual_strategy: ContextualStrategy,
        investigation_context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Prepare execution context for Phase 4."""
        
        execution_context = {
            "complexity_level": complexity_score.level.value,
            "methodology": complexity_score.methodology,
            "estimated_duration_minutes": complexity_score.estimated_duration_minutes,
            "communication_style": contextual_strategy.communication_style,
            "deliverable_format": contextual_strategy.deliverable_format,
            "timeline_allocation": contextual_strategy.estimated_timeline,
            "user_preferences": contextual_strategy.user_preferences,
            "organizational_constraints": contextual_strategy.organizational_constraints
        }
        
        if investigation_context:
            execution_context["investigation_context"] = investigation_context
        
        return execution_context
    
    async def _prepare_fallback_strategy(
        self,
        service_selection: List[ServiceConfiguration],
        complexity_score: ComplexityScore
    ) -> Optional[str]:
        """Prepare fallback strategy in case of service failures."""
        
        if not settings.enable_service_fallback:
            return None
        
        # Determine fallback based on current service selection
        enabled_services = [s.service_type for s in service_selection if s.enabled]
        
        if ServiceType.GRAPHRAG in enabled_services:
            return "fallback_to_qdrant_only"
        elif ServiceType.QDRANT in enabled_services:
            return "fallback_to_postgresql_only"
        elif ServiceType.POSTGRESQL in enabled_services:
            return "fallback_to_mariadb_only"
        else:
            return "degraded_mode_simple_queries_only"
    
    async def health_check(self) -> Dict[str, bool]:
        """Check health of orchestrator and all components."""
        health_status = {
            "orchestrator": self._initialized,
            "mcp_coordinator": False,
            "resource_optimizer": False,
            "health_monitor": False
        }
        
        try:
            health_status["mcp_coordinator"] = await self.mcp_coordinator.health_check()
            health_status["resource_optimizer"] = await self.resource_optimizer.health_check()
            health_status["health_monitor"] = await self.health_monitor.health_check()
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
        
        return health_status