"""
Service Orchestration Module - Phase 3: Service Orchestration

Self-contained module for coordinating database services based on Intelligence Module outputs.
Implements intelligent service selection, resource optimization, and health monitoring.

Main Components:
- ServiceOrchestrator: Core orchestration logic
- MCPServiceCoordinator: MCP client coordination
- ResourceOptimizer: Connection and resource optimization
- HealthMonitor: Service health monitoring with circuit breakers
- ServiceOrchestrationRunner: High-level orchestration interface

Usage Examples:

    # High-level interface (recommended)
    from service_orchestration import orchestrate_investigation
    
    result = await orchestrate_investigation(
        complexity_level="analytical",
        complexity_score=0.4,
        estimated_duration_minutes=15,
        estimated_queries=5,
        methodology="comparative_analysis"
    )
    
    # Direct orchestrator usage
    from service_orchestration import ServiceOrchestrationRunner
    
    runner = ServiceOrchestrationRunner()
    await runner.initialize()
    
    result = await runner.orchestrate(
        complexity_level="computational",
        complexity_score=0.7,
        estimated_duration_minutes=30,
        estimated_queries=10,
        methodology="advanced_analytics"
    )
    
    await runner.cleanup()
    
    # Component-level access
    from service_orchestration import ServiceOrchestrator, ComplexityScore, ContextualStrategy
    
    orchestrator = ServiceOrchestrator()
    await orchestrator.initialize()
    
    # ... use orchestrator directly
"""

from .runner import ServiceOrchestrationRunner, orchestrate_investigation
from .service_orchestrator import (
    ServiceOrchestrator,
    ComplexityLevel,
    ComplexityScore,
    ContextualStrategy,
    ServiceConfiguration,
    OrchestrationResult,
    ServiceType
)
from .mcp_coordinator import MCPServiceCoordinator
from .resource_optimizer import ResourceOptimizer, ResourceAllocation, OptimizationProfile
from .health_monitor import (
    HealthMonitor,
    ServiceHealthMetrics,
    HealthCheckResult,
    ServiceStatus,
    CircuitBreakerState
)
from .config import ServiceOrchestrationSettings, settings
from .orchestration_logging import setup_logger, performance_monitor, log_service_metrics, log_orchestration_event

__all__ = [
    # Main interfaces (recommended entry points)
    "orchestrate_investigation",    # High-level function interface
    "ServiceOrchestrationRunner",  # High-level class interface
    
    # Core orchestration components
    "ServiceOrchestrator",         # Main orchestration logic
    "MCPServiceCoordinator",       # MCP service coordination
    "ResourceOptimizer",           # Resource optimization
    "HealthMonitor",               # Health monitoring
    
    # Data structures
    "ComplexityLevel",             # Complexity enumeration
    "ComplexityScore",             # Complexity score dataclass
    "ContextualStrategy",          # Strategy dataclass
    "ServiceConfiguration",        # Service config dataclass
    "OrchestrationResult",         # Orchestration result dataclass
    "ServiceType",                 # Service type enumeration
    "ResourceAllocation",          # Resource allocation dataclass
    "OptimizationProfile",         # Optimization profile dataclass
    "ServiceHealthMetrics",        # Health metrics dataclass
    "HealthCheckResult",           # Health check result dataclass
    "ServiceStatus",               # Service status enumeration
    "CircuitBreakerState",         # Circuit breaker state enumeration
    
    # Configuration and utilities
    "ServiceOrchestrationSettings", # Configuration class
    "settings",                    # Configuration instance
    "setup_logger",               # Logging setup
    "performance_monitor",        # Performance monitoring decorator
    "log_service_metrics",        # Service metrics logging
    "log_orchestration_event",    # Orchestration event logging
]

__version__ = "1.0.0"
__description__ = "Self-contained service orchestration module with intelligent database coordination"

# Module metadata
__author__ = "Agentic SQL Intelligence System"
__status__ = "Production"
__complexity_levels__ = ["simple", "analytical", "computational", "investigative"]
__supported_services__ = ["mariadb", "postgresql", "lancedb", "graphrag"]

# Quick health check function
async def health_check() -> dict:
    """
    Quick health check of the entire service orchestration module.
    
    Returns:
        Dictionary with health status of all components
    """
    runner = ServiceOrchestrationRunner()
    try:
        await runner.initialize()
        return await runner.health_check()
    except Exception as e:
        return {"error": str(e), "overall_status": "error"}
    finally:
        await runner.cleanup()

# Module initialization check
def verify_module_integrity() -> bool:
    """
    Verify that all required components are properly imported and accessible.
    
    Returns:
        True if module integrity is verified, False otherwise
    """
    try:
        # Check required classes can be instantiated
        from .config import settings
        from .orchestration_logging import setup_logger
        
        # Verify configuration is loaded
        if not hasattr(settings, 'mariadb_service_name'):
            return False
        
        # Verify logger can be created
        logger = setup_logger("integrity_check")
        if not logger:
            return False
        
        return True
        
    except Exception:
        return False

# Export verification result
__module_integrity__ = verify_module_integrity()
