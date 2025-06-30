"""
Main runner for Service Orchestration module.
Production-grade orchestrator with intelligent fallback architecture.
Zero external dependencies beyond module boundary.
"""

from typing import Dict, List, Any, Optional
import asyncio
from dataclasses import asdict

try:
    from .config import settings
    from .orchestration_logging import setup_logger
    from .service_orchestrator import (
        ServiceOrchestrator, 
        ComplexityScore, 
        ContextualStrategy, 
        ComplexityLevel
    )
except ImportError:
    from config import settings
    from orchestration_logging import setup_logger
    from service_orchestrator import (
        ServiceOrchestrator, 
        ComplexityScore, 
        ContextualStrategy, 
        ComplexityLevel
    )


class ServiceOrchestrationRunner:
    """
    Main runner for Phase 3: Service Orchestration.
    Provides high-level interface for service coordination and orchestration.
    """
    
    def __init__(self):
        self.logger = setup_logger("orchestration_runner")
        self.orchestrator = ServiceOrchestrator()
        self._initialized = False
        
        self.logger.info("Service Orchestration Runner initialized")
    
    async def initialize(self) -> None:
        """Initialize the orchestration runner."""
        if self._initialized:
            return
        
        try:
            await self.orchestrator.initialize()
            self._initialized = True
            self.logger.info("Service Orchestration Runner fully initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Service Orchestration Runner: {e}")
            raise
    
    async def orchestrate(
        self,
        complexity_level: str,
        complexity_score: float,
        estimated_duration_minutes: int,
        estimated_queries: int,
        methodology: str,
        user_preferences: Optional[Dict[str, Any]] = None,
        investigation_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Main orchestration interface - simplified for external use.
        
        Args:
            complexity_level: One of 'simple', 'analytical', 'computational', 'investigative'
            complexity_score: Numeric score from 0.0 to 1.0
            estimated_duration_minutes: Expected investigation duration
            estimated_queries: Expected number of database queries
            methodology: Investigation methodology from Intelligence Module
            user_preferences: Optional user preferences dictionary
            investigation_context: Optional investigation context
            
        Returns:
            Orchestration result dictionary ready for Phase 4
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            # Create ComplexityScore from inputs
            complexity_enum = ComplexityLevel(complexity_level.lower())
            complexity_score_obj = ComplexityScore(
                level=complexity_enum,
                score=complexity_score,
                estimated_duration_minutes=estimated_duration_minutes,
                estimated_queries=estimated_queries,
                estimated_services=self._estimate_services_needed(complexity_score),
                confidence=0.8,  # Default confidence
                methodology=methodology
            )
            
            # Create ContextualStrategy from inputs
            contextual_strategy = ContextualStrategy(
                adapted_methodology=methodology,
                estimated_timeline=self._create_timeline(estimated_duration_minutes),
                communication_style="professional",
                deliverable_format="structured_report",
                user_preferences=user_preferences or {},
                organizational_constraints={}
            )
            
            # Perform orchestration
            result = await self.orchestrator.orchestrate_services(
                complexity_score_obj,
                contextual_strategy,
                investigation_context
            )
            
            # Convert result to dictionary for external use
            return {
                "coordinated_services": [
                    {
                        "service_type": service.service_type.value,
                        "enabled": service.enabled,
                        "priority": service.priority,
                        "estimated_load": service.estimated_load,
                        "optimization_settings": service.optimization_settings
                    } for service in result.coordinated_services
                ],
                "optimized_connections": result.optimized_connections,
                "execution_context": result.execution_context,
                "health_status": result.health_status,
                "estimated_performance": result.estimated_performance,
                "fallback_strategy": result.fallback_strategy
            }
            
        except Exception as e:
            self.logger.error(f"Orchestration failed: {e}")
            raise
    
    def _estimate_services_needed(self, complexity_score: float) -> int:
        """Estimate number of services needed based on complexity."""
        if complexity_score < settings.analytical_complexity_threshold:
            return 1  # MariaDB only
        elif complexity_score < settings.computational_complexity_threshold:
            return 2  # MariaDB + PostgreSQL
        elif complexity_score < settings.investigative_complexity_threshold:
            return 3  # MariaDB + PostgreSQL + LanceDB
        else:
            return 4  # All services
    
    def _create_timeline(self, total_duration_minutes: int) -> Dict[str, int]:
        """Create timeline allocation for investigation phases."""
        return {
            "analysis": int(total_duration_minutes * 0.3),
            "data_gathering": int(total_duration_minutes * 0.4),
            "synthesis": int(total_duration_minutes * 0.2),
            "reporting": int(total_duration_minutes * 0.1)
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check of orchestration system."""
        health_status = {
            "runner_initialized": self._initialized,
            "orchestrator_health": {},
            "overall_status": "unknown"
        }
        
        if self._initialized:
            try:
                orchestrator_health = await self.orchestrator.health_check()
                health_status["orchestrator_health"] = orchestrator_health
                
                # Determine overall status
                if all(orchestrator_health.values()):
                    health_status["overall_status"] = "healthy"
                elif any(orchestrator_health.values()):
                    health_status["overall_status"] = "degraded"
                else:
                    health_status["overall_status"] = "unhealthy"
                    
            except Exception as e:
                health_status["orchestrator_health"] = {"error": str(e)}
                health_status["overall_status"] = "error"
        
        return health_status
    
    async def get_service_status(self) -> Dict[str, Any]:
        """Get detailed status of all orchestrated services."""
        if not self._initialized:
            return {"error": "Runner not initialized"}
        
        try:
            # Get health status from health monitor
            health_monitor = self.orchestrator.health_monitor
            overall_health = await health_monitor.get_overall_health_status()
            all_metrics = await health_monitor.get_all_service_metrics()
            
            service_details = {}
            for service_name, metrics in all_metrics.items():
                service_details[service_name] = {
                    "status": metrics.status.value,
                    "consecutive_failures": metrics.consecutive_failures,
                    "total_checks": metrics.total_checks,
                    "total_failures": metrics.total_failures,
                    "average_response_time_ms": f"{metrics.average_response_time_ms:.2f}",
                    "circuit_breaker_state": metrics.circuit_breaker_state.value,
                    "last_check": metrics.last_check_timestamp
                }
            
            return {
                "overall_health": overall_health,
                "service_details": service_details
            }
            
        except Exception as e:
            return {"error": f"Failed to get service status: {e}"}
    
    async def reset_service_health(self, service_name: str) -> Dict[str, str]:
        """Reset health metrics for a specific service."""
        if not self._initialized:
            return {"error": "Runner not initialized"}
        
        try:
            await self.orchestrator.health_monitor.reset_service_metrics(service_name)
            return {"status": f"Reset health metrics for {service_name}"}
            
        except Exception as e:
            return {"error": f"Failed to reset service health: {e}"}
    
    async def cleanup(self) -> None:
        """Cleanup orchestration runner and all components."""
        try:
            if self._initialized:
                # Cleanup orchestrator components
                await self.orchestrator.mcp_coordinator.cleanup()
                await self.orchestrator.resource_optimizer.cleanup()
                await self.orchestrator.health_monitor.cleanup()
                
                self._initialized = False
            
            self.logger.info("Service Orchestration Runner cleaned up")
            
        except Exception as e:
            self.logger.error(f"Error during runner cleanup: {e}")


# Main execution interface
async def orchestrate_investigation(
    complexity_level: str,
    complexity_score: float,
    estimated_duration_minutes: int,
    estimated_queries: int,
    methodology: str,
    user_preferences: Optional[Dict[str, Any]] = None,
    investigation_context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    High-level interface for investigation orchestration.
    
    This is the main entry point for Phase 3: Service Orchestration.
    
    Args:
        complexity_level: Investigation complexity ('simple', 'analytical', 'computational', 'investigative')
        complexity_score: Numeric complexity score (0.0 to 1.0)
        estimated_duration_minutes: Expected investigation duration
        estimated_queries: Expected number of database queries
        methodology: Investigation methodology from Intelligence Module
        user_preferences: Optional user preferences
        investigation_context: Optional investigation context
        
    Returns:
        Orchestration result ready for Phase 4 execution
    """
    runner = ServiceOrchestrationRunner()
    
    try:
        result = await runner.orchestrate(
            complexity_level=complexity_level,
            complexity_score=complexity_score,
            estimated_duration_minutes=estimated_duration_minutes,
            estimated_queries=estimated_queries,
            methodology=methodology,
            user_preferences=user_preferences,
            investigation_context=investigation_context
        )
        
        return result
        
    finally:
        await runner.cleanup()


# Main execution point
if __name__ == "__main__":
    async def main():
        """Example usage of Service Orchestration."""
        logger = setup_logger("orchestration_example")
        
        # Example orchestration
        try:
            result = await orchestrate_investigation(
                complexity_level="analytical",
                complexity_score=0.4,
                estimated_duration_minutes=15,
                estimated_queries=5,
                methodology="comparative_analysis",
                user_preferences={"speed_preference": 0.7}
            )
            
            logger.info("Orchestration completed successfully")
            logger.info(f"Services coordinated: {len(result['coordinated_services'])}")
            logger.info(f"Health status: {result['health_status']}")
            
            # Display coordinated services
            for service in result['coordinated_services']:
                logger.info(f"Service: {service['service_type']} (Priority: {service['priority']})")
            
        except Exception as e:
            logger.error(f"Orchestration failed: {e}")
    
    # Run example
    asyncio.run(main())
