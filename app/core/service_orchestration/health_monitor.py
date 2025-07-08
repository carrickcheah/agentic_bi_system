"""
Health Monitor - Monitors service health and implements circuit breaker patterns.
Self-contained health monitoring for service orchestration.
Zero external dependencies beyond module boundary.
"""

from typing import Dict, List, Optional, Any
import asyncio
import time
from dataclasses import dataclass
from enum import Enum

try:
    from .config import settings
    from .orchestration_logging import setup_logger, performance_monitor, log_service_metrics
except ImportError:
    from config import settings
    from orchestration_logging import setup_logger, performance_monitor, log_service_metrics


class ServiceStatus(Enum):
    """Service health status enumeration."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class CircuitBreakerState(Enum):
    """Circuit breaker state enumeration."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Circuit breaker activated
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class ServiceHealthMetrics:
    """Health metrics for a specific service."""
    service_name: str
    status: ServiceStatus
    last_check_timestamp: float
    consecutive_failures: int
    total_checks: int
    total_failures: int
    average_response_time_ms: float
    circuit_breaker_state: CircuitBreakerState


@dataclass
class HealthCheckResult:
    """Result of a health check operation."""
    service_name: str
    is_healthy: bool
    response_time_ms: float
    error_message: Optional[str]
    timestamp: float


class ServiceCircuitBreaker:
    """
    Circuit breaker implementation for individual services.
    Prevents cascading failures by temporarily disabling unhealthy services.
    """
    
    def __init__(self, service_name: str, failure_threshold: int = 3, recovery_timeout: int = 60):
        self.service_name = service_name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = CircuitBreakerState.CLOSED
        
    async def call(self, health_check_func) -> HealthCheckResult:
        """Execute health check through circuit breaker."""
        current_time = time.time()
        
        # Check if circuit breaker should transition from OPEN to HALF_OPEN
        if (self.state == CircuitBreakerState.OPEN and 
            current_time - self.last_failure_time > self.recovery_timeout):
            self.state = CircuitBreakerState.HALF_OPEN
        
        # If circuit breaker is OPEN, fail fast
        if self.state == CircuitBreakerState.OPEN:
            return HealthCheckResult(
                service_name=self.service_name,
                is_healthy=False,
                response_time_ms=0.0,
                error_message="Circuit breaker is OPEN",
                timestamp=current_time
            )
        
        try:
            # Execute health check
            result = await health_check_func()
            
            if result.is_healthy:
                # Success - reset failure count and close circuit
                self.failure_count = 0
                self.state = CircuitBreakerState.CLOSED
            else:
                # Failure - increment count and possibly open circuit
                self.failure_count += 1
                self.last_failure_time = current_time
                
                if self.failure_count >= self.failure_threshold:
                    self.state = CircuitBreakerState.OPEN
            
            return result
            
        except Exception as e:
            # Exception during health check
            self.failure_count += 1
            self.last_failure_time = current_time
            
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitBreakerState.OPEN
            
            return HealthCheckResult(
                service_name=self.service_name,
                is_healthy=False,
                response_time_ms=0.0,
                error_message=str(e),
                timestamp=current_time
            )


class HealthMonitor:
    """
    Monitors health of all database services and implements circuit breaker patterns.
    Provides comprehensive health monitoring for service orchestration.
    """
    
    def __init__(self):
        self.logger = setup_logger("health_monitor")
        self._service_metrics = {}
        self._circuit_breakers = {}
        self._monitoring_task = None
        self._initialized = False
        self._shutdown_requested = False
        
        self.logger.info("Health Monitor initialized")
    
    async def initialize(self) -> None:
        """Initialize health monitor with service configurations."""
        if self._initialized:
            return
        
        try:
            # Initialize service metrics for all configured services
            service_names = [
                settings.mariadb_service_name,
                settings.postgres_service_name,
                settings.qdrant_service_name,
                settings.graphrag_service_name
            ]
            
            for service_name in service_names:
                self._service_metrics[service_name] = ServiceHealthMetrics(
                    service_name=service_name,
                    status=ServiceStatus.UNKNOWN,
                    last_check_timestamp=0.0,
                    consecutive_failures=0,
                    total_checks=0,
                    total_failures=0,
                    average_response_time_ms=0.0,
                    circuit_breaker_state=CircuitBreakerState.CLOSED
                )
                
                # Initialize circuit breaker if enabled
                if settings.circuit_breaker_enabled:
                    self._circuit_breakers[service_name] = ServiceCircuitBreaker(
                        service_name=service_name,
                        failure_threshold=settings.max_failure_count,
                        recovery_timeout=60  # 1 minute recovery timeout
                    )
            
            # Start continuous health monitoring
            if settings.health_check_interval_seconds > 0:
                self._monitoring_task = asyncio.create_task(self._continuous_monitoring())
            
            self._initialized = True
            self.logger.info("Health Monitor fully initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Health Monitor: {e}")
            raise
    
    async def _continuous_monitoring(self) -> None:
        """Continuous health monitoring background task."""
        self.logger.info("Starting continuous health monitoring")
        
        while not self._shutdown_requested:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(settings.health_check_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in continuous monitoring: {e}")
                await asyncio.sleep(5)  # Short delay before retry
        
        self.logger.info("Continuous health monitoring stopped")
    
    async def _perform_health_checks(self) -> None:
        """Perform health checks for all monitored services."""
        check_tasks = []
        
        for service_name in self._service_metrics.keys():
            task = self._check_individual_service_health(service_name)
            check_tasks.append(task)
        
        # Execute health checks in parallel
        results = await asyncio.gather(*check_tasks, return_exceptions=True)
        
        for i, result in enumerate(results):
            service_name = list(self._service_metrics.keys())[i]
            if isinstance(result, Exception):
                self.logger.error(f"Health check failed for {service_name}: {result}")
            else:
                self._update_service_metrics(service_name, result)
    
    async def _check_individual_service_health(self, service_name: str) -> HealthCheckResult:
        """Check health of an individual service."""
        start_time = time.time()
        
        try:
            # Simulate service health check
            # In a real implementation, this would check actual service connectivity
            await asyncio.sleep(0.05)  # Simulate network latency
            
            # Simulate occasional failures for testing
            import random
            is_healthy = random.random() > 0.1  # 90% success rate
            
            response_time_ms = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                service_name=service_name,
                is_healthy=is_healthy,
                response_time_ms=response_time_ms,
                error_message=None if is_healthy else "Simulated service failure",
                timestamp=time.time()
            )
            
        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                service_name=service_name,
                is_healthy=False,
                response_time_ms=response_time_ms,
                error_message=str(e),
                timestamp=time.time()
            )
    
    def _update_service_metrics(self, service_name: str, result: HealthCheckResult) -> None:
        """Update service metrics based on health check result."""
        if service_name not in self._service_metrics:
            return
        
        metrics = self._service_metrics[service_name]
        
        # Update counters
        metrics.total_checks += 1
        metrics.last_check_timestamp = result.timestamp
        
        if result.is_healthy:
            metrics.consecutive_failures = 0
            if metrics.status != ServiceStatus.HEALTHY:
                self.logger.info(f"Service {service_name} recovered to healthy status")
            metrics.status = ServiceStatus.HEALTHY
        else:
            metrics.total_failures += 1
            metrics.consecutive_failures += 1
            
            # Determine status based on consecutive failures
            if metrics.consecutive_failures >= settings.max_failure_count:
                if metrics.status != ServiceStatus.UNHEALTHY:
                    self.logger.warning(f"Service {service_name} marked as unhealthy")
                metrics.status = ServiceStatus.UNHEALTHY
            elif metrics.consecutive_failures > 1:
                if metrics.status != ServiceStatus.DEGRADED:
                    self.logger.warning(f"Service {service_name} marked as degraded")
                metrics.status = ServiceStatus.DEGRADED
        
        # Update average response time (exponential moving average)
        if metrics.average_response_time_ms == 0.0:
            metrics.average_response_time_ms = result.response_time_ms
        else:
            alpha = 0.3  # Smoothing factor
            metrics.average_response_time_ms = (
                alpha * result.response_time_ms + 
                (1 - alpha) * metrics.average_response_time_ms
            )
        
        # Update circuit breaker state
        if service_name in self._circuit_breakers:
            metrics.circuit_breaker_state = self._circuit_breakers[service_name].state
        
        # Log service metrics
        log_service_metrics(service_name, {
            "status": metrics.status.value,
            "consecutive_failures": metrics.consecutive_failures,
            "response_time_ms": f"{metrics.average_response_time_ms:.2f}",
            "circuit_breaker_state": metrics.circuit_breaker_state.value
        })
    
    @performance_monitor("service_health_check")
    async def check_service_health(self, service_name: str) -> bool:
        """
        Check health of a specific service.
        
        Args:
            service_name: Name of the service to check
            
        Returns:
            True if service is healthy, False otherwise
        """
        if service_name not in self._service_metrics:
            self.logger.warning(f"Unknown service for health check: {service_name}")
            return False
        
        # Use circuit breaker if enabled
        if settings.circuit_breaker_enabled and service_name in self._circuit_breakers:
            circuit_breaker = self._circuit_breakers[service_name]
            result = await circuit_breaker.call(
                lambda: self._check_individual_service_health(service_name)
            )
            self._update_service_metrics(service_name, result)
            return result.is_healthy
        else:
            # Direct health check without circuit breaker
            result = await self._check_individual_service_health(service_name)
            self._update_service_metrics(service_name, result)
            return result.is_healthy
    
    async def get_service_metrics(self, service_name: str) -> Optional[ServiceHealthMetrics]:
        """Get health metrics for a specific service."""
        return self._service_metrics.get(service_name)
    
    async def get_all_service_metrics(self) -> Dict[str, ServiceHealthMetrics]:
        """Get health metrics for all monitored services."""
        return self._service_metrics.copy()
    
    async def get_overall_health_status(self) -> Dict[str, Any]:
        """Get overall health status of all services."""
        if not self._service_metrics:
            return {
                "overall_status": "unknown",
                "healthy_services": 0,
                "total_services": 0,
                "degraded_services": 0,
                "unhealthy_services": 0
            }
        
        healthy_count = 0
        degraded_count = 0
        unhealthy_count = 0
        
        for metrics in self._service_metrics.values():
            if metrics.status == ServiceStatus.HEALTHY:
                healthy_count += 1
            elif metrics.status == ServiceStatus.DEGRADED:
                degraded_count += 1
            elif metrics.status == ServiceStatus.UNHEALTHY:
                unhealthy_count += 1
        
        total_services = len(self._service_metrics)
        
        # Determine overall status
        if unhealthy_count > total_services // 2:
            overall_status = "critical"
        elif degraded_count + unhealthy_count > total_services // 2:
            overall_status = "degraded"
        elif healthy_count == total_services:
            overall_status = "healthy"
        else:
            overall_status = "mixed"
        
        return {
            "overall_status": overall_status,
            "healthy_services": healthy_count,
            "total_services": total_services,
            "degraded_services": degraded_count,
            "unhealthy_services": unhealthy_count,
            "health_percentage": (healthy_count / total_services) * 100 if total_services > 0 else 0
        }
    
    async def reset_service_metrics(self, service_name: str) -> None:
        """Reset health metrics for a specific service."""
        if service_name in self._service_metrics:
            metrics = self._service_metrics[service_name]
            metrics.consecutive_failures = 0
            metrics.total_checks = 0
            metrics.total_failures = 0
            metrics.status = ServiceStatus.UNKNOWN
            metrics.average_response_time_ms = 0.0
            
            # Reset circuit breaker
            if service_name in self._circuit_breakers:
                circuit_breaker = self._circuit_breakers[service_name]
                circuit_breaker.failure_count = 0
                circuit_breaker.state = CircuitBreakerState.CLOSED
                metrics.circuit_breaker_state = CircuitBreakerState.CLOSED
            
            self.logger.info(f"Reset health metrics for service: {service_name}")
    
    async def health_check(self) -> bool:
        """Check health of the health monitor itself."""
        if not self._initialized:
            return False
        
        # Check if monitoring task is running
        if self._monitoring_task and self._monitoring_task.done():
            self.logger.warning("Health monitoring task has stopped")
            return False
        
        # Check if we have metrics for all expected services
        expected_services = [
            settings.mariadb_service_name,
            settings.postgres_service_name,
            settings.qdrant_service_name,
            settings.graphrag_service_name
        ]
        
        for service_name in expected_services:
            if service_name not in self._service_metrics:
                self.logger.error(f"Missing metrics for service: {service_name}")
                return False
        
        return True
    
    async def cleanup(self) -> None:
        """Cleanup health monitor and stop monitoring."""
        try:
            self._shutdown_requested = True
            
            # Cancel monitoring task
            if self._monitoring_task and not self._monitoring_task.done():
                self._monitoring_task.cancel()
                try:
                    await self._monitoring_task
                except asyncio.CancelledError:
                    pass
            
            # Clear metrics and circuit breakers
            self._service_metrics.clear()
            self._circuit_breakers.clear()
            self._initialized = False
            
            self.logger.info("Health Monitor cleaned up")
            
        except Exception as e:
            self.logger.error(f"Error during health monitor cleanup: {e}")
