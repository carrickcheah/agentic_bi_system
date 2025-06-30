#!/usr/bin/env python3
"""
Standalone test suite for Service Orchestration module.
Independent validation of all service orchestration components.
Zero external dependencies beyond module boundary.
"""

import asyncio
import sys
import time
from pathlib import Path
from typing import Dict, Any

# Add module to path for testing
sys.path.insert(0, str(Path(__file__).parent))

from config import settings
from runner import ServiceOrchestrationRunner, orchestrate_investigation
from service_orchestrator import ServiceOrchestrator, ComplexityLevel, ComplexityScore, ContextualStrategy
from mcp_coordinator import MCPServiceCoordinator
from resource_optimizer import ResourceOptimizer
from health_monitor import HealthMonitor
from orchestration_logging import setup_logger


class ServiceOrchestrationTester:
    """Comprehensive test suite for Service Orchestration module."""
    
    def __init__(self):
        self.logger = setup_logger("orchestration_tester")
        self.test_results = {}
        self.passed_tests = 0
        self.failed_tests = 0
    
    def log_test_result(self, test_name: str, passed: bool, details: str = "") -> None:
        """Log test result."""
        status = "PASS" if passed else "FAIL"
        self.test_results[test_name] = {"passed": passed, "details": details}
        
        if passed:
            self.passed_tests += 1
            self.logger.info(f"Test {test_name}: {status}")
        else:
            self.failed_tests += 1
            self.logger.error(f"Test {test_name}: {status} - {details}")
    
    async def test_configuration_loading(self) -> bool:
        """Test configuration loading from settings.env."""
        try:
            # Test required MCP service configurations
            required_services = [
                settings.mariadb_service_name,
                settings.postgres_service_name,
                settings.lancedb_service_name,
                settings.graphrag_service_name
            ]
            
            for service_name in required_services:
                if not service_name:
                    self.log_test_result(
                        "configuration_loading", 
                        False, 
                        f"Missing service name configuration"
                    )
                    return False
            
            # Test threshold configurations
            thresholds = [
                settings.analytical_complexity_threshold,
                settings.computational_complexity_threshold,
                settings.investigative_complexity_threshold
            ]
            
            for i, threshold in enumerate(thresholds):
                if not (0.0 <= threshold <= 1.0):
                    self.log_test_result(
                        "configuration_loading", 
                        False, 
                        f"Invalid threshold value: {threshold}"
                    )
                    return False
            
            # Test operational settings
            if settings.connection_pool_size <= 0:
                self.log_test_result(
                    "configuration_loading", 
                    False, 
                    f"Invalid connection pool size: {settings.connection_pool_size}"
                )
                return False
            
            self.log_test_result(
                "configuration_loading", 
                True, 
                f"All configurations loaded successfully"
            )
            return True
            
        except Exception as e:
            self.log_test_result("configuration_loading", False, str(e))
            return False
    
    async def test_mcp_coordinator(self) -> bool:
        """Test MCP Service Coordinator functionality."""
        try:
            coordinator = MCPServiceCoordinator()
            await coordinator.initialize()
            
            # Test service preparation
            optimization_settings = {
                "connection_pool_size": 5,
                "query_timeout": 30
            }
            
            mariadb_context = await coordinator.prepare_service(
                "mariadb", optimization_settings
            )
            
            if not mariadb_context or mariadb_context.get("service_type") != "mariadb":
                self.log_test_result(
                    "mcp_coordinator", 
                    False, 
                    "Failed to prepare MariaDB service"
                )
                return False
            
            # Test health check
            is_healthy = await coordinator.health_check()
            
            await coordinator.cleanup()
            
            self.log_test_result(
                "mcp_coordinator", 
                True, 
                f"MCP coordinator functioning properly"
            )
            return True
            
        except Exception as e:
            self.log_test_result("mcp_coordinator", False, str(e))
            return False
    
    async def test_resource_optimizer(self) -> bool:
        """Test Resource Optimizer functionality."""
        try:
            optimizer = ResourceOptimizer()
            await optimizer.initialize()
            
            # Create mock complexity score
            complexity_score = ComplexityScore(
                level=ComplexityLevel.ANALYTICAL,
                score=0.4,
                estimated_duration_minutes=15,
                estimated_queries=5,
                estimated_services=2,
                confidence=0.8,
                methodology="comparative_analysis"
            )
            
            # Test connection optimization
            mock_connections = {
                "mariadb": {
                    "service_type": "mariadb",
                    "connection_pool_size": 5,
                    "optimization_features": {},
                    "business_data_optimizations": {}
                }
            }
            
            optimized = await optimizer.optimize_connections(
                mock_connections, complexity_score
            )
            
            if not optimized or "mariadb" not in optimized:
                self.log_test_result(
                    "resource_optimizer", 
                    False, 
                    "Failed to optimize connections"
                )
                return False
            
            # Test resource allocation calculation
            allocations = await optimizer.calculate_resource_allocation(
                ["mariadb", "postgresql"], complexity_score
            )
            
            if len(allocations) != 2:
                self.log_test_result(
                    "resource_optimizer", 
                    False, 
                    "Failed to calculate resource allocations"
                )
                return False
            
            # Test health check
            is_healthy = await optimizer.health_check()
            
            await optimizer.cleanup()
            
            self.log_test_result(
                "resource_optimizer", 
                True, 
                "Resource optimizer functioning properly"
            )
            return True
            
        except Exception as e:
            self.log_test_result("resource_optimizer", False, str(e))
            return False
    
    async def test_health_monitor(self) -> bool:
        """Test Health Monitor functionality."""
        try:
            monitor = HealthMonitor()
            await monitor.initialize()
            
            # Test service health checks
            mariadb_health = await monitor.check_service_health(settings.mariadb_service_name)
            
            # Test overall health status
            overall_health = await monitor.get_overall_health_status()
            
            if "overall_status" not in overall_health:
                self.log_test_result(
                    "health_monitor", 
                    False, 
                    "Failed to get overall health status"
                )
                return False
            
            # Test service metrics
            all_metrics = await monitor.get_all_service_metrics()
            
            if not all_metrics:
                self.log_test_result(
                    "health_monitor", 
                    False, 
                    "Failed to get service metrics"
                )
                return False
            
            # Test health check
            is_healthy = await monitor.health_check()
            
            await monitor.cleanup()
            
            self.log_test_result(
                "health_monitor", 
                True, 
                "Health monitor functioning properly"
            )
            return True
            
        except Exception as e:
            self.log_test_result("health_monitor", False, str(e))
            return False
    
    async def test_service_orchestrator(self) -> bool:
        """Test main Service Orchestrator functionality."""
        try:
            orchestrator = ServiceOrchestrator()
            await orchestrator.initialize()
            
            # Create test inputs
            complexity_score = ComplexityScore(
                level=ComplexityLevel.ANALYTICAL,
                score=0.4,
                estimated_duration_minutes=15,
                estimated_queries=5,
                estimated_services=2,
                confidence=0.8,
                methodology="comparative_analysis"
            )
            
            contextual_strategy = ContextualStrategy(
                adapted_methodology="comparative_analysis",
                estimated_timeline={"analysis": 5, "data_gathering": 6, "synthesis": 3, "reporting": 1},
                communication_style="professional",
                deliverable_format="structured_report",
                user_preferences={"speed_preference": 0.7},
                organizational_constraints={}
            )
            
            # Test orchestration
            result = await orchestrator.orchestrate_services(
                complexity_score, contextual_strategy
            )
            
            if not result.coordinated_services:
                self.log_test_result(
                    "service_orchestrator", 
                    False, 
                    "Failed to coordinate services"
                )
                return False
            
            # Verify expected services are selected
            service_types = [s.service_type.value for s in result.coordinated_services if s.enabled]
            
            if "mariadb" not in service_types:
                self.log_test_result(
                    "service_orchestrator", 
                    False, 
                    "MariaDB service not selected"
                )
                return False
            
            # Test health check
            health_status = await orchestrator.health_check()
            
            self.log_test_result(
                "service_orchestrator", 
                True, 
                f"Orchestrated {len(service_types)} services successfully"
            )
            return True
            
        except Exception as e:
            self.log_test_result("service_orchestrator", False, str(e))
            return False
    
    async def test_orchestration_runner(self) -> bool:
        """Test Service Orchestration Runner functionality."""
        try:
            runner = ServiceOrchestrationRunner()
            await runner.initialize()
            
            # Test orchestration with simple inputs
            result = await runner.orchestrate(
                complexity_level="analytical",
                complexity_score=0.4,
                estimated_duration_minutes=15,
                estimated_queries=5,
                methodology="comparative_analysis",
                user_preferences={"speed_preference": 0.7}
            )
            
            if not result or "coordinated_services" not in result:
                self.log_test_result(
                    "orchestration_runner", 
                    False, 
                    "Failed to get orchestration result"
                )
                return False
            
            # Test health check
            health_status = await runner.health_check()
            
            if health_status.get("overall_status") == "error":
                self.log_test_result(
                    "orchestration_runner", 
                    False, 
                    f"Health check failed: {health_status}"
                )
                return False
            
            # Test service status
            service_status = await runner.get_service_status()
            
            if "error" in service_status:
                self.log_test_result(
                    "orchestration_runner", 
                    False, 
                    f"Service status failed: {service_status['error']}"
                )
                return False
            
            await runner.cleanup()
            
            self.log_test_result(
                "orchestration_runner", 
                True, 
                "Orchestration runner functioning properly"
            )
            return True
            
        except Exception as e:
            self.log_test_result("orchestration_runner", False, str(e))
            return False
    
    async def test_high_level_interface(self) -> bool:
        """Test high-level orchestration interface."""
        try:
            # Test the main orchestrate_investigation function
            result = await orchestrate_investigation(
                complexity_level="simple",
                complexity_score=0.2,
                estimated_duration_minutes=5,
                estimated_queries=2,
                methodology="descriptive_analysis"
            )
            
            if not result or "coordinated_services" not in result:
                self.log_test_result(
                    "high_level_interface", 
                    False, 
                    "Failed to orchestrate investigation"
                )
                return False
            
            # Verify simple complexity only uses MariaDB
            services = result["coordinated_services"]
            enabled_services = [s["service_type"] for s in services if s["enabled"]]
            
            if len(enabled_services) != 1 or "mariadb" not in enabled_services:
                self.log_test_result(
                    "high_level_interface", 
                    False, 
                    f"Unexpected services for simple complexity: {enabled_services}"
                )
                return False
            
            self.log_test_result(
                "high_level_interface", 
                True, 
                "High-level interface functioning properly"
            )
            return True
            
        except Exception as e:
            self.log_test_result("high_level_interface", False, str(e))
            return False
    
    async def test_complexity_levels(self) -> bool:
        """Test service selection for different complexity levels."""
        try:
            complexity_tests = [
                ("simple", 0.1, ["mariadb"]),
                ("analytical", 0.4, ["mariadb", "postgresql"]),
                ("computational", 0.7, ["mariadb", "postgresql", "lancedb"]),
                ("investigative", 0.9, ["mariadb", "postgresql", "lancedb", "graphrag"])
            ]
            
            for complexity_level, score, expected_services in complexity_tests:
                result = await orchestrate_investigation(
                    complexity_level=complexity_level,
                    complexity_score=score,
                    estimated_duration_minutes=10,
                    estimated_queries=3,
                    methodology="test_methodology"
                )
                
                services = result["coordinated_services"]
                enabled_services = [s["service_type"] for s in services if s["enabled"]]
                
                if set(enabled_services) != set(expected_services):
                    self.log_test_result(
                        "complexity_levels", 
                        False, 
                        f"Wrong services for {complexity_level}: got {enabled_services}, expected {expected_services}"
                    )
                    return False
            
            self.log_test_result(
                "complexity_levels", 
                True, 
                "All complexity levels working correctly"
            )
            return True
            
        except Exception as e:
            self.log_test_result("complexity_levels", False, str(e))
            return False
    
    async def test_performance_characteristics(self) -> bool:
        """Test performance characteristics of orchestration."""
        try:
            # Measure orchestration time
            start_time = time.time()
            
            result = await orchestrate_investigation(
                complexity_level="computational",
                complexity_score=0.7,
                estimated_duration_minutes=30,
                estimated_queries=10,
                methodology="advanced_analytics"
            )
            
            orchestration_time = time.time() - start_time
            
            # Orchestration should complete quickly (under 5 seconds)
            if orchestration_time > 5.0:
                self.log_test_result(
                    "performance_characteristics", 
                    False, 
                    f"Orchestration too slow: {orchestration_time:.2f}s"
                )
                return False
            
            # Check performance estimates are present
            if "estimated_performance" not in result:
                self.log_test_result(
                    "performance_characteristics", 
                    False, 
                    "Missing performance estimates"
                )
                return False
            
            performance = result["estimated_performance"]
            required_metrics = [
                "expected_response_time_seconds",
                "estimated_throughput_qps",
                "resource_utilization"
            ]
            
            for metric in required_metrics:
                if metric not in performance:
                    self.log_test_result(
                        "performance_characteristics", 
                        False, 
                        f"Missing performance metric: {metric}"
                    )
                    return False
            
            self.log_test_result(
                "performance_characteristics", 
                True, 
                f"Performance characteristics valid (orchestration: {orchestration_time:.2f}s)"
            )
            return True
            
        except Exception as e:
            self.log_test_result("performance_characteristics", False, str(e))
            return False
    
    def print_test_summary(self) -> None:
        """Print comprehensive test summary."""
        total_tests = self.passed_tests + self.failed_tests
        success_rate = (self.passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        print("\n" + "=" * 60)
        print("SERVICE ORCHESTRATION MODULE TEST SUMMARY")
        print("=" * 60)
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {self.passed_tests}")
        print(f"Failed: {self.failed_tests}")
        print(f"Success Rate: {success_rate:.1f}%")
        print("\nTest Details:")
        
        for test_name, result in self.test_results.items():
            status = "PASS" if result["passed"] else "FAIL"
            print(f"  {test_name}: {status}")
            if result["details"] and not result["passed"]:
                print(f"    Details: {result['details']}")
        
        print("\n" + "=" * 60)
        
        if self.failed_tests == 0:
            print("All tests passed! Service Orchestration module is functioning correctly.")
        else:
            print(f"{self.failed_tests} test(s) failed. Please review the issues above.")
        
        print("=" * 60)
    
    async def run_all_tests(self) -> bool:
        """Run all tests and return overall success."""
        print("Starting Service Orchestration Module Tests...\n")
        
        test_methods = [
            self.test_configuration_loading,
            self.test_mcp_coordinator,
            self.test_resource_optimizer,
            self.test_health_monitor,
            self.test_service_orchestrator,
            self.test_orchestration_runner,
            self.test_high_level_interface,
            self.test_complexity_levels,
            self.test_performance_characteristics
        ]
        
        for test_method in test_methods:
            try:
                await test_method()
            except Exception as e:
                self.log_test_result(test_method.__name__, False, f"Test exception: {e}")
        
        self.print_test_summary()
        return self.failed_tests == 0


async def main() -> int:
    """Main test execution."""
    tester = ServiceOrchestrationTester()
    
    try:
        success = await tester.run_all_tests()
        return 0 if success else 1
        
    except Exception as e:
        print(f"\nFatal error during testing: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
