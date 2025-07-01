#!/usr/bin/env python3
"""Standalone test suite for Phase 4 Investigation module validation."""

import asyncio
import sys
from pathlib import Path

# Add module to path for testing
sys.path.insert(0, str(Path(__file__).parent))

from config import settings
from runner import AutonomousInvestigationEngine, conduct_autonomous_investigation
from investigation_logging import InvestigationLogger
from prompts import InvestigationPrompts, PromptTemplates

class MockMCPClient:
    """Mock MCP client for testing."""
    
    async def list_tables(self):
        return ["sales", "customers", "products"]
    
    async def get_table_schema(self, table_name):
        from types import SimpleNamespace
        schema = SimpleNamespace()
        schema.columns = [
            SimpleNamespace(name="id", type="INT"),
            SimpleNamespace(name="name", type="VARCHAR"),
            SimpleNamespace(name="created_at", type="TIMESTAMP")
        ]
        return schema
    
    async def get_table_count(self, table_name):
        return 1000
    
    async def execute_query(self, query):
        from types import SimpleNamespace
        result = SimpleNamespace()
        result.rows = [{"count": 1000}]
        result.columns = ["count"]
        result.execution_time = 0.1
        return result
    
    async def test_connection(self):
        return True

class MockMCPClientManager:
    """Mock MCP client manager for testing."""
    
    def __init__(self):
        self.mock_client = MockMCPClient()
    
    def get_client(self, client_name):
        return self.mock_client

async def test_configuration():
    """Test configuration loading."""
    print("Testing Configuration...")
    try:
        print(f"Investigation Timeout: {settings.investigation_timeout_minutes} minutes")
        print(f"Confidence Threshold: {settings.confidence_threshold}")
        print(f"MCP Endpoints Count: {len(settings.get_mcp_endpoints())}")
        print(f"Max Hypotheses: {settings.max_hypotheses}")
        print("Configuration test: PASS")
        return True
    except Exception as e:
        print(f"Configuration test: FAIL - {e}")
        return False

async def test_logging():
    """Test investigation logging functionality."""
    print("\nTesting Investigation Logging...")
    try:
        logger = InvestigationLogger("test_investigation")
        logger.log_step_start("test_step", 1)
        logger.log_finding("Test finding", 0.85)
        logger.log_adaptive_reasoning("Test adaptive reasoning")
        logger.log_step_complete("test_step", 1, 2.5)
        logger.log_investigation_summary(7, 60.0, 0.82)
        print("Investigation logging test: PASS")
        return True
    except Exception as e:
        print(f"Investigation logging test: FAIL - {e}")
        return False

async def test_prompts():
    """Test prompt generation functionality."""
    print("\nTesting Prompt Generation...")
    try:
        # Test coordinated services formatting
        services = {
            "mariadb": {"enabled": True, "optimization_settings": {"pool_size": 10}},
            "postgresql": {"enabled": True, "optimization_settings": {"cache": True}}
        }
        formatted = PromptTemplates.format_coordinated_services(services)
        assert "MARIADB: ENABLED" in formatted
        
        # Test investigation prompt formatting
        investigation_prompt = InvestigationPrompts.format_investigation_prompt(
            coordinated_services=services,
            investigation_request="Test investigation request",
            execution_context={"user_role": "analyst"}
        )
        assert "Test investigation request" in investigation_prompt
        
        # Test step prompt formatting
        context = {
            "investigation_request": "Test request",
            "coordinated_services": formatted,
            "execution_context": {"user_role": "analyst"}
        }
        schema_prompt = InvestigationPrompts.format_step_prompt("schema_analysis", context)
        assert "schema analysis" in schema_prompt.lower()
        
        print("Prompt generation test: PASS")
        return True
    except Exception as e:
        import traceback
        print(f"Prompt generation test: FAIL - {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

async def test_investigation_engine():
    """Test autonomous investigation engine initialization."""
    print("\nTesting Investigation Engine...")
    try:
        engine = AutonomousInvestigationEngine()
        
        # Test engine initialization
        assert engine.investigation_id is None
        assert len(engine.step_definitions) == 7
        assert engine.step_definitions[0]["name"] == "schema_analysis"
        assert engine.step_definitions[6]["name"] == "results_synthesis"
        
        # Test step classification methods
        assert engine._classify_investigation_type("Why did sales drop?") == "diagnostic"
        assert engine._classify_investigation_type("Predict next quarter revenue") == "predictive"
        assert engine._classify_investigation_type("Compare Q1 vs Q2") == "comparative"
        assert engine._classify_investigation_type("What are total sales?") == "descriptive"
        
        # Test strategic importance assessment
        assert engine._assess_strategic_importance("Revenue analysis for customers") == "high"
        assert engine._assess_strategic_importance("Process efficiency improvement") == "medium"
        assert engine._assess_strategic_importance("General data exploration") == "low"
        
        print("Investigation engine test: PASS")
        return True
    except Exception as e:
        print(f"Investigation engine test: FAIL - {e}")
        return False

async def test_investigation_execution():
    """Test full investigation execution flow (with mock data)."""
    print("\nTesting Investigation Execution...")
    try:
        # Mock coordinated services
        coordinated_services = {
            "mariadb": {
                "enabled": True,
                "priority": 1,
                "optimization_settings": {"connection_pool_size": 10}
            },
            "postgresql": {
                "enabled": True, 
                "priority": 2,
                "optimization_settings": {"cache_enabled": True}
            }
        }
        
        # Test investigation request
        investigation_request = "Analyze last quarter's sales performance by region"
        
        # Test execution context
        execution_context = {
            "user_role": "business_analyst",
            "business_domain": "sales",
            "urgency_level": "medium",
            "complexity_level": "analytical"
        }
        
        # Run investigation
        # Mock MCP client manager for testing
        mock_mcp_manager = MockMCPClientManager()
        
        results = await conduct_autonomous_investigation(
            coordinated_services=coordinated_services,
            investigation_request=investigation_request,
            execution_context=execution_context,
            mcp_client_manager=mock_mcp_manager
        )
        
        # Validate results structure
        assert results.investigation_id is not None
        assert results.investigation_request == investigation_request
        assert results.status in ["completed", "failed", "partial"]
        assert isinstance(results.total_duration_seconds, float)
        assert 0.0 <= results.overall_confidence <= 1.0
        assert isinstance(results.investigation_findings, dict)
        assert isinstance(results.confidence_scores, dict)
        assert isinstance(results.validation_status, dict)
        assert isinstance(results.business_context, dict)
        assert isinstance(results.adaptive_reasoning_log, list)
        assert isinstance(results.completed_steps, list)
        assert len(results.completed_steps) == 7  # All 7 steps should be attempted
        
        # Validate FastMCP integration - check for database_results in findings
        schema_step = next((step for step in results.completed_steps if step.step_name == "schema_analysis"), None)
        if schema_step and schema_step.findings:
            assert "database_results" in schema_step.findings
            print(f"FastMCP Integration: Schema analysis found {len(schema_step.findings.get('database_results', {}))} database results")
        
        print(f"Investigation ID: {results.investigation_id}")
        print(f"Status: {results.status}")
        print(f"Duration: {results.total_duration_seconds:.2f} seconds")
        print(f"Overall Confidence: {results.overall_confidence:.2f}")
        print(f"Steps Completed: {len(results.completed_steps)}")
        print(f"Findings Categories: {list(results.investigation_findings.keys())}")
        
        print("Investigation execution test: PASS")
        return True
    except Exception as e:
        print(f"Investigation execution test: FAIL - {e}")
        return False

async def test_error_handling():
    """Test error handling and recovery capabilities."""
    print("\nTesting Error Handling...")
    try:
        engine = AutonomousInvestigationEngine()
        
        # Test with invalid services (should handle gracefully)
        invalid_services = {}
        investigation_request = "Test error handling"
        execution_context = {"user_role": "test"}
        
        results = await engine.conduct_investigation(
            coordinated_services=invalid_services,
            investigation_request=investigation_request,
            execution_context=execution_context
        )
        
        # Should complete but with lower confidence or partial status
        assert results is not None
        assert results.investigation_id is not None
        
        print("Error handling test: PASS")
        return True
    except Exception as e:
        print(f"Error handling test: FAIL - {e}")
        return False

async def main():
    """Run all tests."""
    print("Running Phase 4 Investigation Module Tests...\n")
    
    test_results = []
    
    # Run all tests
    test_results.append(await test_configuration())
    test_results.append(await test_logging())
    test_results.append(await test_prompts())
    test_results.append(await test_investigation_engine())
    test_results.append(await test_investigation_execution())
    test_results.append(await test_error_handling())
    
    # Summary
    passed = sum(test_results)
    total = len(test_results)
    
    print(f"\nTest Summary:")
    print(f"Passed: {passed}/{total}")
    print(f"Configuration: {'PASS' if test_results[0] else 'FAIL'}")
    print(f"Logging: {'PASS' if test_results[1] else 'FAIL'}")
    print(f"Prompts: {'PASS' if test_results[2] else 'FAIL'}")
    print(f"Engine: {'PASS' if test_results[3] else 'FAIL'}")
    print(f"Execution: {'PASS' if test_results[4] else 'FAIL'}")
    print(f"Error Handling: {'PASS' if test_results[5] else 'FAIL'}")
    
    if passed == total:
        print("\nAll tests passed! Phase 4 Investigation module is ready.")
        return 0
    else:
        print(f"\n{total - passed} tests failed. Please review and fix issues.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)