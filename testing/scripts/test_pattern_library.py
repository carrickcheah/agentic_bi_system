"""
Comprehensive Testing Script for Pattern Library Functionality

Tests pattern loading, semantic search, recommendations, and integration
with the Qdrant vector database via MCP clients.
"""

import asyncio
import sys
from pathlib import Path
from typing import Dict, Any
import json

# Add the app directory to Python path for imports
app_dir = Path(__file__).parent.parent.parent / "app"
sys.path.insert(0, str(app_dir))

from utils.logging import setup_logging, logger
from fastmcp.client_manager import MCPClientManager
from intelligence.pattern_loader import PatternLoader
from intelligence.pattern_library import PatternLibrary, PatternMatch, BusinessDomain, ComplexityLevel


class PatternLibraryTester:
    """Comprehensive testing suite for pattern library functionality."""
    
    def __init__(self):
        self.client_manager = None
        self.pattern_loader = None
        self.pattern_library = None
        self.test_results = {
            "pattern_loading": {},
            "semantic_search": {},
            "recommendations": {},
            "integration": {},
            "performance": {}
        }
    
    async def setup(self):
        """Setup test environment."""
        logger.info("🔧 Setting up pattern library test environment...")
        
        try:
            # Initialize MCP client manager
            self.client_manager = MCPClientManager()
            await self.client_manager.initialize()
            
            # Initialize pattern loader
            self.pattern_loader = PatternLoader(self.client_manager)
            
            # Initialize pattern library
            self.pattern_library = PatternLibrary(self.client_manager)
            
            logger.info("✅ Test environment setup completed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Test environment setup failed: {e}")
            return False
    
    async def test_pattern_loading(self):
        """Test pattern loading functionality."""
        logger.info("🧪 Testing pattern loading...")
        
        test_results = {
            "load_success": False,
            "patterns_loaded": 0,
            "verification_passed": False,
            "errors": []
        }
        
        try:
            # Test pattern loading
            load_success = await self.pattern_loader.load_all_patterns()
            test_results["load_success"] = load_success
            
            if load_success:
                logger.info("✅ Pattern loading successful")
                
                # Verify patterns were loaded
                verification = await self.pattern_loader.verify_patterns()
                test_results["verification_passed"] = verification.get("success", False)
                test_results["patterns_loaded"] = verification.get("total_searches", 0)
                
                if test_results["verification_passed"]:
                    logger.info("✅ Pattern verification passed")
                else:
                    logger.warning(f"⚠️  Pattern verification issues: {verification}")
            else:
                logger.error("❌ Pattern loading failed")
                test_results["errors"].append("Pattern loading failed")
                
        except Exception as e:
            logger.error(f"❌ Pattern loading test error: {e}")
            test_results["errors"].append(str(e))
        
        self.test_results["pattern_loading"] = test_results
        return test_results["load_success"] and test_results["verification_passed"]
    
    async def test_semantic_search(self):
        """Test semantic search functionality."""
        logger.info("🧪 Testing semantic search...")
        
        test_results = {
            "searches_completed": 0,
            "total_searches": 0,
            "search_results": {},
            "average_results_per_query": 0.0,
            "errors": []
        }
        
        # Test queries across different business domains
        test_queries = [
            {
                "query": "monthly revenue performance analysis",
                "expected_domain": "sales",
                "min_results": 1
            },
            {
                "query": "equipment maintenance optimization",
                "expected_domain": "asset_management", 
                "min_results": 1
            },
            {
                "query": "supply chain inventory management",
                "expected_domain": "supply_chain",
                "min_results": 1
            },
            {
                "query": "customer satisfaction improvement",
                "expected_domain": "customer",
                "min_results": 1
            },
            {
                "query": "cost reduction initiatives",
                "expected_domain": "cost_management",
                "min_results": 1
            }
        ]
        
        test_results["total_searches"] = len(test_queries)
        total_results = 0
        
        try:
            # Initialize pattern library if not already done
            if not self.pattern_library.patterns_loaded:
                await self.pattern_library.initialize()
            
            for test_query in test_queries:
                try:
                    # Create search request
                    from intelligence.pattern_library import PatternSearchRequest
                    search_request = PatternSearchRequest(
                        query=test_query["query"],
                        max_results=5,
                        min_similarity=0.3
                    )
                    
                    # Perform search using the new method
                    matches = await self.pattern_library.search_patterns(search_request)
                    
                    # Analyze results
                    result_info = {
                        "query": test_query["query"],
                        "results_count": len(matches),
                        "top_similarity": matches[0].similarity_score if matches else 0.0,
                        "domain_match": False,
                        "meets_minimum": len(matches) >= test_query["min_results"]
                    }
                    
                    # Check domain relevance
                    if matches:
                        for match in matches[:3]:  # Check top 3 results
                            pattern_domain = match.metadata.get("business_domain", "")
                            if test_query["expected_domain"] in pattern_domain:
                                result_info["domain_match"] = True
                                break
                    
                    test_results["search_results"][test_query["query"]] = result_info
                    total_results += len(matches)
                    
                    if result_info["meets_minimum"]:
                        test_results["searches_completed"] += 1
                        logger.info(f"✅ Search successful: {test_query['query']} ({len(matches)} results)")
                    else:
                        logger.warning(f"⚠️  Insufficient results for: {test_query['query']}")
                
                except Exception as e:
                    logger.error(f"❌ Search failed for '{test_query['query']}': {e}")
                    test_results["errors"].append(f"Search error for '{test_query['query']}': {str(e)}")
            
            # Calculate average results per query
            if test_results["total_searches"] > 0:
                test_results["average_results_per_query"] = total_results / test_results["total_searches"]
            
        except Exception as e:
            logger.error(f"❌ Semantic search testing error: {e}")
            test_results["errors"].append(str(e))
        
        self.test_results["semantic_search"] = test_results
        success_rate = test_results["searches_completed"] / test_results["total_searches"] if test_results["total_searches"] > 0 else 0.0
        logger.info(f"📊 Semantic search success rate: {success_rate:.1%}")
        
        return success_rate >= 0.8  # 80% success rate threshold
    
    async def test_investigation_recommendations(self):
        """Test investigation recommendation functionality."""
        logger.info("🧪 Testing investigation recommendations...")
        
        test_results = {
            "recommendations_generated": 0,
            "total_tests": 0,
            "recommendation_quality": {},
            "errors": []
        }
        
        # Test scenarios with user context
        test_scenarios = [
            {
                "business_query": "How can we improve our monthly sales performance?",
                "user_context": {
                    "roles": ["sales_manager"],
                    "domain": "sales",
                    "complexity": "moderate"
                },
                "expected_patterns": ["revenue", "sales", "performance"]
            },
            {
                "business_query": "Why is our equipment failing more frequently?",
                "user_context": {
                    "roles": ["maintenance_manager"],
                    "domain": "asset_management",
                    "complexity": "complex"
                },
                "expected_patterns": ["equipment", "failure", "maintenance"]
            },
            {
                "business_query": "How to optimize inventory levels?",
                "user_context": {
                    "roles": ["supply_chain_manager"],
                    "domain": "supply_chain",
                    "complexity": "moderate"
                },
                "expected_patterns": ["inventory", "optimization", "supply"]
            }
        ]
        
        test_results["total_tests"] = len(test_scenarios)
        
        try:
            for scenario in test_scenarios:
                try:
                    # Generate recommendation
                    recommendation = await self.pattern_library.recommend_investigation_approach(
                        scenario["business_query"],
                        scenario["user_context"]
                    )
                    
                    # Analyze recommendation quality
                    quality_score = 0.0
                    quality_details = {
                        "success": recommendation.get("success", False),
                        "recommendations_count": len(recommendation.get("recommendations", [])),
                        "relevance_score": 0.0,
                        "context_alignment": False
                    }
                    
                    if recommendation.get("success", False):
                        quality_score += 0.3  # Base score for successful generation
                        
                        recommendations = recommendation.get("recommendations", [])
                        if recommendations:
                            quality_score += 0.2  # Score for having recommendations
                            
                            # Check relevance by looking for expected patterns
                            top_rec = recommendations[0]
                            investigation_text = top_rec.get("investigation_type", "").lower()
                            
                            pattern_matches = sum(1 for pattern in scenario["expected_patterns"] 
                                               if pattern in investigation_text)
                            relevance_score = pattern_matches / len(scenario["expected_patterns"])
                            quality_details["relevance_score"] = relevance_score
                            quality_score += relevance_score * 0.3
                            
                            # Check context alignment
                            if top_rec.get("complexity") == scenario["user_context"]["complexity"]:
                                quality_details["context_alignment"] = True
                                quality_score += 0.2
                    
                    quality_details["total_score"] = quality_score
                    test_results["recommendation_quality"][scenario["business_query"]] = quality_details
                    
                    if quality_score >= 0.6:  # 60% quality threshold
                        test_results["recommendations_generated"] += 1
                        logger.info(f"✅ Good recommendation for: {scenario['business_query'][:50]}...")
                    else:
                        logger.warning(f"⚠️  Low quality recommendation for: {scenario['business_query'][:50]}...")
                
                except Exception as e:
                    logger.error(f"❌ Recommendation failed for scenario: {e}")
                    test_results["errors"].append(f"Recommendation error: {str(e)}")
        
        except Exception as e:
            logger.error(f"❌ Recommendation testing error: {e}")
            test_results["errors"].append(str(e))
        
        self.test_results["recommendations"] = test_results
        success_rate = test_results["recommendations_generated"] / test_results["total_tests"] if test_results["total_tests"] > 0 else 0.0
        logger.info(f"📊 Recommendation success rate: {success_rate:.1%}")
        
        return success_rate >= 0.7  # 70% success rate threshold
    
    async def test_mcp_integration(self):
        """Test MCP client integration."""
        logger.info("🧪 Testing MCP integration...")
        
        test_results = {
            "qdrant_available": False,
            "postgres_available": False,
            "health_check_passed": False,
            "errors": []
        }
        
        try:
            # Test Qdrant client availability
            qdrant_client = self.client_manager.get_client("qdrant")
            test_results["qdrant_available"] = qdrant_client is not None
            
            if qdrant_client:
                logger.info("✅ Qdrant client available")
            else:
                logger.warning("⚠️  Qdrant client not available")
                test_results["errors"].append("Qdrant client not available")
            
            # Test PostgreSQL client availability
            postgres_client = self.client_manager.get_client("postgres")
            test_results["postgres_available"] = postgres_client is not None
            
            if postgres_client:
                logger.info("✅ PostgreSQL client available")
            else:
                logger.warning("⚠️  PostgreSQL client not available")
                test_results["errors"].append("PostgreSQL client not available")
            
            # Test pattern library health check
            health_check = await self.pattern_library.health_check()
            test_results["health_check_passed"] = health_check.get("healthy", False)
            
            if test_results["health_check_passed"]:
                logger.info("✅ Pattern library health check passed")
            else:
                logger.warning(f"⚠️  Pattern library health check failed: {health_check}")
                test_results["errors"].append(f"Health check failed: {health_check}")
        
        except Exception as e:
            logger.error(f"❌ MCP integration test error: {e}")
            test_results["errors"].append(str(e))
        
        self.test_results["integration"] = test_results
        return test_results["qdrant_available"] and test_results["health_check_passed"]
    
    async def test_performance(self):
        """Test performance characteristics."""
        logger.info("🧪 Testing performance...")
        
        test_results = {
            "pattern_loading_time": 0.0,
            "average_search_time": 0.0,
            "recommendation_time": 0.0,
            "performance_acceptable": False,
            "errors": []
        }
        
        try:
            import time
            
            # Test pattern loading performance
            start_time = time.time()
            await self.pattern_library.initialize()
            test_results["pattern_loading_time"] = time.time() - start_time
            
            # Test search performance
            search_times = []
            test_queries = [
                "revenue analysis",
                "equipment maintenance", 
                "inventory optimization"
            ]
            
            for query in test_queries:
                start_time = time.time()
                # Create a simple search
                from intelligence.pattern_library import PatternSearchRequest
                search_request = PatternSearchRequest(query=query, max_results=3)
                await self.pattern_library.search_patterns(search_request)
                search_times.append(time.time() - start_time)
            
            test_results["average_search_time"] = sum(search_times) / len(search_times) if search_times else 0.0
            
            # Test recommendation performance
            start_time = time.time()
            await self.pattern_library.recommend_investigation_approach(
                "How to improve sales performance?",
                {"roles": ["sales_manager"], "domain": "sales"}
            )
            test_results["recommendation_time"] = time.time() - start_time
            
            # Evaluate performance acceptability
            performance_acceptable = (
                test_results["pattern_loading_time"] < 30.0 and  # 30 seconds max
                test_results["average_search_time"] < 2.0 and    # 2 seconds max  
                test_results["recommendation_time"] < 5.0        # 5 seconds max
            )
            test_results["performance_acceptable"] = performance_acceptable
            
            logger.info(f"📊 Pattern loading time: {test_results['pattern_loading_time']:.2f}s")
            logger.info(f"📊 Average search time: {test_results['average_search_time']:.2f}s")
            logger.info(f"📊 Recommendation time: {test_results['recommendation_time']:.2f}s")
        
        except Exception as e:
            logger.error(f"❌ Performance testing error: {e}")
            test_results["errors"].append(str(e))
        
        self.test_results["performance"] = test_results
        return test_results["performance_acceptable"]
    
    async def run_all_tests(self):
        """Run all tests and generate comprehensive report."""
        logger.info("🚀 Starting comprehensive pattern library testing...")
        
        # Setup test environment
        setup_success = await self.setup()
        if not setup_success:
            logger.error("❌ Test setup failed, aborting tests")
            return False
        
        test_suite = [
            ("MCP Integration", self.test_mcp_integration),
            ("Pattern Loading", self.test_pattern_loading),
            ("Semantic Search", self.test_semantic_search),
            ("Investigation Recommendations", self.test_investigation_recommendations),
            ("Performance", self.test_performance)
        ]
        
        passed_tests = 0
        total_tests = len(test_suite)
        
        for test_name, test_func in test_suite:
            logger.info(f"\n{'='*60}")
            logger.info(f"Running {test_name} Tests")
            logger.info(f"{'='*60}")
            
            try:
                test_passed = await test_func()
                if test_passed:
                    logger.info(f"✅ {test_name} tests PASSED")
                    passed_tests += 1
                else:
                    logger.error(f"❌ {test_name} tests FAILED")
            except Exception as e:
                logger.error(f"❌ {test_name} tests ERROR: {e}")
        
        # Generate final report
        await self.generate_test_report(passed_tests, total_tests)
        
        # Cleanup
        await self.cleanup()
        
        return passed_tests == total_tests
    
    async def generate_test_report(self, passed_tests: int, total_tests: int):
        """Generate comprehensive test report."""
        logger.info(f"\n{'='*60}")
        logger.info("PATTERN LIBRARY TEST REPORT")
        logger.info(f"{'='*60}")
        
        success_rate = passed_tests / total_tests if total_tests > 0 else 0.0
        logger.info(f"Overall Success Rate: {success_rate:.1%} ({passed_tests}/{total_tests})")
        
        # Detailed results
        for test_category, results in self.test_results.items():
            logger.info(f"\n{test_category.upper().replace('_', ' ')} RESULTS:")
            
            if isinstance(results, dict):
                for key, value in results.items():
                    if key != "errors":
                        logger.info(f"  {key}: {value}")
                
                if "errors" in results and results["errors"]:
                    logger.info("  Errors:")
                    for error in results["errors"]:
                        logger.info(f"    - {error}")
        
        # Save detailed report to file
        report_file = Path(__file__).parent / "pattern_library_test_report.json"
        try:
            with open(report_file, 'w') as f:
                json.dump({
                    "timestamp": str(asyncio.get_event_loop().time()),
                    "success_rate": success_rate,
                    "passed_tests": passed_tests,
                    "total_tests": total_tests,
                    "detailed_results": self.test_results
                }, f, indent=2)
            logger.info(f"📄 Detailed report saved to: {report_file}")
        except Exception as e:
            logger.warning(f"⚠️  Could not save report file: {e}")
        
        if success_rate >= 0.8:
            logger.info("🎉 Pattern library testing SUCCESSFUL!")
        else:
            logger.warning("⚠️  Pattern library testing needs attention")
    
    async def cleanup(self):
        """Cleanup test environment."""
        try:
            if self.client_manager:
                await self.client_manager.close()
            logger.info("🧹 Test environment cleanup completed")
        except Exception as e:
            logger.warning(f"⚠️  Cleanup warning: {e}")


async def main():
    """Main testing entry point."""
    setup_logging()
    
    logger.info("🧪 Pattern Library Comprehensive Testing")
    logger.info("=========================================")
    
    tester = PatternLibraryTester()
    
    try:
        success = await tester.run_all_tests()
        
        if success:
            logger.info("✅ All pattern library tests passed!")
            return 0
        else:
            logger.error("❌ Some pattern library tests failed!")
            return 1
    
    except Exception as e:
        logger.error(f"❌ Testing error: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)