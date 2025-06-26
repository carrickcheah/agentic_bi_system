#!/usr/bin/env python3
"""
Pattern Library Testing Script

Tests the complete pattern library functionality including:
- Pattern loading from JSON files
- Database schema creation
- Qdrant vector database integration
- Pattern matching and scoring
- Success rate tracking and updates
- Organizational learning features

Usage:
    python testing/scripts/test_pattern_library.py
"""

import sys
import asyncio
import json
from pathlib import Path
from datetime import datetime

# Add app to Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from app.intelligence.pattern_library import (
    PatternLibrary, PatternMatch, PatternStatistics, InvestigationOutcome, DataSource
)
from app.mcp.qdrant_client import QdrantClient
from app.mcp.postgres_client import PostgresClient
from app.utils.logging import logger


class PatternLibraryTester:
    """Comprehensive test suite for Pattern Library functionality."""
    
    def __init__(self):
        self.qdrant_client = None
        self.postgres_client = None
        self.pattern_library = None
        self.test_results = {
            "passed": 0,
            "failed": 0,
            "details": []
        }
    
    async def setup(self):
        """Setup test environment."""
        logger.info("🔧 Setting up Pattern Library test environment...")
        
        try:
            # Initialize MCP clients
            self.qdrant_client = QdrantClient()
            self.postgres_client = PostgresClient()
            
            # Initialize pattern library
            self.pattern_library = PatternLibrary(self.qdrant_client, self.postgres_client)
            
            # Create database schema
            await self._create_database_schema()
            
            logger.info("✅ Test environment setup complete")
            return True
            
        except Exception as e:
            logger.error(f"❌ Test environment setup failed: {e}")
            return False
    
    async def _create_database_schema(self):
        """Create database schema for pattern statistics."""
        schema_file = Path(__file__).parent / "create_pattern_statistics_tables.sql"
        
        if schema_file.exists():
            with open(schema_file, 'r') as f:
                schema_sql = f.read()
            
            # Execute schema creation
            await self.postgres_client.execute_query(schema_sql)
            logger.info("✅ Database schema created successfully")
        else:
            logger.warning("⚠️  Schema file not found, skipping database setup")
    
    async def test_pattern_loading(self):
        """Test pattern loading from JSON files."""
        logger.info("🧪 Testing pattern loading...")
        
        try:
            # Test pattern library initialization
            await self.pattern_library.initialize()
            
            # Verify patterns were loaded
            pattern_count = len(self.pattern_library.patterns)
            expected_count = 150  # Total expected patterns
            
            if pattern_count == expected_count:
                self._record_test_pass(f"Pattern loading: {pattern_count}/{expected_count} patterns loaded")
            else:
                self._record_test_fail(f"Pattern loading: {pattern_count}/{expected_count} patterns loaded")
            
            # Test pattern file structure
            await self._test_pattern_structure()
            
        except Exception as e:
            self._record_test_fail(f"Pattern loading failed: {e}")
    
    async def _test_pattern_structure(self):
        """Test pattern data structure validity."""
        logger.info("🧪 Testing pattern structure...")
        
        required_fields = [
            "information", "metadata"
        ]
        
        required_metadata_fields = [
            "pattern", "user_roles", "business_domain", "timeframe", 
            "complexity", "success_rate", "confidence_indicators", 
            "expected_deliverables", "data_source", "sample_size"
        ]
        
        valid_patterns = 0
        
        for pattern_id, pattern in self.pattern_library.patterns.items():
            # Check required fields
            if all(field in pattern for field in required_fields):
                metadata = pattern["metadata"]
                if all(field in metadata for field in required_metadata_fields):
                    valid_patterns += 1
                else:
                    logger.warning(f"⚠️  Pattern {pattern_id} missing metadata fields")
            else:
                logger.warning(f"⚠️  Pattern {pattern_id} missing required fields")
        
        if valid_patterns == len(self.pattern_library.patterns):
            self._record_test_pass(f"Pattern structure: {valid_patterns} valid patterns")
        else:
            self._record_test_fail(f"Pattern structure: {valid_patterns}/{len(self.pattern_library.patterns)} valid patterns")
    
    async def test_pattern_matching(self):
        """Test pattern matching functionality."""
        logger.info("🧪 Testing pattern matching...")
        
        try:
            # Test queries for different business domains
            test_queries = [
                {
                    "query": "What is our daily production output compared to target?",
                    "user_context": {
                        "role": "production_manager",
                        "department": "production",
                        "experience_level": "intermediate"
                    },
                    "expected_domain": "production"
                },
                {
                    "query": "How can we reduce our defect rate in the assembly line?",
                    "user_context": {
                        "role": "quality_manager", 
                        "department": "quality",
                        "experience_level": "expert"
                    },
                    "expected_domain": "quality"
                },
                {
                    "query": "What are our supplier delivery performance issues?",
                    "user_context": {
                        "role": "procurement_manager",
                        "department": "supply_chain", 
                        "experience_level": "intermediate"
                    },
                    "expected_domain": "supply_chain"
                }
            ]
            
            for i, test_case in enumerate(test_queries, 1):
                matches = await self.pattern_library.find_matching_patterns(
                    test_case["query"],
                    test_case["user_context"],
                    top_k=3
                )
                
                if matches and len(matches) > 0:
                    top_match = matches[0]
                    domain_match = test_case["expected_domain"] in top_match.pattern_data["metadata"]["business_domain"]
                    
                    if domain_match:
                        self._record_test_pass(f"Pattern matching test {i}: Found relevant match")
                        logger.info(f"  📊 Top match: {top_match.pattern_data['information'][:60]}...")
                        logger.info(f"  📊 Score: {top_match.total_score:.3f}")
                    else:
                        self._record_test_fail(f"Pattern matching test {i}: Domain mismatch")
                else:
                    self._record_test_fail(f"Pattern matching test {i}: No matches found")
                    
        except Exception as e:
            self._record_test_fail(f"Pattern matching failed: {e}")
    
    async def test_success_rate_tracking(self):
        """Test success rate tracking and Bayesian updates."""
        logger.info("🧪 Testing success rate tracking...")
        
        try:
            # Create test investigation outcome
            test_pattern_id = "production_001"
            
            # Get initial statistics
            initial_stats = await self.pattern_library.get_pattern_statistics(test_pattern_id)
            initial_success_rate = initial_stats.success_rate if initial_stats else 0.55
            
            # Create successful outcome
            successful_outcome = InvestigationOutcome(
                pattern_id=test_pattern_id,
                investigation_id="test_investigation_001",
                user_id="test_user",
                completion_success=True,
                user_satisfaction_score=0.8,
                accuracy_validation=True,
                implementation_success=True,
                investigation_time_minutes=15.5,
                timestamp=datetime.utcnow()
            )
            
            # Track outcome and update success rate
            updated_stats = await self.pattern_library.track_investigation_outcome(successful_outcome)
            
            # Verify success rate increased
            if updated_stats.success_rate > initial_success_rate:
                self._record_test_pass(f"Success rate tracking: Rate increased from {initial_success_rate:.3f} to {updated_stats.success_rate:.3f}")
            else:
                self._record_test_fail(f"Success rate tracking: Rate did not increase as expected")
            
            # Test failed outcome
            failed_outcome = InvestigationOutcome(
                pattern_id=test_pattern_id,
                investigation_id="test_investigation_002", 
                user_id="test_user",
                completion_success=False,
                user_satisfaction_score=0.3,
                accuracy_validation=False,
                implementation_success=False,
                investigation_time_minutes=8.2,
                timestamp=datetime.utcnow()
            )
            
            # Track failed outcome
            updated_stats_2 = await self.pattern_library.track_investigation_outcome(failed_outcome)
            
            # Verify success rate decreased
            if updated_stats_2.success_rate < updated_stats.success_rate:
                self._record_test_pass(f"Success rate tracking: Rate decreased after failure to {updated_stats_2.success_rate:.3f}")
            else:
                self._record_test_fail(f"Success rate tracking: Rate did not decrease as expected")
                
        except Exception as e:
            self._record_test_fail(f"Success rate tracking failed: {e}")
    
    async def test_data_source_progression(self):
        """Test data source classification progression."""
        logger.info("🧪 Testing data source progression...")
        
        try:
            test_pattern_id = "quality_001"
            
            # Start with bootstrap estimate
            initial_stats = PatternStatistics(
                pattern_id=test_pattern_id,
                success_rate=0.55,
                confidence_interval=(0.45, 0.65),
                data_source=DataSource.BOOTSTRAP_ESTIMATE,
                sample_size=0,
                successes=0,
                failures=0,
                last_updated=datetime.utcnow(),
                usage_count=0
            )
            
            # Add multiple outcomes to progress through data sources
            for i in range(20):
                outcome = InvestigationOutcome(
                    pattern_id=test_pattern_id,
                    investigation_id=f"test_investigation_{i:03d}",
                    user_id="test_user",
                    completion_success=i % 3 != 0,  # 67% success rate
                    user_satisfaction_score=0.7 if i % 3 != 0 else 0.4,
                    accuracy_validation=i % 2 == 0,
                    implementation_success=i % 4 != 0,
                    investigation_time_minutes=10.0 + i,
                    timestamp=datetime.utcnow()
                )
                
                # Update statistics
                if i == 0:
                    # Initialize with first outcome
                    updated_stats = self.pattern_library._bayesian_update_success_rate(initial_stats, 0.7)
                    self.pattern_library.pattern_statistics[test_pattern_id] = updated_stats
                else:
                    current_stats = self.pattern_library.pattern_statistics[test_pattern_id]
                    updated_stats = await self.pattern_library.track_investigation_outcome(outcome)
            
            final_stats = self.pattern_library.pattern_statistics[test_pattern_id]
            
            # Check data source progression
            if final_stats.data_source == DataSource.STATISTICAL_CONFIDENCE:
                self._record_test_pass(f"Data source progression: Reached statistical confidence with {final_stats.sample_size} samples")
            elif final_stats.data_source == DataSource.REAL_USAGE_DATA:
                self._record_test_pass(f"Data source progression: Reached real usage data with {final_stats.sample_size} samples")
            else:
                self._record_test_fail(f"Data source progression: Still at {final_stats.data_source.value} with {final_stats.sample_size} samples")
                
        except Exception as e:
            self._record_test_fail(f"Data source progression failed: {e}")
    
    async def test_domain_patterns(self):
        """Test domain-specific pattern retrieval."""
        logger.info("🧪 Testing domain pattern retrieval...")
        
        try:
            domains = ["production", "quality", "supply_chain", "cost_management", "asset_management"]
            
            for domain in domains:
                patterns = await self.pattern_library.get_domain_patterns(domain)
                
                if patterns and len(patterns) > 0:
                    self._record_test_pass(f"Domain patterns: Found {len(patterns)} patterns for {domain}")
                else:
                    self._record_test_fail(f"Domain patterns: No patterns found for {domain}")
                    
        except Exception as e:
            self._record_test_fail(f"Domain pattern retrieval failed: {e}")
    
    def _record_test_pass(self, message: str):
        """Record a test pass."""
        self.test_results["passed"] += 1
        self.test_results["details"].append(f"✅ {message}")
        logger.info(f"✅ {message}")
    
    def _record_test_fail(self, message: str):
        """Record a test failure."""
        self.test_results["failed"] += 1  
        self.test_results["details"].append(f"❌ {message}")
        logger.error(f"❌ {message}")
    
    async def run_all_tests(self):
        """Run the complete test suite."""
        logger.info("🚀 Starting Pattern Library Test Suite...")
        
        # Setup test environment
        if not await self.setup():
            logger.error("❌ Test setup failed, aborting tests")
            return False
        
        # Run test suite
        await self.test_pattern_loading()
        await self.test_pattern_matching()
        await self.test_success_rate_tracking()
        await self.test_data_source_progression()
        await self.test_domain_patterns()
        
        # Print test summary
        self._print_test_summary()
        
        return self.test_results["failed"] == 0
    
    def _print_test_summary(self):
        """Print comprehensive test summary."""
        total_tests = self.test_results["passed"] + self.test_results["failed"]
        success_rate = self.test_results["passed"] / total_tests if total_tests > 0 else 0
        
        print("\n" + "="*80)
        print("🧪 PATTERN LIBRARY TEST SUMMARY")
        print("="*80)
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {self.test_results['passed']}")
        print(f"Failed: {self.test_results['failed']}")
        print(f"Success Rate: {success_rate:.1%}")
        print("\nDetailed Results:")
        print("-"*40)
        
        for detail in self.test_results["details"]:
            print(detail)
        
        print("="*80)
        
        if self.test_results["failed"] == 0:
            print("🎉 ALL TESTS PASSED! Pattern Library is ready for production.")
        else:
            print("⚠️  Some tests failed. Please review and fix issues before deployment.")
        
        print("="*80)


async def main():
    """Main test runner."""
    tester = PatternLibraryTester()
    success = await tester.run_all_tests()
    
    if success:
        print("\n🎉 Pattern Library test suite completed successfully!")
        print("📝 Next steps:")
        print("   1. Integration with QueryProcessor")
        print("   2. Pattern-aware cache enhancement")
        print("   3. Investigation methodology selection")
        print("   4. Organizational learning implementation")
        return 0
    else:
        print("\n❌ Pattern Library test suite failed!")
        print("   Please fix the failing tests before proceeding.")
        return 1


if __name__ == "__main__":
    import sys
    exit_code = asyncio.run(main())
    sys.exit(exit_code)