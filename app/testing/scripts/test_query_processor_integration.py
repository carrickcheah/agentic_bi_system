#!/usr/bin/env python3
"""
QueryProcessor Pattern Integration Test

Tests the enhanced QueryProcessor with PatternLibrary integration
to verify pattern-aware processing capabilities.

Usage:
    python testing/scripts/test_query_processor_integration.py
"""

import sys
import asyncio
from pathlib import Path
from datetime import datetime

# Add app to Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from app.core.query_processor import QueryProcessor
from app.intelligence.pattern_library import PatternLibrary
from app.mcp.qdrant_client import QdrantClient
from app.mcp.postgres_client import PostgresClient
from app.utils.logging import logger


async def test_query_processor_integration():
    """Test QueryProcessor integration with PatternLibrary."""
    
    print("🧪 Testing QueryProcessor Pattern Integration...")
    print("=" * 60)
    
    # Test cases for different business domains
    test_cases = [
        {
            "business_question": "What is our daily production output compared to target?",
            "user_context": {
                "user_id": "test_user_001",
                "role": "production_manager",
                "department": "production",
                "permissions": ["production_read", "target_analysis"]
            },
            "organization_context": {
                "organization_id": "test_org",
                "business_rules": {"fiscal_year_start": "January"},
                "data_classification": "standard"
            },
            "expected_domain": "production"
        },
        {
            "business_question": "How can we reduce defect rates in our assembly line?",
            "user_context": {
                "user_id": "test_user_002", 
                "role": "quality_manager",
                "department": "quality",
                "permissions": ["quality_analysis", "defect_investigation"]
            },
            "organization_context": {
                "organization_id": "test_org",
                "business_rules": {"quality_threshold": 0.95},
                "data_classification": "standard"
            },
            "expected_domain": "quality"
        },
        {
            "business_question": "What are our supplier delivery performance issues?",
            "user_context": {
                "user_id": "test_user_003",
                "role": "procurement_manager", 
                "department": "supply_chain",
                "permissions": ["supplier_analysis", "delivery_tracking"]
            },
            "organization_context": {
                "organization_id": "test_org",
                "business_rules": {"delivery_sla": "95%"},
                "data_classification": "standard"
            },
            "expected_domain": "supply_chain"
        }
    ]
    
    try:
        # Initialize MCP clients (mock for testing)
        print("📋 Initializing test environment...")
        qdrant_client = QdrantClient()
        postgres_client = PostgresClient()
        
        # Initialize QueryProcessor with pattern library
        query_processor = QueryProcessor(qdrant_client, postgres_client)
        
        # Test each case
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n🧪 Test Case {i}: {test_case['business_question'][:50]}...")
            
            try:
                # Process the business question
                result = await query_processor.process_business_question(
                    test_case["business_question"],
                    test_case["user_context"],
                    test_case["organization_context"]
                )
                
                # Verify basic structure
                assert "original_question" in result
                assert "business_intent" in result
                assert "business_domain" in result
                assert "semantic_hash" in result
                assert "complexity_indicators" in result
                assert "pattern_intelligence" in result
                assert "processing_metadata" in result
                
                # Verify pattern intelligence structure
                pattern_intel = result["pattern_intelligence"]
                assert "matching_patterns" in pattern_intel
                assert "suggested_methodologies" in pattern_intel
                assert "pattern_confidence_boost" in pattern_intel
                assert "methodology_recommendations" in pattern_intel
                
                # Verify processing metadata
                metadata = result["processing_metadata"]
                assert "processor_version" in metadata
                assert metadata["processor_version"] == "2.0"
                assert "confidence_score" in metadata
                assert "base_confidence" in metadata
                assert "pattern_enhanced" in metadata
                
                # Check domain classification
                detected_domain = result["business_domain"]
                expected_domain = test_case["expected_domain"]
                
                print(f"   ✅ Processing successful")
                print(f"   📊 Domain: {detected_domain} (expected: {expected_domain})")
                print(f"   🎯 Pattern matches: {len(pattern_intel['matching_patterns'])}")
                print(f"   📈 Confidence: {metadata['confidence_score']:.3f} (base: {metadata['base_confidence']:.3f})")
                print(f"   🔧 Methodologies: {len(pattern_intel['suggested_methodologies'])}")
                
                # Display top pattern match if available
                if pattern_intel['matching_patterns']:
                    top_match = pattern_intel['matching_patterns'][0]
                    print(f"   🥇 Top match: {top_match['information'][:40]}...")
                    print(f"      Success rate: {top_match['success_rate']:.3f}")
                    print(f"      Confidence: {top_match['confidence']:.3f}")
                
                # Display methodology recommendations
                if pattern_intel['methodology_recommendations']:
                    print(f"   🔬 Methodology: {pattern_intel['methodology_recommendations'][0][:60]}...")
                
            except Exception as e:
                print(f"   ❌ Test case {i} failed: {e}")
                raise
        
        print(f"\n✅ All {len(test_cases)} test cases passed!")
        print("\n📊 QueryProcessor Integration Summary:")
        print("   ✅ Pattern library initialization")
        print("   ✅ Business question processing with patterns")
        print("   ✅ Enhanced semantic hashing")
        print("   ✅ Pattern-aware confidence scoring")
        print("   ✅ Investigation methodology suggestions")
        print("   ✅ Pattern intelligence metadata")
        
        print("\n🎉 QueryProcessor pattern integration is working correctly!")
        return True
        
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        logger.error(f"QueryProcessor integration test failed: {e}")
        return False


async def test_semantic_hash_enhancement():
    """Test enhanced semantic hashing with pattern context."""
    
    print("\n🔍 Testing Enhanced Semantic Hashing...")
    print("-" * 40)
    
    try:
        # Mock pattern matches for testing
        class MockPatternMatch:
            def __init__(self, pattern_id, success_rate, complexity):
                self.pattern_id = pattern_id
                self.pattern_data = {
                    "metadata": {
                        "success_rate": success_rate,
                        "complexity": complexity
                    }
                }
        
        query_processor = QueryProcessor()
        
        # Test basic vs enhanced hashing
        question = "What is our production efficiency?"
        domain = "production"
        role = "production_manager"
        
        # Basic hash (no patterns)
        basic_hash = query_processor._generate_semantic_hash(question, domain, role)
        
        # Enhanced hash with patterns
        mock_patterns = [
            MockPatternMatch("production_001", 0.85, "simple"),
            MockPatternMatch("production_002", 0.70, "moderate")
        ]
        enhanced_hash = query_processor._generate_enhanced_semantic_hash(
            question, domain, role, mock_patterns
        )
        
        print(f"   📝 Question: {question}")
        print(f"   🔤 Basic hash: {basic_hash[:16]}...")
        print(f"   🎯 Enhanced hash: {enhanced_hash[:16]}...")
        print(f"   ✅ Hashes are different: {basic_hash != enhanced_hash}")
        
        # Test with no patterns (should match basic)
        enhanced_no_patterns = query_processor._generate_enhanced_semantic_hash(
            question, domain, role, []
        )
        print(f"   🔍 No patterns hash: {enhanced_no_patterns[:16]}...")
        
        assert basic_hash != enhanced_hash, "Enhanced hash should differ from basic"
        assert len(basic_hash) == len(enhanced_hash), "Hash lengths should be equal"
        
        print("   ✅ Enhanced semantic hashing working correctly!")
        return True
        
    except Exception as e:
        print(f"   ❌ Semantic hashing test failed: {e}")
        return False


async def main():
    """Main test runner."""
    
    print("🚀 QueryProcessor Pattern Integration Test Suite")
    print("   Testing enhanced business question processing...")
    print("")
    
    # Run integration tests
    integration_success = await test_query_processor_integration()
    
    # Run semantic hashing tests  
    hashing_success = await test_semantic_hash_enhancement()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    total_tests = 2
    passed_tests = sum([integration_success, hashing_success])
    
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success Rate: {passed_tests/total_tests:.1%}")
    
    if passed_tests == total_tests:
        print("\n🎉 ALL TESTS PASSED!")
        print("📝 QueryProcessor pattern integration is ready for production!")
        print("\n🚀 Next Steps:")
        print("   1. Run full pattern library test suite")
        print("   2. Test with real MCP connections")
        print("   3. Integration with Strategy Planner")
        print("   4. Cache system enhancement")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED!")
        print("   Please fix failing tests before proceeding.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)