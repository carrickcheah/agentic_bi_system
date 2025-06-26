#!/usr/bin/env python3
"""
Expanded Pattern Library Test

Tests the enhanced pattern library with all 220 patterns across
manufacturing and business domains.

Usage:
    python testing/scripts/test_expanded_patterns.py
"""

import sys
import asyncio
from pathlib import Path

# Add app to Python path
sys.path.append(str(Path(__file__).parent.parent.parent))


async def test_business_domain_classification():
    """Test QueryProcessor classification with new business domains."""
    
    print("🧪 Testing Enhanced Business Domain Classification...")
    print("=" * 60)
    
    try:
        from app.core.query_processor import QueryProcessor
        
        query_processor = QueryProcessor()
        
        # Test cases for all 14 business domains
        test_cases = [
            # Manufacturing domains
            ("What is our daily production output?", "production"),
            ("How can we improve quality control?", "quality"),
            ("What are our supplier delivery issues?", "supply_chain"),
            ("How to reduce manufacturing costs?", "cost_management"),
            ("Equipment maintenance schedule optimization", "asset_management"),
            ("Safety incident trend analysis", "safety"),
            ("Customer demand forecasting", "customer"),
            ("Production planning and scheduling", "planning"),
            ("Workforce productivity analysis", "hr"),
            
            # Business domains
            ("What is our monthly sales revenue?", "sales"),
            ("Product feature adoption rates", "product"),
            ("Marketing campaign ROI analysis", "marketing"),
            ("Process efficiency optimization", "operations"),
            ("Budget variance and forecast accuracy", "finance")
        ]
        
        print(f"Testing {len(test_cases)} business domain classifications...")
        
        correct_classifications = 0
        
        for question, expected_domain in test_cases:
            classified_domain = query_processor._classify_business_domain(question)
            
            if classified_domain == expected_domain:
                correct_classifications += 1
                print(f"   ✅ '{question[:40]}...' → {classified_domain}")
            else:
                print(f"   ❌ '{question[:40]}...' → {classified_domain} (expected: {expected_domain})")
        
        accuracy = correct_classifications / len(test_cases)
        print(f"\n📊 Classification Accuracy: {accuracy:.1%} ({correct_classifications}/{len(test_cases)})")
        
        if accuracy >= 0.8:  # 80% accuracy threshold
            print("✅ Business domain classification working well")
            return True
        else:
            print("❌ Classification accuracy below threshold")
            return False
            
    except Exception as e:
        print(f"❌ Domain classification test failed: {e}")
        return False


async def test_pattern_coverage():
    """Test that all 220 patterns are accessible."""
    
    print("\n📚 Testing Pattern Coverage...")
    print("-" * 40)
    
    pattern_dir = Path(__file__).parent.parent.parent / "app" / "data" / "patterns"
    
    # Expected pattern counts by domain
    expected_patterns = {
        # Manufacturing domains (150 total)
        "production_operations.json": 30,
        "quality_management.json": 25,
        "supply_chain_inventory.json": 25,
        "cost_management.json": 20,
        "asset_equipment.json": 15,
        "safety_compliance.json": 10,
        "customer_demand.json": 10,
        "planning_scheduling.json": 10,
        "hr_workforce.json": 5,
        
        # Business domains (70 total)
        "sales_revenue.json": 20,
        "product_management.json": 15,
        "marketing_campaigns.json": 15,
        "operations_efficiency.json": 10,
        "finance_budgeting.json": 10
    }
    
    total_patterns = 0
    
    for filename, expected_count in expected_patterns.items():
        file_path = pattern_dir / filename
        
        if file_path.exists():
            import json
            with open(file_path, 'r') as f:
                patterns = json.load(f)
            
            actual_count = len(patterns)
            total_patterns += actual_count
            
            if actual_count == expected_count:
                print(f"   ✅ {filename}: {actual_count} patterns")
            else:
                print(f"   ❌ {filename}: {actual_count}/{expected_count} patterns")
                return False
        else:
            print(f"   ❌ {filename}: File not found")
            return False
    
    print(f"\n📊 Total Pattern Coverage: {total_patterns}/220")
    
    if total_patterns == 220:
        print("✅ All 220 patterns accessible")
        return True
    else:
        print("❌ Pattern count mismatch")
        return False


async def test_pattern_quality():
    """Test pattern quality and structure."""
    
    print("\n🔍 Testing Pattern Quality...")
    print("-" * 40)
    
    pattern_dir = Path(__file__).parent.parent.parent / "app" / "data" / "patterns"
    
    # Sample patterns from each new business domain
    business_domain_samples = [
        ("sales_revenue.json", "sales"),
        ("product_management.json", "product"),
        ("marketing_campaigns.json", "marketing"),
        ("operations_efficiency.json", "operations"),
        ("finance_budgeting.json", "finance")
    ]
    
    quality_checks_passed = 0
    
    for filename, domain in business_domain_samples:
        file_path = pattern_dir / filename
        
        try:
            import json
            with open(file_path, 'r') as f:
                patterns = json.load(f)
            
            # Test first pattern structure
            if patterns and len(patterns) > 0:
                pattern = patterns[0]
                
                # Check required fields
                required_fields = ["information", "metadata"]
                required_metadata = [
                    "pattern", "user_roles", "business_domain", "timeframe",
                    "complexity", "success_rate", "confidence_indicators",
                    "expected_deliverables", "data_source", "sample_size"
                ]
                
                has_required_fields = all(field in pattern for field in required_fields)
                has_required_metadata = all(field in pattern["metadata"] for field in required_metadata)
                
                # Check domain consistency
                domain_consistent = pattern["metadata"]["business_domain"] == domain
                
                # Check success rate range
                success_rate = pattern["metadata"]["success_rate"]
                valid_success_rate = 0.0 <= success_rate <= 1.0
                
                if has_required_fields and has_required_metadata and domain_consistent and valid_success_rate:
                    quality_checks_passed += 1
                    print(f"   ✅ {filename}: Quality checks passed")
                else:
                    print(f"   ❌ {filename}: Quality issues detected")
                    
        except Exception as e:
            print(f"   ❌ {filename}: Error loading - {e}")
    
    quality_ratio = quality_checks_passed / len(business_domain_samples)
    print(f"\n📊 Pattern Quality: {quality_ratio:.1%} ({quality_checks_passed}/{len(business_domain_samples)})")
    
    if quality_ratio == 1.0:
        print("✅ All business domain patterns meet quality standards")
        return True
    else:
        print("❌ Some patterns have quality issues")
        return False


async def test_query_processing_with_new_domains():
    """Test enhanced QueryProcessor with new business domains."""
    
    print("\n🚀 Testing Enhanced Query Processing...")
    print("-" * 40)
    
    try:
        from app.core.query_processor import QueryProcessor
        
        query_processor = QueryProcessor()
        
        # Test cases for new business domains
        test_questions = [
            {
                "question": "What is our monthly sales revenue performance vs target?",
                "expected_domain": "sales",
                "expected_complexity": "simple"
            },
            {
                "question": "How effective are our marketing campaigns and what's the ROI?",
                "expected_domain": "marketing", 
                "expected_complexity": "moderate"
            },
            {
                "question": "Which product features have the highest adoption rates?",
                "expected_domain": "product",
                "expected_complexity": "moderate"
            },
            {
                "question": "What are our operational efficiency bottlenecks and cost reduction opportunities?",
                "expected_domain": "operations",
                "expected_complexity": "complex"
            },
            {
                "question": "How is our budget variance and what's driving the financial performance?",
                "expected_domain": "finance",
                "expected_complexity": "moderate"
            }
        ]
        
        successful_tests = 0
        
        for test in test_questions:
            user_context = {
                "user_id": "test_user",
                "role": "manager",
                "department": "business"
            }
            
            org_context = {
                "organization_id": "test_org",
                "business_rules": {},
                "data_classification": "standard"
            }
            
            result = await query_processor.process_business_question(
                test["question"],
                user_context,
                org_context
            )
            
            # Verify results
            domain_correct = result["business_domain"] == test["expected_domain"]
            has_pattern_intelligence = "pattern_intelligence" in result
            has_enhanced_metadata = result["processing_metadata"]["processor_version"] == "2.0"
            
            if domain_correct and has_pattern_intelligence and has_enhanced_metadata:
                successful_tests += 1
                print(f"   ✅ '{test['question'][:40]}...'")
                print(f"      Domain: {result['business_domain']}")
                print(f"      Complexity: {result['complexity_indicators']['complexity_level']}")
                print(f"      Confidence: {result['processing_metadata']['confidence_score']:.3f}")
            else:
                print(f"   ❌ '{test['question'][:40]}...'")
                print(f"      Issues: Domain={domain_correct}, Pattern={has_pattern_intelligence}, Enhanced={has_enhanced_metadata}")
        
        success_rate = successful_tests / len(test_questions)
        print(f"\n📊 Enhanced Processing Success Rate: {success_rate:.1%} ({successful_tests}/{len(test_questions)})")
        
        if success_rate >= 0.8:
            print("✅ Enhanced query processing working correctly")
            return True
        else:
            print("❌ Enhanced query processing has issues")
            return False
            
    except Exception as e:
        print(f"❌ Enhanced processing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main test runner."""
    
    print("🚀 Expanded Pattern Library Test Suite")
    print("   Testing 220 patterns across 14 business domains...")
    print("")
    
    # Run all tests
    tests = [
        ("Business Domain Classification", test_business_domain_classification),
        ("Pattern Coverage", test_pattern_coverage),
        ("Pattern Quality", test_pattern_quality),
        ("Enhanced Query Processing", test_query_processing_with_new_domains)
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        try:
            result = await test_func()
            if result:
                passed_tests += 1
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 EXPANDED PATTERN LIBRARY TEST SUMMARY")
    print("=" * 60)
    
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success Rate: {passed_tests/total_tests:.1%}")
    
    if passed_tests == total_tests:
        print("\n🎉 ALL TESTS PASSED!")
        print("\n✅ Expanded Pattern Library Status:")
        print("   ✅ 220 patterns across 14 business domains")
        print("   ✅ Manufacturing domains (150 patterns)")
        print("   ✅ Business domains (70 patterns)")
        print("   ✅ Enhanced QueryProcessor integration")
        print("   ✅ Pattern-aware business intelligence processing")
        
        print("\n🎯 Pattern Coverage Achievement:")
        print("   📊 Sales & Revenue: 20 patterns")
        print("   🛍️  Product Management: 15 patterns")
        print("   📢 Marketing & Campaigns: 15 patterns")
        print("   ⚙️  Operations & Efficiency: 10 patterns")
        print("   💰 Finance & Budgeting: 10 patterns")
        print("   🏭 Manufacturing & Production: 150 patterns")
        
        print("\n🚀 Ready for Advanced Features:")
        print("   1. Multi-domain pattern correlation analysis")
        print("   2. Cross-functional business intelligence")
        print("   3. Comprehensive pattern-guided investigations")
        print("   4. Full business lifecycle coverage")
        
        return 0
    else:
        print(f"\n❌ {total_tests - passed_tests} tests failed!")
        print("   Please review and fix failing components.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)