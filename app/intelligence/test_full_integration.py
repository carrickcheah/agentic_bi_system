#!/usr/bin/env python3
"""
Comprehensive Integration Test for Intelligence Module
Tests all components including vector-enhanced versions to ensure proper integration.
"""

import asyncio
import time
from pathlib import Path
import sys
from datetime import datetime

# Color codes for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'


async def test_base_components():
    """Test base Intelligence module components."""
    print(f"\n{BLUE}=== Testing Base Intelligence Components ==={RESET}")
    
    try:
        # Test imports
        from domain_expert import DomainExpert, BusinessIntent, BusinessDomain, AnalysisType
        from complexity_analyzer import ComplexityAnalyzer, ComplexityScore, ComplexityLevel
        from business_context import BusinessContextAnalyzer, ContextualStrategy
        from hypothesis_generator import HypothesisGenerator, HypothesisSet
        from pattern_recognizer import PatternRecognizer, DiscoveredPattern
        
        print(f"{GREEN}✅ All base imports successful{RESET}")
        
        # Test DomainExpert
        print(f"\n{YELLOW}1. Testing DomainExpert...{RESET}")
        expert = DomainExpert()
        query = "Why did sales drop last quarter?"
        intent = expert.classify_business_intent(query)
        
        print(f"   Query: {query}")
        print(f"   Domain: {intent.primary_domain.value}")
        print(f"   Analysis Type: {intent.analysis_type.value}")
        print(f"   Confidence: {intent.confidence:.2%}")
        print(f"   {GREEN}✅ DomainExpert working{RESET}")
        
        # Test ComplexityAnalyzer
        print(f"\n{YELLOW}2. Testing ComplexityAnalyzer...{RESET}")
        analyzer = ComplexityAnalyzer()
        complexity = analyzer.analyze_complexity(intent, query)
        
        print(f"   Complexity Level: {complexity.level.value}")
        print(f"   Score: {complexity.overall_score:.2f}")
        print(f"   Methodology: {complexity.methodology.value}")
        print(f"   Estimated Time: {complexity.estimated_execution_time_seconds}s")
        print(f"   {GREEN}✅ ComplexityAnalyzer working{RESET}")
        
        # Test BusinessContextAnalyzer
        print(f"\n{YELLOW}3. Testing BusinessContextAnalyzer...{RESET}")
        context_analyzer = BusinessContextAnalyzer()
        strategy = context_analyzer.analyze_context(
            business_intent=intent,
            complexity_level=complexity.level,
            base_methodology=complexity.methodology,
            user_id="test_user",
            organization_id="test_org"
        )
        
        print(f"   Adapted Methodology: {strategy.adapted_methodology.value}")
        print(f"   Strategic Adjustments: {len(strategy.strategic_adjustments)}")
        print(f"   Compliance Requirements: {len(strategy.compliance_requirements)}")
        print(f"   {GREEN}✅ BusinessContextAnalyzer working{RESET}")
        
        # Test HypothesisGenerator
        print(f"\n{YELLOW}4. Testing HypothesisGenerator...{RESET}")
        hypothesis_gen = HypothesisGenerator()
        hypotheses = hypothesis_gen.generate_hypotheses(
            business_intent=intent,
            contextual_strategy=strategy,
            complexity_level=complexity.level
        )
        
        print(f"   Primary Hypothesis: {hypotheses.primary_hypothesis.description}")
        print(f"   Secondary Hypotheses: {len(hypotheses.secondary_hypotheses)}")
        print(f"   Exploration Areas: {len(hypotheses.exploration_areas)}")
        print(f"   {GREEN}✅ HypothesisGenerator working{RESET}")
        
        # Test PatternRecognizer
        print(f"\n{YELLOW}5. Testing PatternRecognizer...{RESET}")
        pattern_recognizer = PatternRecognizer()
        patterns = pattern_recognizer.recognize_patterns(
            business_intent=intent,
            contextual_strategy=strategy
        )
        
        print(f"   Domain Patterns: {len(patterns.domain_patterns)}")
        print(f"   Query Patterns: {len(patterns.query_patterns)}")
        print(f"   {GREEN}✅ PatternRecognizer working{RESET}")
        
        return True
        
    except Exception as e:
        print(f"{RED}❌ Base component test failed: {e}{RESET}")
        import traceback
        traceback.print_exc()
        return False


async def test_vector_enhanced_components():
    """Test vector-enhanced Intelligence components."""
    print(f"\n{BLUE}=== Testing Vector-Enhanced Components ==={RESET}")
    
    try:
        # Test imports
        from vector_enhanced_domain_expert import VectorEnhancedDomainExpert
        from vector_enhanced_complexity_analyzer import VectorEnhancedComplexityAnalyzer
        from lancedb_pattern_recognizer import LanceDBPatternRecognizer
        
        print(f"{GREEN}✅ Vector-enhanced imports successful{RESET}")
        
        # Test VectorEnhancedDomainExpert
        print(f"\n{YELLOW}1. Testing VectorEnhancedDomainExpert...{RESET}")
        vector_expert = VectorEnhancedDomainExpert()
        await vector_expert.initialize()
        
        query = "Analyze customer churn patterns in Q2"
        if vector_expert.embedder:
            enhanced_intent = await vector_expert.classify_business_intent_with_vectors(query)
            print(f"   Query: {query}")
            print(f"   Domain: {enhanced_intent.primary_domain.value}")
            print(f"   Base Confidence: {enhanced_intent.base_confidence:.2%}")
            print(f"   Pattern Boost: {enhanced_intent.confidence_boost:.2%}")
            print(f"   Final Confidence: {enhanced_intent.confidence:.2%}")
            print(f"   Similar Queries: {len(enhanced_intent.similar_queries)}")
            print(f"   {GREEN}✅ VectorEnhancedDomainExpert working (with vectors){RESET}")
        else:
            # Test fallback
            intent = vector_expert.domain_expert.classify_business_intent(query)
            print(f"   Query: {query}")
            print(f"   Domain: {intent.primary_domain.value}")
            print(f"   Confidence: {intent.confidence:.2%}")
            print(f"   {YELLOW}⚠️  VectorEnhancedDomainExpert working (fallback mode){RESET}")
        
        await vector_expert.cleanup()
        
        # Test VectorEnhancedComplexityAnalyzer
        print(f"\n{YELLOW}2. Testing VectorEnhancedComplexityAnalyzer...{RESET}")
        vector_analyzer = VectorEnhancedComplexityAnalyzer()
        await vector_analyzer.initialize()
        
        # Create a business intent for testing
        from domain_expert import BusinessIntent, BusinessDomain, AnalysisType
        test_intent = BusinessIntent(
            primary_domain=BusinessDomain.SALES,
            secondary_domains=[BusinessDomain.CUSTOMER],
            analysis_type=AnalysisType.TREND,
            confidence=0.85,
            key_entities=["customer", "churn", "Q2"],
            temporal_scope="quarterly",
            investigation_required=True
        )
        
        if vector_analyzer.embedder:
            enhanced_complexity = await vector_analyzer.analyze_complexity_with_vectors(
                test_intent, query
            )
            print(f"   Base Score: {enhanced_complexity.base_score:.2f}")
            print(f"   Pattern Adjustment: {enhanced_complexity.pattern_based_adjustment:.2%}")
            print(f"   Final Score: {enhanced_complexity.overall_score:.2f}")
            print(f"   Historical Patterns: {len(enhanced_complexity.historical_patterns)}")
            print(f"   {GREEN}✅ VectorEnhancedComplexityAnalyzer working (with vectors){RESET}")
        else:
            complexity = vector_analyzer.complexity_analyzer.analyze_complexity(test_intent, query)
            print(f"   Score: {complexity.overall_score:.2f}")
            print(f"   Level: {complexity.level.value}")
            print(f"   {YELLOW}⚠️  VectorEnhancedComplexityAnalyzer working (fallback mode){RESET}")
        
        await vector_analyzer.cleanup()
        
        # Test LanceDBPatternRecognizer
        print(f"\n{YELLOW}3. Testing LanceDBPatternRecognizer...{RESET}")
        pattern_recognizer = LanceDBPatternRecognizer()
        await pattern_recognizer.initialize()
        
        if pattern_recognizer.embedder:
            # Test pattern analysis
            pattern_analysis = await pattern_recognizer.analyze_cross_module_patterns()
            
            print(f"   Business Process Patterns: {len(pattern_analysis['business_process_patterns'])}")
            print(f"   Domain Correlations: {len(pattern_analysis['domain_correlation_patterns'])}")
            print(f"   Temporal Patterns: {len(pattern_analysis['temporal_patterns'])}")
            print(f"   Performance Trends: {len(pattern_analysis['performance_trend_patterns'])}")
            print(f"   {GREEN}✅ LanceDBPatternRecognizer working (with vectors){RESET}")
        else:
            print(f"   {YELLOW}⚠️  LanceDBPatternRecognizer in fallback mode{RESET}")
        
        await pattern_recognizer.cleanup()
        
        return True
        
    except Exception as e:
        print(f"{RED}❌ Vector-enhanced component test failed: {e}{RESET}")
        import traceback
        traceback.print_exc()
        return False


async def test_integrated_workflow():
    """Test complete integrated workflow."""
    print(f"\n{BLUE}=== Testing Integrated Workflow ==={RESET}")
    
    try:
        from runner import IntelligenceModuleRunner
        
        # Initialize runner
        runner = IntelligenceModuleRunner()
        
        # Test complete workflow
        business_question = "Why are we losing market share to competitors in the premium segment?"
        user_context = {
            "user_id": "exec_001",
            "role": "executive",
            "department": "strategy"
        }
        organization_context = {
            "organization_id": "org_001",
            "industry": "retail",
            "size": "enterprise"
        }
        
        print(f"\n{YELLOW}Running complete intelligence planning...{RESET}")
        print(f"Question: {business_question}")
        
        start_time = time.perf_counter()
        result = await runner.plan_investigation_strategy(
            business_question=business_question,
            user_context=user_context,
            organization_context=organization_context
        )
        execution_time = (time.perf_counter() - start_time) * 1000
        
        print(f"\n{GREEN}✅ Intelligence planning completed in {execution_time:.1f}ms{RESET}")
        
        # Display results
        print(f"\n{YELLOW}Results:{RESET}")
        print(f"1. Business Intent:")
        print(f"   - Domain: {result.business_intent.primary_domain.value}")
        print(f"   - Analysis Type: {result.business_intent.analysis_type.value}")
        print(f"   - Confidence: {result.business_intent.confidence:.2%}")
        
        print(f"\n2. Complexity Analysis:")
        print(f"   - Level: {result.complexity_score.level.value}")
        print(f"   - Score: {result.complexity_score.overall_score:.2f}")
        print(f"   - Estimated Time: {result.complexity_score.estimated_execution_time_seconds}s")
        
        print(f"\n3. Contextual Strategy:")
        print(f"   - Methodology: {result.contextual_strategy.adapted_methodology.value}")
        print(f"   - Adjustments: {len(result.contextual_strategy.strategic_adjustments)}")
        print(f"   - Risk Factors: {len(result.contextual_strategy.risk_factors)}")
        
        print(f"\n4. Hypothesis Set:")
        print(f"   - Primary: {result.hypothesis_set.primary_hypothesis.description}")
        print(f"   - Confidence: {result.hypothesis_set.primary_hypothesis.confidence:.2%}")
        print(f"   - Secondary: {len(result.hypothesis_set.secondary_hypotheses)} hypotheses")
        
        return True
        
    except Exception as e:
        print(f"{RED}❌ Integrated workflow test failed: {e}{RESET}")
        import traceback
        traceback.print_exc()
        return False


async def test_vector_integration():
    """Test vector components integration with base components."""
    print(f"\n{BLUE}=== Testing Vector Integration with Base Components ==={RESET}")
    
    try:
        # Test vector + base workflow
        from vector_enhanced_domain_expert import VectorEnhancedDomainExpert
        from vector_enhanced_complexity_analyzer import VectorEnhancedComplexityAnalyzer
        from business_context import BusinessContextAnalyzer
        from hypothesis_generator import HypothesisGenerator
        
        print(f"\n{YELLOW}Creating integrated pipeline...{RESET}")
        
        # Initialize components
        vector_expert = VectorEnhancedDomainExpert()
        vector_analyzer = VectorEnhancedComplexityAnalyzer()
        context_analyzer = BusinessContextAnalyzer()
        hypothesis_gen = HypothesisGenerator()
        
        await vector_expert.initialize()
        await vector_analyzer.initialize()
        
        # Test query
        query = "How can we improve customer retention in our subscription service?"
        
        # Step 1: Vector-enhanced intent classification
        print(f"\n1. Classifying intent with vectors...")
        if vector_expert.embedder:
            enhanced_intent = await vector_expert.classify_business_intent_with_vectors(query)
            print(f"   ✅ Enhanced classification: {enhanced_intent.primary_domain.value}")
            print(f"   Confidence boost: {enhanced_intent.confidence_boost:.2%}")
            
            # Convert to base intent for compatibility
            base_intent = enhanced_intent.base_intent
        else:
            base_intent = vector_expert.domain_expert.classify_business_intent(query)
            print(f"   ⚠️  Fallback classification: {base_intent.primary_domain.value}")
        
        # Step 2: Vector-enhanced complexity analysis
        print(f"\n2. Analyzing complexity with patterns...")
        if vector_analyzer.embedder:
            enhanced_complexity = await vector_analyzer.analyze_complexity_with_vectors(
                base_intent, query
            )
            print(f"   ✅ Enhanced complexity: {enhanced_complexity.level.value}")
            print(f"   Pattern adjustment: {enhanced_complexity.pattern_based_adjustment:.2%}")
            
            # Extract base complexity for compatibility
            base_complexity = enhanced_complexity
        else:
            base_complexity = vector_analyzer.complexity_analyzer.analyze_complexity(
                base_intent, query
            )
            print(f"   ⚠️  Fallback complexity: {base_complexity.level.value}")
        
        # Step 3: Context analysis (base component)
        print(f"\n3. Analyzing business context...")
        strategy = context_analyzer.analyze_context(
            business_intent=base_intent,
            complexity_level=base_complexity.level,
            base_methodology=base_complexity.methodology,
            user_id="test_user",
            organization_id="test_org"
        )
        print(f"   ✅ Strategy: {strategy.adapted_methodology.value}")
        
        # Step 4: Hypothesis generation (base component)
        print(f"\n4. Generating hypotheses...")
        hypotheses = hypothesis_gen.generate_hypotheses(
            business_intent=base_intent,
            contextual_strategy=strategy,
            complexity_level=base_complexity.level
        )
        print(f"   ✅ Generated {len(hypotheses.secondary_hypotheses) + 1} hypotheses")
        
        # Cleanup
        await vector_expert.cleanup()
        await vector_analyzer.cleanup()
        
        print(f"\n{GREEN}✅ Vector integration with base components successful!{RESET}")
        return True
        
    except Exception as e:
        print(f"{RED}❌ Vector integration test failed: {e}{RESET}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all integration tests."""
    print(f"{BLUE}{'=' * 70}{RESET}")
    print(f"{BLUE}Intelligence Module - Comprehensive Integration Test{RESET}")
    print(f"{BLUE}{'=' * 70}{RESET}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    tests = [
        ("Base Components", test_base_components),
        ("Vector-Enhanced Components", test_vector_enhanced_components),
        ("Integrated Workflow", test_integrated_workflow),
        ("Vector Integration", test_vector_integration)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = await test_func()
        except Exception as e:
            print(f"\n{RED}❌ {test_name} crashed: {e}{RESET}")
            results[test_name] = False
    
    # Summary
    print(f"\n{BLUE}{'=' * 70}{RESET}")
    print(f"{BLUE}TEST SUMMARY{RESET}")
    print(f"{BLUE}{'=' * 70}{RESET}")
    
    total = len(results)
    passed = sum(1 for result in results.values() if result)
    
    for test_name, result in results.items():
        status = f"{GREEN}✅ PASS{RESET}" if result else f"{RED}❌ FAIL{RESET}"
        print(f"{test_name}: {status}")
    
    print(f"\nTotal: {total}, Passed: {passed}, Failed: {total - passed}")
    success_rate = (passed / total * 100) if total > 0 else 0
    print(f"Success Rate: {success_rate:.1f}%")
    
    if success_rate == 100:
        print(f"\n{GREEN}🎉 All Intelligence module components are properly integrated!{RESET}")
    elif success_rate >= 75:
        print(f"\n{YELLOW}⚠️  Most components working. Some features may be limited.{RESET}")
    else:
        print(f"\n{RED}❌ Significant integration issues detected.{RESET}")
    
    return 0 if success_rate == 100 else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)