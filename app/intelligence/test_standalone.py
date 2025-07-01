#!/usr/bin/env python3
"""
Standalone Test Suite for Intelligence Module
Comprehensive testing of all Phase 2: Strategy Planning components.
Tests self-contained module architecture and component integration.
"""

import asyncio
import sys
import traceback
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Any

# Add module to path for testing
sys.path.insert(0, str(Path(__file__).parent))

# Test imports - validates all components are importable
try:
    from config import settings
    from intelligence_logging import setup_logger, log_operation, performance_monitor
    from domain_expert import (
        DomainExpert, BusinessDomain, AnalysisType, BusinessIntent
    )
    from complexity_analyzer import (
        ComplexityAnalyzer, ComplexityLevel, InvestigationMethodology, ComplexityScore
    )
    from business_context import (
        BusinessContextAnalyzer, UserRole, OrganizationalContext, 
        UserProfile, OrganizationalProfile, ContextualStrategy
    )
    from hypothesis_generator import (
        HypothesisGenerator, HypothesisType, Hypothesis, HypothesisSet
    )
    from pattern_recognizer import (
        PatternRecognizer, PatternType, DiscoveredPattern, PatternLibraryUpdate
    )
    
    IMPORT_SUCCESS = True
    import_error = None
except Exception as e:
    IMPORT_SUCCESS = False
    import_error = str(e)
    print(f"Import error details: {e}")
    import traceback
    traceback.print_exc()


class IntelligenceModuleTestSuite:
    """Comprehensive test suite for intelligence module components."""
    
    def __init__(self):
        self.logger = setup_logger("test_suite")
        self.results = {
            "configuration": {"passed": 0, "failed": 0, "details": []},
            "domain_expert": {"passed": 0, "failed": 0, "details": []},
            "complexity_analyzer": {"passed": 0, "failed": 0, "details": []},
            "business_context": {"passed": 0, "failed": 0, "details": []},
            "hypothesis_generator": {"passed": 0, "failed": 0, "details": []},
            "pattern_recognizer": {"passed": 0, "failed": 0, "details": []},
            "integration": {"passed": 0, "failed": 0, "details": []}
        }
        
        # Test data
        self.test_queries = [
            "Why did Line 2 efficiency drop 15% last week?",
            "Show me current inventory levels for raw materials",
            "Analyze Q4 revenue performance vs forecast across product lines",
            "What caused the increase in customer complaints this month?",
            "Optimize production schedule for next quarter considering all constraints",
            "Compare maintenance costs between different equipment types"
        ]
        
        self.investigation_history = [
            {
                "query": "Production efficiency analysis",
                "domain": "production",
                "methodology": "multi_phase_root_cause",
                "success": True,
                "duration_minutes": 25,
                "user_role": "manager",
                "complexity_level": "analytical",
                "timestamp": "2024-01-15T10:30:00Z"
            },
            {
                "query": "Quality defect investigation",
                "domain": "quality",
                "methodology": "systematic_analysis",
                "success": True,
                "duration_minutes": 18,
                "user_role": "engineer",
                "complexity_level": "analytical",
                "timestamp": "2024-01-16T14:20:00Z"
            },
            {
                "query": "Supply chain delay analysis",
                "domain": "supply_chain",
                "methodology": "rapid_response",
                "success": False,
                "duration_minutes": 12,
                "user_role": "analyst",
                "complexity_level": "simple",
                "timestamp": "2024-01-17T09:15:00Z"
            },
            {
                "query": "Cost variance investigation",
                "domain": "cost",
                "methodology": "systematic_analysis",
                "success": True,
                "duration_minutes": 30,
                "user_role": "manager",
                "complexity_level": "computational",
                "timestamp": "2024-01-18T11:45:00Z"
            },
            {
                "query": "Customer satisfaction decline",
                "domain": "customer",
                "methodology": "multi_phase_root_cause",
                "success": True,
                "duration_minutes": 45,
                "user_role": "specialist",
                "complexity_level": "investigative",
                "timestamp": "2024-01-19T15:30:00Z"
            }
        ] * 3  # Multiply to get sufficient data
    
    def run_all_tests(self) -> bool:
        """Run all test suites and return overall success."""
        
        print("🧪 Intelligence Module Test Suite")
        print("=" * 60)
        print(f"📅 Test Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Check imports first
        if not IMPORT_SUCCESS:
            print("❌ CRITICAL: Import test failed")
            print(f"   Error: {import_error}")
            return False
        
        print("✅ Import test passed - All components importable")
        print()
        
        # Run test suites
        test_methods = [
            ("Configuration Test", self.test_configuration),
            ("Domain Expert Test", self.test_domain_expert),
            ("Complexity Analyzer Test", self.test_complexity_analyzer),
            ("Business Context Test", self.test_business_context),
            ("Hypothesis Generator Test", self.test_hypothesis_generator),
            ("Pattern Recognizer Test", self.test_pattern_recognizer),
            ("Integration Test", self.test_integration)
        ]
        
        overall_success = True
        
        for test_name, test_method in test_methods:
            print(f"🔍 Running {test_name}...")
            try:
                success = test_method()
                if success:
                    print(f"✅ {test_name} passed")
                else:
                    print(f"❌ {test_name} failed")
                    overall_success = False
            except Exception as e:
                print(f"💥 {test_name} crashed: {str(e)}")
                traceback.print_exc()
                overall_success = False
            print()
        
        # Print summary
        self.print_test_summary()
        
        return overall_success
    
    def test_configuration(self) -> bool:
        """Test configuration loading and validation."""
        
        try:
            # Test settings loading
            self.assert_true(settings is not None, "Settings object exists")
            self.assert_true(hasattr(settings, 'strategy_planning_timeout'), "Has strategy planning timeout")
            self.assert_true(hasattr(settings, 'domain_confidence_threshold'), "Has domain confidence threshold")
            self.assert_true(hasattr(settings, 'complexity_scoring_weights'), "Has complexity scoring weights")
            
            # Test configuration values
            self.assert_true(settings.strategy_planning_timeout > 0, "Strategy planning timeout is positive")
            self.assert_true(0 <= settings.domain_confidence_threshold <= 1, "Domain confidence threshold is valid")
            self.assert_true(isinstance(settings.complexity_scoring_weights, dict), "Complexity weights is dictionary")
            
            # Test weight normalization
            total_weight = sum(settings.complexity_scoring_weights.values())
            self.assert_true(0.95 <= total_weight <= 1.05, f"Complexity weights sum to ~1.0 (got {total_weight})")
            
            self.results["configuration"]["passed"] += 4
            return True
            
        except Exception as e:
            self.results["configuration"]["failed"] += 1
            self.results["configuration"]["details"].append(f"Configuration test failed: {e}")
            return False
    
    def test_domain_expert(self) -> bool:
        """Test domain expert classification functionality."""
        
        try:
            expert = DomainExpert()
            success_count = 0
            
            # Test query classification
            for query in self.test_queries:
                intent = expert.classify_business_intent(query)
                
                # Validate intent structure
                self.assert_true(isinstance(intent, BusinessIntent), f"Intent is BusinessIntent for: {query[:30]}")
                self.assert_true(isinstance(intent.primary_domain, BusinessDomain), "Primary domain is BusinessDomain")
                self.assert_true(isinstance(intent.analysis_type, AnalysisType), "Analysis type is AnalysisType")
                self.assert_true(0 <= intent.confidence <= 1, f"Confidence in valid range: {intent.confidence}")
                self.assert_true(isinstance(intent.key_indicators, list), "Key indicators is list")
                
                success_count += 1
            
            # Test specific domain classifications
            production_query = "Why did Line 2 efficiency drop 15% last week?"
            production_intent = expert.classify_business_intent(production_query)
            self.assert_true(
                production_intent.primary_domain == BusinessDomain.PRODUCTION,
                f"Production query classified correctly: {production_intent.primary_domain}"
            )
            
            quality_query = "What caused the increase in customer complaints?"
            quality_intent = expert.classify_business_intent(quality_query)
            self.assert_true(
                quality_intent.primary_domain in [BusinessDomain.QUALITY, BusinessDomain.CUSTOMER],
                f"Quality/Customer query classified appropriately: {quality_intent.primary_domain}"
            )
            
            # Test validation
            self.assert_true(expert.validate_business_intent(production_intent), "High-confidence intent validates")
            
            self.results["domain_expert"]["passed"] += success_count + 3
            return True
            
        except Exception as e:
            self.results["domain_expert"]["failed"] += 1
            self.results["domain_expert"]["details"].append(f"Domain expert test failed: {e}")
            return False
    
    def test_complexity_analyzer(self) -> bool:
        """Test complexity analysis functionality."""
        
        try:
            analyzer = ComplexityAnalyzer()
            expert = DomainExpert()
            success_count = 0
            
            # Test complexity analysis for different query types
            test_cases = [
                ("Show current production status", ComplexityLevel.SIMPLE),
                ("Compare this month's efficiency vs last month", ComplexityLevel.ANALYTICAL),
                ("Why did Line 2 efficiency drop 15% and what's the root cause?", ComplexityLevel.INVESTIGATIVE)
            ]
            
            for query, expected_complexity in test_cases:
                intent = expert.classify_business_intent(query)
                complexity = analyzer.analyze_complexity(intent, query)
                
                # Validate complexity structure
                self.assert_true(isinstance(complexity, ComplexityScore), f"Complexity is ComplexityScore for: {query[:30]}")
                self.assert_true(isinstance(complexity.level, ComplexityLevel), "Complexity level is ComplexityLevel")
                self.assert_true(isinstance(complexity.methodology, InvestigationMethodology), "Methodology is InvestigationMethodology")
                self.assert_true(0 <= complexity.score <= 1, f"Complexity score in valid range: {complexity.score}")
                self.assert_true(complexity.estimated_duration_minutes > 0, "Duration estimate is positive")
                self.assert_true(complexity.estimated_queries > 0, "Query estimate is positive")
                self.assert_true(complexity.estimated_services > 0, "Service estimate is positive")
                
                success_count += 1
            
            # Test dimension scoring
            simple_query = "Show current status"
            simple_intent = expert.classify_business_intent(simple_query)
            simple_complexity = analyzer.analyze_complexity(simple_intent, simple_query)
            
            complex_query = "Analyze quarterly revenue trends across product lines with statistical significance"
            complex_intent = expert.classify_business_intent(complex_query)
            complex_complexity = analyzer.analyze_complexity(complex_intent, complex_query)
            
            self.assert_true(
                simple_complexity.score <= complex_complexity.score,
                f"Complex query has higher complexity score: {simple_complexity.score} <= {complex_complexity.score}"
            )
            
            self.results["complexity_analyzer"]["passed"] += success_count + 1
            return True
            
        except Exception as e:
            self.results["complexity_analyzer"]["failed"] += 1
            self.results["complexity_analyzer"]["details"].append(f"Complexity analyzer test failed: {e}")
            return False
    
    def test_business_context(self) -> bool:
        """Test business context analysis functionality."""
        
        try:
            context_analyzer = BusinessContextAnalyzer()
            expert = DomainExpert()
            complexity_analyzer = ComplexityAnalyzer()
            success_count = 0
            
            # Create test user and org profiles
            test_user = UserProfile(
                user_id="test_manager",
                role=UserRole.MANAGER,
                experience_level="intermediate",
                preferred_detail_level="summary",
                preferred_speed="fast",
                domain_expertise=[BusinessDomain.PRODUCTION, BusinessDomain.QUALITY],
                investigation_history={"rapid_response": 5, "systematic_analysis": 3},
                success_rate=0.85,
                last_activity=datetime.now(timezone.utc)
            )
            
            test_org = OrganizationalProfile(
                organization_id="test_manufacturing",
                context_type=OrganizationalContext.LEAN_MANUFACTURING,
                primary_domains=[BusinessDomain.PRODUCTION, BusinessDomain.QUALITY],
                investigation_patterns={},
                resource_constraints={"time": "limited", "compute": "standard"},
                methodology_preferences={
                    InvestigationMethodology.RAPID_RESPONSE: 0.9,
                    InvestigationMethodology.SYSTEMATIC_ANALYSIS: 0.7
                },
                time_zone="UTC",
                business_hours={"start_hour": "06:00", "end_hour": "18:00"},
                quality_standards={"confidence_threshold": 0.8, "validation_level": 0.7}
            )
            
            # Register profiles
            context_analyzer.update_user_profile(test_user)
            context_analyzer.update_organizational_profile(test_org)
            
            # Test context analysis
            test_query = "Why did Line 2 efficiency drop 15% last week?"
            business_intent = expert.classify_business_intent(test_query)
            complexity = complexity_analyzer.analyze_complexity(business_intent, test_query)
            
            strategy = context_analyzer.analyze_context(
                business_intent=business_intent,
                complexity_level=complexity.level,
                base_methodology=complexity.methodology,
                user_id="test_manager",
                organization_id="test_manufacturing"
            )
            
            # Validate strategy structure
            self.assert_true(isinstance(strategy, ContextualStrategy), "Strategy is ContextualStrategy")
            self.assert_true(isinstance(strategy.adapted_methodology, InvestigationMethodology), "Adapted methodology is valid")
            self.assert_true(isinstance(strategy.context_adjustments, dict), "Context adjustments is dictionary")
            self.assert_true(isinstance(strategy.estimated_timeline, dict), "Timeline is dictionary")
            self.assert_true(strategy.communication_style in ["concise", "detailed", "technical", "executive", "focused"], 
                           f"Communication style is valid: {strategy.communication_style}")
            
            # Test profile retrieval
            retrieved_user = context_analyzer.get_user_profile("test_manager")
            self.assert_true(retrieved_user.user_id == "test_manager", "User profile retrieved correctly")
            
            retrieved_org = context_analyzer.get_organizational_profile("test_manufacturing")
            self.assert_true(retrieved_org.organization_id == "test_manufacturing", "Org profile retrieved correctly")
            
            self.results["business_context"]["passed"] += 3
            return True
            
        except Exception as e:
            self.results["business_context"]["failed"] += 1
            self.results["business_context"]["details"].append(f"Business context test failed: {e}")
            return False
    
    def test_hypothesis_generator(self) -> bool:
        """Test hypothesis generation functionality."""
        
        try:
            generator = HypothesisGenerator()
            expert = DomainExpert()
            complexity_analyzer = ComplexityAnalyzer()
            context_analyzer = BusinessContextAnalyzer()
            
            # Create test scenario
            test_query = "Why did Line 2 efficiency drop 15% last week?"
            business_intent = expert.classify_business_intent(test_query)
            complexity = complexity_analyzer.analyze_complexity(business_intent, test_query)
            
            # Mock contextual strategy
            contextual_strategy = ContextualStrategy(
                base_methodology=InvestigationMethodology.MULTI_PHASE_ROOT_CAUSE,
                adapted_methodology=InvestigationMethodology.MULTI_PHASE_ROOT_CAUSE,
                context_adjustments={},
                user_preferences={"speed_preference": 0.5},
                organizational_constraints={},
                estimated_timeline={"analysis": 30, "validation": 10, "reporting": 10},
                communication_style="technical",
                deliverable_format="report"
            )
            
            # Generate hypotheses
            hypothesis_set = generator.generate_hypotheses(
                business_intent=business_intent,
                contextual_strategy=contextual_strategy,
                complexity_level=complexity.level,
                investigation_context={"line": "Line 2", "metric": "efficiency"}
            )
            
            # Validate hypothesis set structure
            self.assert_true(isinstance(hypothesis_set, HypothesisSet), "Result is HypothesisSet")
            self.assert_true(len(hypothesis_set.hypotheses) > 0, f"Generated hypotheses: {len(hypothesis_set.hypotheses)}")
            self.assert_true(isinstance(hypothesis_set.investigation_strategy, str), "Investigation strategy is string")
            self.assert_true(len(hypothesis_set.prioritized_sequence) > 0, "Has prioritized sequence")
            self.assert_true(isinstance(hypothesis_set.resource_allocation, dict), "Resource allocation is dictionary")
            
            # Validate individual hypotheses
            for hypothesis in hypothesis_set.hypotheses:
                self.assert_true(isinstance(hypothesis, Hypothesis), "Hypothesis is Hypothesis type")
                self.assert_true(isinstance(hypothesis.type, HypothesisType), "Hypothesis type is HypothesisType")
                self.assert_true(0 <= hypothesis.confidence <= 1, f"Hypothesis confidence in range: {hypothesis.confidence}")
                self.assert_true(1 <= hypothesis.priority <= 5, f"Hypothesis priority in range: {hypothesis.priority}")
                self.assert_true(hypothesis.estimated_effort in ["low", "medium", "high"], 
                               f"Hypothesis effort is valid: {hypothesis.estimated_effort}")
                self.assert_true(len(hypothesis.description) > 0, "Hypothesis has description")
                self.assert_true(len(hypothesis.investigation_approach) > 0, "Hypothesis has investigation approach")
            
            self.results["hypothesis_generator"]["passed"] += 2 + len(hypothesis_set.hypotheses)
            return True
            
        except Exception as e:
            self.results["hypothesis_generator"]["failed"] += 1
            self.results["hypothesis_generator"]["details"].append(f"Hypothesis generator test failed: {e}")
            return False
    
    def test_pattern_recognizer(self) -> bool:
        """Test pattern recognition functionality."""
        
        try:
            recognizer = PatternRecognizer()
            
            # Test pattern analysis with investigation history
            library_update = recognizer.analyze_investigation_patterns(self.investigation_history)
            
            # Validate library update structure
            self.assert_true(isinstance(library_update, PatternLibraryUpdate), "Result is PatternLibraryUpdate")
            self.assert_true(isinstance(library_update.new_patterns, list), "New patterns is list")
            self.assert_true(isinstance(library_update.usage_statistics, dict), "Usage statistics is dictionary")
            self.assert_true(isinstance(library_update.effectiveness_metrics, dict), "Effectiveness metrics is dictionary")
            
            # If patterns were discovered, validate them
            if len(library_update.new_patterns) > 0:
                for pattern in library_update.new_patterns:
                    self.assert_true(isinstance(pattern, DiscoveredPattern), "Pattern is DiscoveredPattern")
                    self.assert_true(isinstance(pattern.type, PatternType), "Pattern type is PatternType")
                    self.assert_true(0 <= pattern.business_value <= 1, f"Business value in range: {pattern.business_value}")
                    self.assert_true(pattern.implementation_complexity in ["low", "medium", "high"], 
                                   f"Implementation complexity is valid: {pattern.implementation_complexity}")
                    self.assert_true(len(pattern.description) > 0, "Pattern has description")
                    self.assert_true(len(pattern.recommended_action) > 0, "Pattern has recommended action")
                    self.assert_true(pattern.evidence.observation_count > 0, "Pattern has observations")
                
                self.results["pattern_recognizer"]["passed"] += 1 + len(library_update.new_patterns)
            else:
                # No patterns discovered is acceptable with limited test data
                self.results["pattern_recognizer"]["passed"] += 1
            
            return True
            
        except Exception as e:
            self.results["pattern_recognizer"]["failed"] += 1
            self.results["pattern_recognizer"]["details"].append(f"Pattern recognizer test failed: {e}")
            return False
    
    def test_integration(self) -> bool:
        """Test end-to-end integration of all components."""
        
        try:
            # Initialize all components
            expert = DomainExpert()
            complexity_analyzer = ComplexityAnalyzer()
            context_analyzer = BusinessContextAnalyzer()
            hypothesis_generator = HypothesisGenerator()
            pattern_recognizer = PatternRecognizer()
            
            # Test complete workflow
            test_query = "Why did Line 2 efficiency drop 15% last week?"
            
            # Step 1: Domain classification
            business_intent = expert.classify_business_intent(test_query)
            self.assert_true(business_intent.confidence > 0.25, "Domain classification has reasonable confidence")
            
            # Step 2: Complexity analysis
            complexity = complexity_analyzer.analyze_complexity(business_intent, test_query)
            self.assert_true(complexity.level in [ComplexityLevel.ANALYTICAL, ComplexityLevel.INVESTIGATIVE], 
                           f"Diagnostic query has appropriate complexity: {complexity.level}")
            
            # Step 3: Context analysis
            strategy = context_analyzer.analyze_context(
                business_intent=business_intent,
                complexity_level=complexity.level,
                base_methodology=complexity.methodology
            )
            self.assert_true(strategy.adapted_methodology in list(InvestigationMethodology), 
                           "Strategy has valid methodology")
            
            # Step 4: Hypothesis generation
            hypothesis_set = hypothesis_generator.generate_hypotheses(
                business_intent=business_intent,
                contextual_strategy=strategy,
                complexity_level=complexity.level
            )
            self.assert_true(len(hypothesis_set.hypotheses) > 0, "Hypotheses generated for diagnostic query")
            
            # Step 5: Pattern analysis
            # Create investigation record
            investigation_record = {
                "query": test_query,
                "domain": business_intent.primary_domain.value,
                "methodology": strategy.adapted_methodology.value,
                "complexity_level": complexity.level.value,
                "success": True,
                "duration_minutes": complexity.estimated_duration_minutes,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            extended_history = self.investigation_history + [investigation_record]
            library_update = pattern_recognizer.analyze_investigation_patterns(extended_history)
            
            # Validate end-to-end flow
            self.assert_true(isinstance(library_update, PatternLibraryUpdate), "Pattern analysis completes successfully")
            
            # Test data consistency
            self.assert_true(
                business_intent.primary_domain == BusinessDomain.PRODUCTION,
                "Production query correctly classified throughout pipeline"
            )
            
            self.assert_true(
                complexity.level in [ComplexityLevel.ANALYTICAL, ComplexityLevel.INVESTIGATIVE],
                "Diagnostic query has appropriate complexity level"
            )
            
            # Test component interaction
            high_priority_hypotheses = [h for h in hypothesis_set.hypotheses if h.priority <= 2]
            self.assert_true(len(high_priority_hypotheses) > 0, "At least one high-priority hypothesis generated")
            
            self.results["integration"]["passed"] += 6
            return True
            
        except Exception as e:
            self.results["integration"]["failed"] += 1
            self.results["integration"]["details"].append(f"Integration test failed: {e}")
            return False
    
    def assert_true(self, condition: bool, message: str):
        """Assert that condition is true."""
        if not condition:
            raise AssertionError(f"Assertion failed: {message}")
    
    def print_test_summary(self):
        """Print comprehensive test summary."""
        
        print("📊 Test Summary")
        print("=" * 60)
        
        total_passed = 0
        total_failed = 0
        
        for component, results in self.results.items():
            passed = results["passed"]
            failed = results["failed"]
            total_passed += passed
            total_failed += failed
            
            status = "✅ PASS" if failed == 0 else "❌ FAIL"
            print(f"{component.replace('_', ' ').title():20} {status:8} ({passed} passed, {failed} failed)")
            
            # Print failure details
            if failed > 0 and results["details"]:
                for detail in results["details"]:
                    print(f"    ⚠️  {detail}")
        
        print("-" * 60)
        success_rate = (total_passed / (total_passed + total_failed)) * 100 if (total_passed + total_failed) > 0 else 0
        print(f"{'Overall':<20} {total_passed} passed, {total_failed} failed ({success_rate:.1f}% success)")
        
        # Final verdict
        if total_failed == 0:
            print("\n🎉 All tests passed! Intelligence Module is ready for integration.")
        else:
            print(f"\n⚠️  {total_failed} test(s) failed. Review and fix issues before integration.")


def main():
    """Main test execution."""
    
    # Create and run test suite
    test_suite = IntelligenceModuleTestSuite()
    success = test_suite.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()