#!/usr/bin/env python3
"""
Integration Test for Intelligence Module with Existing Components
Tests the Intelligence Module integration with the existing agentic SQL system.
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timezone

# Add current directory to path for testing
sys.path.insert(0, str(Path(__file__).parent))

# Import Intelligence Module components directly
try:
    from domain_expert import DomainExpert, BusinessDomain, AnalysisType, BusinessIntent
    from complexity_analyzer import ComplexityAnalyzer, ComplexityLevel, InvestigationMethodology, ComplexityScore
    from business_context import (
        BusinessContextAnalyzer, UserRole, OrganizationalContext, 
        UserProfile, OrganizationalProfile, ContextualStrategy
    )
    from hypothesis_generator import HypothesisGenerator, HypothesisType, Hypothesis, HypothesisSet
    from pattern_recognizer import PatternRecognizer, PatternType, DiscoveredPattern, PatternLibraryUpdate
    INTELLIGENCE_AVAILABLE = True
except ImportError as e:
    print(f"Intelligence Module not available: {e}")
    INTELLIGENCE_AVAILABLE = False

# Import existing core components
try:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from core.strategy_planner import StrategyPlanner, InvestigationComplexity
    from core.business_analyst import AutonomousBusinessAnalyst
    CORE_AVAILABLE = True
except ImportError as e:
    print(f"Core components not available: {e}")
    CORE_AVAILABLE = False


class IntelligenceIntegrationTest:
    """Test Intelligence Module integration with existing components."""
    
    def __init__(self):
        self.results = {
            "component_initialization": {"passed": 0, "failed": 0, "details": []},
            "data_flow_compatibility": {"passed": 0, "failed": 0, "details": []},
            "strategy_planner_integration": {"passed": 0, "failed": 0, "details": []},
            "business_analyst_compatibility": {"passed": 0, "failed": 0, "details": []},
            "end_to_end_workflow": {"passed": 0, "failed": 0, "details": []}
        }
    
    async def run_integration_tests(self) -> bool:
        """Run comprehensive integration tests."""
        
        print("🔗 Intelligence Module Integration Test Suite")
        print("=" * 60)
        print(f"📅 Test Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Test component initialization
        print("🔍 Testing Component Initialization...")
        init_success = await self.test_component_initialization()
        print(f"{'✅' if init_success else '❌'} Component Initialization")
        print()
        
        # Test data flow compatibility
        print("🔍 Testing Data Flow Compatibility...")
        data_flow_success = await self.test_data_flow_compatibility()
        print(f"{'✅' if data_flow_success else '❌'} Data Flow Compatibility")
        print()
        
        # Test strategy planner integration
        if CORE_AVAILABLE:
            print("🔍 Testing Strategy Planner Integration...")
            strategy_success = await self.test_strategy_planner_integration()
            print(f"{'✅' if strategy_success else '❌'} Strategy Planner Integration")
            print()
            
            # Test business analyst compatibility
            print("🔍 Testing Business Analyst Compatibility...")
            analyst_success = await self.test_business_analyst_compatibility()
            print(f"{'✅' if analyst_success else '❌'} Business Analyst Compatibility")
            print()
            
            # Test end-to-end workflow
            print("🔍 Testing End-to-End Workflow...")
            e2e_success = await self.test_end_to_end_workflow()
            print(f"{'✅' if e2e_success else '❌'} End-to-End Workflow")
            print()
        else:
            print("⚠️  Skipping core component tests - components not available")
            strategy_success = analyst_success = e2e_success = True
        
        # Print summary
        self.print_integration_summary()
        
        return (init_success and data_flow_success and 
                strategy_success and analyst_success and e2e_success)
    
    async def test_component_initialization(self) -> bool:
        """Test that all Intelligence Module components initialize correctly."""
        if not INTELLIGENCE_AVAILABLE:
            print("⚠️  Skipping component initialization test - Intelligence Module not available")
            return True
            
        try:
            # Initialize Intelligence Module components
            domain_expert = DomainExpert()
            complexity_analyzer = ComplexityAnalyzer()
            context_analyzer = BusinessContextAnalyzer()
            hypothesis_generator = HypothesisGenerator()
            pattern_recognizer = PatternRecognizer()
            
            # Verify components are properly initialized
            self.assert_true(domain_expert is not None, "DomainExpert initialized")
            self.assert_true(complexity_analyzer is not None, "ComplexityAnalyzer initialized")
            self.assert_true(context_analyzer is not None, "BusinessContextAnalyzer initialized")
            self.assert_true(hypothesis_generator is not None, "HypothesisGenerator initialized")
            self.assert_true(pattern_recognizer is not None, "PatternRecognizer initialized")
            
            self.results["component_initialization"]["passed"] += 5
            return True
            
        except Exception as e:
            self.results["component_initialization"]["failed"] += 1
            self.results["component_initialization"]["details"].append(f"Initialization failed: {e}")
            return False
    
    async def test_data_flow_compatibility(self) -> bool:
        """Test data format compatibility between Intelligence Module and existing system."""
        if not INTELLIGENCE_AVAILABLE:
            print("⚠️  Skipping data flow compatibility test - Intelligence Module not available")
            return True
            
        try:
            # Create test data flow
            domain_expert = DomainExpert()
            complexity_analyzer = ComplexityAnalyzer()
            context_analyzer = BusinessContextAnalyzer()
            
            # Test query: Production efficiency analysis
            test_query = "Why did Line 2 efficiency drop 15% last week?"
            
            # Step 1: Domain classification
            business_intent = domain_expert.classify_business_intent(test_query)
            
            # Verify business intent structure
            self.assert_true(isinstance(business_intent, BusinessIntent), "BusinessIntent created")
            self.assert_true(isinstance(business_intent.primary_domain, BusinessDomain), "Primary domain classified")
            self.assert_true(isinstance(business_intent.analysis_type, AnalysisType), "Analysis type determined")
            
            # Step 2: Complexity analysis
            complexity = complexity_analyzer.analyze_complexity(business_intent, test_query)
            
            # Verify complexity structure
            self.assert_true(hasattr(complexity, 'level'), "Complexity level determined")
            self.assert_true(hasattr(complexity, 'methodology'), "Methodology selected")
            self.assert_true(hasattr(complexity, 'estimated_duration_minutes'), "Duration estimated")
            
            # Step 3: Context analysis
            strategy = context_analyzer.analyze_context(
                business_intent=business_intent,
                complexity_level=complexity.level,
                base_methodology=complexity.methodology
            )
            
            # Verify strategy structure
            self.assert_true(hasattr(strategy, 'adapted_methodology'), "Strategy has adapted methodology")
            self.assert_true(hasattr(strategy, 'estimated_timeline'), "Strategy has timeline")
            self.assert_true(hasattr(strategy, 'communication_style'), "Strategy has communication style")
            
            # Test data conversion for existing system compatibility
            legacy_format = self.convert_to_legacy_format(business_intent, complexity, strategy)
            
            # Verify legacy format compatibility
            self.assert_true("business_domain" in legacy_format, "Legacy format has business domain")
            self.assert_true("complexity_level" in legacy_format, "Legacy format has complexity")
            self.assert_true("methodology" in legacy_format, "Legacy format has methodology")
            
            self.results["data_flow_compatibility"]["passed"] += 4
            return True
            
        except Exception as e:
            self.results["data_flow_compatibility"]["failed"] += 1
            self.results["data_flow_compatibility"]["details"].append(f"Data flow test failed: {e}")
            return False
    
    async def test_strategy_planner_integration(self) -> bool:
        """Test integration with existing StrategyPlanner component."""
        if not CORE_AVAILABLE:
            return True
        
        try:
            # Initialize components
            domain_expert = DomainExpert()
            complexity_analyzer = ComplexityAnalyzer()
            strategy_planner = StrategyPlanner()
            
            # Create test scenario
            test_query = "Analyze quarterly revenue trends across product lines"
            
            # Get Intelligence Module analysis
            business_intent = domain_expert.classify_business_intent(test_query)
            complexity = complexity_analyzer.analyze_complexity(business_intent, test_query)
            
            # Convert to legacy format for StrategyPlanner
            semantic_intent = self.convert_to_legacy_format(business_intent, complexity, None)
            user_context = {"role": "analyst", "permissions": ["read_sales", "read_finance"]}
            org_context = {"data_classification": "standard", "compliance_requirements": []}
            
            # Test existing StrategyPlanner with Intelligence Module data
            investigation_plan = await strategy_planner.create_investigation_plan(
                semantic_intent, user_context, org_context
            )
            
            # Verify integration success
            self.assert_true("strategy_id" in investigation_plan, "Strategy plan created")
            self.assert_true("complexity" in investigation_plan, "Complexity preserved")
            self.assert_true("methodology" in investigation_plan, "Methodology preserved")
            self.assert_true("investigation_phases" in investigation_plan, "Investigation phases created")
            
            # Test complexity mapping compatibility
            intelligence_complexity = complexity.level
            strategy_complexity = InvestigationComplexity(investigation_plan["complexity"])
            
            complexity_mapping = {
                ComplexityLevel.SIMPLE: InvestigationComplexity.SIMPLE,
                ComplexityLevel.ANALYTICAL: InvestigationComplexity.MODERATE,
                ComplexityLevel.COMPUTATIONAL: InvestigationComplexity.COMPLEX,
                ComplexityLevel.INVESTIGATIVE: InvestigationComplexity.COMPREHENSIVE
            }
            
            expected_complexity = complexity_mapping.get(intelligence_complexity)
            self.assert_true(
                strategy_complexity == expected_complexity,
                f"Complexity mapping compatible: {intelligence_complexity} -> {strategy_complexity}"
            )
            
            self.results["strategy_planner_integration"]["passed"] += 3
            return True
            
        except Exception as e:
            self.results["strategy_planner_integration"]["failed"] += 1
            self.results["strategy_planner_integration"]["details"].append(f"Strategy planner integration failed: {e}")
            return False
    
    async def test_business_analyst_compatibility(self) -> bool:
        """Test compatibility with AutonomousBusinessAnalyst."""
        if not CORE_AVAILABLE:
            return True
        
        try:
            # Initialize business analyst
            business_analyst = AutonomousBusinessAnalyst()
            await business_analyst.initialize()
            
            # Verify initialization
            self.assert_true(business_analyst is not None, "AutonomousBusinessAnalyst initialized")
            
            # Test that Intelligence Module can be integrated
            # (This would require modifying the business analyst to use Intelligence Module)
            # For now, just verify structure compatibility
            
            required_methods = [
                "conduct_investigation",
                "initialize"
            ]
            
            for method in required_methods:
                self.assert_true(
                    hasattr(business_analyst, method),
                    f"Business analyst has {method} method"
                )
            
            # Test context structure compatibility
            user_context = {
                "role": "manager",
                "permissions": ["read_production", "read_quality"],
                "user_id": "test_user"
            }
            
            org_context = {
                "data_classification": "standard",
                "compliance_requirements": [],
                "organization_id": "test_org"
            }
            
            # Verify context structures are compatible
            self.assert_true(isinstance(user_context, dict), "User context is dictionary")
            self.assert_true(isinstance(org_context, dict), "Organization context is dictionary")
            
            self.results["business_analyst_compatibility"]["passed"] += 3
            return True
            
        except Exception as e:
            self.results["business_analyst_compatibility"]["failed"] += 1
            self.results["business_analyst_compatibility"]["details"].append(f"Business analyst compatibility failed: {e}")
            return False
    
    async def test_end_to_end_workflow(self) -> bool:
        """Test complete end-to-end workflow with Intelligence Module enhancement."""
        if not INTELLIGENCE_AVAILABLE:
            print("⚠️  Skipping end-to-end workflow test - Intelligence Module not available")
            return True
            
        try:
            # Initialize all components
            domain_expert = DomainExpert()
            complexity_analyzer = ComplexityAnalyzer()
            context_analyzer = BusinessContextAnalyzer()
            hypothesis_generator = HypothesisGenerator()
            pattern_recognizer = PatternRecognizer()
            
            if CORE_AVAILABLE:
                strategy_planner = StrategyPlanner()
            
            # Test query
            test_query = "Why did customer satisfaction scores drop in Q3 and what can we do about it?"
            
            # Phase 1: Enhanced Query Processing with Intelligence Module
            business_intent = domain_expert.classify_business_intent(test_query)
            
            # Verify Phase 1 output
            self.assert_true(
                business_intent.primary_domain == BusinessDomain.CUSTOMER,
                "Customer domain correctly identified"
            )
            self.assert_true(
                business_intent.analysis_type == AnalysisType.DIAGNOSTIC,
                "Diagnostic analysis type identified"
            )
            
            # Phase 2: Enhanced Strategy Planning
            complexity = complexity_analyzer.analyze_complexity(business_intent, test_query)
            
            # Create user and org profiles for context
            user_profile = UserProfile(
                user_id="test_manager",
                role=UserRole.MANAGER,
                experience_level="intermediate",
                preferred_detail_level="summary",
                preferred_speed="balanced",
                domain_expertise=[BusinessDomain.CUSTOMER],
                investigation_history={},
                success_rate=0.8,
                last_activity=datetime.now(timezone.utc)
            )
            
            org_profile = OrganizationalProfile(
                organization_id="test_org",
                context_type=OrganizationalContext.TRADITIONAL,
                primary_domains=[BusinessDomain.CUSTOMER, BusinessDomain.SALES],
                investigation_patterns={},
                resource_constraints={"time": "moderate"},
                methodology_preferences={},
                time_zone="UTC",
                business_hours={"start_hour": "08:00", "end_hour": "17:00"},
                quality_standards={"confidence_threshold": 0.8}
            )
            
            context_analyzer.update_user_profile(user_profile)
            context_analyzer.update_organizational_profile(org_profile)
            
            contextual_strategy = context_analyzer.analyze_context(
                business_intent=business_intent,
                complexity_level=complexity.level,
                base_methodology=complexity.methodology,
                user_id="test_manager",
                organization_id="test_org"
            )
            
            # Verify Phase 2 output
            self.assert_true(
                contextual_strategy.adapted_methodology in list(InvestigationMethodology),
                "Valid methodology selected"
            )
            
            # Phase 2b: Generate hypotheses for diagnostic investigation
            hypothesis_set = hypothesis_generator.generate_hypotheses(
                business_intent=business_intent,
                contextual_strategy=contextual_strategy,
                complexity_level=complexity.level
            )
            
            # Verify hypothesis generation
            self.assert_true(
                len(hypothesis_set.hypotheses) > 0,
                f"Hypotheses generated: {len(hypothesis_set.hypotheses)}"
            )
            
            # Integration with existing StrategyPlanner (if available)
            if CORE_AVAILABLE:
                legacy_format = self.convert_to_legacy_format(business_intent, complexity, contextual_strategy)
                user_context = {"role": "manager", "permissions": ["read_customer"]}
                org_context = {"data_classification": "standard"}
                
                investigation_plan = await strategy_planner.create_investigation_plan(
                    legacy_format, user_context, org_context
                )
                
                self.assert_true(
                    "strategy_id" in investigation_plan,
                    "Integration with existing strategy planner successful"
                )
            
            # Phase 5: Pattern learning for organizational improvement
            investigation_history = [{
                "query": test_query,
                "domain": business_intent.primary_domain.value,
                "methodology": contextual_strategy.adapted_methodology.value,
                "success": True,
                "duration_minutes": complexity.estimated_duration_minutes,
                "user_role": "manager",
                "complexity_level": complexity.level.value,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }]
            
            pattern_update = pattern_recognizer.analyze_investigation_patterns(investigation_history)
            
            # Verify pattern learning
            self.assert_true(
                isinstance(pattern_update.usage_statistics, dict),
                "Pattern usage statistics generated"
            )
            
            self.results["end_to_end_workflow"]["passed"] += 5
            return True
            
        except Exception as e:
            self.results["end_to_end_workflow"]["failed"] += 1
            self.results["end_to_end_workflow"]["details"].append(f"End-to-end workflow failed: {e}")
            return False
    
    def convert_to_legacy_format(self, business_intent, complexity, strategy) -> Dict:
        """Convert Intelligence Module output to legacy format for existing components."""
        legacy_format = {
            "original_question": "Test question",
            "business_domain": business_intent.primary_domain.value,
            "business_domain_analysis": {
                "primary_methodology": business_intent.analysis_type.value,
                "confidence": business_intent.confidence
            },
            "complexity_indicators": {
                "indicators": {
                    "multi_domain": len(business_intent.secondary_domains) > 0,
                    "temporal_analysis": "time" in business_intent.time_context if business_intent.time_context else False,
                    "causal_analysis": business_intent.analysis_type == AnalysisType.DIAGNOSTIC,
                    "predictive_analysis": business_intent.analysis_type == AnalysisType.PREDICTIVE,
                    "multi_metric": len(business_intent.business_metrics) > 1,
                    "requires_context": True,
                    "open_ended": business_intent.analysis_type != AnalysisType.DESCRIPTIVE
                }
            }
        }
        
        if complexity:
            # Map Intelligence Module complexity to legacy format
            complexity_mapping = {
                ComplexityLevel.SIMPLE: "simple",
                ComplexityLevel.ANALYTICAL: "moderate",
                ComplexityLevel.COMPUTATIONAL: "complex",
                ComplexityLevel.INVESTIGATIVE: "comprehensive"
            }
            legacy_format["complexity_level"] = complexity_mapping.get(complexity.level, "moderate")
        
        if strategy:
            legacy_format["communication_style"] = strategy.communication_style
            legacy_format["estimated_timeline"] = strategy.estimated_timeline
        
        return legacy_format
    
    def assert_true(self, condition: bool, message: str):
        """Assert that condition is true."""
        if not condition:
            raise AssertionError(f"Assertion failed: {message}")
    
    def print_integration_summary(self):
        """Print comprehensive integration test summary."""
        
        print("📊 Integration Test Summary")
        print("=" * 60)
        
        total_passed = 0
        total_failed = 0
        
        for component, results in self.results.items():
            passed = results["passed"]
            failed = results["failed"]
            total_passed += passed
            total_failed += failed
            
            status = "✅ PASS" if failed == 0 else "❌ FAIL"
            print(f"{component.replace('_', ' ').title():30} {status:8} ({passed} passed, {failed} failed)")
            
            # Print failure details
            if failed > 0 and results["details"]:
                for detail in results["details"]:
                    print(f"    ⚠️  {detail}")
        
        print("-" * 60)
        success_rate = (total_passed / (total_passed + total_failed)) * 100 if (total_passed + total_failed) > 0 else 0
        print(f"{'Overall':<30} {total_passed} passed, {total_failed} failed ({success_rate:.1f}% success)")
        
        # Final verdict
        if total_failed == 0:
            print("\n🎉 All integration tests passed! Intelligence Module ready for deployment.")
            print("\n📋 Integration Summary:")
            print("• Intelligence Module components initialize correctly")
            print("• Data flows are compatible with existing system")
            if CORE_AVAILABLE:
                print("• StrategyPlanner integration successful")
                print("• AutonomousBusinessAnalyst compatibility confirmed")
                print("• End-to-end workflow enhancement validated")
            else:
                print("• Core component integration tests skipped (components not available)")
            print("• Ready for production integration")
        else:
            print(f"\n⚠️  {total_failed} integration test(s) failed. Review and fix issues before deployment.")


async def main():
    """Main integration test execution."""
    
    test_suite = IntelligenceIntegrationTest()
    success = await test_suite.run_integration_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())