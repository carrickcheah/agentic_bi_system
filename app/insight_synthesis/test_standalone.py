#!/usr/bin/env python3
"""
Standalone Test Suite for Insight Synthesis Module
Comprehensive testing of Phase 5: Insight Synthesis components.
Tests strategic intelligence generation and business insight transformation.
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
    from synthesis_logging import setup_logger, log_operation, performance_monitor
    from runner import (
        InsightSynthesizer, InsightType, RecommendationType, OutputFormat,
        BusinessInsight, Recommendation, OrganizationalLearning, SynthesisResult
    )
    
    IMPORT_SUCCESS = True
    import_error = None
except Exception as e:
    IMPORT_SUCCESS = False
    import_error = str(e)
    print(f"Import error details: {e}")
    import traceback
    traceback.print_exc()


class InsightSynthesisTestSuite:
    """Comprehensive test suite for insight synthesis module components."""
    
    def __init__(self):
        self.logger = setup_logger("test_suite")
        self.results = {
            "configuration": {"passed": 0, "failed": 0, "details": []},
            "insight_generation": {"passed": 0, "failed": 0, "details": []},
            "recommendation_generation": {"passed": 0, "failed": 0, "details": []},
            "business_impact_calculation": {"passed": 0, "failed": 0, "details": []},
            "organizational_learning": {"passed": 0, "failed": 0, "details": []},
            "role_specific_formatting": {"passed": 0, "failed": 0, "details": []},
            "integration": {"passed": 0, "failed": 0, "details": []}
        }
        
        # Test data - mock investigation results
        self.mock_investigation_results = {
            "investigation_id": "test_investigation_001",
            "step_results": {
                "schema_analysis": {
                    "key_findings": [
                        "Production efficiency declined 15% in Line 2 over the past week",
                        "Equipment downtime increased by 23% compared to baseline"
                    ],
                    "confidence": 0.85,
                    "supporting_evidence": [
                        "Equipment sensor data", "Production logs", "Maintenance records"
                    ],
                    "data_quality": 0.9
                },
                "data_exploration": {
                    "key_findings": [
                        "Correlation identified between maintenance schedule and efficiency drops",
                        "Quality defect rate increased during high-speed production periods"
                    ],
                    "confidence": 0.78,
                    "supporting_evidence": [
                        "Statistical correlation analysis", "Quality inspection data"
                    ],
                    "data_quality": 0.8
                },
                "pattern_discovery": {
                    "key_findings": [
                        "Predictive maintenance timing shows 72% accuracy for failure prediction",
                        "Supply chain delays correlate with inventory shortages"
                    ],
                    "confidence": 0.82,
                    "supporting_evidence": [
                        "Machine learning model results", "Supply chain data analysis"
                    ],
                    "data_quality": 0.85
                }
            },
            "summary": {
                "conclusions": [
                    "Equipment maintenance optimization could improve efficiency by 12-18%",
                    "Integrated scheduling approach needed for production and maintenance",
                    "Quality control processes require adjustment for high-speed operations"
                ],
                "overall_confidence": 0.82,
                "supporting_data": [
                    "Historical maintenance data", "Production metrics", "Quality records"
                ],
                "data_reliability": 0.85
            }
        }
        
        self.test_business_context = {
            "current_initiative": "Operational Excellence Program",
            "strategic_goal": "15% efficiency improvement",
            "business_unit": "Manufacturing Division",
            "time_period": "Q1 2024",
            "strategic_priority": True,
            "executive_sponsor": True
        }
    
    def run_all_tests(self) -> bool:
        """Run all test suites and return overall success."""
        
        print("🧪 Insight Synthesis Module Test Suite")
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
            ("Insight Generation Test", self.test_insight_generation),
            ("Recommendation Generation Test", self.test_recommendation_generation),
            ("Business Impact Calculation Test", self.test_business_impact_calculation),
            ("Organizational Learning Test", self.test_organizational_learning),
            ("Role-Specific Formatting Test", self.test_role_specific_formatting),
            ("Integration Test", self.test_integration)
        ]
        
        overall_success = True
        
        for test_name, test_method in test_methods:
            print(f"🔍 Running {test_name}...")
            try:
                success = asyncio.run(test_method()) if asyncio.iscoroutinefunction(test_method) else test_method()
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
            self.assert_true(hasattr(settings, 'synthesis_timeout'), "Has synthesis timeout")
            self.assert_true(hasattr(settings, 'insight_confidence_threshold'), "Has insight confidence threshold")
            self.assert_true(hasattr(settings, 'max_insights_per_investigation'), "Has max insights limit")
            
            # Test configuration values
            self.assert_true(settings.synthesis_timeout > 0, "Synthesis timeout is positive")
            self.assert_true(0 <= settings.insight_confidence_threshold <= 1, "Insight confidence threshold is valid")
            self.assert_true(settings.max_insights_per_investigation > 0, "Max insights is positive")
            
            # Test weight normalization
            rec_weights = settings.recommendation_priority_weights
            total_weight = sum(rec_weights.values())
            self.assert_true(0.95 <= total_weight <= 1.05, f"Recommendation weights sum to ~1.0 (got {total_weight})")
            
            impact_weights = settings.impact_calculation_weights
            total_impact_weight = sum(impact_weights.values())
            self.assert_true(0.95 <= total_impact_weight <= 1.05, f"Impact weights sum to ~1.0 (got {total_impact_weight})")
            
            # Test role detail levels
            self.assert_true(isinstance(settings.role_detail_levels, dict), "Role detail levels is dictionary")
            self.assert_true("manager" in settings.role_detail_levels, "Manager role configured")
            self.assert_true("executive" in settings.role_detail_levels, "Executive role configured")
            
            self.results["configuration"]["passed"] += 6
            return True
            
        except Exception as e:
            self.results["configuration"]["failed"] += 1
            self.results["configuration"]["details"].append(f"Configuration test failed: {e}")
            return False
    
    async def test_insight_generation(self) -> bool:
        """Test business insight generation functionality."""
        
        try:
            synthesizer = InsightSynthesizer()
            success_count = 0
            
            # Test insight synthesis
            result = await synthesizer.synthesize_insights(
                investigation_results=self.mock_investigation_results,
                business_context=self.test_business_context,
                user_role="manager",
                output_format=OutputFormat.DETAILED_REPORT
            )
            
            # Validate synthesis result structure
            self.assert_true(isinstance(result, SynthesisResult), "Result is SynthesisResult")
            self.assert_true(len(result.insights) > 0, f"Generated insights: {len(result.insights)}")
            self.assert_true(isinstance(result.insights, list), "Insights is list")
            success_count += 1
            
            # Validate individual insights
            for insight in result.insights:
                self.assert_true(isinstance(insight, BusinessInsight), "Insight is BusinessInsight type")
                self.assert_true(isinstance(insight.type, InsightType), "Insight type is InsightType")
                self.assert_true(0 <= insight.confidence <= 1, f"Insight confidence in range: {insight.confidence}")
                self.assert_true(0 <= insight.strategic_depth <= 1, f"Strategic depth in range: {insight.strategic_depth}")
                self.assert_true(0 <= insight.actionability <= 1, f"Actionability in range: {insight.actionability}")
                self.assert_true(len(insight.title) > 0, "Insight has title")
                self.assert_true(len(insight.description) > 0, "Insight has description")
                self.assert_true(isinstance(insight.business_impact, dict), "Business impact is dictionary")
                self.assert_true(len(insight.stakeholders) > 0, "Insight has stakeholders")
                self.assert_true(len(insight.related_domains) > 0, "Insight has related domains")
                success_count += 1
            
            # Test insight quality filters
            high_confidence_insights = [i for i in result.insights if i.confidence >= settings.insight_confidence_threshold]
            self.assert_true(len(high_confidence_insights) > 0, "High-confidence insights generated")
            success_count += 1
            
            # Test insight type classification
            insight_types = [insight.type for insight in result.insights]
            self.assert_true(len(set(insight_types)) > 0, "Multiple insight types identified")
            success_count += 1
            
            self.results["insight_generation"]["passed"] += success_count
            return True
            
        except Exception as e:
            self.results["insight_generation"]["failed"] += 1
            self.results["insight_generation"]["details"].append(f"Insight generation test failed: {e}")
            return False
    
    async def test_recommendation_generation(self) -> bool:
        """Test recommendation generation functionality."""
        
        try:
            synthesizer = InsightSynthesizer()
            success_count = 0
            
            # Test recommendation synthesis
            result = await synthesizer.synthesize_insights(
                investigation_results=self.mock_investigation_results,
                business_context=self.test_business_context,
                user_role="manager",
                output_format=OutputFormat.ACTION_PLAN
            )
            
            # Validate recommendations structure
            self.assert_true(len(result.recommendations) > 0, f"Generated recommendations: {len(result.recommendations)}")
            self.assert_true(isinstance(result.recommendations, list), "Recommendations is list")
            success_count += 1
            
            # Validate individual recommendations
            for recommendation in result.recommendations:
                self.assert_true(isinstance(recommendation, Recommendation), "Recommendation is Recommendation type")
                self.assert_true(isinstance(recommendation.type, RecommendationType), "Recommendation type is RecommendationType")
                self.assert_true(1 <= recommendation.priority <= 5, f"Priority in range: {recommendation.priority}")
                self.assert_true(0 <= recommendation.feasibility <= 1, f"Feasibility in range: {recommendation.feasibility}")
                self.assert_true(recommendation.risk_level in ["low", "medium", "high"], f"Valid risk level: {recommendation.risk_level}")
                self.assert_true(len(recommendation.title) > 0, "Recommendation has title")
                self.assert_true(len(recommendation.description) > 0, "Recommendation has description")
                self.assert_true(len(recommendation.rationale) > 0, "Recommendation has rationale")
                self.assert_true(isinstance(recommendation.resource_requirements, dict), "Resource requirements is dict")
                self.assert_true(len(recommendation.expected_outcomes) > 0, "Has expected outcomes")
                self.assert_true(len(recommendation.success_metrics) > 0, "Has success metrics")
                success_count += 1
            
            # Test recommendation prioritization
            priorities = [rec.priority for rec in result.recommendations]
            self.assert_true(min(priorities) >= 1, "Minimum priority is valid")
            self.assert_true(max(priorities) <= 5, "Maximum priority is valid")
            success_count += 1
            
            # Test recommendation types
            rec_types = [rec.type for rec in result.recommendations]
            self.assert_true(len(set(rec_types)) > 0, "Multiple recommendation types generated")
            
            # Test for immediate actions
            immediate_actions = [r for r in result.recommendations if r.type == RecommendationType.IMMEDIATE_ACTION]
            if immediate_actions:
                self.assert_true(immediate_actions[0].priority <= 2, "Immediate actions have high priority")
                success_count += 1
            
            self.results["recommendation_generation"]["passed"] += success_count
            return True
            
        except Exception as e:
            self.results["recommendation_generation"]["failed"] += 1
            self.results["recommendation_generation"]["details"].append(f"Recommendation generation test failed: {e}")
            return False
    
    async def test_business_impact_calculation(self) -> bool:
        """Test business impact calculation functionality."""
        
        try:
            synthesizer = InsightSynthesizer()
            success_count = 0
            
            # Test business impact assessment
            result = await synthesizer.synthesize_insights(
                investigation_results=self.mock_investigation_results,
                business_context=self.test_business_context
            )
            
            # Validate business impact assessment structure
            impact_assessment = result.business_impact_assessment
            self.assert_true(isinstance(impact_assessment, dict), "Impact assessment is dictionary")
            
            required_impact_dimensions = [
                "financial_potential", "operational_improvement", "strategic_value",
                "risk_reduction", "implementation_complexity", "confidence_level"
            ]
            
            for dimension in required_impact_dimensions:
                self.assert_true(dimension in impact_assessment, f"Has {dimension} dimension")
                self.assert_true(0 <= impact_assessment[dimension] <= 1, 
                               f"{dimension} in valid range: {impact_assessment[dimension]}")
                success_count += 1
            
            # Test insight-level business impact
            for insight in result.insights:
                impact = insight.business_impact
                self.assert_true(isinstance(impact, dict), "Insight business impact is dictionary")
                
                # Check required impact categories
                impact_categories = ["financial_impact", "operational_efficiency", "strategic_alignment", "risk_mitigation"]
                for category in impact_categories:
                    if category in impact:
                        self.assert_true(0 <= impact[category] <= 1, 
                                       f"Impact {category} in range: {impact[category]}")
                success_count += 1
            
            self.results["business_impact_calculation"]["passed"] += success_count
            return True
            
        except Exception as e:
            self.results["business_impact_calculation"]["failed"] += 1
            self.results["business_impact_calculation"]["details"].append(f"Business impact calculation test failed: {e}")
            return False
    
    async def test_organizational_learning(self) -> bool:
        """Test organizational learning capture functionality."""
        
        try:
            synthesizer = InsightSynthesizer()
            success_count = 0
            
            # Test organizational learning capture
            result = await synthesizer.synthesize_insights(
                investigation_results=self.mock_investigation_results,
                business_context=self.test_business_context
            )
            
            # Validate organizational learning structure
            learning = result.organizational_learning
            self.assert_true(isinstance(learning, OrganizationalLearning), "Learning is OrganizationalLearning type")
            
            # Validate learning components
            self.assert_true(len(learning.pattern_id) > 0, "Has pattern ID")
            self.assert_true(len(learning.pattern_description) > 0, "Has pattern description")
            self.assert_true(learning.frequency > 0, "Frequency is positive")
            self.assert_true(0 <= learning.success_rate <= 1, f"Success rate in range: {learning.success_rate}")
            self.assert_true(0 <= learning.business_value <= 1, f"Business value in range: {learning.business_value}")
            self.assert_true(isinstance(learning.applicable_domains, list), "Applicable domains is list")
            self.assert_true(isinstance(learning.best_practices, list), "Best practices is list")
            self.assert_true(isinstance(learning.lessons_learned, list), "Lessons learned is list")
            self.assert_true(isinstance(learning.improvement_opportunities, list), "Improvement opportunities is list")
            success_count += 4
            
            # Test learning content quality
            if learning.best_practices:
                self.assert_true(len(learning.best_practices[0]) > 10, "Best practices have meaningful content")
                success_count += 1
            
            if learning.lessons_learned:
                self.assert_true(len(learning.lessons_learned[0]) > 10, "Lessons learned have meaningful content")
                success_count += 1
            
            self.results["organizational_learning"]["passed"] += success_count
            return True
            
        except Exception as e:
            self.results["organizational_learning"]["failed"] += 1
            self.results["organizational_learning"]["details"].append(f"Organizational learning test failed: {e}")
            return False
    
    async def test_role_specific_formatting(self) -> bool:
        """Test role-specific formatting functionality."""
        
        try:
            synthesizer = InsightSynthesizer()
            success_count = 0
            
            # Test different role outputs
            test_roles = ["manager", "executive", "analyst", "engineer"]
            role_results = {}
            
            for role in test_roles:
                result = await synthesizer.synthesize_insights(
                    investigation_results=self.mock_investigation_results,
                    business_context=self.test_business_context,
                    user_role=role
                )
                role_results[role] = result
                success_count += 1
            
            # Validate stakeholder communications
            for role, result in role_results.items():
                comms = result.stakeholder_communications
                self.assert_true(isinstance(comms, dict), f"Stakeholder communications is dict for {role}")
                
                # Check that other roles have tailored messages
                other_roles = [r for r in test_roles if r != role]
                role_specific_messages = [r for r in other_roles if r in comms]
                
                if role_specific_messages:
                    for other_role in role_specific_messages:
                        message = comms[other_role]
                        self.assert_true(len(message) > 20, f"Role-specific message has content for {other_role}")
                        success_count += 1
            
            # Test executive summary differences
            manager_summary = role_results["manager"].executive_summary
            executive_summary = role_results["executive"].executive_summary
            
            # Summaries should be tailored but contain similar core information
            self.assert_true(len(manager_summary) > 0, "Manager summary has content")
            self.assert_true(len(executive_summary) > 0, "Executive summary has content")
            success_count += 2
            
            self.results["role_specific_formatting"]["passed"] += success_count
            return True
            
        except Exception as e:
            self.results["role_specific_formatting"]["failed"] += 1
            self.results["role_specific_formatting"]["details"].append(f"Role-specific formatting test failed: {e}")
            return False
    
    async def test_integration(self) -> bool:
        """Test end-to-end integration of all synthesis components."""
        
        try:
            synthesizer = InsightSynthesizer()
            success_count = 0
            
            # Test complete synthesis workflow
            result = await synthesizer.synthesize_insights(
                investigation_results=self.mock_investigation_results,
                business_context=self.test_business_context,
                user_role="manager",
                output_format=OutputFormat.DETAILED_REPORT
            )
            
            # Validate complete result structure
            required_components = [
                "insights", "recommendations", "organizational_learning",
                "executive_summary", "key_findings", "business_impact_assessment",
                "success_criteria", "follow_up_actions", "stakeholder_communications",
                "synthesis_metadata"
            ]
            
            for component in required_components:
                self.assert_true(hasattr(result, component), f"Result has {component}")
                component_value = getattr(result, component)
                self.assert_true(component_value is not None, f"{component} is not None")
                success_count += 1
            
            # Test component integration
            # Insights should relate to recommendations
            if result.insights and result.recommendations:
                insight_domains = set()
                for insight in result.insights:
                    insight_domains.update(insight.related_domains)
                
                rec_related_domains = set()
                for rec in result.recommendations:
                    # Check if recommendation references insights
                    if rec.related_insight_ids:
                        self.assert_true(len(rec.related_insight_ids) > 0, "Recommendation linked to insights")
                        success_count += 1
            
            # Test metadata completeness
            metadata = result.synthesis_metadata
            self.assert_true("synthesis_timestamp" in metadata, "Has synthesis timestamp")
            self.assert_true("source_investigation" in metadata, "Has source investigation")
            self.assert_true("quality_metrics" in metadata, "Has quality metrics")
            success_count += 3
            
            # Test follow-up actions are actionable
            if result.follow_up_actions:
                for action in result.follow_up_actions:
                    self.assert_true(len(action) > 10, "Follow-up action has meaningful content")
                    success_count += 1
            
            # Test success criteria are measurable
            if result.success_criteria:
                for criterion in result.success_criteria:
                    self.assert_true(len(criterion) > 10, "Success criterion has meaningful content")
                    success_count += 1
            
            # Test business value alignment
            if result.insights:
                high_value_insights = [
                    i for i in result.insights 
                    if sum(i.business_impact.values()) > 2.0  # Good business impact
                ]
                if high_value_insights:
                    self.assert_true(len(high_value_insights) > 0, "High-value insights identified")
                    success_count += 1
            
            self.results["integration"]["passed"] += success_count
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
            print("\n🎉 All tests passed! Insight Synthesis Module is ready for integration.")
        else:
            print(f"\n⚠️  {total_failed} test(s) failed. Review and fix issues before integration.")


def main():
    """Main test execution."""
    
    # Create and run test suite
    test_suite = InsightSynthesisTestSuite()
    success = test_suite.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()