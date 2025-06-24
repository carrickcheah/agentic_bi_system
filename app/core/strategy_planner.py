"""
Strategy Planner - Phase 2 of Five-Phase Workflow

Creates sophisticated investigation strategies based on business intelligence
and domain expertise. Implements autonomous investigation planning with
adaptive methodology selection.

Key Features:
- Business-intelligent investigation planning
- Multi-phase strategy development
- Resource allocation and timeline estimation
- Risk assessment and mitigation planning
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from enum import Enum

from ..utils.logging import logger


class InvestigationComplexity(Enum):
    """Investigation complexity levels."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    COMPREHENSIVE = "comprehensive"


class InvestigationPhase(Enum):
    """Investigation phase types."""
    DISCOVERY = "discovery"
    ANALYSIS = "analysis"
    VALIDATION = "validation"
    SYNTHESIS = "synthesis"


class StrategyPlanner:
    """
    Phase 2: Strategy Planning
    
    Creates comprehensive investigation strategies that balance thoroughness
    with efficiency, adapting to business context and requirements.
    """
    
    def __init__(self):
        self.strategy_templates = self._initialize_strategy_templates()
        self.complexity_mappings = self._initialize_complexity_mappings()
        self.resource_estimates = self._initialize_resource_estimates()
        
    async def create_investigation_plan(
        self,
        semantic_intent: Dict[str, Any],
        user_context: Dict[str, Any],
        organization_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create comprehensive investigation strategy based on business intelligence.
        
        Args:
            semantic_intent: Processed business question intent
            user_context: User information and permissions
            organization_context: Organizational context and business rules
            
        Returns:
            Comprehensive investigation strategy with execution plan
        """
        try:
            logger.info("📋 Creating investigation strategy")
            
            # Determine investigation complexity
            complexity = self._determine_investigation_complexity(
                semantic_intent, user_context, organization_context
            )
            
            # Select investigation methodology
            methodology = semantic_intent.get("business_domain_analysis", {}).get("primary_methodology", "descriptive")
            
            # Create multi-phase plan
            investigation_phases = self._plan_investigation_phases(
                complexity, methodology, semantic_intent
            )
            
            # Estimate resources and timeline
            resource_plan = self._estimate_resources(complexity, investigation_phases)
            
            # Create risk assessment
            risk_assessment = self._assess_investigation_risks(
                semantic_intent, complexity, organization_context
            )
            
            # Generate success criteria
            success_criteria = self._define_success_criteria(
                semantic_intent, user_context, methodology
            )
            
            # Create adaptive checkpoints
            checkpoints = self._create_adaptive_checkpoints(investigation_phases)
            
            investigation_strategy = {
                "strategy_id": f"strat_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                "complexity": complexity.value,
                "methodology": methodology,
                "investigation_phases": investigation_phases,
                "resource_plan": resource_plan,
                "risk_assessment": risk_assessment,
                "success_criteria": success_criteria,
                "adaptive_checkpoints": checkpoints,
                "business_context": {
                    "domain": semantic_intent.get("business_domain"),
                    "urgency": self._assess_urgency(semantic_intent, user_context),
                    "stakeholder_impact": self._assess_stakeholder_impact(semantic_intent),
                    "compliance_requirements": organization_context.get("compliance_requirements", [])
                },
                "execution_metadata": {
                    "created_at": datetime.utcnow().isoformat(),
                    "estimated_duration": resource_plan.get("estimated_duration_minutes"),
                    "confidence_score": self._calculate_strategy_confidence(complexity, methodology),
                    "adaptive_enabled": True
                }
            }
            
            logger.info(f"✅ Strategy created - Complexity: {complexity.value}, Phases: {len(investigation_phases)}")
            return investigation_strategy
            
        except Exception as e:
            logger.error(f"Strategy planning failed: {e}")
            raise
    
    def _determine_investigation_complexity(
        self,
        semantic_intent: Dict[str, Any],
        user_context: Dict[str, Any],
        organization_context: Dict[str, Any]
    ) -> InvestigationComplexity:
        """Determine the complexity level of the investigation."""
        complexity_indicators = semantic_intent.get("complexity_indicators", {}).get("indicators", {})
        
        complexity_factors = [
            complexity_indicators.get("multi_domain", False),
            complexity_indicators.get("temporal_analysis", False),
            complexity_indicators.get("causal_analysis", False),
            complexity_indicators.get("predictive_analysis", False),
            complexity_indicators.get("multi_metric", False),
            complexity_indicators.get("requires_context", False),
            complexity_indicators.get("open_ended", False),
            len(user_context.get("permissions", [])) > 5,  # High permissions suggest complex access
            organization_context.get("data_classification", "standard") == "sensitive"
        ]
        
        complexity_score = sum(complexity_factors) / len(complexity_factors)
        
        if complexity_score < 0.25:
            return InvestigationComplexity.SIMPLE
        elif complexity_score < 0.5:
            return InvestigationComplexity.MODERATE
        elif complexity_score < 0.75:
            return InvestigationComplexity.COMPLEX
        else:
            return InvestigationComplexity.COMPREHENSIVE
    
    def _plan_investigation_phases(
        self,
        complexity: InvestigationComplexity,
        methodology: str,
        semantic_intent: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Plan investigation phases based on complexity and methodology."""
        base_phases = self.strategy_templates.get(complexity, [])
        
        # Customize phases based on methodology
        if methodology == "diagnostic":
            base_phases = self._add_diagnostic_phases(base_phases)
        elif methodology == "predictive":
            base_phases = self._add_predictive_phases(base_phases)
        elif methodology == "temporal":
            base_phases = self._add_temporal_phases(base_phases)
        elif methodology == "comparative":
            base_phases = self._add_comparative_phases(base_phases)
        
        # Add business-specific customizations
        business_domain = semantic_intent.get("business_domain")
        if business_domain in ["sales", "finance"]:
            base_phases = self._add_financial_validation_phases(base_phases)
        elif business_domain == "customer":
            base_phases = self._add_customer_analysis_phases(base_phases)
        
        return base_phases
    
    def _estimate_resources(
        self,
        complexity: InvestigationComplexity,
        investigation_phases: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Estimate resources required for investigation."""
        base_estimates = self.resource_estimates.get(complexity.value, {})
        
        # Calculate phase-specific estimates
        total_duration = sum(phase.get("estimated_duration_minutes", 0) for phase in investigation_phases)
        total_queries = sum(phase.get("estimated_queries", 0) for phase in investigation_phases)
        total_services = len(set(
            service for phase in investigation_phases 
            for service in phase.get("required_services", [])
        ))
        
        return {
            "estimated_duration_minutes": max(total_duration, base_estimates.get("min_duration", 5)),
            "estimated_queries": total_queries,
            "required_services": total_services,
            "complexity_factor": base_estimates.get("complexity_factor", 1.0),
            "parallel_execution_possible": complexity in [InvestigationComplexity.MODERATE, InvestigationComplexity.COMPLEX],
            "resource_requirements": {
                "cpu_intensive": complexity in [InvestigationComplexity.COMPLEX, InvestigationComplexity.COMPREHENSIVE],
                "memory_intensive": len(investigation_phases) > 5,
                "io_intensive": total_queries > 10,
                "cache_beneficial": total_queries > 3
            }
        }
    
    def _assess_investigation_risks(
        self,
        semantic_intent: Dict[str, Any],
        complexity: InvestigationComplexity,
        organization_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess risks associated with the investigation."""
        risks = []
        mitigation_strategies = []
        
        # Complexity-based risks
        if complexity in [InvestigationComplexity.COMPLEX, InvestigationComplexity.COMPREHENSIVE]:
            risks.append("investigation_timeout")
            mitigation_strategies.append("implement_progressive_timeout_extension")
            
            risks.append("resource_exhaustion")
            mitigation_strategies.append("enable_parallel_processing_with_limits")
        
        # Data-based risks
        if semantic_intent.get("complexity_indicators", {}).get("indicators", {}).get("multi_domain", False):
            risks.append("data_inconsistency_across_domains")
            mitigation_strategies.append("implement_cross_domain_validation")
        
        # Permission-based risks
        data_classification = organization_context.get("data_classification", "standard")
        if data_classification == "sensitive":
            risks.append("unauthorized_data_access")
            mitigation_strategies.append("enforce_strict_permission_checking")
        
        # Business-based risks
        business_domain = semantic_intent.get("business_domain")
        if business_domain in ["finance", "hr"]:
            risks.append("regulatory_compliance_violation")
            mitigation_strategies.append("apply_data_masking_and_audit_logging")
        
        return {
            "identified_risks": risks,
            "mitigation_strategies": mitigation_strategies,
            "risk_level": "high" if len(risks) > 3 else "medium" if len(risks) > 1 else "low",
            "monitoring_required": len(risks) > 2,
            "fallback_strategies": self._create_fallback_strategies(complexity, risks)
        }
    
    def _define_success_criteria(
        self,
        semantic_intent: Dict[str, Any],
        user_context: Dict[str, Any],
        methodology: str
    ) -> Dict[str, Any]:
        """Define success criteria for the investigation."""
        base_criteria = {
            "investigation_completion": True,
            "error_free_execution": True,
            "response_time_acceptable": True
        }
        
        # Methodology-specific criteria
        if methodology == "diagnostic":
            base_criteria.update({
                "root_cause_identified": True,
                "confidence_level_adequate": True,
                "actionable_recommendations": True
            })
        elif methodology == "predictive":
            base_criteria.update({
                "prediction_confidence_bounds": True,
                "model_assumptions_validated": True,
                "uncertainty_quantified": True
            })
        elif methodology == "comparative":
            base_criteria.update({
                "fair_baseline_comparison": True,
                "contextual_differences_noted": True,
                "statistical_significance": True
            })
        
        # User role-specific criteria
        user_role = user_context.get("role", "")
        if user_role in ["executive", "director", "vp"]:
            base_criteria.update({
                "strategic_implications_clear": True,
                "executive_summary_provided": True
            })
        elif user_role in ["analyst", "manager"]:
            base_criteria.update({
                "detailed_analysis_provided": True,
                "data_quality_validated": True
            })
        
        return {
            "primary_criteria": base_criteria,
            "quality_metrics": {
                "accuracy_threshold": 0.95,
                "completeness_threshold": 0.90,
                "timeliness_threshold": 0.85
            },
            "business_value_indicators": [
                "actionable_insights_generated",
                "decision_support_provided",
                "organizational_learning_captured"
            ]
        }
    
    def _create_adaptive_checkpoints(
        self,
        investigation_phases: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Create adaptive checkpoints for dynamic strategy adjustment."""
        checkpoints = []
        
        for i, phase in enumerate(investigation_phases):
            if i > 0:  # Add checkpoint after each phase except the first
                checkpoint = {
                    "checkpoint_id": f"checkpoint_{i}",
                    "after_phase": phase.get("phase_name"),
                    "evaluation_criteria": [
                        "phase_success_rate",
                        "data_quality_assessment",
                        "investigation_progress",
                        "resource_utilization"
                    ],
                    "adaptation_options": [
                        "continue_as_planned",
                        "adjust_subsequent_phases",
                        "escalate_complexity",
                        "simplify_approach",
                        "abort_and_fallback"
                    ],
                    "decision_thresholds": {
                        "continue_threshold": 0.7,
                        "adjust_threshold": 0.5,
                        "abort_threshold": 0.3
                    }
                }
                checkpoints.append(checkpoint)
        
        return checkpoints
    
    def _initialize_strategy_templates(self) -> Dict[InvestigationComplexity, List[Dict[str, Any]]]:
        """Initialize investigation strategy templates."""
        return {
            InvestigationComplexity.SIMPLE: [
                {
                    "phase_name": "direct_query",
                    "phase_type": InvestigationPhase.ANALYSIS.value,
                    "estimated_duration_minutes": 2,
                    "estimated_queries": 1,
                    "required_services": ["business_data"],
                    "parallel_execution": False
                }
            ],
            InvestigationComplexity.MODERATE: [
                {
                    "phase_name": "data_discovery",
                    "phase_type": InvestigationPhase.DISCOVERY.value,
                    "estimated_duration_minutes": 3,
                    "estimated_queries": 2,
                    "required_services": ["business_data", "memory"],
                    "parallel_execution": False
                },
                {
                    "phase_name": "analysis_execution",
                    "phase_type": InvestigationPhase.ANALYSIS.value,
                    "estimated_duration_minutes": 5,
                    "estimated_queries": 3,
                    "required_services": ["business_data", "analytics"],
                    "parallel_execution": True
                }
            ],
            InvestigationComplexity.COMPLEX: [
                {
                    "phase_name": "comprehensive_discovery",
                    "phase_type": InvestigationPhase.DISCOVERY.value,
                    "estimated_duration_minutes": 5,
                    "estimated_queries": 4,
                    "required_services": ["business_data", "memory", "vector"],
                    "parallel_execution": False
                },
                {
                    "phase_name": "multi_domain_analysis",
                    "phase_type": InvestigationPhase.ANALYSIS.value,
                    "estimated_duration_minutes": 8,
                    "estimated_queries": 6,
                    "required_services": ["business_data", "analytics", "vector"],
                    "parallel_execution": True
                },
                {
                    "phase_name": "cross_validation",
                    "phase_type": InvestigationPhase.VALIDATION.value,
                    "estimated_duration_minutes": 4,
                    "estimated_queries": 3,
                    "required_services": ["business_data", "memory"],
                    "parallel_execution": False
                }
            ]
        }
    
    def _initialize_complexity_mappings(self) -> Dict[str, InvestigationComplexity]:
        """Initialize complexity mappings for different scenarios."""
        return {
            "simple_descriptive": InvestigationComplexity.SIMPLE,
            "moderate_analytical": InvestigationComplexity.MODERATE,
            "complex_predictive": InvestigationComplexity.COMPLEX,
            "comprehensive_investigative": InvestigationComplexity.COMPREHENSIVE
        }
    
    def _initialize_resource_estimates(self) -> Dict[str, Dict[str, Any]]:
        """Initialize resource estimation templates."""
        return {
            "simple": {
                "min_duration": 2,
                "max_duration": 5,
                "complexity_factor": 1.0,
                "parallel_capable": False
            },
            "moderate": {
                "min_duration": 5,
                "max_duration": 15,
                "complexity_factor": 1.5,
                "parallel_capable": True
            },
            "complex": {
                "min_duration": 15,
                "max_duration": 45,
                "complexity_factor": 2.5,
                "parallel_capable": True
            },
            "comprehensive": {
                "min_duration": 30,
                "max_duration": 120,
                "complexity_factor": 4.0,
                "parallel_capable": True
            }
        }
    
    def _add_diagnostic_phases(self, phases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add diagnostic-specific phases."""
        diagnostic_phase = {
            "phase_name": "root_cause_analysis",
            "phase_type": InvestigationPhase.ANALYSIS.value,
            "estimated_duration_minutes": 6,
            "estimated_queries": 4,
            "required_services": ["business_data", "analytics"],
            "parallel_execution": False
        }
        phases.append(diagnostic_phase)
        return phases
    
    def _add_predictive_phases(self, phases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add predictive-specific phases."""
        predictive_phase = {
            "phase_name": "predictive_modeling",
            "phase_type": InvestigationPhase.ANALYSIS.value,
            "estimated_duration_minutes": 10,
            "estimated_queries": 5,
            "required_services": ["business_data", "analytics", "vector"],
            "parallel_execution": False
        }
        phases.append(predictive_phase)
        return phases
    
    def _add_temporal_phases(self, phases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add temporal analysis phases."""
        temporal_phase = {
            "phase_name": "temporal_trend_analysis",
            "phase_type": InvestigationPhase.ANALYSIS.value,
            "estimated_duration_minutes": 7,
            "estimated_queries": 4,
            "required_services": ["business_data", "analytics"],
            "parallel_execution": True
        }
        phases.append(temporal_phase)
        return phases
    
    def _add_comparative_phases(self, phases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add comparative analysis phases."""
        comparative_phase = {
            "phase_name": "comparative_analysis",
            "phase_type": InvestigationPhase.ANALYSIS.value,
            "estimated_duration_minutes": 5,
            "estimated_queries": 3,
            "required_services": ["business_data", "analytics"],
            "parallel_execution": True
        }
        phases.append(comparative_phase)
        return phases
    
    def _add_financial_validation_phases(self, phases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add financial validation phases."""
        validation_phase = {
            "phase_name": "financial_validation",
            "phase_type": InvestigationPhase.VALIDATION.value,
            "estimated_duration_minutes": 3,
            "estimated_queries": 2,
            "required_services": ["business_data"],
            "parallel_execution": False
        }
        phases.append(validation_phase)
        return phases
    
    def _add_customer_analysis_phases(self, phases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add customer-specific analysis phases."""
        customer_phase = {
            "phase_name": "customer_behavior_analysis",
            "phase_type": InvestigationPhase.ANALYSIS.value,
            "estimated_duration_minutes": 6,
            "estimated_queries": 4,
            "required_services": ["business_data", "vector"],
            "parallel_execution": True
        }
        phases.append(customer_phase)
        return phases
    
    def _assess_urgency(self, semantic_intent: Dict[str, Any], user_context: Dict[str, Any]) -> str:
        """Assess urgency level of investigation."""
        urgent_keywords = ["urgent", "immediately", "asap", "critical", "emergency"]
        question = semantic_intent.get("original_question", "").lower()
        
        if any(keyword in question for keyword in urgent_keywords):
            return "urgent"
        elif user_context.get("role") in ["executive", "director"]:
            return "high"
        else:
            return "standard"
    
    def _assess_stakeholder_impact(self, semantic_intent: Dict[str, Any]) -> str:
        """Assess potential stakeholder impact."""
        high_impact_domains = ["sales", "finance", "customer"]
        business_domain = semantic_intent.get("business_domain", "general")
        
        if business_domain in high_impact_domains:
            return "high"
        elif semantic_intent.get("complexity_indicators", {}).get("indicators", {}).get("multi_domain", False):
            return "high"
        else:
            return "medium"
    
    def _create_fallback_strategies(self, complexity: InvestigationComplexity, risks: List[str]) -> List[str]:
        """Create fallback strategies for risk mitigation."""
        fallback_strategies = []
        
        if "investigation_timeout" in risks:
            fallback_strategies.append("implement_progressive_results_delivery")
        
        if "resource_exhaustion" in risks:
            fallback_strategies.append("enable_graceful_degradation")
        
        if "data_inconsistency_across_domains" in risks:
            fallback_strategies.append("fallback_to_single_domain_analysis")
        
        if complexity in [InvestigationComplexity.COMPLEX, InvestigationComplexity.COMPREHENSIVE]:
            fallback_strategies.append("enable_checkpoint_recovery")
        
        return fallback_strategies
    
    def _calculate_strategy_confidence(self, complexity: InvestigationComplexity, methodology: str) -> float:
        """Calculate confidence in strategy success."""
        base_confidence = {
            InvestigationComplexity.SIMPLE: 0.95,
            InvestigationComplexity.MODERATE: 0.85,
            InvestigationComplexity.COMPLEX: 0.75,
            InvestigationComplexity.COMPREHENSIVE: 0.65
        }.get(complexity, 0.5)
        
        # Adjust based on methodology familiarity
        methodology_confidence = {
            "descriptive": 0.95,
            "diagnostic": 0.85,
            "temporal": 0.90,
            "comparative": 0.88,
            "predictive": 0.70
        }.get(methodology, 0.75)
        
        return (base_confidence + methodology_confidence) / 2