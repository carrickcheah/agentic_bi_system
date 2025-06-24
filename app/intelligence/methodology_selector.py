"""
Investigation Methodology Selector

Selects appropriate investigation methodology based on business intent and complexity:
- Rapid Response: Simple queries (direct metrics)
- Systematic Analysis: Analytical queries (trends, comparisons)
- Scenario Modeling: Computational queries (what-if scenarios)
- Multi-Phase Root Cause: Investigative queries (why questions)
- Strategic Analysis: Strategic recommendations
"""

from typing import Dict, List, Optional, Any
from enum import Enum

from ..utils.logging import logger
from .domain_expert import BusinessIntent, AnalysisType
from .complexity_analyzer import ComplexityAnalysis, BusinessComplexity


class InvestigationMethodology(Enum):
    """Available investigation methodologies."""
    RAPID_RESPONSE = "rapid_response"
    SYSTEMATIC_ANALYSIS = "systematic_analysis"
    SCENARIO_MODELING = "scenario_modeling"
    MULTI_PHASE_ROOT_CAUSE = "multi_phase_root_cause"
    STRATEGIC_ANALYSIS = "strategic_analysis"


class InvestigationMethodologySelector:
    """
    Selects the most appropriate investigation methodology based on
    business intent analysis and complexity assessment.
    """
    
    def __init__(self):
        self.methodology_mapping = self._initialize_methodology_mapping()
        self.complexity_adjustments = self._initialize_complexity_adjustments()
    
    def _initialize_methodology_mapping(self) -> Dict[str, Dict[str, Any]]:
        """Initialize mapping from analysis types to methodologies."""
        return {
            "descriptive": {
                "default": InvestigationMethodology.RAPID_RESPONSE,
                "complex": InvestigationMethodology.SYSTEMATIC_ANALYSIS,
                "description": "Direct data retrieval and basic metrics"
            },
            "diagnostic": {
                "default": InvestigationMethodology.MULTI_PHASE_ROOT_CAUSE,
                "simple": InvestigationMethodology.SYSTEMATIC_ANALYSIS,
                "description": "Root cause analysis and causal investigation"
            },
            "predictive": {
                "default": InvestigationMethodology.SCENARIO_MODELING,
                "simple": InvestigationMethodology.SYSTEMATIC_ANALYSIS,
                "description": "Forecasting and scenario modeling"
            },
            "prescriptive": {
                "default": InvestigationMethodology.STRATEGIC_ANALYSIS,
                "simple": InvestigationMethodology.SYSTEMATIC_ANALYSIS,
                "description": "Strategic recommendations and action plans"
            }
        }
    
    def _initialize_complexity_adjustments(self) -> Dict[BusinessComplexity, Dict[str, Any]]:
        """Initialize complexity-based methodology adjustments."""
        return {
            BusinessComplexity.SIMPLE: {
                "preferred_methodologies": [
                    InvestigationMethodology.RAPID_RESPONSE,
                    InvestigationMethodology.SYSTEMATIC_ANALYSIS
                ],
                "avoid_methodologies": [
                    InvestigationMethodology.MULTI_PHASE_ROOT_CAUSE,
                    InvestigationMethodology.STRATEGIC_ANALYSIS
                ],
                "max_phases": 2,
                "target_duration_minutes": 5
            },
            BusinessComplexity.ANALYTICAL: {
                "preferred_methodologies": [
                    InvestigationMethodology.SYSTEMATIC_ANALYSIS,
                    InvestigationMethodology.SCENARIO_MODELING
                ],
                "avoid_methodologies": [],
                "max_phases": 4,
                "target_duration_minutes": 12
            },
            BusinessComplexity.COMPUTATIONAL: {
                "preferred_methodologies": [
                    InvestigationMethodology.SCENARIO_MODELING,
                    InvestigationMethodology.SYSTEMATIC_ANALYSIS
                ],
                "avoid_methodologies": [
                    InvestigationMethodology.RAPID_RESPONSE
                ],
                "max_phases": 5,
                "target_duration_minutes": 20
            },
            BusinessComplexity.INVESTIGATIVE: {
                "preferred_methodologies": [
                    InvestigationMethodology.MULTI_PHASE_ROOT_CAUSE,
                    InvestigationMethodology.STRATEGIC_ANALYSIS
                ],
                "avoid_methodologies": [
                    InvestigationMethodology.RAPID_RESPONSE
                ],
                "max_phases": 6,
                "target_duration_minutes": 30
            }
        }
    
    async def select_methodology(
        self,
        business_intent: BusinessIntent,
        complexity_analysis: ComplexityAnalysis,
        user_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Select the most appropriate investigation methodology.
        
        Args:
            business_intent: Business intent analysis results
            complexity_analysis: Query complexity analysis results
            user_context: User role, preferences, time constraints
            
        Returns:
            Selected methodology with configuration and reasoning
        """
        logger.info(f"Selecting methodology for {business_intent.analysis_type.value} "
                   f"analysis with {complexity_analysis.classification.value} complexity")
        
        # Get base methodology from analysis type
        analysis_type_key = business_intent.analysis_type.value
        methodology_info = self.methodology_mapping.get(analysis_type_key, {})
        
        # Get complexity constraints
        complexity_constraints = self.complexity_adjustments[complexity_analysis.classification]
        
        # Select methodology based on complexity
        selected_methodology = self._select_with_complexity_adjustment(
            methodology_info, complexity_constraints, complexity_analysis
        )
        
        # Apply user context adjustments
        if user_context:
            selected_methodology = self._apply_user_context_adjustments(
                selected_methodology, user_context, complexity_constraints
            )
        
        # Generate configuration
        methodology_config = self._generate_methodology_config(
            selected_methodology, business_intent, complexity_analysis, user_context
        )
        
        # Generate reasoning
        reasoning = self._generate_selection_reasoning(
            selected_methodology, business_intent, complexity_analysis, methodology_info
        )
        
        return {
            "methodology": selected_methodology,
            "config": methodology_config,
            "reasoning": reasoning,
            "estimated_duration_minutes": methodology_config["estimated_duration_minutes"],
            "estimated_phases": methodology_config["estimated_phases"],
            "confidence_score": self._calculate_confidence_score(
                selected_methodology, business_intent, complexity_analysis
            )
        }
    
    def _select_with_complexity_adjustment(
        self,
        methodology_info: Dict[str, Any],
        complexity_constraints: Dict[str, Any],
        complexity_analysis: ComplexityAnalysis
    ) -> InvestigationMethodology:
        """Select methodology considering complexity constraints."""
        
        # Get default methodology for analysis type
        default_methodology = methodology_info.get("default", InvestigationMethodology.SYSTEMATIC_ANALYSIS)
        
        # Check if default is in preferred methodologies
        preferred = complexity_constraints["preferred_methodologies"]
        avoided = complexity_constraints["avoid_methodologies"]
        
        if default_methodology in avoided:
            # Use most preferred methodology instead
            return preferred[0]
        elif default_methodology in preferred:
            # Use default if it's preferred
            return default_methodology
        else:
            # Use most preferred methodology
            return preferred[0]
    
    def _apply_user_context_adjustments(
        self,
        methodology: InvestigationMethodology,
        user_context: Dict[str, Any],
        complexity_constraints: Dict[str, Any]
    ) -> InvestigationMethodology:
        """Apply user context considerations to methodology selection."""
        
        user_role = user_context.get("role", "").lower()
        time_constraint = user_context.get("time_constraint", "normal")
        detail_preference = user_context.get("detail_preference", "standard")
        
        # Executive users prefer faster, high-level analysis
        if "executive" in user_role or "ceo" in user_role:
            if methodology == InvestigationMethodology.MULTI_PHASE_ROOT_CAUSE:
                return InvestigationMethodology.SYSTEMATIC_ANALYSIS
            elif methodology == InvestigationMethodology.STRATEGIC_ANALYSIS:
                return InvestigationMethodology.SYSTEMATIC_ANALYSIS
        
        # Time-constrained users prefer faster methodologies
        if time_constraint == "urgent":
            fast_methodologies = [
                InvestigationMethodology.RAPID_RESPONSE,
                InvestigationMethodology.SYSTEMATIC_ANALYSIS
            ]
            if methodology not in fast_methodologies:
                # Find fastest allowed methodology
                for fast_method in fast_methodologies:
                    if fast_method in complexity_constraints["preferred_methodologies"]:
                        return fast_method
        
        # Users who prefer minimal detail
        if detail_preference == "minimal":
            if methodology in [InvestigationMethodology.MULTI_PHASE_ROOT_CAUSE, 
                             InvestigationMethodology.STRATEGIC_ANALYSIS]:
                return InvestigationMethodology.SYSTEMATIC_ANALYSIS
        
        return methodology
    
    def _generate_methodology_config(
        self,
        methodology: InvestigationMethodology,
        business_intent: BusinessIntent,
        complexity_analysis: ComplexityAnalysis,
        user_context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate configuration for the selected methodology."""
        
        base_configs = {
            InvestigationMethodology.RAPID_RESPONSE: {
                "estimated_phases": 1,
                "estimated_duration_minutes": 3,
                "depth": "shallow",
                "validation_required": False,
                "real_time_updates": True
            },
            InvestigationMethodology.SYSTEMATIC_ANALYSIS: {
                "estimated_phases": 3,
                "estimated_duration_minutes": 8,
                "depth": "intermediate",
                "validation_required": True,
                "real_time_updates": True
            },
            InvestigationMethodology.SCENARIO_MODELING: {
                "estimated_phases": 4,
                "estimated_duration_minutes": 15,
                "depth": "deep",
                "validation_required": True,
                "scenario_count": 3,
                "confidence_intervals": True
            },
            InvestigationMethodology.MULTI_PHASE_ROOT_CAUSE: {
                "estimated_phases": 5,
                "estimated_duration_minutes": 20,
                "depth": "comprehensive",
                "validation_required": True,
                "hypothesis_testing": True,
                "cross_domain_validation": True
            },
            InvestigationMethodology.STRATEGIC_ANALYSIS: {
                "estimated_phases": 6,
                "estimated_duration_minutes": 25,
                "depth": "comprehensive",
                "validation_required": True,
                "strategic_recommendations": True,
                "implementation_roadmap": True
            }
        }
        
        config = base_configs[methodology].copy()
        
        # Adjust based on complexity
        complexity_multiplier = {
            BusinessComplexity.SIMPLE: 0.7,
            BusinessComplexity.ANALYTICAL: 1.0,
            BusinessComplexity.COMPUTATIONAL: 1.3,
            BusinessComplexity.INVESTIGATIVE: 1.5
        }
        
        multiplier = complexity_multiplier[complexity_analysis.classification]
        config["estimated_duration_minutes"] = int(config["estimated_duration_minutes"] * multiplier)
        
        # Add business domain specific config
        config["primary_domain"] = business_intent.primary_domain.value
        config["secondary_domains"] = [domain.value for domain in business_intent.secondary_domains]
        config["analysis_type"] = business_intent.analysis_type.value
        
        # Add user context specific config
        if user_context:
            config["user_role"] = user_context.get("role")
            config["personalization"] = {
                "detail_level": user_context.get("detail_preference", "standard"),
                "communication_style": user_context.get("communication_style", "professional"),
                "technical_depth": user_context.get("technical_depth", "moderate")
            }
        
        return config
    
    def _generate_selection_reasoning(
        self,
        methodology: InvestigationMethodology,
        business_intent: BusinessIntent,
        complexity_analysis: ComplexityAnalysis,
        methodology_info: Dict[str, Any]
    ) -> str:
        """Generate human-readable reasoning for methodology selection."""
        
        reasons = []
        
        # Primary reason based on analysis type
        analysis_type = business_intent.analysis_type.value
        reasons.append(f"Selected {methodology.value} methodology for {analysis_type} analysis")
        
        # Complexity consideration
        complexity = complexity_analysis.classification.value
        if complexity == "simple":
            reasons.append("optimized for fast response due to simple complexity")
        elif complexity == "investigative":
            reasons.append("comprehensive approach needed for investigative complexity")
        else:
            reasons.append(f"appropriate depth for {complexity} complexity")
        
        # Business domain consideration
        if len(business_intent.secondary_domains) > 0:
            reasons.append(f"spans multiple business domains: {business_intent.primary_domain.value} + {len(business_intent.secondary_domains)} others")
        
        # Confidence consideration
        if business_intent.confidence_score < 0.7:
            reasons.append("includes additional validation due to lower confidence in intent classification")
        
        return "Methodology selection: " + " and ".join(reasons) + "."
    
    def _calculate_confidence_score(
        self,
        methodology: InvestigationMethodology,
        business_intent: BusinessIntent,
        complexity_analysis: ComplexityAnalysis
    ) -> float:
        """Calculate confidence score for methodology selection."""
        
        base_confidence = 0.8
        
        # Adjust based on business intent confidence
        intent_adjustment = (business_intent.confidence_score - 0.5) * 0.2
        base_confidence += intent_adjustment
        
        # Adjust based on complexity analysis confidence
        complexity_adjustment = (complexity_analysis.complexity_score - 0.5) * 0.1
        base_confidence += complexity_adjustment
        
        # Boost confidence for well-matched methodologies
        if business_intent.analysis_type.value in self.methodology_mapping:
            type_info = self.methodology_mapping[business_intent.analysis_type.value]
            if methodology == type_info["default"]:
                base_confidence += 0.1
        
        return min(max(base_confidence, 0.0), 1.0)
    
    def get_methodology_description(self, methodology: InvestigationMethodology) -> str:
        """Get human-readable description of a methodology."""
        
        descriptions = {
            InvestigationMethodology.RAPID_RESPONSE: 
                "Fast, direct metric retrieval for simple questions. Optimized for speed and immediate answers.",
            
            InvestigationMethodology.SYSTEMATIC_ANALYSIS:
                "Structured analysis approach for trend analysis, comparisons, and pattern recognition. Balances depth with efficiency.", 
            
            InvestigationMethodology.SCENARIO_MODELING:
                "Advanced modeling approach for what-if scenarios, forecasting, and computational analysis. Includes confidence intervals and multiple scenarios.",
            
            InvestigationMethodology.MULTI_PHASE_ROOT_CAUSE:
                "Comprehensive root cause analysis with hypothesis testing, cross-domain validation, and evidence-based conclusions.",
            
            InvestigationMethodology.STRATEGIC_ANALYSIS:
                "Executive-level strategic analysis with actionable recommendations, implementation roadmaps, and business impact assessment."
        }
        
        return descriptions.get(methodology, "Advanced business intelligence methodology")