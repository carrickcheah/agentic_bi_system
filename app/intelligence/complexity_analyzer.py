"""
Complexity Analyzer - Investigation Complexity Scoring Component
Self-contained business investigation complexity analysis for manufacturing contexts.
Zero external dependencies beyond module boundary.
"""

from enum import Enum
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import re
import math

try:
    from .config import settings
    from .intelligence_logging import setup_logger, performance_monitor
    from .domain_expert import BusinessDomain, AnalysisType, BusinessIntent
except ImportError:
    from config import settings
    from intelligence_logging import setup_logger, performance_monitor
    from domain_expert import BusinessDomain, AnalysisType, BusinessIntent


class ComplexityLevel(Enum):
    """Investigation complexity levels with time estimates."""
    SIMPLE = "simple"              # 2-5 minutes: Single source, descriptive
    ANALYTICAL = "analytical"      # 5-15 minutes: Multiple sources, comparative
    COMPUTATIONAL = "computational"  # 15-45 minutes: Advanced analytics, modeling
    INVESTIGATIVE = "investigative"  # 30-120 minutes: Multi-domain, predictive


class InvestigationMethodology(Enum):
    """Investigation methodologies aligned with complexity levels."""
    RAPID_RESPONSE = "rapid_response"        # Simple operational queries
    SYSTEMATIC_ANALYSIS = "systematic_analysis"  # Structured multi-step investigations
    SCENARIO_MODELING = "scenario_modeling"   # What-if analysis with variables
    MULTI_PHASE_ROOT_CAUSE = "multi_phase_root_cause"  # Deep diagnostic investigations
    STRATEGIC_ANALYSIS = "strategic_analysis"  # Comprehensive business intelligence


@dataclass
class ComplexityScore:
    """Structured complexity scoring result."""
    level: ComplexityLevel
    methodology: InvestigationMethodology
    score: float  # 0.0 to 1.0
    dimension_scores: Dict[str, float]
    estimated_duration_minutes: int
    estimated_queries: int
    estimated_services: int
    confidence: float
    risk_factors: List[str]
    resource_requirements: Dict[str, Union[str, int]]


@dataclass
class ComplexityDimensions:
    """Complexity analysis dimensions with weights."""
    data_sources: float = 0.0      # Number and variety of data sources
    time_range: float = 0.0        # Historical depth and time span
    analytical_depth: float = 0.0  # Statistical/analytical sophistication
    cross_validation: float = 0.0  # Cross-domain validation requirements
    business_impact: float = 0.0   # Business criticality and scope


class ComplexityAnalyzer:
    """
    Investigation complexity analyzer using multi-dimensional scoring.
    Determines investigation complexity and methodology selection.
    """
    
    def __init__(self):
        self.logger = setup_logger("complexity_analyzer")
        self._complexity_patterns = self._load_complexity_patterns()
        self._methodology_mapping = self._load_methodology_mapping()
        self._duration_estimates = self._load_duration_estimates()
        
        self.logger.info("Complexity Analyzer initialized with scoring models")
    
    def _load_complexity_patterns(self) -> Dict[str, Dict[str, List[str]]]:
        """Load complexity indicator patterns for each dimension."""
        return {
            "data_sources": {
                "simple": [
                    "current", "today", "now", "status", "single", "one",
                    "this", "latest", "recent"
                ],
                "moderate": [
                    "compare", "versus", "against", "between", "multiple",
                    "different", "various", "several"
                ],
                "complex": [
                    "across", "all", "entire", "organization", "company-wide",
                    "comprehensive", "complete", "full", "historical"
                ],
                "advanced": [
                    "integrate", "correlate", "cross-reference", "multi-source",
                    "external", "third-party", "combined", "merged"
                ]
            },
            
            "time_range": {
                "simple": [
                    "today", "now", "current", "this hour", "this shift",
                    "real-time", "instant", "immediate"
                ],
                "moderate": [
                    "this week", "this month", "recent", "last week",
                    "yesterday", "past few", "weekly", "monthly"
                ],
                "complex": [
                    "quarterly", "this quarter", "last quarter", "q1", "q2", "q3", "q4",
                    "year", "annual", "yearly", "seasonal"
                ],
                "advanced": [
                    "historical", "trend", "multi-year", "long-term", "since",
                    "over time", "historical analysis", "time series"
                ]
            },
            
            "analytical_depth": {
                "simple": [
                    "show", "list", "display", "count", "sum", "total",
                    "average", "mean", "simple", "basic"
                ],
                "moderate": [
                    "compare", "analyze", "breakdown", "group by", "segment",
                    "categorize", "variance", "difference"
                ],
                "complex": [
                    "correlation", "pattern", "trend", "forecast", "predict",
                    "model", "regression", "statistical", "significance"
                ],
                "advanced": [
                    "optimize", "simulate", "scenario", "machine learning",
                    "ai", "algorithm", "complex", "advanced analytics"
                ]
            },
            
            "cross_validation": {
                "simple": [
                    "single", "one", "specific", "individual", "isolated"
                ],
                "moderate": [
                    "validate", "check", "verify", "confirm", "cross-check"
                ],
                "complex": [
                    "correlate", "relate", "connect", "link", "associate",
                    "impact", "effect", "influence"
                ],
                "advanced": [
                    "root cause", "comprehensive", "holistic", "systematic",
                    "multi-dimensional", "cross-functional", "enterprise"
                ]
            },
            
            "business_impact": {
                "simple": [
                    "minor", "small", "limited", "local", "departmental"
                ],
                "moderate": [
                    "significant", "important", "division", "plant", "facility"
                ],
                "complex": [
                    "major", "critical", "company", "organization", "strategic"
                ],
                "advanced": [
                    "enterprise", "mission-critical", "business-critical",
                    "transformational", "game-changing", "competitive"
                ]
            }
        }
    
    def _load_methodology_mapping(self) -> Dict[ComplexityLevel, InvestigationMethodology]:
        """Map complexity levels to investigation methodologies."""
        return {
            ComplexityLevel.SIMPLE: InvestigationMethodology.RAPID_RESPONSE,
            ComplexityLevel.ANALYTICAL: InvestigationMethodology.SYSTEMATIC_ANALYSIS,
            ComplexityLevel.COMPUTATIONAL: InvestigationMethodology.SCENARIO_MODELING,
            ComplexityLevel.INVESTIGATIVE: InvestigationMethodology.MULTI_PHASE_ROOT_CAUSE
        }
    
    def _load_duration_estimates(self) -> Dict[ComplexityLevel, Dict[str, Tuple[int, int]]]:
        """Load duration and resource estimates by complexity level."""
        return {
            ComplexityLevel.SIMPLE: {
                "duration_range": (2, 5),      # minutes
                "queries_range": (1, 3),       # number of queries
                "services_range": (1, 2)       # number of services
            },
            ComplexityLevel.ANALYTICAL: {
                "duration_range": (5, 15),     # minutes
                "queries_range": (3, 8),       # number of queries
                "services_range": (2, 3)       # number of services
            },
            ComplexityLevel.COMPUTATIONAL: {
                "duration_range": (15, 45),    # minutes
                "queries_range": (8, 20),      # number of queries
                "services_range": (3, 4)       # number of services
            },
            ComplexityLevel.INVESTIGATIVE: {
                "duration_range": (30, 120),   # minutes
                "queries_range": (15, 50),     # number of queries
                "services_range": (3, 5)       # number of services
            }
        }
    
    @performance_monitor("complexity_analysis")
    def analyze_complexity(self, business_intent: BusinessIntent, query: str) -> ComplexityScore:
        """
        Analyze investigation complexity from business intent and query.
        
        Args:
            business_intent: Classified business intent
            query: Original natural language query
            
        Returns:
            ComplexityScore with detailed analysis
        """
        query_lower = query.lower()
        
        # Calculate dimension scores
        dimensions = self._calculate_dimension_scores(query_lower, business_intent)
        
        # Calculate overall complexity score
        overall_score = self._calculate_overall_score(dimensions)
        
        # Determine complexity level
        complexity_level = self._determine_complexity_level(overall_score, dimensions)
        
        # Select methodology
        methodology = self._select_methodology(complexity_level, business_intent)
        
        # Estimate resources
        duration, queries, services = self._estimate_resources(
            complexity_level, business_intent, dimensions
        )
        
        # Calculate confidence
        confidence = self._calculate_confidence(dimensions, business_intent)
        
        # Identify risk factors
        risk_factors = self._identify_risk_factors(dimensions, business_intent, query_lower)
        
        # Build resource requirements
        resource_requirements = self._build_resource_requirements(
            complexity_level, methodology, duration, queries, services
        )
        
        self.logger.info(
            f"Complexity analyzed: {complexity_level.value} "
            f"({methodology.value}) score: {overall_score:.3f} "
            f"duration: {duration}min"
        )
        
        return ComplexityScore(
            level=complexity_level,
            methodology=methodology,
            score=overall_score,
            dimension_scores={
                "data_sources": dimensions.data_sources,
                "time_range": dimensions.time_range,
                "analytical_depth": dimensions.analytical_depth,
                "cross_validation": dimensions.cross_validation,
                "business_impact": dimensions.business_impact
            },
            estimated_duration_minutes=duration,
            estimated_queries=queries,
            estimated_services=services,
            confidence=confidence,
            risk_factors=risk_factors,
            resource_requirements=resource_requirements
        )
    
    def _calculate_dimension_scores(self, query: str, intent: BusinessIntent) -> ComplexityDimensions:
        """Calculate scores for each complexity dimension."""
        dimensions = ComplexityDimensions()
        
        # Data sources complexity
        dimensions.data_sources = self._score_dimension(query, "data_sources")
        
        # Adjust based on business domain (some domains are inherently more complex)
        domain_multipliers = {
            BusinessDomain.PRODUCTION: 1.0,
            BusinessDomain.QUALITY: 1.1,
            BusinessDomain.SUPPLY_CHAIN: 1.2,
            BusinessDomain.COST: 1.1,
            BusinessDomain.ASSETS: 1.0,
            BusinessDomain.SAFETY: 1.1,
            BusinessDomain.CUSTOMER: 1.2,
            BusinessDomain.PLANNING: 1.3,
            BusinessDomain.HUMAN_RESOURCES: 1.0,
            BusinessDomain.SALES: 1.1,
            BusinessDomain.FINANCE: 1.2,
            BusinessDomain.MARKETING: 1.1,
            BusinessDomain.OPERATIONS: 1.0,
            BusinessDomain.STRATEGIC: 1.4
        }
        
        domain_multiplier = domain_multipliers.get(intent.primary_domain, 1.0)
        dimensions.data_sources = min(dimensions.data_sources * domain_multiplier, 1.0)
        
        # Time range complexity
        dimensions.time_range = self._score_dimension(query, "time_range")
        
        # Adjust based on time context
        if intent.time_context:
            time_multipliers = {
                "today": 0.3,
                "yesterday": 0.4,
                "this_week": 0.5,
                "last_week": 0.6,
                "this_month": 0.7,
                "last_month": 0.8,
                "this_quarter": 0.9,
                "last_quarter": 0.9,
                "this_year": 1.0,
                "last_year": 1.0
            }
            time_multiplier = time_multipliers.get(intent.time_context, 1.0)
            dimensions.time_range = max(dimensions.time_range, time_multiplier)
        
        # Analytical depth complexity
        dimensions.analytical_depth = self._score_dimension(query, "analytical_depth")
        
        # Adjust based on analysis type
        analysis_multipliers = {
            AnalysisType.DESCRIPTIVE: 0.5,
            AnalysisType.DIAGNOSTIC: 0.8,
            AnalysisType.PREDICTIVE: 1.0,
            AnalysisType.PRESCRIPTIVE: 1.2
        }
        analysis_multiplier = analysis_multipliers.get(intent.analysis_type, 0.5)
        dimensions.analytical_depth = max(dimensions.analytical_depth, analysis_multiplier)
        
        # Cross-validation complexity
        dimensions.cross_validation = self._score_dimension(query, "cross_validation")
        
        # Adjust based on secondary domains
        if intent.secondary_domains:
            cross_domain_multiplier = 1.0 + (len(intent.secondary_domains) * 0.2)
            dimensions.cross_validation = min(
                dimensions.cross_validation * cross_domain_multiplier, 1.0
            )
        
        # Business impact complexity
        dimensions.business_impact = self._score_dimension(query, "business_impact")
        
        # Adjust based on urgency
        urgency_multipliers = {
            "low": 0.6,
            "normal": 1.0,
            "high": 1.2,
            "critical": 1.4
        }
        urgency_multiplier = urgency_multipliers.get(intent.urgency_level, 1.0)
        dimensions.business_impact = min(dimensions.business_impact * urgency_multiplier, 1.0)
        
        return dimensions
    
    def _score_dimension(self, query: str, dimension: str) -> float:
        """Score a single complexity dimension."""
        patterns = self._complexity_patterns[dimension]
        scores = []
        
        # Score each complexity level for this dimension
        for level, keywords in patterns.items():
            level_score = sum(1 for keyword in keywords if keyword in query)
            if level_score > 0:
                # Map levels to numeric scores
                level_values = {
                    "simple": 0.25,
                    "moderate": 0.5,
                    "complex": 0.75,
                    "advanced": 1.0
                }
                # Boost scoring to prevent dilution
                normalized_score = min(level_score * 2 / len(keywords), 1.0)
                scores.append(level_values[level] * normalized_score)
        
        if not scores:
            return 0.3  # Default moderate score
        
        # Return highest score for this dimension
        return min(max(scores), 1.0)
    
    def _calculate_overall_score(self, dimensions: ComplexityDimensions) -> float:
        """Calculate weighted overall complexity score."""
        weights = settings.complexity_scoring_weights
        
        overall_score = (
            weights["data_sources"] * dimensions.data_sources +
            weights["time_range"] * dimensions.time_range +
            weights["analytical_depth"] * dimensions.analytical_depth +
            weights["cross_validation"] * dimensions.cross_validation +
            weights["business_impact"] * dimensions.business_impact
        )
        
        return min(overall_score, 1.0)
    
    def _determine_complexity_level(
        self, overall_score: float, dimensions: ComplexityDimensions
    ) -> ComplexityLevel:
        """Determine complexity level from overall score and dimension analysis."""
        
        # Base determination from overall score
        if overall_score >= 0.8:
            base_level = ComplexityLevel.INVESTIGATIVE
        elif overall_score >= 0.6:
            base_level = ComplexityLevel.COMPUTATIONAL
        elif overall_score >= 0.4:
            base_level = ComplexityLevel.ANALYTICAL
        else:
            base_level = ComplexityLevel.SIMPLE
        
        # Adjust based on specific dimension thresholds
        adjustments = []
        
        # High analytical depth pushes towards computational/investigative
        if dimensions.analytical_depth >= 0.8:
            adjustments.append(ComplexityLevel.COMPUTATIONAL)
        
        # High cross-validation requirements push towards investigative
        if dimensions.cross_validation >= 0.7:
            adjustments.append(ComplexityLevel.INVESTIGATIVE)
        
        # Very high business impact pushes towards investigative
        if dimensions.business_impact >= 0.9:
            adjustments.append(ComplexityLevel.INVESTIGATIVE)
        
        # Very low scores across dimensions keep it simple
        max_dimension = max(
            dimensions.data_sources,
            dimensions.time_range,
            dimensions.analytical_depth,
            dimensions.cross_validation,
            dimensions.business_impact
        )
        
        if max_dimension < 0.3:
            return ComplexityLevel.SIMPLE
        
        # Return highest level from base and adjustments
        all_levels = [base_level] + adjustments
        level_values = {
            ComplexityLevel.SIMPLE: 1,
            ComplexityLevel.ANALYTICAL: 2,
            ComplexityLevel.COMPUTATIONAL: 3,
            ComplexityLevel.INVESTIGATIVE: 4
        }
        
        return max(all_levels, key=lambda x: level_values[x])
    
    def _select_methodology(
        self, complexity_level: ComplexityLevel, intent: BusinessIntent
    ) -> InvestigationMethodology:
        """Select investigation methodology based on complexity and intent."""
        
        # Start with default mapping
        methodology = self._methodology_mapping[complexity_level]
        
        # Override based on specific conditions
        
        # Strategic analysis for strategic domain regardless of complexity
        if intent.primary_domain == BusinessDomain.STRATEGIC:
            methodology = InvestigationMethodology.STRATEGIC_ANALYSIS
        
        # Multi-phase root cause for diagnostic analysis with high complexity
        if (intent.analysis_type == AnalysisType.DIAGNOSTIC and 
            complexity_level in [ComplexityLevel.COMPUTATIONAL, ComplexityLevel.INVESTIGATIVE]):
            methodology = InvestigationMethodology.MULTI_PHASE_ROOT_CAUSE
        
        # Scenario modeling for prescriptive analysis
        if intent.analysis_type == AnalysisType.PRESCRIPTIVE:
            methodology = InvestigationMethodology.SCENARIO_MODELING
        
        # Critical urgency may upgrade methodology
        if intent.urgency_level == "critical":
            if methodology == InvestigationMethodology.RAPID_RESPONSE:
                methodology = InvestigationMethodology.SYSTEMATIC_ANALYSIS
        
        return methodology
    
    def _estimate_resources(
        self, 
        complexity_level: ComplexityLevel, 
        intent: BusinessIntent, 
        dimensions: ComplexityDimensions
    ) -> Tuple[int, int, int]:
        """Estimate resource requirements (duration, queries, services)."""
        
        estimates = self._duration_estimates[complexity_level]
        
        # Base estimates
        duration_min, duration_max = estimates["duration_range"]
        queries_min, queries_max = estimates["queries_range"]
        services_min, services_max = estimates["services_range"]
        
        # Calculate position within range based on dimension scores
        avg_dimension_score = (
            dimensions.data_sources + dimensions.time_range + 
            dimensions.analytical_depth + dimensions.cross_validation + 
            dimensions.business_impact
        ) / 5
        
        # Use average dimension score to interpolate within ranges
        duration = int(duration_min + (duration_max - duration_min) * avg_dimension_score)
        queries = int(queries_min + (queries_max - queries_min) * avg_dimension_score)
        services = int(services_min + (services_max - services_min) * avg_dimension_score)
        
        # Adjust for specific conditions
        
        # Secondary domains increase service requirements
        if intent.secondary_domains:
            services += len(intent.secondary_domains)
        
        # High urgency increases parallelization (more services, less duration)
        if intent.urgency_level in ["high", "critical"]:
            services = min(services + 1, 5)
            duration = int(duration * 0.8)  # 20% faster with more resources
        
        # Diagnostic analysis requires more queries
        if intent.analysis_type == AnalysisType.DIAGNOSTIC:
            queries = int(queries * 1.3)
        
        # Clamp to reasonable limits
        duration = max(2, min(duration, 120))
        queries = max(1, min(queries, 50))
        services = max(1, min(services, 5))
        
        return duration, queries, services
    
    def _calculate_confidence(
        self, dimensions: ComplexityDimensions, intent: BusinessIntent
    ) -> float:
        """Calculate confidence in complexity assessment."""
        
        # Base confidence from intent classification confidence
        base_confidence = intent.confidence
        
        # Reduce confidence if dimensions are inconsistent
        dimension_values = [
            dimensions.data_sources,
            dimensions.time_range,
            dimensions.analytical_depth,
            dimensions.cross_validation,
            dimensions.business_impact
        ]
        
        # Calculate standard deviation of dimensions
        mean_dimension = sum(dimension_values) / len(dimension_values)
        variance = sum((x - mean_dimension) ** 2 for x in dimension_values) / len(dimension_values)
        std_dev = math.sqrt(variance)
        
        # High variance reduces confidence
        consistency_factor = max(0.5, 1.0 - std_dev)
        
        # Multiple key indicators increase confidence
        indicator_factor = min(1.0, len(intent.key_indicators) / 5)
        
        final_confidence = base_confidence * consistency_factor * (0.5 + 0.5 * indicator_factor)
        
        return min(final_confidence, 0.95)  # Cap at 95%
    
    def _identify_risk_factors(
        self, 
        dimensions: ComplexityDimensions, 
        intent: BusinessIntent, 
        query: str
    ) -> List[str]:
        """Identify potential risk factors for the investigation."""
        
        risks = []
        
        # High complexity risks
        if dimensions.analytical_depth >= 0.8:
            risks.append("Advanced analytics may require specialized expertise")
        
        if dimensions.cross_validation >= 0.8:
            risks.append("Cross-domain validation may reveal data inconsistencies")
        
        if dimensions.time_range >= 0.8:
            risks.append("Historical data analysis may encounter data quality issues")
        
        if dimensions.data_sources >= 0.8:
            risks.append("Multiple data sources may have integration challenges")
        
        # Business context risks
        if intent.urgency_level == "critical":
            risks.append("Critical urgency may compromise thoroughness")
        
        if intent.confidence < 0.7:
            risks.append("Unclear business intent may lead to investigation drift")
        
        if len(intent.secondary_domains) > 2:
            risks.append("Multiple domains may create conflicting requirements")
        
        # Query-specific risks
        if "optimize" in query or "recommend" in query:
            risks.append("Prescriptive analysis requires validation of recommendations")
        
        if "root cause" in query or "why" in query:
            risks.append("Root cause analysis may require iterative investigation")
        
        if "forecast" in query or "predict" in query:
            risks.append("Predictive analysis accuracy depends on historical data quality")
        
        return risks[:5]  # Limit to top 5 risks
    
    def _build_resource_requirements(
        self,
        complexity_level: ComplexityLevel,
        methodology: InvestigationMethodology,
        duration: int,
        queries: int,
        services: int
    ) -> Dict[str, Union[str, int]]:
        """Build detailed resource requirements dictionary."""
        
        # Service types needed based on complexity
        service_types = {
            ComplexityLevel.SIMPLE: ["primary_database"],
            ComplexityLevel.ANALYTICAL: ["primary_database", "cache_service"],
            ComplexityLevel.COMPUTATIONAL: ["primary_database", "cache_service", "analytics_service"],
            ComplexityLevel.INVESTIGATIVE: ["primary_database", "cache_service", "analytics_service", "pattern_library"]
        }
        
        # Parallel execution capability
        parallel_capability = {
            ComplexityLevel.SIMPLE: "sequential",
            ComplexityLevel.ANALYTICAL: "limited_parallel",
            ComplexityLevel.COMPUTATIONAL: "moderate_parallel",
            ComplexityLevel.INVESTIGATIVE: "high_parallel"
        }
        
        # Memory requirements
        memory_requirements = {
            ComplexityLevel.SIMPLE: "low",
            ComplexityLevel.ANALYTICAL: "moderate",
            ComplexityLevel.COMPUTATIONAL: "high",
            ComplexityLevel.INVESTIGATIVE: "very_high"
        }
        
        return {
            "duration_minutes": duration,
            "estimated_queries": queries,
            "required_services": services,
            "service_types": service_types[complexity_level],
            "parallel_execution": parallel_capability[complexity_level],
            "memory_requirement": memory_requirements[complexity_level],
            "methodology": methodology.value,
            "complexity_level": complexity_level.value
        }
    
    def analyze_query_complexity(self, query: str) -> ComplexityScore:
        """
        Simplified complexity analysis based only on query text.
        Used for true parallel processing without business intent dependency.
        
        Args:
            query: Natural language query
            
        Returns:
            ComplexityScore with basic analysis
        """
        # Simple implementation without performance monitoring
        # Extract basic dimensions from query alone
        dimensions = ComplexityDimensions(
            data_sources=self._estimate_data_sources_from_query(query),
            time_range=self._estimate_time_range_from_query(query),
            analytical_depth=self._estimate_analytical_depth_from_query(query),
            cross_validation=self._estimate_cross_validation_from_query(query),
            business_impact=self._estimate_business_impact_from_query(query)
        )
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(dimensions)
        
        # Determine complexity level and methodology
        level = self._determine_complexity_level(overall_score, dimensions)
        # Simple methodology mapping without business intent
        methodology_map = {
            ComplexityLevel.SIMPLE: InvestigationMethodology.RAPID_RESPONSE,
            ComplexityLevel.ANALYTICAL: InvestigationMethodology.SYSTEMATIC_ANALYSIS,
            ComplexityLevel.COMPUTATIONAL: InvestigationMethodology.SCENARIO_MODELING,
            ComplexityLevel.INVESTIGATIVE: InvestigationMethodology.MULTI_PHASE_ROOT_CAUSE
        }
        methodology = methodology_map.get(level, InvestigationMethodology.SYSTEMATIC_ANALYSIS)
        
        # Simple estimates without business intent
        duration = int(5 + overall_score * 40)  # 5-45 minutes
        queries = int(1 + overall_score * 9)    # 1-10 queries
        services = int(1 + overall_score * 2)    # 1-3 services
        
        return ComplexityScore(
            level=level,
            methodology=methodology,
            score=overall_score,
            dimension_scores={
                "data_sources": dimensions.data_sources,
                "time_range": dimensions.time_range,
                "analytical_depth": dimensions.analytical_depth,
                "cross_validation": dimensions.cross_validation,
                "business_impact": dimensions.business_impact
            },
            estimated_duration_minutes=duration,
            estimated_queries=queries,
            estimated_services=services,
            confidence=0.8,  # Fixed confidence for simplified analysis
            risk_factors=[],
            resource_requirements={
                "computational": "standard",
                "domain_expertise": "general",
                "data_quality": "standard"
            }
        )
    
    def _estimate_data_sources_from_query(self, query: str) -> float:
        """Estimate data source complexity from query text."""
        indicators = ["multiple", "across", "combine", "integrate", "compare"]
        score = sum(0.2 for ind in indicators if ind in query.lower())
        return min(score, 1.0)
    
    def _estimate_time_range_from_query(self, query: str) -> float:
        """Estimate time range complexity from query text."""
        indicators = ["historical", "trend", "over time", "last year", "quarterly"]
        score = sum(0.25 for ind in indicators if ind in query.lower())
        return min(score, 1.0)
    
    def _estimate_analytical_depth_from_query(self, query: str) -> float:
        """Estimate analytical depth from query text."""
        indicators = ["analyze", "predict", "forecast", "optimize", "optimal", "correlation", "maximize", "minimize"]
        score = sum(0.2 for ind in indicators if ind in query.lower())
        return min(score, 1.0)
    
    def _estimate_cross_validation_from_query(self, query: str) -> float:
        """Estimate cross-validation needs from query text."""
        indicators = ["validate", "verify", "confirm", "cross-check", "accuracy"]
        score = sum(0.25 for ind in indicators if ind in query.lower())
        return min(score, 1.0)
    
    def _estimate_business_impact_from_query(self, query: str) -> float:
        """Estimate business impact from query text."""
        indicators = ["strategic", "critical", "urgent", "important", "priority"]
        score = sum(0.25 for ind in indicators if ind in query.lower())
        return min(score, 1.0)
    
    def enhance_complexity_with_intent(
        self, 
        base_complexity: ComplexityScore, 
        business_intent: BusinessIntent
    ) -> ComplexityScore:
        """
        Enhance basic complexity score with business intent context.
        Used after parallel processing to improve accuracy.
        
        Args:
            base_complexity: Initial complexity from query-only analysis
            business_intent: Business context from domain expert
            
        Returns:
            Enhanced ComplexityScore with business-aware adjustments
        """
        # Start with base score
        enhanced_score = base_complexity.score
        
        # Domain-specific complexity multipliers
        domain_multipliers = {
            BusinessDomain.SALES: 1.0,
            BusinessDomain.CUSTOMER: 1.1,        # Customer analysis often complex
            BusinessDomain.SUPPLY_CHAIN: 1.15,   # Supply chain optimization is complex
            BusinessDomain.OPERATIONS: 1.2,      # Operations analysis very complex
            BusinessDomain.FINANCE: 1.25,        # Financial analysis most complex
            BusinessDomain.PRODUCTION: 1.1,      # Production analysis
            BusinessDomain.QUALITY: 1.05,        # Quality metrics
            BusinessDomain.COST: 1.1,            # Cost analysis
            BusinessDomain.STRATEGIC: 1.3        # Strategic planning most complex
        }
        
        # Analysis type multipliers
        analysis_multipliers = {
            AnalysisType.DESCRIPTIVE: 1.0,    # Simple reporting
            AnalysisType.DIAGNOSTIC: 1.1,     # Root cause analysis
            AnalysisType.PREDICTIVE: 1.2,     # Forecasting
            AnalysisType.PRESCRIPTIVE: 1.3    # Optimization
        }
        
        # Apply domain multiplier
        if business_intent.primary_domain in domain_multipliers:
            enhanced_score *= domain_multipliers[business_intent.primary_domain]
        
        # Apply analysis type multiplier
        if business_intent.analysis_type in analysis_multipliers:
            enhanced_score *= analysis_multipliers[business_intent.analysis_type]
        
        # Key indicators-based adjustments (using key_indicators as proxy for entities)
        if len(business_intent.key_indicators) > 3:
            enhanced_score *= 1.1  # Multiple indicators increase complexity
        
        # Confidence adjustment
        if business_intent.confidence < 0.7:
            enhanced_score *= 1.05  # Low confidence needs more investigation
        
        # Cap the score
        enhanced_score = min(enhanced_score, 1.0)
        
        # Recalculate level and methodology based on enhanced score
        # Create a dummy dimensions object from the base complexity
        dimensions = ComplexityDimensions(
            data_sources=base_complexity.dimension_scores.get("data_sources", 0.5),
            time_range=base_complexity.dimension_scores.get("time_range", 0.5),
            analytical_depth=base_complexity.dimension_scores.get("analytical_depth", 0.5),
            cross_validation=base_complexity.dimension_scores.get("cross_validation", 0.5),
            business_impact=base_complexity.dimension_scores.get("business_impact", 0.5)
        )
        
        enhanced_level = self._determine_complexity_level(enhanced_score, dimensions)
        
        # Select methodology based on enhanced understanding
        enhanced_methodology = self._select_methodology_simple(
            enhanced_level, 
            business_intent
        )
        
        # Calculate enhanced estimates
        enhanced_duration = int(10 + enhanced_score * 50)
        enhanced_queries = int(2 + enhanced_score * 12)
        enhanced_services = int(1 + enhanced_score * 3)
        
        # Build enhanced complexity score
        return ComplexityScore(
            level=enhanced_level,
            methodology=enhanced_methodology,
            score=enhanced_score,
            dimension_scores=base_complexity.dimension_scores,
            estimated_duration_minutes=enhanced_duration,
            estimated_queries=enhanced_queries,
            estimated_services=enhanced_services,
            confidence=max(base_complexity.confidence, business_intent.confidence),
            risk_factors=self._identify_risk_factors_simple(
                base_complexity.dimension_scores, 
                business_intent
            ),
            resource_requirements=self._build_resource_requirements(
                enhanced_level, 
                enhanced_methodology,
                enhanced_duration,
                enhanced_queries,
                enhanced_services
            )
        )
    
    def _select_methodology_simple(
        self, 
        complexity_level: ComplexityLevel, 
        business_intent: BusinessIntent
    ) -> InvestigationMethodology:
        """Simplified methodology selection for enhancement."""
        # Base methodology from complexity
        base_methodology = {
            ComplexityLevel.SIMPLE: InvestigationMethodology.RAPID_RESPONSE,
            ComplexityLevel.ANALYTICAL: InvestigationMethodology.SYSTEMATIC_ANALYSIS,
            ComplexityLevel.COMPUTATIONAL: InvestigationMethodology.SCENARIO_MODELING,
            ComplexityLevel.INVESTIGATIVE: InvestigationMethodology.MULTI_PHASE_ROOT_CAUSE
        }
        
        methodology = base_methodology.get(
            complexity_level, 
            InvestigationMethodology.SYSTEMATIC_ANALYSIS
        )
        
        # Upgrade for specific analysis types
        if business_intent.analysis_type == AnalysisType.PRESCRIPTIVE:
            if methodology == InvestigationMethodology.RAPID_RESPONSE:
                methodology = InvestigationMethodology.SYSTEMATIC_ANALYSIS
        
        return methodology
    
    def _identify_risk_factors_simple(
        self, 
        dimensions: Dict[str, float], 
        business_intent: BusinessIntent
    ) -> List[str]:
        """Identify risks based on complexity and business intent."""
        risks = []
        
        # High complexity risks
        if dimensions.get("analytical_depth", 0) >= 0.8:
            risks.append("Advanced analytics may require specialized expertise")
        
        # Business domain risks
        if business_intent.primary_domain == BusinessDomain.FINANCE:
            risks.append("Financial analysis requires high accuracy validation")
        
        # Multi-domain risks
        if len(business_intent.secondary_domains) > 1:
            risks.append("Cross-domain analysis may have data consistency issues")
        
        return risks[:3]  # Top 3 risks


if __name__ == "__main__":
    from domain_expert import DomainExpert
    
    analyzer = ComplexityAnalyzer()
    expert = DomainExpert()
    
    # Test queries with increasing complexity
    test_queries = [
        "Show current production status",
        "Compare this month's efficiency vs last month",
        "Analyze quarterly revenue trends across product lines",
        "Why did Line 2 efficiency drop 15% and what's the root cause?",
        "Optimize production schedule for next quarter considering all constraints"
    ]
    
    print("Complexity Analysis Test")
    print("=" * 60)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        
        # Get business intent first
        intent = expert.classify_business_intent(query)
        
        # Analyze complexity
        complexity = analyzer.analyze_complexity(intent, query)
        
        print(f"Complexity: {complexity.level.value}")
        print(f"Methodology: {complexity.methodology.value}")
        print(f"Score: {complexity.score:.3f}")
        print(f"Duration: {complexity.estimated_duration_minutes} minutes")
        print(f"Queries: {complexity.estimated_queries}")
        print(f"Services: {complexity.estimated_services}")
        print(f"Confidence: {complexity.confidence:.3f}")
        if complexity.risk_factors:
            print(f"Risks: {complexity.risk_factors[0]}")