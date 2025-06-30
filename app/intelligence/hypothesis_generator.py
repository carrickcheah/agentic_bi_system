"""
Hypothesis Generator - Diagnostic Investigation Support Component
Self-contained hypothesis generation for root cause analysis and diagnostic investigations.
Zero external dependencies beyond module boundary.
"""

from enum import Enum
from typing import Dict, List, Optional, Union
from dataclasses import dataclass

try:
    from .config import settings
    from .intelligence_logging import setup_logger, performance_monitor
    from .domain_expert import BusinessDomain, AnalysisType, BusinessIntent
    from .complexity_analyzer import ComplexityLevel, InvestigationMethodology
    from .business_context import ContextualStrategy
except ImportError:
    from config import settings
    from intelligence_logging import setup_logger, performance_monitor
    from domain_expert import BusinessDomain, AnalysisType, BusinessIntent
    from complexity_analyzer import ComplexityLevel, InvestigationMethodology
    from business_context import ContextualStrategy


class HypothesisType(Enum):
    """Types of diagnostic hypotheses."""
    ROOT_CAUSE = "root_cause"          # Direct causal relationship
    CORRELATION = "correlation"        # Statistical correlation
    ENVIRONMENTAL = "environmental"    # External environmental factors
    PROCESS = "process"               # Internal process issues
    SYSTEM = "system"                 # System or technical issues
    HUMAN_FACTOR = "human_factor"     # Human-related factors
    RESOURCE = "resource"             # Resource availability/quality issues


class ConfidenceLevel(Enum):
    """Confidence levels for hypotheses."""
    LOW = "low"           # 0.3-0.5
    MEDIUM = "medium"     # 0.5-0.7
    HIGH = "high"         # 0.7-0.85
    VERY_HIGH = "very_high"  # 0.85+


@dataclass
class Hypothesis:
    """Individual hypothesis for investigation."""
    id: str
    type: HypothesisType
    description: str
    investigation_approach: str
    expected_evidence: List[str]
    data_requirements: List[str]
    confidence: float
    priority: int  # 1 (highest) to 5 (lowest)
    estimated_effort: str  # low, medium, high
    success_indicators: List[str]
    related_domains: List[BusinessDomain]


@dataclass
class HypothesisSet:
    """Set of hypotheses for diagnostic investigation."""
    business_intent: BusinessIntent
    hypotheses: List[Hypothesis]
    investigation_strategy: str
    prioritized_sequence: List[str]  # Hypothesis IDs in order
    resource_allocation: Dict[str, int]  # hypothesis_id -> estimated minutes
    validation_approach: str
    success_criteria: List[str]


class HypothesisGenerator:
    """
    Generates diagnostic hypotheses for root cause analysis.
    Creates structured investigation approaches for complex business problems.
    """
    
    def __init__(self):
        self.logger = setup_logger("hypothesis_generator")
        self._domain_hypothesis_patterns = self._load_domain_patterns()
        self._causality_frameworks = self._load_causality_frameworks()
        self._investigation_templates = self._load_investigation_templates()
        
        self.logger.info("Hypothesis Generator initialized with diagnostic frameworks")
    
    def _load_domain_patterns(self) -> Dict[BusinessDomain, Dict[str, List[str]]]:
        """Load domain-specific hypothesis patterns."""
        return {
            BusinessDomain.PRODUCTION: {
                "common_causes": [
                    "equipment downtime", "setup/changeover delays", "material shortages",
                    "quality issues", "operator efficiency", "process bottlenecks",
                    "maintenance issues", "capacity constraints", "scheduling conflicts"
                ],
                "indicators": [
                    "oee decline", "cycle time increase", "yield drop",
                    "throughput reduction", "downtime events", "quality defects"
                ],
                "data_sources": [
                    "production logs", "equipment sensors", "quality records",
                    "maintenance logs", "operator schedules", "material inventory"
                ]
            },
            
            BusinessDomain.QUALITY: {
                "common_causes": [
                    "material defects", "process variation", "equipment calibration",
                    "operator training", "environmental conditions", "supplier issues",
                    "design specifications", "control system failures"
                ],
                "indicators": [
                    "defect rate increase", "customer complaints", "rework costs",
                    "inspection failures", "specification deviations", "control chart alerts"
                ],
                "data_sources": [
                    "inspection records", "customer feedback", "supplier quality data",
                    "process control charts", "environmental monitoring", "audit results"
                ]
            },
            
            BusinessDomain.SUPPLY_CHAIN: {
                "common_causes": [
                    "supplier delays", "transportation issues", "demand fluctuations",
                    "inventory imbalances", "forecasting errors", "communication gaps",
                    "regulatory changes", "quality issues"
                ],
                "indicators": [
                    "delivery delays", "stock shortages", "excess inventory",
                    "cost increases", "service level drops", "supplier performance decline"
                ],
                "data_sources": [
                    "supplier performance data", "inventory levels", "demand forecasts",
                    "transportation logs", "purchase orders", "delivery confirmations"
                ]
            },
            
            BusinessDomain.COST: {
                "common_causes": [
                    "material cost increases", "labor inefficiencies", "overhead allocation",
                    "process inefficiencies", "waste generation", "energy consumption",
                    "equipment utilization", "volume variances"
                ],
                "indicators": [
                    "cost variance", "margin decline", "budget overruns",
                    "efficiency ratios", "waste percentages", "utilization rates"
                ],
                "data_sources": [
                    "cost accounting systems", "purchasing records", "labor reports",
                    "utility bills", "waste tracking", "production volumes"
                ]
            },
            
            BusinessDomain.CUSTOMER: {
                "common_causes": [
                    "product quality issues", "delivery delays", "service failures",
                    "communication problems", "pricing concerns", "competitive pressure",
                    "changing requirements", "technical support issues"
                ],
                "indicators": [
                    "satisfaction scores", "complaint rates", "retention rates",
                    "response times", "resolution rates", "competitive losses"
                ],
                "data_sources": [
                    "customer surveys", "support tickets", "sales data",
                    "delivery records", "quality metrics", "competitive analysis"
                ]
            },
            
            BusinessDomain.ASSETS: {
                "common_causes": [
                    "maintenance delays", "equipment aging", "operating conditions",
                    "operator practices", "spare parts availability", "preventive maintenance",
                    "environmental factors", "design limitations"
                ],
                "indicators": [
                    "downtime frequency", "mtbf decline", "maintenance costs",
                    "availability reduction", "repair frequency", "performance degradation"
                ],
                "data_sources": [
                    "maintenance logs", "equipment sensors", "work orders",
                    "spare parts inventory", "operator logs", "environmental data"
                ]
            }
        }
    
    def _load_causality_frameworks(self) -> Dict[str, Dict[str, List[str]]]:
        """Load causality analysis frameworks."""
        return {
            "fishbone": {
                "categories": [
                    "people", "process", "equipment", "materials",
                    "environment", "measurement"
                ],
                "analysis_approach": [
                    "identify symptom", "categorize potential causes",
                    "drill down each category", "validate with data",
                    "prioritize by impact and likelihood"
                ]
            },
            
            "5_whys": {
                "steps": [
                    "define the problem", "ask why it occurred",
                    "ask why the cause occurred", "continue 5 levels deep",
                    "validate root cause with evidence"
                ],
                "validation_criteria": [
                    "logical causality", "data support",
                    "actionable solution", "prevents recurrence"
                ]
            },
            
            "fault_tree": {
                "elements": [
                    "top event", "intermediate events", "basic events",
                    "logic gates", "probability assignments"
                ],
                "analysis_flow": [
                    "define failure mode", "identify immediate causes",
                    "break down to basic events", "calculate probabilities",
                    "identify critical paths"
                ]
            },
            
            "statistical": {
                "methods": [
                    "correlation analysis", "regression analysis",
                    "control charts", "hypothesis testing",
                    "time series analysis"
                ],
                "evidence_types": [
                    "correlation coefficients", "p-values",
                    "confidence intervals", "trend analysis",
                    "pattern recognition"
                ]
            }
        }
    
    def _load_investigation_templates(self) -> Dict[InvestigationMethodology, Dict[str, Union[str, List[str]]]]:
        """Load investigation methodology templates."""
        return {
            InvestigationMethodology.MULTI_PHASE_ROOT_CAUSE: {
                "phases": [
                    "problem definition", "hypothesis generation",
                    "data collection", "hypothesis testing",
                    "root cause validation", "solution development"
                ],
                "hypothesis_limit": 5,
                "evidence_threshold": 0.7,
                "validation_approach": "statistical_and_logical"
            },
            
            InvestigationMethodology.SYSTEMATIC_ANALYSIS: {
                "phases": [
                    "symptom analysis", "cause identification",
                    "data validation", "solution recommendation"
                ],
                "hypothesis_limit": 3,
                "evidence_threshold": 0.6,
                "validation_approach": "data_driven"
            },
            
            InvestigationMethodology.RAPID_RESPONSE: {
                "phases": [
                    "immediate assessment", "likely cause identification",
                    "quick validation"
                ],
                "hypothesis_limit": 2,
                "evidence_threshold": 0.5,
                "validation_approach": "expert_judgment"
            }
        }
    
    @performance_monitor("hypothesis_generation")
    def generate_hypotheses(
        self,
        business_intent: BusinessIntent,
        contextual_strategy: ContextualStrategy,
        complexity_level: ComplexityLevel,
        investigation_context: Optional[Dict[str, str]] = None
    ) -> HypothesisSet:
        """
        Generate diagnostic hypotheses for investigation.
        
        Args:
            business_intent: Classified business intent
            contextual_strategy: Context-adapted strategy
            complexity_level: Investigation complexity level
            investigation_context: Additional context for hypothesis generation
            
        Returns:
            HypothesisSet with generated hypotheses and investigation strategy
        """
        # Determine investigation framework
        methodology = contextual_strategy.adapted_methodology
        framework = self._select_causality_framework(methodology, business_intent)
        
        # Generate hypotheses based on domain and indicators
        hypotheses = self._generate_domain_hypotheses(
            business_intent, complexity_level, investigation_context
        )
        
        # Apply causality framework structuring
        structured_hypotheses = self._apply_causality_framework(
            hypotheses, framework, business_intent
        )
        
        # Filter and prioritize based on constraints
        filtered_hypotheses = self._filter_and_prioritize(
            structured_hypotheses, methodology, contextual_strategy
        )
        
        # Create investigation strategy
        investigation_strategy = self._create_investigation_strategy(
            filtered_hypotheses, methodology, complexity_level
        )
        
        # Determine prioritized sequence
        prioritized_sequence = self._determine_investigation_sequence(
            filtered_hypotheses, contextual_strategy
        )
        
        # Allocate resources
        resource_allocation = self._allocate_resources(
            filtered_hypotheses, contextual_strategy.estimated_timeline
        )
        
        # Define validation approach
        validation_approach = self._define_validation_approach(
            methodology, complexity_level
        )
        
        self.logger.info(
            f"Generated {len(filtered_hypotheses)} hypotheses for "
            f"{business_intent.primary_domain.value} investigation"
        )
        
        return HypothesisSet(
            business_intent=business_intent,
            hypotheses=filtered_hypotheses,
            investigation_strategy=investigation_strategy,
            prioritized_sequence=prioritized_sequence,
            resource_allocation=resource_allocation,
            validation_approach=validation_approach,
            success_criteria=self._define_success_criteria(
                filtered_hypotheses, business_intent
            )
        )
    
    def _select_causality_framework(
        self, methodology: InvestigationMethodology, business_intent: BusinessIntent
    ) -> str:
        """Select appropriate causality analysis framework."""
        
        # Map methodology to preferred framework
        methodology_frameworks = {
            InvestigationMethodology.MULTI_PHASE_ROOT_CAUSE: "fault_tree",
            InvestigationMethodology.SYSTEMATIC_ANALYSIS: "fishbone",
            InvestigationMethodology.RAPID_RESPONSE: "5_whys",
            InvestigationMethodology.SCENARIO_MODELING: "statistical",
            InvestigationMethodology.STRATEGIC_ANALYSIS: "fishbone"
        }
        
        base_framework = methodology_frameworks.get(methodology, "fishbone")
        
        # Adjust based on analysis type
        if business_intent.analysis_type == AnalysisType.DIAGNOSTIC:
            if "root cause" in " ".join(business_intent.key_indicators):
                return "5_whys"
            elif len(business_intent.secondary_domains) > 1:
                return "fishbone"  # Good for multi-domain analysis
        
        return base_framework
    
    def _generate_domain_hypotheses(
        self,
        business_intent: BusinessIntent,
        complexity_level: ComplexityLevel,
        investigation_context: Optional[Dict[str, str]]
    ) -> List[Hypothesis]:
        """Generate hypotheses based on domain knowledge."""
        
        hypotheses = []
        domain_patterns = self._domain_hypothesis_patterns.get(
            business_intent.primary_domain, {}
        )
        
        if not domain_patterns:
            # Fallback to general patterns
            domain_patterns = self._domain_hypothesis_patterns[BusinessDomain.PRODUCTION]
        
        common_causes = domain_patterns.get("common_causes", [])
        indicators = domain_patterns.get("indicators", [])
        data_sources = domain_patterns.get("data_sources", [])
        
        # Generate hypotheses for each potential cause
        for i, cause in enumerate(common_causes[:settings.hypothesis_max_count]):
            # Determine hypothesis type
            hyp_type = self._classify_hypothesis_type(cause)
            
            # Calculate confidence based on indicator matching
            confidence = self._calculate_hypothesis_confidence(
                cause, business_intent.key_indicators, indicators
            )
            
            # Skip low-confidence hypotheses unless specifically needed
            if confidence < settings.hypothesis_confidence_threshold:
                continue
            
            # Generate hypothesis
            hypothesis = Hypothesis(
                id=f"hyp_{business_intent.primary_domain.value}_{i+1}",
                type=hyp_type,
                description=self._generate_hypothesis_description(
                    cause, business_intent, investigation_context
                ),
                investigation_approach=self._generate_investigation_approach(
                    cause, hyp_type, complexity_level
                ),
                expected_evidence=self._generate_expected_evidence(
                    cause, indicators, business_intent
                ),
                data_requirements=self._determine_data_requirements(
                    cause, data_sources, business_intent
                ),
                confidence=confidence,
                priority=self._calculate_priority(confidence, hyp_type, business_intent),
                estimated_effort=self._estimate_effort(cause, complexity_level),
                success_indicators=self._define_success_indicators(cause, business_intent),
                related_domains=[business_intent.primary_domain] + business_intent.secondary_domains
            )
            
            hypotheses.append(hypothesis)
        
        return hypotheses
    
    def _classify_hypothesis_type(self, cause: str) -> HypothesisType:
        """Classify hypothesis type based on cause description."""
        
        type_keywords = {
            HypothesisType.ROOT_CAUSE: ["failure", "breakdown", "defect", "error"],
            HypothesisType.PROCESS: ["process", "procedure", "workflow", "setup"],
            HypothesisType.SYSTEM: ["system", "equipment", "software", "technology"],
            HypothesisType.HUMAN_FACTOR: ["operator", "training", "skill", "personnel"],
            HypothesisType.RESOURCE: ["material", "inventory", "capacity", "shortage"],
            HypothesisType.ENVIRONMENTAL: ["environment", "weather", "external", "supplier"],
            HypothesisType.CORRELATION: ["correlation", "pattern", "trend", "statistical"]
        }
        
        cause_lower = cause.lower()
        for hyp_type, keywords in type_keywords.items():
            if any(keyword in cause_lower for keyword in keywords):
                return hyp_type
        
        return HypothesisType.ROOT_CAUSE  # Default
    
    def _calculate_hypothesis_confidence(
        self, cause: str, key_indicators: List[str], domain_indicators: List[str]
    ) -> float:
        """Calculate confidence score for hypothesis."""
        
        base_confidence = 0.5
        
        # Boost confidence if cause relates to key indicators
        cause_words = set(cause.lower().split())
        indicator_words = set(" ".join(key_indicators).lower().split())
        
        overlap = len(cause_words.intersection(indicator_words))
        if overlap > 0:
            base_confidence += min(overlap * 0.1, 0.3)
        
        # Boost confidence if cause relates to domain indicators
        domain_words = set(" ".join(domain_indicators).lower().split())
        domain_overlap = len(cause_words.intersection(domain_words))
        if domain_overlap > 0:
            base_confidence += min(domain_overlap * 0.05, 0.2)
        
        return min(base_confidence, 0.95)
    
    def _generate_hypothesis_description(
        self,
        cause: str,
        business_intent: BusinessIntent,
        investigation_context: Optional[Dict[str, str]]
    ) -> str:
        """Generate detailed hypothesis description."""
        
        # Extract key context
        context_str = ""
        if investigation_context:
            context_items = [f"{k}: {v}" for k, v in investigation_context.items()]
            context_str = f" given {', '.join(context_items[:2])}"
        
        # Create description based on analysis type
        if business_intent.analysis_type == AnalysisType.DIAGNOSTIC:
            return (
                f"The observed issue may be caused by {cause.lower()}{context_str}. "
                f"This hypothesis suggests investigating {cause.lower()} as a potential "
                f"root cause for the reported problem."
            )
        else:
            return (
                f"Analysis suggests that {cause.lower()} may be a contributing factor"
                f"{context_str}. Investigation should examine the relationship between "
                f"{cause.lower()} and the observed patterns."
            )
    
    def _generate_investigation_approach(
        self, cause: str, hyp_type: HypothesisType, complexity_level: ComplexityLevel
    ) -> str:
        """Generate investigation approach for hypothesis."""
        
        # Base approaches by hypothesis type
        type_approaches = {
            HypothesisType.ROOT_CAUSE: "Direct causal analysis with timeline correlation",
            HypothesisType.PROCESS: "Process mapping and deviation analysis",
            HypothesisType.SYSTEM: "System performance monitoring and fault analysis",
            HypothesisType.HUMAN_FACTOR: "Training records and performance correlation analysis",
            HypothesisType.RESOURCE: "Resource utilization and availability analysis",
            HypothesisType.ENVIRONMENTAL: "External factor correlation and impact analysis",
            HypothesisType.CORRELATION: "Statistical correlation and pattern analysis"
        }
        
        base_approach = type_approaches[hyp_type]
        
        # Enhance based on complexity
        if complexity_level in [ComplexityLevel.COMPUTATIONAL, ComplexityLevel.INVESTIGATIVE]:
            return f"{base_approach} with statistical validation and cross-domain verification"
        
        return base_approach
    
    def _generate_expected_evidence(
        self, cause: str, domain_indicators: List[str], business_intent: BusinessIntent
    ) -> List[str]:
        """Generate expected evidence for hypothesis validation."""
        
        evidence = []
        
        # Add cause-specific evidence
        if "equipment" in cause.lower():
            evidence.extend([
                "Equipment downtime logs", "Maintenance records",
                "Performance degradation patterns"
            ])
        elif "material" in cause.lower():
            evidence.extend([
                "Material quality reports", "Supplier performance data",
                "Inventory shortage records"
            ])
        elif "process" in cause.lower():
            evidence.extend([
                "Process variation data", "Procedure compliance records",
                "Quality control measurements"
            ])
        elif "operator" in cause.lower() or "training" in cause.lower():
            evidence.extend([
                "Training completion records", "Performance metrics",
                "Schedule correlation data"
            ])
        
        # Add domain-specific evidence
        relevant_indicators = [
            indicator for indicator in domain_indicators
            if any(word in indicator.lower() for word in business_intent.key_indicators)
        ]
        evidence.extend(relevant_indicators[:3])
        
        return evidence[:5]  # Limit to top 5
    
    def _determine_data_requirements(
        self, cause: str, domain_data_sources: List[str], business_intent: BusinessIntent
    ) -> List[str]:
        """Determine data requirements for hypothesis testing."""
        
        requirements = []
        
        # Add cause-specific data requirements
        cause_lower = cause.lower()
        for source in domain_data_sources:
            if any(word in source.lower() for word in cause_lower.split()):
                requirements.append(source)
        
        # Add intent-specific requirements
        if business_intent.time_context:
            requirements.append(f"Historical data for {business_intent.time_context}")
        
        # Add metrics-specific requirements
        for metric in business_intent.business_metrics:
            requirements.append(f"Time series data for {metric}")
        
        return list(set(requirements))[:5]  # Deduplicate and limit
    
    def _calculate_priority(
        self, confidence: float, hyp_type: HypothesisType, business_intent: BusinessIntent
    ) -> int:
        """Calculate investigation priority (1=highest, 5=lowest)."""
        
        base_priority = 3
        
        # Adjust based on confidence
        if confidence > 0.8:
            base_priority -= 2
        elif confidence > 0.6:
            base_priority -= 1
        elif confidence < 0.4:
            base_priority += 1
        
        # Adjust based on hypothesis type criticality
        type_priorities = {
            HypothesisType.ROOT_CAUSE: -1,      # Higher priority
            HypothesisType.SYSTEM: -1,          # Higher priority
            HypothesisType.PROCESS: 0,          # Normal priority
            HypothesisType.RESOURCE: 0,         # Normal priority
            HypothesisType.HUMAN_FACTOR: 1,     # Lower priority
            HypothesisType.ENVIRONMENTAL: 1,    # Lower priority
            HypothesisType.CORRELATION: 2       # Lowest priority
        }
        
        base_priority += type_priorities.get(hyp_type, 0)
        
        # Adjust based on urgency
        if business_intent.urgency_level == "critical":
            base_priority -= 1
        elif business_intent.urgency_level == "low":
            base_priority += 1
        
        return max(1, min(base_priority, 5))
    
    def _estimate_effort(self, cause: str, complexity_level: ComplexityLevel) -> str:
        """Estimate investigation effort level."""
        
        # Base effort by complexity
        base_efforts = {
            ComplexityLevel.SIMPLE: "low",
            ComplexityLevel.ANALYTICAL: "medium",
            ComplexityLevel.COMPUTATIONAL: "high",
            ComplexityLevel.INVESTIGATIVE: "high"
        }
        
        base_effort = base_efforts[complexity_level]
        
        # Adjust for cause complexity
        complex_causes = ["process", "system", "multi", "complex", "integration"]
        if any(term in cause.lower() for term in complex_causes):
            if base_effort == "low":
                return "medium"
            elif base_effort == "medium":
                return "high"
        
        return base_effort
    
    def _define_success_indicators(
        self, cause: str, business_intent: BusinessIntent
    ) -> List[str]:
        """Define success indicators for hypothesis validation."""
        
        indicators = []
        
        # Add cause-specific indicators
        if "equipment" in cause.lower():
            indicators.append("Equipment performance improvement correlation")
        elif "process" in cause.lower():
            indicators.append("Process variation reduction")
        elif "material" in cause.lower():
            indicators.append("Material quality improvement correlation")
        
        # Add intent-specific indicators
        for metric in business_intent.business_metrics:
            indicators.append(f"Improvement in {metric} after addressing {cause}")
        
        indicators.append("Statistical significance of relationship")
        indicators.append("Logical causality validation")
        
        return indicators[:4]
    
    def _apply_causality_framework(
        self, hypotheses: List[Hypothesis], framework: str, business_intent: BusinessIntent
    ) -> List[Hypothesis]:
        """Apply causality framework structuring to hypotheses."""
        
        if framework == "fishbone":
            return self._apply_fishbone_structure(hypotheses)
        elif framework == "5_whys":
            return self._apply_5_whys_structure(hypotheses, business_intent)
        elif framework == "fault_tree":
            return self._apply_fault_tree_structure(hypotheses)
        elif framework == "statistical":
            return self._apply_statistical_structure(hypotheses)
        
        return hypotheses
    
    def _apply_fishbone_structure(self, hypotheses: List[Hypothesis]) -> List[Hypothesis]:
        """Apply fishbone (Ishikawa) diagram structure."""
        
        # Categorize hypotheses by fishbone categories
        categories = {
            "people": HypothesisType.HUMAN_FACTOR,
            "process": HypothesisType.PROCESS,
            "equipment": HypothesisType.SYSTEM,
            "materials": HypothesisType.RESOURCE,
            "environment": HypothesisType.ENVIRONMENTAL,
            "measurement": HypothesisType.CORRELATION
        }
        
        # Update descriptions to include category context
        for hypothesis in hypotheses:
            for category, hyp_type in categories.items():
                if hypothesis.type == hyp_type:
                    hypothesis.description = (
                        f"[{category.title()} Factor] {hypothesis.description}"
                    )
                    break
        
        return hypotheses
    
    def _apply_5_whys_structure(
        self, hypotheses: List[Hypothesis], business_intent: BusinessIntent
    ) -> List[Hypothesis]:
        """Apply 5 Whys methodology structure."""
        
        # Prioritize root cause hypotheses
        for hypothesis in hypotheses:
            if hypothesis.type == HypothesisType.ROOT_CAUSE:
                hypothesis.priority = max(1, hypothesis.priority - 1)
                hypothesis.investigation_approach = (
                    "Five Whys analysis: " + hypothesis.investigation_approach
                )
        
        return hypotheses
    
    def _apply_fault_tree_structure(self, hypotheses: List[Hypothesis]) -> List[Hypothesis]:
        """Apply fault tree analysis structure."""
        
        # Add probability considerations
        for hypothesis in hypotheses:
            hypothesis.investigation_approach = (
                "Fault tree analysis with probability assessment: " +
                hypothesis.investigation_approach
            )
            hypothesis.expected_evidence.append("Failure probability data")
        
        return hypotheses
    
    def _apply_statistical_structure(self, hypotheses: List[Hypothesis]) -> List[Hypothesis]:
        """Apply statistical analysis structure."""
        
        # Enhance with statistical requirements
        for hypothesis in hypotheses:
            if hypothesis.type != HypothesisType.CORRELATION:
                hypothesis.type = HypothesisType.CORRELATION
            
            hypothesis.investigation_approach = (
                "Statistical correlation analysis: " + hypothesis.investigation_approach
            )
            hypothesis.expected_evidence.extend([
                "Correlation coefficients", "Statistical significance tests"
            ])
        
        return hypotheses
    
    def _filter_and_prioritize(
        self,
        hypotheses: List[Hypothesis],
        methodology: InvestigationMethodology,
        contextual_strategy: ContextualStrategy
    ) -> List[Hypothesis]:
        """Filter and prioritize hypotheses based on constraints."""
        
        template = self._investigation_templates[methodology]
        max_hypotheses = template["hypothesis_limit"]
        
        # Sort by priority (lower number = higher priority)
        sorted_hypotheses = sorted(hypotheses, key=lambda h: (h.priority, -h.confidence))
        
        # Filter by confidence threshold
        threshold = template["evidence_threshold"]
        filtered = [h for h in sorted_hypotheses if h.confidence >= threshold]
        
        # Apply contextual constraints
        speed_pref = contextual_strategy.user_preferences.get("speed_preference", 0.5)
        if speed_pref > 0.7:  # High speed preference
            # Prefer low effort hypotheses
            filtered = sorted(filtered, key=lambda h: (
                h.priority,
                0 if h.estimated_effort == "low" else 1 if h.estimated_effort == "medium" else 2,
                -h.confidence
            ))
        
        return filtered[:max_hypotheses]
    
    def _create_investigation_strategy(
        self,
        hypotheses: List[Hypothesis],
        methodology: InvestigationMethodology,
        complexity_level: ComplexityLevel
    ) -> str:
        """Create overall investigation strategy."""
        
        strategy_parts = []
        
        # Add methodology description
        strategy_parts.append(f"Investigation will follow {methodology.value} methodology")
        
        # Add hypothesis overview
        hypothesis_types = list(set(h.type.value for h in hypotheses))
        strategy_parts.append(
            f"Testing {len(hypotheses)} hypotheses focusing on {', '.join(hypothesis_types)}"
        )
        
        # Add complexity considerations
        if complexity_level in [ComplexityLevel.COMPUTATIONAL, ComplexityLevel.INVESTIGATIVE]:
            strategy_parts.append(
                "Advanced statistical analysis and cross-domain validation will be applied"
            )
        
        # Add validation approach
        template = self._investigation_templates[methodology]
        validation = template["validation_approach"]
        strategy_parts.append(f"Evidence validation will use {validation} approach")
        
        return ". ".join(strategy_parts) + "."
    
    def _determine_investigation_sequence(
        self, hypotheses: List[Hypothesis], contextual_strategy: ContextualStrategy
    ) -> List[str]:
        """Determine optimal investigation sequence."""
        
        # Sort by priority and effort considerations
        speed_pref = contextual_strategy.user_preferences.get("speed_preference", 0.5)
        
        if speed_pref > 0.7:
            # Prioritize quick wins: low effort, high confidence
            sorted_hyp = sorted(hypotheses, key=lambda h: (
                h.priority,
                0 if h.estimated_effort == "low" else 2,
                -h.confidence
            ))
        else:
            # Prioritize by importance: priority and confidence
            sorted_hyp = sorted(hypotheses, key=lambda h: (h.priority, -h.confidence))
        
        return [h.id for h in sorted_hyp]
    
    def _allocate_resources(
        self, hypotheses: List[Hypothesis], timeline: Dict[str, int]
    ) -> Dict[str, int]:
        """Allocate time resources to hypotheses."""
        
        total_analysis_time = timeline.get("analysis", 30)
        
        # Allocate based on effort and priority
        effort_weights = {"low": 1, "medium": 2, "high": 3}
        total_weight = sum(effort_weights[h.estimated_effort] / h.priority for h in hypotheses)
        
        allocation = {}
        for hypothesis in hypotheses:
            weight = effort_weights[hypothesis.estimated_effort] / hypothesis.priority
            allocated_time = int((weight / total_weight) * total_analysis_time)
            allocation[hypothesis.id] = max(allocated_time, 2)  # Minimum 2 minutes
        
        return allocation
    
    def _define_validation_approach(
        self, methodology: InvestigationMethodology, complexity_level: ComplexityLevel
    ) -> str:
        """Define validation approach for investigation."""
        
        template = self._investigation_templates[methodology]
        base_approach = template["validation_approach"]
        
        if complexity_level == ComplexityLevel.INVESTIGATIVE:
            return f"{base_approach} with multi-phase validation and cross-verification"
        elif complexity_level == ComplexityLevel.COMPUTATIONAL:
            return f"{base_approach} with statistical significance testing"
        
        return base_approach
    
    def _define_success_criteria(
        self, hypotheses: List[Hypothesis], business_intent: BusinessIntent
    ) -> List[str]:
        """Define overall success criteria for investigation."""
        
        criteria = []
        
        # Add hypothesis-specific criteria
        if len(hypotheses) > 0:
            criteria.append(f"At least one hypothesis validated with high confidence")
            criteria.append("Clear causal relationship established")
        
        # Add intent-specific criteria
        if business_intent.analysis_type == AnalysisType.DIAGNOSTIC:
            criteria.append("Root cause identified and actionable")
        
        criteria.extend([
            "Evidence supports logical causality",
            "Solution pathway identified",
            "Risk of recurrence assessed"
        ])
        
        return criteria


# Standalone execution for testing
if __name__ == "__main__":
    from domain_expert import DomainExpert, BusinessIntent, BusinessDomain, AnalysisType
    from complexity_analyzer import ComplexityAnalyzer, ComplexityLevel, InvestigationMethodology
    from business_context import BusinessContextAnalyzer, ContextualStrategy
    
    generator = HypothesisGenerator()
    
    # Test hypothesis generation
    business_intent = BusinessIntent(
        primary_domain=BusinessDomain.PRODUCTION,
        secondary_domains=[],
        analysis_type=AnalysisType.DIAGNOSTIC,
        confidence=0.85,
        key_indicators=["efficiency", "line", "drop"],
        business_metrics=["efficiency %"],
        time_context="last_week",
        urgency_level="high"
    )
    
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
    
    hypothesis_set = generator.generate_hypotheses(
        business_intent=business_intent,
        contextual_strategy=contextual_strategy,
        complexity_level=ComplexityLevel.INVESTIGATIVE,
        investigation_context={"line": "Line 2", "metric": "efficiency"}
    )
    
    print("Hypothesis Generation Test")
    print("=" * 50)
    print(f"Generated {len(hypothesis_set.hypotheses)} hypotheses")
    print(f"Investigation Strategy: {hypothesis_set.investigation_strategy}")
    print(f"Prioritized Sequence: {hypothesis_set.prioritized_sequence}")
    print(f"Validation Approach: {hypothesis_set.validation_approach}")
    
    for i, hypothesis in enumerate(hypothesis_set.hypotheses[:3]):
        print(f"\nHypothesis {i+1}:")
        print(f"  Type: {hypothesis.type.value}")
        print(f"  Description: {hypothesis.description[:100]}...")
        print(f"  Confidence: {hypothesis.confidence:.3f}")
        print(f"  Priority: {hypothesis.priority}")
        print(f"  Estimated Effort: {hypothesis.estimated_effort}")