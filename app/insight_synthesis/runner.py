"""
Insight Synthesis Runner - Phase 5: Strategic Business Intelligence Generation
Transforms raw investigation findings into strategic business intelligence with role-specific formatting.
Self-contained orchestrator for insight synthesis workflow.
"""

from enum import Enum
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from datetime import datetime, timezone
import asyncio

try:
    from .config import settings
    from .synthesis_logging import setup_logger, performance_monitor
except ImportError:
    from config import settings
    from synthesis_logging import setup_logger, performance_monitor


class InsightType(Enum):
    """Types of business insights."""
    OPERATIONAL = "operational"          # Day-to-day operations insights
    TACTICAL = "tactical"               # Short-term strategic insights
    STRATEGIC = "strategic"             # Long-term strategic insights
    TRANSFORMATIONAL = "transformational"  # Business transformation insights
    RISK_MITIGATION = "risk_mitigation"   # Risk management insights
    OPPORTUNITY = "opportunity"          # Business opportunity insights
    EFFICIENCY = "efficiency"           # Process efficiency insights
    COMPLIANCE = "compliance"           # Regulatory compliance insights


class RecommendationType(Enum):
    """Types of recommendations."""
    IMMEDIATE_ACTION = "immediate_action"    # Actions needed now
    SHORT_TERM = "short_term"               # Actions within 1-3 months
    LONG_TERM = "long_term"                 # Actions within 6-12 months
    STRATEGIC_INITIATIVE = "strategic_initiative"  # Strategic programs
    PROCESS_IMPROVEMENT = "process_improvement"     # Process changes
    RESOURCE_ALLOCATION = "resource_allocation"     # Resource decisions
    MONITORING = "monitoring"                       # Ongoing monitoring


class OutputFormat(Enum):
    """Output format types."""
    EXECUTIVE_SUMMARY = "executive_summary"
    DETAILED_REPORT = "detailed_report"
    PRESENTATION = "presentation"
    DASHBOARD = "dashboard"
    ACTION_PLAN = "action_plan"
    TECHNICAL_BRIEF = "technical_brief"


@dataclass
class BusinessInsight:
    """Structured business insight."""
    id: str
    type: InsightType
    title: str
    description: str
    business_context: str
    supporting_evidence: List[str]
    confidence: float
    business_impact: Dict[str, float]  # financial, operational, strategic, risk
    strategic_depth: float  # 0.0 to 1.0
    actionability: float   # 0.0 to 1.0
    stakeholders: List[str]
    related_domains: List[str]
    discovery_timestamp: datetime


@dataclass
class Recommendation:
    """Actionable business recommendation."""
    id: str
    type: RecommendationType
    title: str
    description: str
    rationale: str
    implementation_approach: str
    resource_requirements: Dict[str, Any]
    expected_outcomes: List[str]
    success_metrics: List[str]
    priority: int  # 1 (highest) to 5 (lowest)
    timeline: str
    risk_level: str  # low, medium, high
    feasibility: float  # 0.0 to 1.0
    related_insight_ids: List[str]


@dataclass
class OrganizationalLearning:
    """Captured organizational learning."""
    pattern_id: str
    pattern_description: str
    frequency: int
    success_rate: float
    business_value: float
    applicable_domains: List[str]
    best_practices: List[str]
    lessons_learned: List[str]
    improvement_opportunities: List[str]


@dataclass
class SynthesisResult:
    """Complete insight synthesis result."""
    investigation_id: str
    insights: List[BusinessInsight]
    recommendations: List[Recommendation]
    organizational_learning: OrganizationalLearning
    executive_summary: str
    key_findings: List[str]
    business_impact_assessment: Dict[str, Any]
    success_criteria: List[str]
    follow_up_actions: List[str]
    stakeholder_communications: Dict[str, str]  # role -> message
    synthesis_metadata: Dict[str, Any]


class InsightSynthesizer:
    """
    Strategic insight synthesizer for business intelligence generation.
    Transforms investigation findings into actionable business intelligence.
    """
    
    def __init__(self):
        self.logger = setup_logger("insight_synthesizer")
        self._insight_patterns = self._load_insight_patterns()
        self._role_templates = self._load_role_templates()
        self._impact_calculators = self._load_impact_calculators()
        
        self.logger.info("Insight Synthesizer initialized for strategic intelligence generation")
    
    def _load_insight_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Load insight generation patterns."""
        return {
            "efficiency_decline": {
                "insight_type": InsightType.OPERATIONAL,
                "indicators": ["efficiency drop", "performance decline", "productivity loss"],
                "business_impact_areas": ["operational", "financial"],
                "typical_recommendations": [
                    RecommendationType.IMMEDIATE_ACTION,
                    RecommendationType.PROCESS_IMPROVEMENT
                ]
            },
            
            "quality_issues": {
                "insight_type": InsightType.OPERATIONAL,
                "indicators": ["defect rate", "quality problems", "customer complaints"],
                "business_impact_areas": ["operational", "strategic"],
                "typical_recommendations": [
                    RecommendationType.IMMEDIATE_ACTION,
                    RecommendationType.PROCESS_IMPROVEMENT,
                    RecommendationType.MONITORING
                ]
            },
            
            "cost_variance": {
                "insight_type": InsightType.TACTICAL,
                "indicators": ["cost increase", "budget overrun", "variance"],
                "business_impact_areas": ["financial", "operational"],
                "typical_recommendations": [
                    RecommendationType.SHORT_TERM,
                    RecommendationType.RESOURCE_ALLOCATION
                ]
            },
            
            "supply_chain_disruption": {
                "insight_type": InsightType.STRATEGIC,
                "indicators": ["supply delay", "inventory shortage", "supplier issues"],
                "business_impact_areas": ["operational", "strategic", "risk"],
                "typical_recommendations": [
                    RecommendationType.SHORT_TERM,
                    RecommendationType.STRATEGIC_INITIATIVE
                ]
            },
            
            "customer_satisfaction": {
                "insight_type": InsightType.STRATEGIC,
                "indicators": ["satisfaction decline", "customer complaints", "retention"],
                "business_impact_areas": ["strategic", "financial"],
                "typical_recommendations": [
                    RecommendationType.IMMEDIATE_ACTION,
                    RecommendationType.STRATEGIC_INITIATIVE
                ]
            },
            
            "equipment_performance": {
                "insight_type": InsightType.OPERATIONAL,
                "indicators": ["downtime", "maintenance", "equipment failure"],
                "business_impact_areas": ["operational", "financial", "risk"],
                "typical_recommendations": [
                    RecommendationType.IMMEDIATE_ACTION,
                    RecommendationType.LONG_TERM,
                    RecommendationType.MONITORING
                ]
            }
        }
    
    def _load_role_templates(self) -> Dict[str, Dict[str, str]]:
        """Load role-specific communication templates."""
        return {
            "executive": {
                "focus": "strategic impact and business outcomes",
                "detail_level": "high-level summary with key metrics",
                "communication_style": "strategic and outcome-focused",
                "key_elements": "ROI, strategic alignment, competitive advantage"
            },
            
            "manager": {
                "focus": "operational improvements and team actions",
                "detail_level": "actionable insights with clear next steps",
                "communication_style": "practical and action-oriented", 
                "key_elements": "team impact, resource requirements, timelines"
            },
            
            "analyst": {
                "focus": "detailed analysis and methodology",
                "detail_level": "comprehensive findings with supporting data",
                "communication_style": "analytical and data-driven",
                "key_elements": "methodology, data quality, statistical significance"
            },
            
            "engineer": {
                "focus": "technical implementation and system changes",
                "detail_level": "technical specifications and implementation details",
                "communication_style": "technical and solution-focused",
                "key_elements": "technical approach, system requirements, implementation steps"
            },
            
            "specialist": {
                "focus": "domain-specific insights and expert recommendations",
                "detail_level": "specialized knowledge and best practices",
                "communication_style": "expert-level and domain-focused",
                "key_elements": "best practices, industry standards, expert recommendations"
            }
        }
    
    def _load_impact_calculators(self) -> Dict[str, callable]:
        """Load business impact calculation methods."""
        return {
            "financial_impact": self._calculate_financial_impact,
            "operational_efficiency": self._calculate_operational_impact,
            "strategic_alignment": self._calculate_strategic_impact,
            "risk_mitigation": self._calculate_risk_impact
        }
    
    @performance_monitor("insight_synthesis")
    async def synthesize_insights(
        self,
        investigation_results: Dict[str, Any],
        business_context: Optional[Dict[str, Any]] = None,
        user_role: Optional[str] = None,
        output_format: Optional[OutputFormat] = None
    ) -> SynthesisResult:
        """
        Synthesize strategic insights from investigation results.
        
        Args:
            investigation_results: Raw findings from Phase 4 investigation
            business_context: Additional business context
            user_role: User role for tailored output
            output_format: Desired output format
            
        Returns:
            SynthesisResult with strategic insights and recommendations
        """
        if output_format is None:
            output_format = OutputFormat(settings.default_output_format)
        
        # Extract key findings and patterns
        findings = self._extract_key_findings(investigation_results)
        
        # Generate business insights
        insights = await self._generate_insights(findings, business_context)
        
        # Create actionable recommendations
        recommendations = await self._generate_recommendations(insights, findings)
        
        # Capture organizational learning
        learning = self._capture_organizational_learning(insights, findings)
        
        # Calculate business impact assessment
        impact_assessment = self._calculate_business_impact(insights, recommendations)
        
        # Generate executive summary
        executive_summary = self._generate_executive_summary(
            insights, recommendations, impact_assessment
        )
        
        # Create stakeholder communications
        stakeholder_comms = self._generate_stakeholder_communications(
            insights, recommendations, user_role
        )
        
        # Define success criteria and follow-up actions
        success_criteria = self._define_success_criteria(insights, recommendations)
        follow_up_actions = self._generate_follow_up_actions(recommendations)
        
        # Create synthesis metadata
        metadata = self._create_synthesis_metadata(
            investigation_results, business_context, output_format
        )
        
        self.logger.info(
            f"Synthesized {len(insights)} insights and {len(recommendations)} "
            f"recommendations from investigation findings"
        )
        
        return SynthesisResult(
            investigation_id=investigation_results.get("investigation_id", "unknown"),
            insights=insights,
            recommendations=recommendations,
            organizational_learning=learning,
            executive_summary=executive_summary,
            key_findings=[f["description"] for f in findings],
            business_impact_assessment=impact_assessment,
            success_criteria=success_criteria,
            follow_up_actions=follow_up_actions,
            stakeholder_communications=stakeholder_comms,
            synthesis_metadata=metadata
        )
    
    def _extract_key_findings(self, investigation_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract and structure key findings from investigation results."""
        findings = []
        
        # Extract from investigation steps
        if "step_results" in investigation_results:
            for step_name, step_result in investigation_results["step_results"].items():
                if "key_findings" in step_result:
                    for finding in step_result["key_findings"]:
                        findings.append({
                            "source_step": step_name,
                            "description": finding,
                            "confidence": step_result.get("confidence", 0.7),
                            "evidence": step_result.get("supporting_evidence", []),
                            "data_quality": step_result.get("data_quality", 0.8)
                        })
        
        # Extract from summary
        if "summary" in investigation_results:
            summary = investigation_results["summary"]
            if "conclusions" in summary:
                for conclusion in summary["conclusions"]:
                    findings.append({
                        "source_step": "summary",
                        "description": conclusion,
                        "confidence": summary.get("overall_confidence", 0.8),
                        "evidence": summary.get("supporting_data", []),
                        "data_quality": summary.get("data_reliability", 0.8)
                    })
        
        return findings
    
    async def _generate_insights(
        self, 
        findings: List[Dict[str, Any]], 
        business_context: Optional[Dict[str, Any]]
    ) -> List[BusinessInsight]:
        """Generate strategic business insights from findings."""
        insights = []
        
        for i, finding in enumerate(findings):
            # Classify insight type based on finding content
            insight_type = self._classify_insight_type(finding["description"])
            
            # Calculate strategic depth
            strategic_depth = self._calculate_strategic_depth(finding, business_context)
            
            # Calculate business impact
            business_impact = self._calculate_insight_business_impact(finding)
            
            # Determine actionability
            actionability = self._calculate_actionability(finding)
            
            # Filter by confidence threshold
            if finding["confidence"] < settings.insight_confidence_threshold:
                continue
            
            # Create insight
            insight = BusinessInsight(
                id=f"insight_{i+1}",
                type=insight_type,
                title=self._generate_insight_title(finding, insight_type),
                description=self._enhance_insight_description(finding, business_context),
                business_context=self._generate_business_context(finding, business_context),
                supporting_evidence=finding.get("evidence", []),
                confidence=finding["confidence"],
                business_impact=business_impact,
                strategic_depth=strategic_depth,
                actionability=actionability,
                stakeholders=self._identify_stakeholders(finding, insight_type),
                related_domains=self._identify_related_domains(finding),
                discovery_timestamp=datetime.now(timezone.utc)
            )
            
            insights.append(insight)
        
        # Sort by business impact and confidence
        insights.sort(key=lambda x: (
            sum(x.business_impact.values()) * x.confidence
        ), reverse=True)
        
        return insights[:settings.max_insights_per_investigation]
    
    async def _generate_recommendations(
        self, 
        insights: List[BusinessInsight], 
        findings: List[Dict[str, Any]]
    ) -> List[Recommendation]:
        """Generate actionable recommendations from insights."""
        recommendations = []
        
        for insight in insights:
            # Generate recommendations for this insight
            insight_recommendations = self._generate_insight_recommendations(insight)
            
            for rec_data in insight_recommendations:
                recommendation = Recommendation(
                    id=f"rec_{len(recommendations)+1}",
                    type=rec_data["type"],
                    title=rec_data["title"],
                    description=rec_data["description"],
                    rationale=rec_data["rationale"],
                    implementation_approach=rec_data["implementation"],
                    resource_requirements=rec_data["resources"],
                    expected_outcomes=rec_data["outcomes"],
                    success_metrics=rec_data["metrics"],
                    priority=rec_data["priority"],
                    timeline=rec_data["timeline"],
                    risk_level=rec_data["risk_level"],
                    feasibility=rec_data["feasibility"],
                    related_insight_ids=[insight.id]
                )
                recommendations.append(recommendation)
        
        # Sort by priority and feasibility
        recommendations.sort(key=lambda x: (
            x.priority, -x.feasibility
        ))
        
        return recommendations[:settings.recommendation_max_count]
    
    def _classify_insight_type(self, description: str) -> InsightType:
        """Classify insight type based on description content."""
        description_lower = description.lower()
        
        # Check for pattern matches
        for pattern_name, pattern_data in self._insight_patterns.items():
            if any(indicator in description_lower for indicator in pattern_data["indicators"]):
                return pattern_data["insight_type"]
        
        # Default classification based on keywords
        if any(word in description_lower for word in ["immediate", "urgent", "critical"]):
            return InsightType.OPERATIONAL
        elif any(word in description_lower for word in ["strategic", "long-term", "competitive"]):
            return InsightType.STRATEGIC
        elif any(word in description_lower for word in ["efficiency", "process", "optimization"]):
            return InsightType.EFFICIENCY
        elif any(word in description_lower for word in ["risk", "compliance", "regulatory"]):
            return InsightType.RISK_MITIGATION
        else:
            return InsightType.TACTICAL
    
    def _calculate_strategic_depth(
        self, 
        finding: Dict[str, Any], 
        business_context: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate strategic depth of the insight."""
        base_depth = 0.5
        
        description = finding["description"].lower()
        
        # Increase depth for strategic keywords
        strategic_keywords = [
            "competitive", "market", "strategic", "transformation", 
            "innovation", "growth", "sustainability"
        ]
        depth_boost = sum(0.1 for keyword in strategic_keywords if keyword in description)
        
        # Consider business context
        if business_context:
            if business_context.get("strategic_priority", False):
                depth_boost += 0.2
            if business_context.get("executive_sponsor", False):
                depth_boost += 0.1
        
        return min(base_depth + depth_boost, 1.0)
    
    def _calculate_insight_business_impact(self, finding: Dict[str, Any]) -> Dict[str, float]:
        """Calculate business impact across multiple dimensions."""
        impact = {}
        
        for dimension, calculator in self._impact_calculators.items():
            impact[dimension] = calculator(finding)
        
        return impact
    
    def _calculate_financial_impact(self, finding: Dict[str, Any]) -> float:
        """Calculate financial impact score."""
        description = finding["description"].lower()
        
        # High impact financial keywords
        high_impact = ["cost", "revenue", "profit", "savings", "budget", "roi"]
        medium_impact = ["efficiency", "productivity", "waste", "utilization"]
        
        if any(word in description for word in high_impact):
            return 0.8
        elif any(word in description for word in medium_impact):
            return 0.6
        else:
            return 0.4
    
    def _calculate_operational_impact(self, finding: Dict[str, Any]) -> float:
        """Calculate operational efficiency impact score."""
        description = finding["description"].lower()
        
        operational_keywords = [
            "efficiency", "performance", "productivity", "throughput", 
            "downtime", "utilization", "process"
        ]
        
        impact_score = sum(0.1 for keyword in operational_keywords if keyword in description)
        return min(0.4 + impact_score, 1.0)
    
    def _calculate_strategic_impact(self, finding: Dict[str, Any]) -> float:
        """Calculate strategic alignment impact score."""
        description = finding["description"].lower()
        
        strategic_keywords = [
            "strategic", "competitive", "market", "customer", "growth", 
            "innovation", "transformation"
        ]
        
        impact_score = sum(0.15 for keyword in strategic_keywords if keyword in description)
        return min(0.3 + impact_score, 1.0)
    
    def _calculate_risk_impact(self, finding: Dict[str, Any]) -> float:
        """Calculate risk mitigation impact score."""
        description = finding["description"].lower()
        
        risk_keywords = [
            "risk", "compliance", "safety", "regulatory", "audit", 
            "failure", "incident", "violation"
        ]
        
        impact_score = sum(0.12 for keyword in risk_keywords if keyword in description)
        return min(0.2 + impact_score, 1.0)
    
    def _calculate_actionability(self, finding: Dict[str, Any]) -> float:
        """Calculate how actionable the insight is."""
        base_actionability = 0.6
        
        # Boost for specific, measurable findings
        if finding.get("data_quality", 0) > 0.8:
            base_actionability += 0.2
        
        # Boost for clear cause-effect relationships
        description = finding["description"].lower()
        if any(word in description for word in ["caused by", "due to", "because", "correlation"]):
            base_actionability += 0.1
        
        return min(base_actionability, 1.0)
    
    def _generate_insight_title(self, finding: Dict[str, Any], insight_type: InsightType) -> str:
        """Generate a clear, actionable insight title."""
        description = finding["description"]
        
        # Extract key elements
        if "efficiency" in description.lower():
            return f"Production Efficiency Analysis - {insight_type.value.title()} Impact"
        elif "quality" in description.lower():
            return f"Quality Performance Insight - {insight_type.value.title()} Focus"
        elif "cost" in description.lower():
            return f"Cost Management Opportunity - {insight_type.value.title()} Level"
        else:
            return f"Business Intelligence Insight - {insight_type.value.title()}"
    
    def _enhance_insight_description(
        self, 
        finding: Dict[str, Any], 
        business_context: Optional[Dict[str, Any]]
    ) -> str:
        """Enhance finding description with business intelligence perspective."""
        base_description = finding["description"]
        
        # Add business context if available
        if business_context:
            context_note = ""
            if business_context.get("current_initiative"):
                context_note += f" This relates to the ongoing {business_context['current_initiative']} initiative."
            if business_context.get("strategic_goal"):
                context_note += f" This aligns with the strategic goal of {business_context['strategic_goal']}."
            
            return base_description + context_note
        
        return base_description
    
    def _generate_business_context(
        self, 
        finding: Dict[str, Any], 
        business_context: Optional[Dict[str, Any]]
    ) -> str:
        """Generate comprehensive business context for the insight."""
        context_parts = []
        
        # Add source context
        context_parts.append(f"Discovered through {finding.get('source_step', 'investigation')} analysis")
        
        # Add confidence context
        confidence = finding.get("confidence", 0.7)
        if confidence > 0.8:
            context_parts.append("with high confidence")
        elif confidence > 0.6:
            context_parts.append("with moderate confidence")
        else:
            context_parts.append("requiring further validation")
        
        # Add business context if available
        if business_context:
            if business_context.get("business_unit"):
                context_parts.append(f"within {business_context['business_unit']}")
            if business_context.get("time_period"):
                context_parts.append(f"for {business_context['time_period']}")
        
        return ". ".join(context_parts) + "."
    
    def _identify_stakeholders(self, finding: Dict[str, Any], insight_type: InsightType) -> List[str]:
        """Identify relevant stakeholders for the insight."""
        stakeholders = []
        
        # Add stakeholders based on insight type
        if insight_type == InsightType.OPERATIONAL:
            stakeholders.extend(["Operations Manager", "Plant Manager", "Shift Supervisors"])
        elif insight_type == InsightType.STRATEGIC:
            stakeholders.extend(["Executive Team", "VP Operations", "Strategic Planning"])
        elif insight_type == InsightType.EFFICIENCY:
            stakeholders.extend(["Process Engineers", "Continuous Improvement", "Operations"])
        elif insight_type == InsightType.RISK_MITIGATION:
            stakeholders.extend(["Risk Management", "Compliance Officer", "Safety Manager"])
        
        # Add domain-specific stakeholders
        description = finding["description"].lower()
        if "quality" in description:
            stakeholders.append("Quality Manager")
        if "maintenance" in description:
            stakeholders.append("Maintenance Manager")
        if "supply" in description:
            stakeholders.append("Supply Chain Manager")
        
        return list(set(stakeholders))
    
    def _identify_related_domains(self, finding: Dict[str, Any]) -> List[str]:
        """Identify business domains related to the insight."""
        domains = []
        description = finding["description"].lower()
        
        domain_keywords = {
            "production": ["production", "manufacturing", "assembly"],
            "quality": ["quality", "defect", "inspection"],
            "maintenance": ["maintenance", "equipment", "downtime"],
            "supply_chain": ["supply", "inventory", "supplier"],
            "cost": ["cost", "budget", "expense"],
            "safety": ["safety", "incident", "hazard"],
            "customer": ["customer", "satisfaction", "complaint"]
        }
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in description for keyword in keywords):
                domains.append(domain)
        
        return domains if domains else ["operations"]
    
    def _generate_insight_recommendations(self, insight: BusinessInsight) -> List[Dict[str, Any]]:
        """Generate specific recommendations for an insight."""
        recommendations = []
        
        # Base recommendation based on insight type
        if insight.type == InsightType.OPERATIONAL:
            recommendations.append({
                "type": RecommendationType.IMMEDIATE_ACTION,
                "title": "Implement Immediate Operational Fix",
                "description": f"Address the operational issue identified in {insight.title}",
                "rationale": f"Based on {insight.description}",
                "implementation": "Deploy rapid response team to implement immediate corrective actions",
                "resources": {"team_size": 3, "duration_days": 5, "budget": "low"},
                "outcomes": ["Immediate stabilization", "Reduced operational impact"],
                "metrics": ["Performance recovery time", "Impact reduction %"],
                "priority": 1,
                "timeline": "1-3 days",
                "risk_level": "low",
                "feasibility": 0.9
            })
        
        elif insight.type == InsightType.EFFICIENCY:
            recommendations.append({
                "type": RecommendationType.PROCESS_IMPROVEMENT,
                "title": "Process Optimization Initiative",
                "description": f"Optimize processes based on efficiency insights from {insight.title}",
                "rationale": f"Analysis shows significant efficiency improvement potential",
                "implementation": "Conduct detailed process mapping and implement lean improvements",
                "resources": {"team_size": 5, "duration_weeks": 8, "budget": "medium"},
                "outcomes": ["15-25% efficiency improvement", "Cost reduction", "Quality enhancement"],
                "metrics": ["OEE improvement", "Cost per unit", "Cycle time reduction"],
                "priority": 2,
                "timeline": "4-8 weeks",
                "risk_level": "medium",
                "feasibility": 0.8
            })
        
        elif insight.type == InsightType.STRATEGIC:
            recommendations.append({
                "type": RecommendationType.STRATEGIC_INITIATIVE,
                "title": "Strategic Business Initiative",
                "description": f"Launch strategic initiative based on {insight.title}",
                "rationale": f"Strategic analysis indicates significant business opportunity",
                "implementation": "Develop comprehensive business case and implementation roadmap",
                "resources": {"team_size": 10, "duration_months": 6, "budget": "high"},
                "outcomes": ["Competitive advantage", "Market position improvement", "Revenue growth"],
                "metrics": ["Market share", "Revenue impact", "ROI"],
                "priority": 1,
                "timeline": "3-6 months",
                "risk_level": "medium",
                "feasibility": 0.7
            })
        
        # Add monitoring recommendation for all insights
        recommendations.append({
            "type": RecommendationType.MONITORING,
            "title": "Establish Ongoing Monitoring",
            "description": f"Monitor key metrics related to {insight.title}",
            "rationale": "Ensure sustained improvement and early detection of issues",
            "implementation": "Set up automated dashboards and alerting systems",
            "resources": {"team_size": 2, "duration_weeks": 2, "budget": "low"},
            "outcomes": ["Proactive issue detection", "Performance visibility", "Trend analysis"],
            "metrics": ["Alert response time", "Trend accuracy", "Detection rate"],
            "priority": 3,
            "timeline": "1-2 weeks",
            "risk_level": "low",
            "feasibility": 0.95
        })
        
        return recommendations
    
    def _capture_organizational_learning(
        self, 
        insights: List[BusinessInsight], 
        findings: List[Dict[str, Any]]
    ) -> OrganizationalLearning:
        """Capture organizational learning from the investigation."""
        # Identify patterns across insights
        common_patterns = self._identify_common_patterns(insights)
        
        # Calculate success metrics
        avg_confidence = sum(insight.confidence for insight in insights) / len(insights) if insights else 0
        
        # Extract lessons learned
        lessons = self._extract_lessons_learned(insights, findings)
        
        return OrganizationalLearning(
            pattern_id=f"pattern_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            pattern_description=f"Investigation pattern identified across {len(insights)} insights",
            frequency=1,  # First occurrence
            success_rate=avg_confidence,
            business_value=min(sum(sum(insight.business_impact.values()) for insight in insights) / len(insights), 1.0),
            applicable_domains=list(set(domain for insight in insights for domain in insight.related_domains)),
            best_practices=self._extract_best_practices(insights),
            lessons_learned=lessons,
            improvement_opportunities=self._identify_improvement_opportunities(insights)
        )
    
    def _identify_common_patterns(self, insights: List[BusinessInsight]) -> List[str]:
        """Identify common patterns across insights."""
        patterns = []
        
        # Group by insight type
        type_counts = {}
        for insight in insights:
            type_counts[insight.type] = type_counts.get(insight.type, 0) + 1
        
        # Identify dominant patterns
        for insight_type, count in type_counts.items():
            if count > 1:
                patterns.append(f"Multiple {insight_type.value} insights identified")
        
        return patterns
    
    def _extract_lessons_learned(
        self, 
        insights: List[BusinessInsight], 
        findings: List[Dict[str, Any]]
    ) -> List[str]:
        """Extract key lessons learned from the investigation."""
        lessons = []
        
        # High-confidence insights provide reliable lessons
        high_confidence_insights = [i for i in insights if i.confidence > 0.8]
        if high_confidence_insights:
            lessons.append("High-confidence data analysis enables reliable business insights")
        
        # Cross-domain insights provide broader lessons
        multi_domain_insights = [i for i in insights if len(i.related_domains) > 1]
        if multi_domain_insights:
            lessons.append("Cross-functional analysis reveals interconnected business impacts")
        
        # Strategic insights provide long-term lessons
        strategic_insights = [i for i in insights if i.type == InsightType.STRATEGIC]
        if strategic_insights:
            lessons.append("Strategic analysis uncovers opportunities for competitive advantage")
        
        return lessons
    
    def _extract_best_practices(self, insights: List[BusinessInsight]) -> List[str]:
        """Extract best practices from successful insights."""
        practices = []
        
        # High actionability insights suggest good practices
        actionable_insights = [i for i in insights if i.actionability > 0.8]
        if actionable_insights:
            practices.append("Focus on specific, measurable findings for actionable insights")
        
        # Multiple stakeholder insights suggest collaboration
        multi_stakeholder_insights = [i for i in insights if len(i.stakeholders) > 2]
        if multi_stakeholder_insights:
            practices.append("Engage multiple stakeholders for comprehensive business impact")
        
        return practices
    
    def _identify_improvement_opportunities(self, insights: List[BusinessInsight]) -> List[str]:
        """Identify opportunities for improving future investigations."""
        opportunities = []
        
        # Low confidence insights suggest data quality improvements
        low_confidence_insights = [i for i in insights if i.confidence < 0.7]
        if low_confidence_insights:
            opportunities.append("Improve data quality and validation processes")
        
        # Limited strategic depth suggests analysis enhancement
        shallow_insights = [i for i in insights if i.strategic_depth < 0.5]
        if shallow_insights:
            opportunities.append("Enhance strategic analysis capabilities and business context")
        
        return opportunities
    
    def _calculate_business_impact(
        self, 
        insights: List[BusinessInsight], 
        recommendations: List[Recommendation]
    ) -> Dict[str, Any]:
        """Calculate overall business impact assessment."""
        impact_assessment = {
            "financial_potential": 0.0,
            "operational_improvement": 0.0,
            "strategic_value": 0.0,
            "risk_reduction": 0.0,
            "implementation_complexity": 0.0,
            "confidence_level": 0.0
        }
        
        if insights:
            # Aggregate impact across insights
            for insight in insights:
                weight = insight.confidence
                impact_assessment["financial_potential"] += insight.business_impact.get("financial_impact", 0) * weight
                impact_assessment["operational_improvement"] += insight.business_impact.get("operational_efficiency", 0) * weight
                impact_assessment["strategic_value"] += insight.business_impact.get("strategic_alignment", 0) * weight
                impact_assessment["risk_reduction"] += insight.business_impact.get("risk_mitigation", 0) * weight
                impact_assessment["confidence_level"] += insight.confidence
            
            # Average the values
            num_insights = len(insights)
            for key in impact_assessment:
                if key != "implementation_complexity":
                    impact_assessment[key] /= num_insights
        
        # Calculate implementation complexity from recommendations
        if recommendations:
            avg_complexity = sum(
                1.0 - rec.feasibility for rec in recommendations
            ) / len(recommendations)
            impact_assessment["implementation_complexity"] = avg_complexity
        
        return impact_assessment
    
    def _generate_executive_summary(
        self, 
        insights: List[BusinessInsight], 
        recommendations: List[Recommendation],
        impact_assessment: Dict[str, Any]
    ) -> str:
        """Generate executive summary of the investigation insights."""
        summary_parts = []
        
        # Opening statement
        summary_parts.append(
            f"Business intelligence analysis identified {len(insights)} key insights "
            f"with {len(recommendations)} actionable recommendations."
        )
        
        # Key insights summary
        if insights:
            strategic_insights = [i for i in insights if i.type == InsightType.STRATEGIC]
            operational_insights = [i for i in insights if i.type == InsightType.OPERATIONAL]
            
            if strategic_insights:
                summary_parts.append(
                    f"{len(strategic_insights)} strategic insights identified opportunities "
                    f"for competitive advantage and business growth."
                )
            
            if operational_insights:
                summary_parts.append(
                    f"{len(operational_insights)} operational insights highlighted "
                    f"immediate improvement opportunities."
                )
        
        # Business impact summary
        financial_impact = impact_assessment.get("financial_potential", 0)
        if financial_impact > 0.7:
            summary_parts.append("Analysis indicates significant financial impact potential.")
        elif financial_impact > 0.5:
            summary_parts.append("Analysis indicates moderate financial impact potential.")
        
        # Implementation summary
        high_priority_recs = [r for r in recommendations if r.priority <= 2]
        if high_priority_recs:
            summary_parts.append(
                f"{len(high_priority_recs)} high-priority recommendations "
                f"require immediate attention for maximum business impact."
            )
        
        # Confidence statement
        avg_confidence = impact_assessment.get("confidence_level", 0)
        if avg_confidence > 0.8:
            summary_parts.append("Findings are supported by high-quality data and analysis.")
        
        return " ".join(summary_parts)
    
    def _generate_stakeholder_communications(
        self, 
        insights: List[BusinessInsight], 
        recommendations: List[Recommendation],
        user_role: Optional[str]
    ) -> Dict[str, str]:
        """Generate role-specific stakeholder communications."""
        communications = {}
        
        for role, template in self._role_templates.items():
            # Skip if this is the requesting user's role (they get full report)
            if role == user_role:
                continue
            
            message = self._generate_role_specific_message(
                insights, recommendations, role, template
            )
            communications[role] = message
        
        return communications
    
    def _generate_role_specific_message(
        self,
        insights: List[BusinessInsight],
        recommendations: List[Recommendation], 
        role: str,
        template: Dict[str, str]
    ) -> str:
        """Generate a role-specific message."""
        message_parts = []
        
        # Role-specific opening
        if role == "executive":
            message_parts.append(
                f"Strategic analysis identified {len(insights)} business insights "
                f"with potential for competitive advantage."
            )
        elif role == "manager":
            message_parts.append(
                f"Operational analysis identified {len(insights)} actionable insights "
                f"for your team's consideration."
            )
        elif role == "analyst":
            message_parts.append(
                f"Comprehensive analysis generated {len(insights)} validated insights "
                f"with supporting evidence."
            )
        elif role == "engineer":
            message_parts.append(
                f"Technical analysis identified {len(insights)} insights "
                f"requiring engineering implementation."
            )
        
        # Key recommendations for this role
        role_relevant_recs = [
            r for r in recommendations 
            if role.lower() in r.description.lower() or 
               role.lower() in " ".join(r.expected_outcomes).lower()
        ]
        
        if role_relevant_recs:
            message_parts.append(
                f"{len(role_relevant_recs)} recommendations specifically "
                f"relevant to {role} responsibilities."
            )
        
        # Role-specific focus areas
        focus = template["focus"]
        message_parts.append(f"Analysis focuses on {focus}.")
        
        return " ".join(message_parts)
    
    def _define_success_criteria(
        self, 
        insights: List[BusinessInsight], 
        recommendations: List[Recommendation]
    ) -> List[str]:
        """Define success criteria for the insights and recommendations."""
        criteria = []
        
        # Insight-based criteria
        for insight in insights:
            if insight.business_impact.get("financial_impact", 0) > 0.7:
                criteria.append("Achieve measurable financial impact within target timeframe")
            if insight.business_impact.get("operational_efficiency", 0) > 0.7:
                criteria.append("Demonstrate operational efficiency improvements")
            if insight.type == InsightType.STRATEGIC:
                criteria.append("Advance strategic business objectives")
        
        # Recommendation-based criteria
        immediate_recs = [r for r in recommendations if r.type == RecommendationType.IMMEDIATE_ACTION]
        if immediate_recs:
            criteria.append("Implement immediate actions within specified timeline")
        
        strategic_recs = [r for r in recommendations if r.type == RecommendationType.STRATEGIC_INITIATIVE]
        if strategic_recs:
            criteria.append("Establish strategic initiative roadmap and governance")
        
        # General criteria
        criteria.extend([
            "Stakeholder engagement and buy-in achieved",
            "Success metrics established and monitored",
            "Organizational learning captured and applied"
        ])
        
        return criteria
    
    def _generate_follow_up_actions(self, recommendations: List[Recommendation]) -> List[str]:
        """Generate specific follow-up actions."""
        actions = []
        
        # Immediate actions
        immediate_recs = [r for r in recommendations if r.type == RecommendationType.IMMEDIATE_ACTION]
        for rec in immediate_recs:
            actions.append(f"Execute immediate action: {rec.title}")
        
        # Planning actions
        strategic_recs = [r for r in recommendations if r.type == RecommendationType.STRATEGIC_INITIATIVE]
        for rec in strategic_recs:
            actions.append(f"Develop implementation plan for: {rec.title}")
        
        # Monitoring actions
        actions.extend([
            "Establish monitoring dashboard for key metrics",
            "Schedule progress review meetings with stakeholders",
            "Create communication plan for ongoing updates"
        ])
        
        return actions
    
    def _create_synthesis_metadata(
        self,
        investigation_results: Dict[str, Any],
        business_context: Optional[Dict[str, Any]],
        output_format: OutputFormat
    ) -> Dict[str, Any]:
        """Create metadata for the synthesis process."""
        return {
            "synthesis_timestamp": datetime.now(timezone.utc).isoformat(),
            "source_investigation": investigation_results.get("investigation_id", "unknown"),
            "business_context_provided": business_context is not None,
            "output_format": output_format.value,
            "synthesis_version": "1.0.0",
            "quality_metrics": {
                "insight_confidence_threshold": settings.insight_confidence_threshold,
                "business_relevance_threshold": settings.business_relevance_threshold,
                "actionability_threshold": settings.actionability_threshold
            }
        }


# Standalone execution for testing
if __name__ == "__main__":
    import asyncio
    
    synthesizer = InsightSynthesizer()
    
    # Mock investigation results
    mock_results = {
        "investigation_id": "test_001",
        "step_results": {
            "schema_analysis": {
                "key_findings": ["Production efficiency declined 15% in Line 2"],
                "confidence": 0.85,
                "supporting_evidence": ["Equipment logs", "Performance metrics"],
                "data_quality": 0.9
            },
            "pattern_discovery": {
                "key_findings": ["Maintenance schedule correlation identified"],
                "confidence": 0.78,
                "supporting_evidence": ["Maintenance records", "Downtime correlation"],
                "data_quality": 0.8
            }
        },
        "summary": {
            "conclusions": ["Equipment maintenance timing impacts production efficiency"],
            "overall_confidence": 0.82,
            "supporting_data": ["Historical maintenance data", "Production metrics"],
            "data_reliability": 0.85
        }
    }
    
    async def test_synthesis():
        result = await synthesizer.synthesize_insights(
            investigation_results=mock_results,
            business_context={"current_initiative": "Operational Excellence"},
            user_role="manager",
            output_format=OutputFormat.DETAILED_REPORT
        )
        
        print("Insight Synthesis Test")
        print("=" * 50)
        print(f"Generated {len(result.insights)} insights")
        print(f"Generated {len(result.recommendations)} recommendations")
        print(f"Executive Summary: {result.executive_summary}")
        
        if result.insights:
            print(f"\nFirst Insight: {result.insights[0].title}")
            print(f"Confidence: {result.insights[0].confidence:.3f}")
            print(f"Business Impact: {result.insights[0].business_impact}")
        
        if result.recommendations:
            print(f"\nFirst Recommendation: {result.recommendations[0].title}")
            print(f"Priority: {result.recommendations[0].priority}")
            print(f"Feasibility: {result.recommendations[0].feasibility:.3f}")
    
    asyncio.run(test_synthesis())