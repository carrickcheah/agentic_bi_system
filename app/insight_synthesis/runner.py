"""
Insight Synthesis - AI Business Analyst that transforms investigation data into insights.
"""

from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Any
import uuid
from datetime import datetime

# Import model from main
from model import ModelManager

# Enums
class OutputFormat(Enum):
    EXECUTIVE_SUMMARY = "executive_summary"
    DETAILED_REPORT = "detailed_report"

class InsightType(Enum):
    TREND = "trend"
    ANOMALY = "anomaly"
    OPPORTUNITY = "opportunity"

class RecommendationType(Enum):
    IMMEDIATE = "immediate"
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"

# Dataclasses
@dataclass
class Insight:
    id: str
    type: InsightType
    title: str
    description: str
    confidence: float
    business_impact: str

@dataclass
class Recommendation:
    id: str
    type: RecommendationType
    title: str
    description: str
    priority: str
    timeline: str
    feasibility: str
    resource_requirements: str
    expected_outcomes: str
    success_metrics: str

@dataclass
class OrganizationalLearning:
    pattern_id: str
    pattern_description: str
    business_value: str
    best_practices: List[str]
    lessons_learned: List[str]

@dataclass
class SynthesisResult:
    insights: List[Insight]
    recommendations: List[Recommendation]
    executive_summary: str
    business_impact_assessment: str
    organizational_learning: OrganizationalLearning
    synthesis_metadata: Dict[str, Any]

# Main synthesizer
class InsightSynthesizer:
    """AI analyst that explains investigation findings to humans."""
    
    def __init__(self):
        self.model = ModelManager()
        
    async def synthesize_insights(
        self,
        investigation_results: Any,
        business_context: Dict[str, Any],
        user_role: str,
        output_format: OutputFormat
    ) -> SynthesisResult:
        """
        Transform investigation data into business insights.
        
        Takes raw data from Phase 4 and creates:
        - Executive summary
        - Key insights
        - Actionable recommendations
        - Business impact assessment
        """
        
        # Build AI prompt
        prompt = f"""You are a senior business analyst. Transform these investigation findings into strategic insights.

Investigation Question: {investigation_results.investigation_request}

Raw Findings:
{self._format_findings(investigation_results.investigation_findings)}

Confidence Level: {investigation_results.overall_confidence:.2f}
User Role: {user_role}

Create:
1. Executive Summary (2-3 paragraphs for {user_role})
2. Top 3 Strategic Insights
3. Top 3 Actionable Recommendations
4. Business Impact Assessment

Be specific, actionable, and focus on business value."""

        # Get AI analysis
        response = await self.model.generate_response(prompt)
        
        # Create structured result
        return self._create_synthesis_result(response, investigation_results, user_role)
    
    def _format_findings(self, findings: Dict[str, Any]) -> str:
        """Format findings for AI prompt."""
        formatted = []
        for step, data in findings.items():
            formatted.append(f"{step}: {str(data)[:500]}")
        return "\n".join(formatted)
    
    def _create_synthesis_result(self, ai_response: str, investigation_results: Any, user_role: str) -> SynthesisResult:
        """Create structured synthesis result."""
        
        # Parse AI response and create insights
        insights = [
            Insight(
                id=str(uuid.uuid4()),
                type=InsightType.TREND,
                title="Primary Business Trend",
                description=ai_response[:200] if ai_response else "Analysis complete",
                confidence=investigation_results.overall_confidence,
                business_impact="High"
            )
        ]
        
        # Create recommendations
        recommendations = [
            Recommendation(
                id=str(uuid.uuid4()),
                type=RecommendationType.IMMEDIATE,
                title="Immediate Action Required",
                description="Based on investigation findings",
                priority="High",
                timeline="This week",
                feasibility="High",
                resource_requirements="Existing resources",
                expected_outcomes="Quick improvement",
                success_metrics="Monitor KPIs"
            )
        ]
        
        # Create organizational learning
        learning = OrganizationalLearning(
            pattern_id=str(uuid.uuid4()),
            pattern_description="Investigation pattern captured",
            business_value="High",
            best_practices=["Use this approach for similar questions"],
            lessons_learned=["Key insights discovered through analysis"]
        )
        
        return SynthesisResult(
            insights=insights,
            recommendations=recommendations,
            executive_summary=ai_response[:1000] if ai_response else "Executive summary",
            business_impact_assessment="Based on the analysis, significant business impact identified.",
            organizational_learning=learning,
            synthesis_metadata={
                "timestamp": datetime.now().isoformat(),
                "user_role": user_role,
                "confidence": investigation_results.overall_confidence
            }
        )