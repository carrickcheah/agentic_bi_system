"""
Business Intelligence API Routes

REST endpoints for business intelligence capabilities including domain analysis,
complexity assessment, and methodology recommendations.
"""

from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ..dependencies import get_current_user, get_organization_context, require_read
from ...intelligence.domain_expert import BusinessDomainExpert
from ...intelligence.complexity_analyzer import ComplexityAnalyzer
from ...utils.logging import logger

router = APIRouter(prefix="/intelligence", tags=["business-intelligence"])


class DomainAnalysisRequest(BaseModel):
    business_question: str
    organization_context: Optional[Dict[str, Any]] = None


@router.post("/analyze-domain")
async def analyze_business_domain(
    request: DomainAnalysisRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    organization_context: Dict[str, Any] = Depends(get_organization_context),
    _: bool = Depends(require_read)
):
    """Analyze business domain and recommend investigation methodology."""
    try:
        domain_expert = BusinessDomainExpert()
        
        # Create semantic intent (simplified)
        semantic_intent = {
            "original_question": request.business_question,
            "business_domain": "general",  # Would be classified
            "complexity_indicators": {"indicators": {}}
        }
        
        analysis = domain_expert.analyze_business_domain(
            semantic_intent, organization_context
        )
        
        return {"domain_analysis": analysis}
        
    except Exception as e:
        logger.error(f"Domain analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze-complexity")
async def analyze_question_complexity(
    request: DomainAnalysisRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    organization_context: Dict[str, Any] = Depends(get_organization_context),
    _: bool = Depends(require_read)
):
    """Analyze complexity of business question."""
    try:
        complexity_analyzer = ComplexityAnalyzer()
        
        semantic_intent = {
            "original_question": request.business_question,
            "business_domain": "general",
            "business_intent": {"question_type": "unknown"},
            "complexity_indicators": {"indicators": {}}
        }
        
        analysis = await complexity_analyzer.analyze_complexity(
            semantic_intent, organization_context
        )
        
        return {"complexity_analysis": analysis}
        
    except Exception as e:
        logger.error(f"Complexity analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))