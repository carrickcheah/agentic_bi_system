"""
Investigations API Routes

REST endpoints for autonomous business intelligence investigations.
Migrated and expanded from database.py with full investigation workflow support.
"""

from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, status
from pydantic import BaseModel, Field

from ..dependencies import (
    get_current_user, get_organization_context, require_read, require_write
)
from ...core.business_analyst import AutonomousBusinessAnalyst
from ...utils.logging import logger

router = APIRouter(prefix="/investigations", tags=["investigations"])


# Pydantic models
class InvestigationRequest(BaseModel):
    business_question: str = Field(..., description="Natural language business question")
    business_domain: Optional[str] = Field("general", description="Business domain classification")
    priority: Optional[str] = Field("standard", description="Investigation priority level")
    stream_progress: bool = Field(True, description="Whether to stream real-time progress")


class InvestigationResponse(BaseModel):
    investigation_id: str
    status: str
    business_question: str
    insights: Dict[str, Any]
    metadata: Dict[str, Any]


class InvestigationStatus(BaseModel):
    investigation_id: str
    status: str
    progress_percentage: float
    current_phase: str
    estimated_completion: Optional[str]


@router.post("/start", response_model=InvestigationResponse)
async def start_investigation(
    request: InvestigationRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(get_current_user),
    organization_context: Dict[str, Any] = Depends(get_organization_context),
    _: bool = Depends(require_read)
):
    """
    Start a new autonomous business intelligence investigation.
    
    This endpoint initiates the five-phase investigation workflow:
    1. Query Processing - Natural language to business intent
    2. Strategy Planning - Investigation methodology selection
    3. Service Orchestration - Database service coordination
    4. Investigation Execution - Autonomous multi-step analysis
    5. Insight Synthesis - Strategic recommendations generation
    """
    try:
        logger.info(f"🚀 Starting investigation: {request.business_question[:100]}...")
        
        # Create business analyst instance
        analyst = AutonomousBusinessAnalyst()
        await analyst.initialize()
        
        # Start investigation
        investigation_results = []
        async for result in analyst.conduct_investigation(
            business_question=request.business_question,
            user_context=current_user,
            organization_context=organization_context,
            stream_progress=request.stream_progress
        ):
            investigation_results.append(result)
            
            # Return final result
            if result.get("type") == "investigation_completed":
                return InvestigationResponse(
                    investigation_id=result.get("investigation_id"),
                    status="completed",
                    business_question=request.business_question,
                    insights=result.get("insights", {}),
                    metadata=result.get("metadata", {})
                )
        
        # If no completion result found
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Investigation did not complete successfully"
        )
        
    except Exception as e:
        logger.error(f"Investigation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Investigation failed: {str(e)}"
        )


@router.get("/status/{investigation_id}", response_model=InvestigationStatus)
async def get_investigation_status(
    investigation_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    _: bool = Depends(require_read)
):
    """Get current status of an ongoing investigation."""
    try:
        analyst = AutonomousBusinessAnalyst()
        await analyst.initialize()
        
        status_info = await analyst.get_investigation_status(investigation_id)
        
        return InvestigationStatus(
            investigation_id=investigation_id,
            status=status_info.get("status", "unknown"),
            progress_percentage=status_info.get("progress_percentage", 0.0),
            current_phase=status_info.get("current_phase", "unknown"),
            estimated_completion=status_info.get("estimated_completion")
        )
        
    except Exception as e:
        logger.error(f"Failed to get investigation status: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Investigation {investigation_id} not found"
        )


@router.get("/history")
async def get_investigation_history(
    limit: int = 50,
    offset: int = 0,
    business_domain: Optional[str] = None,
    current_user: Dict[str, Any] = Depends(get_current_user),
    organization_context: Dict[str, Any] = Depends(get_organization_context),
    _: bool = Depends(require_read)
):
    """Get investigation history for the organization."""
    try:
        analyst = AutonomousBusinessAnalyst()
        await analyst.initialize()
        
        # Get organizational insights (this would include history)
        insights = await analyst.get_organizational_insights(
            organization_id=organization_context.get("organization_id"),
            business_domain=business_domain
        )
        
        return {
            "investigations": [],  # Would be populated from organizational memory
            "total_count": 0,
            "organizational_insights": insights
        }
        
    except Exception as e:
        logger.error(f"Failed to get investigation history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve investigation history"
        )


@router.post("/collaborate/{investigation_id}")
async def collaborate_on_investigation(
    investigation_id: str,
    expert_feedback: Dict[str, Any],
    current_user: Dict[str, Any] = Depends(get_current_user),
    _: bool = Depends(require_write)
):
    """Enable real-time collaboration on an investigation."""
    try:
        analyst = AutonomousBusinessAnalyst()
        await analyst.initialize()
        
        collaboration_result = await analyst.collaborate_on_investigation(
            investigation_id=investigation_id,
            user_context=current_user,
            expert_feedback=expert_feedback
        )
        
        return {
            "collaboration_id": f"collab_{investigation_id}",
            "status": "feedback_processed",
            "investigation_id": investigation_id,
            "result": collaboration_result
        }
        
    except Exception as e:
        logger.error(f"Collaboration failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process collaboration request"
        )


@router.delete("/{investigation_id}")
async def cancel_investigation(
    investigation_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    _: bool = Depends(require_write)
):
    """Cancel an ongoing investigation."""
    try:
        # Implementation would cancel the investigation
        return {
            "investigation_id": investigation_id,
            "status": "cancelled",
            "cancelled_by": current_user.get("user_id"),
            "cancelled_at": "2024-01-01T00:00:00Z"  # Would use actual timestamp
        }
        
    except Exception as e:
        logger.error(f"Failed to cancel investigation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to cancel investigation"
        )