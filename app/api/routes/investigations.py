"""
Investigation Management API Routes

Handles autonomous SQL investigation lifecycle:
- Starting new investigations
- Getting investigation status
- Managing investigation results
"""

from datetime import datetime
from typing import List, Optional
from uuid import uuid4

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel

from ...core.investigation import SimpleInvestigationEngine
from ...database.models import Investigation, InvestigationStatus
from ..app_factory import get_database_manager


router = APIRouter()


class InvestigationRequest(BaseModel):
    """Request to start a new investigation."""
    query: str
    user_id: Optional[str] = None
    context: Optional[dict] = None


class InvestigationResponse(BaseModel):
    """Response containing investigation details."""
    id: str
    query: str
    status: InvestigationStatus
    created_at: datetime
    updated_at: datetime
    progress: Optional[dict] = None
    results: Optional[dict] = None
    error: Optional[str] = None


@router.post("/", response_model=InvestigationResponse)
async def start_investigation(
    request: InvestigationRequest,
    background_tasks: BackgroundTasks
):
    """
    Start a new autonomous SQL investigation.
    
    The agent will analyze the query and execute an investigation plan
    autonomously, streaming progress updates via WebSocket.
    """
    # Generate unique investigation ID
    investigation_id = str(uuid4())
    
    # Create investigation record
    investigation = Investigation(
        id=investigation_id,
        query=request.query,
        user_id=request.user_id,
        status=InvestigationStatus.PENDING,
        context=request.context or {},
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )
    
    # Save to database
    db_manager = get_database_manager()
    if not db_manager:
        raise HTTPException(status_code=500, detail="Database not available")
    
    await db_manager.save_investigation(investigation)
    
    # Start investigation in background
    background_tasks.add_task(
        run_investigation_background,
        investigation_id,
        request.query,
        request.context or {}
    )
    
    return InvestigationResponse(
        id=investigation.id,
        query=investigation.query,
        status=investigation.status,
        created_at=investigation.created_at,
        updated_at=investigation.updated_at
    )


@router.get("/{investigation_id}", response_model=InvestigationResponse)
async def get_investigation(investigation_id: str):
    """Get investigation status and results."""
    db_manager = get_database_manager()
    if not db_manager:
        raise HTTPException(status_code=500, detail="Database not available")
    
    investigation = await db_manager.get_investigation(investigation_id)
    if not investigation:
        raise HTTPException(status_code=404, detail="Investigation not found")
    
    return InvestigationResponse(
        id=investigation.id,
        query=investigation.query,
        status=investigation.status,
        created_at=investigation.created_at,
        updated_at=investigation.updated_at,
        progress=investigation.progress,
        results=investigation.results,
        error=investigation.error
    )


@router.get("/", response_model=List[InvestigationResponse])
async def list_investigations(
    user_id: Optional[str] = None,
    limit: int = 20,
    offset: int = 0
):
    """List recent investigations for a user."""
    db_manager = get_database_manager()
    if not db_manager:
        raise HTTPException(status_code=500, detail="Database not available")
    
    investigations = await db_manager.list_investigations(
        user_id=user_id,
        limit=limit,
        offset=offset
    )
    
    return [
        InvestigationResponse(
            id=inv.id,
            query=inv.query,
            status=inv.status,
            created_at=inv.created_at,
            updated_at=inv.updated_at,
            progress=inv.progress,
            results=inv.results,
            error=inv.error
        )
        for inv in investigations
    ]


@router.delete("/{investigation_id}")
async def cancel_investigation(investigation_id: str):
    """Cancel a running investigation."""
    db_manager = get_database_manager()
    if not db_manager:
        raise HTTPException(status_code=500, detail="Database not available")
    
    success = await db_manager.cancel_investigation(investigation_id)
    if not success:
        raise HTTPException(status_code=404, detail="Investigation not found")
    
    return {"message": "Investigation cancelled"}


async def run_investigation_background(
    investigation_id: str,
    query: str,
    context: dict
):
    """
    Run investigation in background using the autonomous agent.
    
    This function will be implemented to use the MCP client
    to call database tools autonomously.
    """
    try:
        # Initialize investigation engine
        engine = SimpleInvestigationEngine(investigation_id)
        
        # Run autonomous investigation
        await engine.investigate(query, context)
        
    except Exception as e:
        # Update investigation with error
        db_manager = get_database_manager()
        if db_manager:
            await db_manager.update_investigation_error(investigation_id, str(e))