"""
Session Management API Routes

Handles user sessions, context, and memory management.
Provides endpoints for storing and retrieving investigation context.
"""

from datetime import datetime
from typing import Dict, Any, Optional, List
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from ...database.memory_manager import MemoryManager
from ..app_factory import get_database_manager


router = APIRouter()


class SessionRequest(BaseModel):
    """Request to create or update a session."""
    user_id: str
    context: Dict[str, Any] = {}


class SessionResponse(BaseModel):
    """Session information."""
    session_id: str
    user_id: str
    context: Dict[str, Any]
    created_at: datetime
    updated_at: datetime


class ContextUpdate(BaseModel):
    """Update session context."""
    context: Dict[str, Any]


async def get_memory_manager() -> MemoryManager:
    """Dependency to get memory manager."""
    db_manager = get_database_manager()
    if not db_manager:
        raise HTTPException(status_code=500, detail="Database not available")
    return MemoryManager(db_manager)


@router.post("/", response_model=SessionResponse)
async def create_session(
    request: SessionRequest,
    memory: MemoryManager = Depends(get_memory_manager)
):
    """
    Create a new user session.
    
    Sessions store investigation context and user preferences.
    """
    session_id = str(uuid4())
    
    session_data = {
        "session_id": session_id,
        "user_id": request.user_id,
        "context": request.context,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }
    
    await memory.create_session(session_data)
    
    return SessionResponse(**session_data)


@router.get("/{session_id}", response_model=SessionResponse)
async def get_session(
    session_id: str,
    memory: MemoryManager = Depends(get_memory_manager)
):
    """Get session information and context."""
    session_data = await memory.get_session(session_id)
    
    if not session_data:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return SessionResponse(**session_data)


@router.put("/{session_id}/context", response_model=SessionResponse)
async def update_session_context(
    session_id: str,
    update: ContextUpdate,
    memory: MemoryManager = Depends(get_memory_manager)
):
    """
    Update session context.
    
    Used to store investigation preferences, filters, and user state.
    """
    session_data = await memory.update_session_context(
        session_id, 
        update.context
    )
    
    if not session_data:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return SessionResponse(**session_data)


@router.get("/{session_id}/history", response_model=List[dict])
async def get_session_history(
    session_id: str,
    limit: int = 50,
    memory: MemoryManager = Depends(get_memory_manager)
):
    """
    Get investigation history for a session.
    
    Returns recent queries and results for context.
    """
    history = await memory.get_session_history(session_id, limit)
    return history


@router.post("/{session_id}/cache", response_model=dict)
async def cache_result(
    session_id: str,
    key: str,
    data: Dict[str, Any],
    ttl: Optional[int] = 1800,  # 30 minutes default
    memory: MemoryManager = Depends(get_memory_manager)
):
    """
    Cache investigation results.
    
    Stores query results and analysis for quick retrieval.
    """
    await memory.cache_result(session_id, key, data, ttl)
    
    return {"message": "Result cached successfully", "key": key}


@router.get("/{session_id}/cache/{key}", response_model=Dict[str, Any])
async def get_cached_result(
    session_id: str,
    key: str,
    memory: MemoryManager = Depends(get_memory_manager)
):
    """Get cached investigation result."""
    cached_data = await memory.get_cached_result(session_id, key)
    
    if cached_data is None:
        raise HTTPException(status_code=404, detail="Cached result not found")
    
    return cached_data


@router.delete("/{session_id}")
async def delete_session(
    session_id: str,
    memory: MemoryManager = Depends(get_memory_manager)
):
    """Delete session and all associated data."""
    success = await memory.delete_session(session_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {"message": "Session deleted successfully"}