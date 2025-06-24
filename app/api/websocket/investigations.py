"""
Investigation WebSocket Endpoints

Real-time WebSocket connections for live investigation progress,
results streaming, and interactive investigation sessions.
"""

from typing import Dict, Any
from fastapi import WebSocket, WebSocketDisconnect, Depends
from fastapi.routing import APIRouter
import json
import asyncio

from ..dependencies import get_current_user_ws
from ...utils.logging import logger

router = APIRouter()


class InvestigationConnectionManager:
    """Manages WebSocket connections for investigations."""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.investigation_sessions: Dict[str, set] = {}
    
    async def connect(self, websocket: WebSocket, investigation_id: str, user_id: str):
        """Connect user to investigation session."""
        await websocket.accept()
        connection_id = f"{user_id}_{investigation_id}"
        self.active_connections[connection_id] = websocket
        
        if investigation_id not in self.investigation_sessions:
            self.investigation_sessions[investigation_id] = set()
        self.investigation_sessions[investigation_id].add(connection_id)
        
        logger.info(f"User {user_id} connected to investigation {investigation_id}")
    
    def disconnect(self, investigation_id: str, user_id: str):
        """Disconnect user from investigation session."""
        connection_id = f"{user_id}_{investigation_id}"
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]
        
        if investigation_id in self.investigation_sessions:
            self.investigation_sessions[investigation_id].discard(connection_id)
            if not self.investigation_sessions[investigation_id]:
                del self.investigation_sessions[investigation_id]
        
        logger.info(f"User {user_id} disconnected from investigation {investigation_id}")
    
    async def broadcast_to_investigation(self, investigation_id: str, message: dict):
        """Broadcast message to all users in investigation session."""
        if investigation_id not in self.investigation_sessions:
            return
        
        disconnected = []
        for connection_id in self.investigation_sessions[investigation_id]:
            if connection_id in self.active_connections:
                try:
                    await self.active_connections[connection_id].send_text(json.dumps(message))
                except:
                    disconnected.append(connection_id)
        
        # Clean up disconnected connections
        for connection_id in disconnected:
            user_id = connection_id.split('_')[0]
            self.disconnect(investigation_id, user_id)


manager = InvestigationConnectionManager()


@router.websocket("/ws/investigations/{investigation_id}")
async def investigation_websocket(
    websocket: WebSocket,
    investigation_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user_ws)
):
    """WebSocket endpoint for real-time investigation updates."""
    user_id = current_user.get("user_id", "anonymous")
    
    await manager.connect(websocket, investigation_id, user_id)
    
    try:
        # Send initial connection confirmation
        await websocket.send_text(json.dumps({
            "type": "connection_established",
            "investigation_id": investigation_id,
            "user_id": user_id,
            "timestamp": "2024-06-24T10:30:00Z"
        }))
        
        while True:
            # Receive messages from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Echo message back to all connected users
            await manager.broadcast_to_investigation(investigation_id, {
                "type": "user_message",
                "user_id": user_id,
                "message": message,
                "timestamp": "2024-06-24T10:30:00Z"
            })
            
    except WebSocketDisconnect:
        manager.disconnect(investigation_id, user_id)
    except Exception as e:
        logger.error(f"WebSocket error for investigation {investigation_id}: {e}")
        manager.disconnect(investigation_id, user_id)