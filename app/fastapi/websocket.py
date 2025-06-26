"""
WebSocket Handler for Real-Time Investigation Updates

Provides real-time streaming of investigation progress to frontend clients.
"""

import json
import asyncio
from typing import Dict, Set
from uuid import UUID

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.websockets import WebSocketState

from ..utils.logging import logger


router = APIRouter()


class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""
    
    def __init__(self):
        # Map investigation_id -> set of websockets
        self.active_connections: Dict[str, Set[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, investigation_id: str):
        """Accept WebSocket connection and subscribe to investigation."""
        await websocket.accept()
        
        if investigation_id not in self.active_connections:
            self.active_connections[investigation_id] = set()
        
        self.active_connections[investigation_id].add(websocket)
        logger.info(f"WebSocket connected for investigation {investigation_id}")
    
    def disconnect(self, websocket: WebSocket, investigation_id: str):
        """Remove WebSocket connection."""
        if investigation_id in self.active_connections:
            self.active_connections[investigation_id].discard(websocket)
            
            # Clean up empty investigation connections
            if not self.active_connections[investigation_id]:
                del self.active_connections[investigation_id]
        
        logger.info(f"WebSocket disconnected for investigation {investigation_id}")
    
    async def send_investigation_update(
        self, 
        investigation_id: str, 
        message: dict
    ):
        """Send update to all clients subscribed to an investigation."""
        if investigation_id not in self.active_connections:
            return
        
        # Prepare message
        message_data = json.dumps(message)
        
        # Send to all connected clients
        disconnected_websockets = set()
        
        for websocket in self.active_connections[investigation_id]:
            try:
                if websocket.client_state == WebSocketState.CONNECTED:
                    await websocket.send_text(message_data)
                else:
                    disconnected_websockets.add(websocket)
            except Exception as e:
                logger.warning(f"Failed to send WebSocket message: {e}")
                disconnected_websockets.add(websocket)
        
        # Clean up disconnected websockets
        for websocket in disconnected_websockets:
            self.active_connections[investigation_id].discard(websocket)
    
    async def broadcast_global_message(self, message: dict):
        """Send message to all connected clients."""
        message_data = json.dumps(message)
        
        for investigation_connections in self.active_connections.values():
            for websocket in investigation_connections.copy():
                try:
                    if websocket.client_state == WebSocketState.CONNECTED:
                        await websocket.send_text(message_data)
                except Exception as e:
                    logger.warning(f"Failed to broadcast message: {e}")


# Global connection manager instance
manager = ConnectionManager()


@router.websocket("/ws/{investigation_id}")
async def websocket_endpoint(websocket: WebSocket, investigation_id: str):
    """
    WebSocket endpoint for real-time investigation updates.
    
    Clients connect to receive progress updates, results, and status changes
    for a specific investigation.
    """
    try:
        # Validate investigation ID format
        try:
            UUID(investigation_id)
        except ValueError:
            await websocket.close(code=1000, reason="Invalid investigation ID")
            return
        
        await manager.connect(websocket, investigation_id)
        
        # Send initial connection confirmation
        await websocket.send_text(json.dumps({
            "type": "connection",
            "message": "Connected to investigation updates",
            "investigation_id": investigation_id
        }))
        
        # Keep connection alive and handle client messages
        while True:
            try:
                # Wait for client messages (e.g., ping, status requests)
                data = await websocket.receive_text()
                message = json.loads(data)
                
                # Handle client requests
                if message.get("type") == "ping":
                    await websocket.send_text(json.dumps({
                        "type": "pong",
                        "timestamp": message.get("timestamp")
                    }))
                
                elif message.get("type") == "status_request":
                    # Client requesting current investigation status
                    await websocket.send_text(json.dumps({
                        "type": "status_response",
                        "investigation_id": investigation_id,
                        "message": "Status request received"
                    }))
                
            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Invalid JSON format"
                }))
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                break
    
    except Exception as e:
        logger.error(f"WebSocket connection error: {e}")
    
    finally:
        manager.disconnect(websocket, investigation_id)


async def send_investigation_progress(
    investigation_id: str,
    step: str,
    progress: float,
    details: dict = None
):
    """
    Send progress update for an investigation.
    
    Called by the investigation engine to update connected clients.
    """
    message = {
        "type": "progress",
        "investigation_id": investigation_id,
        "step": step,
        "progress": progress,
        "details": details or {},
        "timestamp": asyncio.get_event_loop().time()
    }
    
    await manager.send_investigation_update(investigation_id, message)


async def send_investigation_result(
    investigation_id: str,
    result: dict
):
    """Send final investigation result to connected clients."""
    message = {
        "type": "result",
        "investigation_id": investigation_id,
        "result": result,
        "timestamp": asyncio.get_event_loop().time()
    }
    
    await manager.send_investigation_update(investigation_id, message)


async def send_investigation_error(
    investigation_id: str,
    error: str
):
    """Send error notification to connected clients."""
    message = {
        "type": "error",
        "investigation_id": investigation_id,
        "error": error,
        "timestamp": asyncio.get_event_loop().time()
    }
    
    await manager.send_investigation_update(investigation_id, message)