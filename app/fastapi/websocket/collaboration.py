"""
Collaboration WebSocket Endpoints

Real-time WebSocket connections for team collaboration including
shared workspaces, live editing, and team communication.
"""

from typing import Dict, Any, Set
from fastapi import WebSocket, WebSocketDisconnect, Depends
from fastapi.routing import APIRouter
import json
import asyncio

from ..dependencies import get_current_user_ws
from ...utils.logging import logger

router = APIRouter()


class CollaborationConnectionManager:
    """Manages WebSocket connections for team collaboration."""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.team_sessions: Dict[str, Set[str]] = {}
        self.user_teams: Dict[str, str] = {}
    
    async def connect(self, websocket: WebSocket, team_id: str, user_id: str):
        """Connect user to team collaboration session."""
        await websocket.accept()
        connection_id = f"{user_id}_{team_id}"
        self.active_connections[connection_id] = websocket
        self.user_teams[user_id] = team_id
        
        if team_id not in self.team_sessions:
            self.team_sessions[team_id] = set()
        self.team_sessions[team_id].add(connection_id)
        
        logger.info(f"User {user_id} joined team {team_id} collaboration")
        
        # Notify team members of new user
        await self.broadcast_to_team(team_id, {
            "type": "user_joined",
            "user_id": user_id,
            "team_id": team_id,
            "timestamp": "2024-06-24T10:30:00Z"
        }, exclude_user=user_id)
    
    def disconnect(self, team_id: str, user_id: str):
        """Disconnect user from team collaboration session."""
        connection_id = f"{user_id}_{team_id}"
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]
        
        if user_id in self.user_teams:
            del self.user_teams[user_id]
        
        if team_id in self.team_sessions:
            self.team_sessions[team_id].discard(connection_id)
            if not self.team_sessions[team_id]:
                del self.team_sessions[team_id]
        
        logger.info(f"User {user_id} left team {team_id} collaboration")
    
    async def broadcast_to_team(self, team_id: str, message: dict, exclude_user: str = None):
        """Broadcast message to all team members."""
        if team_id not in self.team_sessions:
            return
        
        disconnected = []
        for connection_id in self.team_sessions[team_id]:
            user_id = connection_id.split('_')[0]
            if exclude_user and user_id == exclude_user:
                continue
                
            if connection_id in self.active_connections:
                try:
                    await self.active_connections[connection_id].send_text(json.dumps(message))
                except:
                    disconnected.append(connection_id)
        
        # Clean up disconnected connections
        for connection_id in disconnected:
            user_id = connection_id.split('_')[0]
            self.disconnect(team_id, user_id)
    
    def get_team_members(self, team_id: str) -> list:
        """Get list of active team members."""
        if team_id not in self.team_sessions:
            return []
        
        return [conn_id.split('_')[0] for conn_id in self.team_sessions[team_id]]


manager = CollaborationConnectionManager()


@router.websocket("/ws/collaboration/{team_id}")
async def collaboration_websocket(
    websocket: WebSocket,
    team_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user_ws)
):
    """WebSocket endpoint for real-time team collaboration."""
    user_id = current_user.get("user_id", "anonymous")
    
    await manager.connect(websocket, team_id, user_id)
    
    try:
        # Send initial team state
        team_members = manager.get_team_members(team_id)
        await websocket.send_text(json.dumps({
            "type": "team_state",
            "team_id": team_id,
            "active_members": team_members,
            "user_id": user_id,
            "timestamp": "2024-06-24T10:30:00Z"
        }))
        
        while True:
            # Receive messages from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Handle different message types
            if message.get("type") == "cursor_position":
                await manager.broadcast_to_team(team_id, {
                    "type": "cursor_update",
                    "user_id": user_id,
                    "position": message.get("position"),
                    "timestamp": "2024-06-24T10:30:00Z"
                }, exclude_user=user_id)
            
            elif message.get("type") == "text_edit":
                await manager.broadcast_to_team(team_id, {
                    "type": "text_change",
                    "user_id": user_id,
                    "edit": message.get("edit"),
                    "document_id": message.get("document_id"),
                    "timestamp": "2024-06-24T10:30:00Z"
                }, exclude_user=user_id)
            
            elif message.get("type") == "chat_message":
                await manager.broadcast_to_team(team_id, {
                    "type": "team_chat",
                    "user_id": user_id,
                    "message": message.get("message"),
                    "timestamp": "2024-06-24T10:30:00Z"
                })
            
            else:
                # Echo other messages to team
                await manager.broadcast_to_team(team_id, {
                    "type": "team_message",
                    "user_id": user_id,
                    "message": message,
                    "timestamp": "2024-06-24T10:30:00Z"
                })
            
    except WebSocketDisconnect:
        # Notify team of user leaving
        await manager.broadcast_to_team(team_id, {
            "type": "user_left",
            "user_id": user_id,
            "team_id": team_id,
            "timestamp": "2024-06-24T10:30:00Z"
        }, exclude_user=user_id)
        manager.disconnect(team_id, user_id)
    except Exception as e:
        logger.error(f"WebSocket error for team {team_id}: {e}")
        manager.disconnect(team_id, user_id)