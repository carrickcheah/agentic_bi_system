"""
Monitoring WebSocket Endpoints

Real-time WebSocket connections for system monitoring including
live metrics, alerts, and performance dashboards.
"""

from typing import Dict, Any, Set
from fastapi import WebSocket, WebSocketDisconnect, Depends
from fastapi.routing import APIRouter
import json
import asyncio
from datetime import datetime

from ..dependencies import get_current_user_ws
from ...utils.logging import logger

router = APIRouter()


class MonitoringConnectionManager:
    """Manages WebSocket connections for real-time monitoring."""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.monitoring_sessions: Set[str] = set()
        self.metric_subscriptions: Dict[str, Set[str]] = {}
    
    async def connect(self, websocket: WebSocket, user_id: str, subscriptions: list = None):
        """Connect user to monitoring session."""
        await websocket.accept()
        connection_id = f"monitor_{user_id}_{hash(str(datetime.utcnow()))}"
        self.active_connections[connection_id] = websocket
        self.monitoring_sessions.add(connection_id)
        
        # Set up metric subscriptions
        if subscriptions:
            for metric in subscriptions:
                if metric not in self.metric_subscriptions:
                    self.metric_subscriptions[metric] = set()
                self.metric_subscriptions[metric].add(connection_id)
        
        logger.info(f"User {user_id} connected to monitoring with subscriptions: {subscriptions}")
        return connection_id
    
    def disconnect(self, connection_id: str):
        """Disconnect monitoring session."""
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]
        
        self.monitoring_sessions.discard(connection_id)
        
        # Remove from metric subscriptions
        for metric, subscribers in self.metric_subscriptions.items():
            subscribers.discard(connection_id)
        
        # Clean up empty subscription sets
        self.metric_subscriptions = {
            metric: subscribers 
            for metric, subscribers in self.metric_subscriptions.items() 
            if subscribers
        }
        
        logger.info(f"Monitoring connection {connection_id} disconnected")
    
    async def broadcast_metric_update(self, metric_type: str, data: dict):
        """Broadcast metric update to subscribed connections."""
        if metric_type not in self.metric_subscriptions:
            return
        
        message = {
            "type": "metric_update",
            "metric": metric_type,
            "data": data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        disconnected = []
        for connection_id in self.metric_subscriptions[metric_type]:
            if connection_id in self.active_connections:
                try:
                    await self.active_connections[connection_id].send_text(json.dumps(message))
                except:
                    disconnected.append(connection_id)
        
        # Clean up disconnected connections
        for connection_id in disconnected:
            self.disconnect(connection_id)
    
    async def broadcast_alert(self, alert_data: dict):
        """Broadcast alert to all monitoring connections."""
        message = {
            "type": "alert",
            "alert": alert_data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        disconnected = []
        for connection_id in self.monitoring_sessions:
            if connection_id in self.active_connections:
                try:
                    await self.active_connections[connection_id].send_text(json.dumps(message))
                except:
                    disconnected.append(connection_id)
        
        # Clean up disconnected connections
        for connection_id in disconnected:
            self.disconnect(connection_id)


manager = MonitoringConnectionManager()


@router.websocket("/ws/monitoring")
async def monitoring_websocket(
    websocket: WebSocket,
    current_user: Dict[str, Any] = Depends(get_current_user_ws)
):
    """WebSocket endpoint for real-time system monitoring."""
    user_id = current_user.get("user_id", "anonymous")
    connection_id = None
    
    try:
        # Initial connection without subscriptions
        connection_id = await manager.connect(websocket, user_id)
        
        # Send initial monitoring state
        await websocket.send_text(json.dumps({
            "type": "monitoring_connected",
            "user_id": user_id,
            "available_metrics": [
                "system_performance",
                "api_metrics", 
                "database_metrics",
                "cache_metrics",
                "investigation_metrics"
            ],
            "timestamp": datetime.utcnow().isoformat()
        }))
        
        while True:
            # Receive subscription requests from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "subscribe":
                metrics = message.get("metrics", [])
                for metric in metrics:
                    if metric not in manager.metric_subscriptions:
                        manager.metric_subscriptions[metric] = set()
                    manager.metric_subscriptions[metric].add(connection_id)
                
                await websocket.send_text(json.dumps({
                    "type": "subscription_confirmed",
                    "subscribed_metrics": metrics,
                    "timestamp": datetime.utcnow().isoformat()
                }))
            
            elif message.get("type") == "unsubscribe":
                metrics = message.get("metrics", [])
                for metric in metrics:
                    if metric in manager.metric_subscriptions:
                        manager.metric_subscriptions[metric].discard(connection_id)
                
                await websocket.send_text(json.dumps({
                    "type": "unsubscription_confirmed",
                    "unsubscribed_metrics": metrics,
                    "timestamp": datetime.utcnow().isoformat()
                }))
            
            elif message.get("type") == "request_current_data":
                # Send current system state
                await websocket.send_text(json.dumps({
                    "type": "current_data",
                    "system_metrics": {
                        "cpu_usage": 35.2,
                        "memory_usage": 62.8,
                        "active_investigations": 8,
                        "cache_hit_rate": 0.87
                    },
                    "timestamp": datetime.utcnow().isoformat()
                }))
            
    except WebSocketDisconnect:
        if connection_id:
            manager.disconnect(connection_id)
    except Exception as e:
        logger.error(f"WebSocket monitoring error: {e}")
        if connection_id:
            manager.disconnect(connection_id)


# Background task simulation for metric updates
async def simulate_metric_updates():
    """Simulate real-time metric updates (would be replaced with actual monitoring)."""
    while True:
        await asyncio.sleep(5)  # Update every 5 seconds
        
        # Simulate system performance update
        await manager.broadcast_metric_update("system_performance", {
            "cpu_usage": 35.2 + (hash(str(datetime.utcnow())) % 20 - 10),
            "memory_usage": 62.8 + (hash(str(datetime.utcnow())) % 10 - 5),
            "disk_usage": 48.1
        })
        
        # Simulate API metrics update
        await manager.broadcast_metric_update("api_metrics", {
            "requests_per_second": 12.5 + (hash(str(datetime.utcnow())) % 8 - 4),
            "average_response_time": 185 + (hash(str(datetime.utcnow())) % 50 - 25),
            "error_rate": 0.02
        })


# Start background monitoring (would be handled by FastAPI lifespan in production)
# asyncio.create_task(simulate_metric_updates())