"""
Monitoring Middleware

Request monitoring, metrics collection, and performance tracking
for the autonomous business intelligence system.
"""

import time
import uuid
from typing import Dict, Any
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from ...utils.logging import logger


class MonitoringMiddleware(BaseHTTPMiddleware):
    """Request monitoring and metrics collection middleware."""
    
    def __init__(self, app):
        super().__init__(app)
        self.request_metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "avg_response_time": 0.0,
            "active_investigations": 0
        }
        
    async def dispatch(self, request: Request, call_next):
        """Monitor request processing."""
        
        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        request.state.start_time = time.time()
        
        # Update metrics
        self.request_metrics["total_requests"] += 1
        
        # Log request start
        logger.info(
            f"=€ Request started: {request.method} {request.url.path} "
            f"[ID: {request_id[:8]}...]"
        )
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate response time
            end_time = time.time()
            response_time = end_time - request.state.start_time
            
            # Update success metrics
            if 200 <= response.status_code < 400:
                self.request_metrics["successful_requests"] += 1
            else:
                self.request_metrics["failed_requests"] += 1
            
            # Update average response time
            self._update_avg_response_time(response_time)
            
            # Add monitoring headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Response-Time"] = f"{response_time:.3f}s"
            response.headers["X-Server-Instance"] = "business-analyst-1"
            
            # Log request completion
            logger.info(
                f" Request completed: {request.method} {request.url.path} "
                f"[{response.status_code}] [{response_time:.3f}s] [ID: {request_id[:8]}...]"
            )
            
            return response
            
        except Exception as e:
            # Handle errors
            end_time = time.time()
            response_time = end_time - request.state.start_time
            
            self.request_metrics["failed_requests"] += 1
            
            logger.error(
                f"L Request failed: {request.method} {request.url.path} "
                f"[{response_time:.3f}s] [ID: {request_id[:8]}...] - {str(e)}"
            )
            
            raise
    
    def _update_avg_response_time(self, response_time: float):
        """Update rolling average response time."""
        current_avg = self.request_metrics["avg_response_time"]
        total_requests = self.request_metrics["total_requests"]
        
        # Simple rolling average
        self.request_metrics["avg_response_time"] = (
            (current_avg * (total_requests - 1) + response_time) / total_requests
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current monitoring metrics."""
        return {
            "request_metrics": self.request_metrics.copy(),
            "performance": {
                "avg_response_time_ms": self.request_metrics["avg_response_time"] * 1000,
                "success_rate": (
                    self.request_metrics["successful_requests"] / 
                    max(self.request_metrics["total_requests"], 1)
                ),
                "error_rate": (
                    self.request_metrics["failed_requests"] / 
                    max(self.request_metrics["total_requests"], 1)
                )
            },
            "system_health": {
                "status": "healthy",
                "uptime_seconds": time.time(),  # Simplified
                "active_connections": 1  # Simplified
            }
        }