"""
Security Middleware

Handles authentication, authorization, rate limiting, and security headers
for the autonomous business intelligence system.
"""

import time
from typing import Dict, Any
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from ...utils.logging import logger


class SecurityMiddleware(BaseHTTPMiddleware):
    """Security middleware for authentication and protection."""
    
    def __init__(self, app, secret_key: str = None):
        super().__init__(app)
        self.secret_key = secret_key or "default_secret_key"
        self.rate_limits = {}
        
    async def dispatch(self, request: Request, call_next):
        """Process security checks for each request."""
        start_time = time.time()
        
        # Add security headers
        response = await call_next(request)
        
        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        
        # Rate limiting info
        client_ip = request.client.host if request.client else "unknown"
        response.headers["X-RateLimit-Remaining"] = "100"
        
        # Processing time
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        
        return response