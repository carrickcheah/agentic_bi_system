"""
Caching Middleware

Request-level caching middleware for the multi-tier cache cascade system.
Handles cache headers, ETags, and intelligent cache control.
"""

import hashlib
import json
from typing import Optional
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from ...utils.logging import logger


class CachingMiddleware(BaseHTTPMiddleware):
    """Request-level caching middleware."""
    
    def __init__(self, app, cache_ttl: int = 300):
        super().__init__(app)
        self.cache_ttl = cache_ttl
        self.cache_store = {}
        
    async def dispatch(self, request: Request, call_next):
        """Handle request caching logic."""
        
        # Only cache GET requests
        if request.method != "GET":
            return await call_next(request)
        
        # Generate cache key
        cache_key = self._generate_cache_key(request)
        
        # Check if cached
        if cache_key in self.cache_store:
            cached_data = self.cache_store[cache_key]
            logger.info(f"<¯ Cache HIT: {cache_key[:16]}...")
            
            response = Response(
                content=cached_data["content"],
                status_code=cached_data["status_code"],
                headers=cached_data["headers"]
            )
            response.headers["X-Cache-Status"] = "HIT"
            return response
        
        # Process request
        response = await call_next(request)
        
        # Cache successful responses
        if response.status_code == 200:
            response_body = b""
            async for chunk in response.body_iterator:
                response_body += chunk
            
            # Store in cache
            self.cache_store[cache_key] = {
                "content": response_body,
                "status_code": response.status_code,
                "headers": dict(response.headers)
            }
            
            # Add cache headers
            response.headers["X-Cache-Status"] = "MISS"
            response.headers["Cache-Control"] = f"max-age={self.cache_ttl}"
            
            # Return new response with body
            return Response(
                content=response_body,
                status_code=response.status_code,
                headers=response.headers,
                media_type=response.media_type
            )
        
        response.headers["X-Cache-Status"] = "SKIP"
        return response
    
    def _generate_cache_key(self, request: Request) -> str:
        """Generate cache key for request."""
        key_data = {
            "path": str(request.url.path),
            "query": str(request.url.query),
            "user": request.headers.get("Authorization", "")[:20]  # Partial auth
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()