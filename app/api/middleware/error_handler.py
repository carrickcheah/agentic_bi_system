"""
Error Handler Middleware

Graceful error handling and standardized error responses
for the autonomous business intelligence system.
"""

import traceback
from typing import Dict, Any
from fastapi import Request, Response, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from ...utils.logging import logger


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """Global error handling middleware."""
    
    def __init__(self, app, debug: bool = False):
        super().__init__(app)
        self.debug = debug
        
    async def dispatch(self, request: Request, call_next):
        """Handle errors gracefully."""
        try:
            response = await call_next(request)
            return response
            
        except HTTPException as e:
            # Handle FastAPI HTTP exceptions
            logger.warning(f"HTTP Exception: {e.status_code} - {e.detail}")
            return self._create_error_response(
                status_code=e.status_code,
                error_type="http_error",
                message=e.detail,
                request=request
            )
            
        except ValueError as e:
            # Handle validation errors
            logger.warning(f"Validation Error: {str(e)}")
            return self._create_error_response(
                status_code=400,
                error_type="validation_error",
                message=str(e),
                request=request
            )
            
        except ConnectionError as e:
            # Handle database/service connection errors
            logger.error(f"Connection Error: {str(e)}")
            return self._create_error_response(
                status_code=503,
                error_type="service_unavailable",
                message="Service temporarily unavailable",
                request=request
            )
            
        except Exception as e:
            # Handle unexpected errors
            logger.error(f"Unexpected Error: {str(e)}\n{traceback.format_exc()}")
            return self._create_error_response(
                status_code=500,
                error_type="internal_error",
                message="An unexpected error occurred" if not self.debug else str(e),
                request=request,
                traceback=traceback.format_exc() if self.debug else None
            )
    
    def _create_error_response(
        self,
        status_code: int,
        error_type: str,
        message: str,
        request: Request,
        traceback: str = None
    ) -> JSONResponse:
        """Create standardized error response."""
        
        error_response = {
            "error": {
                "type": error_type,
                "message": message,
                "status_code": status_code,
                "timestamp": "2024-01-01T00:00:00Z",  # Would use actual timestamp
                "request_id": getattr(request.state, "request_id", "unknown"),
                "path": str(request.url.path),
                "method": request.method
            }
        }
        
        # Add debug information if in debug mode
        if self.debug and traceback:
            error_response["error"]["traceback"] = traceback
        
        # Add helpful error codes and suggestions
        error_response["error"]["code"] = self._get_error_code(error_type, status_code)
        error_response["error"]["suggestion"] = self._get_error_suggestion(error_type, status_code)
        
        return JSONResponse(
            status_code=status_code,
            content=error_response
        )
    
    def _get_error_code(self, error_type: str, status_code: int) -> str:
        """Get specific error code for the error."""
        codes = {
            ("validation_error", 400): "VALIDATION_FAILED",
            ("http_error", 401): "AUTHENTICATION_REQUIRED",
            ("http_error", 403): "INSUFFICIENT_PERMISSIONS",
            ("http_error", 404): "RESOURCE_NOT_FOUND",
            ("http_error", 429): "RATE_LIMIT_EXCEEDED",
            ("service_unavailable", 503): "SERVICE_UNAVAILABLE",
            ("internal_error", 500): "INTERNAL_SERVER_ERROR"
        }
        return codes.get((error_type, status_code), "UNKNOWN_ERROR")
    
    def _get_error_suggestion(self, error_type: str, status_code: int) -> str:
        """Get helpful suggestion for the error."""
        suggestions = {
            ("validation_error", 400): "Please check your request parameters and try again.",
            ("http_error", 401): "Please provide valid authentication credentials.",
            ("http_error", 403): "You don't have permission to access this resource.",
            ("http_error", 404): "The requested resource was not found.",
            ("http_error", 429): "Please wait before making more requests.",
            ("service_unavailable", 503): "The service is temporarily unavailable. Please try again later.",
            ("internal_error", 500): "An internal error occurred. Please contact support if the issue persists."
        }
        return suggestions.get((error_type, status_code), "Please contact support for assistance.")