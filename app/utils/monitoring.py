"""
Application Monitoring and Metrics

Provides monitoring setup and custom metrics collection.
"""

import time
from functools import wraps
from typing import Callable, Any

from fastapi import FastAPI, Request, Response
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from .logging import logger


# Prometheus metrics
request_count = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status_code']
)

request_duration = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint']
)

sql_query_count = Counter(
    'sql_queries_total',
    'Total SQL queries executed',
    ['database', 'status']
)

sql_query_duration = Histogram(
    'sql_query_duration_seconds',
    'SQL query execution time in seconds',
    ['database']
)

investigation_count = Counter(
    'investigations_total',
    'Total investigations started',
    ['status']
)

mcp_tool_calls = Counter(
    'mcp_tool_calls_total',
    'Total MCP tool calls',
    ['tool', 'status']
)


def setup_monitoring(app: FastAPI):
    """Set up monitoring middleware and endpoints."""
    
    @app.middleware("http")
    async def monitor_requests(request: Request, call_next: Callable) -> Response:
        """Monitor HTTP requests."""
        start_time = time.time()
        
        response = await call_next(request)
        
        # Record metrics
        duration = time.time() - start_time
        
        request_count.labels(
            method=request.method,
            endpoint=str(request.url.path),
            status_code=response.status_code
        ).inc()
        
        request_duration.labels(
            method=request.method,
            endpoint=str(request.url.path)
        ).observe(duration)
        
        # Log slow requests
        if duration > 2.0:  # Log requests slower than 2 seconds
            logger.warning(
                "Slow request detected",
                method=request.method,
                path=str(request.url.path),
                duration=duration,
                status_code=response.status_code
            )
        
        return response
    
    @app.get("/metrics")
    async def get_metrics():
        """Prometheus metrics endpoint."""
        return Response(
            generate_latest(),
            media_type=CONTENT_TYPE_LATEST
        )


def track_sql_execution(database: str):
    """Decorator to track SQL execution metrics."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            status = "success"
            
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                status = "error"
                raise
            finally:
                duration = time.time() - start_time
                
                sql_query_count.labels(
                    database=database,
                    status=status
                ).inc()
                
                sql_query_duration.labels(
                    database=database
                ).observe(duration)
        
        return wrapper
    return decorator


def track_investigation(func: Callable) -> Callable:
    """Decorator to track investigation metrics."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        status = "started"
        
        try:
            result = await func(*args, **kwargs)
            status = "completed"
            return result
        except Exception as e:
            status = "failed"
            raise
        finally:
            investigation_count.labels(status=status).inc()
    
    return wrapper


def track_mcp_call(tool_name: str):
    """Decorator to track MCP tool call metrics."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            status = "success"
            
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                status = "error"
                raise
            finally:
                mcp_tool_calls.labels(
                    tool=tool_name,
                    status=status
                ).inc()
        
        return wrapper
    return decorator


class PerformanceTimer:
    """Context manager for timing operations."""
    
    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        
        logger.info(
            "Operation completed",
            operation=self.operation_name,
            duration=duration,
            success=exc_type is None
        )