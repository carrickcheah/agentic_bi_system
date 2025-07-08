"""
LanceDB Logging Module
Simple logging utilities for the LanceDB module.
"""

import asyncio
import logging
import time
from typing import Any, Dict
from functools import wraps

# Set up module logger
logger = logging.getLogger("lance_db")


def log_operation(operation: str, details: Dict[str, Any] = None):
    """Log an operation with details."""
    msg = f"{operation}"
    if details:
        msg += f" - {details}"
    logger.info(msg)


def log_error(operation: str, error: Exception):
    """Log an error with operation context."""
    logger.error(f"{operation} failed: {error}")


def log_performance(operation: str, duration_ms: float):
    """Log performance metrics."""
    logger.info(f"{operation} completed in {duration_ms:.2f}ms")


def performance_monitor(operation_name: str):
    """Decorator to monitor performance of operations."""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000
                log_performance(operation_name, duration_ms)
                return result
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                log_error(f"{operation_name} (after {duration_ms:.2f}ms)", e)
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000
                log_performance(operation_name, duration_ms)
                return result
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                log_error(f"{operation_name} (after {duration_ms:.2f}ms)", e)
                raise
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def get_logger(name: str = "lance_db") -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(name)


# Configure default logging
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)