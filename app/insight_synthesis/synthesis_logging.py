"""
Synthesis Logging - Local logging for insight synthesis module.
Self-contained logging without external dependencies.
"""

import logging
import sys
import time
from functools import wraps
from typing import Any, Callable


def setup_logger(name: str = "insight_synthesis", level: str = "INFO") -> logging.Logger:
    """Set up isolated logger for the insight synthesis module."""
    logger = logging.getLogger(name)
    
    if not logger.handlers:  # Avoid duplicate handlers
        numeric_level = getattr(logging, level.upper(), logging.INFO)
        logger.setLevel(numeric_level)
        
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(numeric_level)
        
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
    
    return logger


def log_operation(operation_name: str):
    """Decorator to log operation start/end with timing."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            logger = setup_logger("synthesis_operations")
            start_time = time.time()
            
            logger.info(f"Starting {operation_name}")
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                logger.info(f"Completed {operation_name} in {duration:.3f}s")
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"Failed {operation_name} after {duration:.3f}s: {e}")
                raise
        
        return wrapper
    return decorator


def performance_monitor(operation_type: str):
    """Decorator for performance monitoring with detailed metrics."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            logger = setup_logger("synthesis_performance")
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Log performance metrics
                logger.info(
                    f"PERF: {operation_type} completed - "
                    f"duration: {duration:.3f}s"
                )
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                logger.error(
                    f"PERF: {operation_type} failed - "
                    f"duration: {duration:.3f}s, error: {e}"
                )
                raise
        
        return wrapper
    return decorator


# Default logger instance
logger = setup_logger("insight_synthesis", "INFO")