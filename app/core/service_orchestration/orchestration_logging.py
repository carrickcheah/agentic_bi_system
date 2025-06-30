"""
Local logging - avoids naming conflicts with stdlib.
Self-contained logging without external dependencies.
Service orchestration specific logging with performance monitoring.
"""

import logging
import sys
import time
import functools
from typing import Callable, Any, Dict


def setup_logger(name: str = "service_orchestration", level: str = "INFO") -> logging.Logger:
    """Set up isolated logger for the service orchestration module."""
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


def performance_monitor(operation_name: str) -> Callable:
    """
    Decorator for monitoring performance of service orchestration operations.
    
    Args:
        operation_name: Name of the operation being monitored
        
    Returns:
        Decorated function with performance logging
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            logger = setup_logger()
            start_time = time.time()
            
            try:
                logger.debug(f"Starting {operation_name}")
                result = await func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                logger.info(
                    f"Completed {operation_name} in {execution_time:.3f}s"
                )
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(
                    f"Failed {operation_name} after {execution_time:.3f}s: {e}"
                )
                raise
                
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            logger = setup_logger()
            start_time = time.time()
            
            try:
                logger.debug(f"Starting {operation_name}")
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                logger.info(
                    f"Completed {operation_name} in {execution_time:.3f}s"
                )
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(
                    f"Failed {operation_name} after {execution_time:.3f}s: {e}"
                )
                raise
        
        # Return appropriate wrapper based on function type
        if hasattr(func, '__code__') and 'await' in func.__code__.co_names:
            return async_wrapper
        else:
            return sync_wrapper
            
    return decorator


def log_service_metrics(service_name: str, metrics: Dict[str, Any]) -> None:
    """
    Log service-specific metrics for monitoring and observability.
    
    Args:
        service_name: Name of the service
        metrics: Dictionary of metrics to log
    """
    logger = setup_logger()
    metric_strings = [f"{key}={value}" for key, value in metrics.items()]
    logger.info(f"Service metrics [{service_name}]: {', '.join(metric_strings)}")


def log_orchestration_event(event_type: str, details: Dict[str, Any]) -> None:
    """
    Log orchestration events for debugging and monitoring.
    
    Args:
        event_type: Type of orchestration event
        details: Event details dictionary
    """
    logger = setup_logger()
    detail_strings = [f"{key}={value}" for key, value in details.items()]
    logger.info(f"Orchestration event [{event_type}]: {', '.join(detail_strings)}")


# Default logger instance
logger = setup_logger("service_orchestration", "INFO")