"""
Intelligence Module Local Logging - Self-Contained Architecture
Avoids naming conflicts with stdlib and provides isolated logging.
Zero external dependencies beyond module boundary.
"""

import logging
import sys
import time
from typing import Optional
from functools import wraps

try:
    from .config import settings
except ImportError:
    # For standalone execution
    from config import settings


def setup_logger(name: str = "intelligence", level: Optional[str] = None) -> logging.Logger:
    """
    Set up isolated logger for the intelligence module.
    
    Args:
        name: Logger name (prefixed with intelligence)
        level: Log level override (defaults to settings.log_level)
    
    Returns:
        Configured logger instance
    """
    if level is None:
        level = settings.log_level
    
    logger_name = f"intelligence.{name}"
    logger = logging.getLogger(logger_name)
    
    if not logger.handlers:  # Avoid duplicate handlers
        numeric_level = getattr(logging, level.upper(), logging.INFO)
        logger.setLevel(numeric_level)
        
        # Create console handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(numeric_level)
        
        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        
        logger.addHandler(handler)
        logger.propagate = False  # Prevent duplicate logs
    
    return logger


def log_operation(operation: str, details: dict = None, logger_name: str = "operations"):
    """
    Log business operation with structured details.
    
    Args:
        operation: Operation description
        details: Additional operation details
        logger_name: Logger category
    """
    logger = setup_logger(logger_name)
    
    if details:
        detail_str = " | ".join([f"{k}: {v}" for k, v in details.items()])
        logger.info(f"{operation} | {detail_str}")
    else:
        logger.info(operation)


def log_error(operation: str, error: Exception, logger_name: str = "errors"):
    """
    Log error with operation context.
    
    Args:
        operation: Operation that failed
        error: Exception that occurred
        logger_name: Logger category
    """
    logger = setup_logger(logger_name)
    logger.error(f"{operation} failed: {type(error).__name__}: {str(error)}")


def log_performance(operation: str, duration_ms: float, logger_name: str = "performance"):
    """
    Log performance metrics for operations.
    
    Args:
        operation: Operation name
        duration_ms: Operation duration in milliseconds
        logger_name: Logger category
    """
    if settings.detailed_logging:
        logger = setup_logger(logger_name)
        logger.info(f"{operation} completed in {duration_ms:.2f}ms")


def log_strategy_planning(
    query_type: str,
    complexity: str,
    methodology: str,
    duration_ms: float,
    success: bool = True
):
    """
    Log strategy planning specific metrics.
    
    Args:
        query_type: Type of query being planned
        complexity: Determined complexity level
        methodology: Selected methodology
        duration_ms: Planning duration
        success: Whether planning succeeded
    """
    logger = setup_logger("strategy_planning")
    
    status = "SUCCESS" if success else "FAILED"
    logger.info(
        f"Strategy Planning {status} | "
        f"Query: {query_type} | "
        f"Complexity: {complexity} | "
        f"Methodology: {methodology} | "
        f"Duration: {duration_ms:.2f}ms"
    )


def log_pattern_match(
    pattern_id: str,
    similarity_score: float,
    business_domain: str,
    confidence: float
):
    """
    Log pattern matching results.
    
    Args:
        pattern_id: Matched pattern identifier
        similarity_score: Similarity score
        business_domain: Business domain classification
        confidence: Match confidence
    """
    logger = setup_logger("pattern_matching")
    logger.info(
        f"Pattern Match | "
        f"ID: {pattern_id} | "
        f"Similarity: {similarity_score:.3f} | "
        f"Domain: {business_domain} | "
        f"Confidence: {confidence:.3f}"
    )


def log_organizational_learning(
    pattern_id: str,
    old_success_rate: float,
    new_success_rate: float,
    confidence_interval: tuple
):
    """
    Log organizational learning updates.
    
    Args:
        pattern_id: Pattern being updated
        old_success_rate: Previous success rate
        new_success_rate: Updated success rate
        confidence_interval: Bayesian confidence interval
    """
    logger = setup_logger("organizational_learning")
    logger.info(
        f"Learning Update | "
        f"Pattern: {pattern_id} | "
        f"Success Rate: {old_success_rate:.3f} → {new_success_rate:.3f} | "
        f"CI: [{confidence_interval[0]:.3f}, {confidence_interval[1]:.3f}]"
    )


def performance_monitor(operation_name: str):
    """
    Decorator for automatic performance monitoring.
    
    Args:
        operation_name: Name of the operation being monitored
    
    Returns:
        Decorated function with performance logging
    """
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
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def log_health_check(component: str, status: str, details: dict = None):
    """
    Log health check results.
    
    Args:
        component: Component being checked
        status: Health status (HEALTHY, DEGRADED, UNHEALTHY)
        details: Additional health details
    """
    logger = setup_logger("health")
    
    if details:
        detail_str = " | ".join([f"{k}: {v}" for k, v in details.items()])
        logger.info(f"Health Check | {component}: {status} | {detail_str}")
    else:
        logger.info(f"Health Check | {component}: {status}")


# Default logger instance
logger = setup_logger("intelligence", settings.log_level)