"""Local logging for Qdrant module.

Avoids naming conflicts with stdlib logging.
Self-contained logging without external dependencies.
"""

import logging
import sys
from typing import Optional


def setup_logger(
    name: str = "qdrant",
    level: str = "INFO",
    format_string: Optional[str] = None
) -> logging.Logger:
    """Set up isolated logger for the Qdrant module.
    
    Args:
        name: Logger name (default: 'qdrant')
        level: Logging level (default: 'INFO')
        format_string: Custom format string (optional)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers
    if not logger.handlers:
        # Convert string level to numeric
        numeric_level = getattr(logging, level.upper(), logging.INFO)
        logger.setLevel(numeric_level)
        
        # Create console handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(numeric_level)
        
        # Set formatter
        if format_string is None:
            format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        
        formatter = logging.Formatter(format_string)
        handler.setFormatter(formatter)
        
        # Add handler and prevent propagation
        logger.addHandler(handler)
        logger.propagate = False
    
    return logger


# Default logger instance
logger = setup_logger("qdrant", "INFO")


# Performance logging helpers
def log_performance(operation: str, duration_ms: float, success: bool = True):
    """Log performance metrics."""
    status = "SUCCESS" if success else "FAILED"
    logger.info(
        f"Performance: {operation} - {status} - {duration_ms:.2f}ms"
    )


def log_slow_operation(operation: str, duration_ms: float, threshold_ms: float):
    """Log slow operations that exceed threshold."""
    if duration_ms > threshold_ms:
        logger.warning(
            f"Slow operation: {operation} took {duration_ms:.2f}ms "
            f"(threshold: {threshold_ms}ms)"
        )


# Structured logging helpers
def log_circuit_breaker_state_change(old_state: str, new_state: str, reason: str):
    """Log circuit breaker state transitions."""
    logger.info(
        f"Circuit breaker state change: {old_state} -> {new_state} "
        f"(reason: {reason})"
    )


def log_cache_hit(query_hash: str, cache_age_seconds: float):
    """Log cache hit with age information."""
    logger.debug(
        f"Cache HIT: query_hash={query_hash}, age={cache_age_seconds:.1f}s"
    )


def log_cache_miss(query_hash: str):
    """Log cache miss."""
    logger.debug(f"Cache MISS: query_hash={query_hash}")


# Error logging helpers
def log_connection_error(error: Exception, retry_count: int = 0):
    """Log connection errors with retry information."""
    if retry_count > 0:
        logger.error(
            f"Connection error (retry {retry_count}): {type(error).__name__}: {error}"
        )
    else:
        logger.error(f"Connection error: {type(error).__name__}: {error}")


def log_validation_error(field: str, value: any, reason: str):
    """Log validation errors."""
    logger.error(
        f"Validation error: field='{field}', value='{value}', reason='{reason}'"
    )


# Metric logging
class MetricsLogger:
    """Structured metrics logging."""
    
    @staticmethod
    def log_metrics(metrics: dict):
        """Log metrics snapshot."""
        logger.info(
            f"Metrics snapshot: "
            f"queries={metrics.get('total_queries', 0)}, "
            f"cache_hits={metrics.get('cache_hits', 0)}, "
            f"avg_latency={metrics.get('avg_latency_ms', 0):.2f}ms, "
            f"errors={metrics.get('total_errors', 0)}"
        )
    
    @staticmethod
    def log_health_check(healthy: bool, details: dict):
        """Log health check results."""
        if healthy:
            logger.info(f"Health check PASSED: {details}")
        else:
            logger.error(f"Health check FAILED: {details}")