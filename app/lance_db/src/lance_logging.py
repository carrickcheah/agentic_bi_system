"""
Local logging for LanceDB module - avoids naming conflicts with stdlib.
Self-contained logging without external dependencies.
"""

import logging
import sys
from typing import Optional
from pathlib import Path


def setup_logger(name: str = "lancedb", level: str = "INFO") -> logging.Logger:
    """
    Set up isolated logger for the LanceDB module.
    
    Args:
        name: Logger name (default: "lancedb")
        level: Logging level (default: "INFO")
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers
    if not logger.handlers:
        numeric_level = getattr(logging, level.upper(), logging.INFO)
        logger.setLevel(numeric_level)
        
        # Console handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(numeric_level)
        
        # Format with module context
        formatter = logging.Formatter(
            "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        # Prevent propagation to root logger
        logger.propagate = False
    
    return logger


def get_logger(component: Optional[str] = None) -> logging.Logger:
    """
    Get a logger for a specific component within the module.
    
    Args:
        component: Component name (e.g., "embedding", "search")
    
    Returns:
        Logger instance with appropriate naming
    """
    # Load log level from config if available
    try:
        from .config import settings
        level = settings.log_level
    except:
        level = "INFO"
    
    if component:
        return setup_logger(f"lancedb.{component}", level)
    return setup_logger("lancedb", level)


# Default logger instance
logger = get_logger()


# Convenience functions for module-wide logging
def log_operation(operation: str, details: dict = None):
    """Log a LanceDB operation with optional details"""
    try:
        from .config import settings
        if settings.enable_detailed_logging and details:
            logger.info(f"{operation}: {details}")
        else:
            logger.info(operation)
    except:
        logger.info(operation)


def log_error(operation: str, error: Exception):
    """Log an error with operation context"""
    logger.error(f"{operation} failed: {type(error).__name__}: {str(error)}")


def log_performance(operation: str, duration_ms: float):
    """Log performance metrics for operations"""
    logger.info(f"{operation} completed in {duration_ms:.2f}ms")