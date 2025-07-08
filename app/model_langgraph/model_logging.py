"""
Local logging configuration for the model module.

This module provides self-contained logging functionality to eliminate 
dependencies on the parent utils module.
"""

import logging
import sys
from typing import Optional


def setup_logger(
    name: str = "model", 
    level: str = "INFO", 
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Set up a logger for the model module.
    
    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Custom format string for log messages
        
    Returns:
        Configured logger instance
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Create logger
    logger = logging.getLogger(name)
    
    # Set level
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)
    
    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(numeric_level)
    
    # Create formatter
    formatter = logging.Formatter(format_string)
    handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(handler)
    
    # Prevent propagation to avoid duplicate messages
    logger.propagate = False
    
    return logger


# Create default logger instance for the model module
logger = setup_logger("model", "INFO")


def set_log_level(level: str) -> None:
    """
    Change the log level for the model logger.
    
    Args:
        level: New log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)
    for handler in logger.handlers:
        handler.setLevel(numeric_level)


def enable_debug() -> None:
    """Enable debug logging for detailed troubleshooting."""
    set_log_level("DEBUG")


def disable_debug() -> None:
    """Disable debug logging, return to INFO level."""
    set_log_level("INFO")