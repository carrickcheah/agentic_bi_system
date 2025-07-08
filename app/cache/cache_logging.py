"""
Local logging for cache module - avoids naming conflicts.
Self-contained logging without external dependencies.
"""

import logging
import sys

def setup_logger(name: str = "cache", level: str = "INFO") -> logging.Logger:
    """Set up isolated logger for the cache module."""
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

# Default logger instance
logger = setup_logger("cache", "INFO")