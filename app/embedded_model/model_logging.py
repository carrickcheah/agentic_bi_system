"""
Local logging configuration for embedding model module.
Self-contained logging without external dependencies.
"""

import logging
import sys
from pathlib import Path

def setup_logger(name: str = "embedding_model", level: str = "INFO") -> logging.Logger:
    """Set up isolated logger for the embedding model module."""
    logger = logging.getLogger(name)
    
    if not logger.handlers:  # Avoid duplicate handlers
        numeric_level = getattr(logging, level.upper(), logging.INFO)
        logger.setLevel(numeric_level)
        
        # Console handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(numeric_level)
        
        # Format with module context
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        # Prevent propagation to root logger
        logger.propagate = False
    
    return logger

# Default logger instance
logger = setup_logger("embedding_model", "INFO")