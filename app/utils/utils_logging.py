"""
Simple logging utility for the application.
"""

import logging
import sys


def setup_logger(name: str = "app", level: str = "INFO") -> logging.Logger:
    """Set up a basic logger."""
    logger = logging.getLogger(name)
    
    if not logger.handlers:
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


# Default logger
logger = setup_logger("app")