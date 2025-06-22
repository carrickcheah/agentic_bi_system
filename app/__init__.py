"""
Agentic SQL Backend Application

This module contains the core backend implementation for the autonomous SQL investigation agent.
Built to work like Claude Code but for data analysis and SQL operations.
"""

__version__ = "0.1.0"
__author__ = "Agentic SQL Team"

from .config import settings
from .main import app

__all__ = ["app", "settings"]