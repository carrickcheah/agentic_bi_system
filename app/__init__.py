"""
Agentic SQL Backend Application

This module contains the core backend implementation for the autonomous SQL investigation agent.
Built to work like Claude Code but for data analysis and SQL operations.
"""

__version__ = "0.1.0"
__author__ = "Agentic SQL Team"

from .main import (
    model_manager,
    question_checker,
    qdrant_service,
    get_mcp_client_manager,
    get_cache_manager,
    initialize_async_services,
    AgenticBiFlow,
    process_query_with_validation_and_5_phases
)

__all__ = [
    "model_manager",
    "question_checker", 
    "qdrant_service",
    "get_mcp_client_manager",
    "get_cache_manager",
    "initialize_async_services",
    "AgenticBiFlow",
    "process_query_with_validation_and_5_phases"
]