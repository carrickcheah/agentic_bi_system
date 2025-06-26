"""
Utility modules for the Agentic SQL backend.

Provides logging, monitoring, exception handling, and other common utilities.
"""

from .logging import logger, setup_logging
from .exceptions import (
    AgentError,
    InvestigationError,
    InvestigationStepError,
    MemoryError,
    MCPToolError
)

# Only import monitoring for FastAPI mode, not MCP standalone
try:
    from .monitoring import setup_monitoring
    _monitoring_available = True
except ImportError:
    _monitoring_available = False
    def setup_monitoring(*args, **kwargs):
        """Dummy function when monitoring is not available."""
        pass

__all__ = [
    "logger",
    "setup_logging", 
    "setup_monitoring",
    "AgentError",
    "InvestigationError",
    "InvestigationStepError",
    "MemoryError",
    "MCPToolError"
]