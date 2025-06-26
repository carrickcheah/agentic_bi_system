"""
Custom exceptions for the agentic SQL application.
"""

from typing import Optional, Dict, Any


class AgentError(Exception):
    """Base exception for agent-related errors."""
    
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}


class BusinessValidationError(AgentError):
    """Raised when business validation fails."""
    pass


class DatabaseError(AgentError):
    """Raised when database operations fail."""
    pass


class CacheError(AgentError):
    """Raised when cache operations fail."""
    pass


class ConfigurationError(AgentError):
    """Raised when configuration is invalid."""
    pass


class InvestigationError(AgentError):
    """Raised when investigation execution fails."""
    pass


class PatternError(AgentError):
    """Raised when pattern operations fail."""
    pass


class SecurityError(AgentError):
    """Raised when security validation fails."""
    pass


class InvestigationStepError(InvestigationError):
    """Raised when investigation step execution fails."""
    pass


class MemoryError(AgentError):
    """Raised when memory operations fail."""
    pass


class MCPToolError(AgentError):
    """Raised when MCP tool operations fail."""
    pass