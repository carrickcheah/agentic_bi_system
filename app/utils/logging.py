"""
Structured Logging Configuration

Provides consistent logging setup for the application.
"""

import logging
import sys
from typing import Any, Dict

import structlog
from structlog.stdlib import LoggerFactory

try:
    from ..config import settings
except ImportError:
    from config import settings


def setup_logging():
    """Configure structured logging for the application."""
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer() if settings.log_format == "json"
            else structlog.dev.ConsoleRenderer(),
        ],
        context_class=dict,
        logger_factory=LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.log_level.upper()),
    )


# Create logger instance
logger = structlog.get_logger()


def log_sql_execution(
    query: str,
    database: str,
    execution_time: float,
    row_count: int,
    error: str = None
):
    """Log SQL execution with structured data."""
    log_data = {
        "event": "sql_execution",
        "database": database,
        "query": query[:200] + "..." if len(query) > 200 else query,
        "execution_time": execution_time,
        "row_count": row_count,
    }
    
    if error:
        log_data["error"] = error
        logger.error("SQL execution failed", **log_data)
    else:
        logger.info("SQL executed successfully", **log_data)


def log_investigation_step(
    investigation_id: str,
    step: str,
    details: Dict[str, Any] = None
):
    """Log investigation progress step."""
    logger.info(
        "Investigation step",
        investigation_id=investigation_id,
        step=step,
        details=details or {}
    )


def log_mcp_call(
    tool_name: str,
    parameters: Dict[str, Any],
    success: bool,
    execution_time: float = None,
    error: str = None
):
    """Log MCP tool calls."""
    log_data = {
        "event": "mcp_call",
        "tool": tool_name,
        "parameters": parameters,
        "success": success,
    }
    
    if execution_time:
        log_data["execution_time"] = execution_time
    
    if error:
        log_data["error"] = error
        logger.error("MCP tool call failed", **log_data)
    else:
        logger.info("MCP tool call completed", **log_data)