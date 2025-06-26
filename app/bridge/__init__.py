"""
Service Bridge Package

Communication layer between FastAPI frontend and FastMCP backend.
"""

from .service_bridge import ServiceBridge, get_service_bridge

__all__ = ["ServiceBridge", "get_service_bridge"]