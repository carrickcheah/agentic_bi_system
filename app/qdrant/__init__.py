"""Qdrant Vector Database Module.

Production-grade vector search with circuit breaker, caching, and monitoring.
Replaces LanceDB with 10-100x performance improvement.
"""

from .runner import QdrantService, get_qdrant_service
from .config import QdrantSettings, settings

__all__ = [
    "QdrantService",
    "get_qdrant_service",
    "QdrantSettings",
    "settings",
]

__version__ = "1.0.0"
__description__ = "Production Qdrant vector search with ML engineering standards"