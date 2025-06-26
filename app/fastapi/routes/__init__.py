"""
API Routes

REST API endpoints for the autonomous business intelligence system.
"""

from .investigations import router as investigations_router
from .intelligence import router as intelligence_router
from .collaboration import router as collaboration_router
from .analytics import router as analytics_router
from .cache import router as cache_router
from .admin import router as admin_router
from .monitoring import router as monitoring_router
from .database import router as database_router
from .sessions import router as sessions_router

__all__ = [
    "investigations_router",
    "intelligence_router",
    "collaboration_router",
    "analytics_router",
    "cache_router",
    "admin_router",
    "monitoring_router",
    "database_router",
    "sessions_router"
]