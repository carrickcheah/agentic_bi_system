"""
Service Orchestration Layer

Coordinates specialized database services through MCP protocol for business intelligence.

Services:
- Business Data Service: MariaDB with business logic understanding (sales, customers, products)
- Memory Service: PostgreSQL for organizational memory and session management
- Vector Service: Qdrant for semantic search and pattern matching
- Analytics Service: Advanced analytics and computation
- External Service: Supabase for additional operations

Features:
- MCP protocol standardization
- Business-aware query generation
- Service health monitoring
- Intelligent service selection
- Cross-service data correlation
"""

from .service_orchestrator import ServiceOrchestrator
from .business_data_service import BusinessDataService
from .memory_service import MemoryService
from .vector_service import VectorService
from .analytics_service import AnalyticsService
from .external_service import ExternalService

__all__ = [
    "ServiceOrchestrator",
    "BusinessDataService",
    "MemoryService",
    "VectorService", 
    "AnalyticsService",
    "ExternalService"
]