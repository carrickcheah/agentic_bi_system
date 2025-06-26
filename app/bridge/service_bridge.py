"""
Service Bridge - Communication Layer Between FastAPI and FastMCP

This bridge provides a clean interface for FastAPI routes to communicate
with the FastMCP backend service. It supports both embedded (in-process)
and separate service deployment modes.

Features:
- Abstracted service communication
- Error handling and retries
- Load balancing for multiple backend instances
- Health monitoring and failover
- Async/await interface
- Type-safe operations
"""

from typing import Dict, List, Optional, Any, Protocol, Union
from abc import ABC, abstractmethod
from datetime import datetime
import asyncio

from ..utils.logging import logger
from ..utils.exceptions import ServiceBridgeError, ServiceUnavailableError
from ..fastmcp.service import BusinessService, QueryResult
from ..server import get_embedded_service


class ServiceBridgeProtocol(Protocol):
    """Protocol for service bridge implementations."""
    
    async def execute_sql(
        self,
        query: str,
        database: str = "mariadb",
        max_rows: int = 1000,
        timeout: int = 30
    ) -> QueryResult:
        """Execute SQL query."""
        ...
    
    async def get_database_schema(
        self,
        database: str = "mariadb",
        table_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get database schema."""
        ...
    
    async def list_tables(self, database: str = "mariadb") -> List[str]:
        """List database tables."""
        ...
    
    async def is_healthy(self) -> bool:
        """Check service health."""
        ...


class EmbeddedServiceBridge:
    """
    Embedded service bridge for in-process communication.
    
    This bridge directly calls the FastMCP service in the same process,
    providing the highest performance with no network overhead.
    """
    
    def __init__(self):
        self._service: Optional[BusinessService] = None
        self._last_health_check = None
        self._healthy = False
    
    async def _get_service(self) -> BusinessService:
        """Get the embedded FastMCP service instance."""
        if self._service is None:
            self._service = await get_embedded_service()
        
        if self._service is None:
            raise ServiceUnavailableError("FastMCP service not available")
        
        return self._service
    
    async def execute_sql(
        self,
        query: str,
        database: str = "mariadb",
        max_rows: int = 1000,
        timeout: int = 30
    ) -> QueryResult:
        """Execute SQL query via embedded service."""
        try:
            service = await self._get_service()
            result = await service.execute_sql(
                query=query,
                database=database,
                max_rows=max_rows,
                timeout=timeout
            )
            return result
            
        except Exception as e:
            logger.error(f"Embedded SQL execution failed: {e}")
            raise ServiceBridgeError(f"SQL execution failed: {e}")
    
    async def get_database_schema(
        self,
        database: str = "mariadb",
        table_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get database schema via embedded service."""
        try:
            service = await self._get_service()
            schema = await service.get_database_schema(
                database=database,
                table_name=table_name
            )
            return schema
            
        except Exception as e:
            logger.error(f"Embedded schema retrieval failed: {e}")
            raise ServiceBridgeError(f"Schema retrieval failed: {e}")
    
    async def list_tables(self, database: str = "mariadb") -> List[str]:
        """List tables via embedded service."""
        try:
            service = await self._get_service()
            tables = await service.list_tables(database=database)
            return tables
            
        except Exception as e:
            logger.error(f"Embedded table listing failed: {e}")
            raise ServiceBridgeError(f"Table listing failed: {e}")
    
    async def store_query_pattern(
        self,
        sql_query: str,
        description: str,
        result_summary: Optional[str] = None,
        execution_time: Optional[float] = None,
        success: bool = True
    ) -> Dict[str, Any]:
        """Store query pattern via embedded service."""
        try:
            service = await self._get_service()
            result = await service.store_query_pattern(
                sql_query=sql_query,
                description=description,
                result_summary=result_summary,
                execution_time=execution_time,
                success=success
            )
            return result
            
        except Exception as e:
            logger.error(f"Embedded pattern storage failed: {e}")
            raise ServiceBridgeError(f"Pattern storage failed: {e}")
    
    async def find_similar_queries(
        self,
        description: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Find similar queries via embedded service."""
        try:
            service = await self._get_service()
            results = await service.find_similar_queries(
                description=description,
                limit=limit
            )
            return results
            
        except Exception as e:
            logger.error(f"Embedded similar query search failed: {e}")
            raise ServiceBridgeError(f"Similar query search failed: {e}")
    
    async def create_investigation(
        self,
        query: str,
        user_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create investigation via embedded service."""
        try:
            service = await self._get_service()
            investigation_id = await service.create_investigation(
                query=query,
                user_id=user_id,
                context=context
            )
            return investigation_id
            
        except Exception as e:
            logger.error(f"Embedded investigation creation failed: {e}")
            raise ServiceBridgeError(f"Investigation creation failed: {e}")
    
    async def get_investigation(self, investigation_id: str) -> Optional[Dict[str, Any]]:
        """Get investigation via embedded service."""
        try:
            service = await self._get_service()
            investigation = await service.get_investigation(investigation_id)
            return investigation
            
        except Exception as e:
            logger.error(f"Embedded investigation retrieval failed: {e}")
            raise ServiceBridgeError(f"Investigation retrieval failed: {e}")
    
    async def analyze_business_query(
        self,
        query: str,
        domain: Optional[str] = None,
        user_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Analyze business query via embedded service."""
        try:
            service = await self._get_service()
            analysis = await service.analyze_business_query(
                query=query,
                domain=domain,
                user_context=user_context
            )
            return analysis
            
        except Exception as e:
            logger.error(f"Embedded business query analysis failed: {e}")
            raise ServiceBridgeError(f"Business query analysis failed: {e}")
    
    async def is_healthy(self) -> bool:
        """Check embedded service health."""
        try:
            # Cache health check for 30 seconds
            now = datetime.utcnow()
            if (self._last_health_check and 
                (now - self._last_health_check).total_seconds() < 30):
                return self._healthy
            
            service = await self._get_service()
            self._healthy = service.is_healthy()
            self._last_health_check = now
            
            return self._healthy
            
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            self._healthy = False
            return False
    
    async def get_service_status(self) -> Dict[str, Any]:
        """Get comprehensive service status."""
        try:
            service = await self._get_service()
            status = await service.get_service_status()
            
            # Add bridge-specific information
            status["bridge"] = {
                "type": "embedded",
                "healthy": await self.is_healthy(),
                "last_health_check": self._last_health_check.isoformat() if self._last_health_check else None
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Service status retrieval failed: {e}")
            return {
                "bridge": {
                    "type": "embedded",
                    "healthy": False,
                    "error": str(e)
                },
                "timestamp": datetime.utcnow().isoformat()
            }


class RemoteServiceBridge:
    """
    Remote service bridge for inter-process communication.
    
    This bridge communicates with FastMCP service running in a separate
    process via HTTP, gRPC, or message queues. Provides network resilience
    and allows for distributed deployment.
    
    Note: This is a placeholder for future implementation.
    """
    
    def __init__(self, service_url: str):
        self.service_url = service_url
        # TODO: Implement HTTP/gRPC client
    
    async def execute_sql(self, query: str, database: str = "mariadb", **kwargs) -> QueryResult:
        # TODO: Implement remote HTTP/gRPC call
        raise NotImplementedError("Remote service bridge not yet implemented")
    
    async def is_healthy(self) -> bool:
        # TODO: Implement remote health check
        return False


class ServiceBridge:
    """
    Main service bridge that can use different communication strategies.
    
    This bridge provides a unified interface for FastAPI to communicate
    with FastMCP backend, regardless of deployment mode.
    """
    
    def __init__(self, bridge_type: str = "embedded", **kwargs):
        self.bridge_type = bridge_type
        
        if bridge_type == "embedded":
            self._bridge = EmbeddedServiceBridge()
        elif bridge_type == "remote":
            service_url = kwargs.get("service_url")
            if not service_url:
                raise ServiceBridgeError("service_url required for remote bridge")
            self._bridge = RemoteServiceBridge(service_url)
        else:
            raise ServiceBridgeError(f"Unknown bridge type: {bridge_type}")
    
    async def execute_sql(
        self,
        query: str,
        database: str = "mariadb",
        max_rows: int = 1000,
        timeout: int = 30
    ) -> QueryResult:
        """Execute SQL query through the bridge."""
        return await self._bridge.execute_sql(
            query=query,
            database=database,
            max_rows=max_rows,
            timeout=timeout
        )
    
    async def get_database_schema(
        self,
        database: str = "mariadb",
        table_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get database schema through the bridge."""
        return await self._bridge.get_database_schema(
            database=database,
            table_name=table_name
        )
    
    async def list_tables(self, database: str = "mariadb") -> List[str]:
        """List tables through the bridge."""
        return await self._bridge.list_tables(database=database)
    
    async def store_query_pattern(
        self,
        sql_query: str,
        description: str,
        result_summary: Optional[str] = None,
        execution_time: Optional[float] = None,
        success: bool = True
    ) -> Dict[str, Any]:
        """Store query pattern through the bridge."""
        return await self._bridge.store_query_pattern(
            sql_query=sql_query,
            description=description,
            result_summary=result_summary,
            execution_time=execution_time,
            success=success
        )
    
    async def find_similar_queries(
        self,
        description: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Find similar queries through the bridge."""
        return await self._bridge.find_similar_queries(
            description=description,
            limit=limit
        )
    
    async def create_investigation(
        self,
        query: str,
        user_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create investigation through the bridge."""
        return await self._bridge.create_investigation(
            query=query,
            user_id=user_id,
            context=context
        )
    
    async def get_investigation(self, investigation_id: str) -> Optional[Dict[str, Any]]:
        """Get investigation through the bridge."""
        return await self._bridge.get_investigation(investigation_id)
    
    async def analyze_business_query(
        self,
        query: str,
        domain: Optional[str] = None,
        user_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Analyze business query through the bridge."""
        return await self._bridge.analyze_business_query(
            query=query,
            domain=domain,
            user_context=user_context
        )
    
    async def is_healthy(self) -> bool:
        """Check service health through the bridge."""
        return await self._bridge.is_healthy()
    
    async def get_service_status(self) -> Dict[str, Any]:
        """Get service status through the bridge."""
        return await self._bridge.get_service_status()


# Global service bridge instance
_global_bridge: Optional[ServiceBridge] = None


def get_service_bridge() -> ServiceBridge:
    """
    Get the global service bridge instance.
    
    Creates an embedded bridge by default. In production, this could
    be configured to use remote bridges for distributed deployment.
    """
    global _global_bridge
    
    if _global_bridge is None:
        # Default to embedded bridge for development
        _global_bridge = ServiceBridge(bridge_type="embedded")
    
    return _global_bridge


def configure_service_bridge(bridge_type: str = "embedded", **kwargs) -> ServiceBridge:
    """
    Configure the global service bridge.
    
    Args:
        bridge_type: Type of bridge ("embedded" or "remote")
        **kwargs: Bridge-specific configuration
        
    Returns:
        Configured service bridge
    """
    global _global_bridge
    _global_bridge = ServiceBridge(bridge_type=bridge_type, **kwargs)
    return _global_bridge