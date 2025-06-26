"""
FastAPI Application Factory with Service Bridge Integration

This module creates and configures the FastAPI application with:
- All route handlers
- Service bridge for backend communication
- MCP tool mounting for agent communication
- Middleware and lifecycle management
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

from ..config import settings, get_cors_config
from ..utils.logging import setup_logging, logger
from ..utils.monitoring import setup_monitoring
from .routes import investigations, database, sessions
from .websocket import websocket_router
from ..server import initialize_embedded_server, cleanup_embedded_server
from ..bridge.service_bridge import get_service_bridge


# Global embedded server instance
embedded_server = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan management.
    
    Handles startup and shutdown of embedded backend server and services.
    """
    global embedded_server
    
    # Startup
    logger.info("Starting FastAPI with embedded backend server...")
    
    try:
        # Initialize embedded backend server
        logger.info("Initializing embedded backend server...")
        embedded_server = await initialize_embedded_server()
        
        # Store server reference in app state
        app.state.backend_server = embedded_server
        app.state.service_bridge = get_service_bridge()
        
        logger.info("FastAPI with backend server initialized successfully!")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to initialize backend server: {e}")
        raise
    finally:
        # Shutdown
        logger.info("Shutting down FastAPI and backend server...")
        
        if embedded_server:
            await cleanup_embedded_server()
            
        logger.info("FastAPI and backend server shut down gracefully")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application with MCP integration."""
    
    # Setup logging first
    setup_logging()
    
    # Create FastAPI app
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Autonomous SQL Investigation Agent Backend with MCP Tools",
        lifespan=lifespan,
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
    )
    
    # Add middleware
    app.add_middleware(
        CORSMiddleware,
        **get_cors_config()
    )
    
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*"] if settings.debug else ["localhost", "127.0.0.1"]
    )
    
    # Setup monitoring
    setup_monitoring(app)
    
    # Include API routes
    app.include_router(
        investigations.router,
        prefix=f"{settings.api_prefix}/investigations",
        tags=["Investigations"]
    )
    
    app.include_router(
        database.router,
        prefix=f"{settings.api_prefix}/database",
        tags=["Database"]
    )
    
    app.include_router(
        sessions.router,
        prefix=f"{settings.api_prefix}/sessions",
        tags=["Sessions"]
    )
    
    # WebSocket routes
    app.include_router(websocket_router)
    
    # Health check endpoint
    @app.get("/health")
    async def health_check():
        """Health check endpoint for monitoring."""
        service_bridge = get_service_bridge()
        is_healthy = await service_bridge.is_healthy()
        
        return {
            "status": "healthy" if is_healthy else "unhealthy",
            "version": settings.app_version,
            "services": {
                "backend_server": is_healthy,
                "service_bridge": "embedded"
            }
        }
    
    @app.get("/")
    async def root():
        """Root endpoint with API information."""
        return {
            "message": "Agentic SQL Backend API",
            "version": settings.app_version,
            "docs": "/docs",
            "health": "/health",
            "mcp_endpoint": "/mcp"
        }
    
    # Mount MCP tools for agent communication
    try:
        from fastapi_mcp import FastAPIMCP
        
        mcp = FastAPIMCP(app)
        mcp.mount()
        logger.info("MCP tools mounted at /mcp endpoint")
        
    except ImportError:
        logger.warning("FastAPI-MCP not available, skipping MCP integration")
    except Exception as e:
        logger.error(f"Failed to mount MCP tools: {e}")
    
    return app


def get_service_bridge_instance():
    """Get the global service bridge instance for FastAPI routes."""
    return get_service_bridge()


def get_backend_server():
    """Get the embedded backend server instance."""
    return embedded_server