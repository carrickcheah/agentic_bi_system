"""
FastAPI Application Factory with MCP Integration

This module creates and configures the FastAPI application with:
- All route handlers
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
from ..database.connections import DatabaseManager


# Global database manager
db_manager: DatabaseManager = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan management.
    
    Handles startup and shutdown of database connections and other services.
    """
    global db_manager
    
    # Startup
    logger.info("Starting Agentic SQL Backend...")
    
    try:
        # Initialize database connections
        logger.info("Initializing database connections...")
        db_manager = DatabaseManager()
        await db_manager.initialize()
        
        logger.info("✅ All services initialized successfully!")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise
    finally:
        # Shutdown
        logger.info("Shutting down Agentic SQL Backend...")
        
        if db_manager:
            await db_manager.close()
            
        logger.info("✅ All services shut down gracefully")


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
        return {
            "status": "healthy",
            "version": settings.app_version,
            "services": {
                "database": db_manager.is_healthy() if db_manager else False,
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
        logger.info("✅ MCP tools mounted at /mcp endpoint")
        
    except ImportError:
        logger.warning("FastAPI-MCP not available, skipping MCP integration")
    except Exception as e:
        logger.error(f"Failed to mount MCP tools: {e}")
    
    return app


def get_database_manager() -> DatabaseManager:
    """Get the global database manager instance."""
    return db_manager