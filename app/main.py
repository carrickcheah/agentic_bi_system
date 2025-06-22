"""
FastAPI Application Entry Point for Agentic SQL Backend

This module sets up the main FastAPI application with all routes, middleware,
and lifecycle management for the autonomous SQL investigation agent.
"""

import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

from .config import settings, get_cors_config
from .utils.logging import setup_logging, logger
from .utils.monitoring import setup_monitoring
from .api.routes import agent, investigation, faq
from .api.websocket import websocket_router
from .database.connections import DatabaseManager
from .mcp.server import mcp_server
from .model.embeddings import EmbeddingService
from .rag.qdrant_client import QdrantManager


# Global managers
db_manager: DatabaseManager = None
embedding_service: EmbeddingService = None
qdrant_manager: QdrantManager = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan management.
    
    Handles startup and shutdown of all services including:
    - Database connections
    - Vector database
    - Embedding models
    - MCP server
    """
    global db_manager, embedding_service, qdrant_manager
    
    # Startup
    logger.info("Starting Agentic SQL Backend...")
    
    try:
        # Initialize database connections
        logger.info("Initializing database connections...")
        db_manager = DatabaseManager()
        await db_manager.initialize()
        
        # Initialize embedding service
        logger.info("Loading BGE-M3 embedding model...")
        embedding_service = EmbeddingService()
        await embedding_service.initialize()
        
        # Initialize Qdrant
        logger.info("Connecting to Qdrant vector database...")
        qdrant_manager = QdrantManager()
        await qdrant_manager.initialize()
        
        # Start MCP server
        logger.info("Starting MCP tool server...")
        # MCP server will be started separately
        
        logger.info("✅ All services initialized successfully!")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise
    finally:
        # Shutdown
        logger.info("Shutting down Agentic SQL Backend...")
        
        if qdrant_manager:
            await qdrant_manager.close()
            
        if embedding_service:
            await embedding_service.close()
            
        if db_manager:
            await db_manager.close()
            
        logger.info("✅ All services shut down gracefully")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    
    # Setup logging first
    setup_logging()
    
    # Create FastAPI app
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Autonomous SQL Investigation Agent Backend",
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
    
    # Include routers
    app.include_router(
        agent.router,
        prefix=f"{settings.api_prefix}/agent",
        tags=["Agent"]
    )
    
    app.include_router(
        investigation.router,
        prefix=f"{settings.api_prefix}/investigation",
        tags=["Investigation"]
    )
    
    app.include_router(
        faq.router,
        prefix=f"{settings.api_prefix}/faq",
        tags=["FAQ"]
    )
    
    # WebSocket routes
    app.include_router(websocket_router)
    
    # Health check endpoint
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "version": settings.app_version,
            "services": {
                "database": db_manager.is_healthy() if db_manager else False,
                "embeddings": embedding_service.is_ready() if embedding_service else False,
                "qdrant": qdrant_manager.is_connected() if qdrant_manager else False,
            }
        }
    
    @app.get("/")
    async def root():
        """Root endpoint with basic info."""
        return {
            "message": "Agentic SQL Backend API",
            "version": settings.app_version,
            "docs": "/docs",
            "health": "/health"
        }
    
    return app


# Create the app instance
app = create_app()


def main():
    """Run the application."""
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.reload,
        workers=settings.workers,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()