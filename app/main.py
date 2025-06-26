"""
FastAPI Entry Point - Autonomous Business Intelligence System

Main application entry point that initializes the world-class business intelligence
platform with autonomous investigation capabilities.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from .fastapi.app_factory import create_app
from .config import settings
from .utils.logging import logger
from .fastmcp.client_manager import MCPClientManager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    # Startup
    logger.info("🚀 Starting Autonomous Business Intelligence System")
    
    # Initialize MCP client manager
    mcp_manager = MCPClientManager()
    await mcp_manager.initialize()
    app.state.mcp_manager = mcp_manager
    
    logger.info("✅ System initialization completed")
    
    yield
    
    # Shutdown
    logger.info("🔄 Shutting down Business Intelligence System")
    if hasattr(app.state, 'mcp_manager'):
        await app.state.mcp_manager.cleanup()
    logger.info("✅ Shutdown completed")


def create_application() -> FastAPI:
    """Create and configure the FastAPI application."""
    # Create FastAPI app with lifespan
    app = create_app(lifespan=lifespan)
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=getattr(settings, 'cors_origins', ["*"]),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    return app


# Create the application instance
app = create_application()


def main():
    """Run the application."""
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()