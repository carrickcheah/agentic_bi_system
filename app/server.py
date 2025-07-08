"""
replace ur  here.
"""

import asyncio
import signal
from typing import Optional
from contextlib import asynccontextmanager

try:
    # Try relative imports first (when used as module)
    from .config import settings
    from .utils.logging import setup_logger, logger
    from .fastmcp.client_manager import MCPClientManager
except ImportError:
    # Fall back to absolute imports (when run directly)
    from config import settings
    from utils.logging import setup_logger, logger
    from fastmcp.client_manager import MCPClientManager


class BackendService:
    """
    Standalone backend service for database operations.
    
    This service manages all MCP clients and provides business logic
    services independently of any HTTP framework.
    """
    
    def __init__(self):
        self.client_manager: Optional[MCPClientManager] = None
        self._running = False
        self._shutdown_event = asyncio.Event()
    
    async def initialize(self):
        """Initialize the FastMCP server and all services."""
        logger.info("Starting FastMCP Backend Service")
        
        try:
            # Initialize MCP client manager
            logger.info("Initializing MCP client connections...")
            self.client_manager = MCPClientManager()
            await self.client_manager.initialize()
            
            logger.info("FastMCP Backend Service initialized successfully!")
            self._running = True
            
        except Exception as e:
            logger.error(f"Failed to initialize FastMCP service: {e}")
            await self.cleanup()
            raise
    
    async def cleanup(self):
        """Clean up all resources."""
        logger.info("Shutting down FastMCP Backend Service")
        
        self._running = False
        
        try:
            if self.client_manager:
                await self.client_manager.close()
                logger.info("MCP client connections closed")
                
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
        
        logger.info("FastMCP Backend Service shutdown completed")
    
    async def run_forever(self):
        """Run the server until shutdown signal."""
        if not self._running:
            await self.initialize()
        
        logger.info("FastMCP Backend Service is running...")
        logger.info("Press Ctrl+C to shutdown")
        
        # Setup signal handlers
        def signal_handler():
            logger.info("Received shutdown signal")
            self._shutdown_event.set()
        
        # Register signal handlers
        for sig in (signal.SIGTERM, signal.SIGINT):
            asyncio.get_event_loop().add_signal_handler(sig, signal_handler)
        
        try:
            # Wait for shutdown signal
            await self._shutdown_event.wait()
        finally:
            await self.cleanup()
    
    def is_healthy(self) -> bool:
        """Check if the server is healthy."""
        return (
            self._running and 
            self.client_manager and 
            self.client_manager.is_healthy()
        )
    
    def get_client_manager(self) -> Optional[MCPClientManager]:
        """Get the MCP client manager instance."""
        return self.client_manager


# Global server instance for embedded mode
_global_server: Optional[BackendService] = None


@asynccontextmanager
async def get_fastmcp_server():
    """
    Context manager to get a FastMCP server instance.
    
    This can be used for embedded mode within FastAPI or for
    standalone service access.
    """
    global _global_server
    
    if _global_server is None:
        _global_server = BackendService()
        await _global_server.initialize()
    
    try:
        yield _global_server
    finally:
        # Don't cleanup in embedded mode - let FastAPI handle lifecycle
        pass


async def get_embedded_client_manager() -> Optional[MCPClientManager]:
    """
    Get the embedded MCP client manager for use within FastAPI.
    
    Returns None if not initialized.
    """
    global _global_server
    if _global_server and _global_server.is_healthy():
        return _global_server.get_client_manager()
    return None


async def initialize_embedded_server():
    """Initialize the embedded backend server for use within FastAPI."""
    global _global_server
    if _global_server is None:
        _global_server = BackendService()
        await _global_server.initialize()
    return _global_server


async def cleanup_embedded_server():
    """Cleanup the embedded backend server."""
    global _global_server
    if _global_server:
        await _global_server.cleanup()
        _global_server = None


def create_standalone_server() -> BackendService:
    """Create a new standalone backend server instance."""
    return BackendService()


async def main():
    """Main entry point for standalone FastMCP server."""
    # Setup logging
    setup_logging()
    
    logger.info("Starting FastMCP Backend Service in standalone mode")
    logger.info(f"Configuration: {settings.app_name} v{settings.app_version}")
    
    # Create and run server
    server = create_standalone_server()
    
    try:
        await server.run_forever()
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user")
    except Exception as e:
        logger.error(f"FastMCP server error: {e}")
        raise
    finally:
        logger.info("FastMCP Backend Service stopped")


if __name__ == "__main__":
    """Run as standalone server."""
    asyncio.run(main())