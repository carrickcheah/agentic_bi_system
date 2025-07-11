"""
Main orchestration entry point for Agentic BI.
Initializes core services and provides lazy initialization for async components.
"""

from model import ModelManager
from utils import QuestionChecker
from qdrant import get_qdrant_service
from fastmcp.client_manager import MCPClientManager
from cache import CacheManager
import logging

logger = logging.getLogger(__name__)

# Initialize core services synchronously
# OpenAI embeddings are now handled within the model package
model_manager = ModelManager()
question_checker = QuestionChecker(model_manager)

# Initialize Qdrant (async initialization handled lazily)
qdrant_service = None

# Initialize FastMCP (async initialization handled lazily)
mcp_client_manager = None
_mcp_initialized = False

# Initialize Cache (async initialization handled lazily)
cache_manager = None
_cache_initialized = False


async def initialize_async_services():
    """Initialize services that require async setup."""
    global qdrant_service
    
    # Initialize Qdrant with vector search
    qdrant_service = await get_qdrant_service()
    
    return qdrant_service


async def get_mcp_client_manager():
    """Get or initialize MCP client manager on demand."""
    global mcp_client_manager, _mcp_initialized
    
    if not _mcp_initialized:
        mcp_client_manager = MCPClientManager()
        # Don't initialize any services by default - let Phase 3 decide
        _mcp_initialized = True
    
    return mcp_client_manager


async def get_cache_manager():
    """Get or initialize cache manager on demand."""
    global cache_manager, _cache_initialized
    
    if not _cache_initialized:
        # Initialize cache manager with Anthropic cache only
        # No MCP needed for caching - PostgreSQL is only for chat history
        cache_manager = CacheManager()
        await cache_manager.initialize()
        
        _cache_initialized = True
    
    return cache_manager


# Import high-level interfaces
from core import AgenticWorkflow, process_question

# Export all services for other modules
__all__ = [
    "model_manager", 
    "question_checker", 
    "qdrant_service",
    "get_mcp_client_manager",
    "get_cache_manager",
    "initialize_async_services",
    "AgenticWorkflow",
    "process_question"
]


if __name__ == "__main__":
    # Validate and initialize all services when running directly
    import asyncio
    import sys
    
    async def startup(skip_validation=False, chat_mode=False):
        """Initialize and validate all services."""
        if not skip_validation:
            # Validate models
            logger.info("🔍 Validating model API keys...")
            await model_manager.validate_models()
            logger.info("✅ Models validated")
        else:
            logger.info("⏭️  Skipping model validation")
        
        # Initialize only essential services
        logger.info("🔌 Initializing services...")
        await initialize_async_services()
        logger.info("✅ Qdrant initialized")
        
        if not chat_mode:
            # Show available services
            logger.info("\n📦 Available Services:")
            logger.info(f"  - model_manager: {model_manager}")
            logger.info(f"  - question_checker: {question_checker}")
            logger.info(f"  - qdrant_service: {qdrant_service}")
            logger.info(f"  - get_mcp_client_manager: <lazy initialization>")
            logger.info(f"  - get_cache_manager: <lazy initialization>")
            logger.info(f"  - AgenticWorkflow: <clean workflow controller>")
            logger.info(f"  - process_question: <simple interface function>")
            
            logger.info("\n✅ All services ready!")
            logger.info("💡 MCP and Cache will be initialized on first use")
            logger.info("\n💬 Run with --chat for interactive mode")
        else:
            # Start chat mode
            from cli import simple_chat
            await simple_chat()
    
    # Check for flags
    skip_validation = "--skip-validation" in sys.argv
    chat_mode = "--chat" in sys.argv
    
    try:
        asyncio.run(startup(skip_validation, chat_mode))
    except KeyboardInterrupt:
        print("\n\nShutting down gracefully...")