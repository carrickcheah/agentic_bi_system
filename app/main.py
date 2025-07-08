from model import ModelManager
from utils import QuestionChecker
from qdrant import get_qdrant_service
from fastmcp.client_manager import MCPClientManager
from cache import CacheManager

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
        await mcp_client_manager.initialize()
        _mcp_initialized = True
    
    return mcp_client_manager

async def get_cache_manager():
    """Get or initialize cache manager on demand."""
    global cache_manager, _cache_initialized
    
    if not _cache_initialized:
        # Cache depends on MCP for PostgreSQL
        mcp = await get_mcp_client_manager()
        
        cache_manager = CacheManager()
        await cache_manager.initialize()
        
        # Inject PostgreSQL client into cache tiers that need it
        if mcp.postgres:
            cache_manager.postgresql_cache.postgres_client = mcp.postgres
            cache_manager.semantic_cache.postgres_client = mcp.postgres
        
        _cache_initialized = True
    
    return cache_manager

# Export all services for other modules
__all__ = [
    "model_manager", 
    "question_checker", 
    "qdrant_service",
    "get_mcp_client_manager",
    "get_cache_manager",
    "initialize_async_services"
]





if __name__ == "__main__":
    # Validate and initialize all services when running directly
    import asyncio
    import sys
    
    async def startup(skip_validation=False):
        """Initialize and validate all services."""
        if not skip_validation:
            # Validate models
            print("🔍 Validating model API keys...")
            await model_manager.validate_models()
            print("✅ Models validated")
        else:
            print("⏭️  Skipping model validation")
        
        # Initialize only essential services
        print("🔌 Initializing services...")
        await initialize_async_services()
        print("✅ Qdrant initialized")
        
        # Show available services
        print("\n📦 Available Services:")
        print(f"  - model_manager: {model_manager}")
        print(f"  - question_checker: {question_checker}")
        print(f"  - qdrant_service: {qdrant_service}")
        print(f"  - get_mcp_client_manager: <lazy initialization>")
        print(f"  - get_cache_manager: <lazy initialization>")
        
        print("\n✅ All services ready!")
        print("💡 MCP and Cache will be initialized on first use")
    
    # Check for --skip-validation flag
    skip_validation = "--skip-validation" in sys.argv
    
    try:
        asyncio.run(startup(skip_validation))
    except KeyboardInterrupt:
        print("\n\nShutting down gracefully...")