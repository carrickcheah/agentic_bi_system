from model import ModelManager
from utils import QuestionChecker
from qdrant import get_qdrant_service
from fastmcp.client_manager import MCPClientManager

# Initialize core services synchronously
# OpenAI embeddings are now handled within the model package
model_manager = ModelManager()
question_checker = QuestionChecker(model_manager)


# Initialize Qdrant (async initialization handled lazily)
qdrant_service = None

# Initialize FastMCP (async initialization handled lazily)
mcp_client_manager = None

async def initialize_async_services():
    """Initialize services that require async setup."""
    global qdrant_service, mcp_client_manager, business_service
    
    # Initialize Qdrant with vector search
    qdrant_service = await get_qdrant_service()
    
    # Initialize FastMCP
    mcp_client_manager = MCPClientManager()
    await mcp_client_manager.initialize()
    
    # Initialize Business Service
    business_service = BusinessService(mcp_client_manager)
    await business_service.initialize()
    
    return qdrant_service, mcp_client_manager, business_service

# Export all services for other modules
__all__ = [
    "model_manager", 
    "question_checker", 
    "qdrant_service",
    "mcp_client_manager",
    "business_service",
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
        
        # Initialize async services
        print("🔌 Initializing async services...")
        await initialize_async_services()
        print("✅ Qdrant initialized")
        print("✅ FastMCP initialized")
        
        # Show all available services
        print("\n📦 Available Services:")
        print(f"  - model_manager: {model_manager}")
        print(f"  - question_checker: {question_checker}")
        print(f"  - qdrant_service: {qdrant_service}")
        print(f"  - mcp_client_manager: {mcp_client_manager}")
        print(f"  - business_service: {business_service}")
        
        print("\n✅ All services ready!")
    
    # Check for --skip-validation flag
    skip_validation = "--skip-validation" in sys.argv
    asyncio.run(startup(skip_validation))