from model import ModelManager
from utils import QuestionChecker
from embedded_model import get_embedding_manager
from qdrant import get_qdrant_service

# Initialize core services synchronously
model_manager = ModelManager()
question_checker = QuestionChecker(model_manager)

# Initialize embedding manager (may download large model on first run)
try:
    embedding_manager = get_embedding_manager()
except Exception as e:
    print(f"⚠️  Embedding model initialization failed: {e}")
    print("   (This may be due to model download on first run)")
    embedding_manager = None

# Initialize Qdrant (async initialization handled lazily)
qdrant_service = None

async def initialize_async_services():
    """Initialize services that require async setup."""
    global qdrant_service
    
    # Initialize Qdrant with vector search
    qdrant_service = await get_qdrant_service()
    
    return qdrant_service

# Export all services for other modules
__all__ = [
    "model_manager", 
    "question_checker", 
    "embedding_manager",
    "qdrant_service",
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
        print("🔌 Initializing Qdrant...")
        await initialize_async_services()
        print("✅ Qdrant initialized")
        
        # Show all available services
        print("\n📦 Available Services:")
        print(f"  - model_manager: {model_manager}")
        print(f"  - question_checker: {question_checker}")
        print(f"  - embedding_manager: {embedding_manager}")
        print(f"  - qdrant_service: {qdrant_service}")
        
        print("\n✅ All services ready!")
    
    # Check for --skip-validation flag
    skip_validation = "--skip-validation" in sys.argv
    asyncio.run(startup(skip_validation))