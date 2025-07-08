from model import ModelManager
from utils import QuestionChecker
from qdrant import get_qdrant_service

# Initialize core services synchronously
model_manager = ModelManager()
question_checker = QuestionChecker(model_manager)

# OpenAI embeddings are now handled within the model package

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
        print(f"  - qdrant_service: {qdrant_service}")
        
        print("\n✅ All services ready!")
    
    # Check for --skip-validation flag
    skip_validation = "--skip-validation" in sys.argv
    asyncio.run(startup(skip_validation))