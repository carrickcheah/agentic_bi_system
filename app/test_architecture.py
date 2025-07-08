"""
Demo script showing the clean architecture with all services in main.py
"""

print("🏗️  Demonstrating Clean Architecture\n")

# Import all services from main
print("1️⃣  Importing services from main.py...")
from main import (
    model_manager, 
    question_checker, 
    embedding_manager,
    qdrant_service,
    initialize_async_services
)

print("✅ All services imported successfully!\n")

# Show what's available
print("2️⃣  Available Services:")
print(f"   - model_manager: {type(model_manager).__name__}")
print(f"   - question_checker: {type(question_checker).__name__}")
print(f"   - embedding_manager: {'Available' if embedding_manager else 'Not initialized (model download needed)'}")
print(f"   - qdrant_service: {'Not initialized yet (async)' if qdrant_service is None else 'Ready'}")

print("\n3️⃣  Service Capabilities:")
print(f"   - Models available: {model_manager.get_available_models()}")
print(f"   - Current model: {model_manager.get_current_model()}")

print("\n✨ Benefits of this architecture:")
print("   1. All services defined in ONE place (main.py)")
print("   2. No hidden initializations in other modules")
print("   3. Clear dependency graph")
print("   4. Easy to mock for testing")
print("   5. Simple imports for any module")

print("\n📝 Example usage in another module:")
print("""
from main import model_manager, qdrant_service

# Everything is already initialized!
response = await model_manager.generate_response("Hello")
results = await qdrant_service.search_similar_queries("sales data")
""")

print("\n✅ Clean architecture achieved!")