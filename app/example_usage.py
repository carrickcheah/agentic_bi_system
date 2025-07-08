"""
Example of production usage with lazy MCP initialization.
"""
import asyncio
from main import model_manager, get_mcp_client_manager, initialize_async_services
from investigation import InvestigationRunner

async def run_investigation_example():
    """Example of running an investigation with lazy MCP initialization."""
    
    # Initialize core services (no MCP yet)
    print("Initializing core services...")
    await initialize_async_services()
    
    # Your business logic here...
    business_question = "What are the top selling products this month?"
    
    # Only when you need database access, initialize MCP
    print("\nInitializing MCP for database access...")
    mcp_manager = await get_mcp_client_manager()
    print("✅ MCP initialized successfully")
    
    # Now you can use it for investigation
    coordinated_services = {
        "mariadb": mcp_manager.mariadb,
        "model": model_manager
    }
    
    investigation = InvestigationRunner(
        coordinated_services=coordinated_services,
        investigation_request=business_question,
        execution_context={"complexity": 0.3},
        mcp_client_manager=mcp_manager
    )
    
    # Run investigation
    results = await investigation.execute()
    print(f"\nInvestigation completed: {results.investigation_id}")
    
    # Clean up MCP when done
    await mcp_manager.close()

async def simple_query_example():
    """Example of simple database query with lazy MCP."""
    
    # Get MCP manager when needed
    mcp = await get_mcp_client_manager()
    
    # Execute query
    result = await mcp.mariadb.execute_query("SELECT COUNT(*) FROM customers")
    print(f"Customer count: {result}")
    
    # Clean up
    await mcp.close()

if __name__ == "__main__":
    # Run without MCP errors
    asyncio.run(run_investigation_example())