#!/usr/bin/env python3
"""
Test Qdrant MCP functionality - store and retrieve vectors
"""

import asyncio
import os
import sys
from pathlib import Path

# Add app directory to path (we're now inside app/testing/scripts/)
app_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(app_dir))

from mcp.client_manager import MCPClientManager
from config import settings


async def test_qdrant_operations():
    """Test Qdrant store and find operations"""
    print("🧪 Testing Qdrant MCP Functionality")
    print("=" * 50)
    
    # Initialize MCP client manager
    client_manager = MCPClientManager(settings)
    
    try:
        # Initialize clients
        print("\n1️⃣ Initializing MCP clients...")
        await client_manager.initialize()
        print("✅ MCP clients initialized")
        
        # Get Qdrant client
        qdrant = client_manager.qdrant_client
        if not qdrant:
            print("❌ Qdrant client not available")
            return
            
        # Test 1: Store a SQL query pattern
        print("\n2️⃣ Storing SQL query pattern...")
        store_result = await qdrant.store_sql_query(
            sql_query="SELECT COUNT(*) FROM customers WHERE created_at > '2024-01-01'",
            description="Count customers created after January 2024",
            result_summary="Found 1,234 new customers",
            execution_time=0.125,
            success=True
        )
        print(f"✅ Stored SQL query: {store_result}")
        
        # Test 2: Find similar queries
        print("\n3️⃣ Finding similar queries...")
        similar_queries = await qdrant.find_similar_queries(
            description="Count customers from this year",
            limit=3
        )
        print(f"✅ Found {len(similar_queries)} similar queries")
        for i, query in enumerate(similar_queries, 1):
            print(f"   {i}. {query.get('metadata', {}).get('description', 'No description')}")
        
        # Test 3: Store an error pattern
        print("\n4️⃣ Storing error pattern...")
        error_result = await qdrant.store_error_pattern(
            error_message="Column 'created_at' doesn't exist",
            solution="Use 'creation_date' column instead",
            sql_query="SELECT * FROM users WHERE created_at > '2024-01-01'",
            error_type="ColumnNotFound"
        )
        print(f"✅ Stored error pattern: {error_result}")
        
        # Test 4: Find error solutions
        print("\n5️⃣ Finding error solutions...")
        solutions = await qdrant.find_error_solutions(
            error_message="Column doesn't exist",
            limit=2
        )
        print(f"✅ Found {len(solutions)} solutions")
        for i, solution in enumerate(solutions, 1):
            print(f"   {i}. {solution.get('metadata', {}).get('solution', 'No solution')}")
        
        # Test 5: Store schema information
        print("\n6️⃣ Storing schema information...")
        schema_result = await qdrant.store_schema_info(
            table_name="customers",
            schema_description="Customer information including contact details and purchase history",
            columns=[
                {"name": "id", "type": "INTEGER", "primary": True},
                {"name": "name", "type": "VARCHAR(255)"},
                {"name": "email", "type": "VARCHAR(255)", "unique": True},
                {"name": "creation_date", "type": "DATETIME"},
                {"name": "total_purchases", "type": "DECIMAL(10,2)"}
            ],
            database="mariadb"
        )
        print(f"✅ Stored schema info: {schema_result}")
        
        # Test 6: Find relevant tables
        print("\n7️⃣ Finding relevant tables...")
        tables = await qdrant.find_relevant_tables(
            description="customer purchase information",
            limit=3
        )
        print(f"✅ Found {len(tables)} relevant tables")
        for i, table in enumerate(tables, 1):
            print(f"   {i}. {table.get('metadata', {}).get('table_name', 'Unknown table')}")
        
        print("\n✨ All Qdrant operations completed successfully!")
        print("🎯 Your Qdrant vector database is fully operational!")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Cleanup
        await client_manager.cleanup()
        print("\n🧹 Cleanup completed")


if __name__ == "__main__":
    asyncio.run(test_qdrant_operations())