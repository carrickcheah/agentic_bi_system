#!/usr/bin/env python3
"""
Simple MariaDB MCP Connection Test

Tests MariaDB database connectivity via existing MCP infrastructure.
"""

import asyncio
import sys
from pathlib import Path

# Add app directory to path
app_dir = Path(__file__).parent.parent
sys.path.insert(0, str(app_dir))

try:
    from fastmcp.client_manager import MCPClientManager
    from config.cfg_databases import DatabaseConfig
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the app directory with proper dependencies")
    sys.exit(1)


async def test_mariadb_connection():
    """Test MariaDB MCP connection."""
    print("🧪 Testing MariaDB MCP Connection")
    print("=" * 50)
    
    manager = None
    try:
        # Initialize MCP client manager
        print("🔌 Initializing MCP client manager...")
        manager = MCPClientManager()
        await manager.initialize()
        print("✅ MCP client manager initialized")
        
        # Check if MariaDB client is available
        if not manager.mariadb:
            print("❌ MariaDB MCP client not available")
            print("   Check your mcp.json configuration and environment variables")
            return False
        
        print("✅ MariaDB MCP client available")
        
        # Test basic connection
        print("\n📊 Testing basic connection...")
        is_connected = await manager.mariadb.test_connection()
        
        if not is_connected:
            print("❌ MariaDB connection test failed")
            return False
        
        print("✅ MariaDB connection successful")
        
        # Test listing tables
        print("\n📋 Testing table listing...")
        try:
            tables = await manager.mariadb.list_tables()
            print(f"✅ Found {len(tables)} tables: {tables}")
        except Exception as e:
            print(f"⚠️ Table listing failed: {e}")
        
        # Test simple query
        print("\n🔍 Testing simple query...")
        try:
            result = await manager.mariadb.execute_query("SELECT DATABASE(), USER(), VERSION() as version")
            print(f"✅ Query successful:")
            print(f"   Database: {result.data}")
            print(f"   Columns: {result.columns}")
            print(f"   Rows: {result.row_count}")
        except Exception as e:
            print(f"⚠️ Query failed: {e}")
        
        print("\n🎉 MariaDB MCP connection test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return False
        
    finally:
        if manager:
            try:
                await manager.close()
                print("🔌 MCP connections closed")
            except Exception as e:
                print(f"⚠️ Error closing connections: {e}")


async def main():
    """Main function."""
    try:
        success = await test_mariadb_connection()
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())