#!/usr/bin/env python3
"""
MariaDB MCP Client Test

Tests MariaDB connection using Python MCP client.
"""

import asyncio
import json
import os
from pathlib import Path
from typing import Dict, Any

# Import MCP client
try:
    from mcp.client.session import ClientSession
    from mcp.client.stdio import stdio_client
    from mcp.types import StdioServerParameters
except ImportError:
    print("❌ MCP client not available. Install with: uv add 'mcp[cli]'")
    exit(1)


def load_env_vars() -> Dict[str, str]:
    """Load environment variables from .env file."""
    env_file = Path(__file__).parent.parent / ".env"
    env_vars = {}
    
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key.strip()] = value.strip()
    
    return env_vars


class MariaDBMCPTester:
    """Test MariaDB MCP connection using Python client."""
    
    def __init__(self):
        self.env_vars = load_env_vars()
        self.session = None
        self.context = None
    
    async def connect(self) -> bool:
        """Connect to MariaDB MCP server."""
        try:
            print("🔌 Connecting to MariaDB MCP server...")
            
            # Create server parameters
            server_params = StdioServerParameters(
                command="npx",
                args=["-y", "mariadb-mcp-server"],
                env={
                    "MARIADB_HOST": self.env_vars.get("MARIADB_HOST"),
                    "MARIADB_PORT": self.env_vars.get("MARIADB_PORT"),
                    "MARIADB_USER": self.env_vars.get("MARIADB_USER"),
                    "MARIADB_PASSWORD": self.env_vars.get("MARIADB_PASSWORD"),
                    "MARIADB_DATABASE": self.env_vars.get("MARIADB_DATABASE"),
                }
            )
            
            # Create client context
            self.context = stdio_client(server_params)
            
            # Start the client
            read_stream, write_stream = await self.context.__aenter__()
            self.session = ClientSession(read_stream, write_stream)
            
            print("✅ Connected to MariaDB MCP server")
            return True
            
        except Exception as e:
            print(f"❌ Failed to connect: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from MCP server."""
        try:
            if self.context:
                await self.context.__aexit__(None, None, None)
                print("🔌 Disconnected from MariaDB MCP server")
        except Exception as e:
            print(f"⚠️ Error during disconnect: {e}")
    
    async def list_tools(self) -> bool:
        """List available MCP tools."""
        try:
            print("\n🛠️ Listing available tools...")
            result = await self.session.list_tools()
            
            if result.tools:
                print("✅ Available tools:")
                for tool in result.tools:
                    print(f"   - {tool.name}: {tool.description}")
                return True
            else:
                print("⚠️ No tools available")
                return False
                
        except Exception as e:
            print(f"❌ Failed to list tools: {e}")
            return False
    
    async def test_query(self) -> bool:
        """Test a simple database query."""
        try:
            print("\n📊 Testing database query...")
            
            # Try to execute a simple query
            result = await self.session.call_tool(
                "query",
                arguments={
                    "sql": "SELECT DATABASE() as current_db, USER() as current_user, VERSION() as version"
                }
            )
            
            print("✅ Query executed successfully")
            print(f"   Result: {result.content}")
            return True
            
        except Exception as e:
            print(f"❌ Query failed: {e}")
            return False
    
    async def test_table_list(self) -> bool:
        """Test listing tables."""
        try:
            print("\n📋 Testing table listing...")
            
            result = await self.session.call_tool(
                "query",
                arguments={
                    "sql": "SHOW TABLES"
                }
            )
            
            print("✅ Table listing successful")
            print(f"   Result: {result.content}")
            return True
            
        except Exception as e:
            print(f"❌ Table listing failed: {e}")
            return False
    
    async def run_test_suite(self) -> Dict[str, bool]:
        """Run comprehensive test suite."""
        print("🧪 Starting MariaDB MCP Test Suite")
        print("=" * 50)
        
        results = {}
        
        # Test connection
        results["connection"] = await self.connect()
        if not results["connection"]:
            return results
        
        try:
            # Test tools listing
            results["tools_listing"] = await self.list_tools()
            
            # Test basic query
            results["basic_query"] = await self.test_query()
            
            # Test table listing
            results["table_listing"] = await self.test_table_list()
            
        finally:
            await self.disconnect()
        
        return results
    
    def print_results(self, results: Dict[str, bool]):
        """Print test results."""
        print("\n" + "=" * 50)
        print("📊 TEST RESULTS SUMMARY")
        print("=" * 50)
        
        all_passed = True
        for test_name, passed in results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{test_name.replace('_', ' ').title():.<40} {status}")
            if not passed:
                all_passed = False
        
        print("-" * 50)
        overall = "✅ ALL TESTS PASSED" if all_passed else "❌ SOME TESTS FAILED"
        print(f"Overall Status: {overall}")
        
        return all_passed


async def main():
    """Main test function."""
    try:
        tester = MariaDBMCPTester()
        
        print("📋 MariaDB Configuration:")
        print(f"   Host: {tester.env_vars.get('MARIADB_HOST', 'Not set')}")
        print(f"   Port: {tester.env_vars.get('MARIADB_PORT', 'Not set')}")
        print(f"   User: {tester.env_vars.get('MARIADB_USER', 'Not set')}")
        print(f"   Database: {tester.env_vars.get('MARIADB_DATABASE', 'Not set')}")
        print()
        
        results = await tester.run_test_suite()
        success = tester.print_results(results)
        
        if success:
            print("\n🎉 MariaDB MCP connection is working perfectly!")
        else:
            print("\n🔧 Some tests failed. Check your MariaDB configuration.")
        
        return success
        
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
        return False
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)