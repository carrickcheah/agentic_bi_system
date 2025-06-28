#!/usr/bin/env python3
"""
Test MariaDB MCP Connection

Tests MariaDB database connectivity via MCP protocol.
Validates connection, basic queries, and schema inspection.
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List

# Add app directory to path for imports
app_dir = Path(__file__).parent.parent
sys.path.insert(0, str(app_dir))

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MariaDBMCPTester:
    """Test MariaDB connection via MCP protocol."""
    
    def __init__(self):
        self.session = None
        self.server_params = self._build_server_params()
        
    def _build_server_params(self) -> StdioServerParameters:
        """Build MCP server parameters for MariaDB."""
        
        # Load environment variables
        env_file = app_dir / ".env"
        if env_file.exists():
            self._load_env_file(env_file)
        
        return StdioServerParameters(
            command="npx",
            args=["-y", "mariadb-mcp-server"],
            env={
                "MARIADB_HOST": os.getenv("MARIADB_HOST"),
                "MARIADB_PORT": os.getenv("MARIADB_PORT"),
                "MARIADB_USER": os.getenv("MARIADB_USER"),
                "MARIADB_PASSWORD": os.getenv("MARIADB_PASSWORD"),
                "MARIADB_DATABASE": os.getenv("MARIADB_DATABASE"),
            }
        )
    
    def _load_env_file(self, env_file: Path):
        """Load environment variables from .env file."""
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()
    
    async def connect(self) -> bool:
        """Establish MCP connection to MariaDB."""
        try:
            print("🔌 Connecting to MariaDB MCP server...")
            print(f"   Host: {os.getenv('MARIADB_HOST')}:{os.getenv('MARIADB_PORT')}")
            print(f"   Database: {os.getenv('MARIADB_DATABASE')}")
            
            self.session = await stdio_client(self.server_params)
            await self.session.__aenter__()
            
            print("✅ MariaDB MCP connection established")
            return True
            
        except Exception as e:
            print(f"❌ Failed to connect to MariaDB MCP: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from MCP server."""
        if self.session:
            try:
                await self.session.__aexit__(None, None, None)
                print("🔌 Disconnected from MariaDB MCP")
            except Exception as e:
                print(f"⚠️ Error during disconnect: {e}")
    
    async def list_tools(self) -> List[str]:
        """List available MCP tools."""
        try:
            result = await self.session.list_tools()
            tools = [tool.name for tool in result.tools]
            print(f"🛠️ Available tools: {tools}")
            return tools
            
        except Exception as e:
            print(f"❌ Failed to list tools: {e}")
            return []
    
    async def test_basic_query(self) -> bool:
        """Test basic database query via MCP."""
        try:
            print("\n📊 Testing basic query...")
            
            # Try to list tables
            result = await self.session.call_tool(
                "mariadb-query",
                arguments={
                    "sql": "SHOW TABLES;"
                }
            )
            
            print("✅ Query executed successfully")
            print(f"   Result: {result.content}")
            return True
            
        except Exception as e:
            print(f"❌ Basic query failed: {e}")
            return False
    
    async def test_schema_inspection(self) -> bool:
        """Test database schema inspection via MCP."""
        try:
            print("\n🔍 Testing schema inspection...")
            
            # Get database information
            result = await self.session.call_tool(
                "mariadb-describe",
                arguments={
                    "table": None  # List all tables
                }
            )
            
            print("✅ Schema inspection successful")
            print(f"   Schema info: {result.content}")
            return True
            
        except Exception as e:
            print(f"❌ Schema inspection failed: {e}")
            return False
    
    async def test_connection_info(self) -> bool:
        """Test getting connection information."""
        try:
            print("\n📋 Testing connection info...")
            
            result = await self.session.call_tool(
                "mariadb-query",
                arguments={
                    "sql": "SELECT CONNECTION_ID(), USER(), DATABASE(), VERSION();"
                }
            )
            
            print("✅ Connection info retrieved")
            print(f"   Info: {result.content}")
            return True
            
        except Exception as e:
            print(f"❌ Connection info failed: {e}")
            return False
    
    async def run_comprehensive_test(self) -> Dict[str, bool]:
        """Run comprehensive MariaDB MCP test suite."""
        print("🧪 Starting MariaDB MCP Test Suite")
        print("=" * 50)
        
        results = {}
        
        # Test connection
        results["connection"] = await self.connect()
        if not results["connection"]:
            return results
        
        try:
            # Test tools listing
            tools = await self.list_tools()
            results["tools_listing"] = len(tools) > 0
            
            # Test basic query
            results["basic_query"] = await self.test_basic_query()
            
            # Test schema inspection  
            results["schema_inspection"] = await self.test_schema_inspection()
            
            # Test connection info
            results["connection_info"] = await self.test_connection_info()
            
        finally:
            # Always disconnect
            await self.disconnect()
        
        return results
    
    def print_results(self, results: Dict[str, bool]):
        """Print test results summary."""
        print("\n" + "=" * 50)
        print("📋 TEST RESULTS SUMMARY")
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
        
        if all_passed:
            print("\n🎉 MariaDB MCP connection is working perfectly!")
        else:
            print("\n🔧 Check your MariaDB configuration and network connectivity.")


async def main():
    """Main test function."""
    tester = MariaDBMCPTester()
    
    try:
        results = await tester.run_comprehensive_test()
        tester.print_results(results)
        
        # Exit with appropriate code
        exit_code = 0 if all(results.values()) else 1
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())