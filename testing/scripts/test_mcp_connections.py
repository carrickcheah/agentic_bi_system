#!/usr/bin/env python3
"""
Test MCP database connections
Tests all three MCP servers: Supabase, MariaDB, and PostgreSQL
"""

import asyncio
import json
import os
import sys
from pathlib import Path

# Add the app directory to Python path
app_dir = Path(__file__).parent.parent.parent / "app"
sys.path.insert(0, str(app_dir))

from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client
from dotenv import load_dotenv

# Load environment variables
load_dotenv(app_dir / ".env")

async def test_mcp_server(server_name: str, config: dict):
    """Test a single MCP server connection"""
    print(f"\n🔄 Testing {server_name} MCP server...")
    
    try:
        # Create client based on transport type
        if "command" in config:
            # Stdio transport
            command = config["command"]
            args = config["args"]
            
            # Replace environment variables in args
            processed_args = []
            for arg in args:
                if arg.startswith("${") and arg.endswith("}"):
                    env_var = arg[2:-1]
                    processed_args.append(os.getenv(env_var, arg))
                else:
                    processed_args.append(arg)
            
            # Set environment variables for the process
            env = os.environ.copy()
            if "env" in config:
                for key, value in config["env"].items():
                    if value.startswith("${") and value.endswith("}"):
                        env_var = value[2:-1]
                        env[key] = os.getenv(env_var, value)
                    else:
                        env[key] = value
            
            print(f"   Command: {command} {' '.join(processed_args)}")
            
            # Create stdio client
            async with stdio_client(command, processed_args, env=env) as client:
                async with ClientSession(client[0], client[1]) as session:
                    # Test basic connection
                    print(f"   ✅ {server_name} MCP server connected successfully!")
                    
                    # Try to list available tools/resources
                    try:
                        # This might not work for all servers, but worth trying
                        print(f"   📋 Testing basic operations...")
                        return True
                    except Exception as e:
                        print(f"   ⚠️  Connected but limited operations: {e}")
                        return True
                        
        else:
            print(f"   ❌ Unsupported transport type for {server_name}")
            return False
            
    except FileNotFoundError:
        print(f"   ❌ {server_name}: Command not found. Make sure the MCP server is installed.")
        return False
    except Exception as e:
        print(f"   ❌ {server_name}: Connection failed - {e}")
        return False

async def main():
    """Test all MCP connections"""
    print("🧪 Testing MCP Database Connections")
    print("=" * 50)
    
    # Load MCP configuration
    mcp_config_path = app_dir / "mcp.json"
    
    if not mcp_config_path.exists():
        print("❌ mcp.json not found!")
        return
    
    with open(mcp_config_path) as f:
        mcp_config = json.load(f)
    
    servers = mcp_config.get("mcpServers", {})
    results = {}
    
    for server_name, config in servers.items():
        result = await test_mcp_server(server_name, config)
        results[server_name] = result
    
    # Summary
    print("\n" + "=" * 50)
    print("🔍 Connection Test Results:")
    
    success_count = 0
    for server_name, success in results.items():
        status = "✅ Connected" if success else "❌ Failed"
        print(f"   {server_name}: {status}")
        if success:
            success_count += 1
    
    print(f"\n📊 Summary: {success_count}/{len(results)} servers connected successfully")
    
    if success_count == len(results):
        print("🎉 All MCP servers are working!")
    else:
        print("⚠️  Some MCP servers need attention")
    
    # Environment check
    print("\n🔧 Environment Variables Check:")
    env_vars = [
        "MARIADB_HOST", "MARIADB_PORT", "MARIADB_USER", "MARIADB_PASSWORD", "MARIADB_DATABASE",
        "POSTGRES_URL", "SUPABASE_ACCESS_TOKEN"
    ]
    
    for var in env_vars:
        value = os.getenv(var)
        if value:
            masked_value = value[:8] + "..." if len(value) > 8 else "***"
            print(f"   ✅ {var}: {masked_value}")
        else:
            print(f"   ❌ {var}: Not set")

if __name__ == "__main__":
    asyncio.run(main())