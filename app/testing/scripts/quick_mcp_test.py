#!/usr/bin/env python3
"""
Quick MCP configuration test
Validates mcp.json and environment variables
"""

import json
import os
import subprocess
import sys
from pathlib import Path

def test_mcp_config():
    """Test MCP configuration without actual connections"""
    print("🧪 Quick MCP Configuration Test")
    print("=" * 40)
    
    # Check if we're in the right directory
    app_dir = Path(__file__).parent.parent.parent / "app"
    
    # Load environment variables
    env_file = app_dir / ".env"
    if env_file.exists():
        print("✅ .env file found")
        # Simple env file parser
        env_vars = {}
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key] = value
                    os.environ[key] = value
    else:
        print("❌ .env file not found")
        return False
    
    # Load MCP configuration
    mcp_config_path = app_dir / "mcp.json"
    if not mcp_config_path.exists():
        print("❌ mcp.json not found")
        return False
    
    print("✅ mcp.json file found")
    
    with open(mcp_config_path) as f:
        mcp_config = json.load(f)
    
    servers = mcp_config.get("mcpServers", {})
    print(f"📊 Found {len(servers)} MCP servers configured")
    
    # Test each server configuration
    for server_name, config in servers.items():
        print(f"\n🔍 Testing {server_name} configuration:")
        
        # Check command exists
        if "command" in config:
            command = config["command"]
            try:
                result = subprocess.run([command, "--version"], 
                                      capture_output=True, 
                                      text=True, 
                                      timeout=5)
                if result.returncode == 0:
                    print(f"   ✅ Command '{command}' is available")
                else:
                    print(f"   ⚠️  Command '{command}' found but version check failed")
            except (subprocess.TimeoutExpired, FileNotFoundError):
                print(f"   ❌ Command '{command}' not found or not responding")
        
        # Check environment variables
        if "env" in config:
            for env_key, env_value in config["env"].items():
                if env_value.startswith("${") and env_value.endswith("}"):
                    var_name = env_value[2:-1]
                    if var_name in env_vars:
                        masked = env_vars[var_name][:8] + "..." if len(env_vars[var_name]) > 8 else "***"
                        print(f"   ✅ {var_name}: {masked}")
                    else:
                        print(f"   ❌ {var_name}: Missing from .env")
        
        # Check args with env var substitution
        if "args" in config:
            processed_args = []
            for arg in config["args"]:
                if arg.startswith("${") and arg.endswith("}"):
                    var_name = arg[2:-1]
                    if var_name in env_vars:
                        processed_args.append(env_vars[var_name])
                    else:
                        processed_args.append(f"MISSING_{var_name}")
                else:
                    processed_args.append(arg)
            
            print(f"   📋 Command will be: {config['command']} {' '.join(processed_args[:3])}...")
    
    print("\n" + "=" * 40)
    print("✅ Configuration test completed!")
    print("\n💡 To test actual connections, you'll need the MCP servers installed:")
    print("   - npm install -g @modelcontextprotocol/server-postgres")
    print("   - npm install -g @supabase/mcp-server-supabase")
    print("   - Install mcp-server-mariadb-vector")
    
    return True

if __name__ == "__main__":
    test_mcp_config()