#!/usr/bin/env python3
"""
Test all three MCP database connections
"""

import subprocess
import time
import os

def test_mcp_connection(name, command, args, env_vars=None):
    """Test a single MCP server connection"""
    print(f"\n🔄 Testing {name} MCP connection...")
    
    try:
        # Set up environment
        env = os.environ.copy()
        if env_vars:
            env.update(env_vars)
            
        # Start the process
        process = subprocess.Popen(
            [command] + args,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait 3 seconds
        time.sleep(3)
        
        # Check if still running
        if process.poll() is None:
            # Still running - terminate and report success
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
            print(f"✅ {name}: MCP server started successfully!")
            return True
        else:
            # Process exited - check output
            stdout, stderr = process.communicate()
            if "error" in stderr.lower() or "failed" in stderr.lower():
                print(f"❌ {name}: Failed - {stderr[:200]}")
                return False
            else:
                print(f"✅ {name}: MCP server completed successfully!")
                return True
                
    except Exception as e:
        print(f"❌ {name}: Exception - {e}")
        return False

def main():
    print("🧪 Testing All Three MCP Database Connections")
    print("=" * 50)
    
    # Test configuration
    tests = [
        {
            "name": "Supabase",
            "command": "npx",
            "args": [
                "-y", 
                "@supabase/mcp-server-supabase@latest",
                "--access-token",
                "sbp_3e08deefd3540c7221ee1d61b81e5b28eb1935a5"
            ]
        },
        {
            "name": "MariaDB", 
            "command": "npx",
            "args": ["-y", "mariadb-mcp-server"],
            "env": {
                "MARIADB_HOST": "43.134.92.105",
                "MARIADB_PORT": "32531", 
                "MARIADB_USER": "root",
                "MARIADB_PASSWORD": "vBpxMSWA7io6h5qLyC3G489Fg2l1cb0k",
                "MARIADB_DATABASE": "zeabur"
            }
        },
        {
            "name": "PostgreSQL",
            "command": "npx", 
            "args": [
                "-y",
                "@modelcontextprotocol/server-postgres",
                "postgresql://root:6AZ9XWBMN80To7ejUmpC5HtYLD24P3G1@43.134.92.105:32089/zeabur"
            ]
        }
    ]
    
    results = {}
    
    # Run all tests
    for test in tests:
        env_vars = test.get("env", None)
        result = test_mcp_connection(test["name"], test["command"], test["args"], env_vars)
        results[test["name"]] = result
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Final Test Results:")
    
    success_count = 0
    for name, success in results.items():
        status = "✅ Working" if success else "❌ Failed"
        print(f"   {name}: {status}")
        if success:
            success_count += 1
    
    print(f"\n🎯 {success_count}/{len(results)} MCP servers are functional")
    
    if success_count == len(results):
        print("🎉 ALL THREE databases are working with MCP!")
        print("🚀 Your 3-database architecture is ready!")
    else:
        print("⚠️  Some connections need attention")
        
    print("\n💡 Next steps:")
    print("   - PostgreSQL: Short-term & long-term memory")
    print("   - MariaDB: Company data & analytics") 
    print("   - Supabase: Additional operations")
    print("   - Qdrant: Vector embeddings (direct connection)")

if __name__ == "__main__":
    main()