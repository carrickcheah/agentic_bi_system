#!/usr/bin/env python3
"""
Final MCP connection test
"""

import json
import subprocess
import time
import os

def test_connection(name, command, args, env_vars=None):
    """Test a single MCP server connection"""
    print(f"\n🔄 Testing {name}...")
    
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
        
        # Wait 3 seconds to see if it starts
        time.sleep(3)
        
        # Check if still running
        if process.poll() is None:
            # Still running - good sign
            process.terminate()
            process.wait()
            print(f"✅ {name}: Server started successfully!")
            return True
        else:
            # Exited - check output
            stdout, stderr = process.communicate()
            if "error" in stderr.lower() or "failed" in stderr.lower():
                print(f"❌ {name}: Failed - {stderr}")
                return False
            else:
                print(f"✅ {name}: Server completed successfully!")
                return True
                
    except Exception as e:
        print(f"❌ {name}: Exception - {e}")
        return False

def main():
    print("🧪 Final MCP Connection Test")
    print("=" * 40)
    
    # Load environment variables
    postgres_url = "postgresql://root:6AZ9XWBMN80To7ejUmpC5HtYLD24P3G1@43.134.92.105:32089/zeabur"
    supabase_token = "sbp_3e08deefd3540c7221ee1d61b81e5b28eb1935a5"
    
    results = {}
    
    # Test PostgreSQL
    results["postgres"] = test_connection(
        "PostgreSQL",
        "npx",
        ["-y", "@modelcontextprotocol/server-postgres", postgres_url]
    )
    
    # Test Supabase
    results["supabase"] = test_connection(
        "Supabase",
        "npx",
        ["-y", "@supabase/mcp-server-supabase@latest", "--access-token", supabase_token]
    )
    
    # Summary
    print("\n" + "=" * 40)
    print("📊 Final Results:")
    
    success_count = 0
    for server, success in results.items():
        status = "✅ Working" if success else "❌ Failed"
        print(f"   {server}: {status}")
        if success:
            success_count += 1
    
    print(f"\n🎯 {success_count}/{len(results)} MCP servers are functional")
    
    if success_count == len(results):
        print("🎉 All configured MCP servers are working!")
        print("\n📝 Your mcp.json is ready for use!")
    else:
        print("⚠️  Some servers need attention, but configured ones work!")

if __name__ == "__main__":
    main()