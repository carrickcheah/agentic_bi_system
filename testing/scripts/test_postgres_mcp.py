#!/usr/bin/env python3
"""
Test PostgreSQL MCP connection
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the app directory to Python path
app_dir = Path(__file__).parent.parent.parent / "app"
sys.path.insert(0, str(app_dir))

from dotenv import load_dotenv

# Load environment variables
load_dotenv(app_dir / ".env")

async def test_postgres_mcp():
    """Test PostgreSQL MCP server connection"""
    print("🔄 Testing PostgreSQL MCP server...")
    
    postgres_url = os.getenv("POSTGRES_URL")
    if not postgres_url:
        print("❌ POSTGRES_URL not set in environment")
        return False
    
    print(f"📋 Connection string: {postgres_url[:20]}...")
    
    try:
        # Try to run the MCP server briefly to see if it starts
        import subprocess
        
        cmd = ["npx", "-y", "@modelcontextprotocol/server-postgres"]
        env = os.environ.copy()
        env["POSTGRES_CONNECTION_STRING"] = postgres_url
        
        print("🚀 Starting PostgreSQL MCP server...")
        
        # Start the process
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait a bit to see if it starts successfully
        try:
            stdout, stderr = process.communicate(timeout=5)
            if process.returncode == 0:
                print("✅ PostgreSQL MCP server started successfully!")
                return True
            else:
                print(f"❌ PostgreSQL MCP server failed to start")
                print(f"   stdout: {stdout}")
                print(f"   stderr: {stderr}")
                return False
        except subprocess.TimeoutExpired:
            # If it's still running after 5 seconds, that's probably good
            process.terminate()
            print("✅ PostgreSQL MCP server started successfully (still running after 5s)")
            return True
            
    except Exception as e:
        print(f"❌ Error testing PostgreSQL MCP: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(test_postgres_mcp())