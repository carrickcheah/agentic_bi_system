#!/usr/bin/env python3
"""
Simple test for Qdrant MCP server functionality
"""

import subprocess
import time
import os
from pathlib import Path

# Load environment variables
from dotenv import load_dotenv
env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(env_path)

def test_qdrant_mcp():
    """Test Qdrant MCP server directly"""
    print("🧪 Testing Qdrant MCP Server")
    print("=" * 50)
    
    # Get environment variables
    qdrant_url = os.getenv('QDRANT_URL')
    qdrant_api_key = os.getenv('QDRANT_API_KEY')
    
    if not qdrant_url or not qdrant_api_key:
        print("❌ Missing QDRANT_URL or QDRANT_API_KEY in .env file")
        return False
        
    print(f"✅ Qdrant URL: {qdrant_url}")
    print(f"✅ API Key: {'*' * 10}...{qdrant_api_key[-4:]}")
    
    # Test MCP server startup
    print("\n🔄 Starting Qdrant MCP server...")
    
    env = os.environ.copy()
    env.update({
        'QDRANT_URL': qdrant_url,
        'QDRANT_API_KEY': qdrant_api_key,
        'COLLECTION_NAME': 'valiant_vector',
        'EMBEDDING_MODEL': 'sentence-transformers/all-MiniLM-L6-v2'
    })
    
    try:
        # Start the MCP server
        process = subprocess.Popen(
            ['uvx', 'mcp-server-qdrant'],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait for startup
        time.sleep(3)
        
        # Check if running
        if process.poll() is None:
            print("✅ Qdrant MCP server started successfully!")
            
            # Send test initialization
            print("\n🔄 Testing MCP protocol initialization...")
            process.stdin = subprocess.PIPE
            process.stdin.write('{"jsonrpc":"2.0","method":"initialize","params":{"capabilities":{}},"id":1}\n')
            process.stdin.flush()
            
            # Wait for response
            time.sleep(1)
            
            # Terminate
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                
            return True
        else:
            stdout, stderr = process.communicate()
            print(f"❌ Server exited with error:")
            if stderr:
                print(f"   {stderr[:200]}")
            return False
            
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False
    
    
def test_direct_qdrant_connection():
    """Test direct connection to Qdrant cloud"""
    print("\n\n🔍 Testing Direct Qdrant Connection")
    print("=" * 50)
    
    try:
        import httpx
        
        qdrant_url = os.getenv('QDRANT_URL')
        qdrant_api_key = os.getenv('QDRANT_API_KEY')
        
        # Test collections endpoint
        print(f"\n🔄 Connecting to Qdrant at {qdrant_url}")
        
        headers = {'api-key': qdrant_api_key}
        response = httpx.get(f"{qdrant_url}/collections", headers=headers)
        
        if response.status_code == 200:
            print("✅ Successfully connected to Qdrant!")
            collections = response.json()
            print(f"📊 Collections: {collections}")
            return True
        else:
            print(f"❌ Failed to connect: {response.status_code}")
            print(f"   Response: {response.text[:200]}")
            return False
            
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False


def main():
    print("🚀 Qdrant MCP Server Installation Test")
    print("=" * 70)
    
    # Test 1: MCP server
    mcp_success = test_qdrant_mcp()
    
    # Test 2: Direct connection
    direct_success = test_direct_qdrant_connection()
    
    # Summary
    print("\n\n📊 Test Summary")
    print("=" * 50)
    print(f"MCP Server: {'✅ Working' if mcp_success else '❌ Failed'}")
    print(f"Direct Connection: {'✅ Working' if direct_success else '❌ Failed'}")
    
    if mcp_success and direct_success:
        print("\n🎉 Qdrant is fully installed and operational!")
        print("🔹 MCP server can be started with: uvx mcp-server-qdrant")
        print("🔹 Your app can now use Qdrant for vector operations")
    else:
        print("\n⚠️  Some issues need attention")


if __name__ == "__main__":
    main()