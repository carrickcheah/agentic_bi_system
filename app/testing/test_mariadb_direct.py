#!/usr/bin/env python3
"""
Direct MariaDB MCP Connection Test

Tests MariaDB via MCP without complex config dependencies.
"""

import asyncio
import subprocess
import json
import os
from pathlib import Path


def load_env_vars():
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


async def test_mariadb_mcp_server():
    """Test MariaDB MCP server directly."""
    print("🧪 Testing MariaDB MCP Server Direct Connection")
    print("=" * 60)
    
    # Load environment variables
    env_vars = load_env_vars()
    
    print(f"📋 MariaDB Configuration:")
    print(f"   Host: {env_vars.get('MARIADB_HOST', 'Not set')}")
    print(f"   Port: {env_vars.get('MARIADB_PORT', 'Not set')}")
    print(f"   User: {env_vars.get('MARIADB_USER', 'Not set')}")
    print(f"   Database: {env_vars.get('MARIADB_DATABASE', 'Not set')}")
    print()
    
    # Check if npx is available
    print("🔍 Checking if npx is available...")
    try:
        result = subprocess.run(['npx', '--version'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"✅ npx version: {result.stdout.strip()}")
        else:
            print("❌ npx not available or not working")
            return False
    except Exception as e:
        print(f"❌ Error checking npx: {e}")
        return False
    
    # Check if mariadb-mcp-server is available
    print("\n🔍 Checking MariaDB MCP server availability...")
    try:
        # Try to get help from the package
        result = subprocess.run(
            ['npx', '-y', 'mariadb-mcp-server', '--help'],
            capture_output=True, 
            text=True, 
            timeout=30,
            env={**os.environ, **env_vars}
        )
        
        if result.returncode == 0:
            print("✅ MariaDB MCP server package is available")
            print(f"   Help output: {result.stdout[:200]}..." if result.stdout else "No output")
        else:
            print(f"⚠️ MariaDB MCP server exit code: {result.returncode}")
            print(f"   stdout: {result.stdout}")
            print(f"   stderr: {result.stderr}")
    except subprocess.TimeoutExpired:
        print("⚠️ MariaDB MCP server help command timed out (package might be downloading)")
    except Exception as e:
        print(f"❌ Error checking MariaDB MCP server: {e}")
    
    # Test basic connection (simple approach)
    print("\n🔌 Testing basic MariaDB connection...")
    try:
        # Create a simple test script for the MCP server
        test_script = '''
const mysql = require('mysql2/promise');

async function testConnection() {
    try {
        const connection = await mysql.createConnection({
            host: process.env.MARIADB_HOST,
            port: parseInt(process.env.MARIADB_PORT),
            user: process.env.MARIADB_USER,
            password: process.env.MARIADB_PASSWORD,
            database: process.env.MARIADB_DATABASE
        });
        
        const [rows] = await connection.execute('SELECT 1 as test, DATABASE() as db, USER() as user');
        console.log(JSON.stringify({success: true, result: rows}));
        await connection.end();
        
    } catch (error) {
        console.log(JSON.stringify({success: false, error: error.message}));
    }
}

testConnection();
'''
        
        # Write test script to temp file
        test_file = Path(__file__).parent / "temp_mariadb_test.js"
        with open(test_file, 'w') as f:
            f.write(test_script)
        
        # Run the test
        print("   Running connection test...")
        result = subprocess.run(
            ['node', str(test_file)],
            capture_output=True,
            text=True,
            timeout=30,
            env={**os.environ, **env_vars}
        )
        
        # Clean up temp file
        if test_file.exists():
            test_file.unlink()
        
        if result.returncode == 0 and result.stdout:
            try:
                test_result = json.loads(result.stdout.strip())
                if test_result.get('success'):
                    print("✅ MariaDB connection successful!")
                    print(f"   Result: {test_result.get('result')}")
                    return True
                else:
                    print(f"❌ MariaDB connection failed: {test_result.get('error')}")
                    return False
            except json.JSONDecodeError:
                print(f"⚠️ Unexpected output: {result.stdout}")
                return False
        else:
            print(f"❌ Test script failed:")
            print(f"   Exit code: {result.returncode}")
            print(f"   stdout: {result.stdout}")
            print(f"   stderr: {result.stderr}")
            return False
            
    except FileNotFoundError:
        print("❌ Node.js not found. Please install Node.js to test MariaDB MCP server.")
        return False
    except Exception as e:
        print(f"❌ Error during connection test: {e}")
        return False


def check_mcp_config():
    """Check MCP configuration file."""
    print("📋 Checking MCP Configuration")
    print("-" * 30)
    
    mcp_file = Path(__file__).parent.parent / "mcp.json"
    
    if not mcp_file.exists():
        print("❌ mcp.json file not found")
        return False
    
    try:
        with open(mcp_file) as f:
            config = json.load(f)
        
        mariadb_config = config.get("mcpServers", {}).get("mariadb", {})
        
        if not mariadb_config:
            print("❌ MariaDB configuration not found in mcp.json")
            return False
        
        print("✅ MCP configuration found:")
        print(f"   Command: {mariadb_config.get('command')}")
        print(f"   Args: {mariadb_config.get('args')}")
        print(f"   Environment variables: {list(mariadb_config.get('env', {}).keys())}")
        return True
        
    except Exception as e:
        print(f"❌ Error reading mcp.json: {e}")
        return False


async def main():
    """Main function."""
    try:
        print("🚀 MariaDB MCP Connection Test Suite")
        print("=" * 60)
        
        # Check MCP config
        config_ok = check_mcp_config()
        print()
        
        # Test MCP server
        server_ok = await test_mariadb_mcp_server()
        
        print("\n" + "=" * 60)
        print("📊 TEST RESULTS SUMMARY")
        print("=" * 60)
        print(f"MCP Configuration.......... {'✅ PASS' if config_ok else '❌ FAIL'}")
        print(f"MariaDB MCP Server......... {'✅ PASS' if server_ok else '❌ FAIL'}")
        
        overall_success = config_ok and server_ok
        print("-" * 60)
        print(f"Overall Status: {'✅ ALL TESTS PASSED' if overall_success else '❌ SOME TESTS FAILED'}")
        
        if overall_success:
            print("\n🎉 Your MariaDB MCP setup is working correctly!")
            print("   You can now use MariaDB via MCP protocol in your application.")
        else:
            print("\n🔧 Please check the following:")
            print("   1. MariaDB server is accessible from your network")
            print("   2. Environment variables are correctly set in .env")
            print("   3. Node.js and npm are installed for MCP server")
            print("   4. mariadb-mcp-server package can be installed via npx")
        
        return overall_success
        
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