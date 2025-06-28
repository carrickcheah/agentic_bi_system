#!/usr/bin/env python3
"""
Check MariaDB Tables

Simple script to check how many tables are in the MariaDB database.
"""

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


def main():
    """Check MariaDB tables via direct mysql connection."""
    print("📊 Checking MariaDB Tables")
    print("=" * 40)
    
    env_vars = load_env_vars()
    
    # Display connection info
    print(f"🔗 Connecting to:")
    print(f"   Host: {env_vars.get('MARIADB_HOST')}")
    print(f"   Port: {env_vars.get('MARIADB_PORT')}")
    print(f"   Database: {env_vars.get('MARIADB_DATABASE')}")
    print(f"   User: {env_vars.get('MARIADB_USER')}")
    print()
    
    # Try to connect using mysql command if available
    try:
        print("🔍 Attempting to connect with mysql client...")
        
        # Build mysql command
        cmd = [
            "mysql",
            f"--host={env_vars.get('MARIADB_HOST')}",
            f"--port={env_vars.get('MARIADB_PORT')}",
            f"--user={env_vars.get('MARIADB_USER')}",
            f"--password={env_vars.get('MARIADB_PASSWORD')}",
            f"--database={env_vars.get('MARIADB_DATABASE')}",
            "--execute=SHOW TABLES;"
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            tables = [line.strip() for line in result.stdout.strip().split('\n')[1:] if line.strip()]
            print(f"✅ Found {len(tables)} tables:")
            
            if tables:
                for i, table in enumerate(tables, 1):
                    print(f"   {i:2d}. {table}")
            else:
                print("   (No tables found)")
            
            # Get database info
            print(f"\n📋 Database Information:")
            info_cmd = [
                "mysql",
                f"--host={env_vars.get('MARIADB_HOST')}",
                f"--port={env_vars.get('MARIADB_PORT')}",
                f"--user={env_vars.get('MARIADB_USER')}",
                f"--password={env_vars.get('MARIADB_PASSWORD')}",
                f"--database={env_vars.get('MARIADB_DATABASE')}",
                "--execute=SELECT DATABASE() as current_db, USER() as current_user, VERSION() as version;"
            ]
            
            info_result = subprocess.run(info_cmd, capture_output=True, text=True, timeout=10)
            if info_result.returncode == 0:
                print(f"   {info_result.stdout}")
            
        else:
            print(f"❌ MySQL command failed:")
            print(f"   stdout: {result.stdout}")
            print(f"   stderr: {result.stderr}")
            
            # Try alternative connection test
            print(f"\n🔧 Trying telnet test...")
            try:
                telnet_result = subprocess.run(
                    ["telnet", env_vars.get('MARIADB_HOST'), env_vars.get('MARIADB_PORT')],
                    input="\n",
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if "Connected" in telnet_result.stdout or telnet_result.returncode == 0:
                    print("✅ Network connection to MariaDB host is working")
                else:
                    print("❌ Cannot connect to MariaDB host")
            except:
                print("⚠️ Telnet test failed (telnet might not be available)")
    
    except FileNotFoundError:
        print("❌ MySQL client not found. Please install MySQL client:")
        print("   macOS: brew install mysql-client")
        print("   Ubuntu: apt-get install mysql-client")
        print("   Or use Docker: docker run -it --rm mysql:8 mysql ...")
        
        # Show Docker command as alternative
        print(f"\n💡 Alternative Docker command:")
        docker_cmd = f"""docker run -it --rm mysql:8 mysql \\
  --host={env_vars.get('MARIADB_HOST')} \\
  --port={env_vars.get('MARIADB_PORT')} \\
  --user={env_vars.get('MARIADB_USER')} \\
  --password={env_vars.get('MARIADB_PASSWORD')} \\
  --database={env_vars.get('MARIADB_DATABASE')} \\
  --execute="SHOW TABLES;" """
        print(docker_cmd)
        
    except subprocess.TimeoutExpired:
        print("❌ Connection timed out - check network connectivity")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()