"""
MCP Client Manager

Manages MCP client connections for all databases and provides unified interface.
"""

import json
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager

from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client

from ..config import settings
from ..utils.logging import logger
from .mariadb_client import MariaDBClient
from .postgres_client import PostgreSQLClient
from .supabase_client import SupabaseClient
from .qdrant_client import QdrantClient


class MCPClientManager:
    """Manages all MCP database client connections."""
    
    def __init__(self):
        self.clients: Dict[str, Any] = {}
        self.sessions: Dict[str, ClientSession] = {}
        self._initialized = False
        self.config_path = Path(settings.mcp_config_path)
        
        # Database client wrappers
        self.mariadb: Optional[MariaDBClient] = None
        self.postgres: Optional[PostgreSQLClient] = None
        self.supabase: Optional[SupabaseClient] = None
        self.qdrant: Optional[QdrantClient] = None
    
    async def initialize(self):
        """Initialize all MCP client connections."""
        if self._initialized:
            return
        
        try:
            # Load MCP configuration
            mcp_config = self._load_mcp_config()
            
            # Initialize each MCP client
            for server_name, config in mcp_config.get("mcpServers", {}).items():
                await self._init_mcp_client(server_name, config)
            
            # Initialize database client wrappers
            await self._init_database_clients()
            
            self._initialized = True
            logger.info("MCP client manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize MCP clients: {e}")
            raise
    
    def _load_mcp_config(self) -> Dict[str, Any]:
        """Load MCP configuration from mcp.json."""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            
            # Substitute environment variables
            config_str = json.dumps(config)
            for key, value in settings.__dict__.items():
                if isinstance(value, str):
                    config_str = config_str.replace(f"${{{key.upper()}}}", value)
            
            return json.loads(config_str)
            
        except FileNotFoundError:
            logger.error(f"MCP config file not found: {self.config_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in MCP config: {e}")
            raise
    
    async def _init_mcp_client(self, server_name: str, config: Dict[str, Any]):
        """Initialize a single MCP client."""
        try:
            command = config["command"]
            args = config["args"]
            env_vars = config.get("env", {})
            
            # Process environment variables
            env = {}
            for key, value in env_vars.items():
                if value.startswith("${") and value.endswith("}"):
                    env_var = value[2:-1]
                    env[key] = getattr(settings, env_var.lower(), value)
                else:
                    env[key] = value
            
            # Create stdio client
            client = await stdio_client(command, args, env=env)
            self.clients[server_name] = client
            
            # Create session
            session = ClientSession(client[0], client[1])
            self.sessions[server_name] = session
            
            logger.info(f"MCP client '{server_name}' initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize MCP client '{server_name}': {e}")
            raise
    
    async def _init_database_clients(self):
        """Initialize database-specific client wrappers."""
        if "mariadb" in self.sessions:
            self.mariadb = MariaDBClient(self.sessions["mariadb"])
        
        if "postgres" in self.sessions:
            self.postgres = PostgreSQLClient(self.sessions["postgres"])
        
        if "supabase" in self.sessions:
            self.supabase = SupabaseClient(self.sessions["supabase"])
        
        if "qdrant" in self.sessions:
            self.qdrant = QdrantClient(self.sessions["qdrant"])
    
    @asynccontextmanager
    async def get_client_session(self, client_name: str):
        """Get a client session for direct MCP operations."""
        if client_name not in self.sessions:
            raise RuntimeError(f"MCP client '{client_name}' not initialized")
        
        session = self.sessions[client_name]
        try:
            yield session
        except Exception as e:
            logger.error(f"Error in MCP client session '{client_name}': {e}")
            raise
    
    def is_healthy(self) -> bool:
        """Check if MCP clients are healthy."""
        return self._initialized and len(self.sessions) > 0
    
    async def close(self):
        """Close all MCP client connections."""
        try:
            for client_name, client in self.clients.items():
                if hasattr(client, 'close'):
                    await client.close()
                logger.info(f"MCP client '{client_name}' closed")
            
            self.sessions.clear()
            self.clients.clear()
            self._initialized = False
            
        except Exception as e:
            logger.error(f"Error closing MCP clients: {e}")
    
    # Investigation management methods (to be implemented)
    async def save_investigation(self, investigation):
        """Save investigation using PostgreSQL MCP client."""
        if not self.postgres:
            raise RuntimeError("PostgreSQL MCP client not available")
        return await self.postgres.save_investigation(investigation)
    
    async def get_investigation(self, investigation_id: str):
        """Get investigation by ID using PostgreSQL MCP client."""
        if not self.postgres:
            raise RuntimeError("PostgreSQL MCP client not available")
        return await self.postgres.get_investigation(investigation_id)
    
    async def list_investigations(self, user_id: str = None, limit: int = 20, offset: int = 0):
        """List investigations using PostgreSQL MCP client."""
        if not self.postgres:
            raise RuntimeError("PostgreSQL MCP client not available")
        return await self.postgres.list_investigations(user_id, limit, offset)
    
    async def cancel_investigation(self, investigation_id: str):
        """Cancel investigation using PostgreSQL MCP client."""
        if not self.postgres:
            raise RuntimeError("PostgreSQL MCP client not available")
        return await self.postgres.cancel_investigation(investigation_id)
    
    async def update_investigation_error(self, investigation_id: str, error: str):
        """Update investigation with error using PostgreSQL MCP client."""
        if not self.postgres:
            raise RuntimeError("PostgreSQL MCP client not available")
        return await self.postgres.update_investigation_error(investigation_id, error)