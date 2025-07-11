"""
Configuration Management for Agentic SQL Backend

Domain-separated configuration with simple composition for backwards compatibility.
"""

from .cfg_databases import DatabaseSettings
from .cfg_infrastructure import InfrastructureSettings

# Create individual domain instances
databases = DatabaseSettings()
infrastructure = InfrastructureSettings()


class Settings:
    """
    Composed settings with flat access for backwards compatibility.
    
    Domain ownership:
    - databases: Data team
    - infrastructure: DevOps team
    """
    
    # Database settings (Data team)
    mariadb_host = databases.mariadb_host
    mariadb_port = databases.mariadb_port
    mariadb_user = databases.mariadb_user
    mariadb_password = databases.mariadb_password
    mariadb_database = databases.mariadb_database
    postgres_url = databases.postgres_url
    
    # Qdrant vector database (direct HTTP API)
    use_qdrant = databases.use_qdrant
    qdrant_url = databases.qdrant_url
    qdrant_api_key = databases.qdrant_api_key
    collection_name = databases.collection_name
    embedding_model = databases.embedding_model
    
    # Infrastructure settings (DevOps team)
    app_name = infrastructure.app_name
    app_version = infrastructure.app_version
    debug = infrastructure.debug
    api_host = infrastructure.api_host
    api_port = infrastructure.api_port
    api_prefix = infrastructure.api_prefix
    cors_origins = infrastructure.cors_origins
    cors_methods = infrastructure.cors_methods
    cors_headers = infrastructure.cors_headers
    websocket_ping_interval = infrastructure.websocket_ping_interval
    websocket_ping_timeout = infrastructure.websocket_ping_timeout
    reload = infrastructure.reload
    workers = infrastructure.workers
    log_level = infrastructure.log_level
    log_format = infrastructure.log_format
    mcp_config_path = infrastructure.mcp_config_path


# Global settings instance
settings = Settings()


# Backwards Compatibility Helper Functions
def get_mariadb_mcp_config() -> dict:
    """Get MariaDB MCP configuration."""
    return {
        "host": settings.mariadb_host,
        "port": settings.mariadb_port,
        "user": settings.mariadb_user,
        "password": settings.mariadb_password,
        "database": settings.mariadb_database
    }


def get_postgres_mcp_config() -> dict:
    """Get PostgreSQL MCP configuration."""
    return {
        "url": settings.postgres_url
    }


def get_supabase_mcp_config() -> dict:
    """Get Supabase MCP configuration."""
    return {
        "access_token": settings.supabase_access_token
    }


def is_development() -> bool:
    """Check if running in development mode."""
    return settings.debug


def get_cors_config() -> dict:
    """Get CORS configuration."""
    return {
        "allow_origins": settings.cors_origins,
        "allow_credentials": True,
        "allow_methods": settings.cors_methods,
        "allow_headers": settings.cors_headers,
    }