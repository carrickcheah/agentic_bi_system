"""
Configuration Management for Agentic SQL Backend

Domain-separated configuration with simple composition for backwards compatibility.
"""

from .cfg_databases import DatabaseSettings
from .cfg_intelligence import IntelligenceSettings
from .cfg_security import SecuritySettings
from .cfg_infrastructure import InfrastructureSettings

# Create individual domain instances
databases = DatabaseSettings()
intelligence = IntelligenceSettings()
security = SecuritySettings()
infrastructure = InfrastructureSettings()


class Settings:
    """
    Composed settings with flat access for backwards compatibility.
    
    Domain ownership:
    - databases: Data team
    - intelligence: ML team  
    - security: Security team
    - infrastructure: DevOps team
    """
    
    # Database settings (Data team)
    mariadb_host = databases.mariadb_host
    mariadb_port = databases.mariadb_port
    mariadb_user = databases.mariadb_user
    mariadb_password = databases.mariadb_password
    mariadb_database = databases.mariadb_database
    postgres_url = databases.postgres_url
    supabase_access_token = databases.supabase_access_token
    collection_name = databases.collection_name
    embedding_model = databases.embedding_model
    
    # Intelligence settings (ML team)
    anthropic_api_key = intelligence.anthropic_api_key
    anthropic_model = intelligence.anthropic_model
    anthropic_enable_caching = intelligence.anthropic_enable_caching
    cache_system_prompt = intelligence.cache_system_prompt
    cache_schema_info = intelligence.cache_schema_info
    prompt_cache_ttl = intelligence.prompt_cache_ttl
    deepseek_api_key = intelligence.deepseek_api_key
    deepseek_model = intelligence.deepseek_model
    deepseek_base_url = intelligence.deepseek_base_url
    openai_api_key = intelligence.openai_api_key
    openai_model = intelligence.openai_model
    openai_base_url = intelligence.openai_base_url
    embedding_model_name = intelligence.embedding_model_name
    embedding_device = intelligence.embedding_device
    embedding_batch_size = intelligence.embedding_batch_size
    mcp_embedding_model = intelligence.mcp_embedding_model
    max_investigation_steps = intelligence.max_investigation_steps
    investigation_timeout = intelligence.investigation_timeout
    
    # Security settings (Security team)
    max_query_timeout = security.max_query_timeout
    max_rows_returned = security.max_rows_returned
    allowed_sql_keywords = security.allowed_sql_keywords
    dangerous_sql_keywords = security.dangerous_sql_keywords
    rate_limit_requests = security.rate_limit_requests
    rate_limit_window = security.rate_limit_window
    session_timeout = security.session_timeout
    cache_ttl = security.cache_ttl
    
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