"""
Configuration Management for Agentic SQL Backend

Handles all environment-specific settings and configuration management.
"""

from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings and configuration."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )
    
    # Application
    app_name: str = "Agentic SQL Backend"
    app_version: str = "0.1.0"
    debug: bool = False
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8008
    api_prefix: str = "/api/v1"
    
    # Database URLs
    mariadb_url: str = Field(
        description="MariaDB connection URL for company data"
    )
    postgres_url: str = Field(
        description="PostgreSQL connection URL for agent memory"
    )
    
    # Vector Database
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection_name: str = "sql_knowledge"
    
    # AI Models
    anthropic_api_key: str = Field(
        description="Anthropic API key for Claude Sonnet 4"
    )
    anthropic_model: str = "claude-3-5-sonnet-20241022"
    
    # Anthropic Prompt Caching (beta feature)
    anthropic_enable_caching: bool = True
    cache_system_prompt: bool = True
    cache_schema_info: bool = True
    prompt_cache_ttl: int = 3600  # 1 hour cache TTL
    
    # DeepSeek Models
    deepseek_api_key: str = Field(
        description="DeepSeek API key from platform.deepseek.com",
        default=""
    )
    deepseek_model: str = "deepseek-reasoner"
    deepseek_base_url: str = "https://api.deepseek.com"
    
    # OpenAI Models  
    openai_api_key: str = Field(
        description="OpenAI API key",
        default=""
    )
    openai_model: str = "gpt-4.1-nano"
    openai_base_url: str = "https://api.openai.com/v1"
    
    # BGE-M3 Embeddings
    embedding_model_name: str = "BAAI/bge-m3"
    embedding_device: str = "cpu"  # or "cuda" if GPU available
    embedding_batch_size: int = 32
    
    # MCP Configuration
    mcp_host: str = "localhost"
    mcp_port: int = 8001
    
    # Safety and Security
    max_query_timeout: int = 30  # seconds
    max_rows_returned: int = 10000
    allowed_sql_keywords: List[str] = [
        "SELECT", "FROM", "WHERE", "JOIN", "GROUP BY", 
        "ORDER BY", "HAVING", "LIMIT", "UNION", "WITH"
    ]
    dangerous_sql_keywords: List[str] = [
        "DROP", "DELETE", "TRUNCATE", "UPDATE", "INSERT", 
        "ALTER", "CREATE", "GRANT", "REVOKE"
    ]
    
    # Rate Limiting
    rate_limit_requests: int = 100  # requests per window
    rate_limit_window: int = 60     # seconds
    
    # Session Management
    session_timeout: int = 3600     # 1 hour in seconds
    cache_ttl: int = 1800          # 30 minutes in seconds
    
    # Investigation Settings
    max_investigation_steps: int = 20
    investigation_timeout: int = 300  # 5 minutes
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "json"
    
    # CORS
    cors_origins: List[str] = ["http://localhost:3000", "http://localhost:5173"]
    cors_methods: List[str] = ["GET", "POST", "PUT", "DELETE"]
    cors_headers: List[str] = ["*"]
    
    # WebSocket
    websocket_ping_interval: int = 20
    websocket_ping_timeout: int = 10
    
    # Development
    reload: bool = False
    workers: int = 1


# Global settings instance
settings = Settings()


# Database connection strings
def get_mariadb_url() -> str:
    """Get MariaDB connection URL."""
    return settings.mariadb_url


def get_postgres_url() -> str:
    """Get PostgreSQL connection URL."""
    return settings.postgres_url


def get_qdrant_config() -> dict:
    """Get Qdrant configuration."""
    return {
        "url": settings.qdrant_url,
        "collection_name": settings.qdrant_collection_name
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