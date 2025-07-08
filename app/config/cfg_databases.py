"""
Database Configuration Settings

Handles all MCP database configurations for MariaDB, PostgreSQL.
"""

from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    """Database configuration settings."""
    
    model_config = SettingsConfigDict(
        env_file=Path(__file__).parent.parent / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # MariaDB MCP configuration
    mariadb_host: str = Field(description="MariaDB host for MCP connection")
    mariadb_port: int = Field(description="MariaDB port for MCP connection")
    mariadb_user: str = Field(description="MariaDB user for MCP connection")
    mariadb_password: str = Field(description="MariaDB password for MCP connection")
    mariadb_database: str = Field(description="MariaDB database for MCP connection")
    
    # PostgreSQL MCP configuration
    postgres_url: str = Field(description="PostgreSQL connection URL for MCP")
    

    # General vector/embedding configuration
    collection_name: str = Field(default="valiant_vector", description="Collection name for patterns")
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", description="Embedding model for vectors")
    
    # Qdrant configuration (using pydantic settings, NOT os.getenv!)
    use_qdrant: bool = Field(default=True, description="Enable Qdrant vector service")
    qdrant_url: str = Field(
        default="https://1f5d419c-2100-483e-a8c7-e1f2cd0ad2a7.us-east4-0.gcp.cloud.qdrant.io:6333",
        description="Qdrant cloud instance URL"
    )
    qdrant_api_key: str = Field(default="", description="Qdrant API key")