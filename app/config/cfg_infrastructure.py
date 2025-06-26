"""
Infrastructure Configuration Settings

Handles application, API, CORS, WebSocket, logging, and development configuration.
"""

from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class InfrastructureSettings(BaseSettings):
    """Infrastructure and application configuration settings."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Application metadata
    app_name: str = "Agentic SQL Backend"
    app_version: str = "0.1.0"
    debug: bool = False
    
    # API configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8008
    api_prefix: str = "/api/v1"
    
    # CORS configuration
    cors_origins: List[str] = ["http://localhost:3000", "http://localhost:5173"]
    cors_methods: List[str] = ["GET", "POST", "PUT", "DELETE"]
    cors_headers: List[str] = ["*"]
    
    # WebSocket configuration
    websocket_ping_interval: int = 20
    websocket_ping_timeout: int = 10
    
    # Development settings
    reload: bool = False
    workers: int = 1
    
    # Logging configuration
    log_level: str = "INFO"
    log_format: str = "json"
    
    # MCP server configuration
    mcp_config_path: str = "mcp.json"