"""Qdrant Configuration using pydantic settings.

Loads configuration from local settings.env file.
Production-grade settings with circuit breaker and monitoring.
"""

from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class QdrantSettings(BaseSettings):
    """Qdrant configuration - loads from local settings.env"""
    
    model_config = SettingsConfigDict(
        env_file=Path(__file__).parent / "settings.env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Qdrant connection settings (from settings.env)
    qdrant_url: str = Field(
        default="https://1f5d419c-2100-483e-a8c7-e1f2cd0ad2a7.us-east4-0.gcp.cloud.qdrant.io:6333",
        description="Qdrant cloud instance URL"
    )
    api_key: str = Field(description="Qdrant API key")  # Matches settings.env
    
    # Collection configuration
    collection_name: str = Field(
        default="query_patterns",
        description="Collection name for SQL query patterns"
    )
    embedding_dim: int = Field(
        default=1536,
        description="OpenAI text-embedding-3-small dimensions"
    )
    
    # Performance settings
    batch_size: int = Field(
        default=100,
        description="Batch size for bulk operations"
    )
    search_limit: int = Field(
        default=10,
        description="Default search result limit"
    )
    similarity_threshold: float = Field(
        default=0.7,
        description="Minimum similarity score threshold"
    )
    
    # Cache settings
    cache_ttl: int = Field(
        default=300,
        description="Cache TTL in seconds (5 minutes)"
    )
    cache_max_size: int = Field(
        default=1000,
        description="Maximum cache entries"
    )
    
    # Circuit breaker settings
    circuit_breaker_threshold: int = Field(
        default=5,
        description="Consecutive failures before opening circuit"
    )
    circuit_breaker_timeout: int = Field(
        default=60,
        description="Seconds before attempting reset"
    )
    circuit_breaker_half_open_requests: int = Field(
        default=3,
        description="Test requests in half-open state"
    )
    
    # Feature flags (using pydantic, NOT os.getenv!)
    use_qdrant: bool = Field(
        default=True,
        description="Enable Qdrant vector service"
    )
    enable_cache: bool = Field(
        default=True,
        description="Enable query result caching"
    )
    enable_monitoring: bool = Field(
        default=True,
        description="Enable performance monitoring and metrics"
    )
    enable_circuit_breaker: bool = Field(
        default=True,
        description="Enable circuit breaker pattern"
    )
    
    # Operational settings
    timeout_seconds: int = Field(
        default=30,
        description="Request timeout in seconds"
    )
    max_retries: int = Field(
        default=3,
        description="Maximum retry attempts"
    )
    retry_delay: float = Field(
        default=1.0,
        description="Base delay between retries (exponential backoff)"
    )
    
    # Monitoring settings
    metrics_window_size: int = Field(
        default=1000,
        description="Size of metrics sliding window"
    )
    log_slow_queries: bool = Field(
        default=True,
        description="Log queries slower than threshold"
    )
    slow_query_threshold_ms: int = Field(
        default=100,
        description="Slow query threshold in milliseconds"
    )
    
    # Data ingestion settings
    file_path: str = Field(
        default="/Users/carrickcheah/Project/agentic_sql/app/qdrant/patterns",
        description="Path to pattern files for ingestion"
    )


# Create singleton instance
settings = QdrantSettings()


# Validation helper
def validate_settings():
    """Validate required settings are configured."""
    if not settings.api_key:
        raise ValueError("Qdrant API key not configured in settings.env")
    if not settings.qdrant_url:
        raise ValueError("Qdrant URL not configured")
    return True