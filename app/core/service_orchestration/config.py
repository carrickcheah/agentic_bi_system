"""
Self-contained configuration using pydantic-settings.
Loads ALL values from local settings.env - NO hardcoded business defaults.
Service Orchestration configuration for Phase 3 implementation.
"""

from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ServiceOrchestrationSettings(BaseSettings):
    """Service orchestration configuration - loads from local settings.env"""
    
    model_config = SettingsConfigDict(
        env_file=Path(__file__).parent / "settings.env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # MCP Service Connection Settings - NO defaults (force explicit configuration)
    mcp_config_path: str = Field(description="Path to MCP configuration file")
    mariadb_service_name: str = Field(description="MariaDB MCP service name")
    postgres_service_name: str = Field(description="PostgreSQL MCP service name")
    lancedb_service_name: str = Field(description="LanceDB MCP service name")
    graphrag_service_name: str = Field(description="GraphRAG MCP service name")
    
    # Service Selection Thresholds
    analytical_complexity_threshold: float = Field(
        default=0.3, description="Threshold for PostgreSQL activation"
    )
    computational_complexity_threshold: float = Field(
        default=0.6, description="Threshold for LanceDB activation"
    )
    investigative_complexity_threshold: float = Field(
        default=0.8, description="Threshold for GraphRAG activation"
    )
    
    # Performance Optimization Settings
    connection_pool_size: int = Field(
        default=10, description="Maximum connection pool size per service"
    )
    connection_timeout_seconds: int = Field(
        default=30, description="Connection timeout for service initialization"
    )
    cache_warmup_enabled: bool = Field(
        default=True, description="Enable cache warmup during service preparation"
    )
    query_preparation_enabled: bool = Field(
        default=True, description="Enable query context preparation optimization"
    )
    
    # Health Monitoring Configuration
    health_check_interval_seconds: int = Field(
        default=30, description="Interval between health check cycles"
    )
    health_check_timeout_seconds: int = Field(
        default=10, description="Timeout for individual health checks"
    )
    max_failure_count: int = Field(
        default=3, description="Maximum failures before service marked unhealthy"
    )
    circuit_breaker_enabled: bool = Field(
        default=True, description="Enable circuit breaker pattern for service resilience"
    )
    
    # Resource Allocation Settings
    simple_query_max_duration_minutes: int = Field(
        default=5, description="Maximum duration allocation for simple queries"
    )
    analytical_query_max_duration_minutes: int = Field(
        default=15, description="Maximum duration allocation for analytical queries"
    )
    computational_query_max_duration_minutes: int = Field(
        default=45, description="Maximum duration allocation for computational queries"
    )
    investigative_query_max_duration_minutes: int = Field(
        default=120, description="Maximum duration allocation for investigative queries"
    )
    
    # Service Mesh Configuration
    enable_parallel_service_activation: bool = Field(
        default=True, description="Enable parallel activation of multiple services"
    )
    service_coordination_timeout_seconds: int = Field(
        default=60, description="Timeout for multi-service coordination"
    )
    enable_service_fallback: bool = Field(
        default=True, description="Enable fallback to simpler service combinations"
    )
    
    # Logging Configuration
    log_level: str = Field(default="INFO", description="Logging level")
    enable_performance_logging: bool = Field(
        default=True, description="Enable detailed performance logging"
    )
    enable_service_metrics: bool = Field(
        default=True, description="Enable service metrics collection"
    )


# Create singleton instance
settings = ServiceOrchestrationSettings()