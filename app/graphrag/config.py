"""
GraphRAG Configuration Settings

Handles GraphRAG-specific configuration for MCP server integration.
"""

from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field


class GraphRAGConfig(BaseModel):
    """GraphRAG configuration settings."""
    
    # Server configuration
    server_host: str = Field(
        default="localhost",
        description="GraphRAG server host"
    )
    server_port: int = Field(
        default=8001,
        description="GraphRAG server port"
    )
    
    # Data paths
    data_path: Path = Field(
        default=Path("./graphrag_data"),
        description="Path to GraphRAG data directory"
    )
    entities_file: str = Field(
        default="entities.parquet",
        description="Entities parquet file name"
    )
    relationships_file: str = Field(
        default="relationships.parquet", 
        description="Relationships parquet file name"
    )
    communities_file: str = Field(
        default="communities.parquet",
        description="Communities parquet file name"
    )
    community_reports_file: str = Field(
        default="community_reports.parquet",
        description="Community reports parquet file name"
    )
    
    # Performance settings
    timeout_seconds: float = Field(
        default=15.0,
        description="Default timeout for GraphRAG operations"
    )
    max_concurrent_requests: int = Field(
        default=10,
        description="Maximum concurrent GraphRAG requests"
    )
    entity_search_timeout: float = Field(
        default=5.0,
        description="Timeout for entity search operations"
    )
    global_search_timeout: float = Field(
        default=15.0,
        description="Timeout for global search operations"
    )
    
    # Cost controls
    cost_limit_per_query: float = Field(
        default=0.05,
        description="Maximum cost per GraphRAG query in USD"
    )
    daily_budget_limit: float = Field(
        default=100.0,
        description="Daily budget limit for GraphRAG operations in USD"
    )
    
    # Cache settings
    cache_size: int = Field(
        default=10000,
        description="LRU cache size for GraphRAG results"
    )
    cache_ttl_seconds: int = Field(
        default=3600,
        description="Cache TTL in seconds"
    )
    
    # Memory settings
    max_memory_gb: float = Field(
        default=8.0,
        description="Maximum memory usage for GraphRAG server in GB"
    )
    startup_timeout_seconds: float = Field(
        default=300.0,
        description="Timeout for GraphRAG server startup"
    )
    
    # LLM configuration
    llm_model: str = Field(
        default="gpt-4o-mini",
        description="LLM model for GraphRAG operations"
    )
    llm_temperature: float = Field(
        default=0.1,
        description="LLM temperature for GraphRAG operations"
    )
    max_tokens: int = Field(
        default=2000,
        description="Maximum tokens for LLM responses"
    )
    
    # Business intelligence settings
    max_communities_for_global_search: int = Field(
        default=3,
        description="Maximum communities to analyze in global search"
    )
    entity_search_limit: int = Field(
        default=20,
        description="Maximum entities to return in entity search"
    )
    
    # Monitoring settings
    enable_detailed_logging: bool = Field(
        default=True,
        description="Enable detailed logging for GraphRAG operations"
    )
    metrics_export_interval: int = Field(
        default=60,
        description="Metrics export interval in seconds"
    )
    
    class Config:
        """Pydantic configuration."""
        env_prefix = "GRAPHRAG_"
        case_sensitive = False
        
    def get_data_file_path(self, filename: str) -> Path:
        """Get full path to a data file."""
        return self.data_path / filename
    
    def validate_data_files(self) -> bool:
        """Validate that required data files exist."""
        required_files = [
            self.entities_file,
            self.relationships_file, 
            self.communities_file,
            self.community_reports_file
        ]
        
        for filename in required_files:
            file_path = self.get_data_file_path(filename)
            if not file_path.exists():
                return False
        
        return True