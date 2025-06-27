"""
Intelligence Configuration Settings

Handles AI model configurations and investigation settings.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class IntelligenceSettings(BaseSettings):
    """AI model and investigation configuration settings."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Anthropic Claude configuration
    anthropic_api_key: str = Field(description="Anthropic API key for Claude Sonnet 4")
    anthropic_model: str = "claude-sonnet-4-20250514"
    anthropic_enable_caching: bool = True
    cache_system_prompt: bool = True
    cache_schema_info: bool = True
    prompt_cache_ttl: int = 3600  # 1 hour cache TTL
    
    # DeepSeek model configuration
    deepseek_api_key: str = Field(
        description="DeepSeek API key from platform.deepseek.com",
        default=""
    )
    deepseek_model: str = "deepseek-reasoner"
    deepseek_base_url: str = "https://api.deepseek.com"
    
    # OpenAI model configuration
    openai_api_key: str = Field(
        description="OpenAI API key",
        default=""
    )
    openai_model: str = "gpt-4.1-nano"
    openai_base_url: str = "https://api.openai.com/v1"
    
    # Embedding model configuration
    embedding_model_name: str = "BAAI/bge-m3"
    embedding_device: str = "cpu"  # or "cuda" if GPU available
    embedding_batch_size: int = 32
    mcp_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Investigation configuration
    max_investigation_steps: int = 20
    investigation_timeout: int = 300  # 5 minutes
    
    # GraphRAG configuration
    graphrag_data_path: str = Field(
        default="./graphrag_data",
        description="Path to GraphRAG data directory with parquet files"
    )
    graphrag_server_host: str = Field(
        default="localhost",
        description="GraphRAG server host"
    )
    graphrag_server_port: int = Field(
        default=8001,
        description="GraphRAG server port"
    )
    graphrag_timeout: float = Field(
        default=15.0,
        description="Default timeout for GraphRAG operations in seconds"
    )
    graphrag_max_concurrent: int = Field(
        default=10,
        description="Maximum concurrent GraphRAG requests"
    )
    graphrag_cost_limit_per_query: float = Field(
        default=0.05,
        description="Maximum cost per GraphRAG query in USD"
    )
    graphrag_daily_budget_limit: float = Field(
        default=100.0,
        description="Daily budget limit for GraphRAG operations in USD"
    )
    graphrag_cache_size: int = Field(
        default=10000,
        description="LRU cache size for GraphRAG results"
    )
    graphrag_enable_detailed_logging: bool = Field(
        default=True,
        description="Enable detailed logging for GraphRAG operations"
    )
    graphrag_entity_search_timeout: float = Field(
        default=5.0,
        description="Timeout for GraphRAG entity search operations"
    )
    graphrag_global_search_timeout: float = Field(
        default=15.0,
        description="Timeout for GraphRAG global search operations"
    )
    graphrag_max_communities: int = Field(
        default=3,
        description="Maximum communities to analyze in global search"
    )
    graphrag_entity_search_limit: int = Field(
        default=20,
        description="Maximum entities to return in entity search"
    )