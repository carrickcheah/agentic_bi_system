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