"""
Model-specific configuration using pydantic-settings.
Loads settings from app/model/settings.env file.
"""

import os
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelSettings(BaseSettings):
    """AI model configuration settings for the model module."""
    
    model_config = SettingsConfigDict(
        env_file=Path(__file__).parent / "settings.env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Anthropic Configuration
    anthropic_api_key: str = Field(
        description="Anthropic API key for Claude Sonnet 4"
    )
    anthropic_model: str = Field(
        description="Anthropic model to use"
    )
    anthropic_enable_caching: bool = Field(
        default=True,
        description="Enable Anthropic prompt caching for cost savings"
    )
    cache_system_prompt: bool = Field(
        default=True,
        description="Cache system prompts"
    )
    cache_schema_info: bool = Field(
        default=True,
        description="Cache schema information"
    )
    
    # DeepSeek Configuration
    deepseek_api_key: str = Field(
        default="",
        description="DeepSeek API key"
    )
    deepseek_model: str = Field(
        description="DeepSeek model (deepseek-chat for speed, deepseek-coder for coding)"
    )
    deepseek_base_url: str = Field(
        default="https://api.deepseek.com",
        description="DeepSeek API base URL"
    )
    
    # OpenAI Configuration
    openai_api_key: str = Field(
        default="",
        description="OpenAI API key"
    )
    openai_model: str = Field(
        description="OpenAI model (gpt-4o-mini, gpt-4.1-mini, gpt-4.1)"
    )
    openai_base_url: str = Field(
        default="https://api.openai.com/v1",
        description="OpenAI API base URL"
    )


# Create singleton instance
settings = ModelSettings()