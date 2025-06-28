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
        default="claude-sonnet-4-20250514",
        description="Anthropic model to use"
    )
    
    # DeepSeek Configuration
    deepseek_api_key: str = Field(
        default="",
        description="DeepSeek API key"
    )
    deepseek_model: str = Field(
        default="deepseek-reasoner",
        description="DeepSeek model (deepseek-chat for speed, deepseek-reasoner for depth)"
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
        default="gpt-4.1-nano",
        description="OpenAI model (gpt-4o-mini, gpt-4.1-mini, gpt-4.1)"
    )
    openai_base_url: str = Field(
        default="https://api.openai.com/v1",
        description="OpenAI API base URL"
    )


# Create singleton instance
settings = ModelSettings()