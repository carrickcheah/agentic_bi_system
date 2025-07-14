"""
Model-specific configuration using pydantic-settings.
Loads settings from app/model/settings.env file.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any
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
    
    # Embedding Configuration
    embedding_model: str = Field(
        description="OpenAI embedding model (text-embedding-3-small, text-embedding-3-large, text-embedding-ada-002)"
    )
    
    # Model Initialization Configuration
    lazy_model_initialization: bool = Field(
        default=True,
        description="Enable lazy model initialization - models are initialized only when needed instead of at startup"
    )
    validate_all_models: bool = Field(
        default=False,
        description="Validate all model API keys even if primary model works"
    )


# Create singleton instance
settings = ModelSettings()

# System prompts loaded from JSON
_PROMPTS_CACHE: Dict[str, Any] = {}

def load_prompts() -> Dict[str, Any]:
    """
    Load system prompts from system.json file.
    
    Returns:
        Dictionary containing all prompts with metadata
    """
    global _PROMPTS_CACHE
    
    if not _PROMPTS_CACHE:
        prompts_file = Path(__file__).parent / "system.json"
        try:
            with open(prompts_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                _PROMPTS_CACHE = data.get("prompts", {})
        except Exception as e:
            raise RuntimeError(f"Failed to load system prompts from {prompts_file}: {e}")
    
    return _PROMPTS_CACHE

def get_prompt(prompt_name: str) -> str:
    """
    Get a system prompt by name.
    
    Args:
        prompt_name: Name of the prompt to retrieve (sql_agent, default, health_check)
        
    Returns:
        The requested prompt content string
        
    Raises:
        KeyError: If prompt_name is not found
    """
    prompts = load_prompts()
    
    if prompt_name not in prompts:
        available = list(prompts.keys())
        raise KeyError(f"Prompt '{prompt_name}' not found. Available: {available}")
    
    prompt_data = prompts[prompt_name]
    
    # Handle structured prompts (like sql_agent)
    if isinstance(prompt_data.get("content"), dict):
        # For sql_agent, we need to reconstruct the XML format for backward compatibility
        if prompt_name == "sql_agent":
            return _build_sql_agent_prompt(prompt_data["content"])
        else:
            # For other structured prompts, convert to string representation
            return str(prompt_data["content"])
    
    # Handle simple string prompts (like default, health_check)
    return prompt_data.get("content", "")

def _build_sql_agent_prompt(content: Dict[str, Any]) -> str:
    """Build the SQL agent prompt in XML format from JSON structure."""
    
    role = content.get("role", "")
    principles = content.get("core_principles", [])
    methodology = content.get("investigation_methodology", {})
    output_format = content.get("output_format", {})
    constraints = content.get("constraints", [])
    
    prompt_parts = ["<system_prompt>"]
    
    # Role section
    if role:
        prompt_parts.extend([
            "<role>",
            role,
            "</role>",
            ""
        ])
    
    # Core principles
    if principles:
        prompt_parts.append("<core_principles>")
        for principle in principles:
            name = principle.get("name", "")
            desc = principle.get("description", "")
            prompt_parts.extend([
                f'<principle name="{name}">',
                desc,
                "</principle>",
                ""
            ])
        prompt_parts.append("</core_principles>")
        prompt_parts.append("")
    
    # Investigation methodology
    if methodology and methodology.get("phases"):
        prompt_parts.append("<investigation_methodology>")
        for phase in methodology["phases"]:
            name = phase.get("name", "")
            desc = phase.get("description", "")
            prompt_parts.extend([
                f'<phase name="{name}">',
                f"<description>{desc}</description>"
            ])
            
            # Handle outputs, actions, approach, deliverables
            for key in ["outputs", "actions", "approach", "deliverables"]:
                if key in phase:
                    prompt_parts.append(f"<{key}>")
                    for item in phase[key]:
                        prompt_parts.append(f"- {item}")
                    prompt_parts.append(f"</{key}>")
            
            prompt_parts.extend([
                "</phase>",
                ""
            ])
        prompt_parts.append("</investigation_methodology>")
        prompt_parts.append("")
    
    # Output format
    if output_format:
        prompt_parts.append("<output_format>")
        
        if "structure" in output_format:
            prompt_parts.append("<structure>")
            for i, item in enumerate(output_format["structure"], 1):
                prompt_parts.append(f"{i}. **{item.split(':')[0]}**: {':'.join(item.split(':')[1:]).strip()}")
            prompt_parts.append("</structure>")
            prompt_parts.append("")
        
        if "tone" in output_format:
            prompt_parts.append("<tone>")
            for item in output_format["tone"]:
                prompt_parts.append(f"- {item}")
            prompt_parts.append("</tone>")
        
        prompt_parts.append("</output_format>")
        prompt_parts.append("")
    
    # Constraints
    if constraints:
        prompt_parts.append("<constraints>")
        for constraint in constraints:
            prompt_parts.append(f"- {constraint}")
        prompt_parts.append("</constraints>")
    
    prompt_parts.append("</system_prompt>")
    
    return "\n".join(prompt_parts)

def list_available_prompts() -> list:
    """
    List all available prompt names.
    
    Returns:
        List of available prompt names
    """
    prompts = load_prompts()
    return list(prompts.keys())