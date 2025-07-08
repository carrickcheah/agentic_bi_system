"""
AI Model Package

This package provides a production-grade AI model management system with intelligent 
fallback capabilities for the Agentic SQL Backend. It implements fault-tolerant 
architecture with automatic failover between multiple AI providers.

Key Components:
- ModelManager (runner.py): Main orchestrator with fallback logic
- AnthropicModel: Primary AI provider with caching support
- DeepSeekModel: Fast fallback with reasoning capabilities  
- OpenAIModel: Secondary fallback option
- ModelSettings: Type-safe configuration management

Usage:
    from app.model import ModelManager
    
    # Initialize with automatic model discovery
    manager = ModelManager()
    
    # Generate response with intelligent routing
    response = await manager.generate_response("Analyze sales data")
    
    # Monitor system health
    health = await manager.health_check()

Architecture:
    Priority-based fallback system:
    1. Anthropic Claude (primary) - Advanced reasoning with caching
    2. DeepSeek (fallback #1) - Fast, cost-effective reasoning
    3. OpenAI GPT (fallback #2) - Reliable general-purpose AI

Configuration:
    - Self-contained module configuration via settings.env
    - Type-safe pydantic settings with validation
    - Automatic API key discovery and model initialization
    - Production-ready error handling and logging
"""

from .runner import ModelManager
from .anthropic_model import AnthropicModel
from .deepseek_model import DeepSeekModel
from .openai_model import OpenAIModel
from .openai_embedding import OpenAIEmbedding
from .config import ModelSettings, settings

__all__ = [
    "ModelManager",          # Main orchestrator - recommended entry point
    "AnthropicModel",        # Primary AI provider
    "DeepSeekModel",         # Fast fallback provider
    "OpenAIModel",           # Secondary fallback provider
    "OpenAIEmbedding",       # OpenAI embedding model
    "ModelSettings",         # Configuration class
    "settings",              # Configured settings instance
]

__version__ = "1.0.0"
__author__ = "Agentic SQL Team"
__description__ = "Production-grade AI model management with intelligent fallback"