"""
AI Model Package

Contains all AI model integrations for the Agentic SQL Backend.
This package provides fallback support with priority:
1. AnthropicModel (primary)
2. DeepSeekModel (fallback #2)  
3. OpenAIModel (fallback #3)
"""

from .anthropic_model import AnthropicModel
from .deepseek_model import DeepSeekModel
from .openai_model import OpenAIModel

# Easy imports for main.py:
# from app.model import AnthropicModel, DeepSeekModel, OpenAIModel

__all__ = [
    "AnthropicModel",     # Priority 1
    "DeepSeekModel",      # Priority 2
    "OpenAIModel"         # Priority 3
]