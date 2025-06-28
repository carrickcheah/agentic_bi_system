"""
Test script for model-specific configuration.
"""

import sys
from pathlib import Path

# Add the app directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Now import the model config directly
from model.config import settings

def test_config():
    """Test that the model configuration loads correctly."""
    print("🔧 Testing Model Configuration")
    print(f"✅ Config loaded successfully!")
    print(f"📝 Anthropic API Key: {settings.anthropic_api_key[:10]}..." if settings.anthropic_api_key else "❌ No Anthropic API Key")
    print(f"🤖 Anthropic Model: {settings.anthropic_model}")
    print(f"📝 DeepSeek API Key: {settings.deepseek_api_key[:10]}..." if settings.deepseek_api_key else "❌ No DeepSeek API Key")
    print(f"🤖 DeepSeek Model: {settings.deepseek_model}")
    print(f"📝 OpenAI API Key: {settings.openai_api_key[:10]}..." if settings.openai_api_key else "❌ No OpenAI API Key")
    print(f"🤖 OpenAI Model: {settings.openai_model}")
    
    return True

if __name__ == "__main__":
    test_config()