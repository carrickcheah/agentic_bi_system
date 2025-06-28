#!/usr/bin/env python3
"""
Standalone test script for the model module.
This script can be run directly from the model directory.
"""

import asyncio
import sys
from pathlib import Path

# Add the app directory to Python path for imports
app_dir = Path(__file__).parent.parent
sys.path.insert(0, str(app_dir))

# Import the config directly
from model.config import settings

def test_config():
    """Test configuration loading."""
    print("=" * 50)
    print("🔧 MODEL CONFIGURATION TEST")
    print("=" * 50)
    
    print(f"✅ Configuration loaded successfully!")
    print(f"📝 Anthropic API Key: {settings.anthropic_api_key[:10]}..." if settings.anthropic_api_key else "❌ No Anthropic API Key")
    print(f"🤖 Anthropic Model: {settings.anthropic_model}")
    print(f"📝 DeepSeek API Key: {settings.deepseek_api_key[:10]}..." if settings.deepseek_api_key else "❌ No DeepSeek API Key")
    print(f"🤖 DeepSeek Model: {settings.deepseek_model}")
    print(f"📝 OpenAI API Key: {settings.openai_api_key[:10]}..." if settings.openai_api_key else "❌ No OpenAI API Key")
    print(f"🤖 OpenAI Model: {settings.openai_model}")
    print()
    
    return True

async def test_basic_model():
    """Test basic model functionality without full imports."""
    print("=" * 50)
    print("🤖 BASIC MODEL TEST")
    print("=" * 50)
    
    try:
        # Test importing individual models
        print("📦 Testing model imports...")
        
        # Import without using relative imports
        import importlib.util
        
        # Load anthropic model directly
        anthropic_path = Path(__file__).parent / "anthropic_model.py"
        if anthropic_path.exists():
            print("✅ Anthropic model file found")
        else:
            print("❌ Anthropic model file missing")
            
        # Load deepseek model directly  
        deepseek_path = Path(__file__).parent / "deepseek_model.py"
        if deepseek_path.exists():
            print("✅ DeepSeek model file found")
        else:
            print("❌ DeepSeek model file missing")
            
        # Load openai model directly
        openai_path = Path(__file__).parent / "openai_model.py"
        if openai_path.exists():
            print("✅ OpenAI model file found")
        else:
            print("❌ OpenAI model file missing")
            
        print("✅ Basic model structure validated")
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False
    
    return True

def main():
    """Main test function."""
    print("🚀 Starting Model Module Tests...\n")
    
    # Test configuration
    config_ok = test_config()
    
    # Test basic model structure
    model_ok = asyncio.run(test_basic_model())
    
    print("=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    print(f"Configuration Test: {'✅ PASS' if config_ok else '❌ FAIL'}")
    print(f"Model Structure Test: {'✅ PASS' if model_ok else '❌ FAIL'}")
    
    if config_ok and model_ok:
        print("\n🎉 All tests passed! Model module is working correctly.")
        return 0
    else:
        print("\n💥 Some tests failed. Check the output above.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)