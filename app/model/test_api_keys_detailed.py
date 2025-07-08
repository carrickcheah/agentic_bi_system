#!/usr/bin/env python3
"""
Detailed API key test script with enhanced diagnostics.
Tests authentication, model availability, and provides troubleshooting guidance.
"""

import asyncio
import os
from pathlib import Path
from typing import Dict, Any, Optional
import httpx
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class TestSettings(BaseSettings):
    """Test configuration - loads from local settings.env"""
    
    model_config = SettingsConfigDict(
        env_file=Path(__file__).parent / "settings.env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Anthropic Configuration
    anthropic_api_key: str = Field(description="Anthropic API key")
    anthropic_model: str = Field(description="Anthropic model")
    
    # DeepSeek Configuration
    deepseek_api_key: str = Field(description="DeepSeek API key")
    deepseek_model: str = Field(description="DeepSeek model")
    deepseek_base_url: str = Field(description="DeepSeek base URL")
    
    # OpenAI Configuration
    openai_api_key: str = Field(description="OpenAI API key")
    openai_model: str = Field(description="OpenAI model")


class DetailedAPITester:
    """Detailed API key testing with diagnostics."""
    
    def __init__(self):
        self.settings = TestSettings()
        self.test_prompt = "Respond with exactly: 'API test successful'"
        
    def _mask_api_key(self, key: str) -> str:
        """Mask API key for display."""
        if len(key) <= 8:
            return "***"
        return f"{key[:4]}...{key[-4:]}"
    
    async def test_anthropic_detailed(self) -> Dict[str, Any]:
        """Detailed test for Anthropic API."""
        print("\n" + "="*60)
        print("ANTHROPIC API TEST")
        print("="*60)
        
        result = {
            "provider": "Anthropic",
            "api_key_format": "unknown",
            "auth_status": "unknown",
            "model_status": "unknown",
            "response": None,
            "recommendations": []
        }
        
        # Check API key format
        api_key = self.settings.anthropic_api_key
        masked_key = self._mask_api_key(api_key)
        print(f"API Key: {masked_key}")
        
        if api_key.startswith("sk-ant-api"):
            result["api_key_format"] = "valid_format"
            print("✓ API key format looks correct (sk-ant-api...)")
        else:
            result["api_key_format"] = "invalid_format"
            print("✗ API key format looks incorrect (should start with 'sk-ant-api')")
            result["recommendations"].append("Check if the API key is correctly copied from Anthropic console")
        
        # Test authentication
        print(f"\nTesting authentication with model: {self.settings.anthropic_model}")
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": api_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json"
                    },
                    json={
                        "model": self.settings.anthropic_model,
                        "max_tokens": 50,
                        "messages": [
                            {"role": "user", "content": self.test_prompt}
                        ]
                    },
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    result["auth_status"] = "success"
                    result["model_status"] = "available"
                    data = response.json()
                    content = data.get("content", [{}])[0].get("text", "")
                    result["response"] = content
                    print("✓ Authentication successful")
                    print("✓ Model is available")
                    print(f"✓ Response: {content}")
                    
                elif response.status_code == 401:
                    result["auth_status"] = "failed"
                    print("✗ Authentication failed - Invalid API key")
                    result["recommendations"].extend([
                        "Verify the API key is correct and active",
                        "Check if the key has been revoked or expired",
                        "Generate a new API key from https://console.anthropic.com/account/keys"
                    ])
                    
                elif response.status_code == 404:
                    result["auth_status"] = "success"
                    result["model_status"] = "not_found"
                    print("✓ Authentication successful")
                    print("✗ Model not found or not accessible")
                    result["recommendations"].extend([
                        f"Model '{self.settings.anthropic_model}' may not be available",
                        "Try using 'claude-3-opus-20240229' or 'claude-3-sonnet-20240229'",
                        "Check https://docs.anthropic.com/claude/docs/models for available models"
                    ])
                    
                else:
                    print(f"✗ Unexpected response: HTTP {response.status_code}")
                    print(f"Response: {response.text[:200]}...")
                    result["recommendations"].append("Check Anthropic API status at https://status.anthropic.com/")
                    
        except httpx.TimeoutException:
            print("✗ Request timed out")
            result["recommendations"].append("Check your internet connection")
        except Exception as e:
            print(f"✗ Exception: {str(e)}")
            result["recommendations"].append("Check if the API endpoint is accessible from your network")
        
        return result
    
    async def test_deepseek_detailed(self) -> Dict[str, Any]:
        """Detailed test for DeepSeek API."""
        print("\n" + "="*60)
        print("DEEPSEEK API TEST")
        print("="*60)
        
        result = {
            "provider": "DeepSeek",
            "api_key_format": "unknown",
            "auth_status": "unknown",
            "model_status": "unknown",
            "response": None,
            "recommendations": []
        }
        
        # Check API key format
        api_key = self.settings.deepseek_api_key
        masked_key = self._mask_api_key(api_key)
        print(f"API Key: {masked_key}")
        print(f"Base URL: {self.settings.deepseek_base_url}")
        
        if api_key.startswith("sk-"):
            result["api_key_format"] = "valid_format"
            print("✓ API key format looks correct (sk-...)")
        else:
            result["api_key_format"] = "invalid_format"
            print("✗ API key format looks incorrect (should start with 'sk-')")
            result["recommendations"].append("Check if the API key is correctly copied from DeepSeek platform")
        
        # Test authentication
        print(f"\nTesting authentication with model: {self.settings.deepseek_model}")
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.settings.deepseek_base_url}/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": self.settings.deepseek_model,
                        "messages": [
                            {"role": "user", "content": self.test_prompt}
                        ],
                        "max_tokens": 50
                    },
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    result["auth_status"] = "success"
                    result["model_status"] = "available"
                    data = response.json()
                    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    result["response"] = content
                    print("✓ Authentication successful")
                    print("✓ Model is available")
                    print(f"✓ Response: {content}")
                    
                elif response.status_code == 401:
                    result["auth_status"] = "failed"
                    print("✗ Authentication failed - Invalid API key")
                    result["recommendations"].extend([
                        "Verify the API key is correct and active",
                        "Check if the key has been revoked or expired",
                        "Generate a new API key from DeepSeek platform"
                    ])
                    
                elif response.status_code == 404:
                    result["auth_status"] = "success"
                    result["model_status"] = "not_found"
                    print("✓ Authentication successful")
                    print("✗ Model not found or not accessible")
                    result["recommendations"].extend([
                        f"Model '{self.settings.deepseek_model}' may not be available",
                        "Try using 'deepseek-chat' or 'deepseek-coder'",
                        "Check DeepSeek documentation for available models"
                    ])
                    
                else:
                    print(f"✗ Unexpected response: HTTP {response.status_code}")
                    print(f"Response: {response.text[:200]}...")
                    result["recommendations"].append("Check DeepSeek API documentation")
                    
        except httpx.TimeoutException:
            print("✗ Request timed out")
            result["recommendations"].append("Check your internet connection or try a different base URL")
        except Exception as e:
            print(f"✗ Exception: {str(e)}")
            result["recommendations"].append("Check if the API endpoint is accessible from your network")
        
        return result
    
    async def test_openai_detailed(self) -> Dict[str, Any]:
        """Detailed test for OpenAI API."""
        print("\n" + "="*60)
        print("OPENAI API TEST")
        print("="*60)
        
        result = {
            "provider": "OpenAI",
            "api_key_format": "unknown",
            "auth_status": "unknown",
            "model_status": "unknown",
            "response": None,
            "recommendations": []
        }
        
        # Check API key format
        api_key = self.settings.openai_api_key
        masked_key = self._mask_api_key(api_key)
        print(f"API Key: {masked_key}")
        
        if api_key.startswith("sk-proj-") or api_key.startswith("sk-"):
            result["api_key_format"] = "valid_format"
            print("✓ API key format looks correct")
        else:
            result["api_key_format"] = "invalid_format"
            print("✗ API key format looks incorrect")
            result["recommendations"].append("Check if the API key is correctly copied from OpenAI platform")
        
        # Test authentication
        print(f"\nTesting authentication with model: {self.settings.openai_model}")
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": self.settings.openai_model,
                        "messages": [
                            {"role": "user", "content": self.test_prompt}
                        ],
                        "max_tokens": 50
                    },
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    result["auth_status"] = "success"
                    result["model_status"] = "available"
                    data = response.json()
                    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    result["response"] = content
                    print("✓ Authentication successful")
                    print("✓ Model is available")
                    print(f"✓ Response: {content}")
                    
                elif response.status_code == 401:
                    result["auth_status"] = "failed"
                    print("✗ Authentication failed - Invalid API key")
                    result["recommendations"].extend([
                        "Verify the API key is correct and active",
                        "Check if the key has been revoked or expired",
                        "Generate a new API key from https://platform.openai.com/api-keys"
                    ])
                    
                elif response.status_code == 404:
                    result["auth_status"] = "success"
                    result["model_status"] = "not_found"
                    print("✓ Authentication successful")
                    print("✗ Model not found or not accessible")
                    result["recommendations"].extend([
                        f"Model '{self.settings.openai_model}' may not be available",
                        "Try using 'gpt-4o-mini' or 'gpt-3.5-turbo'",
                        "Check https://platform.openai.com/docs/models for available models"
                    ])
                    
                else:
                    print(f"✗ Unexpected response: HTTP {response.status_code}")
                    print(f"Response: {response.text[:200]}...")
                    result["recommendations"].append("Check OpenAI API status at https://status.openai.com/")
                    
        except httpx.TimeoutException:
            print("✗ Request timed out")
            result["recommendations"].append("Check your internet connection")
        except Exception as e:
            print(f"✗ Exception: {str(e)}")
            result["recommendations"].append("Check if the API endpoint is accessible from your network")
        
        return result
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all detailed tests."""
        print("\nDETAILED API KEY TESTING")
        print("Testing all configured AI providers...\n")
        
        results = []
        
        # Test each provider
        results.append(await self.test_anthropic_detailed())
        results.append(await self.test_deepseek_detailed())
        results.append(await self.test_openai_detailed())
        
        # Summary
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        
        for result in results:
            provider = result["provider"]
            auth = "✓" if result["auth_status"] == "success" else "✗"
            model = "✓" if result["model_status"] == "available" else "✗"
            
            print(f"\n{provider}:")
            print(f"  Authentication: {auth}")
            print(f"  Model Access: {model}")
            
            if result["recommendations"]:
                print("  Recommendations:")
                for rec in result["recommendations"]:
                    print(f"    - {rec}")
        
        # Overall status
        working_count = sum(1 for r in results if r["auth_status"] == "success" and r["model_status"] == "available")
        total_count = len(results)
        
        print(f"\n{working_count}/{total_count} providers fully operational")
        
        return {
            "total": total_count,
            "working": working_count,
            "results": results
        }


async def main():
    """Main test execution."""
    tester = DetailedAPITester()
    summary = await tester.run_all_tests()
    
    # Exit with appropriate code
    exit_code = 0 if summary["working"] == summary["total"] else 1
    return exit_code


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)