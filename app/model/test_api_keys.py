#!/usr/bin/env python3
"""
Test script to verify API keys for Anthropic, DeepSeek, and OpenAI models.
Tests authentication and basic functionality for each provider.
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


class APIKeyTester:
    """Test API keys for different AI providers."""
    
    def __init__(self):
        self.settings = TestSettings()
        self.test_prompt = "Say 'Hello World' in exactly 3 words."
        
    async def test_anthropic(self) -> Dict[str, Any]:
        """Test Anthropic API key and model."""
        print("\n=== Testing Anthropic API ===")
        print(f"Model: {self.settings.anthropic_model}")
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": self.settings.anthropic_api_key,
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
                    data = response.json()
                    content = data.get("content", [{}])[0].get("text", "")
                    print(f"SUCCESS: Response received")
                    print(f"Response: {content[:100]}...")
                    return {"status": "success", "provider": "Anthropic", "response": content}
                else:
                    error_msg = f"HTTP {response.status_code}: {response.text}"
                    print(f"FAILED: {error_msg}")
                    return {"status": "failed", "provider": "Anthropic", "error": error_msg}
                    
        except Exception as e:
            error_msg = f"Exception: {str(e)}"
            print(f"FAILED: {error_msg}")
            return {"status": "failed", "provider": "Anthropic", "error": error_msg}
    
    async def test_deepseek(self) -> Dict[str, Any]:
        """Test DeepSeek API key and model."""
        print("\n=== Testing DeepSeek API ===")
        print(f"Model: {self.settings.deepseek_model}")
        print(f"Base URL: {self.settings.deepseek_base_url}")
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.settings.deepseek_base_url}/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.settings.deepseek_api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": self.settings.deepseek_model,
                        "messages": [
                            {"role": "user", "content": self.test_prompt}
                        ],
                        "max_tokens": 50,
                        "temperature": 0.7
                    },
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    data = response.json()
                    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    print(f"SUCCESS: Response received")
                    print(f"Response: {content[:100]}...")
                    return {"status": "success", "provider": "DeepSeek", "response": content}
                else:
                    error_msg = f"HTTP {response.status_code}: {response.text}"
                    print(f"FAILED: {error_msg}")
                    return {"status": "failed", "provider": "DeepSeek", "error": error_msg}
                    
        except Exception as e:
            error_msg = f"Exception: {str(e)}"
            print(f"FAILED: {error_msg}")
            return {"status": "failed", "provider": "DeepSeek", "error": error_msg}
    
    async def test_openai(self) -> Dict[str, Any]:
        """Test OpenAI API key and model."""
        print("\n=== Testing OpenAI API ===")
        print(f"Model: {self.settings.openai_model}")
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.settings.openai_api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": self.settings.openai_model,
                        "messages": [
                            {"role": "user", "content": self.test_prompt}
                        ],
                        "max_tokens": 50,
                        "temperature": 0.7
                    },
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    data = response.json()
                    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    print(f"SUCCESS: Response received")
                    print(f"Response: {content[:100]}...")
                    return {"status": "success", "provider": "OpenAI", "response": content}
                else:
                    error_msg = f"HTTP {response.status_code}: {response.text}"
                    print(f"FAILED: {error_msg}")
                    return {"status": "failed", "provider": "OpenAI", "error": error_msg}
                    
        except Exception as e:
            error_msg = f"Exception: {str(e)}"
            print(f"FAILED: {error_msg}")
            return {"status": "failed", "provider": "OpenAI", "error": error_msg}
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run tests for all providers."""
        print("Starting API Key Tests...")
        print(f"Test prompt: '{self.test_prompt}'")
        
        results = []
        
        # Test each provider
        results.append(await self.test_anthropic())
        results.append(await self.test_deepseek())
        results.append(await self.test_openai())
        
        # Summary
        print("\n=== Test Summary ===")
        success_count = sum(1 for r in results if r["status"] == "success")
        failed_count = sum(1 for r in results if r["status"] == "failed")
        
        for result in results:
            provider = result["provider"]
            status = result["status"].upper()
            print(f"{provider}: {status}")
            if result["status"] == "failed":
                print(f"  Error: {result['error']}")
        
        print(f"\nTotal: {success_count} passed, {failed_count} failed")
        
        return {
            "summary": {
                "total": len(results),
                "passed": success_count,
                "failed": failed_count
            },
            "results": results
        }


async def main():
    """Main test execution."""
    tester = APIKeyTester()
    results = await tester.run_all_tests()
    
    # Exit with appropriate code
    exit_code = 0 if results["summary"]["failed"] == 0 else 1
    return exit_code


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)