"""
Simplified Anthropic model implementation for LangGraph.
Focuses only on API interaction, leaving orchestration to the graph.
"""

from typing import Optional, AsyncGenerator
from anthropic import AsyncAnthropic

from .config import settings
from .model_logging import logger


class AnthropicModel:
    """Anthropic Claude model interface."""
    
    def __init__(self):
        self.client = AsyncAnthropic(
            api_key=settings.anthropic_api_key,
            max_retries=1,  # Let LangGraph handle retries
            timeout=settings.request_timeout
        )
        self.model = settings.anthropic_model
        
    async def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None
    ) -> str:
        """Generate a single response."""
        
        messages = [{"role": "user", "content": prompt}]
        
        # Build API parameters
        params = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages
        }
        
        # Add system prompt if provided
        if system_prompt:
            params["system"] = [{"type": "text", "text": system_prompt}]
        
        # Add caching if enabled
        if settings.anthropic_enable_caching:
            # Cache system prompt
            if system_prompt and settings.cache_system_prompt:
                params["system"][0]["cache_control"] = {"type": "ephemeral"}
            
            # Cache schema info if provided
            if settings.cache_schema_info and "database schema" in prompt.lower():
                messages[0]["cache_control"] = {"type": "ephemeral"}
        
        # Add extended thinking if supported and enabled
        if settings.enable_thinking and self._supports_thinking():
            params["thinking"] = {
                "type": "enabled",
                "budget_tokens": settings.thinking_budget
            }
            params["temperature"] = 1.0  # Required for thinking
        
        # Make API call
        response = await self.client.messages.create(**params)
        
        # Extract content
        if hasattr(response, 'content') and response.content:
            return response.content[0].text
        
        raise ValueError("Empty response from Anthropic")
    
    async def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """Generate a streaming response."""
        
        messages = [{"role": "user", "content": prompt}]
        
        # Build API parameters (same as generate)
        params = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages
        }
        
        if system_prompt:
            params["system"] = [{"type": "text", "text": system_prompt}]
        
        # Add caching if enabled
        if settings.anthropic_enable_caching:
            if system_prompt and settings.cache_system_prompt:
                params["system"][0]["cache_control"] = {"type": "ephemeral"}
            if settings.cache_schema_info and "database schema" in prompt.lower():
                messages[0]["cache_control"] = {"type": "ephemeral"}
        
        # Add extended thinking if supported
        if settings.enable_thinking and self._supports_thinking():
            params["thinking"] = {
                "type": "enabled",
                "budget_tokens": settings.thinking_budget
            }
            params["temperature"] = 1.0
        
        # Stream response
        async with self.client.messages.stream(**params) as stream:
            async for text in stream.text_stream:
                yield text
    
    def _supports_thinking(self) -> bool:
        """Check if model supports extended thinking."""
        thinking_models = ["claude-opus-4", "claude-sonnet-4", "claude-sonnet-3.7"]
        return any(model in self.model for model in thinking_models)
    
    async def health_check(self) -> bool:
        """Simple health check."""
        try:
            response = await self.generate(
                prompt="Respond with 'OK'",
                max_tokens=10,
                temperature=0
            )
            return "OK" in response
        except Exception as e:
            logger.error(f"Anthropic health check failed: {e}")
            return False