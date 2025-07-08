"""
Simplified OpenAI model implementation for LangGraph.
Focuses only on API interaction, leaving orchestration to the graph.
"""

from typing import Optional, AsyncGenerator
from openai import AsyncOpenAI

from .config import settings
from .model_logging import logger


class OpenAIModel:
    """OpenAI GPT model interface."""
    
    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=settings.openai_api_key,
            max_retries=1,  # Let LangGraph handle retries
            timeout=settings.request_timeout
        )
        self.model = settings.openai_model
    
    async def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None
    ) -> str:
        """Generate a single response."""
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        if response.choices and response.choices[0].message.content:
            return response.choices[0].message.content
        
        raise ValueError("Empty response from OpenAI")
    
    async def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """Generate a streaming response."""
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True
        )
        
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    
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
            logger.error(f"OpenAI health check failed: {e}")
            return False