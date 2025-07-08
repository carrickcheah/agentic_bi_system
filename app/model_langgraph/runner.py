"""
LangGraph-based ModelManager with compatibility layer.
Maintains the same API as the original ModelManager while using LangGraph internally.
"""

import asyncio
from typing import Optional, List, Dict

from .graph import create_model_graph
from .state import ModelState
from .config import settings
from .model_logging import logger


class ModelManager:
    """
    LangGraph-based model manager with API compatibility.
    Drop-in replacement for the original ModelManager.
    """
    
    def __init__(self):
        """Initialize the model manager with LangGraph."""
        self.graph = create_model_graph()
        self.models = []  # For compatibility
        self.current_model = None  # For compatibility
        
        # Initialize available models list for compatibility
        if settings.anthropic_api_key:
            self.models.append(("anthropic", None))
        if settings.deepseek_api_key:
            self.models.append(("deepseek", None))
        if settings.openai_api_key:
            self.models.append(("openai", None))
        
        # Set default model
        if self.models:
            self.current_model = self.models[0]
        
        logger.info(f"ModelManager initialized with {len(self.models)} models using LangGraph")
    
    async def generate_response(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        use_system_prompt: bool = True,
        correlation_id: Optional[str] = None
    ) -> str:
        """
        Generate response using LangGraph workflow.
        Maintains compatibility with original API.
        """
        # Create initial state
        initial_state: ModelState = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "use_system_prompt": use_system_prompt,
            "correlation_id": correlation_id,
            "is_streaming": False,
            # Initialize other required fields with None/empty values
            "system_prompt": None,
            "schema_info": None,
            "available_models": {},
            "current_model": None,
            "attempted_models": [],
            "model_health_scores": {},
            "response": None,
            "response_chunks": [],
            "error": None,
            "error_count": 0,
            "last_error_timestamp": None,
            "retry_count": 0,
            "max_retries": 3,
            "start_time": None,
            "end_time": None,
            "model_latencies": {},
            "total_tokens_used": 0,
            "cache_key": None,
            "cache_hit": False,
            "anthropic_cache_enabled": settings.anthropic_enable_caching,
            "should_retry": False,
            "should_fallback": False,
            "next_model": None,
            "final_status": None,
            "business_domain": None,
            "user_role": None,
            "request_priority": None
        }
        
        try:
            # Run the graph
            final_state = await self.graph.ainvoke(initial_state)
            
            # Extract response
            if final_state.get("response"):
                # Update current model for compatibility
                if final_state.get("current_model"):
                    self.current_model = (final_state["current_model"], None)
                
                return final_state["response"]
            else:
                error = final_state.get("error", "Unknown error occurred")
                raise Exception(f"Failed to generate response: {error}")
                
        except Exception as e:
            logger.error(f"ModelManager.generate_response failed: {e}")
            raise
    
    async def generate_response_stream(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        use_system_prompt: bool = True,
        correlation_id: Optional[str] = None
    ):
        """
        Generate streaming response using LangGraph.
        Yields chunks as they are generated.
        """
        # For now, we'll implement a simple non-streaming version
        # Full streaming would require graph modifications
        response = await self.generate_response(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            use_system_prompt=use_system_prompt,
            correlation_id=correlation_id
        )
        
        # Simulate streaming by yielding in chunks
        chunk_size = 50
        for i in range(0, len(response), chunk_size):
            yield response[i:i + chunk_size]
            await asyncio.sleep(0.01)  # Small delay to simulate streaming
    
    def get_current_model(self) -> str:
        """Get the current model name for compatibility."""
        if self.current_model:
            return self.current_model[0]
        return "none"
    
    def get_available_models(self) -> List[str]:
        """Get list of available model names for compatibility."""
        return [model[0] for model in self.models]
    
    async def health_check(self) -> Dict[str, bool]:
        """
        Check health of all models using LangGraph.
        Returns dict of model_name -> is_healthy.
        """
        health_status = {}
        
        # Create a minimal state for health check
        health_state: ModelState = {
            "prompt": "Respond with OK",
            "max_tokens": 10,
            "temperature": 0,
            "use_system_prompt": False,
            "is_streaming": False,
            # Initialize other required fields
            "correlation_id": "health-check",
            "system_prompt": None,
            "schema_info": None,
            "available_models": {},
            "current_model": None,
            "attempted_models": [],
            "model_health_scores": {},
            "response": None,
            "response_chunks": [],
            "error": None,
            "error_count": 0,
            "last_error_timestamp": None,
            "retry_count": 0,
            "max_retries": 1,  # Don't retry health checks
            "start_time": None,
            "end_time": None,
            "model_latencies": {},
            "total_tokens_used": 0,
            "cache_key": None,
            "cache_hit": False,
            "anthropic_cache_enabled": False,
            "should_retry": False,
            "should_fallback": False,
            "next_model": None,
            "final_status": None,
            "business_domain": None,
            "user_role": None,
            "request_priority": None
        }
        
        # Run health check through graph
        try:
            final_state = await self.graph.ainvoke(health_state)
            
            # Extract health information from state
            for model_name in self.get_available_models():
                if model_name in final_state.get("model_health_scores", {}):
                    health_status[model_name] = final_state["model_health_scores"][model_name] > 0.5
                else:
                    health_status[model_name] = False
                    
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            # Mark all as unhealthy on error
            for model_name in self.get_available_models():
                health_status[model_name] = False
        
        return health_status
    
    async def validate_models(self) -> None:
        """
        Validate all configured models for compatibility.
        Raises exception if no models are available.
        """
        if not self.models:
            raise ValueError("No models configured. Please check your API keys.")
        
        # Run health check
        health_status = await self.health_check()
        healthy_models = [model for model, healthy in health_status.items() if healthy]
        
        if not healthy_models:
            raise ValueError("No healthy models available")
        
        logger.info(f"Validated {len(healthy_models)} healthy models: {healthy_models}")


# For backward compatibility
async def test_model_manager():
    """Test the LangGraph ModelManager."""
    manager = ModelManager()
    
    # Test basic generation
    response = await manager.generate_response(
        prompt="What is 2+2?",
        max_tokens=100,
        temperature=0
    )
    print(f"Response: {response}")
    
    # Test streaming
    print("\nStreaming response:")
    async for chunk in manager.generate_response_stream(
        prompt="Count from 1 to 5",
        max_tokens=100
    ):
        print(chunk, end="", flush=True)
    print()
    
    # Test health check
    health = await manager.health_check()
    print(f"\nHealth status: {health}")


if __name__ == "__main__":
    asyncio.run(test_model_manager())