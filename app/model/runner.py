"""
Model Manager with Fallback Logic

Manages the fallback system between Anthropic, DeepSeek, and OpenAI models.
Priority: Anthropic -> DeepSeek -> OpenAI
"""

from typing import Dict, Any, List, Optional, Union
from .config import settings
from .model_logging import logger

from .anthropic_model import AnthropicModel
from .deepseek_model import DeepSeekModel  
from .openai_model import OpenAIModel


class ModelManager:
    """
    Manages AI model fallback system with automatic failover.
    
    Priority order:
    1. AnthropicModel (primary)
    2. DeepSeekModel (fallback #2)
    3. OpenAIModel (fallback #3)
    """
    
    def __init__(self):
        self.models = []
        self.current_model = None
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize models in priority order."""
        
        # Priority 1: Anthropic (primary)
        try:
            if settings.anthropic_api_key:
                anthropic_model = AnthropicModel()
                self.models.append(("anthropic", anthropic_model))
                logger.info("Anthropic model added to fallback chain")
        except Exception as e:
            logger.warning(f"Failed to initialize Anthropic model: {e}")
        
        # Priority 2: DeepSeek (fallback #2)  
        try:
            if settings.deepseek_api_key:
                deepseek_model = DeepSeekModel()
                self.models.append(("deepseek", deepseek_model))
                logger.info("DeepSeek model added to fallback chain")
        except Exception as e:
            logger.warning(f"Failed to initialize DeepSeek model: {e}")
        
        # Priority 3: OpenAI (fallback #3)
        try:
            if settings.openai_api_key:
                openai_model = OpenAIModel()
                self.models.append(("openai", openai_model))
                logger.info("OpenAI model added to fallback chain")
        except Exception as e:
            logger.warning(f"Failed to initialize OpenAI model: {e}")
        
        if not self.models:
            raise RuntimeError("No AI models available! Check your API keys.")
        
        # Set primary model
        self.current_model = self.models[0]
        logger.info(f"Primary model set to: {self.current_model[0]}")
    
    async def _try_with_fallback(self, method_name: str, *args, **kwargs):
        """
        Try method with automatic fallback to next available model.
        
        Args:
            method_name: Name of method to call
            *args, **kwargs: Arguments to pass to method
            
        Returns:
            Result from successful model call
            
        Raises:
            RuntimeError: If all models fail
        """
        last_error = None
        
        for model_name, model in self.models:
            try:
                logger.info(f"Trying {method_name} with {model_name} model")
                
                # Get the method from the model
                method = getattr(model, method_name)
                
                # Filter kwargs for models that don't support caching
                filtered_kwargs = kwargs.copy()
                if model_name != "anthropic" and "schema_info" in filtered_kwargs:
                    # Remove schema_info for non-Anthropic models (fallbacks)
                    filtered_kwargs.pop("schema_info")
                    logger.debug(f"Removed schema_info for {model_name} model (caching not supported)")
                
                # Call the method
                result = await method(*args, **filtered_kwargs)
                
                # Update current model if successful and different
                if self.current_model[0] != model_name:
                    logger.info(f"Switched to {model_name} model")
                    self.current_model = (model_name, model)
                
                return result
                
            except Exception as e:
                last_error = e
                logger.warning(f"{model_name} model failed for {method_name}: {e}")
                continue
        
        # All models failed
        raise RuntimeError(f"All models failed for {method_name}. Last error: {last_error}")
    
    async def generate_response(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        use_system_prompt: bool = True,
        schema_info: Optional[Dict] = None
    ) -> str:
        """Generate response with fallback support and optional caching."""
        return await self._try_with_fallback(
            "generate_response",
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            use_system_prompt=use_system_prompt,
            schema_info=schema_info
        )
    
    async def analyze_sql_query(self, query: str, schema_info: Dict) -> Dict[str, Any]:
        """Analyze SQL query with fallback support."""
        return await self._try_with_fallback(
            "analyze_sql_query",
            query,
            schema_info
        )
    
    async def synthesize_investigation_results(
        self,
        original_query: str,
        findings: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Synthesize results with fallback support."""
        return await self._try_with_fallback(
            "synthesize_investigation_results",
            original_query,
            findings
        )
    
    async def health_check(self) -> Dict[str, bool]:
        """Check health of all available models."""
        health_status = {}
        
        for model_name, model in self.models:
            try:
                is_healthy = await model.health_check()
                health_status[model_name] = is_healthy
                logger.info(f"{model_name} model health: {'✅' if is_healthy else '❌'}")
            except Exception as e:
                health_status[model_name] = False
                logger.error(f"{model_name} model health check failed: {e}")
        
        return health_status
    
    def get_current_model(self) -> str:
        """Get name of currently active model."""
        return self.current_model[0] if self.current_model else "none"
    
    def get_available_models(self) -> List[str]:
        """Get list of available model names."""
        return [model_name for model_name, _ in self.models]
    

if __name__ == "__main__":
    async def main():
        manager = ModelManager()
        print(await manager.health_check())

    import asyncio
    asyncio.run(main())