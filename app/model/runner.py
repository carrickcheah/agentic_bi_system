"""
Enhanced Model Manager with Comprehensive Error Handling
Production-ready model management with proper error handling, monitoring, and resilience.
"""

from typing import Dict, Any, List, Optional, Union, Tuple
import uuid
import time
import asyncio
from datetime import datetime
from .config import settings
from .model_logging import logger

from .anthropic_model import AnthropicModel
from .deepseek_model import DeepSeekModel  
from .openai_model import OpenAIModel

# Import our error handling system
try:
    from ..core.error_handling import (
        error_boundary, with_error_handling, ExternalServiceError,
        ValidationError, ResourceExhaustedError, ErrorCategory, ErrorSeverity,
        validate_input, safe_execute
    )
except ImportError:
    # Fallback for standalone mode
    from contextlib import nullcontext
    
    def error_boundary(*args, **kwargs):
        return nullcontext()
    
    def with_error_handling(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    def validate_input(data, validator, operation, component, correlation_id=None):
        return validator(data)
    
    def safe_execute(func, operation, component, correlation_id=None, default_return=None, raise_on_failure=True):
        try:
            return func()
        except Exception as e:
            if raise_on_failure:
                raise
            return default_return
    
    class ExternalServiceError(Exception):
        pass

class ModelHealthMonitor:
    """Monitor model health and performance"""
    
    def __init__(self):
        self.model_stats = {}
        self.failure_counts = {}
        self.last_success = {}
        self.circuit_breaker_state = {}  # open, closed, half-open
    
    def record_success(self, model_name: str, response_time_ms: float):
        """Record successful model operation"""
        if model_name not in self.model_stats:
            self.model_stats[model_name] = {"success_count": 0, "total_time_ms": 0}
        
        self.model_stats[model_name]["success_count"] += 1
        self.model_stats[model_name]["total_time_ms"] += response_time_ms
        self.last_success[model_name] = datetime.utcnow()
        self.failure_counts[model_name] = 0  # Reset failure count
        self.circuit_breaker_state[model_name] = "closed"
    
    def record_failure(self, model_name: str, error: str):
        """Record model failure"""
        if model_name not in self.failure_counts:
            self.failure_counts[model_name] = 0
        
        self.failure_counts[model_name] += 1
        
        # Circuit breaker logic
        if self.failure_counts[model_name] >= 5:
            self.circuit_breaker_state[model_name] = "open"
            logger.warning(
                f"Circuit breaker opened for {model_name} due to repeated failures",
                extra={"model": model_name, "failure_count": self.failure_counts[model_name]}
            )
    
    def is_model_available(self, model_name: str) -> bool:
        """Check if model is available (circuit breaker check)"""
        state = self.circuit_breaker_state.get(model_name, "closed")
        if state == "open":
            # Check if we should try half-open
            last_failure_time = getattr(self, f"last_failure_{model_name}", datetime.utcnow())
            if (datetime.utcnow() - last_failure_time).seconds > 300:  # 5 minutes
                self.circuit_breaker_state[model_name] = "half-open"
                return True
            return False
        return True
    
    def get_model_stats(self) -> Dict[str, Any]:
        """Get model performance statistics"""
        stats = {}
        for model_name, data in self.model_stats.items():
            if data["success_count"] > 0:
                avg_response_time = data["total_time_ms"] / data["success_count"]
            else:
                avg_response_time = 0
            
            stats[model_name] = {
                "success_count": data["success_count"],
                "failure_count": self.failure_counts.get(model_name, 0),
                "avg_response_time_ms": avg_response_time,
                "circuit_breaker_state": self.circuit_breaker_state.get(model_name, "closed"),
                "last_success": self.last_success.get(model_name)
            }
        
        return stats

class EnhancedModelManager:
    """
    Production-ready model manager with comprehensive error handling.
    
    Features:
    - Structured error handling
    - Circuit breaker pattern
    - Performance monitoring
    - Correlation tracking
    - Input validation
    - Resource management
    """
    
    def __init__(self, validate_on_init: bool = True):
        self.models: List[Tuple[str, Any]] = []
        self.current_model: Optional[Tuple[str, Any]] = None
        self.health_monitor = ModelHealthMonitor()
        self.initialization_correlation_id = str(uuid.uuid4())
        self.validate_on_init = validate_on_init
        self._model_configs = {}  # Store model configurations for lazy init
        self._initialized_models = set()  # Track which models are initialized
        
        # Prepare model configurations instead of initializing
        self._prepare_model_configs()
    
    def _prepare_model_configs(self):
        """Prepare model configurations for lazy initialization."""
        
        logger.info(
            "Preparing model configurations",
            extra={"correlation_id": self.initialization_correlation_id}
        )
        
        # Check which models have API keys configured
        if settings.anthropic_api_key:
            self._model_configs["anthropic"] = self._init_anthropic
            logger.info("Anthropic model configuration prepared")
        
        if settings.deepseek_api_key:
            self._model_configs["deepseek"] = self._init_deepseek
            logger.info("DeepSeek model configuration prepared")
        
        if settings.openai_api_key:
            self._model_configs["openai"] = self._init_openai
            logger.info("OpenAI model configuration prepared")
        
        if not self._model_configs:
            raise ExternalServiceError("No model API keys configured")
        
        # Check if lazy initialization is enabled
        lazy_init = getattr(settings, 'lazy_model_initialization', True)
        
        if not lazy_init:
            # Initialize all models eagerly (old behavior)
            self._initialize_all_models()
        else:
            logger.info("Lazy model initialization enabled - models will be initialized on demand")
    
    def _init_anthropic(self) -> Optional[AnthropicModel]:
        """Initialize Anthropic model with validation"""
        if not settings.anthropic_api_key:
            logger.info("Anthropic API key not provided, skipping")
            return None
        
        if len(settings.anthropic_api_key) < 10:
            raise ValidationError("Anthropic API key appears invalid (too short)")
        
        return AnthropicModel()
    
    def _init_deepseek(self) -> Optional[DeepSeekModel]:
        """Initialize DeepSeek model with validation"""
        if not settings.deepseek_api_key:
            logger.info("DeepSeek API key not provided, skipping")
            return None
        
        if len(settings.deepseek_api_key) < 10:
            raise ValidationError("DeepSeek API key appears invalid (too short)")
        
        return DeepSeekModel()
    
    def _init_openai(self) -> Optional[OpenAIModel]:
        """Initialize OpenAI model with validation"""
        if not settings.openai_api_key:
            logger.info("OpenAI API key not provided, skipping")
            return None
        
        if len(settings.openai_api_key) < 10:
            raise ValidationError("OpenAI API key appears invalid (too short)")
        
        return OpenAIModel()
    
    def validate_generate_params(self, prompt: str, max_tokens: int, temperature: float) -> dict:
        """Validate generation parameters"""
        def validator(data):
            prompt, max_tokens, temperature = data
            
            if not isinstance(prompt, str) or len(prompt.strip()) == 0:
                raise ValueError("Prompt must be a non-empty string")
            
            if len(prompt) > 100000:
                raise ValueError("Prompt exceeds maximum length (100,000 chars)")
            
            if not isinstance(max_tokens, int) or max_tokens <= 0:
                raise ValueError("max_tokens must be a positive integer")
            
            if max_tokens > 32000:
                raise ValueError("max_tokens exceeds maximum (32,000)")
            
            if not isinstance(temperature, (int, float)):
                raise ValueError("temperature must be a number")
            
            if not 0.0 <= temperature <= 2.0:
                raise ValueError("temperature must be between 0.0 and 2.0")
            
            return {
                "prompt": prompt.strip(),
                "max_tokens": min(max_tokens, 32000),
                "temperature": max(0.0, min(temperature, 2.0))
            }
        
        return validate_input(
            (prompt, max_tokens, temperature),
            validator,
            operation="validate_generation_params",
            component="model_manager"
        )
    
    @with_error_handling(
        operation="generate_response",
        component="model_manager",
        timeout_seconds=120.0
    )
    async def generate_response(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        use_system_prompt: bool = True,
        schema_info: Optional[str] = None,
        correlation_id: Optional[str] = None
    ) -> str:
        """Generate response with comprehensive error handling and fallback."""
        
        correlation_id = correlation_id or str(uuid.uuid4())
        
        # Validate inputs
        validated_params = self.validate_generate_params(prompt, max_tokens, temperature)
        
        logger.info(
            "Starting response generation",
            extra={
                "correlation_id": correlation_id,
                "prompt_length": len(validated_params["prompt"]),
                "max_tokens": validated_params["max_tokens"],
                "temperature": validated_params["temperature"]
            }
        )
        
        # Ensure at least one model is initialized (lazy initialization)
        if not self.models:
            # Try to initialize the primary model (first configured)
            primary_model_name = next(iter(self._model_configs.keys()))
            primary_result = self._initialize_model_lazy(primary_model_name)
            if primary_result:
                self.current_model = primary_result
            else:
                # Try all configured models
                for model_name in self._model_configs:
                    result = self._initialize_model_lazy(model_name)
                    if result:
                        self.current_model = result
                        break
                
                if not self.current_model:
                    raise ExternalServiceError("No models could be initialized")
        
        # Try current model first
        if self.current_model:
            model_name, model = self.current_model
            
            if self.health_monitor.is_model_available(model_name):
                try:
                    response = await self._generate_with_model(
                        model_name, model, validated_params, use_system_prompt, schema_info, correlation_id
                    )
                    return response
                    
                except ExternalServiceError as e:
                    logger.warning(
                        f"Primary model {model_name} failed, attempting fallback",
                        extra={"correlation_id": correlation_id, "error": str(e), "model": model_name}
                    )
                    self.health_monitor.record_failure(model_name, str(e))
            else:
                logger.warning(
                    f"Model {model_name} unavailable (circuit breaker open)",
                    extra={"correlation_id": correlation_id, "model": model_name}
                )
        
        # Try fallback models
        for model_name, model in self.models:
            if (model_name, model) == self.current_model:
                continue  # Already tried
            
            if not self.health_monitor.is_model_available(model_name):
                continue
            
            try:
                logger.info(
                    f"Attempting fallback to {model_name}",
                    extra={"correlation_id": correlation_id, "model": model_name}
                )
                
                response = await self._generate_with_model(
                    model_name, model, validated_params, use_system_prompt, schema_info, correlation_id
                )
                
                # Update current model to successful one
                self.current_model = (model_name, model)
                logger.info(
                    f"Switched to {model_name} model",
                    extra={"correlation_id": correlation_id, "model": model_name}
                )
                
                return response
                
            except Exception as e:
                logger.warning(
                    f"Fallback to {model_name} failed: {str(e)}",
                    extra={"correlation_id": correlation_id, "error": str(e), "model": model_name}
                )
                self.health_monitor.record_failure(model_name, str(e))
                continue
        
        # All models failed
        raise ExternalServiceError("All models failed to generate response")
    
    async def _generate_with_model(
        self,
        model_name: str,
        model: Any,
        params: dict,
        use_system_prompt: bool,
        schema_info: Optional[str],
        correlation_id: str
    ) -> str:
        """Generate response with specific model and proper error handling."""
        
        start_time = time.time()
        
        try:
            logger.info(
                f"Trying generate_response with {model_name} model",
                extra={"correlation_id": correlation_id, "model": model_name}
            )
            
            response = await model.generate_response(
                params["prompt"],
                params["max_tokens"],
                params["temperature"],
                use_system_prompt,
                schema_info
            )
            
            if not response or len(response.strip()) == 0:
                raise ExternalServiceError(f"{model_name} returned empty response")
            
            response_time_ms = (time.time() - start_time) * 1000
            self.health_monitor.record_success(model_name, response_time_ms)
            
            logger.info(
                f"Response generated successfully with {model_name}",
                extra={
                    "correlation_id": correlation_id,
                    "model": model_name,
                    "response_length": len(response),
                    "response_time_ms": response_time_ms
                }
            )
            
            return response
            
        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000
            
            # Classify the error
            error_msg = str(e).lower()
            
            if "401" in error_msg or "authentication" in error_msg or "api key" in error_msg:
                raise ExternalServiceError(f"{model_name} authentication failed")
            elif "429" in error_msg or "rate limit" in error_msg:
                raise ExternalServiceError(f"{model_name} rate limited")
            elif "timeout" in error_msg:
                raise ExternalServiceError(f"{model_name} request timeout")
            else:
                raise ExternalServiceError(f"{model_name} API error: {str(e)}")
    
    def get_current_model(self) -> str:
        """Get current model name."""
        if self.current_model:
            return self.current_model[0]
        return "none"
    
    def get_available_models(self) -> List[str]:
        """Get list of available model names."""
        return [name for name, _ in self.models]
    
    def get_health_stats(self) -> Dict[str, Any]:
        """Get comprehensive health statistics."""
        return {
            "current_model": self.get_current_model(),
            "available_models": self.get_available_models(),
            "model_stats": self.health_monitor.get_model_stats(),
            "total_models": len(self.models)
        }
    
    def force_model_switch(self, model_name: str) -> bool:
        """Force switch to specific model if available."""
        for name, model in self.models:
            if name == model_name:
                self.current_model = (name, model)
                logger.info(
                    f"Manually switched to {model_name} model",
                    extra={"model": model_name}
                )
                return True
        return False
    
    async def validate_models(self):
        """Validate API keys for models, stopping after first success."""
        validated_models = []
        
        # Check if we should validate all models or stop at first success
        validate_all = settings.validate_all_models if hasattr(settings, 'validate_all_models') else False
        
        # For lazy initialization, we need to initialize models first
        if not self.models:
            # Initialize at least the primary model for validation
            for model_name in self._model_configs:
                result = self._initialize_model_lazy(model_name)
                if result and not validate_all:
                    # If we only need one model and got it, stop
                    break
        
        for model_name, model in self.models:
            try:
                logger.info(f"Validating {model_name} API key...")
                # Quick test to validate API key
                # Use higher max_tokens for models with thinking budgets
                test_max_tokens = 15000 if model_name == "anthropic" else 10
                await model.generate_response(
                    "Hi", max_tokens=test_max_tokens, temperature=0, use_system_prompt=False,
                    enable_thinking=False  # Disable thinking for validation to avoid token issues
                )
                logger.info(f"✅ {model_name} API key validated successfully")
                validated_models.append((model_name, model))
                
                # If primary model works and we're not validating all, stop here
                if not validate_all and (self.current_model is None or model_name == self.current_model[0]):
                    logger.info(f"Primary model {model_name} validated. Skipping fallback validation.")
                    # Set as current model if we don't have one yet
                    if self.current_model is None:
                        self.current_model = (model_name, model)
                    break
                    
            except Exception as e:
                logger.warning(f"❌ {model_name} API key validation failed: {str(e)}")
                self.health_monitor.record_failure(model_name, str(e))
        
        # Update models list with validated ones first
        if validated_models:
            # Put validated models first, keep others as fallback
            validated_names = [name for name, _ in validated_models]
            other_models = [(name, model) for name, model in self.models if name not in validated_names]
            self.models = validated_models + other_models
            self.current_model = self.models[0]
            logger.info(f"Models ready: {[name for name, _ in validated_models]} (validated), {[name for name, _ in other_models]} (fallback)")
        else:
            raise ExternalServiceError("No models passed API key validation")

    def _initialize_all_models(self):
        """Initialize all configured models (eager initialization)."""
        for model_name, init_func in self._model_configs.items():
            try:
                model_instance = safe_execute(
                    init_func,
                    operation=f"initialize_{model_name}",
                    component="model_manager",
                    correlation_id=self.initialization_correlation_id,
                    default_return=None,
                    raise_on_failure=False
                )
                
                if model_instance:
                    self.models.append((model_name, model_instance))
                    self._initialized_models.add(model_name)
                    logger.info(
                        f"{model_name.title()} model initialized and added to fallback chain",
                        extra={
                            "correlation_id": self.initialization_correlation_id,
                            "model": model_name
                        }
                    )
                    
            except Exception as e:
                logger.warning(
                    f"Failed to initialize {model_name} model",
                    extra={
                        "correlation_id": self.initialization_correlation_id,
                        "model": model_name,
                        "error": str(e)
                    }
                )
        
        # Set primary model
        if self.models:
            self.current_model = self.models[0]
            logger.info(
                f"Primary model set to: {self.current_model[0]}",
                extra={
                    "correlation_id": self.initialization_correlation_id,
                    "primary_model": self.current_model[0]
                }
            )
    
    def _initialize_model_lazy(self, model_name: str) -> Optional[Tuple[str, Any]]:
        """Initialize a specific model on demand."""
        if model_name in self._initialized_models:
            # Already initialized
            for name, model in self.models:
                if name == model_name:
                    return (name, model)
            return None
        
        if model_name not in self._model_configs:
            logger.warning(f"Model {model_name} not configured")
            return None
        
        try:
            init_func = self._model_configs[model_name]
            model_instance = safe_execute(
                init_func,
                operation=f"lazy_initialize_{model_name}",
                component="model_manager",
                correlation_id=self.initialization_correlation_id,
                default_return=None,
                raise_on_failure=False
            )
            
            if model_instance:
                self.models.append((model_name, model_instance))
                self._initialized_models.add(model_name)
                logger.info(
                    f"{model_name.title()} model initialized lazily",
                    extra={
                        "correlation_id": self.initialization_correlation_id,
                        "model": model_name
                    }
                )
                return (model_name, model_instance)
                
        except Exception as e:
            logger.warning(
                f"Failed to lazily initialize {model_name} model",
                extra={
                    "correlation_id": self.initialization_correlation_id,
                    "model": model_name,
                    "error": str(e)
                }
            )
        
        return None

# Maintain compatibility with existing code
ModelManager = EnhancedModelManager