"""
LangGraph node implementations for model orchestration.
Each node represents a specific operation in the model workflow.
"""

import asyncio
import time
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List

from .state import ModelState, ModelStatus, StreamEvent
from .config import settings
from .model_logging import logger
from .prompts import get_prompt


# Simplified model imports - we'll create these next
from .anthropic_model import AnthropicModel
from .deepseek_model import DeepSeekModel
from .openai_model import OpenAIModel


async def initialize_node(state: ModelState) -> ModelState:
    """
    Initialize the request state with defaults and generate correlation ID.
    This is the entry point for all requests.
    """
    logger.info(
        "Initializing model request",
        extra={
            "correlation_id": state.get("correlation_id"),
            "prompt_preview": state["prompt"][:100] + "..." if len(state["prompt"]) > 100 else state["prompt"]
        }
    )
    
    # Set defaults
    state["correlation_id"] = state.get("correlation_id") or str(uuid.uuid4())
    state["start_time"] = datetime.now()
    state["attempted_models"] = []
    state["response_chunks"] = []
    state["error_count"] = 0
    state["retry_count"] = 0
    state["max_retries"] = 3
    state["cache_hit"] = False
    state["should_retry"] = False
    state["should_fallback"] = False
    state["model_latencies"] = {}
    
    # Set system prompt if requested
    if state.get("use_system_prompt") and not state.get("system_prompt"):
        state["system_prompt"] = get_prompt("sql_agent")
    
    # Initialize available models based on configuration
    state["available_models"] = {}
    if settings.anthropic_api_key:
        state["available_models"]["anthropic"] = ModelStatus.UNKNOWN
    if settings.deepseek_api_key:
        state["available_models"]["deepseek"] = ModelStatus.UNKNOWN
    if settings.openai_api_key:
        state["available_models"]["openai"] = ModelStatus.UNKNOWN
    
    logger.info(
        f"Request initialized with {len(state['available_models'])} available models",
        extra={"correlation_id": state["correlation_id"]}
    )
    
    return state


async def check_model_health_node(state: ModelState) -> ModelState:
    """
    Check health status of all available models.
    Updates the available_models status based on recent performance.
    """
    logger.info(
        "Checking model health status",
        extra={"correlation_id": state["correlation_id"]}
    )
    
    # For now, we'll use a simplified health check
    # In production, this would check circuit breaker states, recent failures, etc.
    
    # Default all models to healthy if they have API keys
    for model_name in state["available_models"]:
        # Check if we have recent failures for this model
        if model_name in state.get("model_health_scores", {}):
            score = state["model_health_scores"][model_name]
            if score > 0.8:
                state["available_models"][model_name] = ModelStatus.HEALTHY
            elif score > 0.5:
                state["available_models"][model_name] = ModelStatus.DEGRADED
            else:
                state["available_models"][model_name] = ModelStatus.UNHEALTHY
        else:
            # No health data, assume healthy
            state["available_models"][model_name] = ModelStatus.HEALTHY
    
    healthy_models = [
        name for name, status in state["available_models"].items()
        if status == ModelStatus.HEALTHY
    ]
    
    logger.info(
        f"Health check complete: {len(healthy_models)} healthy models",
        extra={
            "correlation_id": state["correlation_id"],
            "healthy_models": healthy_models
        }
    )
    
    return state


async def select_model_node(state: ModelState) -> ModelState:
    """
    Select the best model based on availability, health, and priority.
    Considers previous attempts and failures.
    """
    logger.info(
        "Selecting model for request",
        extra={"correlation_id": state["correlation_id"]}
    )
    
    # Model priority order
    model_priority = ["anthropic", "deepseek", "openai"]
    
    # Filter out already attempted models
    available_models = [
        model for model in model_priority
        if model in state["available_models"]
        and model not in state["attempted_models"]
        and state["available_models"][model] in [ModelStatus.HEALTHY, ModelStatus.DEGRADED]
    ]
    
    if not available_models:
        logger.error(
            "No available models remaining",
            extra={"correlation_id": state["correlation_id"]}
        )
        state["error"] = "All models have been attempted or are unhealthy"
        state["final_status"] = "failed"
        return state
    
    # Select the first available model
    selected_model = available_models[0]
    state["current_model"] = selected_model
    state["next_model"] = available_models[1] if len(available_models) > 1 else None
    
    logger.info(
        f"Selected model: {selected_model}",
        extra={
            "correlation_id": state["correlation_id"],
            "fallback_available": state["next_model"] is not None
        }
    )
    
    return state


async def generate_response_node(state: ModelState) -> ModelState:
    """
    Generate response using the selected model.
    Handles both streaming and non-streaming responses.
    """
    model_name = state["current_model"]
    if not model_name:
        state["error"] = "No model selected"
        return state
    
    logger.info(
        f"Generating response with {model_name}",
        extra={"correlation_id": state["correlation_id"]}
    )
    
    start_time = time.time()
    
    try:
        # Initialize the appropriate model
        if model_name == "anthropic":
            model = AnthropicModel()
        elif model_name == "deepseek":
            model = DeepSeekModel()
        elif model_name == "openai":
            model = OpenAIModel()
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Record attempt
        state["attempted_models"].append(model_name)
        
        # Generate response
        if state.get("is_streaming", False):
            # Streaming response
            chunks = []
            async for chunk in model.generate_stream(
                prompt=state["prompt"],
                max_tokens=state["max_tokens"],
                temperature=state["temperature"],
                system_prompt=state.get("system_prompt")
            ):
                chunks.append(chunk)
                state["response_chunks"].append(chunk)
            
            state["response"] = "".join(chunks)
        else:
            # Non-streaming response
            response = await model.generate(
                prompt=state["prompt"],
                max_tokens=state["max_tokens"],
                temperature=state["temperature"],
                system_prompt=state.get("system_prompt")
            )
            state["response"] = response
        
        # Record success
        latency = time.time() - start_time
        state["model_latencies"][model_name] = latency
        state["final_status"] = "success"
        
        # Update health score (simple version)
        if "model_health_scores" not in state:
            state["model_health_scores"] = {}
        state["model_health_scores"][model_name] = 1.0  # Success = perfect health
        
        logger.info(
            f"Response generated successfully in {latency:.2f}s",
            extra={
                "correlation_id": state["correlation_id"],
                "model": model_name,
                "response_length": len(state["response"])
            }
        )
        
    except Exception as e:
        # Record failure
        latency = time.time() - start_time
        state["model_latencies"][model_name] = latency
        state["error"] = str(e)
        state["error_count"] += 1
        state["last_error_timestamp"] = datetime.now()
        
        # Update health score
        if "model_health_scores" not in state:
            state["model_health_scores"] = {}
        current_score = state["model_health_scores"].get(model_name, 1.0)
        state["model_health_scores"][model_name] = max(0, current_score - 0.3)  # Decrease health
        
        # Determine if we should retry or fallback
        if state["retry_count"] < state["max_retries"] and "rate limit" in str(e).lower():
            state["should_retry"] = True
        elif state["next_model"]:
            state["should_fallback"] = True
        else:
            state["final_status"] = "failed"
        
        logger.error(
            f"Failed to generate response: {e}",
            extra={
                "correlation_id": state["correlation_id"],
                "model": model_name,
                "error_count": state["error_count"]
            }
        )
    
    return state


async def handle_error_node(state: ModelState) -> ModelState:
    """
    Handle errors with retry logic and exponential backoff.
    """
    if not state.get("should_retry"):
        return state
    
    logger.info(
        f"Retrying request (attempt {state['retry_count'] + 1}/{state['max_retries']})",
        extra={"correlation_id": state["correlation_id"]}
    )
    
    # Exponential backoff
    wait_time = 2 ** state["retry_count"]
    await asyncio.sleep(wait_time)
    
    # Increment retry count
    state["retry_count"] += 1
    
    # Clear error state for retry
    state["error"] = None
    state["should_retry"] = False
    
    # Remove current model from attempted list to retry it
    if state["current_model"] in state["attempted_models"]:
        state["attempted_models"].remove(state["current_model"])
    
    return state


async def collect_metrics_node(state: ModelState) -> ModelState:
    """
    Collect and log performance metrics for the request.
    """
    state["end_time"] = datetime.now()
    total_duration = (state["end_time"] - state["start_time"]).total_seconds()
    
    metrics = {
        "correlation_id": state["correlation_id"],
        "total_duration": total_duration,
        "models_attempted": state["attempted_models"],
        "final_model": state.get("current_model"),
        "final_status": state.get("final_status", "unknown"),
        "error_count": state["error_count"],
        "retry_count": state["retry_count"],
        "cache_hit": state["cache_hit"],
        "model_latencies": state["model_latencies"],
        "response_length": len(state.get("response", "")) if state.get("response") else 0
    }
    
    logger.info(
        "Request completed",
        extra=metrics
    )
    
    # Emit metrics event if streaming
    if state.get("is_streaming"):
        event = StreamEvent(
            type="metrics",
            content=None,
            model=state.get("current_model"),
            timestamp=datetime.now(),
            metadata=metrics
        )
        # In a real implementation, this would be emitted to a stream
        logger.debug(f"Stream event: {event}")
    
    return state


async def validate_input_node(state: ModelState) -> ModelState:
    """
    Validate input parameters and set appropriate defaults.
    """
    # Validate prompt
    if not state.get("prompt"):
        state["error"] = "Prompt is required"
        state["final_status"] = "failed"
        return state
    
    # Validate and set defaults
    state["max_tokens"] = min(
        state.get("max_tokens", 2048),
        16000  # Maximum allowed
    )
    
    state["temperature"] = max(0.0, min(
        state.get("temperature", 0.7),
        2.0
    ))
    
    # Check for Anthropic cache settings
    if state.get("current_model") == "anthropic" and settings.anthropic_enable_caching:
        state["anthropic_cache_enabled"] = True
    
    return state