"""
LangGraph edge routing logic for model orchestration.
Defines conditional routing between nodes based on state.
"""

from typing import Literal
from .state import ModelState, ModelStatus


def route_after_initialization(state: ModelState) -> Literal["validate_input", "collect_metrics"]:
    """
    Route after initialization node.
    If no models available, skip to metrics (will fail).
    """
    if not state.get("available_models"):
        state["error"] = "No models configured"
        state["final_status"] = "failed"
        return "collect_metrics"
    
    return "validate_input"


def route_after_validation(state: ModelState) -> Literal["check_health", "collect_metrics"]:
    """
    Route after input validation.
    If validation failed, skip to metrics.
    """
    if state.get("error") or state.get("final_status") == "failed":
        return "collect_metrics"
    
    return "check_health"


def route_after_health_check(state: ModelState) -> Literal["select_model", "collect_metrics"]:
    """
    Route after health check.
    If no healthy models, skip to metrics.
    """
    healthy_models = [
        name for name, status in state["available_models"].items()
        if status in [ModelStatus.HEALTHY, ModelStatus.DEGRADED]
    ]
    
    if not healthy_models:
        state["error"] = "No healthy models available"
        state["final_status"] = "failed"
        return "collect_metrics"
    
    return "select_model"


def route_after_model_selection(state: ModelState) -> Literal["generate_response", "collect_metrics"]:
    """
    Route after model selection.
    If no model selected, skip to metrics.
    """
    if not state.get("current_model"):
        return "collect_metrics"
    
    return "generate_response"


def route_after_generation(state: ModelState) -> Literal["handle_error", "collect_metrics"]:
    """
    Route after response generation.
    Determine if we need error handling or can finish.
    """
    # Success - go to metrics
    if state.get("response") and not state.get("error"):
        return "collect_metrics"
    
    # Error occurred - check if we should retry or fallback
    if state.get("should_retry") or state.get("should_fallback"):
        return "handle_error"
    
    # No recovery possible - go to metrics
    return "collect_metrics"


def route_after_error_handling(state: ModelState) -> Literal["generate_response", "select_model", "collect_metrics"]:
    """
    Route after error handling.
    Determine next action based on error type.
    """
    # If we should retry with same model
    if state.get("retry_count", 0) < state.get("max_retries", 3) and not state.get("should_fallback"):
        return "generate_response"
    
    # If we should try a different model
    if state.get("should_fallback") and state.get("next_model"):
        state["should_fallback"] = False  # Reset flag
        return "select_model"
    
    # No more options - finish
    return "collect_metrics"


def determine_final_status(state: ModelState) -> ModelState:
    """
    Determine the final status of the request.
    Called before metrics collection.
    """
    if state.get("response"):
        state["final_status"] = "success"
    elif state.get("error"):
        state["final_status"] = "failed"
    else:
        state["final_status"] = "partial"
    
    return state


# Health monitoring edge logic
def should_trigger_health_check(state: ModelState) -> bool:
    """
    Determine if we should trigger an async health check.
    This runs in parallel to main request flow.
    """
    # Check if any model has unknown status
    unknown_models = [
        name for name, status in state.get("available_models", {}).items()
        if status == ModelStatus.UNKNOWN
    ]
    
    # Check if any model health score is below threshold
    low_health_models = [
        name for name, score in state.get("model_health_scores", {}).items()
        if score < 0.5
    ]
    
    return bool(unknown_models or low_health_models)


# Circuit breaker edge logic
def should_open_circuit_breaker(model_name: str, state: ModelState) -> bool:
    """
    Determine if circuit breaker should open for a model.
    Based on consecutive failures and error rate.
    """
    # Get model-specific error count (would be tracked in production)
    error_count = state.get("error_count", 0)
    
    # Simple logic: open after 3 consecutive failures
    # In production, this would consider time windows and error rates
    return error_count >= 3


def should_close_circuit_breaker(model_name: str, state: ModelState) -> bool:
    """
    Determine if circuit breaker can close (half-open -> closed).
    Based on successful health checks.
    """
    # Check if model is responding to health checks
    health_score = state.get("model_health_scores", {}).get(model_name, 0)
    
    # Close if health score is good
    return health_score > 0.8


# Caching edge logic
def should_check_cache(state: ModelState) -> bool:
    """
    Determine if we should check cache before generating.
    """
    # Check cache if:
    # 1. Anthropic model with caching enabled
    # 2. Request has been seen before (would check cache key in production)
    
    return (
        state.get("current_model") == "anthropic" and
        state.get("anthropic_cache_enabled", False)
    )


def should_store_in_cache(state: ModelState) -> bool:
    """
    Determine if response should be stored in cache.
    """
    # Store if:
    # 1. Request was successful
    # 2. Response is substantial (not an error message)
    # 3. Caching is enabled for the model
    
    return (
        state.get("final_status") == "success" and
        state.get("response") and
        len(state.get("response", "")) > 100 and
        state.get("anthropic_cache_enabled", False)
    )