"""
LangGraph state definitions for model orchestration.
Defines the shared state that flows through the graph.
"""

from typing import TypedDict, List, Dict, Any, Optional, Literal
from datetime import datetime
from enum import Enum


class ModelStatus(Enum):
    """Status of each model in the system."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class ModelChoice(Enum):
    """Available model choices."""
    ANTHROPIC = "anthropic"
    DEEPSEEK = "deepseek"
    OPENAI = "openai"


class ModelState(TypedDict):
    """
    State that flows through the LangGraph model orchestration.
    This is the shared context between all nodes.
    """
    
    # Input parameters
    prompt: str
    max_tokens: int
    temperature: float
    use_system_prompt: bool
    system_prompt: Optional[str]
    schema_info: Optional[Dict[str, Any]]
    correlation_id: str
    
    # Model selection and health
    available_models: Dict[str, ModelStatus]  # model_name -> status
    current_model: Optional[str]
    attempted_models: List[str]  # Track which models we've tried
    model_health_scores: Dict[str, float]  # model_name -> health score (0-1)
    
    # Response tracking
    response: Optional[str]
    response_chunks: List[str]  # For streaming support
    is_streaming: bool
    
    # Error handling
    error: Optional[str]
    error_count: int
    last_error_timestamp: Optional[datetime]
    retry_count: int
    max_retries: int
    
    # Performance metrics
    start_time: datetime
    end_time: Optional[datetime]
    model_latencies: Dict[str, float]  # model_name -> latency in seconds
    total_tokens_used: int
    
    # Caching and optimization
    cache_key: Optional[str]
    cache_hit: bool
    anthropic_cache_enabled: bool
    
    # Routing decisions
    should_retry: bool
    should_fallback: bool
    next_model: Optional[str]
    final_status: Optional[Literal["success", "failed", "partial"]]
    
    # Business context (for logging and analysis)
    business_domain: Optional[str]
    user_role: Optional[str]
    request_priority: Optional[Literal["low", "medium", "high", "critical"]]


class HealthCheckState(TypedDict):
    """
    Specialized state for health check operations.
    Runs independently from main request flow.
    """
    
    model_name: str
    check_timestamp: datetime
    is_healthy: bool
    response_time: float
    error_message: Optional[str]
    consecutive_failures: int
    circuit_breaker_open: bool


class EmbeddingState(TypedDict):
    """
    Specialized state for embedding generation.
    Separate flow from text generation.
    """
    
    text: str
    model: str  # Usually "openai"
    embedding_dimension: int
    embedding: Optional[List[float]]
    error: Optional[str]
    latency: float


# Response streaming event types
class StreamEvent(TypedDict):
    """Events emitted during streaming."""
    
    type: Literal["chunk", "error", "done", "metrics"]
    content: Optional[str]
    model: Optional[str]
    timestamp: datetime
    metadata: Optional[Dict[str, Any]]


# Model performance tracking
class ModelMetrics(TypedDict):
    """Aggregated metrics for each model."""
    
    model_name: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_latency: float
    p95_latency: float
    p99_latency: float
    health_score: float
    last_used: datetime
    last_failure: Optional[datetime]
    circuit_breaker_trips: int