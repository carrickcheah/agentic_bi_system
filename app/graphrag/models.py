"""
GraphRAG Pydantic Models

Data models for GraphRAG operations and MCP interface.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field


class GraphRAGEntitySearchRequest(BaseModel):
    """Request model for GraphRAG entity search."""
    
    query: str = Field(
        description="Natural language search query for entities"
    )
    limit: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum number of entities to return"
    )
    timeout: Optional[float] = Field(
        default=None,
        description="Timeout for this search operation"
    )


class GraphRAGEntity(BaseModel):
    """GraphRAG entity model."""
    
    id: str = Field(description="Entity unique identifier")
    name: str = Field(description="Entity name")
    type: str = Field(description="Entity type (person, organization, etc.)")
    description: Optional[str] = Field(
        default=None,
        description="Entity description"
    )
    importance_score: Optional[float] = Field(
        default=None,
        description="Entity importance score"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional entity metadata"
    )


class GraphRAGEntitySearchResponse(BaseModel):
    """Response model for GraphRAG entity search."""
    
    status: str = Field(description="Operation status")
    entities: List[GraphRAGEntity] = Field(
        description="List of matching entities"
    )
    count: int = Field(description="Number of entities returned")
    execution_time_seconds: float = Field(
        description="Execution time in seconds"
    )
    cache_hit: bool = Field(
        default=False,
        description="Whether result came from cache"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional response metadata"
    )


class GraphRAGGlobalSearchRequest(BaseModel):
    """Request model for GraphRAG global search."""
    
    query: str = Field(
        description="Business question for global analysis"
    )
    max_communities: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum business domains to analyze"
    )
    timeout: Optional[float] = Field(
        default=None,
        description="Timeout for this search operation"
    )
    max_cost: Optional[float] = Field(
        default=None,
        description="Maximum cost for this operation"
    )


class GraphRAGCommunityInsight(BaseModel):
    """GraphRAG community insight model."""
    
    community_id: str = Field(description="Community identifier")
    community_name: str = Field(description="Community name")
    insight: str = Field(description="Generated insight")
    confidence_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score for insight"
    )
    supporting_entities: List[str] = Field(
        description="Supporting entity names"
    )
    cost: float = Field(
        description="Cost for generating this insight"
    )


class GraphRAGGlobalSearchResponse(BaseModel):
    """Response model for GraphRAG global search."""
    
    status: str = Field(description="Operation status")
    search_id: str = Field(description="Unique search identifier")
    insights: List[GraphRAGCommunityInsight] = Field(
        description="Cross-domain business insights"
    )
    synthesized_answer: str = Field(
        description="Final synthesized answer"
    )
    metadata: Dict[str, Any] = Field(
        description="Operation metadata"
    )


class GraphRAGServerHealth(BaseModel):
    """GraphRAG server health status model."""
    
    status: str = Field(description="Overall server status")
    uptime_seconds: float = Field(description="Server uptime")
    memory_usage_gb: float = Field(description="Current memory usage")
    active_requests: int = Field(description="Number of active requests")
    cache_hit_rate: float = Field(description="Cache hit rate percentage")
    total_requests: int = Field(description="Total requests processed")
    errors_last_hour: int = Field(description="Errors in last hour")
    data_files_loaded: bool = Field(description="Whether data files are loaded")
    last_updated: datetime = Field(description="Last health check timestamp")


class GraphRAGPerformanceMetrics(BaseModel):
    """GraphRAG performance metrics model."""
    
    entity_search_avg_time: float = Field(
        description="Average entity search time"
    )
    global_search_avg_time: float = Field(
        description="Average global search time"
    )
    cache_hit_percentage: float = Field(
        description="Cache hit percentage"
    )
    total_cost_today: float = Field(
        description="Total cost today in USD"
    )
    requests_per_minute: float = Field(
        description="Average requests per minute"
    )
    error_rate_percentage: float = Field(
        description="Error rate percentage"
    )


class GraphRAGErrorResponse(BaseModel):
    """GraphRAG error response model."""
    
    status: str = Field(default="error", description="Status indicator")
    error_type: str = Field(description="Type of error")
    error_message: str = Field(description="Human-readable error message")
    error_code: Optional[str] = Field(
        default=None,
        description="Error code for programmatic handling"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Error timestamp"
    )
    request_id: Optional[str] = Field(
        default=None,
        description="Request ID for tracing"
    )
    fallback_available: bool = Field(
        default=False,
        description="Whether fallback option is available"
    )
    retry_after_seconds: Optional[float] = Field(
        default=None,
        description="Suggested retry delay"
    )