"""
Intelligence Module Configuration - Self-Contained Architecture
Implements Phase 2: Strategy Planning configuration with pydantic-settings.
NO hardcoded business defaults - all values from environment.
"""

from pathlib import Path
from typing import List, Optional
from pydantic import Field, validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class IntelligenceSettings(BaseSettings):
    """Intelligence Module configuration - loads from local settings.env"""
    
    model_config = SettingsConfigDict(
        env_file=Path(__file__).parent / "settings.env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Core Strategy Planning Configuration
    strategy_planning_timeout: int = Field(
        default=30, 
        description="Maximum time for strategy planning in seconds"
    )
    
    # Domain Classification Settings  
    domain_confidence_threshold: float = Field(
        default=0.25,
        description="Minimum confidence for domain classification"
    )
    
    # Intent Classification Settings
    enable_intent_pre_classification: bool = Field(
        default=True,
        description="Enable intent pre-classification for greetings vs business queries"
    )
    intent_confidence_threshold: float = Field(
        default=0.3,
        description="Minimum confidence for intent classification"
    )
    business_intent_threshold: float = Field(
        default=0.2,
        description="Threshold for defaulting to business intent when uncertain"
    )
    max_greeting_words: int = Field(
        default=10,
        description="Maximum words in query to consider for greeting classification"
    )
    
    # Complexity Analysis Settings
    complexity_scoring_weights: dict = Field(
        default={
            "data_sources": 0.3,
            "time_range": 0.2,
            "analytical_depth": 0.25,
            "cross_validation": 0.15,
            "business_impact": 0.1
        },
        description="Complexity scoring dimension weights"
    )
    
    # Pattern Recognition Settings
    pattern_similarity_threshold: float = Field(
        default=0.85,
        description="Minimum similarity for pattern matching"
    )
    
    pattern_cache_size: int = Field(
        default=1000,
        description="Maximum number of cached patterns"
    )
    
    # Business Context Settings
    context_adaptation_enabled: bool = Field(
        default=True,
        description="Enable context-aware strategy adaptation"
    )
    
    user_role_weights: dict = Field(
        default={
            "manager": {"speed": 0.8, "detail": 0.4, "automation": 0.9},
            "analyst": {"speed": 0.4, "detail": 0.9, "automation": 0.6},
            "engineer": {"speed": 0.5, "detail": 0.8, "automation": 0.7},
            "executive": {"speed": 0.9, "detail": 0.3, "automation": 0.8}
        },
        description="Role-based strategy preferences"
    )
    
    # Hypothesis Generation Settings
    hypothesis_max_count: int = Field(
        default=5,
        description="Maximum hypotheses to generate per investigation"
    )
    
    hypothesis_confidence_threshold: float = Field(
        default=0.3,
        description="Minimum confidence for hypothesis inclusion"
    )
    
    # Performance Settings
    concurrent_strategy_limit: int = Field(
        default=100,
        description="Maximum concurrent strategy planning operations"
    )
    
    cache_ttl_seconds: int = Field(
        default=3600,
        description="Strategy plan cache time-to-live"
    )
    
    # Organizational Learning Settings
    learning_enabled: bool = Field(
        default=True,
        description="Enable organizational learning from investigations"
    )
    
    success_rate_confidence_interval: float = Field(
        default=0.95,
        description="Confidence interval for success rate calculations"
    )
    
    # Business Domain Configuration
    supported_domains: List[str] = Field(
        default=[
            "production", "quality", "supply_chain", "cost", "assets",
            "safety", "customer", "planning", "human_resources", "sales",
            "finance", "marketing", "operations", "strategic"
        ],
        description="Supported business intelligence domains"
    )
    
    # Integration Settings
    mcp_client_timeout: int = Field(
        default=15,
        description="MCP client timeout in seconds"
    )
    
    pattern_library_enabled: bool = Field(
        default=True,
        description="Enable pattern library integration"
    )
    
    # Logging Configuration
    log_level: str = Field(
        default="INFO",
        description="Logging level for intelligence module"
    )
    
    detailed_logging: bool = Field(
        default=False,
        description="Enable detailed performance logging"
    )
    
    # Health Monitoring Settings
    health_check_enabled: bool = Field(
        default=True,
        description="Enable health monitoring"
    )
    
    health_check_interval: int = Field(
        default=60,
        description="Health check interval in seconds"
    )
    
    @validator('domain_confidence_threshold')
    def validate_confidence_threshold(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('Confidence threshold must be between 0.0 and 1.0')
        return v
    
    @validator('pattern_similarity_threshold')
    def validate_similarity_threshold(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('Similarity threshold must be between 0.0 and 1.0')
        return v
    
    @validator('complexity_scoring_weights')
    def validate_complexity_weights(cls, v):
        total_weight = sum(v.values())
        if not 0.95 <= total_weight <= 1.05:  # Allow small floating point variance
            raise ValueError(f'Complexity weights must sum to 1.0, got {total_weight}')
        return v
    
    @validator('log_level')
    def validate_log_level(cls, v):
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f'Log level must be one of {valid_levels}')
        return v.upper()


# Create singleton instance
settings = IntelligenceSettings()