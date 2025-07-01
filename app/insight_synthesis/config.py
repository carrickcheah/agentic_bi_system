"""
Insight Synthesis Module Configuration - Self-Contained Architecture
Implements Phase 5: Insight Synthesis configuration with pydantic-settings.
NO hardcoded business defaults - all values from environment.
"""

from pathlib import Path
from typing import List, Dict, Optional
from pydantic import Field, validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class InsightSynthesisSettings(BaseSettings):
    """Insight Synthesis Module configuration - loads from local settings.env"""
    
    model_config = SettingsConfigDict(
        env_file=Path(__file__).parent / "settings.env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Core Synthesis Configuration
    synthesis_timeout: int = Field(
        default=45, 
        description="Maximum time for insight synthesis in seconds"
    )
    
    # Insight Generation Settings  
    insight_confidence_threshold: float = Field(
        default=0.7,
        description="Minimum confidence for insight inclusion"
    )
    
    max_insights_per_investigation: int = Field(
        default=10,
        description="Maximum insights to generate per investigation"
    )
    
    # Strategic Analysis Settings
    strategic_depth_levels: Dict[str, float] = Field(
        default={
            "operational": 0.4,
            "tactical": 0.6,
            "strategic": 0.8,
            "transformational": 1.0
        },
        description="Strategic depth level thresholds"
    )
    
    # Role-Specific Formatting Settings
    role_formatting_enabled: bool = Field(
        default=True,
        description="Enable role-specific insight formatting"
    )
    
    role_detail_levels: Dict[str, str] = Field(
        default={
            "analyst": "detailed",
            "manager": "summary",
            "engineer": "technical",
            "executive": "strategic",
            "specialist": "detailed"
        },
        description="Detail level preferences by role"
    )
    
    # Organizational Learning Settings
    learning_capture_enabled: bool = Field(
        default=True,
        description="Enable organizational learning capture"
    )
    
    pattern_extraction_threshold: float = Field(
        default=0.8,
        description="Minimum threshold for pattern extraction"
    )
    
    success_metric_tracking: bool = Field(
        default=True,
        description="Enable success metric tracking"
    )
    
    # Recommendation Generation Settings
    recommendation_max_count: int = Field(
        default=5,
        description="Maximum recommendations per insight"
    )
    
    recommendation_priority_weights: Dict[str, float] = Field(
        default={
            "business_impact": 0.4,
            "implementation_feasibility": 0.3,
            "resource_requirement": 0.2,
            "risk_level": 0.1
        },
        description="Recommendation priority scoring weights"
    )
    
    # Performance Settings
    concurrent_synthesis_limit: int = Field(
        default=50,
        description="Maximum concurrent synthesis operations"
    )
    
    synthesis_cache_ttl: int = Field(
        default=1800,  # 30 minutes
        description="Synthesis result cache time-to-live in seconds"
    )
    
    # Quality Assurance Settings
    insight_validation_enabled: bool = Field(
        default=True,
        description="Enable insight quality validation"
    )
    
    business_relevance_threshold: float = Field(
        default=0.75,
        description="Minimum business relevance score for insights"
    )
    
    actionability_threshold: float = Field(
        default=0.6,
        description="Minimum actionability score for recommendations"
    )
    
    # Integration Settings
    investigation_results_timeout: int = Field(
        default=30,
        description="Timeout for receiving investigation results in seconds"
    )
    
    external_context_enabled: bool = Field(
        default=True,
        description="Enable external context integration"
    )
    
    # Output Format Settings
    supported_output_formats: List[str] = Field(
        default=[
            "executive_summary", "detailed_report", "presentation", 
            "dashboard", "action_plan", "technical_brief"
        ],
        description="Supported output format types"
    )
    
    default_output_format: str = Field(
        default="detailed_report",
        description="Default output format"
    )
    
    # Collaboration Settings
    real_time_sharing_enabled: bool = Field(
        default=True,
        description="Enable real-time insight sharing"
    )
    
    stakeholder_notification_enabled: bool = Field(
        default=True,
        description="Enable stakeholder notifications"
    )
    
    # Memory and Context Settings
    context_window_size: int = Field(
        default=5,
        description="Number of previous investigations to consider for context"
    )
    
    cross_investigation_correlation: bool = Field(
        default=True,
        description="Enable correlation across multiple investigations"
    )
    
    # Logging Configuration
    log_level: str = Field(
        default="INFO",
        description="Logging level for insight synthesis module"
    )
    
    detailed_logging: bool = Field(
        default=False,
        description="Enable detailed synthesis logging"
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
    
    # Business Impact Calculation Settings
    impact_calculation_weights: Dict[str, float] = Field(
        default={
            "financial_impact": 0.4,
            "operational_efficiency": 0.3,
            "strategic_alignment": 0.2,
            "risk_mitigation": 0.1
        },
        description="Business impact calculation weights"
    )
    
    @validator('insight_confidence_threshold')
    def validate_insight_confidence(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('Insight confidence threshold must be between 0.0 and 1.0')
        return v
    
    @validator('pattern_extraction_threshold')
    def validate_pattern_threshold(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('Pattern extraction threshold must be between 0.0 and 1.0')
        return v
    
    @validator('recommendation_priority_weights')
    def validate_recommendation_weights(cls, v):
        total_weight = sum(v.values())
        if not 0.95 <= total_weight <= 1.05:  # Allow small floating point variance
            raise ValueError(f'Recommendation priority weights must sum to 1.0, got {total_weight}')
        return v
    
    @validator('impact_calculation_weights')
    def validate_impact_weights(cls, v):
        total_weight = sum(v.values())
        if not 0.95 <= total_weight <= 1.05:  # Allow small floating point variance
            raise ValueError(f'Impact calculation weights must sum to 1.0, got {total_weight}')
        return v
    
    @validator('log_level')
    def validate_log_level(cls, v):
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f'Log level must be one of {valid_levels}')
        return v.upper()
    
    @validator('default_output_format')
    def validate_output_format(cls, v, values):
        if 'supported_output_formats' in values:
            if v not in values['supported_output_formats']:
                raise ValueError(f'Default output format must be in supported formats')
        return v


# Create singleton instance
settings = InsightSynthesisSettings()