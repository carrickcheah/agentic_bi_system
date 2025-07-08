"""
Self-contained configuration for Phase 4 Investigation module.
Loads ALL values from local settings.env - NO hardcoded business defaults.
"""

from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import Field, validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class InvestigationSettings(BaseSettings):
    """Phase 4 Investigation configuration - loads from local settings.env"""
    
    model_config = SettingsConfigDict(
        env_file=Path(__file__).parent / "settings.env",  # CRITICAL: Relative to module
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Investigation Parameters - Core investigation behavior
    investigation_timeout_minutes: int = Field(description="Maximum duration for complete investigation")
    max_concurrent_queries: int = Field(description="Maximum parallel database queries")
    confidence_threshold: float = Field(description="Minimum confidence for accepting findings")
    max_hypotheses: int = Field(description="Maximum number of hypotheses to generate")
    cross_validation_sources: int = Field(description="Minimum sources required for cross-validation")
    
    # MCP Service Integration - Database service endpoints
    mariadb_mcp_endpoint: str = Field(description="MariaDB MCP service endpoint")
    postgres_mcp_endpoint: str = Field(description="PostgreSQL MCP service endpoint") 
    qdrant_endpoint: str = Field(description="Qdrant service endpoint")
    graphrag_mcp_endpoint: str = Field(description="GraphRAG MCP service endpoint")
    
    # Performance Configuration - Operational parameters
    query_timeout_seconds: int = Field(default=30, description="Individual query timeout")
    step_timeout_minutes: int = Field(default=10, description="Individual step timeout")
    parallel_processing_enabled: bool = Field(default=True, description="Enable parallel processing")
    max_retry_attempts: int = Field(default=3, description="Maximum retry attempts for failed operations")
    
    # Validation Configuration - Quality assurance settings
    enable_cross_validation: bool = Field(default=True, description="Enable cross-source validation")
    validation_confidence_threshold: float = Field(default=0.7, description="Minimum confidence for validation")
    enable_statistical_validation: bool = Field(default=True, description="Enable statistical validation")
    min_data_points_for_analysis: int = Field(default=10, description="Minimum data points for statistical analysis")
    
    # State Management - Investigation state persistence
    enable_state_persistence: bool = Field(default=True, description="Enable investigation state persistence")
    checkpoint_interval_steps: int = Field(default=2, description="Steps between investigation checkpoints")
    enable_recovery_mode: bool = Field(default=True, description="Enable investigation recovery from failures")
    max_investigation_history: int = Field(default=100, description="Maximum investigation history to retain")
    
    # Logging Configuration - Investigation audit trail
    log_level: str = Field(default="INFO", description="Logging level for investigation module")
    enable_detailed_logging: bool = Field(default=True, description="Enable detailed investigation logging")
    enable_step_logging: bool = Field(default=True, description="Enable per-step logging")
    enable_performance_logging: bool = Field(default=True, description="Enable performance metrics logging")
    
    @validator('confidence_threshold', 'validation_confidence_threshold')
    def validate_confidence_range(cls, v):
        """Ensure confidence thresholds are between 0.0 and 1.0."""
        if not 0.0 <= v <= 1.0:
            raise ValueError('Confidence thresholds must be between 0.0 and 1.0')
        return v
    
    @validator('log_level')
    def validate_log_level(cls, v):
        """Ensure log level is valid."""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f'Log level must be one of: {valid_levels}')
        return v.upper()
    
    def get_mcp_endpoints(self) -> Dict[str, str]:
        """Get all MCP service endpoints as a dictionary."""
        return {
            'mariadb': self.mariadb_mcp_endpoint,
            'postgresql': self.postgres_mcp_endpoint,
            'qdrant': self.qdrant_endpoint,
            'graphrag': self.graphrag_mcp_endpoint
        }
    
    def get_investigation_limits(self) -> Dict[str, Any]:
        """Get investigation operational limits."""
        return {
            'timeout_minutes': self.investigation_timeout_minutes,
            'max_concurrent_queries': self.max_concurrent_queries,
            'max_hypotheses': self.max_hypotheses,
            'cross_validation_sources': self.cross_validation_sources,
            'query_timeout_seconds': self.query_timeout_seconds,
            'step_timeout_minutes': self.step_timeout_minutes,
            'max_retry_attempts': self.max_retry_attempts
        }
    
    def get_validation_config(self) -> Dict[str, Any]:
        """Get validation and quality assurance configuration."""
        return {
            'enable_cross_validation': self.enable_cross_validation,
            'confidence_threshold': self.confidence_threshold,
            'validation_confidence_threshold': self.validation_confidence_threshold,
            'enable_statistical_validation': self.enable_statistical_validation,
            'min_data_points_for_analysis': self.min_data_points_for_analysis
        }
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return {
            'log_level': self.log_level,
            'enable_detailed_logging': self.enable_detailed_logging,
            'enable_step_logging': self.enable_step_logging,
            'enable_performance_logging': self.enable_performance_logging
        }


class InvestigationRuntimeConfig:
    """Runtime configuration for active investigations."""
    
    def __init__(self, settings: InvestigationSettings):
        self.settings = settings
        self._active_investigations: Dict[str, Dict[str, Any]] = {}
        self._performance_metrics: Dict[str, Any] = {}
    
    def register_investigation(self, investigation_id: str, context: Dict[str, Any]) -> None:
        """Register a new active investigation."""
        self._active_investigations[investigation_id] = {
            'context': context,
            'start_time': 'current_timestamp',
            'current_step': 0,
            'completed_steps': [],
            'status': 'active'
        }
    
    def update_investigation_progress(self, investigation_id: str, step: int, status: str) -> None:
        """Update investigation progress."""
        if investigation_id in self._active_investigations:
            self._active_investigations[investigation_id]['current_step'] = step
            self._active_investigations[investigation_id]['status'] = status
    
    def get_investigation_status(self, investigation_id: str) -> Optional[Dict[str, Any]]:
        """Get current investigation status."""
        return self._active_investigations.get(investigation_id)
    
    def cleanup_completed_investigations(self) -> None:
        """Remove completed investigations beyond history limit."""
        completed = [
            inv_id for inv_id, inv_data in self._active_investigations.items()
            if inv_data.get('status') in ['completed', 'failed']
        ]
        
        if len(completed) > self.settings.max_investigation_history:
            # Remove oldest completed investigations
            for inv_id in completed[:-self.settings.max_investigation_history]:
                del self._active_investigations[inv_id]
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get aggregated performance metrics."""
        return self._performance_metrics.copy()


# Create singleton instance
settings = InvestigationSettings()

# Create runtime configuration
runtime_config = InvestigationRuntimeConfig(settings)