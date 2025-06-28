"""
Self-contained configuration using pydantic-settings.
Loads ALL values from local settings.env - NO hardcoded business defaults.
"""

import os
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LanceDBSettings(BaseSettings):
    """LanceDB module configuration - loads from local settings.env"""
    
    model_config = SettingsConfigDict(
        env_file=Path(__file__).parent / "settings.env",  # CRITICAL: Relative to module
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Required business values - NO defaults (force explicit configuration)
    lancedb_path: str = Field(description="Path to LanceDB data directory")
    embedding_model: str = Field(description="Sentence transformer model name")
    similarity_threshold: float = Field(description="Minimum similarity for cache hits")
    
    # Zeabur deployment support - NO defaults
    zeabur_mount_path: str = Field(description="Zeabur volume mount path")
    
    # Optional operational values - reasonable defaults only
    enable_query_cache: bool = Field(default=True, description="Enable query result caching")
    cache_ttl_hours: int = Field(default=24, description="Cache time-to-live in hours")
    max_cache_size: int = Field(default=100000, description="Maximum cached queries")
    
    # Index configuration - operational defaults
    enable_hnsw_index: bool = Field(default=True, description="Enable HNSW indexing")
    index_nprobe: int = Field(default=20, description="Number of probes for index search")
    index_refine_factor: int = Field(default=10, description="Refine factor for search")
    
    # Logging configuration
    log_level: str = Field(default="INFO", description="Logging level")
    enable_detailed_logging: bool = Field(default=False, description="Enable detailed operation logs")
    
    @property
    def data_path(self) -> str:
        """Get appropriate data path based on environment"""
        # Check if running on Zeabur
        if os.getenv("ZEABUR_ENVIRONMENT"):
            # Check if volume is mounted and writable
            if os.path.exists(self.zeabur_mount_path) and os.access(self.zeabur_mount_path, os.W_OK):
                return self.zeabur_mount_path
            else:
                raise RuntimeError(f"Zeabur volume not mounted or not writable at {self.zeabur_mount_path}")
        
        # Local development path
        return self.lancedb_path
    
    def validate_paths(self):
        """Validate that required paths are accessible"""
        data_dir = Path(self.data_path)
        
        # Create directory if it doesn't exist (local development)
        if not os.getenv("ZEABUR_ENVIRONMENT"):
            data_dir.mkdir(parents=True, exist_ok=True)
        
        # Verify path is writable
        if not os.access(data_dir.parent, os.W_OK):
            raise PermissionError(f"Cannot write to directory: {data_dir.parent}")


# Create singleton instance
try:
    settings = LanceDBSettings()
except Exception as e:
    # Provide helpful error message if settings.env is missing
    import sys
    print(f"ERROR: Failed to load LanceDB configuration: {e}", file=sys.stderr)
    print("Please ensure settings.env exists in the lance_db directory with required values.", file=sys.stderr)
    raise