"""
BGE-M3 Embedding Model Configuration
Self-contained configuration using pydantic-settings.
"""

from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class EmbeddingSettings(BaseSettings):
    """BGE-M3 embedding model configuration."""
    
    model_config = SettingsConfigDict(
        env_file=Path(__file__).parent / "settings.env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Model configuration
    model_name: str = Field(
        default="BAAI/bge-m3",
        description="Hugging Face model identifier"
    )
    model_cache_dir: str = Field(
        default="/tmp/bge_models",
        description="Directory to cache downloaded models"
    )
    embedding_dimension: int = Field(
        default=1024,
        description="Output embedding dimension"
    )
    max_sequence_length: int = Field(
        default=8192,
        description="Maximum input sequence length"
    )
    
    # Processing configuration
    batch_size: int = Field(
        default=32,
        description="Batch size for embedding generation"
    )
    device: str = Field(
        default="cpu",
        description="Device to run model on (cpu/cuda/mps)"
    )
    use_fp16: bool = Field(
        default=False,
        description="Use half precision for faster inference"
    )
    
    # Feature flags
    return_dense: bool = Field(
        default=True,
        description="Return dense embeddings"
    )
    return_sparse: bool = Field(
        default=False,
        description="Return sparse embeddings"
    )
    return_colbert: bool = Field(
        default=False,
        description="Return ColBERT vectors"
    )
    
    # Performance settings
    num_workers: int = Field(
        default=4,
        description="Number of data loading workers"
    )
    prefetch_factor: int = Field(
        default=2,
        description="Number of batches to prefetch"
    )


# Create singleton instance
settings = EmbeddingSettings()