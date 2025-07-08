"""
BGE-M3 Embedding Model Module
Provides production-ready text embeddings for vector search.
"""

from .runner import (
    EmbeddingModelManager,
    get_embedding_manager,
    embed_text,
    embed_texts,
    embed_text_async,
    embed_texts_async
)
from .config import EmbeddingSettings, settings

__all__ = [
    "EmbeddingModelManager",
    "get_embedding_manager",
    "embed_text",
    "embed_texts", 
    "embed_text_async",
    "embed_texts_async",
    "EmbeddingSettings",
    "settings"
]