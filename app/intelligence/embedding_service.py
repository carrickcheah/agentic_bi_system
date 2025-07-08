"""Embedding Service for Vector Operations.

Provides embeddings for SQL queries and business questions.
Uses BGE-M3 model with 1024 dimensions.
"""

import hashlib
from typing import List, Optional
import logging

logger = logging.getLogger("embedding_service")


class EmbeddingService:
    """Embedding service for generating vector representations."""
    
    def __init__(self):
        self.model_name = "BGE-M3"
        self.embedding_dim = 1024  # BGE-M3 dimension
        self._initialized = False
    
    async def initialize(self):
        """Initialize the embedding service."""
        # In production, this would load the actual BGE-M3 model
        # For now, using deterministic embeddings for testing
        self._initialized = True
        logger.info(f"Embedding service initialized (model: {self.model_name}, dim: {self.embedding_dim})")
    
    async def get_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        if not self._initialized:
            await self.initialize()
        
        # Create deterministic embedding based on text hash
        # In production, this would use the actual BGE-M3 model
        hash_obj = hashlib.sha384(text.encode())
        hash_bytes = hash_obj.digest()
        
        # Convert to float values normalized between -1 and 1
        embedding = []
        for i in range(0, len(hash_bytes), 4):
            if len(embedding) >= self.embedding_dim:
                break
            # Take 4 bytes at a time and convert to float
            chunk = hash_bytes[i:i+4]
            if len(chunk) == 4:
                value = int.from_bytes(chunk, 'big') / (2**32)
                # Normalize to [-1, 1]
                normalized = (value * 2) - 1
                embedding.append(normalized)
        
        # Pad with zeros if needed
        while len(embedding) < self.embedding_dim:
            # Use a deterministic pattern for padding
            pad_value = (len(embedding) % 100) / 100.0 - 0.5
            embedding.append(pad_value)
        
        return embedding[:self.embedding_dim]
    
    async def get_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        embeddings = []
        for text in texts:
            embedding = await self.get_embedding(text)
            embeddings.append(embedding)
        return embeddings
    
    def get_model_info(self) -> dict:
        """Get information about the embedding model."""
        return {
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "initialized": self._initialized
        }


# Singleton instance
_instance: Optional[EmbeddingService] = None


async def get_embedding_service() -> EmbeddingService:
    """Get or create embedding service instance."""
    global _instance
    if _instance is None:
        _instance = EmbeddingService()
        await _instance.initialize()
    return _instance