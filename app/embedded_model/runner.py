"""
BGE-M3 Embedding Model Runner
Production-ready embedding generation with BGE-M3 model.
"""

import os
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import logging

from .config import settings
from .model_logging import logger

# Try to import FlagEmbedding, fallback to sentence-transformers
try:
    from FlagEmbedding import BGEM3FlagModel
    USE_FLAG_EMBEDDING = True
except ImportError:
    logger.warning("FlagEmbedding not installed, using sentence-transformers fallback")
    from sentence_transformers import SentenceTransformer
    USE_FLAG_EMBEDDING = False


class EmbeddingModelManager:
    """
    Production-ready BGE-M3 embedding model manager.
    
    Features:
    - Automatic model download and caching
    - Batch processing for efficiency
    - Device optimization (CPU/GPU/MPS)
    - Multiple embedding types (dense/sparse/colbert)
    - Error handling and fallback
    """
    
    def __init__(self):
        """Initialize embedding model."""
        self.model = None
        self.device = self._get_device()
        
        # Create cache directory
        Path(settings.model_cache_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize model
        self._initialize_model()
        
        logger.info(f"EmbeddingModelManager initialized on {self.device}")
    
    def _get_device(self) -> str:
        """Determine best available device."""
        if settings.device != "auto":
            return settings.device
        
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def _initialize_model(self):
        """Initialize BGE-M3 model with appropriate backend."""
        try:
            if USE_FLAG_EMBEDDING:
                # Use official FlagEmbedding implementation
                self.model = BGEM3FlagModel(
                    settings.model_name,
                    cache_dir=settings.model_cache_dir,
                    use_fp16=settings.use_fp16
                )
                logger.info(f"Loaded BGE-M3 model using FlagEmbedding from {settings.model_name}")
            else:
                # Fallback to sentence-transformers
                self.model = SentenceTransformer(
                    settings.model_name,
                    cache_folder=settings.model_cache_dir,
                    device=self.device
                )
                logger.info(f"Loaded BGE-M3 model using sentence-transformers from {settings.model_name}")
                
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise RuntimeError(f"Could not initialize embedding model: {e}")
    
    def generate_embeddings(
        self,
        texts: Union[str, List[str]],
        batch_size: Optional[int] = None,
        show_progress: bool = False
    ) -> Union[np.ndarray, Dict[str, Any]]:
        """
        Generate embeddings for text(s).
        
        Args:
            texts: Single text or list of texts
            batch_size: Override default batch size
            show_progress: Show progress bar for large batches
            
        Returns:
            - If using FlagEmbedding: Dict with dense/sparse/colbert embeddings
            - If using sentence-transformers: numpy array of dense embeddings
        """
        # Ensure texts is a list
        if isinstance(texts, str):
            texts = [texts]
            single_text = True
        else:
            single_text = False
        
        # Validate inputs
        if not texts:
            raise ValueError("No texts provided for embedding")
        
        # Truncate texts to max length
        texts = [text[:settings.max_sequence_length] for text in texts]
        
        batch_size = batch_size or settings.batch_size
        
        try:
            if USE_FLAG_EMBEDDING:
                # Use FlagEmbedding with multiple embedding types
                embeddings = self.model.encode(
                    texts,
                    batch_size=batch_size,
                    return_dense=settings.return_dense,
                    return_sparse=settings.return_sparse,
                    return_colbert_vecs=settings.return_colbert
                )
                
                # Process output
                if settings.return_dense and not settings.return_sparse and not settings.return_colbert:
                    # Only dense embeddings requested
                    result = embeddings['dense_vecs']
                else:
                    # Multiple types requested
                    result = embeddings
                    
            else:
                # Use sentence-transformers (dense only)
                embeddings = self.model.encode(
                    texts,
                    batch_size=batch_size,
                    show_progress_bar=show_progress,
                    convert_to_numpy=True
                )
                result = embeddings
            
            # Return single embedding if single text input
            if single_text and isinstance(result, np.ndarray):
                return result[0]
                
            return result
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise RuntimeError(f"Failed to generate embeddings: {e}")
    
    async def generate_embeddings_async(
        self,
        texts: Union[str, List[str]],
        batch_size: Optional[int] = None
    ) -> Union[np.ndarray, Dict[str, Any]]:
        """
        Async wrapper for embedding generation.
        
        Note: The underlying models are synchronous, but this provides
        an async interface for consistency with other services.
        """
        import asyncio
        
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.generate_embeddings,
            texts,
            batch_size
        )
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings."""
        return settings.embedding_dimension
    
    def health_check(self) -> Dict[str, Any]:
        """Check model health and status."""
        try:
            # Test with simple embedding
            test_embedding = self.generate_embeddings("test")
            
            health_status = {
                "status": "healthy",
                "model": settings.model_name,
                "device": self.device,
                "embedding_dim": self.get_embedding_dimension(),
                "backend": "FlagEmbedding" if USE_FLAG_EMBEDDING else "sentence-transformers",
                "features": {
                    "dense": settings.return_dense,
                    "sparse": settings.return_sparse,
                    "colbert": settings.return_colbert
                }
            }
            
            if isinstance(test_embedding, np.ndarray):
                health_status["test_embedding_shape"] = test_embedding.shape
            
            return health_status
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "model": settings.model_name
            }


# Singleton instance
_embedding_manager = None

def get_embedding_manager() -> EmbeddingModelManager:
    """Get or create singleton embedding manager."""
    global _embedding_manager
    if _embedding_manager is None:
        _embedding_manager = EmbeddingModelManager()
    return _embedding_manager


# Convenience functions
def embed_text(text: str) -> np.ndarray:
    """Generate embedding for single text."""
    manager = get_embedding_manager()
    return manager.generate_embeddings(text)


def embed_texts(texts: List[str], batch_size: Optional[int] = None) -> np.ndarray:
    """Generate embeddings for multiple texts."""
    manager = get_embedding_manager()
    return manager.generate_embeddings(texts, batch_size)


async def embed_text_async(text: str) -> np.ndarray:
    """Async generate embedding for single text."""
    manager = get_embedding_manager()
    return await manager.generate_embeddings_async(text)


async def embed_texts_async(texts: List[str], batch_size: Optional[int] = None) -> np.ndarray:
    """Async generate embeddings for multiple texts."""
    manager = get_embedding_manager()
    return await manager.generate_embeddings_async(texts, batch_size)


# Standalone testing
if __name__ == "__main__":
    import asyncio
    
    async def test():
        print("Testing BGE-M3 Embedding Model\n" + "="*50)
        
        # Initialize
        manager = get_embedding_manager()
        
        # Health check
        health = manager.health_check()
        print(f"\nHealth Status: {health}")
        
        # Test single text
        text = "What were last month's sales by region?"
        embedding = embed_text(text)
        print(f"\nSingle text embedding shape: {embedding.shape if isinstance(embedding, np.ndarray) else type(embedding)}")
        
        # Test batch
        texts = [
            "Show me Q4 revenue trends",
            "Customer retention analysis",
            "Production efficiency metrics"
        ]
        embeddings = embed_texts(texts)
        print(f"\nBatch embeddings shape: {embeddings.shape if isinstance(embeddings, np.ndarray) else type(embeddings)}")
        
        # Test async
        async_embedding = await embed_text_async("Test async embedding")
        print(f"\nAsync embedding shape: {async_embedding.shape if isinstance(async_embedding, np.ndarray) else type(async_embedding)}")
        
        print("\n All tests passed!")
    
    asyncio.run(test())