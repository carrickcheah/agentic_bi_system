"""
Embedding generation component using BGE-M3 model.
Handles all embedding operations for SQL queries.
"""

import time
from typing import List, Union
import numpy as np

from sentence_transformers import SentenceTransformer

try:
    from .config import settings
    from .lance_logging import get_logger, log_operation, log_error, log_performance
except ImportError:
    # For standalone execution
    from config import settings
    from lance_logging import get_logger, log_operation, log_error, log_performance


class EmbeddingGenerator:
    """BGE-M3 embedding generator for SQL queries."""
    
    def __init__(self):
        self.model = None
        self.logger = get_logger("embedding")
        self._initialized = False
        self.model_name = settings.embedding_model
    
    async def initialize(self):
        """Initialize the embedding model."""
        if self._initialized:
            self.logger.warning("Embedding generator already initialized")
            return
        
        try:
            start_time = time.time()
            
            log_operation("Loading embedding model", {"model": self.model_name})
            
            # Load the sentence transformer model
            self.model = SentenceTransformer(self.model_name)
            
            # Warm up the model with a test embedding
            _ = self.model.encode("SELECT * FROM test", convert_to_numpy=True)
            
            self._initialized = True
            
            duration_ms = (time.time() - start_time) * 1000
            log_performance(f"Model loading ({self.model_name})", duration_ms)
            
        except Exception as e:
            log_error("Model initialization", e)
            raise RuntimeError(f"Failed to initialize embedding model: {e}")
    
    async def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text (SQL query)
        
        Returns:
            Numpy array of embeddings
        """
        if not self._initialized:
            raise RuntimeError("Embedding generator not initialized")
        
        try:
            start_time = time.time()
            
            # Generate embedding
            embedding = self.model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=True  # BGE-M3 works better with normalized embeddings
            )
            
            duration_ms = (time.time() - start_time) * 1000
            if settings.enable_detailed_logging:
                log_performance(f"Generate embedding ({len(text)} chars)", duration_ms)
            
            return embedding
            
        except Exception as e:
            log_error("Generate embedding", e)
            raise
    
    async def generate_batch_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for multiple texts in batch.
        
        Args:
            texts: List of input texts
        
        Returns:
            Numpy array of embeddings (n_texts, embedding_dim)
        """
        if not self._initialized:
            raise RuntimeError("Embedding generator not initialized")
        
        try:
            start_time = time.time()
            
            # Generate embeddings in batch
            embeddings = self.model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                batch_size=32,  # Optimize for memory usage
                show_progress_bar=False
            )
            
            duration_ms = (time.time() - start_time) * 1000
            log_performance(f"Generate batch embeddings ({len(texts)} texts)", duration_ms)
            
            return embeddings
            
        except Exception as e:
            log_error("Generate batch embeddings", e)
            raise
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by the model."""
        if not self._initialized:
            raise RuntimeError("Embedding generator not initialized")
        
        return self.model.get_sentence_embedding_dimension()
    
    def is_healthy(self) -> bool:
        """Check if the embedding generator is healthy."""
        if not self._initialized:
            return False
        
        try:
            # Test with a simple query
            test_embedding = self.model.encode("SELECT 1", convert_to_numpy=True)
            return test_embedding is not None and len(test_embedding) > 0
        except:
            return False
    
    async def cleanup(self):
        """Clean up resources."""
        # SentenceTransformer doesn't need explicit cleanup
        self._initialized = False
        self.logger.info("Embedding generator cleaned up")
    
    def preprocess_sql(self, sql_query: str) -> str:
        """
        Preprocess SQL query for better embedding generation.
        
        Args:
            sql_query: Raw SQL query
        
        Returns:
            Preprocessed query
        """
        # Remove excessive whitespace
        processed = " ".join(sql_query.split())
        
        # Keep SQL structure but normalize formatting
        # BGE-M3 works better with natural structure preserved
        
        return processed