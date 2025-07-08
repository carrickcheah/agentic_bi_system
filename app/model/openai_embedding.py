"""
OpenAI Embedding Model Implementation.
Self-contained module for generating embeddings using OpenAI's text-embedding models.
"""

import asyncio
from typing import List, Optional, Union
import numpy as np
from openai import AsyncOpenAI, OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import settings
from .model_logging import logger


class OpenAIEmbeddingModel:
    """OpenAI embedding model with automatic retry and error handling."""
    
    def __init__(self):
        """Initialize OpenAI embedding model."""
        self.api_key = settings.openai_api_key
        self.model = settings.embedding_model
        
        if not self.api_key:
            raise ValueError("OpenAI API key not configured in settings.env")
        
        # Initialize both sync and async clients
        self.client = OpenAI(api_key=self.api_key)
        self.async_client = AsyncOpenAI(api_key=self.api_key)
        
        # Model dimensions mapping
        self.model_dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536
        }
        
        self.dimension = self.model_dimensions.get(self.model, 1536)
        logger.info(f"Initialized OpenAI embedding model: {self.model} with {self.dimension} dimensions")
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def embed_text(self, text: Union[str, List[str]], dimensions: Optional[int] = None) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings for text synchronously.
        
        Args:
            text: Single text string or list of texts
            dimensions: Optional dimension size (only for text-embedding-3-* models)
            
        Returns:
            Single embedding vector or list of embedding vectors
        """
        is_single = isinstance(text, str)
        texts = [text] if is_single else text
        
        try:
            # Prepare request parameters
            params = {
                "model": self.model,
                "input": texts
            }
            
            # Add dimensions parameter if specified and model supports it
            if dimensions and self.model.startswith("text-embedding-3-"):
                params["dimensions"] = dimensions
                logger.debug(f"Using custom dimensions: {dimensions}")
            
            response = self.client.embeddings.create(**params)
            
            embeddings = [data.embedding for data in response.data]
            logger.debug(f"Generated embeddings for {len(texts)} text(s)")
            
            return embeddings[0] if is_single else embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def embed_text_async(self, text: Union[str, List[str]], dimensions: Optional[int] = None) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings for text asynchronously.
        
        Args:
            text: Single text string or list of texts
            dimensions: Optional dimension size (only for text-embedding-3-* models)
            
        Returns:
            Single embedding vector or list of embedding vectors
        """
        is_single = isinstance(text, str)
        texts = [text] if is_single else text
        
        try:
            # Prepare request parameters
            params = {
                "model": self.model,
                "input": texts
            }
            
            # Add dimensions parameter if specified and model supports it
            if dimensions and self.model.startswith("text-embedding-3-"):
                params["dimensions"] = dimensions
                logger.debug(f"Using custom dimensions: {dimensions}")
            
            response = await self.async_client.embeddings.create(**params)
            
            embeddings = [data.embedding for data in response.data]
            logger.debug(f"Generated embeddings for {len(texts)} text(s)")
            
            return embeddings[0] if is_single else embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    def embed_batch(self, texts: List[str], batch_size: int = 100, dimensions: Optional[int] = None) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts with automatic batching.
        
        Args:
            texts: List of texts to embed
            batch_size: Maximum texts per API call (OpenAI limit is typically 2048)
            dimensions: Optional dimension size
            
        Returns:
            List of embedding vectors
        """
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = self.embed_text(batch, dimensions)
            all_embeddings.extend(embeddings)
            logger.debug(f"Processed batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
        
        return all_embeddings
    
    async def embed_batch_async(self, texts: List[str], batch_size: int = 100, dimensions: Optional[int] = None) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts asynchronously with automatic batching.
        
        Args:
            texts: List of texts to embed
            batch_size: Maximum texts per API call
            dimensions: Optional dimension size
            
        Returns:
            List of embedding vectors
        """
        tasks = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            task = self.embed_text_async(batch, dimensions)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        # Flatten results
        all_embeddings = []
        for batch_embeddings in results:
            all_embeddings.extend(batch_embeddings)
        
        logger.debug(f"Processed {len(texts)} texts in {len(tasks)} batches")
        return all_embeddings
    
    def compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score between -1 and 1
        """
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # Compute cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return float(similarity)
    
    async def health_check(self) -> bool:
        """Check if the embedding service is healthy."""
        try:
            # Try to embed a simple test text
            test_text = "health check"
            embedding = await self.embed_text_async(test_text)
            
            # Verify embedding has expected dimensions
            if len(embedding) > 0:
                logger.debug("OpenAI embedding service health check passed")
                return True
            return False
            
        except Exception as e:
            logger.error(f"OpenAI embedding service health check failed: {e}")
            return False
    
    def get_model_info(self) -> dict:
        """Get information about the current embedding model."""
        return {
            "model": self.model,
            "dimensions": self.dimension,
            "supports_custom_dimensions": self.model.startswith("text-embedding-3-"),
            "api_key_configured": bool(self.api_key)
        }


# Convenience function for direct import
def create_embedding_model() -> OpenAIEmbeddingModel:
    """Create and return an OpenAI embedding model instance."""
    return OpenAIEmbeddingModel()


# Example usage and testing
if __name__ == "__main__":
    async def test_embeddings():
        """Test the embedding functionality."""
        model = OpenAIEmbeddingModel()
        
        # Test single text
        text = "This is a test sentence for embedding generation."
        embedding = await model.embed_text_async(text)
        print(f"Single text embedding dimension: {len(embedding)}")
        
        # Test batch
        texts = [
            "First test sentence.",
            "Second test sentence.",
            "Third test sentence."
        ]
        embeddings = await model.embed_batch_async(texts)
        print(f"Batch embeddings count: {len(embeddings)}")
        
        # Test similarity
        similarity = model.compute_similarity(embeddings[0], embeddings[1])
        print(f"Similarity between first two texts: {similarity:.4f}")
        
        # Test health check
        is_healthy = await model.health_check()
        print(f"Health check: {'PASS' if is_healthy else 'FAIL'}")
        
        # Model info
        info = model.get_model_info()
        print(f"Model info: {info}")
    
    # Run test
    asyncio.run(test_embeddings())