"""
OpenAI embedding model implementation.
Provides text embedding functionality for vector search.
"""

from typing import List, Optional
from openai import AsyncOpenAI

from .config import settings
from .model_logging import logger


class OpenAIEmbedding:
    """OpenAI embedding model interface."""
    
    def __init__(self):
        if not settings.openai_api_key:
            raise ValueError("OpenAI API key not configured for embeddings")
            
        self.client = AsyncOpenAI(
            api_key=settings.openai_api_key,
            timeout=settings.request_timeout
        )
        self.model = settings.embedding_model
        
        logger.info(f"OpenAI embedding initialized with model: {self.model}")
    
    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            List of float values representing the embedding
        """
        try:
            response = await self.client.embeddings.create(
                model=self.model,
                input=text
            )
            
            if response.data and response.data[0].embedding:
                return response.data[0].embedding
            else:
                raise ValueError("Empty embedding response")
                
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embeddings (each embedding is a list of floats)
        """
        try:
            response = await self.client.embeddings.create(
                model=self.model,
                input=texts
            )
            
            embeddings = []
            for item in response.data:
                if item.embedding:
                    embeddings.append(item.embedding)
                else:
                    raise ValueError(f"Empty embedding for text")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings for this model.
        
        Returns:
            Embedding dimension size
        """
        dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536
        }
        
        return dimensions.get(self.model, 1536)