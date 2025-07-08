"""
LanceDB SQL Embeddings & Business Intelligence Package

This package provides a production-grade SQL query embedding and similarity search 
system with business intelligence pattern discovery for the Agentic BI Backend. 
It implements high-performance vector storage and semantic search capabilities for 
SQL query caching, organizational learning, and business pattern discovery.

Key Components:
- SQLEmbeddingService (runner.py): Main orchestrator with async operations
- EmbeddingGenerator: BGE-M3 model for SQL query and business pattern embeddings
- VectorSearcher: LanceDB vector similarity search operations
- BusinessPatternIngestion: Business intelligence pattern processing system
- BusinessPatternSearcher: Advanced semantic search for business patterns
- LanceDBSettings: Type-safe configuration management

SQL Query Usage:
    from app.lance_db import SQLEmbeddingService
    
    # Initialize service
    service = SQLEmbeddingService()
    await service.initialize()
    
    # Store SQL query with embedding
    query_id = await service.store_sql_query({
        "sql_query": "SELECT * FROM users WHERE age > 25",
        "database": "mariadb",
        "execution_time_ms": 45.2
    })
    
    # Find similar queries for cache hits
    similar = await service.find_similar_queries("SELECT * FROM users WHERE age > 30")

Business Pattern Usage:
    # Ingest business intelligence patterns
    stats = await service.ingest_business_patterns()
    
    # Search patterns semantically
    patterns = await service.search_business_patterns(
        query="sales revenue analysis",
        search_type="semantic",
        domain_filter="sales"
    )
    
    # Get role-specific recommendations
    recommendations = await service.get_recommended_patterns(
        user_role="sales_manager",
        complexity_preference="moderate"
    )
    
    # Monitor system health
    health = await service.health_check()

Architecture:
    Self-contained dual-purpose vector database system:
    1. BGE-M3 embeddings - Semantic understanding of SQL queries and business patterns
    2. LanceDB storage - High-performance vector database with dual tables
    3. Similarity search - Fast cache hit detection (5-15ms) and pattern discovery
    4. Business intelligence - 300+ patterns across 14 business domains
    5. Zeabur deployment - Production-ready cloud storage

Configuration:
    - Self-contained module configuration via settings.env
    - Type-safe pydantic settings with validation
    - Automatic path detection (local/Zeabur)
    - Production-ready error handling and logging
    - Business pattern directory management
"""

from .runner import SQLEmbeddingService
from .src.embedding_component import EmbeddingGenerator
from .src.search_component import VectorSearcher
from .src.pattern_ingestion import BusinessPatternIngestion
from .src.pattern_search_component import BusinessPatternSearcher
from .config import LanceDBSettings, settings
from .src.lance_logging import get_logger

__all__ = [
    "SQLEmbeddingService",          # Primary entry point - recommended for both SQL and patterns
    "EmbeddingGenerator",           # Direct access if needed
    "VectorSearcher",               # Direct access if needed
    "BusinessPatternIngestion",     # Pattern ingestion system
    "BusinessPatternSearcher",      # Pattern search capabilities
    "LanceDBSettings",              # Configuration class
    "settings",                     # Configured settings instance
    "get_logger",                   # Logging utility
]

__version__ = "1.1.0"
__description__ = "Self-contained LanceDB module for SQL query embeddings and business intelligence pattern discovery"