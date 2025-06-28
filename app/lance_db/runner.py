"""
Main SQL Embedding Service orchestrator with intelligent error handling.
Implements production-grade vector storage and search for SQL queries.
"""

import json
import time
import hashlib
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

import lancedb
import pandas as pd

try:
    from .config import settings
    from .lance_logging import logger, log_operation, log_error, log_performance
    from .embedding_component import EmbeddingGenerator
    from .search_component import VectorSearcher
    from .pattern_ingestion import BusinessPatternIngestion
    from .pattern_search_component import BusinessPatternSearcher
except ImportError:
    # For standalone execution
    from config import settings
    from lance_logging import logger, log_operation, log_error, log_performance
    from embedding_component import EmbeddingGenerator
    from search_component import VectorSearcher
    from pattern_ingestion import BusinessPatternIngestion
    from pattern_search_component import BusinessPatternSearcher


class SQLEmbeddingService:
    """Production-grade SQL embedding service with LanceDB backend and business pattern discovery."""
    
    def __init__(self):
        self.db = None
        self.table = None
        self.embedding_generator = None
        self.vector_searcher = None
        
        # Business pattern components
        self.pattern_ingestion = None
        self.pattern_searcher = None
        self.patterns_table = None
        
        self._initialized = False
        
        # Validate configuration on init
        try:
            settings.validate_paths()
            logger.info(f"SQLEmbeddingService initialized with data path: {settings.data_path}")
        except Exception as e:
            log_error("Configuration validation", e)
            raise
    
    async def initialize(self):
        """Initialize LanceDB connection and components."""
        if self._initialized:
            logger.warning("Service already initialized")
            return
        
        try:
            start_time = time.time()
            
            # Initialize LanceDB connection
            log_operation("Connecting to LanceDB", {"path": settings.data_path})
            self.db = await lancedb.connect_async(settings.data_path)
            
            # Create or open table
            await self._init_table()
            
            # Initialize components
            self.embedding_generator = EmbeddingGenerator()
            await self.embedding_generator.initialize()
            
            self.vector_searcher = VectorSearcher(self.table)
            
            # Initialize business pattern components
            await self._init_pattern_components()
            
            self._initialized = True
            
            duration_ms = (time.time() - start_time) * 1000
            log_performance("Service initialization", duration_ms)
            
        except Exception as e:
            log_error("Service initialization", e)
            raise RuntimeError(f"Failed to initialize SQL embedding service: {e}")
    
    async def _init_table(self):
        """Initialize the SQL embeddings table."""
        table_name = "sql_query_embeddings"
        
        try:
            existing_tables = await self.db.table_names()
            
            if table_name not in existing_tables:
                # Create new table with schema
                log_operation("Creating new table", {"name": table_name})
                
                # Create empty dataframe with proper schema
                schema_data = {
                    "id": pd.Series(dtype="str"),
                    "sql_query": pd.Series(dtype="str"),
                    "normalized_sql": pd.Series(dtype="str"),
                    "vector": pd.Series(dtype="object"),  # Will be vector type
                    "database": pd.Series(dtype="str"),
                    "query_type": pd.Series(dtype="str"),
                    "execution_time_ms": pd.Series(dtype="float64"),
                    "row_count": pd.Series(dtype="int32"),
                    "user_id": pd.Series(dtype="str"),
                    "timestamp": pd.Series(dtype="datetime64[ns]"),
                    "success": pd.Series(dtype="bool"),
                    "metadata": pd.Series(dtype="str"),
                }
                
                df = pd.DataFrame(schema_data)
                self.table = await self.db.create_table(table_name, data=df)
                logger.info(f"Created table: {table_name}")
            else:
                # Open existing table
                self.table = await self.db.open_table(table_name)
                logger.info(f"Opened existing table: {table_name}")
                
        except Exception as e:
            log_error("Table initialization", e)
            raise
    
    async def store_sql_query(self, query_data: Dict[str, Any]) -> str:
        """
        Store SQL query with its embedding.
        
        Args:
            query_data: Dictionary containing:
                - sql_query: The SQL query string
                - database: Database name (default: "mariadb")
                - query_type: Query type (default: "simple")
                - execution_time_ms: Execution time
                - row_count: Number of rows returned
                - user_id: User who ran the query
                - metadata: Additional metadata dict
        
        Returns:
            Query ID (hash)
        """
        if not self._initialized:
            raise RuntimeError("Service not initialized. Call initialize() first.")
        
        try:
            start_time = time.time()
            
            # Generate embedding
            sql_query = query_data["sql_query"]
            embedding = await self.embedding_generator.generate_embedding(sql_query)
            
            # Normalize SQL for better matching
            normalized_sql = self._normalize_sql(sql_query)
            
            # Generate unique ID
            query_id = hashlib.md5(sql_query.encode()).hexdigest()
            
            # Prepare record
            record = {
                "id": query_id,
                "sql_query": sql_query,
                "normalized_sql": normalized_sql,
                "vector": embedding,
                "database": query_data.get("database", "mariadb"),
                "query_type": query_data.get("query_type", "simple"),
                "execution_time_ms": query_data.get("execution_time_ms", 0.0),
                "row_count": query_data.get("row_count", 0),
                "user_id": query_data.get("user_id", "system"),
                "timestamp": datetime.now(),
                "success": query_data.get("success", True),
                "metadata": json.dumps(query_data.get("metadata", {}))
            }
            
            # Add to LanceDB
            await self.table.add([record])
            
            duration_ms = (time.time() - start_time) * 1000
            log_performance(f"Store SQL query {query_id[:8]}", duration_ms)
            
            return query_id
            
        except Exception as e:
            log_error("Store SQL query", e)
            raise
    
    async def find_similar_queries(
        self, 
        query: str, 
        threshold: Optional[float] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Find similar SQL queries using vector search.
        
        Args:
            query: The query to find similarities for
            threshold: Similarity threshold (default from config)
            limit: Maximum number of results
        
        Returns:
            List of similar queries with metadata
        """
        if not self._initialized:
            raise RuntimeError("Service not initialized. Call initialize() first.")
        
        try:
            start_time = time.time()
            
            # Use configured threshold if not provided
            if threshold is None:
                threshold = settings.similarity_threshold
            
            # Generate embedding for query
            query_embedding = await self.embedding_generator.generate_embedding(query)
            
            # Search for similar queries
            results = await self.vector_searcher.search_similar(
                query_embedding,
                threshold=threshold,
                limit=limit
            )
            
            duration_ms = (time.time() - start_time) * 1000
            log_performance(f"Find similar queries (found {len(results)})", duration_ms)
            
            return results
            
        except Exception as e:
            log_error("Find similar queries", e)
            raise
    
    async def get_query_by_id(self, query_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific query by its ID."""
        if not self._initialized:
            raise RuntimeError("Service not initialized. Call initialize() first.")
        
        try:
            # Search by ID
            results = await self.table.search().where(f"id = '{query_id}'").limit(1).to_list()
            
            if results:
                result = results[0]
                # Parse metadata JSON
                result["metadata"] = json.loads(result.get("metadata", "{}"))
                return result
            
            return None
            
        except Exception as e:
            log_error(f"Get query {query_id}", e)
            raise
    
    async def health_check(self) -> Dict[str, bool]:
        """Check health of all components."""
        health_status = {
            "lancedb_connection": False,
            "table_accessible": False,
            "embedding_generator": False,
            "vector_searcher": False,
            "patterns_table_accessible": False,
            "pattern_ingestion": False,
            "pattern_searcher": False
        }
        
        try:
            # Check LanceDB connection
            if self.db is not None:
                tables = await self.db.table_names()
                health_status["lancedb_connection"] = True
                health_status["table_accessible"] = "sql_query_embeddings" in tables
            
            # Check embedding generator
            if self.embedding_generator:
                health_status["embedding_generator"] = self.embedding_generator.is_healthy()
            
            # Check vector searcher
            if self.vector_searcher:
                health_status["vector_searcher"] = True
            
            # Check pattern components
            if self.patterns_table is not None:
                try:
                    # Test patterns table accessibility
                    pattern_tables = await self.db.table_names()
                    health_status["patterns_table_accessible"] = "business_patterns" in pattern_tables
                except Exception:
                    health_status["patterns_table_accessible"] = False
            
            # Check pattern ingestion component
            if self.pattern_ingestion and self.pattern_ingestion._initialized:
                health_status["pattern_ingestion"] = True
            
            # Check pattern searcher component
            if self.pattern_searcher:
                health_status["pattern_searcher"] = True
            
            logger.info(f"Health check status: {health_status}")
            
        except Exception as e:
            log_error("Health check", e)
        
        return health_status
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about stored queries."""
        if not self._initialized:
            raise RuntimeError("Service not initialized. Call initialize() first.")
        
        try:
            # Get total count
            df = await self.table.to_pandas()
            
            stats = {
                "total_queries": len(df),
                "databases": df["database"].value_counts().to_dict() if len(df) > 0 else {},
                "query_types": df["query_type"].value_counts().to_dict() if len(df) > 0 else {},
                "success_rate": (df["success"].sum() / len(df) * 100) if len(df) > 0 else 0,
                "avg_execution_time_ms": df["execution_time_ms"].mean() if len(df) > 0 else 0,
                "total_rows_returned": df["row_count"].sum() if len(df) > 0 else 0
            }
            
            return stats
            
        except Exception as e:
            log_error("Get statistics", e)
            raise
    
    def _normalize_sql(self, sql_query: str) -> str:
        """Normalize SQL for better matching."""
        import re
        
        # Remove extra whitespace
        normalized = " ".join(sql_query.split())
        
        # Convert to lowercase
        normalized = normalized.lower()
        
        # Replace specific values with placeholders
        # Numbers
        normalized = re.sub(r'\b\d+\b', '?', normalized)
        # Quoted strings
        normalized = re.sub(r"'[^']*'", "'?'", normalized)
        normalized = re.sub(r'"[^"]*"', '"?"', normalized)
        
        return normalized
    
    async def _init_pattern_components(self):
        """Initialize business pattern ingestion and search components."""
        try:
            # Initialize pattern ingestion (shares DB connection and embedding generator)
            self.pattern_ingestion = BusinessPatternIngestion()
            self.pattern_ingestion.db = self.db
            self.pattern_ingestion.embedding_generator = self.embedding_generator
            self.pattern_ingestion._initialized = True
            
            # Initialize patterns table
            await self.pattern_ingestion._init_patterns_table()
            self.patterns_table = self.pattern_ingestion.patterns_table
            
            # Initialize pattern searcher
            self.pattern_searcher = BusinessPatternSearcher(
                self.patterns_table, 
                self.embedding_generator
            )
            
            log_operation("Pattern components initialized")
            
        except Exception as e:
            log_error("Pattern components initialization", e)
            # Don't fail the whole service if patterns fail
            logger.warning("Pattern components failed to initialize, SQL functionality still available")
    
    # Business Pattern Methods
    
    async def ingest_business_patterns(self) -> Dict[str, Any]:
        """
        Ingest all business intelligence patterns from the patterns directory.
        
        Returns:
            Dictionary with ingestion statistics
        """
        if not self._initialized:
            raise RuntimeError("Service not initialized. Call initialize() first.")
        
        if not self.pattern_ingestion:
            raise RuntimeError("Pattern ingestion not available")
        
        try:
            log_operation("Starting business pattern ingestion")
            stats = await self.pattern_ingestion.ingest_all_patterns()
            log_operation("Business pattern ingestion completed", stats)
            return stats
            
        except Exception as e:
            log_error("Business pattern ingestion", e)
            raise
    
    async def search_business_patterns(
        self,
        query: str,
        search_type: str = "semantic",
        domain_filter: Optional[str] = None,
        complexity_filter: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search business intelligence patterns using semantic similarity.
        
        Args:
            query: Search query text
            search_type: "semantic", "workflow", or "hybrid"
            domain_filter: Filter by domain (e.g., "sales", "production")
            complexity_filter: Filter by complexity ("simple", "moderate", "complex")
            limit: Maximum number of results
        
        Returns:
            List of matching business patterns
        """
        if not self._initialized:
            raise RuntimeError("Service not initialized. Call initialize() first.")
        
        if not self.pattern_searcher:
            raise RuntimeError("Pattern search not available")
        
        try:
            results = await self.pattern_searcher.search_patterns(
                query=query,
                search_type=search_type,
                domain_filter=domain_filter,
                complexity_filter=complexity_filter,
                limit=limit
            )
            
            log_operation(f"Pattern search completed", {
                "query": query[:50],
                "results_count": len(results),
                "search_type": search_type
            })
            
            return results
            
        except Exception as e:
            log_error("Business pattern search", e)
            raise
    
    async def get_patterns_by_domain(
        self,
        domain: str,
        complexity: Optional[str] = None,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Get all patterns for a specific business domain."""
        if not self.pattern_searcher:
            raise RuntimeError("Pattern search not available")
        
        return await self.pattern_searcher.get_patterns_by_domain(domain, complexity, limit)
    
    async def get_recommended_patterns(
        self,
        user_role: str,
        complexity_preference: str = "moderate",
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get recommended patterns for a specific user role."""
        if not self.pattern_searcher:
            raise RuntimeError("Pattern search not available")
        
        return await self.pattern_searcher.get_recommended_patterns(
            user_role, complexity_preference, limit
        )
    
    async def get_pattern_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about ingested business patterns."""
        if not self.pattern_ingestion:
            raise RuntimeError("Pattern ingestion not available")
        
        return await self.pattern_ingestion.get_ingestion_statistics()
    
    async def cleanup(self):
        """Clean up resources."""
        try:
            if self.embedding_generator:
                await self.embedding_generator.cleanup()
            
            self._initialized = False
            logger.info("SQL embedding service cleaned up")
            
        except Exception as e:
            log_error("Cleanup", e)


# Main execution point for standalone testing
if __name__ == "__main__":
    import asyncio
    
    async def main():
        service = SQLEmbeddingService()
        await service.initialize()
        
        # Test health check
        health = await service.health_check()
        print(f"Health Status: {health}")
        
        # Test storing a query
        test_query = {
            "sql_query": "SELECT * FROM users WHERE age > 25",
            "database": "test_db",
            "query_type": "simple",
            "execution_time_ms": 45.2,
            "row_count": 150,
            "user_id": "test_user"
        }
        
        query_id = await service.store_sql_query(test_query)
        print(f"Stored query with ID: {query_id}")
        
        # Test finding similar queries
        similar = await service.find_similar_queries("SELECT * FROM users WHERE age > 30")
        print(f"Found {len(similar)} similar queries")
        
        await service.cleanup()
    
    asyncio.run(main())