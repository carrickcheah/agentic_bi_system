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
    from .enhanced_schema import (
        EnhancedSQLQuery, QueryContent, SemanticContext, TechnicalMetadata,
        UserContext, InvestigationContext, ExecutionResults, LearningMetadata,
        BusinessIntelligence, create_enhanced_query_from_simple, validate_enhanced_query
    )
except ImportError:
    
    from config import settings
    from lance_logging import logger, log_operation, log_error, log_performance
    from embedding_component import EmbeddingGenerator
    from search_component import VectorSearcher
    from pattern_ingestion import BusinessPatternIngestion
    from pattern_search_component import BusinessPatternSearcher
    # Simplified schema for standalone mode
    class EnhancedSQLQuery:
        def __init__(self):
            self.query_content = type('obj', (object,), {'sql_query': ''})()
        def to_lancedb_record(self):
            return {}
    def create_enhanced_query_from_simple(*args, **kwargs):
        return EnhancedSQLQuery()
    def validate_enhanced_query(query):
        return []


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
        """Initialize the enhanced SQL embeddings table."""
        table_name = "sql_query_embeddings"
        
        try:
            existing_tables = await self.db.table_names()
            
            if table_name not in existing_tables:
                # Create new table with enhanced schema
                log_operation("Creating enhanced table", {"name": table_name})
                
                # Enhanced schema with production-grade fields
                schema_data = {
                    # Core identification
                    "id": pd.Series(dtype="str"),
                    "sql_query": pd.Series(dtype="str"),
                    "normalized_sql": pd.Series(dtype="str"),
                    "vector": pd.Series(dtype="object"),  # BGE-M3 embedding vector
                    
                    # Key classification fields for indexing
                    "database": pd.Series(dtype="str"),
                    "query_type": pd.Series(dtype="str"),
                    "business_domain": pd.Series(dtype="str"),
                    "user_id": pd.Series(dtype="str"),
                    "execution_status": pd.Series(dtype="str"),
                    "success": pd.Series(dtype="bool"),
                    
                    # Performance metrics
                    "execution_time_ms": pd.Series(dtype="float64"),
                    "row_count": pd.Series(dtype="int32"),
                    "complexity_score": pd.Series(dtype="int32"),
                    "usage_frequency": pd.Series(dtype="int32"),
                    
                    # Timestamps
                    "timestamp": pd.Series(dtype="datetime64[ns]"),
                    "created_at": pd.Series(dtype="datetime64[ns]"),
                    "updated_at": pd.Series(dtype="datetime64[ns]"),
                    "last_executed_at": pd.Series(dtype="datetime64[ns]"),
                    
                    # JSON storage for complex nested data
                    "query_content_json": pd.Series(dtype="str"),
                    "semantic_context_json": pd.Series(dtype="str"),
                    "technical_metadata_json": pd.Series(dtype="str"),
                    "user_context_json": pd.Series(dtype="str"),
                    "investigation_context_json": pd.Series(dtype="str"),
                    "execution_results_json": pd.Series(dtype="str"),
                    "learning_metadata_json": pd.Series(dtype="str"),
                    "business_intelligence_json": pd.Series(dtype="str"),
                    "collaboration_json": pd.Series(dtype="str"),
                    "version_control_json": pd.Series(dtype="str"),
                    "caching_json": pd.Series(dtype="str"),
                    "monitoring_json": pd.Series(dtype="str"),
                    "security_json": pd.Series(dtype="str"),
                    "automation_json": pd.Series(dtype="str"),
                    "embeddings_json": pd.Series(dtype="str"),
                    
                    # Tags and custom fields
                    "tags_json": pd.Series(dtype="str"),
                    "custom_fields_json": pd.Series(dtype="str"),
                    
                    # Legacy compatibility (for migration)
                    "metadata": pd.Series(dtype="str"),
                }
                
                df = pd.DataFrame(schema_data)
                self.table = await self.db.create_table(table_name, data=df)
                logger.info(f"Created enhanced table: {table_name}")
            else:
                # Open existing table
                self.table = await self.db.open_table(table_name)
                logger.info(f"Opened existing table: {table_name}")
                
                # Check if migration is needed
                await self._check_and_migrate_schema()
                
        except Exception as e:
            log_error("Table initialization", e)
            raise
    
    async def store_sql_query(self, query_data: Dict[str, Any]) -> str:
        """
        Store SQL query with enhanced metadata and embedding.
        
        Args:
            query_data: Dictionary containing query information. Can be:
                - Legacy format: {sql_query, database, query_type, execution_time_ms, row_count, user_id, metadata}
                - Enhanced format: EnhancedSQLQuery object or comprehensive dictionary
        
        Returns:
            Query ID
        """
        if not self._initialized:
            raise RuntimeError("Service not initialized. Call initialize() first.")
        
        try:
            start_time = time.time()
            
            # Handle both legacy and enhanced formats
            if isinstance(query_data, EnhancedSQLQuery):
                enhanced_query = query_data
            elif "sql_query" in query_data:
                # Convert legacy format to enhanced
                business_context = query_data.get("metadata", {}) if isinstance(query_data.get("metadata"), dict) else {}
                enhanced_query = create_enhanced_query_from_simple(
                    sql_query=query_data["sql_query"],
                    database=query_data.get("database", "mariadb"),
                    user_id=query_data.get("user_id", "system"),
                    business_context=business_context
                )
                
                # Apply execution results if available
                if "execution_time_ms" in query_data:
                    enhanced_query.execution_results.execution_time_ms = query_data["execution_time_ms"]
                if "row_count" in query_data:
                    enhanced_query.execution_results.rows_returned = query_data["row_count"]
                if "success" in query_data:
                    enhanced_query.execution_results.success = query_data["success"]
                    enhanced_query.execution_results.execution_status = (
                        "completed" if query_data["success"] else "failed"
                    )
                
                # Update execution tracking
                enhanced_query.execution_results.execution_count = 1
                enhanced_query.execution_results.first_executed_at = datetime.now()
                enhanced_query.execution_results.last_executed_at = datetime.now()
                enhanced_query.learning_metadata.usage_frequency = 1
                enhanced_query.learning_metadata.first_used_at = datetime.now()
                enhanced_query.learning_metadata.last_used_at = datetime.now()
            else:
                raise ValueError("Invalid query_data format. Must contain 'sql_query' field.")
            
            # Validate enhanced query
            validation_errors = validate_enhanced_query(enhanced_query)
            if validation_errors:
                raise ValueError(f"Query validation failed: {', '.join(validation_errors)}")
            
            # Generate embeddings
            sql_query = enhanced_query.query_content.sql_query
            query_embedding = self.embedding_generator.generate_embedding(sql_query)
            enhanced_query.embeddings.query_embedding = query_embedding
            enhanced_query.embeddings.embedding_created_at = datetime.now()
            
            # Generate description embedding if available
            if enhanced_query.query_content.readable_description:
                desc_embedding = self.embedding_generator.generate_embedding(
                    enhanced_query.query_content.readable_description
                )
                enhanced_query.embeddings.description_embedding = desc_embedding
            
            # Generate business context embedding
            business_text = f"{enhanced_query.query_content.business_question or ''} {enhanced_query.semantic_context.business_domain.value}"
            if business_text.strip():
                context_embedding = self.embedding_generator.generate_embedding(business_text.strip())
                enhanced_query.embeddings.business_context_embedding = context_embedding
            
            # Convert to LanceDB record format
            record = enhanced_query.to_lancedb_record()
            record["vector"] = query_embedding  # Set primary vector for search
            
            # Add to LanceDB
            await self.table.add([record])
            
            duration_ms = (time.time() - start_time) * 1000
            log_performance(f"Store enhanced SQL query {enhanced_query._id}", duration_ms)
            
            logger.info(f"Stored enhanced query: {enhanced_query._id} | Domain: {enhanced_query.semantic_context.business_domain.value} | Type: {enhanced_query.query_content.query_type.value}")
            
            return enhanced_query._id
            
        except Exception as e:
            log_error("Store enhanced SQL query", e)
            raise
    
    async def find_similar_queries(
        self, 
        query: str, 
        threshold: Optional[float] = None,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Find similar SQL queries using enhanced vector search.
        
        Args:
            query: The query to find similarities for
            threshold: Similarity threshold (default from config)
            limit: Maximum number of results
            filters: Additional filters (business_domain, user_id, query_type, etc.)
        
        Returns:
            List of similar queries with enhanced metadata
        """
        if not self._initialized:
            raise RuntimeError("Service not initialized. Call initialize() first.")
        
        try:
            start_time = time.time()
            
            # Use configured threshold if not provided
            if threshold is None:
                threshold = settings.similarity_threshold
            
            # Generate embedding for query
            query_embedding = self.embedding_generator.generate_embedding(query)
            
            # Build filter conditions
            filter_conditions = []
            if filters:
                if "business_domain" in filters:
                    filter_conditions.append(f"business_domain = '{filters['business_domain']}'")
                if "user_id" in filters:
                    filter_conditions.append(f"user_id = '{filters['user_id']}'")
                if "query_type" in filters:
                    filter_conditions.append(f"query_type = '{filters['query_type']}'")
                if "database" in filters:
                    filter_conditions.append(f"database = '{filters['database']}'")
                if "success" in filters:
                    filter_conditions.append(f"success = {filters['success']}")
            
            # Search for similar queries with enhanced filtering
            results = await self.vector_searcher.search_similar(
                query_embedding,
                threshold=threshold,
                limit=limit,
                filter_expression=" AND ".join(filter_conditions) if filter_conditions else None
            )
            
            # Enhance results with parsed JSON fields
            enhanced_results = []
            for result in results:
                enhanced_result = dict(result)
                
                # Parse JSON fields back to objects for richer response
                json_fields = [
                    "query_content_json", "semantic_context_json", "technical_metadata_json",
                    "user_context_json", "investigation_context_json", "execution_results_json",
                    "learning_metadata_json", "business_intelligence_json", "tags_json"
                ]
                
                for field in json_fields:
                    if field in result and result[field]:
                        try:
                            field_name = field.replace("_json", "")
                            enhanced_result[field_name] = json.loads(result[field])
                        except (json.JSONDecodeError, TypeError):
                            pass
                
                enhanced_results.append(enhanced_result)
            
            duration_ms = (time.time() - start_time) * 1000
            log_performance(f"Find enhanced similar queries (found {len(results)})", duration_ms)
            
            logger.info(f"Found {len(results)} similar queries for: {query[:50]}... | Filters: {filters}")
            
            return enhanced_results
            
        except Exception as e:
            log_error("Find enhanced similar queries", e)
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
    
    async def _check_and_migrate_schema(self):
        """Check if schema migration is needed and perform it."""
        try:
            # Get current table schema
            df = await self.table.to_pandas()
            current_columns = set(df.columns)
            
            # Required enhanced columns
            required_columns = {
                "business_domain", "execution_status", "complexity_score", "usage_frequency",
                "query_content_json", "semantic_context_json", "technical_metadata_json",
                "user_context_json", "investigation_context_json", "execution_results_json",
                "learning_metadata_json", "business_intelligence_json"
            }
            
            missing_columns = required_columns - current_columns
            
            if missing_columns:
                log_operation("Schema migration needed", {"missing_columns": list(missing_columns)})
                await self._migrate_table_schema(missing_columns)
            else:
                logger.info("Table schema is up to date")
                
        except Exception as e:
            log_error("Schema migration check", e)
            # Don't fail initialization for migration issues
            logger.warning("Schema migration check failed, continuing with existing schema")
    
    async def _migrate_table_schema(self, missing_columns: set):
        """Migrate existing table to enhanced schema."""
        try:
            logger.info(f"Migrating table schema, adding columns: {missing_columns}")
            
            # For LanceDB, we need to recreate the table with new schema
            # First, backup existing data
            existing_data = await self.table.to_pandas()
            
            if len(existing_data) > 0:
                logger.info(f"Backing up {len(existing_data)} existing records")
                
                # Convert existing records to enhanced format
                enhanced_records = []
                for _, row in existing_data.iterrows():
                    try:
                        # Create enhanced query from legacy data
                        legacy_data = row.to_dict()
                        enhanced_query = EnhancedSQLQuery.from_legacy_data(legacy_data)
                        
                        # Generate embeddings if missing
                        if not enhanced_query.embeddings.query_embedding and enhanced_query.query_content.sql_query:
                            embedding = self.embedding_generator.generate_embedding(enhanced_query.query_content.sql_query)
                            enhanced_query.embeddings.query_embedding = embedding
                            enhanced_query.embeddings.embedding_created_at = datetime.now()
                        
                        enhanced_record = enhanced_query.to_lancedb_record()
                        enhanced_record["vector"] = enhanced_query.embeddings.query_embedding
                        enhanced_records.append(enhanced_record)
                        
                    except Exception as e:
                        logger.warning(f"Failed to migrate record {row.get('id', 'unknown')}: {e}")
                
                # Drop existing table
                table_name = "sql_query_embeddings"
                await self.db.drop_table(table_name)
                
                # Recreate with enhanced schema
                await self._init_table()
                
                # Restore migrated data
                if enhanced_records:
                    await self.table.add(enhanced_records)
                    logger.info(f"Migrated {len(enhanced_records)} records to enhanced schema")
            
            log_operation("Schema migration completed")
            
        except Exception as e:
            log_error("Schema migration", e)
            raise RuntimeError(f"Schema migration failed: {e}")
    
    async def store_enhanced_query(self, enhanced_query: EnhancedSQLQuery) -> str:
        """Store an already-constructed enhanced query."""
        return await self.store_sql_query(enhanced_query)
    
    async def get_query_by_enhanced_id(self, query_id: str) -> Optional[EnhancedSQLQuery]:
        """Get enhanced query by ID and reconstruct full object."""
        try:
            # Search by ID
            results = await self.table.search().where(f"id = '{query_id}'").limit(1).to_list()
            
            if results:
                result = results[0]
                
                # Reconstruct enhanced query from stored data
                enhanced_query = EnhancedSQLQuery()
                enhanced_query._id = result["id"]
                
                # Parse JSON fields back to objects
                json_fields = {
                    "query_content_json": "query_content",
                    "semantic_context_json": "semantic_context",
                    "technical_metadata_json": "technical_metadata",
                    "user_context_json": "user_context",
                    "investigation_context_json": "investigation_context",
                    "execution_results_json": "execution_results",
                    "learning_metadata_json": "learning_metadata",
                    "business_intelligence_json": "business_intelligence",
                    "collaboration_json": "collaboration",
                    "version_control_json": "version_control",
                    "caching_json": "caching",
                    "monitoring_json": "monitoring",
                    "security_json": "security",
                    "automation_json": "automation",
                    "embeddings_json": "embeddings",
                    "tags_json": "tags",
                    "custom_fields_json": "custom_fields"
                }
                
                for json_field, attr_name in json_fields.items():
                    if json_field in result and result[json_field]:
                        try:
                            parsed_data = json.loads(result[json_field])
                            setattr(enhanced_query, attr_name, parsed_data)
                        except (json.JSONDecodeError, TypeError):
                            pass
                
                return enhanced_query
            
            return None
            
        except Exception as e:
            log_error(f"Get enhanced query {query_id}", e)
            raise
    
    async def search_by_business_domain(
        self, 
        business_domain: str, 
        limit: int = 20,
        additional_filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search queries by business domain with enhanced filtering."""
        try:
            filter_conditions = [f"business_domain = '{business_domain}'"]
            
            if additional_filters:
                if "query_type" in additional_filters:
                    filter_conditions.append(f"query_type = '{additional_filters['query_type']}'")
                if "user_id" in additional_filters:
                    filter_conditions.append(f"user_id = '{additional_filters['user_id']}'")
                if "success" in additional_filters:
                    filter_conditions.append(f"success = {additional_filters['success']}")
            
            filter_expression = " AND ".join(filter_conditions)
            
            # Use vector search with domain filtering
            results = await self.table.search().where(filter_expression).limit(limit).to_list()
            
            logger.info(f"Found {len(results)} queries for business domain: {business_domain}")
            return results
            
        except Exception as e:
            log_error(f"Search by business domain {business_domain}", e)
            raise
    
    async def get_usage_analytics(self, days: int = 30) -> Dict[str, Any]:
        """Get comprehensive usage analytics for the enhanced schema."""
        try:
            # Get recent data
            from_date = datetime.now() - pd.Timedelta(days=days)
            
            df = await self.table.to_pandas()
            
            # Filter to recent records
            if 'created_at' in df.columns:
                recent_df = df[pd.to_datetime(df['created_at']) >= from_date]
            else:
                recent_df = df
            
            analytics = {
                "total_queries": len(df),
                "recent_queries": len(recent_df),
                "time_period_days": days,
                "business_domains": {},
                "query_types": {},
                "top_users": {},
                "performance_metrics": {},
                "success_rates": {},
                "complexity_distribution": {}
            }
            
            if len(recent_df) > 0:
                # Business domain distribution
                if 'business_domain' in recent_df.columns:
                    analytics["business_domains"] = recent_df['business_domain'].value_counts().to_dict()
                
                # Query type distribution
                if 'query_type' in recent_df.columns:
                    analytics["query_types"] = recent_df['query_type'].value_counts().to_dict()
                
                # Top users
                if 'user_id' in recent_df.columns:
                    analytics["top_users"] = recent_df['user_id'].value_counts().head(10).to_dict()
                
                # Performance metrics
                if 'execution_time_ms' in recent_df.columns:
                    execution_times = recent_df['execution_time_ms'].dropna()
                    if len(execution_times) > 0:
                        analytics["performance_metrics"] = {
                            "avg_execution_time_ms": float(execution_times.mean()),
                            "median_execution_time_ms": float(execution_times.median()),
                            "min_execution_time_ms": float(execution_times.min()),
                            "max_execution_time_ms": float(execution_times.max())
                        }
                
                # Success rates
                if 'success' in recent_df.columns:
                    success_rate = recent_df['success'].mean() if len(recent_df) > 0 else 0
                    analytics["success_rates"] = {
                        "overall_success_rate": float(success_rate),
                        "total_successful": int(recent_df['success'].sum()),
                        "total_failed": int((~recent_df['success']).sum())
                    }
                
                # Complexity distribution
                if 'complexity_score' in recent_df.columns:
                    complexity_scores = recent_df['complexity_score'].dropna()
                    if len(complexity_scores) > 0:
                        analytics["complexity_distribution"] = {
                            "avg_complexity": float(complexity_scores.mean()),
                            "complexity_counts": complexity_scores.value_counts().to_dict()
                        }
            
            return analytics
            
        except Exception as e:
            log_error("Get usage analytics", e)
            raise
    
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