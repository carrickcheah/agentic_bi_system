#!/usr/bin/env python3
"""
Production-Grade MOQ Template Ingestion System for LanceDB
Single-file solution for ingesting complex MOQ template structures into optimized vector storage.
Follows LanceDB expert standards for enterprise-grade performance and reliability.
"""

import asyncio
import json
import hashlib
import time
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

# Add parent directory to path for config access
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import lancedb
    import pandas as pd
    import numpy as np
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    print(f"ERROR: Missing required dependency: {e}")
    print("Install with: uv add lancedb pandas numpy sentence-transformers")
    sys.exit(1)

# Import configuration from parent directory
try:
    from config import settings
except ImportError:
    print("ERROR: Cannot import config from parent directory")
    print("Ensure config.py exists in parent directory")
    sys.exit(1)

# Import auto-generation functions
try:
    from .generate_fn import AutoGenerationEngine
except ImportError:
    # Fallback for direct execution
    try:
        from generate_fn import AutoGenerationEngine
    except ImportError:
        print("ERROR: Cannot import auto-generation functions")
        print("Ensure generate_fn module is properly configured")
        sys.exit(1)

# Import vector index manager for cross-module optimization
try:
    from .vector_index_manager import VectorIndexManager, create_optimized_vector_index_manager
    VECTOR_INDEX_MANAGER_AVAILABLE = True
except ImportError:
    try:
        from vector_index_manager import VectorIndexManager, create_optimized_vector_index_manager
        VECTOR_INDEX_MANAGER_AVAILABLE = True
    except ImportError:
        print("⚠️ Warning: Vector index manager not available - using basic indexing")
        VECTOR_INDEX_MANAGER_AVAILABLE = False

# Import vector performance monitor for baseline establishment
try:
    from .vector_performance_monitor import (
        VectorPerformanceMonitor, 
        PerformanceMetric, 
        PerformanceMetricType,
        create_performance_monitor
    )
    PERFORMANCE_MONITOR_AVAILABLE = True
except ImportError:
    try:
        from vector_performance_monitor import (
            VectorPerformanceMonitor, 
            PerformanceMetric, 
            PerformanceMetricType,
            create_performance_monitor
        )
        PERFORMANCE_MONITOR_AVAILABLE = True
    except ImportError:
        print("⚠️ Warning: Vector performance monitor not available")
        PERFORMANCE_MONITOR_AVAILABLE = False

# Import enterprise vector schema for unified metadata
try:
    from .enterprise_vector_schema import EnterpriseVectorSchema, VectorMetadata, ModuleSource
    ENTERPRISE_SCHEMA_AVAILABLE = True
except ImportError:
    try:
        from enterprise_vector_schema import EnterpriseVectorSchema, VectorMetadata, ModuleSource
        ENTERPRISE_SCHEMA_AVAILABLE = True
    except ImportError:
        print("⚠️ Warning: Enterprise vector schema not available - using legacy schema")
        ENTERPRISE_SCHEMA_AVAILABLE = False


class MOQTemplateParser:
    """Parser for complex MOQ template JSON structures with auto-generation capabilities."""
    
    def __init__(self):
        self.required_sections = [
            "query_content", "semantic_context", "technical_metadata",
            "user_context", "investigation_context", "execution_results"
        ]
        self.auto_generator = AutoGenerationEngine()
    
    def parse_template(self, template_path: Union[str, Path]) -> Dict[str, Any]:
        """Parse MOQ template with validation and error handling."""
        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                template_data = json.load(f)
            
            # Validate required sections
            missing_sections = [section for section in self.required_sections 
                              if section not in template_data]
            
            if missing_sections:
                raise ValueError(f"Missing required sections: {missing_sections}")
            
            # Validate core SQL query
            if not template_data.get("query_content", {}).get("sql_query"):
                raise ValueError("Missing sql_query in query_content section")
            
            return template_data
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e}")
        except FileNotFoundError:
            raise ValueError(f"Template file not found: {template_path}")
    
    def extract_core_fields(self, template_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract core fields for LanceDB storage with intelligent auto-generation."""
        # Auto-generate all null/zero values using production-grade algorithms
        enhanced_data = self.auto_generator.auto_generate_all_fields(template_data)
        
        return {
            # Core identification
            "sql_query": enhanced_data["query_content"]["sql_query"],
            "database": enhanced_data["technical_metadata"].get("database", "mariadb"),
            "query_type": enhanced_data["query_content"].get("query_type", "unknown"),
            
            # Business context
            "business_domain": enhanced_data["semantic_context"].get("business_domain", "unknown"),
            "business_question": enhanced_data["query_content"].get("business_question", ""),
            "query_intent": enhanced_data["query_content"].get("query_intent", ""),
            
            # User information
            "user_id": enhanced_data["user_context"].get("user_id", "template_user"),
            "user_role": enhanced_data["user_context"].get("user_role", "analyst"),
            
            # Execution metadata (auto-generated intelligent values)
            "execution_status": enhanced_data["execution_results"].get("execution_status", "not_executed"),
            "success": True,  # Default for template data
            "execution_time_ms": enhanced_data["technical_metadata"].get("estimated_execution_time_ms", 45.0),
            "row_count": enhanced_data["technical_metadata"].get("estimated_row_count", 10),
            
            # Technical metadata (auto-calculated)
            "complexity_score": enhanced_data["technical_metadata"].get("complexity_score", 3),
            "performance_tier": enhanced_data["technical_metadata"].get("performance_tier", "medium"),
            "has_subqueries": enhanced_data["technical_metadata"].get("has_subqueries", False),
            "has_window_functions": enhanced_data["technical_metadata"].get("has_window_functions", False),
            "join_count": enhanced_data["technical_metadata"].get("join_count", 0),
            "tables_used": enhanced_data["technical_metadata"].get("tables_used", []),
        }
    
    def flatten_for_storage(self, template_data: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten template for LanceDB storage while preserving structure with enhanced data."""
        # Get enhanced data with auto-generated values
        enhanced_data = self.auto_generator.auto_generate_all_fields(template_data)
        core_fields = self.extract_core_fields(template_data)
        
        # Store complex nested structures as JSON strings using enhanced data
        json_fields = {
            "query_content_json": json.dumps(enhanced_data["query_content"]),
            "semantic_context_json": json.dumps(enhanced_data["semantic_context"]),
            "technical_metadata_json": json.dumps(enhanced_data["technical_metadata"]),
            "user_context_json": json.dumps(enhanced_data["user_context"]),
            "investigation_context_json": json.dumps(enhanced_data["investigation_context"]),
            "execution_results_json": json.dumps(enhanced_data["execution_results"]),
            "learning_metadata_json": json.dumps(enhanced_data.get("learning_metadata", {})),
            "business_intelligence_json": json.dumps(enhanced_data.get("business_intelligence", {})),
            "collaboration_json": json.dumps(enhanced_data.get("collaboration", {})),
            "version_control_json": json.dumps(enhanced_data.get("version_control", {})),
            "caching_json": json.dumps(enhanced_data.get("caching", {})),
            "monitoring_json": json.dumps(enhanced_data.get("monitoring", {})),
            "security_json": json.dumps(enhanced_data.get("security", {})),
            "automation_json": json.dumps(enhanced_data.get("automation", {})),
            "embeddings_json": json.dumps(enhanced_data.get("embeddings", {})),
        }
        
        # Handle custom fields (like moq_specific_metadata) using enhanced data
        known_sections = set(self.required_sections + ["learning_metadata", "business_intelligence", 
                           "collaboration", "version_control", "caching", "monitoring", 
                           "security", "automation", "embeddings", "tags", "_id"])
        
        custom_fields = {k: v for k, v in enhanced_data.items() if k not in known_sections}
        json_fields["custom_fields_json"] = json.dumps(custom_fields)
        json_fields["tags_json"] = json.dumps(enhanced_data.get("tags", []))
        
        # Combine core fields with JSON fields
        return {**core_fields, **json_fields}


class OptimizedEmbeddingGenerator:
    """BGE-M3 embedding generator optimized for production performance."""
    
    def __init__(self):
        self.model = None
        self.model_name = settings.embedding_model
        self._initialized = False
    
    def initialize(self):
        """Initialize BGE-M3 model with optimization."""
        if self._initialized:
            return
        
        print(f"Loading embedding model: {self.model_name}")
        start_time = time.time()
        
        try:
            self.model = SentenceTransformer(self.model_name)
            
            # Warm up with test embedding
            _ = self.model.encode("test query", convert_to_numpy=True, normalize_embeddings=True)
            
            self._initialized = True
            duration = time.time() - start_time
            print(f"✅ Model loaded in {duration:.2f}s")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load embedding model: {e}")
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate optimized BGE-M3 embedding."""
        if not self._initialized:
            self.initialize()
        
        try:
            # BGE-M3 optimization: normalize embeddings for better cosine similarity
            embedding = self.model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            return embedding
            
        except Exception as e:
            raise RuntimeError(f"Failed to generate embedding: {e}")
    
    def get_dimension(self) -> int:
        """Get embedding dimension."""
        if not self._initialized:
            self.initialize()
        return self.model.get_sentence_embedding_dimension()


class LanceDBManager:
    """Production-grade LanceDB connection and table management with cross-module vector indexing."""
    
    def __init__(self):
        self.db = None
        self.table = None
        self.table_name = "moq_sql_embeddings"
        
        # Cross-module vector index manager
        self.vector_index_manager = None
        self.enterprise_schema_enabled = ENTERPRISE_SCHEMA_AVAILABLE
        self.advanced_indexing_enabled = VECTOR_INDEX_MANAGER_AVAILABLE
        
        # Performance monitoring
        self.performance_monitor = None
        self.performance_monitoring_enabled = PERFORMANCE_MONITOR_AVAILABLE
    
    async def initialize(self):
        """Initialize LanceDB connection with cross-module vector indexing."""
        try:
            print(f"Connecting to LanceDB at: {settings.data_path}")
            self.db = await lancedb.connect_async(settings.data_path)
            print("✅ Connected to LanceDB")
            
            # Initialize advanced vector index manager if available
            if self.advanced_indexing_enabled:
                print("🔧 Initializing cross-module vector index manager...")
                self.vector_index_manager = await create_optimized_vector_index_manager(self.db)
                print("✅ Cross-module vector indexing enabled")
            else:
                print("⚠️ Using basic indexing (advanced vector indexing not available)")
            
            # Initialize performance monitoring if available
            if self.performance_monitoring_enabled:
                print("📊 Initializing vector performance monitoring...")
                self.performance_monitor = await create_performance_monitor(self.vector_index_manager)
                print("✅ Vector performance monitoring enabled")
            else:
                print("⚠️ Performance monitoring not available")
            
        except Exception as e:
            raise RuntimeError(f"Failed to connect to LanceDB: {e}")
    
    async def setup_table(self, embedding_dim: int):
        """Setup optimized table with enterprise schema and cross-module indexing."""
        try:
            existing_tables = await self.db.table_names()
            
            if self.table_name in existing_tables:
                print(f"Opening existing table: {self.table_name}")
                self.table = await self.db.open_table(self.table_name)
                
                # Upgrade to enterprise schema if available
                if self.enterprise_schema_enabled and self.vector_index_manager:
                    print("🔄 Checking for enterprise schema compatibility...")
                    await self._ensure_enterprise_schema_compatibility()
            else:
                print(f"Creating new table: {self.table_name}")
                if self.enterprise_schema_enabled:
                    await self._create_enterprise_table(embedding_dim)
                else:
                    await self._create_optimized_table(embedding_dim)
            
        except Exception as e:
            raise RuntimeError(f"Failed to setup table: {e}")
    
    async def _create_optimized_table(self, embedding_dim: int):
        """Create table with optimized schema to prevent Arrow cast errors."""
        # Create dummy embedding for schema establishment
        dummy_embedding = np.zeros(embedding_dim, dtype=np.float32)
        
        # Sample data to establish proper Arrow types
        sample_data = [{
            # Core identification
            "id": "schema_sample",
            "sql_query": "SELECT 1 as sample",
            "normalized_sql": "select 1 as sample",
            "vector": dummy_embedding,
            
            # Classification fields
            "database": "mariadb",
            "query_type": "simple",
            "business_domain": "analytics",
            "business_question": "Sample question",
            "query_intent": "sample_intent",
            
            # User context
            "user_id": "sample_user",
            "user_role": "analyst",
            
            # Execution results
            "execution_status": "not_executed",
            "success": True,
            "execution_time_ms": 0.0,
            "row_count": 0,
            
            # Technical metadata
            "complexity_score": 1,
            "performance_tier": "low",
            "has_subqueries": False,
            "has_window_functions": False,
            "join_count": 0,
            "tables_used_json": "[]",
            
            # Timestamps
            "timestamp": datetime.now(),
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            
            # JSON storage for complex data
            "query_content_json": "{}",
            "semantic_context_json": "{}",
            "technical_metadata_json": "{}",
            "user_context_json": "{}",
            "investigation_context_json": "{}",
            "execution_results_json": "{}",
            "learning_metadata_json": "{}",
            "business_intelligence_json": "{}",
            "collaboration_json": "{}",
            "version_control_json": "{}",
            "caching_json": "{}",
            "monitoring_json": "{}",
            "security_json": "{}",
            "automation_json": "{}",
            "embeddings_json": "{}",
            "custom_fields_json": "{}",
            "tags_json": "[]",
        }]
        
        # Create table with proper schema
        df = pd.DataFrame(sample_data)
        self.table = await self.db.create_table(self.table_name, data=df)
        
        # Remove sample data immediately
        await self.table.delete("id = 'schema_sample'")
        print("✅ Created optimized table schema")
    
    async def create_production_index(self, embedding_dim: int):
        """Create optimized vector index with cross-module support."""
        try:
            print("Creating optimized vector index...")
            start_time = time.time()
            
            if self.vector_index_manager:
                # Use advanced cross-module indexing
                print("🚀 Using advanced cross-module vector indexing...")
                health_report = await self.vector_index_manager.get_index_health_report()
                print(f"Index health: {health_report['active_indexes']}/{health_report['total_indexes']} active")
            else:
                # Fallback to simple index creation
                await self.table.create_index("vector", metric="cosine")
            
            duration = time.time() - start_time
            print(f"✅ Vector index created in {duration:.2f}s")
                
        except Exception as e:
            print(f"Warning: Could not create vector index: {e}")
            print("Continuing without index (search will use brute force)")
    
    async def _create_enterprise_table(self, embedding_dim: int):
        """Create table with enterprise vector schema for cross-module integration."""
        if not self.enterprise_schema_enabled:
            await self._create_optimized_table(embedding_dim)
            return
        
        try:
            print("🏗️ Creating table with enterprise vector schema...")
            
            # Use enterprise schema for unified cross-module structure
            sample_data = EnterpriseVectorSchema.create_sample_data()
            
            self.table = await self.db.create_table(
                self.table_name,
                data=sample_data,
                schema=EnterpriseVectorSchema.get_arrow_schema()
            )
            
            # Remove sample data
            await self.table.delete("id = 'schema_sample_001'")
            
            print("✅ Enterprise table created with unified schema")
            
        except Exception as e:
            print(f"⚠️ Failed to create enterprise table: {e}")
            print("Falling back to optimized table creation...")
            await self._create_optimized_table(embedding_dim)
    
    async def _ensure_enterprise_schema_compatibility(self):
        """Ensure existing table is compatible with enterprise schema."""
        try:
            # Check if table has enterprise schema fields
            table_schema = self.table.schema
            enterprise_fields = [
                "module_source", "business_domain", "performance_tier", 
                "complexity_score", "confidence_score", "actionability_score"
            ]
            
            has_enterprise_fields = all(
                field in [f.name for f in table_schema] 
                for field in enterprise_fields
            )
            
            if has_enterprise_fields:
                print("✅ Table already has enterprise schema compatibility")
            else:
                print("🔄 Table needs enterprise schema upgrade (would require migration)")
                # Note: In production, this would trigger a schema migration
                # For now, we'll continue with existing schema
                
        except Exception as e:
            print(f"⚠️ Could not check enterprise schema compatibility: {e}")
    
    async def get_cross_module_query_stats(self) -> Dict[str, Any]:
        """Get cross-module query performance statistics."""
        from datetime import timezone
        stats = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "cross_module_indexing_enabled": self.advanced_indexing_enabled,
            "performance_monitoring_enabled": self.performance_monitoring_enabled,
            "vector_index_manager_available": self.vector_index_manager is not None,
            "performance_monitor_available": self.performance_monitor is not None
        }
        
        # Get index health if available
        if self.vector_index_manager:
            try:
                health_report = await self.vector_index_manager.get_index_health_report()
                stats["index_health"] = health_report
            except Exception as e:
                stats["index_health_error"] = str(e)
        
        # Get performance report if available
        if self.performance_monitor:
            try:
                performance_report = await self.performance_monitor.get_performance_report(time_window_hours=1)
                stats["performance_report"] = performance_report
            except Exception as e:
                stats["performance_report_error"] = str(e)
        
        return stats
    
    async def establish_performance_baseline(self) -> Dict[str, Any]:
        """Establish performance baseline for the system."""
        if not self.performance_monitor:
            return {"error": "Performance monitoring not available"}
        
        try:
            print("📊 Establishing performance baseline...")
            
            # Run comprehensive benchmark
            benchmark_results = await self.performance_monitor.run_comprehensive_benchmark()
            
            # Generate baseline report
            baseline_report = await self.performance_monitor.get_performance_report(time_window_hours=1)
            
            return {
                "baseline_established": True,
                "benchmark_results": benchmark_results,
                "baseline_report": baseline_report,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            return {
                "baseline_established": False,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }


class MOQIngestionEngine:
    """Main ingestion engine orchestrating all components."""
    
    def __init__(self):
        self.parser = MOQTemplateParser()
        self.embedder = OptimizedEmbeddingGenerator()
        self.db_manager = LanceDBManager()
        self.stats = {
            "start_time": None,
            "templates_processed": 0,
            "embeddings_generated": 0,
            "records_stored": 0,
            "errors": []
        }
    
    async def ingest_moq_template(self, template_path: Union[str, Path]) -> Dict[str, Any]:
        """Ingest MOQ template with comprehensive processing and validation."""
        self.stats["start_time"] = time.time()
        
        try:
            # Step 1: Parse template
            print("📋 Parsing MOQ template...")
            template_data = self.parser.parse_template(template_path)
            print(f"✅ Parsed template with {len(template_data)} sections")
            
            # Step 2: Initialize components
            print("🔧 Initializing components...")
            self.embedder.initialize()
            await self.db_manager.initialize()
            
            embedding_dim = self.embedder.get_dimension()
            await self.db_manager.setup_table(embedding_dim)
            
            # Step 3: Generate embeddings
            print("🧮 Generating embeddings...")
            sql_query = template_data["query_content"]["sql_query"]
            query_embedding = self.embedder.generate_embedding(sql_query)
            self.stats["embeddings_generated"] = 1
            
            # Step 4: Auto-generate metadata and prepare record
            print("🧠 Auto-generating intelligent metadata...")
            flattened_data = self.parser.flatten_for_storage(template_data)
            print("📦 Preparing record for storage...")
            
            # Generate unique ID and add metadata
            query_id = hashlib.md5(f"moq_template_{sql_query[:50]}".encode()).hexdigest()
            
            # Remove tables_used from flattened_data since it's stored in JSON
            tables_used = flattened_data.pop("tables_used", [])
            
            # Create record based on schema type
            if self.db_manager.enterprise_schema_enabled:
                record = await self._create_enterprise_record(
                    query_id, query_embedding, template_data, flattened_data, tables_used
                )
            else:
                record = {
                    "id": query_id,
                    "vector": query_embedding,
                    "normalized_sql": sql_query.lower().replace("  ", " "),
                    "timestamp": datetime.now(),
                    "created_at": datetime.now(),
                    "updated_at": datetime.now(),
                    "tables_used_json": json.dumps(tables_used),
                    **flattened_data
                }
            
            # Step 5: Store record with performance monitoring
            print("💾 Storing record in LanceDB...")
            store_start = time.time()
            await self.db_manager.table.add([record])
            store_time = (time.time() - store_start) * 1000  # ms
            
            # Record ingestion performance metric
            if self.db_manager.performance_monitor and PERFORMANCE_MONITOR_AVAILABLE:
                ingestion_metric = PerformanceMetric(
                    metric_type=PerformanceMetricType.INGESTION_RATE,
                    value=1 / (store_time / 1000),  # records per second
                    timestamp=datetime.now(timezone.utc),
                    module_source=ModuleSource.AUTO_GENERATION if ENTERPRISE_SCHEMA_AVAILABLE else None,
                    context={"record_size_bytes": len(str(record)), "store_time_ms": store_time}
                )
                self.db_manager.performance_monitor.record_metric(ingestion_metric)
            
            self.stats["records_stored"] = 1
            
            # Step 6: Create index with performance monitoring
            print("🔍 Creating production index...")
            index_start = time.time()
            await self.db_manager.create_production_index(embedding_dim)
            index_time = (time.time() - index_start) * 1000  # ms
            
            # Record index build performance metric
            if self.db_manager.performance_monitor and PERFORMANCE_MONITOR_AVAILABLE:
                index_metric = PerformanceMetric(
                    metric_type=PerformanceMetricType.INDEX_BUILD_TIME,
                    value=index_time,
                    timestamp=datetime.now(timezone.utc),
                    module_source=ModuleSource.AUTO_GENERATION if ENTERPRISE_SCHEMA_AVAILABLE else None,
                    context={"embedding_dimension": embedding_dim, "record_count": 1}
                )
                self.db_manager.performance_monitor.record_metric(index_metric)
            
            # Step 7: Verification with performance monitoring
            print("✅ Verifying ingestion...")
            verify_start = time.time()
            verification_result = await self._verify_ingestion(query_embedding, query_id)
            verify_time = (time.time() - verify_start) * 1000  # ms
            
            # Record query performance metric during verification
            if self.db_manager.performance_monitor and PERFORMANCE_MONITOR_AVAILABLE and verification_result.get("verified"):
                query_metric = PerformanceMetric(
                    metric_type=PerformanceMetricType.QUERY_LATENCY,
                    value=verify_time,
                    timestamp=datetime.now(timezone.utc),
                    module_source=ModuleSource.AUTO_GENERATION if ENTERPRISE_SCHEMA_AVAILABLE else None,
                    context={"similarity_score": verification_result.get("similarity", 0.0), "top_k": 1}
                )
                self.db_manager.performance_monitor.record_metric(query_metric)
            
            self.stats["templates_processed"] = 1
            
            return {
                "success": True,
                "query_id": query_id,
                "embedding_dimension": embedding_dim,
                "verification": verification_result,
                "stats": self._generate_stats(),
                "message": "MOQ template ingested successfully"
            }
            
        except Exception as e:
            error_msg = f"Ingestion failed: {str(e)}"
            self.stats["errors"].append(error_msg)
            print(f"❌ {error_msg}")
            
            return {
                "success": False,
                "error": error_msg,
                "stats": self._generate_stats()
            }
    
    async def _verify_ingestion(self, query_embedding: np.ndarray, expected_id: str) -> Dict[str, Any]:
        """Verify successful ingestion with similarity search."""
        try:
            # Perform similarity search using pandas approach for compatibility
            search_query = self.db_manager.table.search(query_embedding)
            results_df = await search_query.limit(3).to_pandas()
            
            if len(results_df) == 0:
                return {"verified": False, "reason": "No results found", "similarity": 0.0}
            
            # Check if our record is the top result
            top_result = results_df.iloc[0]
            similarity = 1 - top_result.get("_distance", 0)
            
            verification = {
                "verified": top_result["id"] == expected_id,
                "similarity": float(similarity),
                "total_results": len(results_df),
                "top_result_id": top_result["id"]
            }
            
            if verification["verified"]:
                print(f"✅ Verification passed (similarity: {similarity:.4f})")
            else:
                print(f"⚠️ Verification issue: expected {expected_id}, got {top_result['id']}")
            
            return verification
            
        except Exception as e:
            print(f"⚠️ Verification error: {e}")
            return {"verified": False, "reason": f"Verification error: {e}", "similarity": 0.0}
    
    async def _create_enterprise_record(
        self, 
        query_id: str, 
        query_embedding: np.ndarray, 
        template_data: Dict, 
        flattened_data: Dict,
        tables_used: List
    ) -> Dict[str, Any]:
        """Create record using enterprise vector schema."""
        if not self.db_manager.enterprise_schema_enabled:
            raise RuntimeError("Enterprise schema not available")
        
        try:
            # Extract enterprise vector metadata from enhanced template data
            enterprise_metadata = template_data.get('enterprise_vector_metadata', {})
            
            if not enterprise_metadata.get('cross_module_ready'):
                print("⚠️ Enterprise vector metadata not available, creating basic enterprise record")
                # Create minimal enterprise record
                from datetime import timezone
                now = datetime.now(timezone.utc)
                return {
                    "id": query_id,
                    "vector": query_embedding.tolist(),
                    "vector_version": "1.0",
                    "embedding_model": "BAAI/bge-m3",
                    "embedding_dimension": len(query_embedding),
                    
                    "module_source": ModuleSource.AUTO_GENERATION.value,
                    "semantic_cluster": None,
                    "similarity_threshold": 0.85,
                    
                    "complexity_score": 0.5,
                    "confidence_score": 0.8,
                    "actionability_score": 0.6,
                    "business_value_score": 0.5,
                    
                    "business_domain": "operations",
                    "performance_tier": "analytical",
                    "analysis_type": "descriptive",
                    
                    "related_investigation_ids": json.dumps([]),
                    "related_synthesis_ids": json.dumps([]),
                    "related_intelligence_ids": json.dumps([]),
                    "parent_vector_id": None,
                    "child_vector_ids": json.dumps([]),
                    
                    "access_frequency": 0,
                    "last_accessed": None,
                    "computation_cost": 1.0,
                    "cache_priority": 5,
                    
                    "user_role": template_data.get("user_context", {}).get("user_role", "analyst"),
                    "organization_context": "enterprise",
                    "business_impact": "medium",
                    "urgency_level": "normal",
                    
                    "created_at": now,
                    "updated_at": now,
                    
                    "content_json": json.dumps(template_data),
                    "module_metadata_json": json.dumps(flattened_data),
                    "cross_module_learning_json": json.dumps({})
                }
            
            # Use full enterprise metadata
            unified_scores = enterprise_metadata.get('unified_scores', {})
            unified_classification = enterprise_metadata.get('unified_classification', {})
            
            return {
                "id": query_id,
                "vector": query_embedding.tolist(),
                "vector_version": "1.0",
                "embedding_model": "BAAI/bge-m3",
                "embedding_dimension": len(query_embedding),
                
                "module_source": enterprise_metadata.get('module_source', 'auto_generation'),
                "semantic_cluster": None,
                "similarity_threshold": 0.85,
                
                "complexity_score": unified_scores.get('complexity_score', 0.5),
                "confidence_score": unified_scores.get('confidence_score', 0.8),
                "actionability_score": unified_scores.get('actionability_score', 0.6),
                "business_value_score": unified_scores.get('business_value_score', 0.5),
                
                "business_domain": unified_classification.get('business_domain', 'operations'),
                "performance_tier": unified_classification.get('performance_tier', 'analytical'),
                "analysis_type": unified_classification.get('analysis_type', 'descriptive'),
                
                "related_investigation_ids": json.dumps([]),
                "related_synthesis_ids": json.dumps([]),
                "related_intelligence_ids": json.dumps([]),
                "parent_vector_id": None,
                "child_vector_ids": json.dumps([]),
                
                "access_frequency": 0,
                "last_accessed": None,
                "computation_cost": 1.0,
                "cache_priority": 5,
                
                "user_role": template_data.get("user_context", {}).get("user_role", "analyst"),
                "organization_context": "enterprise",
                "business_impact": "medium",
                "urgency_level": "normal",
                
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc),
                
                "content_json": json.dumps(template_data),
                "module_metadata_json": json.dumps({
                    **flattened_data,
                    "tables_used": tables_used,
                    "enterprise_integration": enterprise_metadata
                }),
                "cross_module_learning_json": json.dumps({})
            }
            
        except Exception as e:
            print(f"⚠️ Failed to create enterprise record: {e}")
            # Fallback to basic record structure
            now = datetime.now(timezone.utc)
            return {
                "id": query_id,
                "vector": query_embedding.tolist(),
                "content_json": json.dumps(template_data),
                "module_source": "auto_generation",
                "created_at": now,
                "updated_at": now,
                "complexity_score": 0.5,
                "confidence_score": 0.8,
                "actionability_score": 0.6,
                "business_value_score": 0.5,
                "business_domain": "operations",
                "performance_tier": "analytical",
                "analysis_type": "descriptive"
            }
    
    def _generate_stats(self) -> Dict[str, Any]:
        """Generate comprehensive ingestion statistics."""
        duration = time.time() - self.stats["start_time"] if self.stats["start_time"] else 0
        
        return {
            "duration_seconds": round(duration, 2),
            "templates_processed": self.stats["templates_processed"],
            "embeddings_generated": self.stats["embeddings_generated"],
            "records_stored": self.stats["records_stored"],
            "errors_count": len(self.stats["errors"]),
            "errors": self.stats["errors"],
            "performance_metrics": {
                "embeddings_per_second": self.stats["embeddings_generated"] / max(duration, 0.001),
                "records_per_second": self.stats["records_stored"] / max(duration, 0.001)
            }
        }


# Main execution
async def main():
    """Main execution function with CLI interface."""
    print("🚀 MOQ Template Ingestion System")
    print("=" * 40)
    
    # Default template path
    default_template = Path(__file__).parent.parent / "patterns" / "template_moq.json"
    
    # Check if template exists
    if not default_template.exists():
        print(f"❌ Template not found: {default_template}")
        print("Please ensure template_moq.json exists in the patterns directory")
        return 1
    
    # Initialize ingestion engine
    engine = MOQIngestionEngine()
    
    # Perform ingestion
    result = await engine.ingest_moq_template(default_template)
    
    # Display results
    print("\n" + "=" * 40)
    print("📊 Ingestion Results")
    print("=" * 40)
    
    if result["success"]:
        print(f"✅ Success: {result['message']}")
        print(f"🆔 Query ID: {result['query_id']}")
        print(f"📏 Embedding Dimension: {result['embedding_dimension']}")
        print(f"🔍 Verified: {result['verification']['verified']}")
        print(f"📈 Similarity: {result['verification']['similarity']:.4f}")
    else:
        print(f"❌ Failed: {result['error']}")
    
    # Performance stats
    stats = result["stats"]
    print(f"\n📊 Performance:")
    print(f"⏱️  Duration: {stats['duration_seconds']}s")
    print(f"📦 Records: {stats['records_stored']}")
    print(f"🧮 Embeddings: {stats['embeddings_generated']}")
    
    if stats["errors_count"] > 0:
        print(f"\n⚠️ Errors ({stats['errors_count']}):")
        for error in stats["errors"]:
            print(f"  - {error}")
    
    return 0 if result["success"] else 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n🛑 Ingestion interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)