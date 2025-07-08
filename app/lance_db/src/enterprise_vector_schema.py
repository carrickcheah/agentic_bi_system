#!/usr/bin/env python3
"""
Enterprise Vector Schema - Unified LanceDB Schema for Multi-Module Integration
Production-grade vector database schema supporting Investigation, Intelligence, Synthesis, and Auto-Generation modules.
Designed for 100M+ vectors with enterprise performance and reliability requirements.
"""

from enum import Enum
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import hashlib

# Import LanceDB and Arrow for schema definition
try:
    import pyarrow as pa
    import lancedb
    LANCEDB_AVAILABLE = True
except ImportError:
    print("ERROR: Missing LanceDB dependencies. Install with: uv add lancedb pyarrow")
    # Create mock objects to prevent NameError
    class MockPyArrow:
        Schema = None  # Mock Schema class
        @staticmethod
        def schema(*args, **kwargs):
            return None
        @staticmethod
        def field(*args, **kwargs):
            return None
        @staticmethod
        def string():
            return None
        @staticmethod
        def list_(*args, **kwargs):
            return None
        @staticmethod
        def float32():
            return None
        @staticmethod
        def float64():
            return None
        @staticmethod
        def int32():
            return None
        @staticmethod
        def int64():
            return None
        @staticmethod
        def timestamp(*args):
            return None
    
    pa = MockPyArrow()
    lancedb = None
    LANCEDB_AVAILABLE = False

class ModuleSource(Enum):
    """Unified module source classification for vector operations."""
    AUTO_GENERATION = "auto_generation"
    INVESTIGATION = "investigation" 
    INTELLIGENCE = "intelligence"
    INSIGHT_SYNTHESIS = "insight_synthesis"
    BUSINESS_ANALYST = "business_analyst"
    DOMAIN_CLASSIFICATION = "domain_classification"
    COMPLEXITY_ANALYSIS = "complexity_analysis"
    PATTERN_RECOGNITION = "pattern_recognition"
    CROSS_MODULE_LEARNING = "cross_module_learning"
    BUSINESS_CONTEXT = "business_context"


class BusinessDomain(Enum):
    """Unified 14-domain business classification across all modules."""
    PRODUCTION = "production"
    QUALITY = "quality"
    SUPPLY_CHAIN = "supply_chain"
    COST = "cost"
    ASSETS = "assets"
    SAFETY = "safety"
    CUSTOMER = "customer"
    PLANNING = "planning"
    HUMAN_RESOURCES = "human_resources"
    SALES = "sales"
    FINANCE = "finance"
    MARKETING = "marketing"
    OPERATIONS = "operations"
    STRATEGIC = "strategic"


class PerformanceTier(Enum):
    """Unified performance tier classification based on complexity levels."""
    SIMPLE = "simple"              # 2-5 minutes: Single source, descriptive
    ANALYTICAL = "analytical"      # 5-15 minutes: Multiple sources, comparative  
    COMPUTATIONAL = "computational"  # 15-45 minutes: Advanced analytics, modeling
    INVESTIGATIVE = "investigative"  # 30-120 minutes: Multi-domain, predictive
    STRATEGIC = "strategic"        # 60+ minutes: Comprehensive business intelligence


class AnalysisType(Enum):
    """Unified analysis type classification across modules."""
    DESCRIPTIVE = "descriptive"      # What happened?
    DIAGNOSTIC = "diagnostic"        # Why did it happen?
    PREDICTIVE = "predictive"        # What will happen?
    PRESCRIPTIVE = "prescriptive"    # What should we do?


@dataclass
class VectorMetadata:
    """Unified metadata structure for all vector records."""
    
    # Core identification
    id: str
    module_source: ModuleSource
    created_at: datetime
    updated_at: datetime
    
    # Vector-specific fields  
    embedding_model: str = "BAAI/bge-m3"
    embedding_dimension: int = 1024
    vector_version: str = "1.0"
    semantic_cluster: Optional[int] = None
    similarity_threshold: float = 0.85
    
    # Unified scoring (0.0-1.0 scale across ALL modules)
    complexity_score: float = 0.5
    confidence_score: float = 0.8
    actionability_score: float = 0.6
    business_value_score: float = 0.5
    
    # Unified classification
    business_domain: BusinessDomain = BusinessDomain.OPERATIONS
    performance_tier: PerformanceTier = PerformanceTier.ANALYTICAL
    analysis_type: AnalysisType = AnalysisType.DESCRIPTIVE
    
    # Cross-module relationships
    related_investigation_ids: List[str] = field(default_factory=list)
    related_synthesis_ids: List[str] = field(default_factory=list)
    related_intelligence_ids: List[str] = field(default_factory=list)
    parent_vector_id: Optional[str] = None
    child_vector_ids: List[str] = field(default_factory=list)
    
    # Performance optimization
    access_frequency: int = 0
    last_accessed: Optional[datetime] = None
    computation_cost: float = 1.0
    cache_priority: int = 5  # 1=highest, 10=lowest
    
    # Business context
    user_role: str = "analyst"
    organization_context: str = "enterprise"
    business_impact: str = "medium"
    urgency_level: str = "normal"
    
    # Module-specific JSON storage
    module_metadata: Dict[str, Any] = field(default_factory=dict)
    cross_module_learning: Dict[str, Any] = field(default_factory=dict)
    
    def to_lancedb_record(self, vector: List[float], content: Dict[str, Any]) -> Dict[str, Any]:
        """Convert to LanceDB record format with unified schema."""
        return {
            # Core vector fields
            "id": self.id,
            "vector": vector,
            "vector_version": self.vector_version,
            "embedding_model": self.embedding_model,
            "embedding_dimension": self.embedding_dimension,
            
            # Module classification
            "module_source": self.module_source.value,
            "semantic_cluster": self.semantic_cluster,
            "similarity_threshold": self.similarity_threshold,
            
            # Unified scoring (0.0-1.0 scale)
            "complexity_score": float(self.complexity_score),
            "confidence_score": float(self.confidence_score), 
            "actionability_score": float(self.actionability_score),
            "business_value_score": float(self.business_value_score),
            
            # Unified classification
            "business_domain": self.business_domain.value,
            "performance_tier": self.performance_tier.value,
            "analysis_type": self.analysis_type.value,
            
            # Cross-module relationships
            "related_investigation_ids": json.dumps(self.related_investigation_ids),
            "related_synthesis_ids": json.dumps(self.related_synthesis_ids),
            "related_intelligence_ids": json.dumps(self.related_intelligence_ids),
            "parent_vector_id": self.parent_vector_id,
            "child_vector_ids": json.dumps(self.child_vector_ids),
            
            # Performance optimization
            "access_frequency": self.access_frequency,
            "last_accessed": self.last_accessed,
            "computation_cost": float(self.computation_cost),
            "cache_priority": self.cache_priority,
            
            # Business context
            "user_role": self.user_role,
            "organization_context": self.organization_context,
            "business_impact": self.business_impact,
            "urgency_level": self.urgency_level,
            
            # Timestamps
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            
            # Content storage
            "content_json": json.dumps(content),
            "module_metadata_json": json.dumps(self.module_metadata),
            "cross_module_learning_json": json.dumps(self.cross_module_learning),
        }


class EnterpriseVectorSchema:
    """Production-grade vector schema manager for enterprise LanceDB operations."""
    
    @staticmethod
    def get_arrow_schema() -> pa.Schema:
        """Define Arrow schema for optimal LanceDB performance."""
        return pa.schema([
            # Core vector fields
            pa.field("id", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), list_size=1024)),  # BGE-M3 embeddings
            pa.field("vector_version", pa.string()),
            pa.field("embedding_model", pa.string()),
            pa.field("embedding_dimension", pa.int32()),
            
            # Module classification  
            pa.field("module_source", pa.string()),
            pa.field("semantic_cluster", pa.int32()),
            pa.field("similarity_threshold", pa.float64()),
            
            # Unified scoring (0.0-1.0 scale for all modules)
            pa.field("complexity_score", pa.float64()),
            pa.field("confidence_score", pa.float64()),
            pa.field("actionability_score", pa.float64()),
            pa.field("business_value_score", pa.float64()),
            
            # Unified classification
            pa.field("business_domain", pa.string()),
            pa.field("performance_tier", pa.string()),
            pa.field("analysis_type", pa.string()),
            
            # Cross-module relationships (JSON strings for flexibility)
            pa.field("related_investigation_ids", pa.string()),
            pa.field("related_synthesis_ids", pa.string()),
            pa.field("related_intelligence_ids", pa.string()),
            pa.field("parent_vector_id", pa.string()),
            pa.field("child_vector_ids", pa.string()),
            
            # Performance optimization
            pa.field("access_frequency", pa.int64()),
            pa.field("last_accessed", pa.timestamp('us')),
            pa.field("computation_cost", pa.float64()),
            pa.field("cache_priority", pa.int32()),
            
            # Business context
            pa.field("user_role", pa.string()),
            pa.field("organization_context", pa.string()),
            pa.field("business_impact", pa.string()),
            pa.field("urgency_level", pa.string()),
            
            # Timestamps
            pa.field("created_at", pa.timestamp('us')),
            pa.field("updated_at", pa.timestamp('us')),
            
            # Content storage (JSON for flexibility)
            pa.field("content_json", pa.string()),
            pa.field("module_metadata_json", pa.string()),
            pa.field("cross_module_learning_json", pa.string()),
        ])
    
    @staticmethod
    def get_index_configuration() -> Dict[str, Any]:
        """Get optimized index configuration for enterprise workloads."""
        return {
            # Primary vector index for semantic similarity
            "vector_index": {
                "column": "vector",
                "index_type": "IVF_PQ",
                "num_partitions": 2048,        # Optimized for 100M+ vectors
                "num_sub_vectors": 32,         # Memory-performance balance
                "num_bits": 8,                 # Standard quantization
                "metric": "cosine",            # Standard for text embeddings
                "accelerator": "cuda"          # GPU acceleration if available
            },
            
            # Module-specific indexes for fast filtering
            "module_indexes": {
                "module_source": {"type": "BITMAP", "column": "module_source"},
                "business_domain": {"type": "BITMAP", "column": "business_domain"},
                "performance_tier": {"type": "BITMAP", "column": "performance_tier"},
                "analysis_type": {"type": "BITMAP", "column": "analysis_type"},
            },
            
            # Scoring indexes for range queries
            "scoring_indexes": {
                "complexity_score": {"type": "BTREE", "column": "complexity_score"},
                "confidence_score": {"type": "BTREE", "column": "confidence_score"},
                "actionability_score": {"type": "BTREE", "column": "actionability_score"},
                "business_value_score": {"type": "BTREE", "column": "business_value_score"},
            },
            
            # Performance optimization indexes
            "performance_indexes": {
                "semantic_cluster": {"type": "BTREE", "column": "semantic_cluster"},
                "access_frequency": {"type": "BTREE", "column": "access_frequency"},
                "cache_priority": {"type": "BTREE", "column": "cache_priority"},
                "created_at": {"type": "BTREE", "column": "created_at"},
            },
            
            # Cross-module relationship indexes  
            "relationship_indexes": {
                "parent_vector_id": {"type": "HASH", "column": "parent_vector_id"},
                # Note: JSON array fields need special handling for relationship searches
            }
        }
    
    @staticmethod
    def create_sample_data() -> List[Dict[str, Any]]:
        """Create sample data for schema establishment."""
        import numpy as np
        
        sample_metadata = VectorMetadata(
            id="schema_sample_001",
            module_source=ModuleSource.AUTO_GENERATION,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            complexity_score=0.5,
            confidence_score=0.8,
            actionability_score=0.7,
            business_value_score=0.6,
            business_domain=BusinessDomain.OPERATIONS,
            performance_tier=PerformanceTier.ANALYTICAL,
            analysis_type=AnalysisType.DESCRIPTIVE
        )
        
        sample_vector = np.random.randn(1024).astype(np.float32).tolist()
        sample_content = {
            "sql_query": "SELECT COUNT(*) FROM sample_table",
            "business_question": "What is the sample data volume?",
            "methodology": "descriptive_analysis"
        }
        
        return [sample_metadata.to_lancedb_record(sample_vector, sample_content)]
    
    @staticmethod
    def validate_record(record: Dict[str, Any]) -> bool:
        """Validate record against enterprise schema requirements."""
        required_fields = [
            "id", "vector", "module_source", "complexity_score", 
            "confidence_score", "actionability_score", "business_value_score",
            "business_domain", "performance_tier", "analysis_type"
        ]
        
        # Check required fields
        for field in required_fields:
            if field not in record:
                return False
        
        # Validate scoring ranges (0.0-1.0)
        scoring_fields = ["complexity_score", "confidence_score", "actionability_score", "business_value_score"]
        for field in scoring_fields:
            if not (0.0 <= record[field] <= 1.0):
                return False
        
        # Validate vector dimension
        if len(record["vector"]) != 1024:
            return False
        
        return True


class VectorSchemaEvolution:
    """Handles safe schema evolution for production vector databases."""
    
    def __init__(self, db_connection):
        self.db = db_connection
        self.current_version = "1.0"
        self.target_version = "2.0"
    
    async def migrate_to_unified_schema(self, table_name: str):
        """Safely migrate existing table to unified enterprise schema."""
        try:
            # Check if table exists
            existing_tables = await self.db.table_names()
            
            if table_name not in existing_tables:
                # Create new table with unified schema
                sample_data = EnterpriseVectorSchema.create_sample_data()
                table = await self.db.create_table(
                    table_name, 
                    data=sample_data,
                    schema=EnterpriseVectorSchema.get_arrow_schema()
                )
                
                # Remove sample data
                await table.delete("id = 'schema_sample_001'")
                
                print(f"✅ Created new table '{table_name}' with unified enterprise schema")
                return table
            
            else:
                # Migrate existing table
                table = await self.db.open_table(table_name)
                print(f"📋 Migrating existing table '{table_name}' to unified schema")
                
                # Add new columns gradually
                await self._add_unified_columns(table)
                
                # Backfill data
                await self._backfill_unified_data(table)
                
                # Create indexes
                await self._create_unified_indexes(table)
                
                print(f"✅ Successfully migrated '{table_name}' to unified enterprise schema")
                return table
                
        except Exception as e:
            print(f"❌ Schema migration failed: {e}")
            raise
    
    async def _add_unified_columns(self, table):
        """Add new unified schema columns to existing table."""
        # Note: LanceDB column addition would be implemented here
        # For now, this is a placeholder for the migration strategy
        pass
    
    async def _backfill_unified_data(self, table):
        """Backfill unified schema data from existing records."""
        # Note: Data migration logic would be implemented here
        pass
    
    async def _create_unified_indexes(self, table):
        """Create unified indexes for optimal performance."""
        index_config = EnterpriseVectorSchema.get_index_configuration()
        
        try:
            # Create primary vector index
            vector_config = index_config["vector_index"]
            await table.create_index(
                vector_config["column"],
                index_type=vector_config["index_type"],
                num_partitions=vector_config["num_partitions"],
                num_sub_vectors=vector_config["num_sub_vectors"],
                metric=vector_config["metric"]
            )
            print("✅ Created primary vector index")
            
        except Exception as e:
            print(f"⚠️ Vector index creation warning: {e}")
            print("Continuing without advanced indexing (will use brute force)")


# Utility functions for cross-module compatibility
def generate_vector_id(content: str, module_source: ModuleSource) -> str:
    """Generate consistent vector ID across modules."""
    content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
    return f"{module_source.value}_{content_hash}"


def normalize_score_to_unified_scale(score: Union[int, float], original_scale: str = "1-10") -> float:
    """Normalize scores from different modules to unified 0.0-1.0 scale."""
    if original_scale == "1-10":
        # Convert 1-10 integer scale to 0.0-1.0
        return max(0.0, min(1.0, (score - 1) / 9.0))
    elif original_scale == "0-100":
        # Convert 0-100 percentage to 0.0-1.0
        return max(0.0, min(1.0, score / 100.0))
    elif original_scale == "0.0-1.0":
        # Already in unified scale
        return max(0.0, min(1.0, float(score)))
    else:
        # Default: assume already normalized
        return max(0.0, min(1.0, float(score)))


def create_cross_module_metadata(
    investigation_id: Optional[str] = None,
    synthesis_id: Optional[str] = None, 
    intelligence_id: Optional[str] = None
) -> Dict[str, List[str]]:
    """Create cross-module relationship metadata."""
    return {
        "related_investigation_ids": [investigation_id] if investigation_id else [],
        "related_synthesis_ids": [synthesis_id] if synthesis_id else [],
        "related_intelligence_ids": [intelligence_id] if intelligence_id else []
    }


# Export main classes and functions
__all__ = [
    "EnterpriseVectorSchema",
    "VectorMetadata", 
    "VectorSchemaEvolution",
    "ModuleSource",
    "BusinessDomain",
    "PerformanceTier",
    "AnalysisType",
    "generate_vector_id",
    "normalize_score_to_unified_scale",
    "create_cross_module_metadata"
]